"""Pitch tracking (CREPE + pYIN) and the per-frame features the later stages share, cached by stem hash."""

import os

import librosa
import numpy as np
import torch

from .common import FMAX, FMIN, HOP, HOP_S, SR, atomic_savez, log, release_gpu_memory, sha1_array

CREPE_MODELS = ("full", "tiny")


PITCH_BINS = 360  # CREPE's pitch bins, 20 cents apart
BIN0_CENTS = 1997.3794084376191  # the first bin, in cents above 10 Hz


def bin_of_hz(hz):
    return (1200 * np.log2(np.asarray(hz, dtype=float) / 10.0) - BIN0_CENTS) / 20.0


def hz_of_bin(b):
    return 10.0 * 2 ** ((20.0 * np.asarray(b, dtype=float) + BIN0_CENTS) / 1200)


def crepe_probabilities(mono, device="cpu", model="full"):
    """CREPE's probability for each pitch bin in every 10 ms frame: (frames, PITCH_BINS), before any decoding."""
    import torchcrepe

    y16 = librosa.resample(mono, orig_sr=SR, target_sr=16000)
    audio = torch.from_numpy(y16)[None]
    out = []
    # the batch bounds memory: the full model needs about 1 GB per 100 frames in flight, on the GPU or the CPU
    with torch.no_grad():
        for frames in torchcrepe.preprocess(audio, 16000, 160, 512 if device != "cpu" else 128, device):
            out.append(torchcrepe.infer(frames, model, device=device).cpu().numpy())
    return np.concatenate(out)


def decode_pitch(probs, jump=1e-6, width=12, floor=1e-3):
    """The most likely pitch path through CREPE's bin probabilities: (f0 in Hz, the probability at the chosen bin).

    Between frames the path moves smoothly, within `width` bins, or, with probability `jump`, leaps to any bin.
    The leap is what a bass line needs. torchcrepe's own Viterbi decoder allows smooth moves only, so after a
    jump of a fifth or more it lags behind or stays on the old note, where the probability is low: the new note
    is lost and the old, lower one is held over it. On real bass guitar recordings the right pitch for notes
    after such a jump went from 63% to 90% of notes once leaps were allowed (docs/evaluation.md).
    CREPE's lowest bin is about 32 Hz; bins outside 33 Hz..FMAX are not considered."""
    q = probs.astype(np.float64).copy()
    lo, hi = int(np.ceil(bin_of_hz(33.0))), int(np.floor(bin_of_hz(FMAX)))
    q[:, :lo] = 0
    q[:, hi + 1 :] = 0
    frames, bins = q.shape
    emit = np.log(q + floor)
    offsets = np.arange(-width + 1, width)
    smooth = np.maximum(width - np.abs(offsets), 0).astype(float)
    smooth = np.log((1 - jump) * smooth / smooth.sum())
    leap = np.log(jump / bins)
    before = np.arange(bins)[:, None] + offsets[None, :]  # the bins a smooth move can come from
    inside = (before >= 0) & (before < bins)
    before = np.clip(before, 0, bins - 1)
    rows = np.arange(bins)
    score = emit[0].copy()
    back = np.zeros((frames, bins), np.int32)
    for t in range(1, frames):
        cand = np.where(inside, score[before] + smooth[None, :], -np.inf)
        k = cand.argmax(1)
        by_move = cand[rows, k]
        top = int(score.argmax())
        by_leap = score[top] + leap
        leaps = by_leap > by_move
        back[t] = np.where(leaps, top, before[rows, k])
        score = np.where(leaps, by_leap, by_move) + emit[t]
    path = np.zeros(frames, np.int64)
    path[-1] = int(score.argmax())
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    near = np.clip(path[:, None] + np.arange(-4, 5)[None, :], 0, bins - 1)  # refine within 80 cents of the bin
    w = np.take_along_axis(q, near, 1)
    fine = (w * near).sum(1) / np.maximum(w.sum(1), 1e-9)
    return hz_of_bin(fine).astype(np.float32), q[np.arange(frames), path].astype(np.float32)


def track_crepe(probs):
    """CREPE f0 (Hz) and periodicity per 10 ms frame from its bin probabilities (see decode_pitch)."""
    from scipy.ndimage import median_filter

    f0, periodicity = decode_pitch(probs)
    return f0, median_filter(periodicity, 3, mode="nearest")


def track_pyin(mono):
    y22 = librosa.resample(mono, orig_sr=SR, target_sr=22050)
    f0, _, _ = librosa.pyin(y22, fmin=FMIN, fmax=FMAX, sr=22050, frame_length=4096, hop_length=int(22050 * HOP_S))
    return f0


def features(bass_mono, cache_dir, device="cpu", crepe_model="full"):
    """Per-10 ms features of the bass stem: CREPE f0 + periodicity, pYIN f0, RMS (dB), onsets, CQT.

    `crepe_model` is "full" (the default, most accurate) or "tiny" (several times faster, for tests).
    What is cached is what the models gave (CREPE's bin probabilities, pYIN's f0), not the pitch path decoded
    from it: the decoding takes a second and can then change without running the models again."""
    stem_id = sha1_array(bass_mono)
    suffix = "" if crepe_model == "full" else "-" + crepe_model
    path = os.path.join(cache_dir, f"crepe-{stem_id[:16]}{suffix}.npz")
    if os.path.exists(path):
        log.info("pitch tracking: cached")
        tr = np.load(path)
        probs, f0_p = tr["probs"], tr["f0_p"]
    else:
        from concurrent.futures import ThreadPoolExecutor

        log.info("pitch tracking: CREPE on %s, pYIN alongside", device)
        with ThreadPoolExecutor(1) as pool:
            pyin = pool.submit(track_pyin, bass_mono)
            try:  # CREPE on the GPU (MPS/CUDA) is ~10x faster than on the CPU and matches it within 0.4 semitone
                probs = crepe_probabilities(bass_mono, device, crepe_model)
            except Exception as e:
                if device == "cpu":
                    raise
                log.warning("CREPE on %s failed (%s); using the CPU", device, e)
                probs = crepe_probabilities(bass_mono, "cpu", crepe_model)
            f0_p = pyin.result()
        release_gpu_memory()
        probs = probs.astype(np.float16)  # what the cache holds, so a cached run decodes the very same numbers
        atomic_savez(path, compressed=True, probs=probs, f0_p=f0_p)
    f0_c, per_c = track_crepe(probs)
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=HOP)[0] + 1e-9)
    env = librosa.onset.onset_strength(y=bass_mono, sr=SR, hop_length=HOP, fmax=1000)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True)
    cqt = np.abs(
        librosa.cqt(bass_mono, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72, bins_per_octave=12)
    )  # bin k = MIDI 21 + k
    return dict(f0_c=f0_c, per_c=per_c, f0_p=f0_p, rms_db=rms_db, onsets=onsets, cqt=cqt, stem_id=stem_id)
