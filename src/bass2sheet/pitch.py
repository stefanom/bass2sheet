"""Pitch tracking (CREPE + pYIN) and the per-frame features the later stages share, cached by stem hash."""

import os

import librosa
import numpy as np
import torch

from .common import FMAX, FMIN, HOP, HOP_S, SR, atomic_savez, log, sha1_array


CREPE_MODELS = ("full", "tiny")


def track_crepe(mono, device="cpu", model="full"):
    import torchcrepe

    y16 = librosa.resample(mono, orig_sr=SR, target_sr=16000)
    audio = torch.from_numpy(y16)[None]
    # CREPE's lowest bin is about 32 Hz; a lower fmin masks every bin (output: flat 31.7 Hz, NaN periodicity)
    pitch, periodicity = torchcrepe.predict(audio, 16000, hop_length=160, fmin=33.0, fmax=FMAX, model=model,
                                            decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                            batch_size=2048, device=device)
    periodicity = torchcrepe.filter.median(periodicity, 3)
    return pitch[0].cpu().numpy(), periodicity[0].cpu().numpy()


def track_pyin(mono):
    y22 = librosa.resample(mono, orig_sr=SR, target_sr=22050)
    f0, _, _ = librosa.pyin(y22, fmin=FMIN, fmax=FMAX, sr=22050, frame_length=4096, hop_length=int(22050 * HOP_S))
    return f0


def features(bass_mono, cache_dir, device="cpu", crepe_model="full"):
    """Per-10 ms features of the bass stem: CREPE f0 + periodicity, pYIN f0, RMS (dB), onsets, CQT.

    `crepe_model` is "full" (the default, most accurate) or "tiny" (several times faster, for tests)."""
    stem_id = sha1_array(bass_mono)
    path = os.path.join(cache_dir, f"tracks-{stem_id[:16]}{'' if crepe_model == 'full' else '-' + crepe_model}.npz")
    if os.path.exists(path):
        log.info("pitch tracking: cached")
        tr = np.load(path)
        f0_c, per_c, f0_p = tr["f0_c"], tr["per_c"], tr["f0_p"]
    else:
        from concurrent.futures import ThreadPoolExecutor

        log.info("pitch tracking: CREPE on %s, pYIN alongside", device)
        with ThreadPoolExecutor(1) as pool:
            pyin = pool.submit(track_pyin, bass_mono)
            try:  # CREPE on the GPU (MPS/CUDA) is ~10x faster than on the CPU and matches it within 0.4 semitone
                f0_c, per_c = track_crepe(bass_mono, device, crepe_model)
            except Exception as e:
                if device == "cpu":
                    raise
                log.warning("CREPE on %s failed (%s); using the CPU", device, e)
                f0_c, per_c = track_crepe(bass_mono, "cpu", crepe_model)
            f0_p = pyin.result()
        atomic_savez(path, f0_c=f0_c, per_c=per_c, f0_p=f0_p)
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=HOP)[0] + 1e-9)
    env = librosa.onset.onset_strength(y=bass_mono, sr=SR, hop_length=HOP, fmax=1000)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True)
    cqt = np.abs(librosa.cqt(bass_mono, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72,
                             bins_per_octave=12))  # bin k = MIDI 21 + k
    return dict(f0_c=f0_c, per_c=per_c, f0_p=f0_p, rms_db=rms_db, onsets=onsets, cqt=cqt, stem_id=stem_id)
