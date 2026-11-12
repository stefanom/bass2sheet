"""Turn per-frame pitch into notes: hysteresis voicing, pluck-only splits of repeated pitches, legato ends,
and octave sanity checks against the stem's spectrum."""

from dataclasses import asdict, dataclass

import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import HOP, HOP_S, SR


@dataclass
class SegParams:
    enter_per: float = 0.45  # CREPE periodicity that counts as clearly pitched (pitch estimate, note starts)
    stay_per: float = 0.15  # a sounding note ends only after periodicity stays below this ...
    unpitched_frames: int = 20  # ... for 200 ms (each pluck briefly drops periodicity)
    decay_db: float = 15.0  # ... or once it decays this far below its own peak
    attack_db: float = 3.0  # a repeated pitch is a new note only if the low-band energy jumps this much
    flux_support: float = 1.0  # splitting a held note also needs this much spectral flux at the jump (0 = no)
    flux_thr: float = 3.0  # ... or the spectral flux alone reaches this (0 = never): a re-pluck of a ringing note
    strict_replucks: bool = False  # a repeated pitch is a new note only on a bright pluck: fewer, longer notes
    split_held: bool = True  # split a held note at each pluck inside it: repeated notes the tracker saw as one
    split_min_frames: int = 8  # ... when both parts last at least this long (shorter cuts are the note's own attack)
    merge_gap_frames: int = 10  # same-pitch notes closer than 100 ms without a new pluck are one note
    rise_before: int = 2  # pluck rise: peak in the next rise_after frames minus trough in the last rise_before
    rise_after: int = 3
    attack_look: int = 1  # look for that rise within this many frames of a note start
    trust_per: float = 0.5  # CREPE's median periodicity over a note from which its octave stands against pYIN's
    octave_up_db: float = 25.0  # octave this much louder than the fundamental: the note is an octave up
    min_voiced_frames: int = 6  # 60 ms of clearly pitched sound makes a note
    refine_onsets: bool = True  # move each note start to where its pluck begins to sound (1 ms resolution)
    refine_rise_db: float = 6.0  # a pluck: the envelope rises at least this much from the trough before it
    refine_level: float = 0.1  # the start is where the envelope has made this share of that rise
    refine_back_s: float = 0.06  # how far before / after the frame-level start to look
    refine_ahead_s: float = 0.05

    def as_dict(self):
        return asdict(self)


def low_band_db(bass_mono):
    """30-400 Hz energy per 10 ms frame, in dB."""
    return pluck_features(bass_mono)[0]


def pluck_features(bass_mono):
    """Two views of a pluck per 10 ms frame: the 30-400 Hz level in dB, whose jump marks a note that starts from
    a quieter moment, and the spectral flux over 30-4000 Hz (mean rise of the dB spectrum across frequency), which
    also marks a re-pluck of a note that is still ringing, where the level barely moves but the attack's
    brightness returns."""
    S = np.abs(librosa.stft(bass_mono, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = librosa.amplitude_to_db(S[(f > 30) & (f < 400)].sum(0) + 1e-9)
    flux = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S[(f > 30) & (f < 4000)] + 1e-9), sr=SR, hop_length=HOP
    )
    return low, flux


def attack_rise(low_db, before, after):
    n = len(low_db)
    pad = np.pad(low_db, (before, after), mode="edge")
    return sliding_window_view(pad[before:], after + 1).max(1) - sliding_window_view(pad[: n + before], before + 1).min(
        1
    )


def refine_onsets(notes, bass_mono, p):
    """Move note starts from the 10 ms analysis frames to where the pluck starts to sound.

    Frame-level starts come from an onset detector and from CREPE's pitch changes; both see a note only once it
    fills a good part of their window, which puts starts some 15-25 ms late. Here each start moves to the point
    where a 1 ms envelope has made `refine_level` of its rise from the trough before the pluck to the pluck's
    peak. Notes without such a rise (slides, hammer-ons, legato pitch changes) keep their start. Sets
    nt["plucked"] and keeps the previous note from overlapping the new start."""
    from scipy.ndimage import uniform_filter1d

    step = 44  # samples: about 1 ms
    env = np.sqrt(np.maximum(uniform_filter1d(bass_mono.astype(np.float64) ** 2, size=353), 0))[::step]  # 8 ms window
    at = lambda t: int(round(t * SR / step))
    ratio = 10 ** (p.refine_rise_db / 20)
    for i, nt in enumerate(notes):
        ts = nt["start"]
        lo = max(0, at(ts - p.refine_back_s), at(notes[i - 1]["start"] + 0.03) if i else 0)
        hi = min(len(env) - 1, at(ts + p.refine_ahead_s))
        nt["plucked"] = False
        if hi - lo < 5:
            continue
        nxt = notes[i + 1]["start"] if i + 1 < len(notes) else ts + 1
        pk_hi = min(len(env) - 1, at(min(ts + 0.12, nxt + 0.02)))
        pk_lo = max(lo, at(ts - 0.02))
        if pk_hi <= pk_lo:
            continue
        k_peak = pk_lo + int(np.argmax(env[pk_lo : pk_hi + 1]))
        end_trough = min(hi, k_peak)
        if end_trough <= lo:
            continue
        k_trough = lo + int(np.argmin(env[lo : end_trough + 1]))
        base, peak = env[k_trough], env[k_peak]
        if peak < ratio * max(base, 1e-6) or k_peak <= k_trough:
            continue
        seg = env[k_trough : k_peak + 1]
        k = k_trough + int(np.argmax(seg >= base + p.refine_level * (peak - base)))
        if lo <= k <= hi:
            new = k * step / SR
            if i and notes[i - 1]["end"] > new - 1e-9 or i and abs(notes[i - 1]["end"] - ts) < 0.011:
                notes[i - 1]["end"] = max(notes[i - 1]["start"] + 0.02, new)  # no overlap; keep legato joins joined
            nt["start"], nt["plucked"] = new, True
            nt["end"] = max(nt["end"], new + 0.02)
    return notes


def segment_notes(feat, bass_mono, p=None):
    """Notes as dicts {start, end (s), pitch (MIDI), agree, flag}; flag is None, "octave" or "uncertain"."""
    p = p or SegParams()
    f0_c, per_c, f0_p, rms_db, onsets = feat["f0_c"], feat["per_c"], feat["f0_p"], feat["rms_db"], feat["onsets"]
    cqt = feat["cqt"][:60]  # A0..G#5
    low_db, flux = pluck_features(bass_mono)
    n = min(len(f0_c), len(f0_p), len(rms_db), len(low_db))
    f0_c, per_c, f0_p, rms_db, low_db = f0_c[:n], per_c[:n], f0_p[:n], rms_db[:n], low_db[:n]
    rise = attack_rise(low_db, p.rise_before, p.rise_after)
    # pluck strength: 1.0 is the threshold of either view (see pluck_features); flux peaks a frame or two late
    flux = np.pad(flux, (0, max(0, n - len(flux))))[:n]
    flux_near = np.maximum.reduce([flux, np.r_[flux[1:], 0], np.r_[flux[2:], 0, 0]])
    level = rise / p.attack_db
    strong_flux = flux_near / p.flux_thr if p.flux_thr > 0 else 0
    # Splitting a note the tracker held needs strict evidence: the level of the lowest notes ripples with their
    # period, so a jump in level counts only when it is also bright. Keeping apart two notes the tracker already
    # saw as separate needs less: a jump in level is enough, as it always was.
    bright_level = np.minimum(level, flux_near / p.flux_support) if p.flux_support > 0 else level
    pluck = np.maximum(bright_level, strong_flux)
    attack = pluck if p.strict_replucks else np.maximum(level, strong_flux)
    floor = np.percentile(rms_db, 99) - 30
    voiced = (rms_db > floor) & (per_c > p.enter_per)
    midi_c = np.where(voiced, librosa.hz_to_midi(np.maximum(f0_c, 1e-6)), np.nan)
    midi_p = np.where(np.isfinite(f0_p), librosa.hz_to_midi(np.where(np.isfinite(f0_p), f0_p, 1.0)), np.nan)
    rounded = np.where(np.isfinite(midi_c), np.round(midi_c), -1).astype(int)

    boundaries = set(int(o) for o in onsets if 0 <= o < n)
    for t in range(1, n - 4):  # pitch changes that persist at least 40 ms
        if rounded[t] != rounded[t - 1] and rounded[t] >= 0 and np.all(rounded[t : t + 4] == rounded[t]):
            boundaries.add(t)
    starts = sorted(boundaries | {0})
    notes = []
    for s, e in zip(starts, starts[1:] + [n]):
        v = np.isfinite(midi_c[s:e])
        if v.sum() < p.min_voiced_frames:
            continue
        idx = np.nonzero(v)[0] + s
        pitch = int(round(np.median(midi_c[idx])))
        alt = midi_p[idx]
        alt = alt[np.isfinite(alt)]
        p_med = int(round(np.median(alt))) if len(alt) else None
        agree = float(np.mean(np.abs(alt - pitch) <= 0.5)) if len(alt) else 0.0
        flag = None
        sure = float(np.median(per_c[idx])) >= p.trust_per
        if p_med is not None and abs(p_med - pitch) == 12 and not sure:  # the trackers disagree by an octave
            # ...and CREPE is not confident. (When it is, its octave stands: pYIN smooths over jumps and stays on
            # the note before, and the energy test below sides with the lower note whenever that note still rings.)
            low, high = min(p_med, pitch), max(p_med, pitch)
            frames = slice(idx[0], idx[-1] + 1)
            e_low = cqt[low - 21, frames].mean() if 0 <= low - 21 < 60 else 0
            e_high = cqt[high - 21, frames].mean() if 0 <= high - 21 < 60 else 0
            pitch = low if low >= 28 and e_low >= 0.3 * e_high else high
            flag = "octave"
        elif agree < 0.5:
            pc = np.array([cqt[k::12, idx].sum(0) for k in range(12)])
            if np.mean((pc.argmax(0) + 9) % 12 == pitch % 12) < 0.6:  # CQT bin 0 is A
                flag = "uncertain"
        core = idx[:30]  # fundamental far below its own octave: CREPE locked an octave low
        if (
            21 <= pitch
            and pitch - 9 < cqt.shape[0]
            and 20 * np.log10((cqt[pitch - 9, core].mean() + 1e-9) / (cqt[pitch - 21, core].mean() + 1e-9))
            > p.octave_up_db
        ):
            pitch += 12
            flag = "octave"
        start = s if idx[0] - s <= 8 else idx[0]
        notes.append(dict(s=start, last=idx[-1] + 1, pitch=pitch, agree=agree, flag=flag))
    # hold each note while it still sounds: up to the next note, a decay_db drop, or a long unpitched stretch
    unpitched = per_c < p.stay_per
    for i, nt in enumerate(notes):
        limit = notes[i + 1]["s"] if i + 1 < len(notes) else n
        stop = max(floor, rms_db[nt["s"] : nt["last"]].max() - p.decay_db)
        t, run = nt["last"], 0
        while t < limit and rms_db[t] > stop:
            run = run + 1 if unpitched[t] else 0
            if run >= p.unpitched_frames:
                t -= p.unpitched_frames - 1
                break
            t += 1
        nt["e"] = max(t, nt["last"])
    # a held note with a pluck inside it is several notes: fast repeats are too short for the tracker to see each
    # one, so the note before swallows them. Each part takes its own pitch where the tracker has one.
    if p.split_held:
        from scipy.signal import find_peaks

        plucks = find_peaks(pluck, height=1.0, distance=5)[0]
        split = []
        for nt in notes:
            cuts = [int(t) for t in plucks if nt["s"] + p.split_min_frames <= t <= nt["e"] - p.split_min_frames]
            edges = [nt["s"], *cuts, nt["e"]]
            for a, b in zip(edges, edges[1:]):
                part = dict(nt, s=a, e=b, last=min(max(nt["last"], a + 1), b))
                own = midi_c[a:b]
                own = own[np.isfinite(own)]
                if cuts and len(own) >= 3:
                    part["pitch"] = int(round(np.median(own)))
                split.append(part)
        notes = split
    # a repeated pitch without a new pluck continues the previous note
    merged = []
    for nt in notes:
        prev = merged[-1] if merged else None
        if (
            prev
            and nt["pitch"] == prev["pitch"]
            and nt["s"] - prev["e"] <= p.merge_gap_frames
            and attack[max(0, nt["s"] - p.attack_look) : nt["s"] + p.attack_look + 1].max() < 1.0
        ):
            prev["e"] = nt["e"]
            prev["agree"] = min(prev["agree"], nt["agree"])
            prev["flag"] = prev["flag"] or nt["flag"]
        else:
            merged.append(dict(nt))
    for nt in merged:
        nt["start"], nt["end"] = nt.pop("s") * HOP_S, nt.pop("e") * HOP_S
        del nt["last"]
    if p.refine_onsets:
        refine_onsets(merged, bass_mono, p)
    return merged, midi_c, midi_p
