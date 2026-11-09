"""Beat grid, bar phase and quantization; key and chord estimates."""

import librosa
import numpy as np

from .common import NOTE_NAMES, SR, spelling


def beats_and_downbeats(drums, bass, harmony):
    """Tracked pulses, quarter-note tempo, pulses per quarter, and the pulse index of the first downbeat.

    In a rock beat the kick plays beats 1 and 3 and the snare 2 and 4: if kick and snare peak on alternate
    pulses, the pulses are quarter notes; if they share a parity, eighth notes. Beat 1 is the kick pulse where
    chords change and the bass attacks most."""
    sr = 22050
    d = librosa.resample(drums.mean(0), orig_sr=SR, target_sr=sr)
    b = librosa.resample(bass.mean(0), orig_sr=SR, target_sr=sr)
    o = librosa.resample(harmony.mean(0), orig_sr=SR, target_sr=sr)
    tempo, pulses = librosa.beat.beat_track(y=d + b, sr=sr, hop_length=256, units="time", tightness=120)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(pulses) < 16:
        raise ValueError("fewer than 16 beats tracked")
    S = np.abs(librosa.stft(d, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = lambda lo, hi: librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(freqs >= lo) & (freqs < hi)]),
                                                      sr=sr, hop_length=256)
    t_env = librosa.times_like(band(150, 400), sr=sr, hop_length=256)
    at = lambda e: np.interp(pulses, t_env, e)
    sn, kick = at(band(150, 400)), at(band(30, 120))
    bass_on = at(librosa.onset.onset_strength(y=b, sr=sr, hop_length=256))
    parity = lambda v: int(v[1::2].mean() > v[0::2].mean())
    per_quarter = 1 if parity(kick) != parity(sn) else 2
    change = np.zeros(len(pulses))  # harmonic change entering each pulse
    for y in (o, b):
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        tc = librosa.times_like(c, sr=sr, hop_length=512)
        sync = np.array([c[:, (tc >= p0) & (tc < p1)].mean(1) if ((tc >= p0) & (tc < p1)).any() else np.zeros(12)
                         for p0, p1 in zip(pulses[:-1], list(pulses[1:]))] + [np.zeros(12)])
        sync /= np.linalg.norm(sync, axis=1, keepdims=True) + 1e-9
        change[1:] += 1 - (sync[1:] * sync[:-1]).sum(1)
    bar = 4 * per_quarter
    if per_quarter == 1:
        candidates = [p for p in range(bar) if p % 2 == parity(kick)]
    else:  # snare on 2 and 4 = pulses p+2 and p+6
        backbeat = lambda p: sn[(p + 2) % bar::bar].mean() + sn[(p + 6) % bar::bar].mean()
        candidates = [p for p in range(bar) if backbeat(p) >= 0.9 * max(backbeat(q) for q in range(bar))]
    norm = lambda v: v / (v.mean() + 1e-9)
    score = {p: norm(change)[p::bar].mean() + 0.5 * norm(kick)[p::bar].mean() + 0.5 * norm(bass_on)[p::bar].mean()
             for p in candidates}
    phase = max(score, key=score.get)
    return tempo / per_quarter, pulses, per_quarter, phase


def beat_grid(pulses, per_quarter, phase, end):
    """16th-note times covering 0..end (tracked pulses extended at their median spacing) and the grid index of
    the earliest downbeat."""
    sub = 4 // per_quarter
    step = float(np.median(np.diff(pulses)))
    n_pre = int(np.ceil(pulses[0] / step)) + 1
    n_post = int(np.ceil((end - pulses[-1]) / step)) + 2
    ext = np.r_[pulses[0] - step * np.arange(n_pre, 0, -1), pulses, pulses[-1] + step * np.arange(1, n_post)]
    grid = np.concatenate([np.linspace(b0, b1, sub + 1)[:-1] for b0, b1 in zip(ext[:-1], ext[1:])] + [ext[-1:]])
    first_down = (phase + n_pre) % (4 * per_quarter)
    return grid, first_down * sub


def quantize(notes, grid, origin):
    """Snap to the 16th grid; 4/4 bars counted from grid index `origin` (bar 0 is a pickup)."""
    for nt in notes:
        gs = int(np.argmin(np.abs(grid - nt["start"])))
        ge = int(np.argmin(np.abs(grid - nt["end"])))
        rel = gs - origin
        nt["grid_start"], nt["grid_len"] = rel, max(1, ge - gs)
        nt["bar"] = rel // 16 + 1
        nt["beat"] = (rel % 16) // 4 + 1
        nt["sixteenth"] = rel % 4 + 1
    out = []
    for nt in notes:  # a re-attack of the same pitch inside one sixteenth is one note
        if out and nt["pitch"] == out[-1]["pitch"] and nt["grid_start"] == out[-1]["grid_start"]:
            prev = out[-1]
            prev["end"] = nt["end"]
            prev["grid_len"] = max(1, int(np.argmin(np.abs(grid - nt["end"]))) - (prev["grid_start"] + origin))
            prev["flag"] = prev["flag"] or nt["flag"]
            continue
        out.append(nt)
    return out


TRIADS = {"": (0, 4, 7), "m": (0, 3, 7)}


def chords_per_bar(harmony, notes, grid, origin):
    """Rough chord per bar: bar-averaged chroma of the guitars/keys matched to major/minor triads, with a small
    bonus for the bass note."""
    y = librosa.resample(harmony.mean(0), orig_sr=SR, target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512)
    t = librosa.times_like(chroma, sr=22050, hop_length=512)
    bar_starts = grid[origin::16]
    out = []
    for k, (b0, b1) in enumerate(zip(bar_starts[:-1], bar_starts[1:])):
        sel = (t >= b0) & (t < b1)
        if not sel.any():
            continue
        c = chroma[:, sel].mean(1)
        bass_pc = np.zeros(12)
        for nt in notes:
            ov = min(nt["end"], b1) - max(nt["start"], b0)
            if ov > 0:
                bass_pc[nt["pitch"] % 12] += ov
        best = None
        for root in range(12):
            for q, iv in TRIADS.items():
                tmpl = np.zeros(12)
                tmpl[[(root + i) % 12 for i in iv]] = 1
                score = np.dot(c, tmpl) / (np.linalg.norm(c) * np.linalg.norm(tmpl) + 1e-9)
                score += 0.15 * (bass_pc[root] / (bass_pc.sum() + 1e-9))
                if best is None or score > best[0]:
                    best = (score, root, q)
        names = spelling(best[1], best[2] == "")
        out.append(dict(bar=k + 1, root=best[1], quality=best[2], score=float(best[0]), names=names,
                        chord=names[best[1]] + best[2]))
    return out, chroma


def estimate_key(chroma):
    """(name, root pitch class, is_major) from Krumhansl profiles."""
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    c = chroma.mean(1)
    best = max((np.corrcoef(c, np.roll(prof, r))[0, 1], r, prof is major) for r in range(12) for prof in (major, minor))
    _, root, is_major = best
    names = spelling(root, is_major)
    return names[root] + (" major" if is_major else " minor"), root, is_major
