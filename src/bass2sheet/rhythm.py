"""Beat grid, bar phase and quantization; key and chord estimates."""

import librosa
import numpy as np

from .common import SR, spelling


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
    band = lambda lo, hi: librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S[(freqs >= lo) & (freqs < hi)]), sr=sr, hop_length=256
    )
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
        sync = np.array(
            [
                c[:, (tc >= p0) & (tc < p1)].mean(1) if ((tc >= p0) & (tc < p1)).any() else np.zeros(12)
                for p0, p1 in zip(pulses[:-1], list(pulses[1:]))
            ]
            + [np.zeros(12)]
        )
        sync /= np.linalg.norm(sync, axis=1, keepdims=True) + 1e-9
        change[1:] += 1 - (sync[1:] * sync[:-1]).sum(1)
    bar = 4 * per_quarter
    if per_quarter == 1:
        candidates = [p for p in range(bar) if p % 2 == parity(kick)]
    else:  # snare on 2 and 4 = pulses p+2 and p+6
        backbeat = lambda p: sn[(p + 2) % bar :: bar].mean() + sn[(p + 6) % bar :: bar].mean()
        candidates = [p for p in range(bar) if backbeat(p) >= 0.9 * max(backbeat(q) for q in range(bar))]
    norm = lambda v: v / (v.mean() + 1e-9)
    score = {
        p: norm(change)[p::bar].mean() + 0.5 * norm(kick)[p::bar].mean() + 0.5 * norm(bass_on)[p::bar].mean()
        for p in candidates
    }
    phase = max(score, key=score.get)
    return tempo / per_quarter, pulses, per_quarter, phase


def model_beats(mix_mono, device="cpu", cache_path=None):
    """Beats and downbeats in seconds from Beat This! (Foscarin, Schlüter and Widmer, ISMIR 2024) on the full mix.
    The model downloads on first use (77 MB). Results are cached at `cache_path`."""
    import os

    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["beats"], d["downbeats"]
    from beat_this.inference import Audio2Beats

    try:
        beats, downbeats = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)(mix_mono, SR)
    except Exception:
        if device == "cpu":
            raise
        beats, downbeats = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)(mix_mono, SR)
    beats, downbeats = np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
    if cache_path:
        from .common import atomic_savez

        atomic_savez(cache_path, beats=beats, downbeats=downbeats)
    return beats, downbeats


def grid_from_model_beats(beats, downbeats):
    """Tempo, pulses, pulses per quarter (1), the pulse index of beat 1, and how well the downbeats fit 4/4.

    The score is written in 4/4, so beat 1 is the phase (0-3) that most model downbeats fall on. The fit is the
    share of downbeats on that phase: near 1 for a 4/4 song, about 0.25-0.5 for 3/4, 6/8 or changing meters."""
    if len(beats) < 16:
        raise ValueError("fewer than 16 beats tracked")
    tempo = float(60.0 / np.median(np.diff(beats)))
    idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats], dtype=int)
    near = np.array([abs(beats[k] - d) < 0.07 for k, d in zip(idx, downbeats)], dtype=bool)
    idx = idx[near] if len(idx) else idx
    if len(idx) < 2:
        return tempo, beats, 1, 0, None
    counts = np.bincount(idx % 4, minlength=4)
    return tempo, beats, 1, int(counts.argmax()), float(counts.max() / counts.sum())


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
            if nt.get("also") or prev.get("also"):
                prev["also"] = sorted({*prev.get("also", []), *nt.get("also", [])})
            continue
        out.append(nt)
    return out


def place(nt, g0, g1):
    """A copy of a quantized note written at sixteenths g0..g1 (counted from bar 1)."""
    return dict(nt, grid_start=g0, grid_len=g1 - g0, bar=g0 // 16 + 1, beat=(g0 % 16) // 4 + 1, sixteenth=g0 % 4 + 1)


def simplify(notes, busy=0.34, close=2):
    """The same line written for easy reading: in a bar that moves in eighths, the few notes on an odd sixteenth
    are moved to an eighth, and rests of up to `close` sixteenths are closed by holding the note before them.

    A bar where more than `busy` of the notes start on odd sixteenths is a sixteenth-note line and keeps its
    onsets. Elsewhere a held note that anticipates the next eighth moves onto it; a one-sixteenth note moves back
    to the eighth before it, or is dropped when another note already starts there. Pitches never change."""
    notes = sorted(notes, key=lambda n: (n["grid_start"], n["start"]))
    spans = []  # [g0, g1, note]: cut where the next note starts, as the score does
    for k, nt in enumerate(notes):
        nxt = notes[k + 1]["grid_start"] if k + 1 < len(notes) else None
        if nxt is not None and nxt == nt["grid_start"]:
            continue  # the earlier of two notes on one sixteenth is an ornament: not kept
        end = nt["grid_start"] + nt["grid_len"]
        spans.append([nt["grid_start"], max(nt["grid_start"] + 1, end if nxt is None else min(end, nxt)), nt])
    by_bar = {}
    for sp in spans:
        by_bar.setdefault(sp[0] // 16, []).append(sp)
    taken = {sp[0] for sp in spans}
    kept = []
    for bar in sorted(by_bar):
        group = by_bar[bar]
        odd = [sp for sp in group if sp[0] % 2 == 1]
        if not odd or len(odd) > busy * len(group):
            kept += group
            continue
        for sp in group:
            g0, g1, _ = sp
            if g0 % 2 == 0:
                kept.append(sp)
            elif g1 - g0 >= 2 and g0 + 1 not in taken:  # an anticipation: onto the eighth it leans on
                taken.discard(g0)
                taken.add(g0 + 1)
                sp[0] = g0 + 1
                kept.append(sp)
            elif g0 - 1 not in taken:  # a short note: back onto the eighth before it
                taken.discard(g0)
                taken.add(g0 - 1)
                sp[0], sp[1] = g0 - 1, max(g1, g0 + 1)
                kept.append(sp)
            else:  # no room on either eighth: a passing sixteenth, left out
                taken.discard(g0)
    kept.sort(key=lambda sp: sp[0])
    out = []
    for k, (g0, g1, nt) in enumerate(kept):
        nxt = kept[k + 1][0] if k + 1 < len(kept) else None
        if nxt is not None and (g1 > nxt or nxt - g1 <= close):
            g1 = nxt
        if out and out[-1]["pitch"] == nt["pitch"] and out[-1]["grid_start"] == g0:
            continue
        out.append(place(nt, g0, max(g1, g0 + 1)))
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
        out.append(
            dict(
                bar=k + 1,
                root=best[1],
                quality=best[2],
                score=float(best[0]),
                names=names,
                chord=names[best[1]] + best[2],
            )
        )
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
