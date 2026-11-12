"""Two notes played together (a double stop), found in the spectrum of a stretch of the bass track.

The pitch trackers follow one note at a time, and with two sounding they follow the upper one, flip between the
two, or lose the pitch altogether. So a stretch is looked at again here, by its spectral peaks: the note that
explains most of them is found first, and what it leaves unexplained must be a note of its own to count as a
second one. The tests are strict on purpose. A second note that is not there is worse on a sheet than one that
is missing, and a bass track separated from a mix carries other instruments' notes, the kick drum and the note
before still ringing.

An octave is never written as two notes: every partial of the upper note is also one of the lower note's, so
the two cannot be told from one low note, and it is not something bass players do. Notes a step apart are
turned down as well: that is a slide, a bend or the neighbouring note of the line caught in the same stretch.
"""

import numpy as np

from .common import HOP_S, SR

LOWEST, HIGHEST = 27, 67  # the notes looked for: D#1 to G4


def hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def peaks_of(mono, start, end, lead=0.03, longest=0.5, depth=30.0, lo=32.0, hi=1300.0):
    """Spectral peaks [(Hz, dB below the loudest)] of mono[start + lead : min(end, start + longest)] that lie
    within `depth` dB of the loudest. `lead` skips the pluck's noise."""
    from scipy.signal import find_peaks

    seg = mono[int((start + lead) * SR) : int(min(end, start + longest) * SR)].astype(np.float64)
    if len(seg) < int(0.09 * SR):
        return []
    size = 1 << 16
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), size))
    freq = np.fft.rfftfreq(size, 1 / SR)
    keep = (freq >= lo) & (freq <= hi)
    spec, freq = spec[keep], freq[keep]
    if spec.max() <= 0:
        return []
    db = 20 * np.log10(spec / spec.max() + 1e-12)
    apart = max(1, int(1.5 * (SR / len(seg)) / (freq[1] - freq[0])))  # two peaks are at least a main lobe apart
    idx, _ = find_peaks(db, height=-depth, distance=apart, prominence=6)
    return [(float(freq[i]), float(db[i])) for i in idx]


def explain(peaks, f0, tol=0.03, partials=10):
    """{partial number: index of the peak it matches} for a note at f0."""
    out = {}
    for k in range(1, partials + 1):
        near = [i for i, (f, _) in enumerate(peaks) if abs(f / (k * f0) - 1) <= tol]
        if near:
            out[k] = max(near, key=lambda i: peaks[i][1])
    return out


def strength(peaks, f0, used=()):
    """(How much of the peaks a note at f0 accounts for, its partials). The amount is the linear amplitude of its
    first six partials, leaving out the peaks in `used`; it is 0 when this is not a note: fewer than two partials
    sound, neither of the first two does, or only even ones do (then the note is an octave up)."""
    mine = {k: i for k, i in explain(peaks, f0).items() if i not in used}
    low = [k for k in mine if k <= 6]
    if len(low) < 2 or not (1 in mine or 2 in mine) or not any(k % 2 for k in low):
        return 0.0, mine
    return sum(10 ** (peaks[mine[k]][1] / 20) * (1.0 if k <= 4 else 0.5) for k in low), mine


def candidates(peaks, floor=-14.0):
    """The notes whose first four partials could be one of the stronger peaks."""
    out = set()
    for f, db in peaks:
        if db >= floor:
            for k in (1, 2, 3, 4):
                exact = 69 + 12 * np.log2(f / k / 440.0)
                m = int(round(exact))
                if LOWEST <= m <= HIGHEST and abs(exact - m) < 0.35:
                    out.add(m)
    return sorted(out)


def sieve(peaks, given=None, second=0.35, strong=-14.0):
    """The notes in a set of peaks, low to high: one, or two when what the first leaves unexplained is a note of
    its own. `given` is the pitch a tracker heard there; it is the first note if the peaks bear it out. The second
    must be worth `second` of the first, have two of its first three partials among the unexplained peaks within
    `strong` dB of the loudest, and not be the first note's octave or one of its partials."""
    cands = candidates(peaks)
    scored = {m: strength(peaks, float(hz(m))) for m in cands}
    scored = {m: s for m, s in scored.items() if s[0] > 0}
    if not scored:
        return []
    first = given if given in scored else max(scored, key=lambda m: scored[m][0])
    worth, mine = scored[first]
    used = set(mine.values())
    best = None
    for m in cands:
        if (m - first) % 12 == 0 or abs(m - first) in (19, 28, 31):
            continue
        s, own = strength(peaks, float(hz(m)), used)
        clear = [k for k in (1, 2, 3) if k in own and peaks[own[k]][1] >= strong]
        if s >= second * worth and len(clear) >= 2 and (best is None or s > best[1]):
            best = (m, s)
    return sorted([first, best[0]]) if best else [first]


def _spectrum(mono, a_s, b_s):
    seg = mono[max(0, int(a_s * SR)) : int(b_s * SR)].astype(np.float64)
    if len(seg) < int(0.05 * SR):
        return None
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 15))
    return np.fft.rfftfreq(1 << 15, 1 / SR), 20 * np.log10(spec / len(seg) + 1e-12)  # comparable between stretches


def _level(spectrum, f):
    freq, db = spectrum
    near = (freq >= f * 0.972) & (freq <= f * 1.029)
    return float(db[near].max()) if near.any() else -120.0


def notes_in(mono, start, end, given=None, rise=5.0, hold=20.0, second=0.35, strong=-14.0, rest=None, own=3.0):
    """The notes sounding together in mono[start:end], low to high: [] (no pitch to be found), [pitch] or two.

    A second note stands only if each of the two has a partial the other cannot explain, and that partial
    was at least `rise` dB quieter just before the stretch (a note still ringing, or another instrument holding
    one, was there before) and has not dropped by more than `hold` dB in the second half of the stretch (the
    tail of the note before, or the pluck, is gone by then). With `rest`, the record without the bass track,
    it must also be `own` dB louder in the bass track than there: a kick drum or a piano note that leaked into
    the bass track is louder where it belongs."""
    peaks = peaks_of(mono, start, end)
    got = sieve(peaks, given, second, strong)
    if len(got) < 2:
        return got
    alone = [given if given in got else got[0]]
    if got[1] - got[0] <= 2:
        return alone
    stop = min(end, start + 0.5)
    middle = (start + 0.03 + stop) / 2
    early, before, late = (
        _spectrum(mono, start + 0.03, middle),
        _spectrum(mono, start - 0.16, start - 0.02),
        _spectrum(mono, middle, stop),
    )
    whole = _spectrum(mono, start + 0.03, stop)
    elsewhere = None if rest is None else _spectrum(rest, start + 0.03, stop)
    for note, other in (got, got[::-1]):
        f0 = float(hz(note))
        mine, theirs = explain(peaks, f0), set(explain(peaks, float(hz(other))).values())
        mine_only = [k for k in (1, 2, 3) if k in mine and mine[k] not in theirs and peaks[mine[k]][1] >= strong - 4]
        if not mine_only:
            return [other]
        if early and before and max(_level(early, k * f0) - _level(before, k * f0) for k in mine_only) < rise:
            return [other]
        if early and late and max(_level(late, k * f0) - _level(early, k * f0) for k in mine_only) < -hold:
            return [other]
        if whole and elsewhere:
            if float(np.median([_level(whole, k * f0) - _level(elsewhere, k * f0) for k in mine_only])) < own:
                return [other]
    return got


def rose(mono, t, pitch, rise=5.0, beside=None):
    """True when a note of this pitch starts at t: one of its first three partials is `rise` dB louder in the
    120 ms after t than just before. False when it was already sounding (or when there is too little to tell).
    Partials it shares with a note `beside` it are left out: those rise when the other note starts."""
    after, before = _spectrum(mono, t + 0.03, t + 0.15), _spectrum(mono, t - 0.16, t - 0.02)
    if not after or not before:
        return False
    f0 = float(hz(pitch))
    shared = [] if beside is None else [j * float(hz(beside)) for j in range(1, 7)]
    own = [k for k in (1, 2, 3) if all(abs(k * f0 / f - 1) > 0.03 for f in shared)] or [1, 2, 3]
    return max(_level(after, k * f0) - _level(before, k * f0) for k in own) >= rise


def join_pieces(notes, bass_mono, rise):
    """A pair and a single note beside it are one event when the single note is one of the pair's two and does
    not start anew there: the tracker found the pitch for part of the stretch only, or the two notes' beating
    looked like a pluck."""
    out = []
    for n in notes:
        prev = out[-1] if out else None
        if prev and n["start"] - prev["end"] <= 0.06:
            if prev.get("also") and not n.get("also") and n["pitch"] in (prev["pitch"], *prev["also"]):
                other = next(q for q in (prev["pitch"], *prev["also"]) if q != n["pitch"])
                if not rose(bass_mono, n["start"], n["pitch"], rise, other):
                    prev["end"] = n["end"]
                    continue
            short = prev["end"] - prev["start"] < 0.15
            if n.get("also") and not prev.get("also") and short and prev["pitch"] in (n["pitch"], *n["also"]):
                other = next(q for q in (n["pitch"], *n["also"]) if q != prev["pitch"])
                if not rose(bass_mono, n["start"], prev["pitch"], rise, other):
                    out[-1] = dict(n, start=prev["start"])
                    continue
        out.append(dict(n))
    return out


def find_together(notes, bass_mono, sounding, onsets, attacked, p, rest=None):
    """Look at every stretch of the bass track again for two notes played together.

    `notes` are the single notes already found ({start, end, pitch, ...}, in order); `sounding` says per 10 ms
    frame whether the track is loud enough to hold a note, `onsets` are the frames where something starts, and
    `attacked(frame)` whether a pluck starts there; `rest` is the record without the bass track, when there is
    one (see notes_in). Returns the notes with `also` (the pitches sounding with a
    note, above it) set where two were found:
      - a note with a second one under or over it becomes the lower of the two, with the upper in `also`;
      - notes that follow each other without a new pluck and are the two notes of one pair are one event
        (the tracker flipped between them);
      - a stretch that sounds but has no note, because the tracker lost the pitch, gets one if it holds a pair."""
    frame = lambda t: int(round(t / HOP_S))
    look = lambda a, b, given=None: notes_in(
        bass_mono,
        a,
        b,
        given,
        p.together_rise,
        p.together_hold,
        p.together_second,
        p.together_strong,
        rest,
        p.together_own,
    )
    out, k = [], 0
    while k < len(notes):
        group = [notes[k]]
        while (
            k + len(group) < len(notes)
            and notes[k + len(group)]["start"] - group[-1]["end"] <= 0.06
            and not attacked(frame(notes[k + len(group)]["start"]))
        ):
            group.append(notes[k + len(group)])
        k += len(group)
        if len(group) > 1 and len({n["pitch"] for n in group}) == 2:
            pair = (
                look(group[0]["start"], group[-1]["end"])
                if group[-1]["end"] - group[0]["start"] >= p.together_min_s
                else []
            )
            if len(pair) == 2 and {n["pitch"] for n in group} == set(pair):
                first = dict(group[0], end=group[-1]["end"], pitch=pair[0], also=[pair[1]])
                first["agree"] = min(n["agree"] for n in group)
                out.append(first)
                continue
        for n in group:
            pair = look(n["start"], n["end"], n["pitch"]) if n["end"] - n["start"] >= p.together_min_s else []
            out.append(dict(n, pitch=pair[0], also=[pair[1]]) if len(pair) == 2 and n["pitch"] in pair else n)
    # stretches that sound but have no note
    filled, edges = [], [0.0] + [t for n in out for t in (n["start"], n["end"])] + [len(sounding) * HOP_S]
    for a, b in zip(edges[::2], edges[1::2]):
        fa, fb = frame(a), min(len(sounding), frame(b))
        cuts = [fa] + [int(t) for t in onsets if fa < t < fb] + [fb]
        for s, e in zip(cuts, cuts[1:]):
            while s < e and not sounding[s]:
                s += 1
            stop = s
            while stop < e and sounding[stop]:
                stop += 1
            if (stop - s) * HOP_S >= max(p.together_min_s, 0.15):
                pair = look(s * HOP_S, stop * HOP_S)
                if len(pair) == 2:
                    filled.append(
                        dict(start=s * HOP_S, end=stop * HOP_S, pitch=pair[0], also=[pair[1]], agree=0.0, flag=None)
                    )
    return join_pieces(sorted(out + filled, key=lambda n: n["start"]), bass_mono, p.together_rise)
