"""Someone else's transcription of a song, lined up with the recording so its bass part can stand as an answer key.

For most well-known songs somebody has written the bass line down: a MIDI file of the whole arrangement, a Guitar
Pro tab, a score. Such a file counts bars and beats, not the seconds of your recording, and it may be in another
key, at another tempo, with an intro of another length or a verse more or less. `align` finds, for every moment of
the file, the moment of the recording it belongs to, or that it belongs to none:

  1. Both are turned into chroma (how strongly each of the twelve pitch classes sounds, every tenth of a second),
     once for everything that sounds and once for the bass alone: the recording from its sound and from its
     lowest three octaves, the file from all of its pitched notes and from its bass part.
  2. A first look on coarse frames says which transposition fits and how the file's tempo relates to the
     recording's: the one, of all that are tried, at which the two line up best.
  3. With the file brought to the recording's tempo, the two are lined up again by rules that fit music: they run
     side by side, the tempo may drift slowly, and where one has a stretch the other lacks, that stretch is
     jumped over rather than stretched to fit (a local alignment with gaps, Gotoh's, on the chroma similarity).
     What lies between two jumps is a `segment`.
  4. Inside each segment the timing is settled by the onsets: the file's note starts, drums included, are slid
     against the moments something starts in the recording, along one smooth course through the segment.

It is a person's reading of the song, not the truth, and the recording decides nothing about its notes: only
where they fall in time. dev/reference_eval.py measures how close the timing comes on songs whose timing is known.
"""

import os
import subprocess
import tempfile

import numpy as np

from .common import Bass2SheetError

HOP = 0.1  # s: the frames the two are lined up on
ASR, AHOP = 22050, 110  # the recording's onsets are looked for at this rate, every 5 ms
FINE = AHOP / ASR
THETA = 0.2  # a frame pair gains its similarity above this, and loses below it
SLIP = 0.75  # what a tenth of a second of tempo drift costs, in such gains
JUMP = 10.0  # ... and what leaving out a stretch of one of the two costs
TRUSTED = 0.15  # s: how far the onsets may move the first map before it costs them
MIDI_ENDINGS = (".mid", ".midi", ".smf", ".kar")


def read(path, mscore=None):
    """The file as a pretty_midi.PrettyMIDI. A MIDI file is read as it is; anything else MuseScore can open (Guitar
    Pro, MusicXML, its own files) is turned into MIDI by MuseScore first."""
    import pretty_midi

    if not os.path.isfile(path):
        raise Bass2SheetError(f"no such file: {path}")
    if path.lower().endswith(MIDI_ENDINGS):
        try:
            return pretty_midi.PrettyMIDI(path)
        except Exception as e:  # pretty_midi and mido raise many kinds
            raise Bass2SheetError(f"{os.path.basename(path)} is not a MIDI file that can be read ({e})") from e
    from .score import find_mscore

    binary = find_mscore(mscore)
    if not binary:
        raise Bass2SheetError(
            f"{os.path.basename(path)} is not a MIDI file, and MuseScore, which could turn it into one, "
            "is not installed"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "reference.mid")
        done = subprocess.run([binary, "-o", out, path], capture_output=True, text=True, timeout=300)
        if done.returncode or not os.path.exists(out):
            raise Bass2SheetError(f"MuseScore could not read {os.path.basename(path)}: {done.stderr.strip()[-300:]}")
        return pretty_midi.PrettyMIDI(out)


def bass_track(pm):
    """(index of the instrument that is the bass, [the others that could have been]). A bass is called one, by its
    track's name or its General MIDI program, and plays low; with none called so, it is the lowest line."""
    rows = []
    for k, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < 8:
            continue
        median = float(np.median([n.pitch for n in inst.notes]))
        named = "bass" in (inst.name or "").lower() or 32 <= inst.program <= 39
        rows.append((named and median <= 55, len(inst.notes), -median, k))
    if not rows:
        raise Bass2SheetError("the file has no pitched part with notes in it")
    called = sorted((r for r in rows if r[0]), reverse=True)
    if called:
        return called[0][3], [r[3] for r in called[1:]]
    lowest = sorted(rows, key=lambda r: -r[2])  # by median pitch, lowest first
    return lowest[0][3], [r[3] for r in lowest[1:3]]


def _unit(a):
    """Columns centred on the piece's own mean and brought to length one: what is the same all through a song (its
    key) says nothing about where one is in it."""
    a = a - a.mean(axis=1, keepdims=True)
    return (a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-9)).astype(np.float32)


def _folded(roll, shift):
    """A piano roll (128 pitches by frames) as chroma, `shift` semitones up. Each note also sounds a little at its
    third and fifth partials, as it does in a recording."""
    roll = np.log1p(roll / 40.0)
    out = np.zeros((12, roll.shape[1]))
    for pitch in np.flatnonzero(roll.any(axis=1)):
        pc = (pitch + shift) % 12
        out[pc] += roll[pitch]
        out[(pc + 7) % 12] += 0.3 * roll[pitch]
        out[(pc + 4) % 12] += 0.1 * roll[pitch]
    return out


def file_parts(pm, hop, n, shift=0, bass=None):
    """What the file sounds like at frames `hop` seconds of file time apart, moved `shift` semitones up, as chroma:
    of all its pitched notes, and of its bass part alone (instrument `bass`; left out when there is none)."""
    times = np.arange(n) * hop
    parts = [_folded(pm.get_piano_roll(times=times), shift)]
    if bass is not None:
        parts.append(_folded(pm.instruments[bass].get_piano_roll(times=times), shift))
    return parts


def sound_parts(y22, hop, low=True):
    """What a recording (mono, at ASR) sounds like on frames `hop` seconds apart, as chroma: of everything in it,
    and of its lowest three octaves alone, where the bass is."""
    import librosa

    n = int(len(y22) / ASR / hop)
    edges = (np.arange(n + 1) * hop * ASR / 512).astype(int)

    def framed(c):
        edges_ = np.minimum(edges, c.shape[1])
        out = np.zeros((12, n))
        for k in range(n):
            if edges_[k + 1] > edges_[k]:
                out[:, k] = c[:, edges_[k] : edges_[k + 1]].mean(axis=1)
        return out

    c1 = librosa.note_to_hz("C1")
    parts = [framed(librosa.feature.chroma_cqt(y=y22, sr=ASR, hop_length=512, fmin=c1, n_octaves=7))]
    if low:
        deep = np.abs(librosa.cqt(y22, sr=ASR, hop_length=512, fmin=c1, n_bins=36, bins_per_octave=12))
        parts.append(framed(deep[:12] + deep[12:24] + deep[24:]))
    return parts


def _side_by_side():
    """The alignment itself, compiled on first use: the best-scoring way to run two sequences side by side, with a
    cost for drifting by a frame (`slip`) and a larger one for leaving a stretch of either out (`jump`). Returns
    the frame pairs that are matched, in order; a gap in either column is a jump."""
    import numba

    @numba.njit(cache=True)
    def run(S, slip, jump):
        n, m = S.shape
        low = np.float32(-1e9)
        M = np.zeros((n, m), np.float32)
        X = np.full((n, m), low, np.float32)  # file frame i left out, recording at j
        Y = np.full((n, m), low, np.float32)  # recording frame j left out, file at i
        bm = np.zeros((n, m), np.int8)
        bx = np.zeros((n, m), np.int8)
        by = np.zeros((n, m), np.int8)
        best, bi, bj = np.float32(0.0), 0, 0
        for i in range(n):
            for j in range(m):
                v, b = np.float32(0.0), 0
                if i > 0 and j > 0:
                    if M[i - 1, j - 1] > v:
                        v, b = M[i - 1, j - 1], 1
                    if X[i - 1, j - 1] > v:
                        v, b = X[i - 1, j - 1], 4
                    if Y[i - 1, j - 1] > v:
                        v, b = Y[i - 1, j - 1], 5
                    if j > 1 and M[i - 1, j - 2] - slip > v:
                        v, b = M[i - 1, j - 2] - slip, 2
                    if i > 1 and M[i - 2, j - 1] - slip > v:
                        v, b = M[i - 2, j - 1] - slip, 3
                M[i, j] = v + S[i, j]
                bm[i, j] = b
                if i > 0:
                    if M[i - 1, j] - jump >= X[i - 1, j]:
                        X[i, j], bx[i, j] = M[i - 1, j] - jump, 0
                    else:
                        X[i, j], bx[i, j] = X[i - 1, j], 1
                if j > 0:
                    v, b = M[i, j - 1] - jump, 0
                    if Y[i, j - 1] > v:
                        v, b = Y[i, j - 1], 1
                    if X[i, j - 1] - jump > v:
                        v, b = X[i, j - 1] - jump, 2
                    Y[i, j], by[i, j] = v, b
                if M[i, j] > best:
                    best, bi, bj = M[i, j], i, j
        pairs = np.zeros((n + m, 2), np.int64)
        k, i, j, state = 0, bi, bj, 0
        while i >= 0 and j >= 0:
            if state == 0:
                pairs[k, 0], pairs[k, 1] = i, j
                k += 1
                b = bm[i, j]
                if b == 0:
                    break
                if b == 1:
                    i, j = i - 1, j - 1
                elif b == 2:
                    i, j = i - 1, j - 2
                elif b == 3:
                    i, j = i - 2, j - 1
                elif b == 4:
                    i, j, state = i - 1, j - 1, 1
                else:
                    i, j, state = i - 1, j - 1, 2
            elif state == 1:
                b = bx[i, j]
                i -= 1
                if b == 0:
                    state = 0
            else:
                b = by[i, j]
                j -= 1
                if b == 0:
                    state = 0
                elif b == 2:
                    state = 1
        return pairs[:k][::-1], best

    return run


_run = None


def side_by_side(S, slip=SLIP, jump=JUMP):
    global _run
    if _run is None:
        _run = _side_by_side()
    pairs, score = _run(np.ascontiguousarray(S, dtype=np.float32), np.float32(slip), np.float32(jump))
    return pairs, float(score)


def _similar(mine, theirs):
    """Similarity of every frame of the file to every frame of the recording: the mean over their parts (see
    file_parts), averaged over half a second along the diagonals. A moment is like another when what leads up to
    it and what follows are alike too."""
    k = min(len(mine), len(theirs))
    s = sum(_unit(a).T @ _unit(b) for a, b in zip(mine[:k], theirs[:k])) / k
    out = np.zeros_like(s)
    weight = np.zeros_like(s)
    for d in range(-2, 3):
        i0, j0 = max(0, d), max(0, d)
        i1, j1 = s.shape[0] + min(0, d), s.shape[1] + min(0, d)
        out[i0 - d : i1 - d, j0 - d : j1 - d] += s[i0:i1, j0:j1]
        weight[i0 - d : i1 - d, j0 - d : j1 - d] += 1
    return out / weight


def _first_look(pm, y22, length, bass):
    """(semitones the file is to be moved up to be in the recording's key, seconds of recording to a second of
    file), found on coarse frames by trying: the three transpositions whose notes fit the recording's best, each
    at every tempo from under half to over twice the file's own, lined up as `align` does it. The best one's
    tempo is then read off the way its frames run."""
    hop = 0.25
    sound = sound_parts(y22, hop)
    whole = file_parts(pm, hop, max(8, int(length / hop)))[0].mean(axis=1)
    fit = [float(np.corrcoef(np.roll(whole, k), sound[0].mean(axis=1))[0, 1]) for k in range(12)]
    best = (-1.0, 0, 1.0, None)
    for ratio in np.geomspace(0.45, 1 / 0.45, 65):
        mine = file_parts(pm, hop / ratio, max(8, int(length * ratio / hop)), 0, bass)
        for shift in np.argsort(fit)[-3:]:
            pairs, score = side_by_side(_similar([np.roll(part, shift, axis=0) for part in mine], sound) - THETA)
            if score > best[0]:
                best = (score, int(shift), float(ratio), pairs)
    _, shift, ratio, pairs = best
    if pairs is None or len(pairs) < 16:
        raise Bass2SheetError("the file and the recording have nothing in common that could be lined up")
    runs = np.split(pairs, np.flatnonzero((np.diff(pairs[:, 0]) > 2) | (np.diff(pairs[:, 1]) > 2)) + 1)
    longest = max(runs, key=len)
    if len(longest) >= 40:  # how the recording's frames run against the file's, over the longest stretch
        ratio *= float(np.polyfit(longest[:, 0], longest[:, 1], 1)[0])
    return (shift if shift <= 6 else shift - 12), ratio


def _onsets(y22):
    """How strongly something starts in the recording, every FINE seconds, above its own running level."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    env = librosa.onset.onset_strength(y=y22, sr=ASR, hop_length=AHOP, n_fft=1024, lag=2, max_size=1)
    return np.maximum(env - uniform_filter1d(env, size=int(0.5 / FINE)), 0).astype(np.float32)


def _beats_of(pm):
    """(when, how much it counts) for every note start of the file, in order: the starts that are heard as starts
    in a recording count most. Kick, snare and the bass count fully, other drums and other pitched notes less."""
    rows = []
    for inst in pm.instruments:
        for n in inst.notes:
            if inst.is_drum:
                rows.append((n.start, 1.0 if n.pitch in (35, 36, 38, 40) else 0.5))
            else:
                rows.append((n.start, 1.0 if 32 <= inst.program <= 39 else 0.3))
    rows.sort()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=np.float32)


def _fit(starts, weights, env, reach):
    """How well a stretch of note starts (seconds of the recording, each with its weight) falls on the recording's
    onsets when it is moved by each shift from -reach to +reach, FINE seconds apart; None where there is too
    little to go by."""
    from scipy.ndimage import gaussian_filter1d

    if len(starts) < 8:
        return None
    r = int(round(reach / FINE))
    k0 = int(starts.min() / FINE) - r - 20
    train = np.zeros(int(starts.max() / FINE) + r + 20 - k0, np.float32)
    np.add.at(train, np.round(starts / FINE).astype(int) - k0, weights)
    train = gaussian_filter1d(train, 2.0)
    piece = np.zeros(len(train), np.float32)  # the recording's onsets over the same stretch; nothing beyond its ends
    lo, hi = max(k0, 0), min(k0 + len(train), len(env))
    if hi - lo < 20:
        return None
    piece[lo - k0 : hi - k0] = env[lo:hi]
    return np.correlate(piece, train[r : len(train) - r], mode="valid")  # one value per shift, -r .. +r


def _follow(starts, weights, file_t, sound_t, env, reach=0.4, each=1.0, half=2.0, bend=0.06):
    """By how much the first map (`file_t` to `sound_t`) is to be corrected along a segment, so that the file's
    note starts fall on the recording's onsets: one smooth course of shifts within `reach` seconds, never
    changing by more than `bend` seconds from one second to the next. A course, not a shift for every stretch on
    its own: where the music repeats within the beat, a stretch fits as well half a beat off, and only its
    neighbours say which it is. And when a whole song fits as well half a beat off, the first map says which it
    is: shifts beyond TRUSTED seconds cost a little."""
    centres = np.arange(file_t[0], file_t[-1] + each, each)
    r, m = int(round(reach / FINE)), int(round(bend / FINE))
    fits = np.zeros((len(centres), 2 * r + 1))
    placed = np.interp(starts, file_t, sound_t)
    for k, c in enumerate(centres):
        near = (starts >= c - half) & (starts <= c + half)
        fit = _fit(placed[near], weights[near], env, reach)
        if fit is not None and fit.std() > 0:
            fits[k] = (fit - fit.mean()) / fit.std()
    fits -= 2.5 * np.maximum(np.abs(np.arange(-r, r + 1)) * FINE - TRUSTED, 0.0)  # a quarter of a point a tenth
    best = np.zeros_like(fits)
    back = np.zeros(fits.shape, np.int64)
    best[0] = fits[0]
    for k in range(1, len(centres)):
        reached = np.full(2 * r + 1, -np.inf)
        for d in range(-m, m + 1):  # from the shift d steps away, at a small price for each step
            lo, hi = max(0, d), 2 * r + 1 + min(0, d)
            cand = best[k - 1, lo - d : hi - d] - 0.01 * abs(d)
            better = cand > reached[lo:hi]
            reached[lo:hi] = np.where(better, cand, reached[lo:hi])
            back[k, lo:hi] = np.where(better, np.arange(lo - d, hi - d), back[k, lo:hi])
        best[k] = reached + fits[k]
    course = np.zeros(len(centres), np.int64)
    course[-1] = int(np.argmax(best[-1]))
    for k in range(len(centres) - 1, 0, -1):
        course[k - 1] = back[k, course[k]]
    return np.interp(file_t, centres, (course - r) * FINE)


def _tighten(starts, weights, env, reach):
    """How many seconds later than `starts` the recording's onsets are, over this stretch: the shift, within
    `reach`, at which the note starts fall best on the onsets, and how much better than the other shifts."""
    score = _fit(starts, weights, env, reach)
    if score is None or score.std() == 0:
        return 0.0, 0.0
    k = int(np.argmax(score))
    sure = float((score[k] - np.median(score)) / score.std())
    d = float(k - len(score) // 2)
    if 0 < k < len(score) - 1:  # the top of the parabola through the peak and its neighbours
        a, b, c = score[k - 1], score[k], score[k + 1]
        if a - 2 * b + c < 0:
            d += 0.5 * (a - c) / (a - 2 * b + c)
    return d * FINE, sure


class Alignment:
    """Where the moments of a file fall in a recording. `segments` are the stretches the two have in common, each
    dict(file=(t0, t1), sound=(t0, t1), like=similarity); `shift` is the semitones the file is moved up by to be in
    the recording's key; `ratio` the seconds of recording to a second of file."""

    def __init__(self, segments, anchors, shift, ratio):
        self.segments, self.anchors, self.shift, self.ratio = segments, anchors, shift, ratio

    def at(self, t):
        """The recording's time for the file's times `t`; NaN where the file is at something the recording lacks."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.full(len(t), np.nan)
        for seg, (ft, st) in zip(self.segments, self.anchors):
            inside = (t >= seg["file"][0]) & (t <= seg["file"][1])
            out[inside] = np.interp(t[inside], ft, st)
        return out

    def covered(self):
        """The spans of the recording that the file speaks of."""
        return [list(seg["sound"]) for seg in self.segments]

    def notes(self, pm, track):
        """The notes of one of the file's parts in the recording's time and key, as (start, end, pitch): those
        that start in a stretch the two have in common."""
        raw = sorted((n.start, n.end, n.pitch) for n in pm.instruments[track].notes)
        if not raw:
            return []
        starts, ends = self.at([n[0] for n in raw]), self.at([n[1] for n in raw])
        out = []
        for (s, e, p), a, b in zip(raw, starts, ends):
            if np.isfinite(a):
                b = b if np.isfinite(b) and b > a else a + max(0.03, (e - s) * self.ratio)
                out.append((float(a), float(b), int(p + self.shift)))
        return out


def align(pm, mono, sr, shortest=6.0, bass="find"):
    """Line a file (pretty_midi.PrettyMIDI) up with a recording (mono samples at `sr`). Stretches the two have in
    common for less than `shortest` seconds are not kept. `bass` is the file's bass part (an instrument's index),
    found by `bass_track` unless given; None lines the two up by the whole arrangement alone."""
    import librosa
    from scipy.ndimage import median_filter, uniform_filter1d

    y22 = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=sr, target_sr=ASR, res_type="soxr_hq")
    length = pm.get_end_time()
    if length < 10 or len(y22) < 10 * ASR:
        raise Bass2SheetError("too short to line up: the file and the recording need ten seconds each")
    if bass == "find":
        try:
            bass = bass_track(pm)[0]
        except Bass2SheetError:
            bass = None
    shift, ratio = _first_look(pm, y22, length, bass)
    # The file at the recording's tempo: frame i of both is HOP seconds of recording, and i * HOP / ratio of file.
    n = int(length * ratio / HOP)
    S = _similar(file_parts(pm, HOP / ratio, n, shift, bass), sound_parts(y22, HOP)) - THETA
    pairs, _ = side_by_side(S)
    if len(pairs) < shortest / HOP:
        raise Bass2SheetError("the file and the recording have nothing in common that could be lined up")
    env = _onsets(y22)
    starts, weights = _beats_of(pm)
    cuts = np.flatnonzero((np.diff(pairs[:, 0]) > 2) | (np.diff(pairs[:, 1]) > 2)) + 1
    segments, anchors = [], []
    for piece in np.split(pairs, cuts):
        if len(piece) < shortest / HOP:
            continue
        i, j = piece[:, 0].astype(np.float64), piece[:, 1].astype(np.float64)
        # A first map, from the frames: smoothed over eight seconds, it is right to a few tenths of a second.
        lag = median_filter(j - i, size=min(len(piece) // 2 * 2 - 1, 81), mode="nearest")
        lag = uniform_filter1d(lag, size=min(len(piece), 81), mode="nearest")
        file_t, sound_t = i * HOP / ratio, (i + lag) * HOP
        f0, f1 = file_t[0], file_t[-1]
        here = (starts >= f0) & (starts <= f1)
        mine, heavy = starts[here], weights[here]
        if len(mine) >= 8:
            sound_t = sound_t + _follow(mine, heavy, file_t, sound_t, env)  # then the onsets say where exactly
            centres = np.arange(f0, f1, 1.0)
            moved, sure = np.zeros(len(centres)), np.zeros(len(centres))
            for k, c in enumerate(centres):  # ... and, a few seconds at a time, to the millisecond
                near = (mine >= c - 2.0) & (mine <= c + 2.0)
                moved[k], sure[k] = _tighten(np.interp(mine[near], file_t, sound_t), heavy[near], env, 0.04)
            good = sure >= 2.0
            if good.sum() >= 2:
                smooth = median_filter(np.interp(centres, centres[good], moved[good]), size=5, mode="nearest")
                sound_t = sound_t + np.interp(file_t, centres, smooth)
        sound_t = np.maximum.accumulate(np.clip(sound_t, 0.0, len(y22) / ASR))
        like = float(np.mean(S[piece[:, 0], piece[:, 1]]) + THETA)
        segments.append(
            dict(file=(float(f0), float(f1)), sound=(float(sound_t[0]), float(sound_t[-1])), like=round(like, 3))
        )
        anchors.append((file_t, sound_t))
    if not segments:
        raise Bass2SheetError("the file and the recording have nothing in common that could be lined up")
    return Alignment(segments, anchors, shift, ratio)


# ------------------------------------------------------------------------- an answer key for one of the songs


def answer_key(path, mono, sr, bass_mono=None, part=None, mscore=None):
    """Read a file, line it up with a recording and take its bass part as an answer key: dict(file, part, others,
    shift, octave, ratio, segments, covered, notes=[[start, end, pitch]]).

    `part` is the instrument to take in place of the one `bass_track` finds. With `bass_mono`, the recording's
    isolated bass, the key's notes are also moved by whole octaves to where that bass sounds: files often write
    the bass an octave from its sound. That is one shift for the whole file; which notes it has is left alone."""
    import pretty_midi

    pm = read(path, mscore)
    found, others = bass_track(pm)
    if part is not None and not (0 <= part < len(pm.instruments) and pm.instruments[part].notes):
        raise Bass2SheetError(f"the file has no part {part} with notes in it")
    part = found if part is None else part
    al = align(pm, mono, sr, bass=part)
    notes = al.notes(pm, part)
    octave, agree = 0, None
    if bass_mono is not None and notes:
        from .evaluate import octave_shift_of

        octave, agree = octave_shift_of(np.asarray(bass_mono, dtype=np.float32), sr, notes)
    called = lambda k: dict(
        index=int(k),
        name=(pm.instruments[k].name or "").strip()
        or pretty_midi.program_to_instrument_name(pm.instruments[k].program),
        notes=len(pm.instruments[k].notes),
    )
    return dict(
        file=os.path.basename(path),
        part=called(part),
        others=[called(k) for k in ([found] if part != found else []) + list(others)],
        shift=int(al.shift),
        octave=int(octave),
        octave_agreement=None if agree is None else round(float(agree), 3),
        ratio=round(al.ratio, 4),
        segments=[dict(file=list(s["file"]), sound=list(s["sound"]), like=s["like"]) for s in al.segments],
        covered=[[round(a, 3), round(b, 3)] for a, b in al.covered()],
        notes=[[round(s, 4), round(e, 4), int(p + octave)] for s, e, p in notes],
    )


def _inside(notes, spans, slack=0.05):
    return [n for n in notes if any(a - slack <= n[0] <= b + slack for a, b in spans)]


def _common(a, b):
    """The spans that are in both lists of spans."""
    out = [[max(x0, y0), min(x1, y1)] for x0, x1 in a for y0, y1 in b]
    return [s for s in out if s[1] > s[0]]


def against(key_notes, notes, spans, near=0.1):
    """A version of a song against an answer key, on `spans` of the recording: dict(key_notes, notes, f (starts
    within 50 ms), f_100ms, f_note_name (the octave ignored), and what becomes of the key's notes: `same` (the
    version has the note, within `near` seconds), `lower` and `higher` (it has it an octave or two away),
    `other` (it has another pitch there), `none` (it has nothing there), as shares)."""
    from . import evaluate

    key_notes, notes = _inside(sorted(map(tuple, key_notes)), spans), _inside(sorted(notes), spans)
    s = evaluate.score(key_notes, notes) if key_notes and notes else {}
    fate = dict(same=0, lower=0, higher=0, other=0, none=0)
    starts = np.array([n[0] for n in notes])
    for start, _, pitch in key_notes:
        there = [notes[k][2] for k in range(*np.searchsorted(starts, [start - near, start + near]))]
        if pitch in there:
            fate["same"] += 1
        elif not there:
            fate["none"] += 1
        else:
            apart = [q - pitch for q in there if (q - pitch) % 12 == 0]
            fate["other" if not apart else "lower" if apart[0] < 0 else "higher"] += 1
    return dict(
        key_notes=len(key_notes),
        notes=len(notes),
        f=round(s.get("f", 0.0), 3),
        f_100ms=round(s.get("f_100ms", 0.0), 3),
        f_note_name=round(s.get("f_note_name", 0.0), 3),
        **{k: round(v / max(1, len(key_notes)), 3) for k, v in fate.items()},
    )


def add(workdir, song_id, path, part=None, mscore=None):
    """Line a file up with one of the copilot's songs and keep its bass part as an answer key for that song, in
    <song>/reference/<name>.json. Returns (name, the key)."""
    import json
    import time

    import soundfile as sf

    from . import copilot, judge
    from .common import atomic_write_text

    if song_id not in copilot.load_songs(workdir):
        raise Bass2SheetError(f"no song {song_id} in {workdir}")
    folder = copilot.song_dir(workdir, song_id)
    record, sr = sf.read(os.path.join(folder, "audio", "record.flac"), dtype="float32", always_2d=True)
    bass, _ = sf.read(os.path.join(folder, "audio", "bass.flac"), dtype="float32", always_2d=True)
    key = answer_key(path, record.mean(1), sr, bass.mean(1), part, mscore)
    key["when"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    name = judge.slug(os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.join(folder, "reference"), exist_ok=True)
    atomic_write_text(os.path.join(folder, "reference", name + ".json"), json.dumps(key))
    return name, key


def keys(workdir, song_id):
    """{name: answer key} of a song, in the order of their names."""
    import json

    from . import copilot

    folder = os.path.join(copilot.song_dir(workdir, song_id), "reference")
    out = {}
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                out[name[: -len(".json")]] = json.load(f)
    return out


def compare(workdir, song_id):
    """A song's versions against each of its answer keys: [dict(name, key, tool=against(...) on all the key
    covers, worked=None or dict(spans, tool=..., yours=...) on what the person has worked through)]."""
    from . import copilot

    found = keys(workdir, song_id)
    if not found:
        return []
    desk = copilot.Desk(workdir, song_id)
    flat = lambda notes: sorted((n["start"], n["end"], q) for n in notes for q in (n["pitch"], *n.get("also", [])))
    tool, yours = flat(desk.machine["notes"]), flat(desk.notes)
    rows = []
    for name, key in found.items():
        row = dict(name=name, key=key, tool=against(key["notes"], tool, key["covered"]), worked=None)
        spans = _common(key["covered"], desk.worked())
        if spans and _inside(sorted(map(tuple, key["notes"])), spans):
            row["worked"] = dict(
                spans=spans, tool=against(key["notes"], tool, spans), yours=against(key["notes"], yours, spans)
            )
        rows.append(row)
    return rows


def _line(who, a):
    off = a["lower"] + a["higher"]
    return (
        f"| {who} | {a['key_notes']} | {a['f_100ms']:.2f} | {a['f_note_name']:.2f} | {a['same']:.0%} | "
        f"{off:.0%} | {a['other']:.0%} | {a['none']:.0%} |"
    )


def report(workdir):
    """The songs that have answer keys, each version against each key, as Markdown; nothing where there are none."""
    from . import copilot
    from .common import clock

    lines = []
    for song_id, song in copilot.load_songs(workdir).items():
        rows = (
            compare(workdir, song_id)
            if os.path.isdir(os.path.join(copilot.song_dir(workdir, song_id), "reference"))
            else []
        )
        if not rows:
            continue
        finder = (
            "MuScriptor's notes"
            if copilot.Desk(workdir, song_id).machine["settings"].get("notes") == "muscriptor"
            else "The tool's notes"
        )
        lines += [f"## {song.get('title', song_id)}", ""]
        for row in rows:
            key = row["key"]
            length = sum(b - a for a, b in key["covered"])
            moved = [f"moved {key['shift']:+d} semitones to the recording's key"] if key["shift"] else []
            moved += [f"moved {key['octave'] // 12:+d} octave to where the bass sounds"] if key["octave"] else []
            lines += [
                f'**{key["file"]}**: its part "{key["part"]["name"]}", {len(key["notes"])} notes lined up over '
                f"{clock(length)} of the recording" + ("".join(", " + m for m in moved)) + ".",
                "",
                "| | Key's notes | Score | Score, octave ignored | Has the note | An octave away | Another pitch "
                "| Nothing there |",
                "|---|---|---|---|---|---|---|---|",
                _line(finder, row["tool"]),
            ]
            if row["worked"]:
                lines += [
                    _line(finder + ", where you have worked", row["worked"]["tool"]),
                    _line("Your version, where you have worked", row["worked"]["yours"]),
                ]
            lines.append("")
    if not lines:
        return ""
    head = [
        "# Against the answer keys you gave",
        "",
        "An answer key is somebody's transcription, lined up with your recording. The score counts a note as right "
        "when the key has that pitch starting within a tenth of a second.",
        "",
    ]
    return "\n".join(head + lines) + "\n"
