"""A song's harmony: its chords, where they change, and its key or keys.

The record without its bass and drums is heard eighth note by eighth note on the tool's own beat grid, and each
eighth is matched against every chord of a small vocabulary. What speaks for a chord is which notes sound in
that eighth and what the bass plays under it (the tool's own bass notes: the bass holds the root most of the
time). The chords are then read as one line through the song that changes where the music does: a change on a
bar line costs little, one in the middle of a bar more, one anywhere else a lot. The key is read from the
chords: each key makes some chords likely and others rare, and the key may change during the song.

How right this is, measured against chord annotations, is in docs/evaluation.md. dev/chords_eval.py measures it
and fits the weights below; they are fitted on BabySlakh, not on the songs it is measured on.
"""

import librosa
import numpy as np

from .common import SR, spelling

# The vocabulary. Every chord found is one of these on one of the twelve roots, named as a chord symbol ends:
# C, Cm, C7, Cmaj7, Cm7.
QUALITIES = ("", "m", "7", "maj7", "m7")
TONES = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10)}
TRIAD = {"": "", "m": "m", "7": "", "maj7": "", "m7": "m"}  # the triad under each

STEP = 2  # sixteenths in a frame: chords are looked for eighth by eighth
SR_H, HOP_H = 22050, 512  # the spectrum's sample rate and hop (23 ms)
LOWEST, OCTAVES = "C2", 5  # the notes the chords are read from: C2 to B6
COMPRESS = 100.0  # loudness is compressed before notes are compared, so one loud note does not hide the others
BASS_SHARE = 0.6  # the bass must hold another chord note this much of the chord's time to be written under it
QUIET = 0.03  # a frame where the instruments have less than this share of their level when they play has no chord

# What a chord change costs entering a frame, by the sixteenth of the bar the frame starts on: little on a bar
# line, more in the middle of the bar, a lot anywhere else. Set by hand. In BabySlakh's MIDI the chords change
# on 62% of the bar lines and in the middle of 13% of the bars (dev/chords_eval.py --fit).
CHANGE_COST = {0: 1.0, 2: 5.0, 4: 5.0, 6: 5.0, 8: 3.0, 10: 5.0, 12: 5.0, 14: 5.0}

# What each frame is matched on, each a vector over the twelve pitch classes (see `hear`).
GROUPS = ("notes", "bass")

# The weights of the match, fitted on the 16 songs of BabySlakh (CC BY 4.0) whose chords in each eighth are
# read from the songs' own MIDI notes (dev/chords_eval.py --slakh DIR --fit). One row per quality of QUALITIES;
# in a row, twelve numbers for each group of GROUPS, by the pitch class's distance above the chord's root.
# fmt: off
WEIGHTS = np.array([
    # "": the instruments' notes, then the bass's
    [ 10.40,  -9.90,   3.71, -15.92,  11.14,  -2.45,  -2.40,  11.22,  -8.44,  -5.90,  -5.45,  -4.37,
       3.33,  -1.23,  -0.00,  -0.69,   0.67,  -0.52,  -1.06,   1.57,  -1.64,  -0.10,  -0.18,  -0.39],
    # "m": the instruments' notes, then the bass's
    [ 16.48,  -3.44,   0.63,  12.55, -11.10,  -2.18,  -4.29,  16.76,  -8.24,  -1.27,  -2.92,  -7.96,
       2.92,  -0.68,  -0.76,   0.68,   0.55,  -0.19,  -2.93,   1.17,   0.56,  -0.03,  -1.22,  -0.38],
    # "7": the instruments' notes, then the bass's
    [  9.32,  -2.78,  -1.76, -13.21,  13.31,  -3.41,  -6.97,  12.90,  -4.30,  -4.50,  14.64,  -8.54,
       3.04,  -0.55,  -0.02,   0.33,   1.98,  -0.53,  -0.27,   1.45,  -0.35,   0.81,  -0.46,   0.32],
    # "maj7": the instruments' notes, then the bass's
    [  9.63, -12.12,   0.78, -10.25,  11.22,  -6.05,   3.09,  11.98, -11.23,  -1.74,  -8.99,  14.60,
       2.97,  -1.29,  -0.49,  -1.81,  -1.35,  -0.80,  -0.14,  -0.06,  -1.48,   0.17,  -1.19,  -0.70],
    # "m7": the instruments' notes, then the bass's
    [ 12.14,  -5.93,  -2.70,   9.34,  -5.74,  -1.50,  -6.95,  14.39,  -6.03,  -3.17,  14.27,  -9.29,
       2.99,  -1.44,  -0.86,   0.79,   0.05,   0.21,  -0.28,   1.01,  -1.63,   0.03,   0.65,  -0.50],
])
# fmt: on
BIASES = np.array([6.30, -1.14, -0.77, -1.42, -2.37])

# How much of a song in a major key and in a minor key each triad takes, by its root's distance above the tonic
# in semitones, as levels set by hand from common practice: the tonic most, then the fourth and fifth steps
# (in minor the flat sixth and seventh), then the key's other triads, then the chords pop borrows.
IN_MAJOR = {
    "": {0: 5, 5: 3.5, 7: 3.5, 10: 0.5, 2: 0.3, 4: 0.3, 9: 0.3, 3: 0.3, 8: 0.3},
    "m": {9: 2, 2: 1.5, 4: 1.2, 5: 0.4, 0: 0.2, 7: 0.2},
}
IN_MINOR = {
    "m": {0: 5, 7: 2, 5: 1.5},
    "": {8: 3, 10: 2.5, 3: 1.5, 7: 1.2, 5: 1.2, 1: 0.4, 0: 0.2},
}
RARE = 0.1  # a triad not listed above
KEY_CHANGE = 100.0  # what a change of key costs, in the units of a frame's fit: a few borrowed chords are no new key
KEY_SHARPNESS = 8.0  # how fast a key's score falls with its fit


# ------------------------------------------------------------------------------------------------- what is heard


def spectrum(mono):
    """How loud each semitone from C2 to B6 is, 43 times a second: (60, frames). `mono` is at the working rate.
    The instrument's own tuning is measured first, so a record a little sharp or flat is heard in tune."""
    y = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=SR, target_sr=SR_H)
    frames = 1 + len(y) // HOP_H
    if not len(y) or float(np.abs(y).max()) < 1e-4:
        return np.zeros((12 * OCTAVES, frames), np.float32)
    tuning = float(librosa.estimate_tuning(y=y, sr=SR_H, bins_per_octave=36))
    below = librosa.note_to_hz(LOWEST) * 2 ** (-1 / 36)  # three bins a semitone, the note in the middle one
    c = np.abs(
        librosa.cqt(y, sr=SR_H, hop_length=HOP_H, fmin=below, n_bins=36 * OCTAVES, bins_per_octave=36, tuning=tuning)
    )
    return c.reshape(12 * OCTAVES, 3, -1).sum(1).astype(np.float32)


def chroma(spec):
    """The spectrum folded into the twelve pitch classes, after compression: (12, frames), C first."""
    x = spec.astype(np.float64)
    top = float(np.percentile(x, 99.9)) if x.size else 0.0
    x = np.log1p(COMPRESS * x / top) if top > 0 else x
    out = np.zeros((12, x.shape[1]))
    low = int(round(librosa.note_to_midi(LOWEST))) % 12
    for k in range(x.shape[0]):
        out[(low + k) % 12] += x[k]
    return out


def _per_frame(columns, edges):
    """The mean of `columns` (rows, one column every HOP_H samples) between consecutive times in `edges`."""
    total = np.concatenate([np.zeros((columns.shape[0], 1)), np.cumsum(columns, axis=1)], axis=1)
    k = np.clip(np.round(np.asarray(edges) * SR_H / HOP_H).astype(int), 0, columns.shape[1])
    return ((total[:, k[1:]] - total[:, k[:-1]]) / np.maximum(k[1:] - k[:-1], 1)).T


def _level(mono, edges, hop=SR // 100):
    """How loud `mono` (at the working rate) is between consecutive times in `edges`: the level it has for half
    of the time between them or more, so that a chord that comes in at the very end of a frame does not make
    the frame sound."""
    n = len(mono) // hop
    loud = np.sqrt((mono[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(1)) if n else np.zeros(0)
    k = np.clip(np.round(np.asarray(edges) * SR / hop).astype(int), 0, n)
    return np.array([np.median(loud[a:b]) if b > a else 0.0 for a, b in zip(k[:-1], k[1:])])


def _unit(rows):
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-9)


def frame_starts(grid, origin, length):
    """The grid indices where the frames start: every eighth, in step with the bar lines, from the first that
    reaches into the song to the last that starts before its end."""
    grid = np.asarray(grid, dtype=float)
    idx = np.arange(int(origin) % STEP, len(grid) - STEP, STEP)
    return idx[(grid[idx + STEP] > 0) & (grid[idx] < length)]


def held_per_frame(notes, edges):
    """(frames, 12): the share of each frame in which each pitch class is held by `notes` (dicts with start,
    end and pitch). Of bass notes played together (`also`) the lowest counts."""
    out = np.zeros((len(edges) - 1, 12))
    for nt in notes:
        a, b, pc = nt["start"], nt["end"], min([nt["pitch"], *nt.get("also", [])]) % 12
        k0 = max(0, int(np.searchsorted(edges, a, side="right")) - 1)
        k1 = min(len(edges) - 1, int(np.searchsorted(edges, b, side="left")))
        for k in range(k0, k1):
            over = min(b, edges[k + 1]) - max(a, edges[k])
            if over > 0:
                out[k, pc] += over / (edges[k + 1] - edges[k])
    return out


def hear(stems, notes, grid, origin):
    """What the chords are found from, frame by frame (an eighth note each), or None where the record has no
    instruments beside bass and drums. A dict:

    index   the grid index each frame starts on           edges  the frames' boundaries in seconds
    place   the sixteenth of the bar it starts on (0-15)  level  how loud the instruments are in it
    groups  what a chord is matched on, each (frames, 12) over the pitch classes, C first:
            notes  the instruments' notes in the frame
            bass   the share of the frame in which the bass holds each pitch class"""
    grid = np.asarray(grid, dtype=float)
    played = np.asarray(stems["harmony"]).mean(0)
    index = frame_starts(grid, origin, len(played) / SR)
    if float(np.abs(played).max(initial=0.0)) < 1e-4 or len(index) < 2:
        return None
    edges = np.r_[grid[index], grid[index[-1] + STEP]]
    return dict(
        index=index,
        edges=edges,
        place=(index - int(origin)) % 16,
        level=_level(played, edges),
        groups=dict(notes=_unit(_per_frame(chroma(spectrum(played)), edges)), bass=held_per_frame(notes, edges)),
    )


# ------------------------------------------------------------------------------------------------- the chords


def from_each_root(groups):
    """(frames, 12 roots, 12 * groups): every group's vector as it looks from each root, so that one set of
    weights serves all twelve."""
    a = np.stack([groups[g] for g in GROUPS], axis=1)
    return np.stack([np.roll(a, -r, axis=2) for r in range(12)], axis=1).reshape(len(a), 12, -1)


def chord_fit(heard, weights=None, biases=None):
    """(frames, 12 * qualities + 1): the log-probability of each chord in each frame, taken alone, then the
    score of "no chord". A chord's column is 12 * (its quality's place in QUALITIES) + its root."""
    weights = WEIGHTS if weights is None else weights
    biases = BIASES if biases is None else biases
    s = from_each_root(heard["groups"]) @ np.asarray(weights).T + np.asarray(biases)  # frames, roots, qualities
    s = s - s.max(axis=(1, 2), keepdims=True)
    s = s - np.log(np.exp(s).sum(axis=(1, 2), keepdims=True))
    fit = s.transpose(0, 2, 1).reshape(len(s), -1)
    silent = heard["level"] < QUIET * np.percentile(heard["level"], 90)  # against the level when they play
    none = np.where(silent, fit.max(1) + 5.0, fit.max(1) - 12.0)
    return np.concatenate([fit, none[:, None]], axis=1)


def change_costs(place):
    return np.array([CHANGE_COST[int(p) - int(p) % 2] for p in place])


def best_line(fit, cost):
    """The line of chords through the frames that fits best, when changing chord entering frame t costs
    cost[t]: the column of `fit` chosen in each frame."""
    n, m = fit.shape
    score = fit[0].copy()
    came_from = np.zeros((n, m), np.int32)
    same = np.arange(m)
    for t in range(1, n):
        order = np.argsort(score)
        other = np.where(same == order[-1], order[-2], order[-1])  # the best chord to change from
        change = score[other] - cost[t]
        came_from[t] = np.where(score >= change, same, other)
        score = np.maximum(score, change) + fit[t]
    line = np.zeros(n, np.int64)
    line[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        line[t - 1] = came_from[t, line[t]]
    return line


def likelihoods(fit, cost):
    """(frames, columns): how likely each chord is in each frame given the whole song, under the same costs as
    `best_line`. Each row sums to 1."""
    n, m = fit.shape
    e = np.exp(fit - fit.max(1, keepdims=True))
    w = np.exp(-np.asarray(cost, dtype=float))
    before = np.zeros((n, m))
    before[0] = e[0] / e[0].sum()
    for t in range(1, n):
        before[t] = e[t] * (before[t - 1] + w[t] * (1 - before[t - 1]))
        before[t] /= before[t].sum()
    after = np.ones((n, m))
    for t in range(n - 2, -1, -1):
        v = e[t + 1] * after[t + 1]
        after[t] = v + w[t + 1] * (v.sum() - v)
        after[t] /= after[t].max()
    both = before * after
    return both / both.sum(1, keepdims=True)


def _triad_mass(likely, state):
    """Per frame, the likelihood of the chord's root with its third, whatever the seventh."""
    root, triad = state % 12, TRIAD[QUALITIES[state // 12]]
    return sum(likely[:, 12 * q + root] for q, name in enumerate(QUALITIES) if TRIAD[name] == triad)


def _chords(line, likely, heard, length):
    """The line as chords: runs of frames with the same chord; frames of no chord are left out."""
    out, k, n = [], 0, len(line)
    index, edges, bass = heard["index"], heard["edges"], heard["groups"]["bass"]
    while k < n:
        j = k
        while j + 1 < n and line[j + 1] == line[k]:
            j += 1
        state = int(line[k])
        if state < 12 * len(QUALITIES):
            root, quality = state % 12, QUALITIES[state // 12]
            held = bass[k : j + 1].sum(0)
            under = int(held.argmax())
            slash = held.sum() > 0 and under != root and (under - root) % 12 in TONES[quality]
            out.append(
                dict(
                    start=round(float(min(max(edges[k], 0.0), length)), 3),
                    end=round(float(min(max(edges[j + 1], 0.0), length)), 3),
                    at=int(index[k]),
                    until=int(index[j] + STEP),
                    root=root,
                    quality=quality,
                    bass=under if slash and held[under] >= BASS_SHARE * held.sum() else None,
                    score=round(share_right(float(_triad_mass(likely[k : j + 1], state).mean())), 3),
                )
            )
        k = j + 1
    return out


# A chord's likelihood (see `likelihoods`) against the share of chords that likely which have the right root and
# third, measured on BabySlakh (dev/chords_eval.py --fit). Likelihoods run high; the score given out is the share
# that was measured.
SHARE_RIGHT = (
    (0.0, 0.0),
    (0.284, 0.16),
    (0.36, 0.16),
    (0.462, 0.18),
    (0.547, 0.45),
    (0.647, 0.45),
    (0.753, 0.67),
    (0.854, 0.67),
    (0.927, 0.76),
    (0.974, 0.87),
    (0.996, 0.95),
    (1.0, 0.99),
)


def share_right(likelihood):
    xs, ys = zip(*SHARE_RIGHT)
    return float(np.interp(likelihood, xs, ys))


# --------------------------------------------------------------------------------------------------- the key


def key_fit():
    """(24, 12 * qualities): the log of how likely each chord is in each key. Key k is the major key on pitch
    class k for k < 12, the minor key on k - 12 otherwise."""
    table = np.zeros((24, 12 * len(QUALITIES)))
    for k in range(24):
        share = IN_MAJOR if k < 12 else IN_MINOR
        for q, quality in enumerate(QUALITIES):
            for root in range(12):
                table[k, 12 * q + root] = share[TRIAD[quality]].get((root - k) % 12, RARE)
    return np.log(table / table.sum(1, keepdims=True))


def key_line(likely):
    """The key in force in each frame, and (frames, 24) how well each key explains each frame's chords."""
    fit = likely[:, : 12 * len(QUALITIES)] @ key_fit().T
    n = len(fit)
    score = fit[0].copy()
    came_from = np.zeros((n, 24), np.int32)
    same = np.arange(24)
    for t in range(1, n):
        top = int(score.argmax())
        change = score[top] - KEY_CHANGE
        came_from[t] = np.where(score >= change, same, top)
        score = np.maximum(score, change) + fit[t]
    line = np.zeros(n, np.int64)
    line[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        line[t - 1] = came_from[t, line[t]]
    return line, fit


def key_name(root, major):
    return spelling(root, major)[root] + (" major" if major else " minor")


def _keys(line, fit, heard, length):
    """The line of keys as spans, each with the key that explains the span's chords next best."""
    out, a, n = [], 0, len(line)
    index, edges = heard["index"], heard["edges"]
    for t in range(1, n + 1):
        if t < n and line[t] == line[a]:
            continue
        mean = fit[a:t].mean(0)
        share = np.exp(KEY_SHARPNESS * (mean - mean.max()))
        share /= share.sum()
        k = int(line[a])
        other = int(next(c for c in np.argsort(-share) if c != k))
        out.append(
            dict(
                start=round(float(min(max(edges[a], 0.0), length)), 3),
                end=round(float(min(max(edges[t], 0.0), length)), 3),
                at=int(index[a]),
                until=int(index[t - 1] + STEP),
                root=k % 12,
                major=k < 12,
                score=round(float(share[k]), 3),
                other=dict(root=other % 12, major=other < 12),
            )
        )
        a = t
    return out


# ------------------------------------------------------------------------------------------------- all of it


def read(heard, length, weights=None, biases=None):
    """The harmony (see `find`) from what `hear` made of a song of `length` seconds."""
    fit, cost = chord_fit(heard, weights, biases), change_costs(heard["place"])
    line, likely = best_line(fit, cost), likelihoods(fit, cost)
    chords = _chords(line, likely, heard, length)
    return dict(keys=_keys(*key_line(likely), heard, length) if chords else [], chords=chords)


def find(stems, notes, grid, origin):
    """The song's harmony from its stems, its bass notes and its beat grid (sixteenths; bar 1 starts at index
    `origin`). Returns dict(keys, chords), both in time order and plain data.

    chords  each dict(start, end, at, until, root, quality, bass, score):
            start, end  in seconds
            at, until   indices into the grid: the chord sounds from grid[at] to grid[until]
            root        pitch class, 0 = C ... 11 = B
            quality     one of QUALITIES: "" (major), "m", "7", "maj7", "m7"
            bass        the pitch class the bass holds under it when that is another note of the chord (C/E),
                        else None
            score       0 to 1: the share of chords found this surely that have the right root and third
            Where only drums and bass play there is no chord: the list has a gap.
    keys    each dict(start, end, at, until, root, major, score, other): the key over a span, how sure, and
            `other`, a dict(root, major): the key that explains the span's chords next best. For most songs
            that is the relative major or minor. The spans cover the song from its start to its end without
            gaps. Empty when no chord was found."""
    heard = hear(stems, notes, grid, origin)
    if heard is None:
        return dict(keys=[], chords=[])
    return read(heard, np.asarray(stems["harmony"]).shape[-1] / SR)


def main_key(found, notes=()):
    """(name, root, is major): the key the song is in for the longest. Where no chord was found, the key the
    bass notes fit best."""
    if found["keys"]:
        held = {}
        for k in found["keys"]:
            held[(k["root"], k["major"])] = held.get((k["root"], k["major"]), 0.0) + k["end"] - k["start"]
        root, major = max(held, key=held.get)
        return key_name(root, major), root, major
    from .rhythm import estimate_key

    played = np.zeros((12, 1))
    for nt in notes:
        played[nt["pitch"] % 12] += nt["end"] - nt["start"]
    return estimate_key(played) if played.any() else ("C major", 0, True)


def per_bar(found, origin, key):
    """The chords as the stored data has always had them: one dict(bar, root, quality, score, names, chord) for
    each bar from bar 1 on that has a chord, the one that holds most of the bar, as a plain triad ("" major,
    "m" minor). `key` is (name, root, is major) and decides between sharps and flats."""
    names = spelling(key[1], key[2])
    most = {}
    for c in found["chords"]:
        g0, g1 = c["at"] - origin, c["until"] - origin
        for bar in range(max(g0 // 16, 0), (g1 - 1) // 16 + 1):
            held = min(g1, 16 * (bar + 1)) - max(g0, 16 * bar)
            if held > most.get(bar, (0, None))[0]:
                most[bar] = (held, c)
    out = []
    for bar in sorted(most):
        c = most[bar][1]
        triad = TRIAD[c["quality"]]
        out.append(
            dict(
                bar=bar + 1,
                root=c["root"],
                quality=triad,
                score=c["score"],
                names=names,
                chord=names[c["root"]] + triad,
            )
        )
    return out


# ------------------------------------------------------------------------------------------- for the sheet

LETTERS = "CDEFGAB"
NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTER_UP = (0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6)  # how many letters above a note's the note a semitone step up takes


def note_name(pitch_class, above, plain):
    """The name of a pitch class as a step above the note named `above` (a tonic, a chord's root): the letter
    that step takes, with the sharp or flat that makes the pitch. F# above D, Gb above Eb; in A major the chord
    on the fifth step is E and its third G#. Where that would need a double sign, or E#, B#, Cb or Fb, the
    name is taken from `plain`, a list of the twelve names."""
    step = (pitch_class - NATURAL[above[0]] - {"": 0, "#": 1, "b": -1}[above[1:]]) % 12
    letter = LETTERS[(LETTERS.index(above[0]) + LETTER_UP[step]) % 7]
    sign = {0: "", 1: "#", -1: "b"}.get((pitch_class - NATURAL[letter] + 6) % 12 - 6)
    if sign is None or letter + sign in ("E#", "B#", "Cb", "Fb"):
        return plain[pitch_class]
    return letter + sign


def for_sheet(data):
    """What the sheet writes over each bar, from a song's stored data or from what was heard in it: a dict with
    `harmony` and `origin` or, for a song prepared before there was harmony, with `chords`, a triad per bar.
    Returns {bar: [(sixteenth of the bar, chord)]}: one chord on beat 1 and, where the second half of the bar
    has another, one on beat 3. Each chord is a dict(root, quality, bass, names); `names` are the twelve note
    names to spell the chord with, and the bass notes played under it: its root by its step in the key in force,
    its other notes by their step above the root."""
    found = data.get("harmony")
    if not found or not found["chords"]:  # the old shape: dict(bar, root, quality "" or "m", names, chord)
        return {
            c["bar"]: [(0, dict(root=c["root"], quality=c["quality"], bass=None, names=c["names"]))]
            for c in data["chords"]
        }
    origin = int(data["origin"])
    halves = {}  # half bar -> (sixteenths held, chord): the chord that holds most of it
    for c in found["chords"]:
        g0, g1 = c["at"] - origin, c["until"] - origin
        for h in range(g0 // 8, (g1 - 1) // 8 + 1):
            held = min(g1, 8 * (h + 1)) - max(g0, 8 * h)
            if held > halves.get(h, (0, None))[0]:
                halves[h] = (held, c)
    out = {}
    for h in sorted(halves):
        bar, c = h // 2 + 1, halves[h][1]
        if h % 2 and bar in out and all(out[bar][0][1][f] == c[f] for f in ("root", "quality", "bass")):
            continue
        key = next((k for k in found["keys"] if k["at"] <= c["at"] < k["until"]), found["keys"][0])
        plain = spelling(key["root"], key["major"])
        names = list(plain)
        names[c["root"]] = note_name(c["root"], plain[key["root"]], plain)
        for tone in TONES[c["quality"]][1:]:
            names[(c["root"] + tone) % 12] = note_name((c["root"] + tone) % 12, names[c["root"]], plain)
        out.setdefault(bar, []).append(
            (8 * (h % 2), dict(root=c["root"], quality=c["quality"], bass=c["bass"], names=names))
        )
    return out
