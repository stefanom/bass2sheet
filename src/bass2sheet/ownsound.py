"""The transcription checked against the bass track by the track's own sound.

Every note that was found is also a sample: what this bass sounds like, on this recording, at that pitch. So a
note's pitch can be checked without any model of what a bass sounds like: do the *other* notes with that pitch,
or with a neighbouring pitch shifted to it, sound like this one? A note is never compared with itself, or
everything would match.

`check` answers that for each note, and names the pitch whose notes explain it better when there is one.
`rebuild` plays the whole transcription with those samples, each note by a sample cut from another note: the
transcription in the instrument's own voice, to set beside the bass track.

What this can and cannot see (measured in docs/evaluation.md): it finds a wrong note among right ones, and it
is blind to a mistake made every time, because then the wrong notes explain each other. It compares with the
bass track, so it has nothing to say about whether that track is the bass.
"""

import numpy as np

from .common import SR

F_LO, F_HI = 30.0, 1500.0  # the part of the spectrum that is compared
PER_SEMI = 4  # bins per semitone: a shift in pitch is a shift along this axis
N_BINS = int(np.ceil(12 * PER_SEMI * np.log2(F_HI / F_LO)))
NFFT = 1 << 15
_EDGES = np.searchsorted(np.fft.rfftfreq(NFFT, 1 / SR), F_LO * 2 ** (np.arange(N_BINS + 1) / (12 * PER_SEMI)))
SKIP = 0.02  # the pluck's first 20 ms are noise, the same for every note
LENGTHS = (0.06, 0.1, 0.15, 0.2, 0.3, 0.5)  # two notes are compared over the longest of these that both last
FLOOR = 0.1  # what lies below this share of the spectrum's peak does not count
POWER = 0.5  # the spectrum is compressed: how loud each partial is depends on how the note was played, not on its pitch
MARGIN = 0.10  # another pitch explains a note "better" from this much more overlap
MAX_SHIFT = 2  # a sample plays pitches up to this many semitones from its own


def spectrum(mono, start, length):
    """Magnitude spectrum of mono[start + SKIP : start + SKIP + length] on a log-frequency axis, less a floor,
    compressed, with unit sum. None when there is too little sound to take one."""
    a = int((start + SKIP) * SR)
    seg = mono[a : a + int(length * SR)]
    if len(seg) < int(0.04 * SR):
        return None
    full = np.abs(np.fft.rfft(seg.astype(np.float64) * np.hanning(len(seg)), NFFT))
    bands = np.maximum.reduceat(full[: _EDGES[-1]], _EDGES[:-1])  # the loudest line in each quarter-semitone
    out = np.convolve(bands, [0.25, 0.5, 0.25], mode="same")
    out = np.maximum(out - FLOOR * out.max(), 0) ** POWER
    total = out.sum()
    return out / total if total > 0 else None


def shifted(spec, semitones):
    """A spectrum moved along the pitch axis, as the same sound played that much higher or lower would have it."""
    k = int(round(semitones * PER_SEMI))
    out = np.zeros_like(spec)
    if k > 0:
        out[k:] = spec[:-k]
    elif k < 0:
        out[:k] = spec[-k:]
    else:
        out[:] = spec
    total = out.sum()
    return out / total if total > 0 else out


def overlap(a, b):
    """How much two unit-sum spectra have in common: 1 for the same sound, near 0 for two different notes."""
    return float(np.minimum(a, b).sum() / np.maximum(a, b).sum())


def _length(seconds):
    fits = [x for x in LENGTHS if x <= seconds + 1e-9]
    return fits[-1] if fits else None


def _moved(rows, semitones):
    """The spectra in `rows`, each moved along the pitch axis (see shifted)."""
    k = int(round(semitones * PER_SEMI))
    out = np.zeros_like(rows)
    if k > 0:
        out[:, k:] = rows[:, :-k]
    elif k < 0:
        out[:, :k] = rows[:, -k:]
    else:
        out[:] = rows
    total = out.sum(1, keepdims=True)
    return out / np.where(total > 0, total, 1.0)


class Bank:
    """The single notes of a transcription, as samples of the instrument. `notes` are dicts with start, end and
    pitch; a note with others played together (`also`) is not a sample. Of a pitch with many notes, `most`
    spread through the song are used."""

    def __init__(self, mono, notes, most=40):
        self.mono, self.notes, self.most = mono, notes, most
        self.lasts = [n["end"] - n["start"] - SKIP for n in notes]
        self.by_pitch = {}
        for i, n in enumerate(notes):
            if not n.get("also") and self.lasts[i] >= LENGTHS[0]:
                self.by_pitch.setdefault(n["pitch"], []).append(i)
        self._spectra, self._stacks = {}, {}

    def spec(self, i, length):
        key = (i, length)
        if key not in self._spectra:
            self._spectra[key] = spectrum(self.mono, self.notes[i]["start"], length)
        return self._spectra[key]

    def stack(self, pitch, length, played_as=None):
        """(the notes of `pitch` that last `length`, their spectra over that length as rows), the spectra moved
        to the pitch `played_as` when that is another one."""
        key = (pitch, length, pitch if played_as is None else played_as)
        if key not in self._stacks:
            if key[2] != pitch:
                idx, rows = self.stack(pitch, length)
                self._stacks[key] = (idx, _moved(rows, key[2] - pitch))
            else:
                idx = [j for j in self.by_pitch.get(pitch, []) if self.lasts[j] >= length - 1e-9]
                if len(idx) > self.most:
                    idx = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, self.most)]
                rows = [(j, self.spec(j, length)) for j in idx]
                rows = [(j, r) for j, r in rows if r is not None]
                self._stacks[key] = (
                    np.array([j for j, _ in rows], dtype=int),
                    np.array([r for _, r in rows]) if rows else np.zeros((0, N_BINS)),
                )
        return self._stacks[key]

    def samples(self, pitch, exclude, max_shift=MAX_SHIFT):
        """The notes to play `pitch` with, never the note `exclude`: those of the nearest pitch first."""
        out = []
        for d in range(max_shift + 1):
            for p in sorted({pitch - d, pitch + d}):
                out += [j for j in self.by_pitch.get(p, []) if j != exclude]
        return out

    def explained(self, i, pitch=None, length=None, enough=5):
        """How well other notes, played at `pitch` (default: the note's own), explain the sound of note i:
        (the median overlap of its spectrum with theirs, 0..1; the stretch in seconds they were compared over),
        or None when the note is too short to tell or there is nothing to compare it with.

        Notes of that very pitch are asked first: over the longest stretch that the note and `enough` of them
        last (a long note is rare in a fast line, and the few there are tend to be several notes run together).
        Only when no stretch has `enough` are the notes a semitone away asked as well, moved to the pitch, then
        those two away. The median is not taken in by a few notes with a wrong label. With `length`, only that
        stretch is tried: two pitches are compared fairly only over the same one."""
        pitch = self.notes[i]["pitch"] if pitch is None else pitch
        longest = _length(self.lasts[i])
        if longest is None:
            return None
        lengths = [length] if length else [x for x in reversed(LENGTHS) if x <= longest]
        most = None
        for d in range(MAX_SHIFT + 1):
            for stretch in lengths:
                mine = self.spec(i, stretch)
                if mine is None:
                    continue
                values = []
                for p in range(pitch - d, pitch + d + 1):
                    idx, rows = self.stack(p, stretch, pitch)
                    rows = rows[idx != i]
                    if len(rows):
                        values.append(np.minimum(rows, mine).sum(1) / np.maximum(rows, mine).sum(1))
                count = sum(map(len, values))
                if count >= enough:
                    return float(np.median(np.concatenate(values))), stretch
                if count and (most is None or count > most[0]):
                    most = (count, values, stretch)
        return (float(np.median(np.concatenate(most[1]))), most[2]) if most else None

    def pitches_near(self, pitch, span):
        """The pitches within `span` semitones that some sample can play."""
        reach = {p + d for p in self.by_pitch for d in range(-MAX_SHIFT, MAX_SHIFT + 1)}
        return sorted(q for q in reach if q != pitch and abs(q - pitch) <= span)


ALONE = 0.6  # a note with no other of its pitch is still doubted when another pitch explains it this well
UNLIKE = 0.5  # a note the others of its pitch explain less than this does not sound like them


def doubtful(heard):
    """True for a note (an entry of check's result) that does not sound like the song's other notes of its
    pitch, or that another pitch's notes explain better."""
    return bool(heard) and (heard["explained"] < UNLIKE or heard["better"] is not None)


def check(mono, notes, margin=MARGIN, span=12):
    """What the bass track's own sound says of each note: a list, one entry per note, of None (a chord, too short
    to tell, or nothing to compare with) or dict(explained, better, by): how well the other notes of its pitch
    explain it (0..1), and, when the notes of another pitch within `span` semitones explain it better by
    `margin`, that pitch and by how much (otherwise None and 0.0). A note whose pitch no other note is near
    cannot be explained by its own; it is reported (with explained 0) only when another pitch explains it well."""
    bank = Bank(mono, notes)
    out = []
    for i, n in enumerate(notes):
        if n.get("also") or _length(bank.lasts[i]) is None:
            out.append(None)
            continue
        own = bank.explained(i)
        best, value = None, own[0] if own else 0.0
        for q in bank.pitches_near(n["pitch"], span):
            other = bank.explained(i, q, length=own[1] if own else None)
            if other and other[0] > value:
                best, value = q, other[0]
        if own is None:
            out.append(dict(explained=0.0, better=best, by=value) if best is not None and value >= ALONE else None)
            continue
        better = best is not None and value - own[0] >= margin
        out.append(dict(explained=own[0], better=best if better else None, by=value - own[0] if better else 0.0))
    return out


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _stretched(y, n):
    """A sample made to last n samples: its attack as it is, the rest slowed down."""
    import librosa

    head = min(int(0.08 * SR), len(y) // 2)
    body = y[head:]
    want = n - head
    if len(body) < 2048 or want <= len(body):
        return y
    slow = librosa.effects.time_stretch(body.astype(np.float32), rate=len(body) / want, n_fft=2048)
    return np.concatenate([y[:head], slow[:want]])


def rebuild(mono, notes, quietest=0.3):
    """The transcription played with the bass track's own sound (mono float32, as long as the track): each note
    by a sample cut from ANOTHER note of the same pitch, or of a neighbouring pitch played faster or slower, at
    the level the track has there. A note written where the track is nearly silent is still played, at
    `quietest` times the song's typical note, so that a note that is not there can be heard. A note whose pitch
    no other note comes near stays silent."""
    bank = Bank(mono, notes)
    out = np.zeros(len(mono))
    pre = int(0.005 * SR)
    level = [_rms(mono[int(n["start"] * SR) : int(min(n["end"], n["start"] + LENGTHS[-1]) * SR)]) for n in notes]
    sounding = [v for v in level if v > 0]
    floor = quietest * float(np.median(sounding)) if sounding else 0.0
    for i, n in enumerate(notes):
        tones = (n["pitch"], *n.get("also", []))
        want_len = int((n["end"] - n["start"]) * SR) + pre
        for pitch in tones:
            found = bank.samples(pitch, i) or bank.samples(pitch, i, max_shift=5)
            if not found:
                continue
            # of the nearest pitch, a sample that lasts as long as the note if there is one, and then the one
            # nearest to it in length: a short note is played by a short one. It is not picked for sounding
            # like the note it stands in for: that would let a wrong label borrow a sound that hides it.
            near = [j for j in found if abs(notes[j]["pitch"] - pitch) == abs(notes[found[0]]["pitch"] - pitch)]
            lasts = n["end"] - n["start"]
            long_enough = [j for j in near if bank.lasts[j] + SKIP >= lasts - 0.03]
            best = min(long_enough or near, key=lambda j: abs(bank.lasts[j] + SKIP - lasts))
            m = notes[best]
            y = mono[max(0, int(m["start"] * SR) - pre) : int(m["end"] * SR)].astype(np.float64)
            ratio = 2 ** ((pitch - m["pitch"]) / 12)
            if ratio != 1.0:  # played faster or slower: the plain way to change a sample's pitch
                y = np.interp(np.arange(int(len(y) / ratio)) * ratio, np.arange(len(y)), y)
            if len(y) < want_len:
                y = _stretched(y, want_len)
            y = y[:want_len].copy()
            if len(y) < pre + 64:
                continue
            fade_in, fade_out = int(0.003 * SR), min(int(0.01 * SR), len(y) // 4)
            y[:fade_in] *= np.linspace(0, 1, fade_in)
            y[-fade_out:] *= np.linspace(1, 0, fade_out)
            have = _rms(y[pre : pre + int(LENGTHS[-1] * SR)])
            if have > 0:
                y *= max(level[i], floor) / np.sqrt(len(tones)) / have
            a = max(0, int(n["start"] * SR) - pre)
            y = y[: len(out) - a]
            out[a : a + len(y)] += y
    return out.astype(np.float32)
