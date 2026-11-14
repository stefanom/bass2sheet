"""Repeats: the bars of a song that sound alike, and what one of them says about the others.

Music repeats itself, and the tool does not hear every repeat the same way: a riff comes out right in some bars
and wrong in others. The repeats are found in the sound of the bass track, never in the notes, since the notes
are what may be wrong. Two bars are compared sixteenth by sixteenth: where they sound the same, they should be
written the same.

From that follow three kinds of help (see docs/copilot.md and, for how far each can be trusted,
docs/evaluation.md):
  - the song's patterns: which bars are repeats of which (`family`);
  - differences: a sixteenth written unlike its repeats that sound the same there (`differences`), with what
    they have instead: what the person fixed or left standing counts for more than what the tool wrote;
  - a fix carried to the repeats: the other bars that sound the same at that spot and are written as this one
    was before the fix (`same_mistake`), and the surgery that makes one bar read as another over a stretch of
    sixteenths (`copy_region`).

Bars are the tool's bars: sixteen steps of its beat grid, counted from its first downbeat. Where the grid slips
(an odd bar, a missed beat), the bars after it are still repeats of each other, but not of those before.
"""

import collections

import numpy as np

from . import ownsound
from .common import SR

SLOTS = 16  # sixteenths to a bar
ALIKE = 0.7  # two sixteenths sound the same from this much overlap of their spectra
RELATED = 0.5  # two bars are repeats of each other when this share of their sixteenths sound the same (or are silent)
SOUNDING = 4  # ... and at least this many sound in both
PLUCK_DB = 2.0  # a note start is taken from a repeat only where the level rises this much in this bar too
HELD_DB = 4.0  # ... and a note start is dropped only where it rises less than this
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def measure(mono, grid, origin):
    """What the bars sound like, from the bass track: dict(alike (bars, bars, 16) float16: how alike two bars sound
    in each sixteenth, 0..1; sounding (bars, 16): whether the bass sounds there; rise (bars, 16): how much the
    low-band level rises at the sixteenth, in dB, which tells a plucked note from one that is held)."""
    from .notes import attack_rise, low_band_db

    grid = np.asarray(grid, dtype=float)
    bars = max(0, (len(grid) - 1 - origin) // SLOTS)
    n = bars * SLOTS
    specs = np.zeros((n, ownsound.N_BINS), np.float32)
    level = np.full(n, -120.0)
    lift = attack_rise(low_band_db(np.asarray(mono, dtype=np.float32)), 2, 3)
    rise = np.zeros(n, np.float32)
    for g in range(n):
        t0, t1 = grid[origin + g], grid[origin + g + 1]
        seg = mono[max(0, int(t0 * SR)) : max(0, int(t1 * SR))]
        if len(seg):
            level[g] = 20 * np.log10(np.sqrt(np.mean(np.square(seg, dtype=np.float64))) + 1e-9)
        if t0 >= 0:
            s = ownsound.spectrum(mono, t0, t1 - t0 - ownsound.SKIP)
            if s is not None:
                specs[g] = s
            f = int(t0 * 100)
            if f < len(lift):
                rise[g] = lift[max(0, f - 3) : f + 4].max()
    sounding = (level > np.percentile(level, 95) - 25) & (specs.sum(1) > 0) if n else np.zeros(0, bool)
    specs = specs.reshape(bars, SLOTS, -1)
    alike = np.zeros((bars, bars, SLOTS), np.float16)
    for k in range(SLOTS):
        a = specs[:, k]
        for i in range(bars):  # row by row: a whole song at once would take gigabytes
            alike[i, :, k] = np.minimum(a[i], a).sum(1) / np.maximum(np.maximum(a[i], a).sum(1), 1e-9)
    return dict(alike=alike, sounding=sounding.reshape(bars, SLOTS), rise=rise.reshape(bars, SLOTS))


def letter(k):
    """A, B, ... Z, AA, AB, ..."""
    return LETTERS[k] if k < 26 else LETTERS[k // 26 - 1] + LETTERS[k % 26]


class Repeats:
    """The repeats of one song. `measured` is what `measure` gave; `grid` and `origin` are the tool's beat grid."""

    def __init__(self, measured, grid, origin):
        self.grid, self.first = np.asarray(grid, dtype=float), int(origin)
        alike = np.asarray(measured["alike"], dtype=np.float32)
        self.sounding, self.rise = np.asarray(measured["sounding"], bool), np.asarray(measured["rise"], np.float32)
        self.bars = len(self.sounding)
        self.n = self.bars * SLOTS
        both = self.sounding[:, None, :] & self.sounding[None, :, :]
        neither = ~self.sounding[:, None, :] & ~self.sounding[None, :, :]
        self.same = both & (alike >= ALIKE)  # sounds the same, and sounds at all
        self.alike = alike
        self.related = ((self.same | neither).mean(-1) >= RELATED) & (both.sum(-1) >= SOUNDING)
        np.fill_diagonal(self.related, False)
        self.mid = (self.grid[self.first : -1] + self.grid[self.first + 1 :]) / 2  # the middle of every sixteenth
        self._index = np.arange(len(self.grid), dtype=float)
        # patterns: a bar joins the first earlier pattern whose first bar it repeats; a bar with no repeat has none
        leaders, family = [], np.full(self.bars, -1)
        for i in range(self.bars):
            if not self.related[i].any():
                continue
            for f, lead in enumerate(leaders):
                if self.related[i, lead]:
                    family[i] = f
                    break
            else:
                leaders.append(i)
                family[i] = len(leaders) - 1
        while True:  # two patterns whose bars are mostly repeats of each other are one
            ids = sorted(set(family[family >= 0].tolist()))
            best = None
            for x, f in enumerate(ids):
                for h in ids[x + 1 :]:
                    share = self.related[np.ix_(family == f, family == h)].mean()
                    if share >= 0.5 and (best is None or share > best[0]):
                        best = (share, f, h)
            if best is None:
                break
            family[family == best[2]] = best[1]
        order = {f: k for k, f in enumerate(dict.fromkeys(family[family >= 0].tolist()))}  # lettered as they come
        self.family = np.array([order.get(int(f), -1) for f in family])

    # ---- where things are

    def slot_time(self, g):
        """When sixteenth g of the song starts (g counts from the first bar)."""
        return float(self.grid[self.first + g])

    def bar_of(self, t):
        """The bar a moment falls in, or None before the first bar or after the last."""
        g = int(np.searchsorted(self.grid, t, side="right")) - 1 - self.first
        return g // SLOTS if 0 <= g < self.n else None

    def to_bar(self, t, i, j):
        """The moment of bar j that corresponds to moment t of bar i: as far into its sixteenth."""
        p = np.interp(t, self.grid, self._index) + SLOTS * (j - i)
        return float(np.interp(p, self._index, self.grid))

    def cover(self, note):
        """The sixteenths [a, b) whose middle a note covers."""
        a, b = np.searchsorted(self.mid, [note["start"], note["end"]])
        return int(a), int(min(b, self.n))

    def tokens(self, notes):
        """What is written, sixteenth by sixteenth: (pitch (bars, 16), 0 where nothing; start (bars, 16): a note
        starts there; owner (bars, 16): which of `notes`, -1 where none)."""
        pitch, start, owner = np.zeros(self.n, int), np.zeros(self.n, bool), np.full(self.n, -1)
        for idx, n in enumerate(notes):
            a, b = self.cover(n)
            if a < b:
                pitch[a:b], owner[a:b] = n["pitch"], idx
                start[a + 1 : b] = False
                start[a] = True
        shape = (self.bars, SLOTS)
        return pitch.reshape(shape), start.reshape(shape), owner.reshape(shape)

    def within(self, spans):
        """The sixteenths whose middle lies in one of the spans [(start, end)]: (bars, 16) bool."""
        mid = self.mid[: self.n].reshape(self.bars, SLOTS)
        out = np.zeros(mid.shape, bool)
        for a, b in spans:
            out |= (mid >= a) & (mid <= b)
        return out

    def trust(self, notes, machine_pitch, worked):
        """How far each sixteenth is the person's word: 2 where they changed something; 1 where they worked
        through it and left it (`worked`: spans of the song, in seconds); 0 where it is only what the tool
        wrote."""
        pitch, _, owner = self.tokens(notes)
        mine = np.array([n.get("state", "machine") != "machine" for n in notes] + [False])
        level = np.where((pitch != machine_pitch) | mine[owner], 2, 0)
        return np.maximum(level, self.within(worked).astype(int))

    # ---- what the repeats say

    def _allowed(self, i, k, start_here, start_there):
        """A difference in where a note starts counts only if this bar's own sound agrees with it."""
        if start_there and not start_here:
            return self.rise[i, k] >= PLUCK_DB
        if start_here and not start_there:
            return self.rise[i, k] < HELD_DB
        return True

    def differences(self, notes, machine_pitch, worked, kept=()):
        """Where a bar is written unlike its repeats that sound the same there: see `differences_of`. `kept`
        are spans the person said to leave as they are written: nothing is marked there."""
        pitch, start, _ = self.tokens(notes)
        return self.differences_of(pitch, start, self.trust(notes, machine_pitch, worked), self.within(kept))

    def differences_of(self, pitch, start, level, settled=None):
        """From what is written (`tokens`) and whose word it is (`trust`): a list of
        dict(bar, k0, k1, source, why, count), one per stretch of sixteenths k0..k1 of `bar` (counted from 0)
        that bar `source` has differently while sounding the same. Only what the tool wrote is marked: a
        sixteenth the person has been through (`level` above 0) is theirs. `why` is "fixed" when the source is
        something the person changed, "passed" when they played past it and left it: one such repeat is
        enough. It is "most" when all are only the tool's and most of the repeats (`count` of them, at least
        two) have another note there. All three are signs worth a listen, not certainties."""
        code = pitch * 2 + start
        out = []
        for i in range(self.bars):
            js = np.flatnonzero(self.related[i])
            if not len(js):
                continue
            flagged, values = {}, {}
            for k in np.flatnonzero(self.sounding[i]):
                peers = js[self.same[i, js, k]]
                if not len(peers) or (settled is not None and settled[i, k]):
                    continue
                if level[i, k] > 0:
                    continue  # the person has been through this: it is their word
                top = level[peers, k].max()
                best = peers[level[peers, k] == top]
                value, count = collections.Counter(code[best, k].tolist()).most_common(1)[0]
                if value == code[i, k] or count <= len(best) / 2:
                    continue
                if top == 0 and (count < 2 or not value // 2 or not pitch[i, k]):
                    continue  # the tool against the tool: only where both have a note, and two repeats agree
                if value // 2 == pitch[i, k] and not self._allowed(i, k, bool(start[i, k]), bool(value % 2)):
                    continue
                agreeing = best[code[best, k] == value]
                why = "fixed" if top == 2 else "passed" if top == 1 else "most"
                last = flagged.get(int(k) - 1)
                if last and last[1] == why and last[0] in agreeing:
                    source = last[0]  # the repeat that supplied the sixteenth before, as long as it can
                else:
                    source = int(agreeing[np.argmax(self.alike[i, agreeing, k])])
                flagged[int(k)] = (source, why, int(count))
                values[int(k)] = int(value)
            # a note against silence counts only as a whole note: one this bar has and the repeat has not, from
            # its start to its end, or one the repeat has from its start. How long a note is held is not marked.
            dropped = set()
            for k in sorted(flagged):
                there, here = values[k] // 2, int(pitch[i, k])
                if there and here:
                    continue
                if here:  # every sixteenth of this bar's note must be silent in the repeat
                    a = k
                    while a > 0 and pitch[i, a] == here and not start[i, a]:
                        a -= 1
                    b = k + 1
                    while b < SLOTS and pitch[i, b] == here and not start[i, b]:
                        b += 1
                    whole = start[i, a] and all(values.get(q, 1) // 2 == 0 and q in flagged for q in range(a, b))
                else:  # the repeat's note must start in the stretch that is marked
                    a = k
                    while (
                        not values[a] % 2 and a - 1 in flagged and values[a - 1] // 2 == there and not pitch[i, a - 1]
                    ):
                        a -= 1
                    whole = bool(values[a] % 2)
                if not whole:
                    dropped.add(k)
            # the same notes with the line between two of them a sixteenth earlier or later are not a difference
            # worth marking: a note that starts halfway between two sixteenths falls now on one, now on the other
            for k in sorted(flagged):
                j, there, here = flagged[k][0], values[k] // 2, int(pitch[i, k])
                for d in (-1, 1):
                    if not 0 <= k + d < SLOTS or not 0 <= k - d < SLOTS:
                        continue
                    if there != here and pitch[i, k + d] == there and pitch[j, k - d] == here:
                        dropped.add(k)
                    if (
                        there == here == pitch[i, k + d] == pitch[j, k + d]
                        and start[i, k + d] == bool(values[k] % 2)
                        and start[j, k + d] == start[i, k]
                    ):
                        dropped.add(k)
            flagged = {k: v for k, v in flagged.items() if k not in dropped}
            run = []
            for k in sorted(flagged):  # stretches of sixteenths that one repeat can supply
                if run and k == run[-1] + 1 and flagged[k][:2] == flagged[run[0]][:2]:
                    run.append(k)
                    continue
                if run:
                    out.append(self._stretch(i, run, flagged))
                run = [k]
            if run:
                out.append(self._stretch(i, run, flagged))
        return out

    @staticmethod
    def _stretch(i, run, flagged):
        source, why, _ = flagged[run[0]]
        return dict(bar=i, k0=run[0], k1=run[-1], source=source, why=why, count=min(flagged[k][2] for k in run))

    def changed(self, before, after):
        """The stretches of sixteenths where what is written changed: [(bar, k0, k1)], from two `tokens` results."""
        diff = (before[0] != after[0]) | (before[1] != after[1])
        out = []
        for i in np.flatnonzero(diff.any(1)):
            ks = np.flatnonzero(diff[i])
            k0 = prev = int(ks[0])
            for k in map(int, ks[1:]):
                if k != prev + 1:
                    out.append((int(i), k0, prev))
                    k0 = k
                prev = k
            out.append((int(i), k0, prev))
        return out

    def same_mistake(self, before, after, level, settled=None):
        """After a fix: the other bars that sound the same where something was changed, and are written there as
        the fixed bar was before: [dict(source, bar, k0, k1)]. A bar the person has already been through (`level`
        above 0 anywhere in the stretch, or `settled`: they said to keep what is written) is left alone: if it
        differs from what they have now fixed, it shows as a difference instead. So is one that sounds
        different in more than a fifth of the stretch."""
        code_before = before[0] * 2 + before[1]
        code_now = after[0] * 2 + after[1]
        out = []
        for i, k0, k1 in self.changed(before, after):
            ks = np.arange(k0, k1 + 1)
            for j in np.flatnonzero(self.related[i]):
                if (level[j, ks] >= 1).any() or (settled is not None and settled[j, ks].any()):
                    continue
                if not (code_now[j, ks] == code_before[i, ks]).all():
                    continue
                sounds = self.sounding[i, ks] | self.sounding[j, ks]
                if not sounds.any() or self.same[i, j, ks][sounds].mean() < 0.8:
                    continue
                starts = (after[1][i, ks] != after[1][j, ks]) & (after[0][i, ks] == after[0][j, ks])
                if any(not self._allowed(j, k, bool(after[1][j, k]), bool(after[1][i, k])) for k in ks[starts]):
                    continue
                out.append(dict(source=int(i), bar=int(j), k0=k0, k1=k1))
        return out

    def unlike(self, pitch, start, i, j):
        """Where bar j is written unlike bar i while the two sound the same there: [(k0, k1)], stretches of
        sixteenths, from what is written (`tokens`). This is one bar against one other, for a person to look at
        and hear; nothing follows from it by itself. As in `differences_of`: a difference only in where a note
        starts counts where bar j's own sound agrees with it; a note against silence counts only as a whole
        note (how long a note is held is not a difference); and the same notes with the line between two of
        them a sixteenth earlier or later are the same notes."""
        inside = lambda q: 0 <= q < SLOTS
        flagged = []
        for k in range(SLOTS):
            there, here = int(pitch[i, k]), int(pitch[j, k])
            if (there, bool(start[i, k])) == (here, bool(start[j, k])) or not self.same[i, j, k]:
                continue
            if there == here:  # the same note, begun here in one bar and held on in the other
                if not self._allowed(j, k, bool(start[j, k]), bool(start[i, k])):
                    continue
                if any(
                    inside(k + d)
                    and pitch[i, k + d] == pitch[j, k + d] == there
                    and start[i, k + d] == start[j, k]
                    and start[j, k + d] == start[i, k]
                    for d in (-1, 1)
                ):
                    continue
            elif there and here:
                if any(
                    inside(k + d) and inside(k - d) and pitch[j, k + d] == there and pitch[i, k - d] == here
                    for d in (-1, 1)
                ):
                    continue
            else:  # a note one bar has and the other has not: all of it, from its start, or it is not counted
                x, other, p = (i, j, there) if there else (j, i, here)
                a = k
                while a > 0 and pitch[x, a] == p and not start[x, a]:
                    a -= 1
                b = k + 1
                while b < SLOTS and pitch[x, b] == p and not start[x, b]:
                    b += 1
                if not (start[x, a] and all(pitch[other, q] == 0 and self.same[i, j, q] for q in range(a, b))):
                    continue
            flagged.append(k)
        out = []
        for k in flagged:
            if out and k == out[-1][1] + 1:
                out[-1] = (out[-1][0], k)
            else:
                out.append((k, k))
        return out

    # ---- making one bar read as another

    def pieces(self, notes, i, j, k0, k1):
        """What bar i has from sixteenth k0 to k1, as it would lie in bar j: [dict(start, end, pitch, also,
        starts_here, ends_here, slot)]. A note that begins before the stretch or goes on after it is cut at its
        edge."""
        g0, g1 = i * SLOTS + k0, i * SLOTS + k1 + 1
        d0, d1 = self.slot_time(j * SLOTS + k0), self.slot_time(j * SLOTS + k1 + 1)
        out = []
        for n in sorted(notes, key=lambda n: n["start"]):
            a, b = self.cover(n)
            if a >= b or b <= g0 or a >= g1:
                continue
            starts_here, ends_here = a >= g0, b <= g1
            out.append(
                dict(
                    start=round(max(self.to_bar(n["start"], i, j), 0.0) if starts_here else d0, 4),
                    end=round(self.to_bar(n["end"], i, j) if ends_here else d1, 4),
                    pitch=n["pitch"],
                    also=list(n.get("also", [])),
                    starts_here=starts_here,
                    ends_here=ends_here,
                    slot=max(a, g0) + SLOTS * (j - i),  # the sixteenth of the song it starts on, in bar j
                )
            )
        return out
