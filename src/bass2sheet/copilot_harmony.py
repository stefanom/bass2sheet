"""A song's harmony, for the copilot's harmony page: the chords in time, named against a home key, the job each
does, the few loops the song is made of, and what the bass plays under each chord.

The page is for understanding a song before playing it, and for remembering it. So everything here is said
against a home: the note the song comes back to. The tool finds a key; the person may pick another (a song in A
minor and one in C major use the same notes, and which is home is heard, not counted). Their choice is kept
beside the song, in harmony.json.

The map of the page is a map of jobs, from functional harmony: a chord rests (home), drifts (away) or pulls
(tension). Where a chord sits says three things: its job (the region), how far outside the key it is (the row),
and how bright it is (left to right, on the line of fifths). The rules are rules of thumb, written down in `job`,
`ring` and `bright`; they are not measured from anything.
"""

import collections
import json
import math
import os

import numpy as np

from . import copilot, repeats
from .common import Bass2SheetError, atomic_write_text
from .rhythm import spelling
from .score import STRINGS, frets

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "copilot_harmony.html")

# A chord's step above home, named on the major scale of home whatever the key's mode: a song in a minor key
# then reads i, bIII, iv, v, bVI, bVII, which is how the people who play these songs write them.
STEPS = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")
# What a bass note is to the chord it sounds under, by its distance above the chord's root in semitones.
ROLES = {0: "root", 3: "third", 4: "third", 7: "fifth", 10: "seventh", 11: "seventh"}
ROLE_ORDER = ("root", "fifth", "third", "seventh", "other")
# How a chord's root moves to the next one's, by the distance up in semitones (the nearer way round is named).
MOTIONS = {
    1: "up a semitone",
    2: "up a tone",
    3: "up a minor third",
    4: "up a major third",
    5: "up a fourth",
    6: "a tritone",
    7: "down a fourth",
    8: "down a major third",
    9: "down a minor third",
    10: "down a tone",
    11: "down a semitone",
}


def minor_like(quality):
    """Whether a chord of this quality has a minor third: "m", "m7", "dim" do; "", "7", "maj7", "sus4" do not."""
    return (quality.startswith("m") and not quality.startswith("maj")) or quality.startswith("dim")


def numeral(root, quality, home):
    """A chord as a step above home: `IV`, `vi`, `V7`, `bVII`, `ii7`, `vii°`. `home` is (root, major). Small
    letters are chords with a minor third."""
    step = STEPS[(root - home[0]) % 12]
    if quality == "m7b5":
        return step.lower() + "ø7"
    if quality.startswith("dim"):
        return step.lower() + "°" + quality[3:]
    if minor_like(quality):
        return step.lower() + quality[1:]
    return step + quality


def key_name(home):
    return spelling(home[0], home[1])[home[0]] + (" major" if home[1] else " minor")


def relative(home):
    """The key with the same notes and the other mode: A minor for C major, C major for A minor."""
    return ((home[0] - 3) % 12, False) if home[1] else ((home[0] + 3) % 12, True)


def role(pitch, root):
    """What a bass note is to a chord: its root, third, fifth or seventh, or `other` (a passing note)."""
    return ROLES.get((pitch - root) % 12, "other")


# ---- the map: a chord's job, how far outside the key it is, and how bright it is

# The notes of a chord, in semitones above its root.
TONES = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
}
MAJOR = frozenset((0, 2, 4, 5, 7, 9, 11))
MINOR = frozenset((0, 2, 3, 5, 7, 8, 10, 11))  # the natural minor scale, and the raised seventh its V chord needs
JOBS = ("rest", "drift", "pull")
# The job of a plain chord by its step above home, where no rule below says otherwise.
BY_STEP = ("rest", "drift", "drift", "rest", "rest", "drift", "pull", "pull", "drift", "rest", "drift", "pull")
# A root's place on the line of fifths from home, by its step above home; and, for a diminished chord that passes
# between two notes of the key, the place of the sharp it is heard as.
LINE = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)
LINE_PASSING = (0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5)
# How bright a chord's own notes make it: their mean place on the line of fifths from its root.
OWN_LIGHT = {
    "": 5 / 3,
    "m": -2 / 3,
    "dim": -3,
    "aug": 4,
    "sus4": 0,
    "6": 2,
    "m6": 0.25,
    "7": 0.75,
    "maj7": 2.5,
    "m7": -1,
    "m7b5": -2.75,
    "dim7": -4.5,
}
# The chords a map always shows, as (step above home, quality): the ones a song in that mode is likely to use.
STOPS = {
    True: dict(
        rest=((0, ""), (9, "m"), (4, "m"), (0, "m"), (3, "")),
        drift=((5, ""), (2, "m"), (5, "m"), (8, ""), (10, ""), (2, "dim"), (1, ""), (7, "m")),
        pull=((7, ""), (11, "dim"), (9, "7"), (4, "7"), (2, "7"), (0, "7"), (11, "7"), (1, "7")),
    ),
    False: dict(
        rest=((0, "m"), (3, ""), (0, "")),
        drift=((5, "m"), (2, "dim"), (8, ""), (10, ""), (5, ""), (1, ""), (2, "m"), (7, "m")),
        pull=((7, ""), (11, "dim"), (0, "7"), (2, "7"), (3, "7"), (10, "7"), (5, "7"), (1, "7")),
    ),
}
# The picture is 760 by 534. Each job has a region (x, y, width, height); a region's rows are the distances from
# the key, the row that faces the middle of the picture first.
REGIONS = dict(drift=(6, 30, 330, 190), pull=(348, 30, 406, 190), rest=(220, 318, 320, 190))
ROWS = dict(drift=(182, 124, 66), pull=(182, 124, 66), rest=(356, 414, 472))
UNIT, INSIDE, APART = (
    60,
    24,
    39,
)  # pixels for one step of brightness; a chord's least distance from an edge, and from a chord
WHEEL = dict(box=(600, 430), centre=(300, 215), maj=180, min=130, dim=86)  # a wheel is drawn 600 by 430


def tones(quality):
    return TONES.get(quality) or TONES["m" if minor_like(quality) else ""]


def kind(quality):
    """The family a quality belongs to: `dom` (a dominant seventh; also augmented and suspended chords, which
    lean the same way), `dim`, `min` or `maj`."""
    if quality in ("7", "9", "13", "7sus4", "aug", "sus4"):
        return "dom"
    if quality.startswith("dim") or quality == "m7b5":
        return "dim"
    return "min" if minor_like(quality) else "maj"


def job(root, quality, home):
    """What a chord does in a key: `rest` (home), `drift` (away) or `pull` (tension)."""
    step, family = (root - home[0]) % 12, kind(quality)
    if family == "dom":
        return "rest" if step == 0 and quality == "sus4" else "pull"
    if family == "dim":  # the one on the second step leads away (ii° to V); the others lean on a neighbour
        return "drift" if step == 2 and quality in ("dim", "m7b5") else "pull"
    if family == "maj" and step in ((2, 4, 9, 11) if home[1] else (2, 4)):
        return "pull"  # II, III, VI and VII as major chords are the dominants of other chords of the key
    if family == "min" and step == 7:
        return "drift"  # a minor v does not pull
    return BY_STEP[step]


def ring(root, quality, home):
    """How far outside the key a chord is: 0, every note is in the key; 1, one step out (its notes are in the
    key of the other mode on the same home; or it is the dominant of a chord of the key, or a diminished chord a
    semitone under one); 2, further."""
    step = (root - home[0]) % 12
    notes = {(step + t) % 12 for t in tones(quality)}
    own, other = (MAJOR, MINOR) if home[1] else (MINOR, MAJOR)
    if notes <= own:
        return 0
    if notes <= other:
        return 1
    family = kind(quality)
    if (family == "dom" and (step - 7) % 12 in own) or (family == "dim" and (step + 1) % 12 in own):
        return 1
    return 2


def bright(root, quality, home):
    """How bright a chord is in a key: its root's place on the line of fifths from home, and its own notes'."""
    step = (root - home[0]) % 12
    passing = kind(quality) == "dim" and step not in (MAJOR if home[1] else MINOR)
    return (LINE_PASSING if passing else LINE)[step] + OWN_LIGHT.get(
        quality, OWN_LIGHT["m" if minor_like(quality) else ""]
    )


def far(root, quality, home):
    """A chord's distance from rest, for the profile of a song: 0 at home, more for a chord that drifts, most for
    one that pulls; more the further outside the key it is, and a little more if it holds a tritone."""
    does, out = job(root, quality, home), ring(root, quality, home)
    if does == "rest" and out == 0 and root == home[0]:
        return 0.0
    notes = tones(quality)
    tritone = any((a - b) % 12 == 6 for a in notes for b in notes)
    return round(dict(rest=0.8, drift=2.2, pull=3.6)[does] + 0.8 * out + (0.6 if tritone else 0), 2)


def place(step, quality):
    """Which spot of a map a chord goes to: chords of one step and one family share a spot (C and Cmaj7; Am and
    Am7), and so do V and V7, which a song uses as one."""
    family = kind(quality)
    return step, "maj" if family == "dom" and step == 7 and quality in ("7", "9", "13") else family


def _across(light, region):
    """Where a brightness falls across a region, in pixels: the regular chords of the region are spread in the
    order of their brightness, a step of brightness 60 pixels wide and no gap wider than that, the run centred."""
    x0, _, width, _ = region["box"]
    marks, at = region["marks"], region["at"]
    if light <= marks[0]:  # darker or brighter than every regular chord: a little beyond the last one
        along = at[0] - min(0.5, (marks[0] - light) * 0.3)
    elif light >= marks[-1]:
        along = at[-1] + min(0.5, (light - marks[-1]) * 0.3)
    else:
        along = float(np.interp(light, marks, at))
    x = x0 + width / 2 + (along - at[-1] / 2) * UNIT
    return min(max(x, x0 + INSIDE), x0 + width - INSIDE)


def roads(home, spent):
    """The map of jobs for a home and a song: its regular chords and the song's own, each where its job, its
    distance from the key and its brightness put it. `spent` is the song's seconds on each (root, quality).
    Returns the spots, and for each of the song's chords the number of its spot."""
    regions = {}
    for does in JOBS:
        marks = sorted({bright((home[0] + step) % 12, q, home) for step, q in STOPS[home[1]][does]})
        at = [0.0]
        for a, b in zip(marks, marks[1:]):
            at.append(at[-1] + min(1.0, b - a))
        regions[does] = dict(box=REGIONS[does], marks=marks, at=at)
    spots, index = [], {}

    def spot(step, quality, regular):
        root = (home[0] + step) % 12
        does, out = job(root, quality, home), ring(root, quality, home)
        x, y = _across(bright(root, quality, home), regions[does]), ROWS[does][out]
        if not regular:  # not a regular chord: slid along its row to the nearest free place, else to the next row
            box, taken = REGIONS[does], [(s["x"], s["y"]) for s in spots]
            for row in sorted(range(3), key=lambda r: abs(r - out)):
                free = [x + 13 * k for k in range(-28, 29) if box[0] + INSIDE <= x + 13 * k <= box[0] + box[2] - INSIDE]
                free = [c for c in free if all(math.hypot(c - tx, ROWS[does][row] - ty) >= APART for tx, ty in taken)]
                if free:
                    x, y = min(free, key=lambda c: abs(c - x)), ROWS[does][row]
                    break
        index[place(step, quality)] = len(spots)
        spots.append(dict(root=root, quality=quality, job=does, ring=out, regular=regular, seconds=0.0))
        spots[-1].update(x=round(float(x), 1), y=y, r=19 if out == 0 else 18)

    for does in JOBS:
        for step, quality in STOPS[home[1]][does]:
            spot(step, quality, True)
    return _settle(home, spent, spots, index, lambda step, quality: spot(step, quality, False))


def wheel(home, spent, by="fifths"):
    """The same chords on a circle, home at the top: every major chord on the outer ring, every minor chord on
    the middle one and every diminished chord inside. By fifths, neighbours are a fifth apart and a minor chord
    sits under the major chord with the same notes; by semitones, neighbours are a semitone apart and the three
    chords on one note sit on one spoke."""
    top = home[0] if home[1] or by != "fifths" else (home[0] + 3) % 12  # the major chord at the top
    lift = dict(maj=0, min=3, dim=1) if by == "fifths" else dict(maj=0, min=0, dim=0)
    spots, index = [], {}
    for family, quality in (("maj", ""), ("min", "m"), ("dim", "dim")):
        for root in range(12):
            turn = (root + lift[family] - top) % 12
            turn = turn * 7 % 12 if by == "fifths" else turn
            angle = math.radians(30 * turn)
            cx, cy = WHEEL["centre"]
            x, y = cx + WHEEL[family] * math.sin(angle), cy - WHEEL[family] * math.cos(angle)
            index[((root - home[0]) % 12, family)] = len(spots)
            does, out = job(root, quality, home), ring(root, quality, home)
            spots.append(dict(root=root, quality=quality, job=does, ring=out, regular=True, seconds=0.0))
            spots[-1].update(x=round(x, 1), y=round(y, 1), r=dict(maj=22, min=19, dim=15)[family], turn=turn)
    return _settle(home, spent, spots, index, None)


def _settle(home, spent, spots, index, add):
    """Put a song's chords on a map's spots: each spot gets the seconds the song spends there and the quality
    the song uses most there. A chord with no spot gets one (`add`); on a wheel, which has no ring for them, a
    dominant goes with the major chords. Returns the spots, and the number of the spot of each (root, quality)."""
    used, where = collections.defaultdict(collections.Counter), {}
    for (root, quality), seconds in spent.items():
        step, family = place((root - home[0]) % 12, quality)
        if (step, family) not in index:
            if add is None:
                family = "maj"
            else:
                add(step, quality)
        used[index[(step, family)]][quality] += seconds
        where[(root, quality)] = index[(step, family)]
    for at, qualities in used.items():
        quality = qualities.most_common(1)[0][0]
        spots[at].update(quality=quality, seconds=round(sum(qualities.values()), 1))
        if add is None:  # on a wheel a spot's colour follows the chord the song has there
            root = spots[at]["root"]
            spots[at].update(job=job(root, quality, home), ring=ring(root, quality, home))
    return spots, where


# ---- the song in parts: the few loops it is made of

LOOPS = (1, 2, 3, 4, 6, 8, 12, 16)  # how many bars a loop may be


def _round(labels, i, size):
    """How many bars from bar i on go round one loop of `size` bars: the first round is the loop, each further
    one must be like it (three bars in four; every bar of a loop of one or two), and a last one may stop early."""
    loop = labels[i : i + size]
    if len(loop) < size:
        return 0
    n = size
    while True:
        nxt = labels[i + n : i + n + size]
        if len(nxt) == size and sum(a == b for a, b in zip(loop, nxt)) >= (size if size <= 2 else 0.75 * size):
            n += size
            continue
        if n >= 2 * size:  # a last round that stops early
            k = 0
            while k < len(nxt) and nxt[k] == loop[k]:
                k += 1
            n += k
        return n


def _agreed(labels, i, n, size):
    """(the loop, the bars that differ from it) of n bars from bar i on that go round `size` bars: the loop is
    what most rounds have in each bar."""
    loop = [collections.Counter(labels[i + k : i + n : size]).most_common(1)[0][0] for k in range(size)]
    return loop, sum(labels[i + j] != loop[j % size] for j in range(n))


def parts(labels):
    """A song as parts, from what each bar holds (`labels`, anything that can be compared): a part is a loop of
    a few bars going round at least twice, or a passage that does not repeat. [dict(first, bars, loop, skip,
    letter)]: `first` counts bars from 0; `loop` is what one round holds, bar by bar (what most rounds have
    there); parts with the same loop have the same letter, and one that begins part of the way round it says how
    many bars in (`skip`).

    Of all the ways to cut the song, the one is taken that puts the most bars into loops: a bar in a loop counts
    one, a bar of a loop that differs from what most rounds have there costs three, a bar of a passage costs
    one, and every loop costs a little, the longer the more, so that the song is not cut finer than it need be
    and four bars twice are not called a loop of eight."""
    n = len(labels)
    best, cut = [0.0] * (n + 1), [None] * (n + 1)  # from bar i on: the best that can be had, and how it begins
    for i in range(n - 1, -1, -1):
        best[i], cut[i] = best[i + 1] - 1, (1, 0)  # a bar of a passage
        for size in LOOPS:
            bars = _round(labels, i, size)
            if bars >= max(2 * size, 4):  # a loop goes round at least twice and lasts four bars or more
                good = bars - 3 * _agreed(labels, i, bars, size)[1] - 0.5 - 0.01 * size + best[i + bars]
                if good > best[i]:
                    best[i], cut[i] = good, (bars, size)
    out, i, letters = [], 0, {}
    while i < n:
        bars, size = cut[i]
        if size == 0 and out and out[-1]["size"] == 0:
            out[-1]["bars"] += 1
        else:
            out.append(dict(first=i, bars=bars, size=size))
        i += bars
    known = []  # the loops so far
    for part in out:
        size = part.pop("size")
        loop, skip = tuple(_agreed(labels, part["first"], part["bars"], size or part["bars"])[0]), 0
        if size and loop not in known:  # an earlier loop, begun part of the way round, is that loop
            turned = [(other, s) for other in known for s in range(1, size) if other[s:] + other[:s] == loop]
            if turned:
                loop, skip = turned[0]
            else:
                known.append(loop)
        part.update(loop=list(loop), skip=skip, letter=repeats.letter(letters.setdefault(loop, len(letters))))
    return out


# ---- what the bass plays over a chord

DEGREES = ("R", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")


def degree(pitch, root, base):
    """A bass note as a step of the chord it sounds under: R, 3, 5, b7; `8` is the root an octave above `base`
    (the pitch of the root the line stands on), an arrow marks other notes above that octave or below the root."""
    up = pitch - base
    name = DEGREES[(pitch - root) % 12]
    if name == "R":
        return "R" if up == 0 else "8" if up == 12 else "R↑" if up > 0 else "R↓"
    return name + ("↑" if up > 12 else "↓" if up < 0 else "")


def chords_of(machine):
    """The tool's chords as stretches of time, in order: [dict(start, end, root, quality, score)]. A song
    prepared when the tool wrote one chord a bar has one a bar; a bar where it heard no harmony has none."""
    grid, origin, length = machine["grid"], machine["origin"], machine["length"]
    out = []
    for c in machine.get("chords") or []:
        i = origin + 16 * (c["bar"] - 1)
        if not 0 <= i < len(grid) or float(c.get("score", 1.0)) <= 0:
            continue
        end = grid[i + 16] if i + 16 < len(grid) else length
        start = max(0.0, grid[i])
        if end - start <= 0.05:
            continue
        chord = dict(start=round(start, 3), end=round(min(end, max(length, start)), 3), root=int(c["root"]))
        out.append(dict(chord, quality=c.get("quality", ""), score=round(float(c.get("score", 1.0)), 3)))
    return out


def read_home(folder):
    """The home the person chose for a song, as (root, major), or None."""
    saved = copilot.read_json(os.path.join(folder, "harmony.json")) or {}
    home = saved.get("home")
    return (int(home[0]) % 12, bool(home[1])) if home else None


def set_home(folder, home):
    """Keep the person's choice of home, (root, major); None goes back to the key the tool found."""
    path = os.path.join(folder, "harmony.json")
    saved = copilot.read_json(path) or {}
    if home is None:
        saved.pop("home", None)
    else:
        root, major = home
        if not (isinstance(root, int) and 0 <= root < 12 and isinstance(major, bool)):
            raise Bass2SheetError("a home is one of the twelve notes, and major or minor")
        saved["home"] = [root, major]
    atomic_write_text(path, json.dumps(saved))


def _under(notes, start, end):
    """The bass note a chord stands on: the one sounding when the chord starts, else the first that starts in it."""
    for n in notes:
        if n["start"] <= start + 0.06 < n["end"]:
            return n
    return next((n for n in notes if start <= n["start"] < end), None)


def _bass(machine, notes, chords, bars):
    """The bass line against the chords: each note with where it is played on the neck, the chord it sounds
    under, what it is to that chord, and the sixteenth of its bar it starts on."""
    grid = np.asarray(machine["grid"])
    style = machine["settings"].get("tab", "low")
    played = frets([int(n["pitch"]) for n in notes], style, also=[n.get("also", []) for n in notes])
    starts = [c["start"] for c in chords]
    base = {}  # per chord: the pitch of the root the line stands on there
    out = []
    for n, (string, fret) in zip(notes, played):
        k = int(np.searchsorted(starts, n["start"] + 0.03, side="right")) - 1
        under = chords[k] if k >= 0 and n["start"] + 0.03 < chords[k]["end"] else None
        bar = max(0, int(np.searchsorted(bars, n["start"] + 0.03, side="right")) - 1)
        slot = int(np.argmin(np.abs(grid - n["start"]))) - (machine["origin"] + 16 * bar)
        note = dict(start=round(n["start"], 3), end=round(n["end"], 3), pitch=int(n["pitch"]), string=string, fret=fret)
        note.update(bar=bar + 1, slot=max(0, min(15, slot)), chord=k if under else None, degree=None, role=None)
        if under:
            root = under["root"]
            if k not in base:
                here = [m["pitch"] for m in notes if under["start"] - 0.03 <= m["start"] < under["end"] - 0.03]
                roots = [q for q in here if (q - root) % 12 == 0]
                base[k] = min(roots) if roots else n["pitch"] - (n["pitch"] - root) % 12
            note.update(degree=degree(n["pitch"], root, base[k]), role=role(n["pitch"], root))
        out.append(note)
    return out


def _loops(song_parts, bass, n_bars):
    """The distinct parts of a song, for remembering it: each loop once, with how often it comes and what the
    bass plays over each of its bars (the line most rounds have there, and the share of rounds that have it)."""
    in_bar = collections.defaultdict(list)
    for note in bass:
        in_bar[note["bar"] - 1].append((note["slot"], note["degree"], note["role"]))
    loops = {}
    for part in song_parts:
        loop = loops.setdefault(part["letter"], dict(letter=part["letter"], size=part["size"], loop=part["loop"]))
        loop.setdefault("lines", [collections.Counter() for _ in range(part["size"])])
        loop.update(bars=loop.get("bars", 0) + part["bars"], times=loop.get("times", 0) + 1)
        loop["rounds"] = round(loop.get("rounds", 0) + part["bars"] / part["size"], 2)
        for k in range(part["bars"]):
            bar = part["bar"] - 1 + k
            if bar < n_bars:
                loop["lines"][(k + part["skip"]) % part["size"]][tuple(in_bar[bar])] += 1
    for loop in loops.values():
        lines = []
        for seen in loop.pop("lines"):
            line, n = seen.most_common(1)[0] if seen else ((), 0)
            lines.append(
                dict(
                    notes=[dict(slot=s, degree=d, role=r) for s, d, r in line],
                    share=round(n / max(1, sum(seen.values())), 2),
                )
            )
        shapes = {tuple((n["slot"], n["degree"]) for n in line["notes"]) for line in lines if line["notes"]}
        loop.update(bass=lines, same=len(shapes) == 1 and loop["size"] > 1)
    return sorted(loops.values(), key=lambda loop: -loop["bars"])


def view(desk):
    """Everything the harmony page shows of a song (see the page for how). Times are seconds of the record."""
    machine = desk.machine
    found = (int(machine["key"][1]) % 12, bool(machine["key"][2]))
    chosen = read_home(desk.folder)
    home = chosen or found
    names = spelling(home[0], home[1])
    notes = sorted(desk.notes, key=lambda n: n["start"])
    bars = [machine["grid"][i] for i in range(machine["origin"], len(machine["grid"]), 16)]
    bar_ends = bars[1:] + [max(machine["length"], bars[-1])] if bars else []
    patterns = desk.patterns()["bars"]
    said = lambda root, quality: dict(  # noqa: E731
        root=root,
        quality=quality,
        name=names[root] + quality,
        numeral=numeral(root, quality, home),
        job=job(root, quality, home),
    )
    chords, spent, played = [], collections.Counter(), collections.Counter()
    for c in chords_of(machine):
        on = _under(notes, c["start"], c["end"])
        for n in notes:  # how long the bass spends on each part of the chord
            both = min(n["end"], c["end"]) - max(n["start"], c["start"])
            if both > 0:
                played[role(n["pitch"], c["root"])] += both
        spent[(c["root"], c["quality"])] += c["end"] - c["start"]
        chord = dict(c, **said(c["root"], c["quality"]), bar=max(1, sum(t <= c["start"] + 0.01 for t in bars)))
        chord.update(ring=ring(c["root"], c["quality"], home), far=far(c["root"], c["quality"], home))
        chord.update(tones=[(c["root"] + t) % 12 for t in tones(c["quality"])])
        chord.update(
            bass=None if on is None else int(on["pitch"]) % 12,
            role=None if on is None else role(on["pitch"], c["root"]),
        )
        chords.append(chord)
    maps = {}
    made = dict(roads=roads(home, spent), wheel=wheel(home, spent), semitones=wheel(home, spent, "semitones"))
    for name, (spots, where) in made.items():
        for spot in spots:
            spot.update(
                name=names[spot["root"]] + spot["quality"], numeral=numeral(spot["root"], spot["quality"], home)
            )
        maps[name] = dict(box=(760, 534) if name == "roads" else WHEEL["box"], spots=spots)
        for chord in chords:
            chord.setdefault("on", {})[name] = where[(chord["root"], chord["quality"])]
    # what each bar holds: its chords in order (a chord counts in a bar it fills a tenth of), then the parts
    held = [
        tuple((c["root"], c["quality"]) for c in chords if min(c["end"], t1) - max(c["start"], t0) > 0.1 * (t1 - t0))
        for t0, t1 in zip(bars, bar_ends)
    ]
    song_parts = [
        dict(
            letter=part["letter"],
            bar=part["first"] + 1,
            bars=part["bars"],
            size=len(part["loop"]),
            skip=part["skip"],
            start=round(max(0.0, bars[part["first"]]), 3),
            end=round(bar_ends[part["first"] + part["bars"] - 1], 3),
            loop=[[said(*chord) for chord in bar] for bar in part["loop"]],
        )
        for part in parts(held)
    ]
    bass = _bass(machine, notes, chords, bars)
    runs = [c for k, c in enumerate(chords) if k == 0 or _same(c) != _same(chords[k - 1])]
    moves = collections.Counter((_same(a), _same(b)) for a, b in zip(runs, runs[1:]))
    motions = collections.Counter(
        (b["root"] - a["root"]) % 12 for a, b in zip(runs, runs[1:]) if a["root"] != b["root"]
    )
    total, sung = sum(spent.values()) or 1.0, sum(played.values()) or 1.0
    homes = [found, relative(found)] + ([chosen] if chosen and chosen not in (found, relative(found)) else [])
    return dict(
        id=desk.song_id,
        title=desk.song.get("title", desk.song_id),
        artist=desk.song.get("artist", ""),
        length=machine["length"],
        tempo=machine["tempo"],
        home=dict(root=home[0], major=home[1], name=key_name(home), yours=chosen is not None),
        homes=[dict(root=h[0], major=h[1], name=key_name(h), found=h == found) for h in homes],
        names=names,
        bars=[
            dict(bar=k + 1, time=round(t, 3), pattern=patterns[k]["pattern"] if k < len(patterns) else None)
            for k, t in enumerate(bars)
        ],
        chords=chords,
        maps=maps,
        regions={does: dict(box=REGIONS[does]) for does in JOBS},
        parts=song_parts,
        loops=_loops(song_parts, bass, len(bars)),
        bass=bass,
        strings=STRINGS,
        places=[dict(said(*key), seconds=round(s, 1), share=round(s / total, 3)) for key, s in spent.most_common()],
        moves=[dict(a=said(*a), b=said(*b), times=n) for (a, b), n in moves.most_common()],
        motions=[dict(up=up, name=MOTIONS[up], times=n) for up, n in motions.most_common()],
        roles={r: round(played[r] / sung, 3) for r in ROLE_ORDER},
        on_root=round(sum(c["role"] == "root" for c in chords) / max(1, sum(c["role"] is not None for c in chords)), 3),
        fine=bool((machine.get("harmony") or {}).get("chords")),  # chords found beat by beat, or one a bar
        audio={name: f"/data/{desk.song_id}/audio/{name}.flac" for name in ("record", "nobass", "bass")},
    )


def _same(chord):
    return chord["root"], chord["quality"]
