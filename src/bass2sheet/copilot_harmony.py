"""A song's harmony, for the copilot: a chord for every bar, the key, the parts the bass line goes round, what
the bass plays in them and where on the neck.

Two pages use it. The song's own page is where what goes on the sheet is changed, the chords with the notes: every
bar has the tool's best guess at its chords (`guessed`) until the person chooses others (`set_bars`); for a chord
the page offers a few to try in its place (`offers`, `suggest`), ranked by the bass notes of the bar, the key and
what the tool hears. The key and the fingering are picked there too, and the neck shows where the notes are
played (`for_notes`). The harmony page shows how the song moves and changes nothing (`view`).

A guess is not agreed to. The pages draw it apart from what the person chose, and a bass note is named by what it
is to its chord only where the chord is the person's; elsewhere it is a step of the key.

Everything is said against a home: the note the song comes back to. The tool guesses a key; the person may pick
another (a song in A minor and one in C major use the same notes, and which is home is heard, not counted).
What they chose is kept beside the song, in harmony.json.

The map of the harmony page is a map of jobs, from functional harmony: a chord rests (home), drifts (away) or
pulls (tension). Where a chord sits says three things: its job (the region), how far outside the key it is (the
row), and how bright it is (left to right, on the line of fifths). The rules are rules of thumb, written down in
`job`, `ring` and `bright`; they are not measured from anything.
"""

import collections
import functools
import json
import math
import os

import numpy as np

from . import copilot, harmony, repeats
from .common import Bass2SheetError, atomic_write_text
from .rhythm import spelling
from .score import STRINGS, TAB_STYLES, effort, frets, hand

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

TONES = harmony.TONES  # the notes of a chord, in semitones above its root, by its kind
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


def _agreed(labels, i, n, size, skip=0):
    """(the loop, the bars that differ from it) of n bars from bar i on that go round `size` bars, the first of
    them being bar `skip` of the loop: the loop is what most rounds have in each of its bars."""
    seen = [collections.Counter() for _ in range(size)]
    for j in range(n):
        seen[(j + skip) % size][labels[i + j]] += 1
    loop = [c.most_common(1)[0][0] if c else None for c in seen]
    return loop, sum(labels[i + j] != loop[(j + skip) % size] for j in range(n))


def parts(labels, home=None):
    """A song as parts, from what each bar holds (`labels`, anything that can be compared and hashed): a part
    is a loop of a few bars going round at least twice, or a passage that does not repeat. [dict(first, bars,
    loop, skip, letter)]: `first` counts bars from 0; `loop` is what one round holds, bar by bar (what most
    rounds have there); `skip` is the bar of the loop the part begins on, for a part that comes in part of the
    way round; parts with the same loop have the same letter.

    Of all the ways to cut the song, the one is taken that puts the most bars into loops: a bar in a loop counts
    one, a bar of a loop that differs from what most rounds have there costs three, a bar of a passage costs
    one, and every loop costs a little, the longer the more, so that the song is not cut finer than it need be
    and four bars twice are not called a loop of eight. A loop may be come into part of the way round (a bar
    that leads in), as it may be left part of the way round. Where a loop could as well begin on one bar as on
    another, whole rounds are taken before cut ones, and then a loop that begins on a bar that holds `home`,
    when that is given: a phrase mostly starts at home."""
    return [dict(part, loop=list(part["loop"])) for part in _parts(tuple(labels), home)]


@functools.lru_cache(maxsize=16)
def _parts(labels, home):
    n = len(labels)
    whole = {}  # (bar, size) -> the bars that go round the loop that starts there
    best, cut = [0.0] * (n + 1), [None] * (n + 1)  # from bar i on: the best that can be had, and how it begins
    for i in range(n - 1, -1, -1):
        best[i], cut[i] = best[i + 1] - 1, (1, 0, 0)  # a bar of a passage
        for size in LOOPS:
            for lead in range(size):  # the bars before the first whole round: the end of a round
                start = i + lead
                if start + size > n:
                    break
                if lead and labels[i:start] != labels[start + size - lead : start + size]:
                    continue
                if (start, size) not in whole:
                    whole[(start, size)] = _round(labels, start, size)
                bars = whole[(start, size)]
                if bars < max(2 * size, 4):  # a loop goes round at least twice and lasts four bars or more
                    continue
                skip = (size - lead) % size
                good = lead + bars - 3 * _agreed(labels, i, lead + bars, size, skip)[1] - 0.5 - 0.01 * size
                good -= 0.2 * ((lead > 0) + (bars % size > 0))  # whole rounds are likelier than cut ones
                good += 0.3 * (home is not None and labels[start] == home) + best[i + lead + bars]
                if good > best[i]:
                    best[i], cut[i] = good, (lead + bars, size, skip)
    out, i, letters = [], 0, {}
    while i < n:
        bars, size, skip = cut[i]
        if size == 0 and out and out[-1]["size"] == 0:
            out[-1]["bars"] += 1
        else:
            out.append(dict(first=i, bars=bars, size=size, skip=skip))
        i += bars
    known = []  # the loops so far
    for part in out:
        size, skip = part.pop("size"), part["skip"]
        loop = tuple(_agreed(labels, part["first"], part["bars"], size or part["bars"], skip)[0])
        if size and loop not in known:
            # An earlier loop taken from another bar on is that loop; and so is one that differs from it in a
            # single bar of four or more (a bar heard otherwise). The nearest first.
            turned = [
                (sum(a != b for a, b in zip(other[s:] + other[:s], loop)), s, other)
                for other in known
                if len(other) == size
                for s in range(size)
            ]
            near = min(turned, default=None, key=lambda found: found[:2])
            if near is not None and near[0] <= (1 if size >= 4 else 0):
                loop, skip = near[2], (skip + near[1]) % size
            else:
                known.append(loop)
        part.update(loop=loop, skip=skip, letter=repeats.letter(letters.setdefault(loop, len(letters))))
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


KEY_STEPS = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")


def step(pitch, home):
    """A bass note as a step of the key: 1 is home, 5 the fifth above it, b3 and b7 the flat third and seventh."""
    return KEY_STEPS[(pitch - home[0]) % 12]


# ---- what the tool hears, which is used for two things only: its guess of the key, and ranking what it offers

_ASKED = set()  # the songs whose harmony was asked for and could not be had: not asked for at every look


def with_harmony(desk):
    """The song's machine data, with its harmony heard now if the song was prepared before there was any
    (copilot.hear_harmony: a few seconds, once)."""
    if "harmony" not in desk.machine and desk.folder not in _ASKED:
        _ASKED.add(desk.folder)
        if copilot.hear_harmony(desk.workdir, desk.song_id) is not None:
            desk.machine = copilot.read_json(os.path.join(desk.folder, "machine.json"))
    return desk.machine


def heard_chords(machine):
    """The chords the tool hears in a song, as stretches of time: [dict(start, end, root, quality, score)].
    Eighth by eighth where the song has its harmony (harmony.find), else the chord a bar it was given long ago.
    Nobody sees these as the song's chords: they rank the chords the page offers (`suggest`)."""
    found = (machine.get("harmony") or {}).get("chords")
    if found:
        return [
            dict(start=c["start"], end=c["end"], root=int(c["root"]), quality=c["quality"], score=float(c["score"]))
            for c in found
        ]
    grid, origin, out = machine["grid"], machine["origin"], []
    for c in machine.get("chords") or []:
        i = origin + 16 * (c["bar"] - 1)
        if 0 <= i < len(grid) - 16 and float(c.get("score", 1.0)) > 0:
            out.append(
                dict(start=grid[i], end=grid[i + 16], root=int(c["root"]), quality=c.get("quality", ""), score=0.5)
            )
    return out


def guessed_keys(machine):
    """The keys the tool would call home, the likeliest first, as (root, major): the key it reads from the
    chords it hears, the key with the same notes, the next best it found, and the key the song was first given."""
    keys = (machine.get("harmony") or {}).get("keys") or []
    out = []
    if keys:
        main = harmony.main_key(machine["harmony"], machine.get("notes") or ())
        out.append((int(main[1]) % 12, bool(main[2])))
        out.append(relative(out[0]))
        out += [
            (int(k["other"]["root"]) % 12, bool(k["other"]["major"])) for k in keys if (k["root"], k["major"]) == out[0]
        ][:1]
    given = (int(machine["key"][1]) % 12, bool(machine["key"][2]))
    out += [given, relative(given)]
    return list(dict.fromkeys(out))


def read_home(folder):
    """The home the person chose for a song, as (root, major), or None."""
    saved = copilot.read_json(os.path.join(folder, "harmony.json")) or {}
    home = saved.get("home")
    return (int(home[0]) % 12, bool(home[1])) if home else None


def set_home(folder, home):
    """Keep the person's choice of home, (root, major); None goes back to the key the tool guesses."""
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


def home_of(folder, machine):
    """(root, major) of a song's home: the one the person chose, else the tool's first guess."""
    return read_home(folder) or guessed_keys(machine)[0]


# ---- a song's chords: the tool's best guess for every bar, and the person's where they chose

BEATS = 4  # the beats of a bar that a chord may start on


def guessed(machine):
    """The tool's best guess at a song's chords: {bar, counted from 0: [(beat, root, quality)]} for the bars it
    hears a chord in. A major or minor triad on beat 1 and, where the second half of the bar has another, one on
    beat 3: what the sheet has always been given of the chords heard eighth by eighth (harmony.for_sheet), which
    is right more often than a chord a bar (see docs/evaluation.md). A song prepared long ago has the chord a
    bar it was given."""
    found = machine.get("harmony") or {}
    if not found.get("chords"):
        old = [c for c in machine.get("chords") or [] if float(c.get("score", 1.0)) > 0]
        return {c["bar"] - 1: [(0, int(c["root"]), c.get("quality", ""))] for c in old if c["bar"] >= 1}
    out = {}
    for bar, symbols in harmony.for_sheet(dict(harmony=found, origin=machine["origin"])).items():
        chords = []
        for at, c in symbols:
            chord = (at // 4, int(c["root"]), harmony.TRIAD[c["quality"]])
            if not chords or chords[-1][1:] != chord[1:]:  # a seventh after its triad is the same chord here
                chords.append(chord)
        if bar >= 1:
            out[bar - 1] = chords
    return out


def _from_marks(marks):
    """The chords a person wrote as marks on beats, when that was how it was done: {bar: [(beat, root,
    quality)]}. A chord lasted from its mark to the next mark of any kind; a mark without a chord is left out."""
    marks, out = sorted(marks, key=lambda m: m["at"]), {}
    for k, m in enumerate(marks):
        if not isinstance(m.get("chord"), list):
            continue
        at, chord = int(m["at"]), (int(m["chord"][0]) % 12, str(m["chord"][1]))
        until = int(marks[k + 1]["at"]) if k + 1 < len(marks) else 16 * (at // 16 + 1)
        out.setdefault(at // 16, []).append((at % 16 // 4, *chord))
        for bar in range(at // 16 + 1, (until - 1) // 16 + 1):
            out[bar] = [(0, *chord)]
    return out


def read_bars(folder):
    """The chords the person chose, by bar: {bar, counted from 0: [(beat, root, quality)]}, the beats counted
    from 0 and in order; an empty list is a bar they said has no chord."""
    saved = copilot.read_json(os.path.join(folder, "harmony.json")) or {}
    if "bars" not in saved:
        return _from_marks(saved.get("marks") or [])
    return {int(bar): [(int(b), int(r) % 12, str(q)) for b, r, q in chords] for bar, chords in saved["bars"].items()}


def set_bars(folder, changes):
    """Keep the person's chords for some bars. `changes` is [dict(bar, chords)]: `bar` counted from 0, `chords`
    the whole of what the bar holds as [[beat, root, quality]] ([] for no chord), or None to go back to the
    tool's guess there."""
    if not (isinstance(changes, list) and changes):
        raise Bass2SheetError("no bar was given")
    bars = read_bars(folder)
    for change in changes:
        bar, chords = (change.get("bar"), change.get("chords")) if isinstance(change, dict) else (None, 0)
        if not (isinstance(bar, int) and not isinstance(bar, bool) and bar >= 0):
            raise Bass2SheetError("a chord is put on a bar of the song")
        if chords is None:
            bars.pop(bar, None)
            continue
        whole = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
        ok = isinstance(chords, list) and all(isinstance(c, (list, tuple)) and len(c) == 3 for c in chords)
        if not (
            ok and all(whole(b) and whole(r) and 0 <= b < BEATS and 0 <= r < 12 and q in TONES for b, r, q in chords)
        ):
            kinds = ", ".join(q or "major" for q in TONES)
            raise Bass2SheetError(f"a chord is a beat of the bar, one of the twelve notes, and one of: {kinds}")
        beats = [b for b, _, _ in chords]
        if beats != sorted(set(beats)):
            raise Bass2SheetError("the chords of a bar start on different beats, in order")
        bars[bar] = [(b, r, q) for b, r, q in chords]
    path = os.path.join(folder, "harmony.json")
    saved = copilot.read_json(path) or {}
    if "marks" in saved:  # what was written as marks is in `bars` now; the marks are kept as they were
        saved["marks_before"] = saved.pop("marks")
    saved["bars"] = {str(bar): [list(c) for c in chords] for bar, chords in sorted(bars.items())}
    atomic_write_text(path, json.dumps(saved))


def bar_chords(folder, machine):
    """What each bar of a song holds, as things stand: [dict(chords, yours)] for every bar, `chords` being
    [(beat, root, quality)]: the person's where they chose (`yours`), else the tool's best guess."""
    n_bars = len(range(machine["origin"], len(machine["grid"]), 16))
    mine, guess = read_bars(folder), guessed(machine)
    return [dict(chords=mine[k] if k in mine else guess.get(k, []), yours=k in mine) for k in range(n_bars)]


def starts_on(notes, start, end):
    """The note a stretch of the bass line starts on: the first to begin in its first eighth (a note may come in
    a touch early), else a note held into it from before for a quarter of it or more. A note of the bar before
    that rings on over the bar line is not it. None when there is neither."""
    span = end - start
    for n in notes:
        if start - 0.08 <= n["start"] < start + min(0.125 * span, 0.3) and n["end"] > start + 0.02:
            return n
    for n in notes:
        if n["start"] < start and n["end"] - start >= 0.25 * span:
            return n
    return None


def main_notes(notes, bars, bar_ends):
    """The note the bass line stands on in each bar, as a pitch class, or None for a bar without bass: the note
    it spends most of the bar on, where the note it starts the bar with counts for half a bar more. A line
    comes back with other fills each round, and still stands on the same notes."""
    out = []
    for t0, t1 in zip(bars, bar_ends):
        held, first = collections.Counter(), starts_on(notes, t0, t1)
        for n in notes:
            both = min(n["end"], t1) - max(n["start"], t0)
            if both > 0.05:  # a note that comes in a hair before the next bar line is not of this bar
                held[n["pitch"] % 12] += both / (t1 - t0)
        if first is not None:
            held[first["pitch"] % 12] += 0.5
        out.append(max(held, key=lambda pc: (held[pc], -pc)) if held else None)
    return out


def song_parts(machine, notes, home):
    """The song cut into parts by its bass line: by the note the line stands on in each bar (main_notes), never
    by chords. See `parts`; a loop starts on the bar that stands on home (`home`, as (root, major)) where it
    could start on several."""
    grid, origin = machine["grid"], machine["origin"]
    bars = [grid[i] for i in range(origin, len(grid), 16)]
    bar_ends = bars[1:] + [max(machine["length"], bars[-1])] if bars else []
    return parts(["-" if pc is None else pc for pc in main_notes(notes, bars, bar_ends)], home[0])


# ---- chords to offer in place of the one a bar has

# What each thing a suggestion is ranked by counts for. Set by hand; how well the ranking does is measured in
# docs/evaluation.md.
OFFER = dict(
    first=2.0,  # the bass plays the chord's root where the stretch starts
    held=1.5,  # the share of the stretch the bass spends on the root
    fifth=0.5,  # ... and on the chord's fifth
    third=1.0,  # ... on its third, against the third of the other kind (major or minor)
    key=0.5,  # how common the chord is in the key
    heard=2.0,  # the share of the stretch in which the tool hears a chord on this root, by how sure it is
    kind=0.5,  # ... and of this kind
    picked=6.0,  # the person put this chord at the same place in another round
)


def suggest(notes, start, end, home, heard=(), picked=(), most=8, weights=None):
    """The chords to offer for a stretch of a song, the likeliest first: [(root, quality)], major and minor
    chords only. They are ranked by the bass notes in the stretch (`notes`: where the stretch starts the bass
    is mostly on the root, and it spends its time on the root, the fifth and the third), by the key (`home`,
    as (root, major): the chords a key is made of come first), by what the tool hears there (`heard`: chords
    as chords_of gives them, each counting for the share of the stretch it fills and how sure the tool is), and
    by what the person put at the same place in other rounds (`picked`: [(root, quality)])."""
    w = dict(OFFER, **(weights or {}))
    span = max(end - start, 1e-6)
    on, first = np.zeros(12), starts_on(notes, start, end)
    first = None if first is None else first["pitch"] % 12
    for n in notes:
        both = min(n["end"], end) - max(n["start"], start)
        if both > 0:
            on[n["pitch"] % 12] += both / span
    if on.sum() > 0:
        on = on / on.sum()
    table = harmony.IN_MAJOR if home[1] else harmony.IN_MINOR
    sure, kinds = np.zeros(12), collections.Counter()
    for c in heard:
        both = (min(c["end"], end) - max(c["start"], start)) / span
        if both > 0:
            sure[c["root"]] += both * c["score"]
            kinds[(c["root"], "m" if minor_like(c["quality"]) else "")] += both * c["score"]
    chosen = collections.Counter((int(root), "m" if minor_like(quality) else "") for root, quality in picked)
    ranked = []
    for root in range(12):
        for quality, third, other in (("", 4, 3), ("m", 3, 4)):
            common = table[quality].get((root - home[0]) % 12, harmony.RARE)
            score = (
                w["first"] * (first == root)
                + w["held"] * on[root]
                + w["fifth"] * on[(root + 7) % 12]
                + w["third"] * (on[(root + third) % 12] - on[(root + other) % 12])
                + w["key"] * math.log(common / 5.0)
                + w["heard"] * sure[root]
                + w["kind"] * kinds[(root, quality)]
                + w["picked"] * chosen[(root, quality)]
            )
            ranked.append((-score, root, quality))
    return [(root, quality) for _, root, quality in sorted(ranked)[:most]]


def _said(home, names):
    """How the pages are told a chord: its name, its step above home and its job, and its notes."""
    return lambda root, quality: dict(
        root=root,
        quality=quality,
        name=names[root] + quality,
        numeral=numeral(root, quality, home),
        job=job(root, quality, home),
        tones=[(root + t) % 12 for t in tones(quality)],
    )


def _spans(machine):
    """(when each bar starts, when it ends, a function from (bar, beat) to when that beat is)."""
    grid, origin = machine["grid"], machine["origin"]
    bars = [grid[i] for i in range(origin, len(grid), 16)]
    bar_ends = bars[1:] + [max(machine["length"], bars[-1])] if bars else []
    beat = lambda k, b: float(grid[min(len(grid) - 1, origin + 16 * k + 4 * b)])  # noqa: E731
    return bars, bar_ends, beat


def _stretches(folder, machine, said):
    """A song's chords as stretches of time in order: [dict(bar, beat, start, end, yours)] with each chord as
    `said` gives it; `bar` and `beat` count from 0, and a bar without a chord has no stretch."""
    bars, bar_ends, when = _spans(machine)
    out = []
    for k, entry in enumerate(bar_chords(folder, machine)):
        for i, (beat, root, quality) in enumerate(entry["chords"]):
            end = bar_ends[k] if i + 1 == len(entry["chords"]) else when(k, entry["chords"][i + 1][0])
            start = max(0.0, when(k, beat))
            out.append(dict(said(root, quality), bar=k, beat=beat, start=round(start, 3), end=round(end, 3)))
            out[-1]["yours"] = entry["yours"]
    return out


def on_the_neck(desk, chords=None):
    """Where each note of a song is played, for the neck of the song's page: {note id: [string, fret, where the
    hand is, what the note is, its part of the chord]}. What a note is: a step of the key, or of the chord it
    sounds under where the person chose that chord (and then its part of the chord: root, third, fifth...)."""
    machine = desk.machine
    home = home_of(desk.folder, machine)
    if chords is None:
        chords = _stretches(desk.folder, machine, _said(home, spelling(*home)))
    notes = sorted(desk.notes, key=lambda n: n["start"])
    placed = _placed(machine, notes, copilot.tab_style(desk.folder))
    bass = _bass(machine, notes, chords, _spans(machine)[0], placed, home)
    said = lambda b: [b["string"], b["fret"], b["hand"], b["degree"] or b["step"], b["role"]]  # noqa: E731
    return {n["id"]: said(b) for n, b in zip(notes, bass)}


def for_notes(desk):
    """What the song's own page needs besides its notes, to show and change what goes on the sheet with them:
    dict(home, homes, spelled, qualities, chords, yours, tab, tabs, neck, strings). `chords` are the song's
    chords as stretches (`_stretches`) and `yours` the bars the person chose them for; `home` is the key and
    `homes` the tool's guesses at it, the likeliest first; `spelled` are the twelve notes as the key names them
    and `qualities` the kinds of chord there are; `tab` is how the notes are put on the neck, `tabs` what each
    way asks of the hand over the whole song, and `neck` where each note is played (`on_the_neck`)."""
    machine = with_harmony(desk)
    guesses, chosen = guessed_keys(machine), read_home(desk.folder)
    home = chosen or guesses[0]
    names = spelling(*home)
    chords = _stretches(desk.folder, machine, _said(home, names))
    ways = fingerings(machine, sorted(desk.notes, key=lambda n: n["start"]))
    return dict(
        home=dict(root=home[0], major=home[1], name=key_name(home), yours=chosen is not None),
        homes=[dict(root=h[0], major=h[1], name=key_name(h), guess=k == 0) for k, h in enumerate(guesses[:3])],
        spelled=names,
        qualities=list(TONES),
        chords=chords,
        yours=sorted(read_bars(desk.folder)),
        tab=copilot.tab_style(desk.folder),
        tabs=[dict(id=style, **effort(ways[style])) for style in TAB_STYLES],
        neck=on_the_neck(desk, chords),
        strings=STRINGS,
    )


def offers(desk, bar, beat=0, most=6):
    """What the page shows for a chord of a bar that the person may change: dict(bar, beat, start, end, chord,
    offers, alike). `bar` and `beat` count from 0 and name a chord of the bar, or the start of a bar that has
    none. `offers` are the chords to try in its place, the likeliest first (see `suggest`), over the stretch the
    chord lasts. `alike` are the other bars where the bass stands on the same note and the same chord is
    written on that beat, where a change made here may be wanted too: `part` those in the parts with this
    part's letter, `song` all of them."""
    machine = with_harmony(desk)
    bars, bar_ends, when = _spans(machine)
    whole = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
    if not (whole(bar) and whole(beat) and 0 <= bar < len(bars) and 0 <= beat < BEATS):
        raise Bass2SheetError("there is no such bar of the song, or no such beat in it")
    home = home_of(desk.folder, machine)
    said = _said(home, spelling(*home))
    per_bar = bar_chords(desk.folder, machine)
    mine = per_bar[bar]["chords"]
    at = next((i for i, c in enumerate(mine) if c[0] == beat), None)
    if at is None and (mine or beat):
        raise Bass2SheetError("no chord of that bar starts on that beat")
    start = when(bar, beat)
    end = bar_ends[bar] if at is None or at + 1 == len(mine) else when(bar, mine[at + 1][0])
    now = None if at is None else mine[at][1:]
    notes = sorted(desk.notes, key=lambda n: n["start"])
    ranked = suggest(notes, start, end, home, heard=heard_chords(machine), most=most + 1)
    stands = main_notes(notes, bars, bar_ends)
    cut = song_parts(machine, notes, home)
    letter = {k: p["letter"] for p in cut for k in range(p["first"], p["first"] + p["bars"])}
    there = lambda k: next((c[1:] for c in per_bar[k]["chords"] if c[0] == beat), None)  # noqa: E731
    alike = [k for k in range(len(bars)) if k != bar and stands[k] == stands[bar] and there(k) == now]
    if at is None:  # a bar without a chord is like the other bars without one
        alike = [k for k in alike if not per_bar[k]["chords"]]
    return dict(
        bar=bar,
        beat=beat,
        start=round(max(0.0, start), 3),
        end=round(end, 3),
        chord=None if now is None else dict(said(*now), yours=per_bar[bar]["yours"]),
        offers=[said(root, quality) for root, quality in ranked if (root, quality) != now][:most],
        alike=dict(part=[k for k in alike if letter.get(k) == letter.get(bar)], song=alike),
    )


def _per_bar(chords, bars, bar_ends):
    """What each bar holds: its chords in order. A chord counts in a bar it fills a fifth of (a beat); one that
    goes on after another of the same name counts once."""
    out = []
    for t0, t1 in zip(bars, bar_ends):
        held = [c for c in chords if min(c["end"], t1) - max(c["start"], t0) > 0.2 * (t1 - t0)]
        out.append([c for k, c in enumerate(held) if k == 0 or _same(c) != _same(held[k - 1])])
    return out


def _under(notes, start, end):
    """The bass note a chord stands on: the one its stretch starts on, else the first that starts in it."""
    return starts_on(notes, start, end) or next((n for n in notes if start <= n["start"] < end), None)


def set_tab(folder, style):
    """Keep how the person wants this song put on the neck (one of score.TAB_STYLES); None goes back to the way
    every song gets (copilot.TAB). The song's score and its play-along page follow (copilot.tab_style)."""
    if style is not None and style not in TAB_STYLES:
        raise Bass2SheetError("a song is put on the neck in one of these ways: " + ", ".join(TAB_STYLES))
    path = os.path.join(folder, "harmony.json")
    saved = copilot.read_json(path) or {}
    if style is None:
        saved.pop("tab", None)
    else:
        saved["tab"] = style
    atomic_write_text(path, json.dumps(saved))


_FINGERED = {}  # the last fingerings worked out, so that a page asked for again does not wait for them


def _placed(machine, notes, style):
    """The notes on the neck in one way (score.TAB_STYLES): [(string, fret)]."""
    grid = np.asarray(machine["grid"])
    when = [int(np.argmin(np.abs(grid - n["start"]))) - machine["origin"] for n in notes]
    pitches, also = [int(n["pitch"]) for n in notes], [tuple(n.get("also", [])) for n in notes]
    key = (style, tuple(pitches), tuple(when), tuple(also))
    if key not in _FINGERED:
        if len(_FINGERED) >= 24:
            _FINGERED.pop(next(iter(_FINGERED)))
        bar_of = [g // 16 for g in when]
        _FINGERED[key] = frets(pitches, style, also=[list(a) for a in also], starts=when, bars=bar_of)
    return _FINGERED[key]


def fingerings(machine, notes):
    """The notes on the neck in each way there is (score.TAB_STYLES): {style: [(string, fret)]}."""
    return {style: _placed(machine, notes, style) for style in TAB_STYLES}


def _bass(machine, notes, chords, bars, played, home):
    """The bass line: each note with where it is played on the neck (`played`), where the fretting hand is
    then, the sixteenth of its bar it starts on, the chord it sounds under, and what it is: a step of the key
    (`step`) and, under a chord the person chose, what it is to that chord (`degree`, `role`)."""
    grid = np.asarray(machine["grid"])
    starts = [c["start"] for c in chords]
    base = {}  # per chord: the pitch of the root the line stands on there
    out = []
    for n, (string, fret), at in zip(notes, played, hand(played)):
        k = int(np.searchsorted(starts, n["start"] + 0.03, side="right")) - 1
        under = chords[k] if k >= 0 and n["start"] + 0.03 < chords[k]["end"] else None
        bar = max(0, int(np.searchsorted(bars, n["start"] + 0.03, side="right")) - 1)
        slot = int(np.argmin(np.abs(grid - n["start"]))) - (machine["origin"] + 16 * bar)
        note = dict(start=round(n["start"], 3), end=round(n["end"], 3), pitch=int(n["pitch"]), string=string, fret=fret)
        note.update(hand=at, bar=bar + 1, slot=max(0, min(15, slot)), step=step(n["pitch"], home))
        note.update(chord=k if under else None, degree=None, role=None)
        if under and under["yours"]:
            root = under["root"]
            if k not in base:
                here = [m["pitch"] for m in notes if under["start"] - 0.03 <= m["start"] < under["end"] - 0.03]
                roots = [q for q in here if (q - root) % 12 == 0]
                base[k] = min(roots) if roots else n["pitch"] - (n["pitch"] - root) % 12
            note.update(degree=degree(n["pitch"], root, base[k]), role=role(n["pitch"], root))
        out.append(note)
    return out


def _loops(shown, bass, n_bars):
    """The distinct parts of a song, for remembering it: each loop once, with how often it comes and what the
    bass plays over each of its bars: the line most rounds have there, and the share of rounds that have it. A
    note is named by the chord it is played over where the bar has chords throughout (`of` is "chord"), else as
    a step of the key ("key")."""
    in_bar = collections.defaultdict(list)
    for note in bass:
        in_bar[note["bar"] - 1].append(note)
    line_of = {}
    for bar, played in in_bar.items():
        by_chord = all(n["degree"] is not None for n in played)
        line_of[bar] = tuple(
            (n["slot"], n["degree"] if by_chord else n["step"], n["role"] if by_chord else None) for n in played
        )
    loops = {}
    for part in shown:
        loop = loops.setdefault(part["letter"], dict(letter=part["letter"], size=part["size"], loop=part["loop"]))
        loop.setdefault("stands", part["stands"])
        loop.setdefault("lines", [collections.Counter() for _ in range(part["size"])])
        loop.update(bars=loop.get("bars", 0) + part["bars"], times=loop.get("times", 0) + 1)
        loop["rounds"] = round(loop.get("rounds", 0) + part["bars"] / part["size"], 2)
        for k in range(part["bars"]):
            bar = part["bar"] - 1 + k
            if bar < n_bars:
                loop["lines"][(k + part["skip"]) % part["size"]][line_of.get(bar, ())] += 1
    for loop in loops.values():
        lines = []
        for seen in loop.pop("lines"):
            line, n = seen.most_common(1)[0] if seen else ((), 0)
            lines.append(
                dict(
                    notes=[dict(slot=s, said=d, role=r) for s, d, r in line],
                    of="chord" if line and all(r is not None for _, _, r in line) else "key",
                    share=round(n / max(1, sum(seen.values())), 2),
                )
            )
        shapes = {tuple((n["slot"], n["said"]) for n in line["notes"]) for line in lines if line["notes"]}
        loop.update(bass=lines, same=len(shapes) == 1 and loop["size"] > 1)
    return sorted(loops.values(), key=lambda loop: -loop["bars"])


def view(desk):
    """Everything the harmony page shows of a song (see the page for how). Times are seconds of the record.

    The page changes nothing of the song: its chords are what each bar holds as things stand (bar_chords), the
    person's where they chose on the song's own page and the tool's best guess elsewhere. The tool cuts the
    song into parts by the notes its bass line stands on and names each bass note as a step of the key; under a
    chord the person chose, a bass note is also named by what it is to that chord."""
    machine = with_harmony(desk)
    guesses = guessed_keys(machine)
    chosen = read_home(desk.folder)
    home = chosen or guesses[0]
    names = spelling(home[0], home[1])
    notes = sorted(desk.notes, key=lambda n: n["start"])
    grid, origin = machine["grid"], machine["origin"]
    bars = [grid[i] for i in range(origin, len(grid), 16)]
    bar_ends = bars[1:] + [max(machine["length"], bars[-1])] if bars else []
    family = [p["pattern"] for p in desk.patterns()["bars"]]
    stands = main_notes(notes, bars, bar_ends)
    said = _said(home, names)
    cut = song_parts(machine, notes, home)
    per_bar, when = bar_chords(desk.folder, machine), _spans(machine)[2]
    chords, spent, played = [], collections.Counter(), collections.Counter()
    for k, entry in enumerate(per_bar):
        for i, (beat, root, quality) in enumerate(entry["chords"]):
            start = max(0.0, when(k, beat))
            end = bar_ends[k] if i + 1 == len(entry["chords"]) else when(k, entry["chords"][i + 1][0])
            if end - start <= 0.05:
                continue
            on = _under(notes, start, end)
            if entry["yours"]:  # how long the bass spends on each part of a chord the person chose
                for n in notes:
                    both = min(n["end"], end) - max(n["start"], start)
                    if both > 0:
                        played[role(n["pitch"], root)] += both
            spent[(root, quality)] += end - start
            chord = dict(said(root, quality), bar=k + 1, beat=beat + 1, start=round(start, 3), end=round(end, 3))
            chord.update(yours=entry["yours"], ring=ring(root, quality, home), far=far(root, quality, home))
            chord.update(
                bass=None if on is None else int(on["pitch"]) % 12,
                role=None if on is None else role(on["pitch"], root),
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
    in_bar = _per_bar(chords, bars, bar_ends)
    held = [tuple(_same(c) for c in bar) for bar in in_bar]
    shown = collections.defaultdict(collections.Counter)  # what most rounds of a loop hold in each of its bars
    for part in cut:
        for k in range(part["bars"]):
            if held[part["first"] + k]:
                shown[(part["letter"], (k + part["skip"]) % len(part["loop"]))][held[part["first"] + k]] += 1
    the_parts = [
        dict(
            letter=part["letter"],
            bar=part["first"] + 1,
            bars=part["bars"],
            size=len(part["loop"]),
            skip=part["skip"],
            start=round(max(0.0, bars[part["first"]]), 3),
            end=round(bar_ends[part["first"] + part["bars"] - 1], 3),
            loop=[
                [said(*chord) for chord in (shown[(part["letter"], at)].most_common(1) or [((), 1)])[0][0]]
                for at in range(len(part["loop"]))
            ],
            stands=[None if pc == "-" else names[pc] for pc in part["loop"]],  # the note the bass stands on, bar by bar
        )
        for part in cut
    ]
    tab, ways = copilot.tab_style(desk.folder), fingerings(machine, notes)
    bass = _bass(machine, notes, chords, bars, ways[tab], home)
    runs = [c for k, c in enumerate(chords) if k == 0 or _same(c) != _same(chords[k - 1])]
    moves = collections.Counter((_same(a), _same(b)) for a, b in zip(runs, runs[1:]))
    motions = collections.Counter(
        (b["root"] - a["root"]) % 12 for a, b in zip(runs, runs[1:]) if a["root"] != b["root"]
    )
    on_steps = collections.Counter()
    for n in notes:
        on_steps[step(n["pitch"], home)] += n["end"] - n["start"]
    total, sung, long = sum(spent.values()) or 1.0, sum(played.values()) or 1.0, sum(on_steps.values()) or 1.0
    mine = [c for c in chords if c["yours"]]
    return dict(
        id=desk.song_id,
        title=desk.song.get("title", desk.song_id),
        artist=desk.song.get("artist", ""),
        length=machine["length"],
        tempo=machine["tempo"],
        home=dict(root=home[0], major=home[1], name=key_name(home), yours=chosen is not None),
        homes=[
            dict(root=h[0], major=h[1], name=key_name(h), guess=k == 0)
            for k, h in enumerate(guesses[:3] + ([chosen] if chosen and chosen not in guesses[:3] else []))
        ],
        names=names,
        bars=[
            dict(
                bar=k + 1,
                time=round(t, 3),
                end=round(bar_ends[k], 3),
                beats=[round(float(grid[min(len(grid) - 1, origin + 16 * k + 4 * b)]), 3) for b in range(4)],
                pattern=family[k] if k < len(family) else None,  # the bars that sound the same in the bass track
                stands_on=None if stands[k] is None else names[stands[k]],
                holds=[said(*chord) for chord in held[k]],
                chords=[dict(said(root, quality), beat=beat + 1) for beat, root, quality in per_bar[k]["chords"]],
                yours=per_bar[k]["yours"],
            )
            for k, t in enumerate(bars)
        ],
        chords=chords,
        yours=sum(entry["yours"] for entry in per_bar),  # the bars whose chords the person chose
        guessed=sum(bool(entry["chords"]) and not entry["yours"] for entry in per_bar),  # and those left to the tool
        covered=round(total / max(machine["length"], 1e-9), 3) if chords else 0.0,  # the share of the song with a chord
        qualities=list(TONES),
        maps=maps,
        regions={does: dict(box=REGIONS[does]) for does in JOBS},
        parts=the_parts,
        loops=_loops(the_parts, bass, len(bars)),
        bass=bass,
        strings=STRINGS,
        tab=tab,  # how the notes are put on the neck, and what each way asks of the hand over the whole song
        tabs=[dict(id=style, **effort(ways[style])) for style in TAB_STYLES],
        places=[dict(said(*key), seconds=round(s, 1), share=round(s / total, 3)) for key, s in spent.most_common()],
        moves=[dict(a=said(*a), b=said(*b), times=n) for (a, b), n in moves.most_common()],
        motions=[dict(up=up, name=MOTIONS[up], times=n) for up, n in motions.most_common()],
        roles={r: round(played[r] / sung, 3) for r in ROLE_ORDER} if mine else None,  # under the person's chords
        steps=[dict(step=name, share=round(t / long, 3)) for name, t in on_steps.most_common()],
        on_root=round(sum(c["role"] == "root" for c in mine) / max(1, sum(c["role"] is not None for c in mine)), 3),
        audio={name: f"/data/{desk.song_id}/audio/{name}.flac" for name in ("record", "nobass", "bass")},
    )


def for_sheet(folder, machine):
    """The chords a song's score writes over its bars, as things stand: the person's where they chose, the
    tool's best guess elsewhere (bar_chords). As harmony.for_sheet gives them: {bar, counted from 1: [(sixteenth
    of the bar, dict(root, quality, bass, names))]}."""
    home = home_of(folder, machine)
    names = spelling(*home)
    out = {}
    for k, entry in enumerate(bar_chords(folder, machine)):
        for beat, root, quality in entry["chords"]:
            spelled = list(names)
            spelled[root] = harmony.note_name(root, names[home[0]], names)
            for tone in tones(quality)[1:]:
                spelled[(root + tone) % 12] = harmony.note_name((root + tone) % 12, spelled[root], names)
            out.setdefault(k + 1, []).append((4 * beat, dict(root=root, quality=quality, bass=None, names=spelled)))
    return out


def chosen_share(folder, machine):
    """(the bars whose chords the person chose, the bars that have chords at all)."""
    per_bar = bar_chords(folder, machine)
    return sum(e["yours"] for e in per_bar), sum(bool(e["chords"]) for e in per_bar)


def _same(chord):
    return chord["root"], chord["quality"]
