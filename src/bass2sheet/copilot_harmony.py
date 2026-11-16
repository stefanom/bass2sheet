"""A song's harmony, for the copilot's harmony page: the parts its bass line goes round, what the bass plays in
them, and the chords the person writes over it, named against a home key with the job each does.

The page is for understanding a song before playing it, and for remembering it. The tool writes no chord: a
bass player needs the notes and how the song is built, and the tool's guesses at the chords are wrong too often
to stand as the song's. The person says where a chord starts (a mark on a beat) and what it is; the tool cuts the
song into parts from its bass line, names each bass note as a step of the key, and offers a few chords to try on
a beat (`suggest`), ranked by the bass notes, the key, what it hears itself and what the person put in the other
rounds. Their chords, their home and their fingering are kept beside the song, in harmony.json.

Everything is said against a home: the note the song comes back to. The tool guesses a key; the person may pick
another (a song in A minor and one in C major use the same notes, and which is home is heard, not counted).

The map of the page is a map of jobs, from functional harmony: a chord rests (home), drifts (away) or pulls
(tension). Where a chord sits says three things: its job (the region), how far outside the key it is (the row),
and how bright it is (left to right, on the line of fifths). The rules are rules of thumb, written down in `job`,
`ring` and `bright`; they are not measured from anything.
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


# ---- the person's chords: marks on beats

NO_CHORD = "none"  # what a mark holds where the person says: no chord from here
REMOVE = "remove"  # what a change says to take a mark away


def read_marks(folder):
    """The chord marks the person made in a song, in order: [dict(at, chord)]. `at` counts sixteenths from the
    start of bar 1 and falls on a beat. `chord` is (root, quality); or None, where a change of chord is marked
    and the chord is still to be chosen; or "none", where the person says there is no chord from here."""
    saved = copilot.read_json(os.path.join(folder, "harmony.json")) or {}
    marks = []
    for m in saved.get("marks") or []:
        chord = m.get("chord")
        marks.append(
            dict(at=int(m["at"]), chord=(int(chord[0]) % 12, str(chord[1])) if isinstance(chord, list) else chord)
        )
    return sorted(marks, key=lambda m: m["at"])


def set_marks(folder, changes):
    """Make, change or take away chord marks. `changes` is [dict(at, chord)]: `chord` as read_marks gives it
    (a chord as [root, quality]), or "remove" to take the mark at `at` away."""
    if not (isinstance(changes, list) and changes):
        raise Bass2SheetError("no chord mark was given")
    marks = {m["at"]: m["chord"] for m in read_marks(folder)}
    for change in changes:
        at, chord = (change.get("at"), change.get("chord")) if isinstance(change, dict) else (None, None)
        if not (isinstance(at, int) and not isinstance(at, bool) and at >= 0 and at % 4 == 0):
            raise Bass2SheetError("a chord starts on a beat of a bar")
        if isinstance(chord, (list, tuple)):
            ok = len(chord) == 2 and isinstance(chord[0], int) and not isinstance(chord[0], bool)
            if not (ok and 0 <= chord[0] < 12 and chord[1] in TONES):
                raise Bass2SheetError(
                    "a chord is one of the twelve notes with one of: " + ", ".join(q or "major" for q in TONES)
                )
            marks[at] = (chord[0], chord[1])
        elif chord == REMOVE:
            marks.pop(at, None)
        elif chord is None or chord == NO_CHORD:
            marks[at] = chord
        else:
            raise Bass2SheetError("a mark holds a chord, no chord, or a chord still to choose")
    path = os.path.join(folder, "harmony.json")
    saved = copilot.read_json(path) or {}
    saved["marks"] = [dict(at=at, chord=list(c) if isinstance(c, tuple) else c) for at, c in sorted(marks.items())]
    atomic_write_text(path, json.dumps(saved))


def stretches(machine, marks, part_ends):
    """What the marks say, as stretches of the song in order: [dict(at, until, start, end, chord)]. A mark's
    chord lasts until the next mark, or the end of the part it is in (`part_ends`: where each part ends, in
    sixteenths from the start of bar 1), or the end of the song. Times are seconds; a mark beyond the end of the
    song is left out."""
    grid, origin = machine["grid"], machine["origin"]
    total = len(grid) - 1 - origin
    time = lambda g: float(grid[origin + g])  # noqa: E731
    out = []
    for k, m in enumerate(marks):
        if m["at"] >= total:
            continue
        nxt = [marks[k + 1]["at"]] if k + 1 < len(marks) else []
        until = min([total] + nxt + [e for e in part_ends if e > m["at"]][:1])
        end = min(time(until), max(machine["length"], time(m["at"])))
        out.append(
            dict(at=m["at"], until=until, start=round(max(0.0, time(m["at"])), 3), end=round(end, 3), chord=m["chord"])
        )
    return out


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


def place_of(cut, at):
    """Where a sixteenth is in the song's form: (the part it is in, the bar of the part's loop, the round), or
    None beyond the last part."""
    bar = at // 16
    for part in cut:
        if part["first"] <= bar < part["first"] + part["bars"]:
            k = bar - part["first"] + part["skip"]
            return part, k % len(part["loop"]), k // len(part["loop"])
    return None


def same_place(cut, at):
    """The same beat of the same bar of the loop, in every other round of the parts with this part's letter:
    the sixteenths, in order."""
    here = place_of(cut, at)
    if here is None:
        return []
    letter, pos, out = here[0]["letter"], here[1], []
    for part in cut:
        if part["letter"] == letter:
            for k in range(part["bars"]):
                there = 16 * (part["first"] + k) + at % 16
                if (k + part["skip"]) % len(part["loop"]) == pos and there != at:
                    out.append(there)
    return out


def sounding(spans, at):
    """The chord the person has at a sixteenth, (root, quality), or None."""
    for span in spans:
        if span["at"] <= at < span["until"]:
            return span["chord"] if isinstance(span["chord"], tuple) else None
    return None


def copy_round(desk, at):
    """Write the chords of the round that holds sixteenth `at` into the other rounds of the parts with the same
    letter, where a round has no mark of its own yet: a round the person wrote is never written over. Returns
    how many rounds were filled."""
    machine = desk.machine
    notes = sorted(desk.notes, key=lambda n: n["start"])
    cut, marks = song_parts(machine, notes, home_of(desk.folder, machine)), read_marks(desk.folder)
    here = place_of(cut, at)
    if here is None:
        raise Bass2SheetError("there is no part of the song there")
    part, size = here[0], len(here[0]["loop"])

    def rounds(p):  # the rounds of a part as (first sixteenth, one past the last), cut to the part
        first, last = 16 * p["first"], 16 * (p["first"] + p["bars"])
        k = first - 16 * p["skip"]
        while k < last:
            yield max(k, first), min(k + 16 * size, last), k
            k += 16 * size

    mine = next(r for r in rounds(part) if r[0] <= at < r[1])
    spans = stretches(machine, marks, [16 * (p["first"] + p["bars"]) for p in cut])
    source = [(m["at"] - mine[2], m["chord"]) for m in marks if mine[0] <= m["at"] < mine[1]]
    carried = sounding(spans, mine[0])  # a chord that comes into the round from before it starts the copy too
    if carried and not any(offset == mine[0] - mine[2] for offset, _ in source):
        source.insert(0, (mine[0] - mine[2], carried))
    if not source:
        raise Bass2SheetError("this round has no chords to copy yet")
    changes, filled = [], 0
    for p in cut:
        if p["letter"] != part["letter"]:
            continue
        for first, last, zero in rounds(p):
            if (first, last) == mine[:2] or any(first <= m["at"] < last for m in marks):
                continue
            new = [dict(at=zero + offset, chord=chord) for offset, chord in source if first <= zero + offset < last]
            before = [chord for offset, chord in source if zero + offset < first]
            if before and not any(c["at"] == first for c in new):  # a round come into part of the way: the chord
                new.insert(0, dict(at=first, chord=before[-1]))  # that sounds there in the round copied starts it
            changes += new
            filled += bool(new)
    if changes:
        set_marks(
            desk.folder,
            [dict(c, chord=list(c["chord"]) if isinstance(c["chord"], tuple) else c["chord"]) for c in changes],
        )
    return filled


# ---- chords to offer: the person writes a song's chords, and the tool says which are likely

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


def offers(desk, at, most=6):
    """What the page shows for a beat the person is putting a chord on: dict(offers, rounds). `offers` are the
    chords to try, the likeliest first (see `suggest`), over the stretch from the beat to the next mark, a bar
    at most. `rounds` are the same place in the other rounds of the part, each with the chord the person has
    there now: where the chord chosen here may be wanted too."""
    machine = with_harmony(desk)
    grid, origin = machine["grid"], machine["origin"]
    n_bars = max(0, (len(grid) - 1 - origin) // 16)
    if not (isinstance(at, int) and 0 <= at < 16 * n_bars and at % 4 == 0):
        raise Bass2SheetError("a chord starts on a beat of a bar of the song")
    home = home_of(desk.folder, machine)
    names = spelling(*home)
    notes = sorted(desk.notes, key=lambda n: n["start"])
    cut, marks = song_parts(machine, notes, home), read_marks(desk.folder)
    spans = stretches(machine, marks, [16 * (p["first"] + p["bars"]) for p in cut])
    later = [m["at"] for m in marks if m["at"] > at] + [
        16 * (p["first"] + p["bars"]) for p in cut if 16 * (p["first"] + p["bars"]) > at
    ]
    until = min([at + 16, 16 * n_bars] + later)
    there = [(g, sounding(spans, g)) for g in same_place(cut, at)]
    ranked = suggest(
        notes,
        float(grid[origin + at]),
        float(grid[origin + until]),
        home,
        heard=heard_chords(machine),
        picked=[chord for _, chord in there if chord],
        most=most,
    )
    said = lambda root, quality: dict(  # noqa: E731
        root=root,
        quality=quality,
        name=names[root] + quality,
        numeral=numeral(root, quality, home),
        job=job(root, quality, home),
    )
    yours = {(root, "m" if minor_like(quality) else "") for _, chord in there if chord for root, quality in [chord]}
    return dict(
        at=at,
        offers=[dict(said(root, quality), yours=(root, quality) in yours) for root, quality in ranked],
        rounds=[dict(at=g, bar=g // 16 + 1, chord=None if chord is None else said(*chord)) for g, chord in there],
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


def fingerings(machine, notes):
    """The notes on the neck in each way there is (score.TAB_STYLES): {style: [(string, fret)]}."""
    grid = np.asarray(machine["grid"])
    when = [int(np.argmin(np.abs(grid - n["start"]))) - machine["origin"] for n in notes]
    pitches, also = [int(n["pitch"]) for n in notes], [tuple(n.get("also", [])) for n in notes]
    key = (tuple(pitches), tuple(when), tuple(also))
    if key not in _FINGERED:
        if len(_FINGERED) >= 8:
            _FINGERED.pop(next(iter(_FINGERED)))
        bar_of = [g // 16 for g in when]
        _FINGERED[key] = {
            style: frets(pitches, style, also=[list(a) for a in also], starts=when, bars=bar_of) for style in TAB_STYLES
        }
    return _FINGERED[key]


def _bass(machine, notes, chords, bars, played, home):
    """The bass line: each note with where it is played on the neck (`played`), where the fretting hand is
    then, the sixteenth of its bar it starts on, and what it is: a step of the key (`step`) and, where the
    person wrote a chord, what it is to that chord (`degree`, `role`)."""
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
        if under:
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

    The song's chords are the person's: the marks they made (read_marks). The tool cuts the song into parts by
    the notes its bass line stands on, names each bass note as a step of the key, guesses the key, and offers
    chords to choose from (`offers`); it writes no chord itself."""
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
    said = lambda root, quality: dict(  # noqa: E731
        root=root,
        quality=quality,
        name=names[root] + quality,
        numeral=numeral(root, quality, home),
        job=job(root, quality, home),
    )
    cut = song_parts(machine, notes, home)
    marks = read_marks(desk.folder)
    spans = stretches(machine, marks, [16 * (p["first"] + p["bars"]) for p in cut])
    chords, spent, played = [], collections.Counter(), collections.Counter()
    for span in spans:
        if not isinstance(span["chord"], tuple) or span["end"] - span["start"] <= 0.05:
            continue
        root, quality = span["chord"]
        on = _under(notes, span["start"], span["end"])
        for n in notes:  # how long the bass spends on each part of the chord
            both = min(n["end"], span["end"]) - max(n["start"], span["start"])
            if both > 0:
                played[role(n["pitch"], root)] += both
        spent[(root, quality)] += span["end"] - span["start"]
        chord = dict(said(root, quality), at=span["at"], until=span["until"], start=span["start"], end=span["end"])
        chord.update(ring=ring(root, quality, home), far=far(root, quality, home))
        chord.update(tones=[(root + t) % 12 for t in tones(quality)], bar=span["at"] // 16 + 1)
        chord.update(
            bass=None if on is None else int(on["pitch"]) % 12, role=None if on is None else role(on["pitch"], root)
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
    carried = [sounding(spans, 16 * k) for k in range(len(bars))]  # the chord a bar starts in
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
                comes_in=None if carried[k] is None else said(*carried[k]),
            )
            for k, t in enumerate(bars)
        ],
        marks=[
            dict(
                at=span["at"],
                until=span["until"],
                bar=span["at"] // 16 + 1,
                beat=span["at"] % 16 // 4 + 1,
                start=span["start"],
                end=span["end"],
                chord=said(*span["chord"]) if isinstance(span["chord"], tuple) else None,
                open=span["chord"] is None,  # a change is marked and its chord is still to be chosen
            )
            for span in spans
        ],
        chords=chords,
        written=len(chords),
        to_choose=sum(span["chord"] is None for span in spans),
        covered=round(total / max(machine["length"], 1e-9), 3) if chords else 0.0,  # the share of the song with a chord
        qualities=list(TONES),
        palette={  # every chord the page may put on a beat, with its notes
            f"{root},{quality}": dict(said(root, quality), tones=[(root + t) % 12 for t in tones(quality)])
            for root in range(12)
            for quality in TONES
        },
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
        roles={r: round(played[r] / sung, 3) for r in ROLE_ORDER} if chords else None,
        steps=[dict(step=name, share=round(t / long, 3)) for name, t in on_steps.most_common()],
        on_root=round(sum(c["role"] == "root" for c in chords) / max(1, sum(c["role"] is not None for c in chords)), 3),
        audio={name: f"/data/{desk.song_id}/audio/{name}.flac" for name in ("record", "nobass", "bass")},
    )


def for_sheet(folder, machine, notes):
    """The chords a song's score writes over its bars: the person's, as long as the harmony page shows each
    (to the next mark or the end of its part, which is cut from `notes`), and none where they wrote none. As
    harmony.for_sheet gives them: {bar: [(sixteenth of the bar, dict(root, quality, bass, names))]}, a chord
    where it starts and again at the start of every bar it goes on in."""
    home = home_of(folder, machine)
    names = spelling(*home)
    n_bars = max(0, (len(machine["grid"]) - 1 - machine["origin"]) // 16)
    marks, out = read_marks(folder), {}
    cut = song_parts(machine, sorted(notes, key=lambda n: n["start"]), home) if marks else []
    for span in stretches(machine, marks, [16 * (p["first"] + p["bars"]) for p in cut]):
        if isinstance(span["chord"], tuple):
            root, quality = span["chord"]
            spelled = list(names)
            spelled[root] = harmony.note_name(root, names[home[0]], names)
            for tone in tones(quality)[1:]:
                spelled[(root + tone) % 12] = harmony.note_name((root + tone) % 12, spelled[root], names)
            chord = dict(root=root, quality=quality, bass=None, names=spelled)
            for g in [span["at"]] + list(range(16 * (span["at"] // 16 + 1), min(span["until"], 16 * n_bars), 16)):
                out.setdefault(g // 16 + 1, []).append((g % 16, chord))
    return out


def _same(chord):
    return chord["root"], chord["quality"]
