"""A song's harmony, for the copilot's harmony page: the chords in time, named against a home key, how they
follow each other, and what the bass plays under each.

The page is for understanding a song before playing it. So everything here is said against a home: the note the
song comes back to. The tool finds a key; the person may pick another (a song in A minor and one in C major use
the same notes, and which is home is heard, not counted). Their choice is kept beside the song, in harmony.json.
"""

import collections
import json
import os

from . import copilot
from .common import Bass2SheetError, atomic_write_text
from .rhythm import spelling

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


def view(desk):
    """Everything the harmony page shows of a song (see the page for how). Times are seconds of the record."""
    machine = desk.machine
    found = (int(machine["key"][1]) % 12, bool(machine["key"][2]))
    chosen = read_home(desk.folder)
    home = chosen or found
    names = spelling(home[0], home[1])
    notes = sorted(desk.notes, key=lambda n: n["start"])
    bars = [machine["grid"][i] for i in range(machine["origin"], len(machine["grid"]), 16)]
    patterns = desk.patterns()["bars"]
    chords, spent, played = [], collections.Counter(), collections.Counter()
    for c in chords_of(machine):
        on = _under(notes, c["start"], c["end"])
        bar = max(1, sum(t <= c["start"] + 0.01 for t in bars))
        for n in notes:  # how long the bass spends on each part of the chord
            both = min(n["end"], c["end"]) - max(n["start"], c["start"])
            if both > 0:
                played[role(n["pitch"], c["root"])] += both
        spent[(c["root"], c["quality"])] += c["end"] - c["start"]
        chords.append(
            dict(
                c,
                bar=bar,
                name=names[c["root"]] + c["quality"],
                numeral=numeral(c["root"], c["quality"], home),
                bass=None if on is None else int(on["pitch"]) % 12,
                role=None if on is None else role(on["pitch"], c["root"]),
            )
        )
    runs = [c for k, c in enumerate(chords) if k == 0 or (c["root"], c["quality"]) != _same(chords[k - 1])]
    moves = collections.Counter((_same(a), _same(b)) for a, b in zip(runs, runs[1:]))
    motions = collections.Counter(
        (b["root"] - a["root"]) % 12 for a, b in zip(runs, runs[1:]) if a["root"] != b["root"]
    )
    total, sung = sum(spent.values()) or 1.0, sum(played.values()) or 1.0
    where = lambda key: dict(root=key[0], quality=key[1], name=names[key[0]] + key[1], numeral=numeral(*key, home))
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
        places=[dict(where(key), seconds=round(s, 1), share=round(s / total, 3)) for key, s in spent.most_common()],
        moves=[dict(a=where(a), b=where(b), times=n) for (a, b), n in moves.most_common()],
        motions=[dict(up=up, name=MOTIONS[up], times=n) for up, n in motions.most_common()],
        roles={r: round(played[r] / sung, 3) for r in ROLE_ORDER},
        fine=bool((machine.get("harmony") or {}).get("chords")),  # chords found beat by beat, or one a bar
        audio={name: f"/data/{desk.song_id}/audio/{name}.flac" for name in ("record", "nobass", "bass")},
    )


def _same(chord):
    return chord["root"], chord["quality"]
