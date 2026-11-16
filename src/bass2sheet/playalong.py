"""Play along: the song without its bass, and the score moving with it, on a phone.

The copilot (copilot.py) is where a person makes a song's score their own. This is where they play it. The page
(`playalong_page.html`, served at `/play` by the copilot's server) plays the record with the bass taken out, and
shows the score written from the person's notes one line at a time: the line being played large at the top, the
next one under it, the note to play highlighted.

Three things are made for it, each on first use and kept in the song's folder:

- **The song without the bass** (`audio/nobass.flac`): the other separated stems summed, at the level of
  `record.flac` and `bass.flac`. A song prepared from now on gets it with the other two (`write_backing`); for a
  song prepared before, it is made from the stems in the cache (`backing`), or, when they are gone, by taking the
  isolated bass out of the record.
- **The score as pictures, one per line** (`play/b<bars>/`): MuseScore engraves the person's notes with every
  line on a page of its own and says where it put every measure and every note (its "positions" of measures and
  segments). The pictures are cut to the lines (`engrave`, `lines_of`).
- **When each of those is played**: a bar or a note is written on a sixteenth of the song's own beat grid, and
  that sixteenth's time in the recording is its time (`grid_time`). The highlight follows the record's tempo as
  it drifts, not a metronome.

Nothing here changes a song: the page sends no fix, and the server refuses any that does not come from the
machine it runs on (`from_this_machine`), so it can listen on the home network (`serve --lan`).
"""

import collections
import functools
import hashlib
import hmac
import http.cookies
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import soundfile as sf

from . import copilot, score
from .common import DEFAULTS, SR, Bass2SheetError, atomic_write_text, cache_home, sha1_file

PAGE_PATH = os.path.join(os.path.dirname(__file__), "playalong_page.html")
SOUNDS = ("nobass", "record", "bass")  # what can be heard: the song without the bass, all of it, the bass alone
BARS = (2, 3, 4)  # the most bars to a line that can be asked for; a busy song gets fewer (see `lines`)
DPI = 360  # of the pictures: a line is about 2200 points wide, enough for a phone's screen at its full sharpness
SPATIUM = 1.75 / 25.4  # the distance between two staff lines, in inches (MuseScore's own)
LINE = 80.0  # staff spaces across a line: on a phone held sideways the line being played then fills the width
PAGE_HEIGHT = 48.0  # staff spaces: room for one line of staff and tab with the highest notes a bass has above it
ENGRAVING = "lines 2"  # with another way of engraving, the pictures kept from this one are made again
STYLE = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.70"><Style>
<pageWidth>{width:.4f}</pageWidth><pageHeight>{height:.4f}</pageHeight>
<pagePrintableWidth>{printable:.4f}</pagePrintableWidth>
<pageEvenLeftMargin>0.1</pageEvenLeftMargin><pageOddLeftMargin>0.1</pageOddLeftMargin>
<pageEvenTopMargin>0.05</pageEvenTopMargin><pageEvenBottomMargin>0.05</pageEvenBottomMargin>
<pageOddTopMargin>0.05</pageOddTopMargin><pageOddBottomMargin>0.05</pageOddBottomMargin><pageTwosided>0</pageTwosided>
<createMultiMeasureRests>0</createMultiMeasureRests><enableVerticalSpread>0</enableVerticalSpread>
<akkoladeDistance>4.5</akkoladeDistance>
<showMeasureNumberOne>1</showMeasureNumberOne><measureNumberSystem>1</measureNumberSystem>
<showHeader>0</showHeader><showFooter>0</showFooter>
</Style></museScore>
"""
_TIME_MARK = re.compile(
    r'<direction placement="above"><direction-type><words font-size="7" font-style="italic">[^<]*</words>'
    r"</direction-type><staff>1</staff></direction>"
)
_HEAD = re.compile(r"<movement-title>.*?</identification>", re.S)

_making = collections.defaultdict(threading.Lock)  # one maker at a time for each file or folder that is made


# --------------------------------------------------------------------------------------- the song without the bass


def _fitted(target, part):
    """(how many times `part` is in `target`, by least squares; the share of `target` that leaves unexplained)."""
    power, whole = float(np.sum(part * part)), float(np.sum(target * target))
    if power <= 0 or whole <= 0:
        return 0.0, 1.0
    gain = float(np.sum(target * part)) / power
    left = target - gain * part
    return gain, float(np.sum(left * left)) / whole


def write_backing(folder, rest, scale, mix=None, bass=None):
    """Write `nobass.flac` in a song's audio folder: `rest`, everything but the bass as (2, n) or (n,), times
    `scale`, what the record beside it was multiplied by.

    With the record (`mix`) and the isolated bass as the separator gave them, the rest is first brought to the
    level it has in the record. A separator turns a loud record down before it takes it apart, so its stems can
    add up to a quieter record than the one that plays; fitted back, switching between the record and the record
    without the bass changes only the bass."""
    y = np.atleast_2d(np.asarray(rest, dtype=np.float64))
    if mix is not None and bass is not None:
        gain, _ = _fitted(np.atleast_2d(np.asarray(mix, dtype=np.float64)), y + np.atleast_2d(bass))
        y = y * (gain if gain > 0 else 1.0)
    y = y * scale
    y = y / max(1.0, float(np.abs(y).max()) / 0.99) if y.size else y
    out = os.path.join(folder, "nobass.flac")
    sf.write(out + ".part", y.T, SR, subtype="PCM_16", format="FLAC")
    os.replace(out + ".part", out)
    return out


def _fit(a, n):
    a = np.atleast_2d(a)[:, :n]
    return np.pad(a, ((0, 0), (0, n - a.shape[1]))) if a.shape[1] < n else a


def _kept_stems(workdir, song, n):
    """The stems a song was prepared from, if they can still be had without separating again: the ones the
    person gave, or the separator's in the cache. (everything but the bass, the bass), each (2, n), or None."""
    from .separate import load_stems

    opts = copilot.read_json(os.path.join(workdir, "settings.json"), {})
    settings = dict(DEFAULTS, **(opts.get("defaults") or {}), **(song.get("settings") or {}))
    stems = None
    if song.get("stems_dir"):
        if os.path.isdir(song["stems_dir"]):
            stems = load_stems(song["stems_dir"], n)
    elif song.get("path") and os.path.isfile(song["path"]):
        cache = cache_home(opts.get("cache_dir"))
        kept = os.path.join(cache, f"stems-{sha1_file(song['path'])[:16]}-{settings['separator']}.npz")
        if os.path.exists(kept):
            with np.load(kept) as z:
                stems = {name: z[name] for name in z.files}
    if not stems or "bass" not in stems or len(stems) < 2:
        return None
    rest = sum(_fit(s, n).astype(np.float64) for name, s in stems.items() if name != "bass")
    return rest, _fit(stems["bass"], n).astype(np.float64)


def backing(workdir, song_id):
    """The path of a song's record without its bass, made if it is not there yet (or is older than the record).

    From the stems the song was prepared from, where they are still to be had: the same sum a newly prepared
    song gets, at the level that makes the stems add up to `record.flac`. Without them, or if they no longer add
    up to this record, the isolated bass is taken out of the record instead, as much of it as the record holds:
    as good where the separator's stems add up to the record, a little of the bass left in where they do not."""
    folder = os.path.join(copilot.song_dir(workdir, song_id), "audio")
    record, bass, out = (os.path.join(folder, name + ".flac") for name in ("record", "bass", "nobass"))
    if not (os.path.exists(record) and os.path.exists(bass)):
        raise Bass2SheetError(f"{song_id} is not prepared")
    with _making[out]:
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(record):
            return out
        whole = sf.read(record, dtype="float32", always_2d=True)[0].T.astype(np.float64)
        n = whole.shape[1]
        song = copilot.load_songs(workdir).get(song_id, {})
        try:
            kept = _kept_stems(workdir, song, n)
        except (Bass2SheetError, OSError, ValueError, KeyError):
            kept = None
        if kept is not None:
            gain, unexplained = _fitted(whole, kept[0] + kept[1])
            if gain > 0 and unexplained < 0.05:  # they are this record's stems
                return write_backing(folder, kept[0], gain)
        mono = _fit(sf.read(bass, dtype="float32", always_2d=True)[0].mean(1), n)[0].astype(np.float64)
        rest = np.stack([channel - max(0.0, _fitted(channel, mono)[0]) * mono for channel in whole])
        return write_backing(folder, rest, 1.0)


def sound(workdir, song_id, name, percent=100):
    """The path of what is played: `name` of SOUNDS at `percent` of the song's speed, made if need be."""
    if name not in SOUNDS or (percent != 100 and percent not in copilot.SPEEDS):
        raise Bass2SheetError(f"no {name} at {percent}%")
    if name == "nobass":
        backing(workdir, song_id)
    if percent != 100:
        return copilot.slowed(workdir, song_id, name, percent)
    path = os.path.join(copilot.song_dir(workdir, song_id), "audio", f"{name}.flac")
    if not os.path.exists(path):
        raise Bass2SheetError(f"{song_id} is not prepared")
    return path


def _ready(workdir, song_id, name, percent):
    folder = os.path.join(copilot.song_dir(workdir, song_id), "audio")
    base, record = os.path.join(folder, f"{name}.flac"), os.path.join(folder, "record.flac")
    if not os.path.exists(base) or (name == "nobass" and os.path.getmtime(base) < os.path.getmtime(record)):
        return False
    if percent == 100:
        return True
    slow = os.path.join(folder, f"{name}-{percent}.flac")
    return os.path.exists(slow) and os.path.getmtime(slow) >= os.path.getmtime(base)


class Sounds:
    """What is being made for the page, in the background: slowing a record down takes most of a minute, longer
    than a phone waits for an answer. The page asks until it is there."""

    def __init__(self, workdir):
        self.workdir, self.lock, self.making = workdir, threading.Lock(), {}

    def ask(self, song_id, name, percent):
        """{ready: True, url} when it can be played, {ready: False} while it is being made (started by the first
        asking); a failure is told once, as Bass2SheetError."""
        if song_id not in copilot.load_songs(self.workdir):
            raise Bass2SheetError(f"no song {song_id}")
        if name not in SOUNDS or (percent != 100 and percent not in copilot.SPEEDS):
            raise Bass2SheetError(f"no {name} at {percent}%")
        key = (song_id, name, percent)
        with self.lock:
            job = self.making.get(key)
            if job is not None and not job["thread"].is_alive():
                del self.making[key]
                if job["error"]:
                    raise Bass2SheetError(job["error"])
                job = None
            if job is None and _ready(self.workdir, song_id, name, percent):
                tail = "" if percent == 100 else f"-{percent}"
                return dict(ready=True, url=f"/play/{song_id}/audio/{name}{tail}.flac")
            if job is None:
                job = self.making[key] = dict(error=None)
                job["thread"] = threading.Thread(target=self._make, args=(key, job), daemon=True)
                job["thread"].start()
        return dict(ready=False)

    def _make(self, key, job):
        try:
            sound(self.workdir, *key)
        except (Bass2SheetError, OSError, ValueError) as e:
            job["error"] = str(e) or type(e).__name__


# ------------------------------------------------------------------------------------------- when things are played


def grid_time(grid, i):
    """When sixteenth `i` of the song's grid is played, in seconds. Before the first and after the last the
    grid goes on at the pace it has there."""
    n = len(grid)
    if 0 <= i < n:
        return float(grid[i])
    if n < 2:
        return float(grid[0]) if n else 0.0
    if i < 0:
        return float(grid[0] + i * (grid[1] - grid[0]))
    return float(grid[-1] + (i - n + 1) * (grid[-1] - grid[-2]))


def written(machine, notes):
    """The person's notes as the score writes them (what copilot.sheet engraves)."""
    from .rhythm import quantize, simplify

    settings = dict(DEFAULTS, **machine.get("settings", {}))
    out = [
        dict(
            start=n["start"],
            end=n["end"],
            pitch=n["pitch"],
            flag=None,
            **({"also": n["also"]} if n.get("also") else {}),
        )
        for n in sorted(notes, key=lambda n: n["start"])
    ]
    out = quantize(out, np.array(machine["grid"]), machine["origin"])
    return simplify(out) if settings["rhythm"] == "simple" else out


def score_xml(machine, notes, bars, tab=None, chords=None, key=None):
    """The score to engrave for the page: (MusicXML with every line of `bars` bars on a page of its own and no
    title, the number of its first bar, the sixteenths counted from bar 1 where a note starts, those where one
    sounds). `tab` is how the notes are put on the neck (copilot.tab_style); `chords` is what is written over
    the bars (copilot.sheet_chords; by default nothing), and `key` the key it is written in (copilot.sheet_key;
    by default the key the song was given)."""
    tab = tab or copilot.TAB
    if not notes:
        raise Bass2SheetError("there are no notes to play along with")
    quantized = written(machine, notes)
    enters = max(1, min(n["grid_start"] // 16 + 1 for n in quantized))  # lines are counted from where the bass enters
    xml = score.musicxml(
        quantized,
        chords or {},
        key or tuple(machine["key"]),
        machine["tempo"],
        grid_time(machine["grid"], machine["origin"]),
        "",
        "",
        {},
        draft=False,
        phrase_start=1 + (enters - 1) % bars,
        bars_per_line=bars,
        tab_style=tab,
    )
    xml = _TIME_MARK.sub("", _HEAD.sub("", xml)).replace('<print new-system="yes"/>', '<print new-page="yes"/>')
    xml = xml.replace("<part-name>Bass</part-name>", "<part-name/>")  # the first line as wide as the others
    events = score.written_events(quantized, tab)
    first_bar = min(1, min(e["grid_start"] // 16 + 1 for e in events))  # as score.musicxml counts it
    starts = {e["g0"] for e in events if not e["grace"]}
    sounding = {g for e in events if not e["grace"] for g in range(e["g0"], e["g1"])}
    return xml, first_bar, starts, sounding


def positions(text):
    """MuseScore's positions file (of measures, or of segments) as a list in score order: where each is on its
    page in points of the picture (`x`, `y`, `w`, `h`), its `page`, and when the score's own tempo plays it
    (`ms`)."""
    root = ElementTree.fromstring(text)
    when = {e.get("elid"): float(e.get("position")) for e in root.iter("event")}
    out = []
    for e in root.iter("element"):
        box = {k: float(e.get(a)) / 12 for k, a in (("x", "x"), ("y", "y"), ("w", "sx"), ("h", "sy"))}
        out.append(dict(box, page=int(e.get("page")), ms=when[e.get("id")]))
    return out


def lines_of(measures, segments, first_bar, grid, origin, starts=(), sounding=()):
    """The score as the lines MuseScore laid out, each with its bars and what is played in them.

    A line is the measures that share a page and a height on it. Of each bar: its number, where it lies across
    the page (`x0`, `x1`) and when the record plays it (`t`). Of each mark, a note or a rest: where it is (`x`,
    `w`), when (`t`), and its kind: `n` a note starts, `t` a note goes on (tied), `r` a rest. Which sixteenth of
    its bar a mark is on is read from the score's own clock: how far it is between its bar and the next. Its
    time is that sixteenth of the song's grid."""
    if not measures:
        raise Bass2SheetError("MuseScore laid out no measures")
    lasts = [b["ms"] - a["ms"] for a, b in zip(measures, measures[1:])]
    lasts.append(lasts[-1] if lasts else 1.0)
    lines, k = [], 0
    for i, m in enumerate(measures):
        bar = first_bar + i
        if not lines or (lines[-1]["page"], round(lines[-1]["top"])) != (m["page"], round(m["y"])):
            lines.append(dict(page=m["page"], top=m["y"], bottom=m["y"] + m["h"], bars=[], marks=[]))
        line = lines[-1]
        line["bars"].append(dict(bar=bar, x0=m["x"], x1=m["x"] + m["w"], t=grid_time(grid, origin + (bar - 1) * 16)))
        until = m["ms"] + lasts[i] - 0.5 if i + 1 < len(measures) else float("inf")
        while k < len(segments) and segments[k]["ms"] < until:
            s = segments[k]
            k += 1
            at = 16 * (s["ms"] - m["ms"]) / lasts[i]
            if abs(at - round(at)) > 0.2 or not 0 <= round(at) < 16:
                raise Bass2SheetError("MuseScore's positions do not fit the score's sixteenths")
            g = (bar - 1) * 16 + round(at)
            kind = "n" if g in starts else "t" if g in sounding else "r"
            line["marks"].append(dict(x=s["x"], w=s["w"], t=grid_time(grid, origin + g), kind=kind))
    for line in lines:
        line["end"] = grid_time(grid, origin + line["bars"][-1]["bar"] * 16)
    return lines


# --------------------------------------------------------------------------------------------- the score as pictures


def _read_gray(path):
    import av

    with av.open(path) as c:
        return next(c.decode(video=0)).to_ndarray(format="gray")


def _write_gray(path, a):
    import av

    a = np.ascontiguousarray(a, dtype=np.uint8)
    with av.open(path, "w", format="image2") as c:
        stream = c.add_stream("png")
        stream.width, stream.height, stream.pix_fmt = a.shape[1], a.shape[0], "gray"
        for packet in stream.encode(av.VideoFrame.from_ndarray(a, format="gray")):
            c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def style():
    return STYLE.format(width=LINE * SPATIUM + 0.2, printable=LINE * SPATIUM, height=PAGE_HEIGHT * SPATIUM)


def engrave(xml, first_bar, grid, origin, starts, sounding, out_dir, mscore, tag="", timeout=300):
    """Engrave the score and cut it into its lines: one picture per line in `out_dir` (`line-<tag>-<k>.png`, all
    the same size, the staff at the same place in each), and what the page needs to move along them. Returns
    {width, height, lines}, the positions across a line as fractions of its width."""
    work = tempfile.mkdtemp(prefix=".engrave-", dir=out_dir)
    try:
        paths = {k: os.path.join(work, f"score.{k}") for k in ("musicxml", "mss", "png", "mpos", "spos", "json")}
        atomic_write_text(paths["musicxml"], xml)
        atomic_write_text(paths["mss"], style())
        job = [{"in": paths["musicxml"], "out": [paths["png"], paths["mpos"], paths["spos"]]}]
        atomic_write_text(paths["json"], json.dumps(job))
        try:
            r = subprocess.run(
                [mscore, "-S", paths["mss"], "-r", str(DPI), "-j", paths["json"]],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise Bass2SheetError(f"MuseScore took more than {timeout} s over the score") from e
        if r.returncode != 0 or not (os.path.exists(paths["mpos"]) and os.path.exists(paths["spos"])):
            tail = "\n".join(line for line in (r.stderr or "").splitlines() if "sandbox_extension" not in line)[-600:]
            raise Bass2SheetError(f"MuseScore did not engrave the score (exit {r.returncode}): {tail}")
        with open(paths["mpos"]) as f, open(paths["spos"]) as g:
            lines = lines_of(positions(f.read()), positions(g.read()), first_bar, grid, origin, starts, sounding)
        pages = {}
        for line in lines:  # MuseScore counts the pages from 0 here and names their pictures from 1
            page = os.path.join(work, f"score-{line['page'] + 1}.png")
            if line["page"] not in pages:
                if not os.path.exists(page):
                    raise Bass2SheetError("MuseScore wrote no picture of one of the score's pages")
                pages[line["page"]] = _read_gray(page)
        # what is drawn around each line's staves, measured from the top line of its staff; a line has its own
        # band of the page, should a page hold two (a line MuseScore had to break again)
        above = below = 0.0
        left, right = float("inf"), 0.0
        for line in lines:
            ink = pages[line["page"]] < 200
            same = [o for o in lines if o["page"] == line["page"]]
            j = same.index(line)
            lo = 0 if j == 0 else int((same[j - 1]["bottom"] + line["top"]) / 2)
            hi = ink.shape[0] if j == len(same) - 1 else int((line["bottom"] + same[j + 1]["top"]) / 2)
            rows = np.flatnonzero(ink[lo:hi].any(1)) + lo
            cols = np.flatnonzero(ink[lo:hi].any(0))
            if len(rows):
                above, below = max(above, line["top"] - rows[0]), max(below, rows[-1] + 1 - line["top"])
                left, right = min(left, cols[0]), max(right, cols[-1] + 1)
        pad = round(0.7 * SPATIUM * DPI)
        left, right = int(max(0, left - pad)), int(right + pad)
        width, height = right - left, round(above + below) + 2 * pad
        across = lambda x: round(min(1.0, max(0.0, (x - left) / width)), 5)
        out = []
        for i, line in enumerate(lines):
            page = pages[line["page"]]
            y0 = round(line["top"] - above) - pad
            cut = np.full((height, width), 255, np.uint8)
            r0, r1 = max(0, y0), min(page.shape[0], y0 + height)
            c1 = min(page.shape[1], right)
            cut[r0 - y0 : r1 - y0, : c1 - left] = page[r0:r1, left:c1]
            name = f"line-{tag}-{i:03d}.png" if tag else f"line-{i:03d}.png"
            _write_gray(os.path.join(work, name), cut)
            out.append(
                dict(
                    file=name,
                    end=round(line["end"], 3),
                    bars=[[b["bar"], across(b["x0"]), across(b["x1"]), round(b["t"], 3)] for b in line["bars"]],
                    marks=[
                        [across(m["x"]), round(m["w"] / width, 5), round(m["t"], 3), m["kind"]] for m in line["marks"]
                    ],
                )
            )
        for line in out:
            os.replace(os.path.join(work, line["file"]), os.path.join(out_dir, line["file"]))
        return dict(width=int(width), height=int(height), lines=out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def lines(workdir, song_id, bars=3, mscore=None):
    """A song's score for the page, as {key, bars, width, height, lines} with the pictures beside it in
    `<song>/play/b<bars>/`. Engraved from the person's notes as they stand; kept until they change.

    `bars` is the most bars a line gets. A line is always as wide as the page, so a song too busy for that many
    has its lines broken again by MuseScore, into lines of uneven length. When that happens to more than a tenth
    of them, the song is engraved with a bar less to the line (`bars` in the result says how many it got)."""
    if bars not in BARS:
        raise Bass2SheetError(f"lines of {bars} bars are not made: {', '.join(map(str, BARS))} are")
    folder = copilot.song_dir(workdir, song_id)
    machine, mine = (copilot.read_json(os.path.join(folder, name)) for name in ("machine.json", "notes.json"))
    if machine is None or mine is None:
        raise Bass2SheetError(f"{song_id} is not prepared")
    tab = copilot.tab_style(folder)
    chords, home = copilot.sheet_chords(folder, machine), copilot.sheet_key(folder, machine)
    xml, first_bar, starts, sounding = score_xml(machine, mine["notes"], bars, tab, chords, home)
    grid = json.dumps([machine["origin"], machine["grid"]])  # the same score on another grid is played at other times
    key = hashlib.sha1("\n".join([ENGRAVING, str(DPI), style(), grid, xml]).encode()).hexdigest()[:10]
    out_dir = os.path.join(folder, "play", f"b{bars}")
    kept_path = os.path.join(out_dir, "score.json")
    with _making[out_dir]:
        kept = copilot.read_json(kept_path)
        if (
            kept
            and kept.get("key") == key
            and all(os.path.exists(os.path.join(out_dir, line["file"])) for line in kept["lines"])
        ):
            return kept
        ms = score.find_mscore(mscore)
        if not ms:
            raise Bass2SheetError("the score is engraved by MuseScore 4, which is not installed (musescore.org)")
        os.makedirs(out_dir, exist_ok=True)
        for fit in range(bars, 0, -1):
            if fit < bars:
                xml, first_bar, starts, sounding = score_xml(machine, mine["notes"], fit, tab, chords, home)
            made = engrave(xml, first_bar, machine["grid"], machine["origin"], starts, sounding, out_dir, ms, key)
            meant = xml.count('<print new-page="yes"/>') + 1
            if len(made["lines"]) - meant <= 0.1 * meant:
                break
        made = dict(key=key, bars=fit, **made)
        atomic_write_text(kept_path, json.dumps(made))
        keep = {line["file"] for line in made["lines"]} | {"score.json"}
        for name in os.listdir(out_dir):  # the pictures of the notes as they were before
            if name not in keep and name.startswith("line-"):
                os.remove(os.path.join(out_dir, name))
        return made


# ---------------------------------------------------------------------------------------------------- for the page


def songs(workdir):
    out = []
    for song_id, song in copilot.load_songs(workdir).items():
        machine = copilot.read_json(os.path.join(copilot.song_dir(workdir, song_id), "machine.json"))
        if machine is not None:
            out.append(
                dict(
                    id=song_id,
                    title=song.get("title", song_id),
                    artist=song.get("artist", ""),
                    length=machine["length"],
                    tempo=round(machine["tempo"]),
                )
            )
    return out


def public(workdir, song_id, bars=3):
    """Everything the page needs of one song."""
    song = copilot.load_songs(workdir).get(song_id)
    if song is None:
        raise Bass2SheetError(f"no song {song_id}")
    made = lines(workdir, song_id, bars)
    machine = copilot.read_json(os.path.join(copilot.song_dir(workdir, song_id), "machine.json"))
    return dict(
        id=song_id,
        title=song.get("title", song_id),
        artist=song.get("artist", ""),
        length=machine["length"],
        tempo=machine["tempo"],
        sounds=list(SOUNDS),
        speeds=[100, *copilot.SPEEDS],
        images=f"/play/{song_id}/b{bars}/",
        **made,
    )


def _png(rgb):
    """A PNG of an (h, w, 3) array of bytes."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    head = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


@functools.lru_cache(maxsize=4)
def icon(size=180):
    """The picture the phone puts on its home screen: four strings and the note to play."""
    y, x = np.mgrid[0:size, 0:size] / size
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (31, 35, 40)
    for at in (0.32, 0.44, 0.56, 0.68):
        img[(np.abs(y - at) < 0.012) & (x > 0.14) & (x < 0.86)] = (225, 228, 232)
    img[(x - 0.4) ** 2 + (y - 0.56) ** 2 < 0.095**2] = (255, 196, 0)
    img[(np.abs(x - 0.66) < 0.012) & (y > 0.2) & (y < 0.8)] = (255, 196, 0)  # the playhead
    return _png(img)


MANIFEST = dict(
    name="bass2sheet play along",
    short_name="Play along",
    start_url="/play",
    scope="/play",
    display="fullscreen",
    orientation="landscape",
    background_color="#1f2328",
    theme_color="#1f2328",
    icons=[
        dict(src="/play/icon-180.png", sizes="180x180", type="image/png"),
        dict(src="/play/icon-512.png", sizes="512x512", type="image/png"),
    ],
)


def awake(workdir):
    """The path of a few seconds of black, silent film, or None if it cannot be made. A phone dims its screen
    under a page that only plays sound; one that plays a film stays lit. (Over plain http, which is what a home
    network gets, a page cannot ask the phone to stay awake any other way.)"""
    path = os.path.join(workdir, ".play-awake.mp4")
    with _making[path]:
        if not os.path.exists(path):
            import imageio_ffmpeg

            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
            cmd += ["-f", "lavfi", "-i", "color=c=black:s=32x32:r=5", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono"]
            cmd += ["-t", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-c:a", "aac"]
            cmd += ["-b:a", "16k", "-movflags", "+faststart", "-f", "mp4", path + ".part"]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                return None
            if r.returncode != 0 or not os.path.exists(path + ".part"):
                return None
            os.replace(path + ".part", path)
    return path


_FILES = re.compile(r"^/play/([^/]+)/(?:(b[234])/(line-[\w-]+\.png)|audio/((?:nobass|record|bass)(?:-\d+)?\.flac))$")


def send_path(handler, path, ctype, cache="no-store"):
    """A file, whole or the byte range asked for (a phone will not play sound without ranges)."""
    if not os.path.isfile(path):
        return handler.send_json(dict(error="not found"), 404)
    size = os.path.getsize(path)
    start, end, status = 0, size - 1, 200
    m = re.match(r"bytes=(\d*)-(\d*)$", handler.headers.get("Range", ""))
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start, end = int(m.group(1)), min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        else:
            start = max(0, size - int(m.group(2)))
        if start > end:
            return handler.send_body(b"", ctype, 416, {"Content-Range": f"bytes */{size}"})
        status = 206
    with open(path, "rb") as f:
        f.seek(start)
        body = f.read(end - start + 1)
    extra = {"Accept-Ranges": "bytes"}
    if status == 206:
        extra["Content-Range"] = f"bytes {start}-{end}/{size}"
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", cache)
    for k, v in extra.items():
        handler.send_header(k, v)
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


_sounds = {}


def get(handler, path, q):
    """Answer a GET under /play or /api/play, for the copilot's server (`handler` is its request handler)."""
    workdir = handler.workdir
    try:
        if path in ("/play", "/play/"):
            with open(PAGE_PATH, "rb") as f:
                return handler.send_body(f.read(), "text/html; charset=utf-8")
        if path == "/play/manifest.webmanifest":
            # On another device the app starts from the address it was added at: that one carries its key.
            mine = from_this_machine(handler.client_address[0])
            manifest = MANIFEST if mine else {k: v for k, v in MANIFEST.items() if k != "start_url"}
            return handler.send_body(json.dumps(manifest).encode(), "application/manifest+json")
        icon_ = re.match(r"^/play/icon-(180|512)\.png$", path)
        if icon_:
            return handler.send_body(icon(int(icon_.group(1))), "image/png")
        if path == "/play/awake.mp4":
            film = awake(workdir)
            return send_path(handler, film, "video/mp4") if film else handler.send_json(dict(error="not found"), 404)
        if path == "/api/play/songs":
            return handler.send_json(songs(workdir))
        if path == "/api/play/song":
            return handler.send_json(public(workdir, q.get("id", ""), int(q.get("bars", "3"))))
        if path == "/api/play/sound":
            maker = _sounds.setdefault(workdir, Sounds(workdir))
            return handler.send_json(maker.ask(q.get("id", ""), q.get("name", ""), int(q.get("speed", "100"))))
        m = _FILES.match(path)
        if m and m.group(1) in copilot.load_songs(workdir):
            folder = copilot.song_dir(workdir, m.group(1))
            if m.group(3):  # a line's picture: its name changes when the notes do, so the phone may keep it
                kept = "public, max-age=31536000, immutable"
                return send_path(handler, os.path.join(folder, "play", m.group(2), m.group(3)), "image/png", kept)
            return send_path(handler, os.path.join(folder, "audio", m.group(4)), "audio/flac")
        return handler.send_json(dict(error="not found"), 404)
    except ValueError:
        return handler.send_json(dict(error="not a number"), 400)
    except Bass2SheetError as e:
        return handler.send_json(dict(error=str(e)), 404)


# ------------------------------------------------------------------------------------------------ the home network


_HOME = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]  # what routers give out


def _ip(address):
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def from_this_machine(address):
    """Whether a request's sender is the machine the server runs on: only then may it change anything."""
    ip = _ip(address)
    return ip is not None and ip.is_loopback


def on_this_network(address):
    """Whether a request's sender is this machine or one on a home network with it. `--lan` is for the phone in
    the same room: should the computer be on a network that gives it an address the whole internet can reach,
    the songs are still not served to anyone further away."""
    ip = _ip(address)
    return ip is not None and (ip.is_loopback or ip.is_link_local or any(ip in net for net in _HOME))


def addresses():
    """This machine's addresses on a home network, the likeliest first. Not the address a phone's hotspot or
    another IPv6-only network gives it for its own use (192.0.0.x): nothing else can reach it there."""
    found = []
    try:  # the address the machine would send from: no packet leaves
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.255.255", 9))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    program = shutil.which("ifconfig") or ("/sbin/ifconfig" if os.path.exists("/sbin/ifconfig") else None)
    if program:
        try:
            listed = subprocess.run([program], capture_output=True, text=True, timeout=5).stdout
            found += re.findall(r"\binet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", listed)
        except (OSError, subprocess.TimeoutExpired):
            pass
    out = []
    for a in found:
        ip = _ip(a)
        if ip is not None and ip.version == 4 and any(ip in net for net in _HOME) and a not in out:
            out.append(a)
    return sorted(out, key=lambda a: not a.startswith("192.168."))


COOKIE = "bass2sheet_key"
WRONG, REST = 20, 600.0  # wrong keys in a row before no key is taken for a while, and for how long (seconds)
OPEN = re.compile(r"^/play/(manifest\.webmanifest|icon-(180|512)\.png)$")  # fetched without the cookie; no song in them
PAGES = ("/", "/play", "/play/", "/song", "/score", "/tests")
REFUSAL = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>bass2sheet</title><body style="font: 17px -apple-system, Helvetica, Arial, sans-serif; margin: 2em">
<p><b>This page needs the link from your computer.</b></p>
<p>On the computer bass2sheet runs on, open its first page. The link for a phone is at the top; it ends in a key
of eight digits.</p>"""


def phone_key(workdir):
    """The key a device other than this machine has to show before it is served anything: eight digits, made once
    and kept in the working directory, so that a link saved on a phone goes on working."""
    path = os.path.join(workdir, "phone_key")
    try:
        with open(path) as f:
            key = f.read().strip()
        if re.fullmatch(r"\d{8}", key):
            return key
    except OSError:
        pass
    key = f"{secrets.randbelow(10**8):08d}"
    os.makedirs(workdir, exist_ok=True)
    atomic_write_text(path, key + "\n")
    os.chmod(path, 0o600)
    return key


class Door:
    """Who is let in from the network when the server listens on it (`serve --lan`). A home network's addresses
    are every network's addresses, the office's and the cafe's too, and a laptop moves. So being on the network is
    not enough: a device shows the key, once in the link it opens and from then on in a cookie."""

    def __init__(self, key):
        self.key, self.wrong, self.rests_until, self.lock = key, 0, 0.0, threading.Lock()

    def knows(self, shown):
        """Whether `shown` is the key. After WRONG wrong ones in a row no key is taken for REST seconds, the right
        one included: nobody finds it by trying them all."""
        with self.lock:
            if time.monotonic() < self.rests_until:
                return False
            if hmac.compare_digest(str(shown).encode(), self.key.encode()):
                self.wrong = 0
                return True
            self.wrong += 1
            if self.wrong >= WRONG:
                self.wrong, self.rests_until = 0, time.monotonic() + REST
            return False


def admit(handler, path, q):
    """Whether a GET may be answered. This machine: always. Another one: only when the server was started to
    listen on the network, the device is on a home network with it, and it shows the key, in the link or in the
    cookie the link left. If not, the refusal has been sent and this is False."""
    address = handler.client_address[0]
    if from_this_machine(address):
        return True
    door = getattr(handler, "door", None)
    if door is None or not on_this_network(address):
        handler.send_json(dict(error="this page is for the network it is on"), 403)
        return False
    if OPEN.match(path):
        return True
    if "key" in q and door.knows(q["key"]):
        kept = f"{COOKIE}={door.key}; Max-Age=31536000; Path=/; HttpOnly; SameSite=Lax"
        handler.also = {"Set-Cookie": kept}
        return True
    try:
        held = http.cookies.SimpleCookie(handler.headers.get("Cookie", "")).get(COOKIE)
    except http.cookies.CookieError:
        held = None
    if held is not None and hmac.compare_digest(held.value.encode(), door.key.encode()):
        if path in ("/play", "/play/"):
            # The key stays in the address: what the phone adds to its home screen starts from there, and an app
            # on the home screen does not have the browser's cookies.
            handler.send_response(302)
            handler.send_header("Location", "/play?" + urlencode(dict(q, key=door.key)))
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return False
        return True
    if path in PAGES:
        handler.send_body(REFUSAL.encode(), "text/html; charset=utf-8", 403)
    else:
        handler.send_json(dict(error="this device has not shown the key"), 403)
    return False


def links(port, key):
    """Where a phone opens the play-along page: one link for each home-network address of this machine."""
    return [f"http://{a}:{port}/play?key={key}" for a in addresses()]


def phone(port, door):
    """What the first page says about playing along on a phone: whether the server listens for one, and the links."""
    return dict(on=door is not None, links=links(port, door.key) if door else [])


def announce(port, key):
    """What `serve --lan` prints: where to open the play-along page on a phone."""
    found = links(port, key)
    if not found:
        return [
            "Listening on the network, but this machine has no home-network address: it is not on Wi-Fi, or it is "
            "on a phone's hotspot, where the phone cannot reach it.",
            f"On this machine the play-along page is at http://localhost:{port}/play",
        ]
    lines_ = [f"On your phone, on the same Wi-Fi, open {found[0]}"]
    lines_ += [f"  (or {a})" for a in found[1:]]
    lines_ += [
        "  Hold the phone sideways. Share > Add to Home Screen makes it an app that opens full screen.",
        "  The key at the end of the link lets that device in; any other device on the network is refused.",
        "  A device that is let in can listen and read along; only this machine can change a score.",
    ]
    return lines_
