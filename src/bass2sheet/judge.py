"""Judging with a person in the loop.

Scores against reference transcriptions say whether the tool is exact. They cannot say whether its choices suit the
person who will play from the sheet. Here the person judges the pipeline one step at a time, on their own songs
and on each step's own output: the isolated bass tracks, the notes heard in them, the count, the written rhythm,
the tab. Each step sets one of the tool's standing choices (common.DEFAULTS) against its alternative on short
passages where the two differ, by ear and by eye, and the person picks without knowing which version is which.

The steps of a song open in order, and each is built on what was picked at the steps before it, so nobody
judges notes found in a bass track they turned down. A working directory holds everything: the versions
(snapshots that are never overwritten), the comparisons with their audio, pictures and score snippets, and the
picks. Every judged passage is kept, so when a later change to the tool alters one, it comes back as a new
comparison between what the person picked and what the tool writes now.
"""

import collections
import functools
import hashlib
import json
import os
import random
import re
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
import zlib

import librosa
import numpy as np
import soundfile as sf

from . import render, score
from .common import ALTERNATIVES, DEFAULTS, SR, Bass2SheetError, atomic_write_text, clean_name, sha1_file
from .rhythm import simplify

# The pipeline, one step at a time, in the order the tool runs it. Each step sets one of the tool's standing
# choices (common.DEFAULTS) against its alternative and is judged on that step's own output. `kind` is what the
# person hears and sees there:
#   stem     the two isolated bass tracks, and a pitch picture of each
#   heard    the notes found in the isolated bass, before any bars, drawn on its pitch picture
#   bars     the record with each version's count, and the sheet
#   writing  the sheet played as written
#   tab      the tab (the sound is the same)
STEPS = [
    dict(
        id="separator",
        setting="separator",
        kind="stem",
        name="Isolate the bass",
        about="Two models take the bass out of the record. A and B are the two isolated bass tracks. The "
        "pictures show what sounds in each: time from left to right, low notes at the bottom, darker where it "
        "is louder.",
        ask="Which isolated bass is cleaner and more complete?",
        labels=dict(roformer="bass isolated by BS-RoFormer", demucs="bass isolated by Demucs"),
    ),
    dict(
        id="replucks",
        setting="replucks",
        kind="heard",
        name="Find the notes",
        about="The notes the tool hears in the isolated bass, before any beats or bars. The thin line is the "
        "pitch it tracked and the boxes are the notes it made of it. One version starts a new note at every "
        "pluck it hears, the other only at a clear, bright pluck.",
        ask="Which set of notes is what the bass plays?",
        labels=dict(every="a new note at every re-pluck", clear="a new note only at a clear re-pluck"),
    ),
    dict(
        id="together",
        setting="together",
        kind="heard",
        name="Notes played together",
        about="Where the tool hears two notes played together, one version writes both and the other writes one "
        "note at a time, as the pitch tracker heard it. The passages are the ones where that makes a difference.",
        ask="Which is what the bass plays?",
        labels=dict(write="two notes where two are played together", single="one note at a time"),
    ),
    dict(
        id="barlines",
        setting="beat_tracker",
        kind="bars",
        name="Count the bars",
        about="The two versions put the beats and bar lines in different places. Each plays the record with its "
        "own count: a high click on beat 1, lower clicks on the other beats.",
        ask="Which count and which bar lines fit the music?",
        labels=dict(beat_this="beats from the Beat This! model", librosa="beats from the built-in tracker"),
    ),
    dict(
        id="rhythm",
        setting="rhythm",
        kind="writing",
        name="Write the rhythm",
        about="One version writes the rhythm as played. The other moves stray sixteenths onto eighths and "
        "closes short rests. What you hear is each sheet played exactly as written.",
        ask="Which would you rather read and play from?",
        labels=dict(literal="rhythm written as played", simple="rhythm simplified"),
    ),
    dict(
        id="tab",
        setting="tab",
        kind="tab",
        name="Place it on the neck",
        about="The same notes with two fingerings: one stays low on the neck and uses open strings, the other "
        "keeps the hand in one position.",
        ask="Which tab would you rather play from?",
        labels=dict(low="tab low on the neck, with open strings", position="tab with the hand in one position"),
    ),
]
CHANGE = dict(
    id="change",
    name="Changed since you judged",
    about="The tool has changed since you judged this passage, and now writes it differently from the version "
    "you picked.",
    ask="Which is better?",
    labels=dict(picked="the version you picked before", current="what the tool writes now"),
)
PICTURE_KINDS = ("stem", "heard")  # judged on a pitch picture of the bass track, not on a score
SELECT = dict(stem="notes", heard="notes", bars="bars", writing="writing", tab="tab")  # what a passage must differ in
ROLL_LO, ROLL_HI, ROLL_W = 26, 62, 1200  # the pitch pictures: D1 to D4, this many pixels wide
PICKS = ("A", "B", "same", "neither")
SHEET_REV = 2  # what a comparison's score snippets hold; older ones are engraved again (2: with cursor positions)
SNIPPET_DPI = 150
SNIPPET_RIGHT = (0.3 + 10.4) * SNIPPET_DPI  # where the system ends on the snippet page below, in pixels
NOTEHEAD = 13.4  # pixels at SNIPPET_DPI: the cursor runs through the middle of a note, not along its left edge
SNIPPET_STYLE = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.70"><Style>
<pageWidth>11</pageWidth><pageHeight>4</pageHeight><pagePrintableWidth>10.4</pagePrintableWidth>
<pageEvenLeftMargin>0.3</pageEvenLeftMargin><pageOddLeftMargin>0.3</pageOddLeftMargin>
<pageEvenTopMargin>0.3</pageEvenTopMargin><pageEvenBottomMargin>0.3</pageEvenBottomMargin>
<pageOddTopMargin>0.3</pageOddTopMargin><pageOddBottomMargin>0.3</pageOddBottomMargin><pageTwosided>0</pageTwosided>
<createMultiMeasureRests>0</createMultiMeasureRests><showMeasureNumberOne>1</showMeasureNumberOne>
<measureNumberSystem>1</measureNumberSystem><showFooter>0</showFooter><showHeader>0</showHeader>
</Style></museScore>
"""


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "song"


def choice_of(choice_id):
    return CHANGE if choice_id == "change" else next(s for s in STEPS if s["id"] == choice_id)


def song_entry(audio_path):
    name = clean_name(audio_path)
    artist, _, title = name.partition(" - ") if " - " in name else ("", "", name)
    return dict(id=slug(name), path=os.path.abspath(audio_path), title=title.strip(), artist=artist.strip())


# ---------------------------------------------------------------------------------------------------- versions


@functools.lru_cache(maxsize=1)
def code_id():
    """A hash of the modules that decide what is heard in a song, so results kept from an older tool are not
    taken for the current one's."""
    h = hashlib.sha1()
    for name in ("audio", "common", "notes", "pipeline", "pitch", "rhythm", "separate", "together"):
        with open(os.path.join(os.path.dirname(__file__), name + ".py"), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:8]


@functools.lru_cache(maxsize=3)
def _song(audio_path, cache_dir, stems_dir, separator, device):
    from .audio import decode, validate
    from .separate import load_stems, separate

    wav = decode(audio_path)
    validate(wav)
    audio_id = sha1_file(audio_path)
    if stems_dir:
        return wav, load_stems(stems_dir, wav.shape[1]), audio_id
    mix_path = os.path.join(cache_dir, f"mix-{audio_id[:16]}.wav")
    if not os.path.exists(mix_path):
        sf.write(mix_path, wav.T, SR, subtype="FLOAT")
    return wav, separate(separator, wav, mix_path, audio_id, cache_dir, device), audio_id


def load_song(audio_path, cache_dir=None, device=None, stems_dir=None, separator="roformer"):
    """(wav, stems, audio id, cache dir, device) for one separator, from the cache where possible. The last few
    songs stay in memory: a step's comparisons ask for the same song several times."""
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir or "~/.cache/bass2sheet"))
    os.makedirs(cache_dir, exist_ok=True)
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wav, stems, audio_id = _song(
        os.path.abspath(audio_path), cache_dir, stems_dir, None if stems_dir else separator, device
    )
    return wav, stems, audio_id, cache_dir, device


def hear(audio_path, settings, cache_dir=None, device=None, stems_dir=None, crepe_model="full", keep=None):
    """The notes heard in a song under `settings` (separator, replucks, together, beat_tracker), on their beat
    grid, with the key and the chords, as plain data. Heavy the first time; with stems and pitch tracks in the cache it
    takes seconds. `keep` is a directory where results are kept between runs, by settings and by code_id()."""
    from .pipeline import find_notes, seg_params

    together = settings.get("together", DEFAULTS["together"])  # versions from before the choice existed
    parts = [code_id(), "given" if stems_dir else settings["separator"], settings["replucks"], together]
    path = os.path.join(keep, "-".join([*parts, settings["beat_tracker"], crepe_model]) + ".json") if keep else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    wav, stems, audio_id, cache_dir, device = load_song(audio_path, cache_dir, device, stems_dir, settings["separator"])
    c = find_notes(
        stems,
        cache_dir,
        seg_params(settings["replucks"], together),
        device,
        crepe_model,
        wav.mean(0),
        audio_id,
        settings["beat_tracker"],
    )
    found = {k: c[k] for k in ("notes", "grid", "origin", "tempo", "key", "chords")}
    heard = json.loads(json.dumps(found, default=lambda x: x.tolist()))  # numpy values to plain ones
    if path:
        os.makedirs(keep, exist_ok=True)
        atomic_write_text(path, json.dumps(heard))
    return heard


def make_version(settings, heard):
    """How the tool writes a song under `settings`, from the notes as heard. Plain data, ready for JSON."""
    notes = simplify(heard["notes"]) if settings["rhythm"] == "simple" else heard["notes"]
    plain = json.loads(json.dumps(dict(notes=notes, chords=heard["chords"]), default=float))
    key = heard["key"]
    return dict(
        settings=dict(settings),
        tab=settings["tab"],
        notes=plain["notes"],
        grid=[round(float(t), 4) for t in heard["grid"]],
        origin=int(heard["origin"]),
        tempo=float(heard["tempo"]),
        key=[key[0], int(key[1]), bool(key[2])],
        chords=plain["chords"],
    )


def version_hash(v):
    core = [
        v["tab"],
        v["origin"],
        [round(t, 3) for t in v["grid"]],
        [(n["grid_start"], n["grid_len"], n["pitch"], bool(n.get("flag")), *n.get("also", [])) for n in v["notes"]],
    ]
    return hashlib.sha1(json.dumps(core).encode()).hexdigest()[:10]


def save_version(workdir, song_id, v):
    """Snapshots are never overwritten: the file name is a hash of what the version writes."""
    rel = os.path.join("versions", song_id, f"{version_hash(v)}.json")
    path = os.path.join(workdir, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write_text(path, json.dumps(v))
    return rel


def load_version(workdir, rel):
    with open(os.path.join(workdir, rel)) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------ what a version writes


def events(v):
    """The notes as the score shows them (see score.written_events), without ornaments."""
    return [e for e in score.written_events(v["notes"], v["tab"]) if not e["grace"]]


def g_time(v, g):
    """Where sixteenth `g` (counted from bar 1) falls in the recording."""
    grid = v["grid"]
    return grid[min(max(v["origin"] + g, 0), len(grid) - 1)]


def bar_time(v, bar):
    return g_time(v, 16 * (bar - 1))


def bar_times(v):
    return {k + 1: v["grid"][i] for k, i in enumerate(range(v["origin"], len(v["grid"]), 16))}


def same_grid(va, vb):
    ga, gb = va["grid"], vb["grid"]
    return va["origin"] == vb["origin"] and len(ga) == len(gb) and max(abs(a - b) for a, b in zip(ga, gb)) < 0.005


def grids_agree(va, vb, tol=0.035):
    """The two versions count the music the same way: nine in ten of A's beats and bar lines, where it has
    notes, have one of B's within `tol` seconds."""
    ea = events(va)
    if not ea:
        return True
    t0, t1 = g_time(va, min(e["g0"] for e in ea)), g_time(va, max(e["g1"] for e in ea))

    def share(a, b):
        a = np.array([t for t in a if t0 - tol <= t <= t1 + tol])
        b = np.array(b)
        return 1.0 if not len(a) else float(np.mean([np.abs(b - t).min() <= tol for t in a])) if len(b) else 0.0

    beats = lambda v: v["grid"][v["origin"] % 4 :: 4]
    bars = lambda v: v["grid"][v["origin"] % 16 :: 16]
    return share(beats(va), beats(vb)) >= 0.9 and share(bars(va), bars(vb)) >= 0.9


KEYS = {  # a note played together with others (`also`) is another note than the same one alone
    "notes": lambda e: (e["pitch"], *e.get("also", [])),
    "writing": lambda e: (e["pitch"], *e.get("also", []), e["g1"] - e["g0"]),
    "tab": lambda e: (e["pitch"], e["string"], e["fret"], *(tuple(x) for x in e.get("also_tab", []))),
}


def unmatched_events(va, vb, kind):
    """(A's written notes that B lacks, B's that A lacks). On one grid a note matches when it has the same place
    and the same `kind` key; across two grids, when it starts within 60 ms."""
    key = KEYS.get(kind, KEYS["notes"])
    ea, eb = events(va), events(vb)
    if same_grid(va, vb):
        a, b = {(e["g0"], *key(e)) for e in ea}, {(e["g0"], *key(e)) for e in eb}
        return [e for e in ea if (e["g0"], *key(e)) not in b], [e for e in eb if (e["g0"], *key(e)) not in a]
    pool = [(g_time(vb, e["g0"]), key(e)) for e in eb]
    used = [False] * len(pool)
    only_a = []
    for e in ea:
        t, k = g_time(va, e["g0"]), key(e)
        hit = next((i for i, (tb, kb) in enumerate(pool) if not used[i] and kb == k and abs(tb - t) <= 0.06), None)
        if hit is None:
            only_a.append(e)
        else:
            used[hit] = True
    return only_a, [e for e, u in zip(eb, used) if not u]


def unmatched(va, vb, kind):
    """The same, as [(A's bar, what, side)]: `what` is the note's place in its bar and its `kind` key."""
    key = KEYS.get(kind, KEYS["notes"])
    only_a, only_b = unmatched_events(va, vb, kind)
    if same_grid(va, vb):
        return [
            (e["g0"] // 16 + 1, (e["g0"] % 16, *key(e)), side)
            for side, es in (("a", only_a), ("b", only_b))
            for e in es
        ]
    starts = np.array(sorted(bar_times(va).items()))  # [bar, time]
    bar_at = lambda t: int(starts[max(0, np.searchsorted(starts[:, 1], t + 0.03) - 1), 0])
    out = []
    for side, v, es in (("a", va, only_a), ("b", vb, only_b)):
        for e in es:
            t = g_time(v, e["g0"])
            out.append((bar_at(t), (round(t - bar_time(va, bar_at(t)), 1), *key(e)), side))
    return out


def bars_covering(v, t0, t1):
    """The version's own bars that cover the recording from t0 to t1, or None."""
    times = bar_times(v)
    step = float(np.median(np.diff(sorted(times.values())))) if len(times) > 1 else 2.0
    inside = [bar for bar, t in times.items() if t0 - 0.5 * step <= t < t1 - 0.5 * step]
    return (min(inside), max(inside)) if inside else None


def pick_windows(va, vb, kind, count):
    """Up to `count` passages of version A's bars where the two versions differ most, each a different figure
    (a riff that repeats is judged once). Passages are 4 bars, or 2 at slow tempos. When the versions differ in
    where the bars fall, every passage differs, so the passages are spread through the song."""
    ea = events(va)
    if not ea or count < 1:
        return []
    starts = sorted(bar_times(va).values())
    bar_len = float(np.median(np.diff(starts))) if len(starts) > 1 else 2.0
    span = 4 if bar_len <= 2.7 else 2
    first, last = max(1, min(e["g0"] for e in ea) // 16 + 1), max((e["g1"] - 1) // 16 + 1 for e in ea)
    windows = list(range(first, last + 1, span))
    if kind == "bars":
        per_bar = {}
        for e in ea:
            per_bar[e["g0"] // 16 + 1] = per_bar.get(e["g0"] // 16 + 1, 0) + 1
        full = [s for s in windows if sum(per_bar.get(b, 0) for b in range(s, s + span)) >= 2 * span]
        picks = (
            sorted({full[int(round(i))] for i in np.linspace(0, len(full) - 1, min(count, len(full)))}) if full else []
        )
        return [dict(bars=(s, s + span - 1), weight=None) for s in picks]
    diff = unmatched(va, vb, kind)
    times_b = [g_time(vb, e["g0"]) for e in events(vb)]
    scored = []
    for s in windows:
        mine = [(bar - s, what, side) for bar, what, side in diff if s <= bar < s + span]
        if mine:
            t0, t1 = bar_time(va, s), bar_time(va, s + span)
            in_a = sum(1 for e in ea if s <= e["g0"] // 16 + 1 < s + span)
            in_b = sum(1 for t in times_b if t0 - 0.03 <= t < t1 - 0.03)
            thin = min(in_a, in_b) < 3  # one version has next to nothing here (an intro, a fade): judged last
            scored.append((thin, len(mine), s, frozenset(mine)))
    chosen, seen = [], set()
    for _, weight, s, figure in sorted(scored, key=lambda x: (x[0], -x[1], x[2])):
        if figure in seen:
            continue
        seen.add(figure)
        chosen.append(dict(bars=(s, s + span - 1), weight=weight))
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda w: w["bars"])


def passage(v, t0, t1, kind):
    """What the version writes between t0 and t1 of the recording: [seconds from t0, pitch, ...] per note, with
    the note's length for `writing` and its string and fret for `tab`."""
    out = []
    for e in events(v):
        t = g_time(v, e["g0"])
        if t0 - 0.02 <= t < t1 - 0.02:
            row = [round(t - t0, 3), e["pitch"]]
            if kind == "writing":
                row.append(round(g_time(v, e["g1"]) - t, 3))
            elif kind == "tab":
                row += [e["string"], e["fret"]]
            out.append(row)
    return out


def same_passage(p, q, tol=0.06):
    """Two passages say the same thing: the same notes in order, with times within `tol` seconds."""
    if len(p) != len(q):
        return False
    for a, b in zip(p, q):
        if abs(a[0] - b[0]) > tol or a[1] != b[1]:
            return False
        for x, y in zip(a[2:], b[2:]):
            if (abs(x - y) > tol) if isinstance(x, float) else (x != y):
                return False
    return True


# ------------------------------------------------------------------------------------------------------ media


def written_notes(v):
    """The sheet played exactly as written: each note from its written start to its written end."""
    return [
        dict(
            start=g_time(v, e["g0"]),
            end=g_time(v, e["g1"]),
            pitch=e["pitch"],
            flag=e.get("flag"),
            also=e.get("also", []),
        )
        for e in events(v)
    ]


class SongAudio:
    """One song as one separator took it apart: the record, the isolated bass and everything else, for building
    what the person hears and sees."""

    def __init__(
        self, audio_path, cache_dir=None, device=None, stems_dir=None, separator="roformer", crepe_model="full"
    ):
        wav, stems, _, self.cache_dir, self.device = load_song(audio_path, cache_dir, device, stems_dir, separator)
        self.mix = wav.astype(np.float64)
        self.bass = stems["bass"].astype(np.float64)
        self.bass_mono = stems["bass"].mean(0)  # as the pipeline takes it: the pitch tracks are cached by its hash
        self.rest = sum(s for name, s in stems.items() if name != "bass").astype(np.float64)
        self.n = wav.shape[1]
        self.crepe_model = crepe_model
        self._synth, self._feat = {}, None

    def level(self, s):
        """A synthesized bass at the level of the real one."""
        real = self.bass.mean(0)
        act = np.abs(real) > 1e-4
        return s * (np.sqrt(np.mean(real[act] ** 2) / (np.mean(s[act] ** 2) + 1e-12)) if act.any() else 1.0)

    def synth(self, v):
        """The version's bass as its sheet is written."""
        key = ("written", version_hash(v))
        if key not in self._synth:
            self._synth[key] = self.level(render.synth(written_notes(v), self.n))
        return self._synth[key]

    def synth_heard(self, v):
        """The version's notes as they were heard, each from where it starts to where it ends in the recording."""
        key = ("heard", version_hash(v))
        if key not in self._synth:
            self._synth[key] = self.level(render.synth(v["notes"], self.n))
        return self._synth[key]

    def pitch_line(self, w0, w1, floor=0.3):
        """The pitch the tracker followed between w0 and w1: [seconds from w0, MIDI pitch] every 20 ms, or None
        where it was not confident of a pitch."""
        from .pitch import features

        if self._feat is None:
            self._feat = features(self.bass_mono, self.cache_dir, self.device, self.crepe_model)
        f0, per = self._feat["f0_c"], self._feat["per_c"]
        out = []
        for i in range(max(0, int(w0 * 100)), min(len(f0), int(w1 * 100) + 1), 2):
            sure = per[i] >= floor and f0[i] > 0
            out.append([round(i / 100 - w0, 2), round(float(69 + 12 * np.log2(f0[i] / 440)), 2)] if sure else None)
        return out


def clip(x, t0, t1, fade=0.02):
    """Stereo excerpt with short fades."""
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    y = x[:, max(0, int(t0 * SR)) : int(t1 * SR)].copy()
    k = min(int(fade * SR), y.shape[1] // 2)
    if k:
        ramp = np.linspace(0, 1, k)
        y[:, :k] *= ramp
        y[:, -k:] *= ramp[::-1]
    return y


def clicks(v, t0, t1):
    """A count on the version's beats between t0 and t1: a high click on beat 1 of each bar, lower ones between."""
    out = np.zeros(max(1, int(t1 * SR) - max(0, int(t0 * SR))))
    t = np.arange(int(0.04 * SR)) / SR
    for i in range(v["origin"] % 4, len(v["grid"]), 4):
        when = v["grid"][i]
        if not t0 <= when < t1:
            continue
        down = (i - v["origin"]) % 16 == 0
        burst = (0.6 if down else 0.35) * np.sin(2 * np.pi * (1760 if down else 1100) * t) * np.exp(-t * 90)
        s = int((when - t0) * SR)
        n = min(len(burst), len(out) - s)
        out[s : s + n] += burst[:n]
    return out


def write_audio(path, x, scale):
    sf.write(path, np.clip(x * scale, -1, 1).T, SR, format="FLAC", subtype="PCM_16")


def snippet_xml(v, bars):
    return score.musicxml(
        v["notes"],
        v["chords"],
        tuple(v["key"]),
        v["tempo"],
        bar_time(v, 1),
        "",
        "",
        {},
        False,
        tab_style=v["tab"],
        bars=bars,
        bar_times=bar_times(v),
    )


def engrave_snippets(jobs, mscore, timeout=900):
    """[(MusicXML path, PNG path, positions path)] through one MuseScore run: the page as an image, and where
    each note and rest sits on it. Returns an error string or None."""
    if not jobs:
        return None
    folder = os.path.dirname(jobs[0][0])
    style, job = os.path.join(folder, ".snippet.mss"), os.path.join(folder, ".snippets.json")
    with open(style, "w") as f:
        f.write(SNIPPET_STYLE)
    with open(job, "w") as f:
        json.dump([{"in": x, "out": [png, spos]} for x, png, spos in jobs], f)
    try:
        r = subprocess.run(
            [mscore, "-S", style, "-r", str(SNIPPET_DPI), "-j", job], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"MuseScore timed out after {timeout} s"
    finally:
        for p in (style, job):
            if os.path.exists(p):
                os.remove(p)
    missing = 0
    for _, png, spos in jobs:
        page = png[:-4] + "-1.png"  # MuseScore numbers the pages of an image export
        if os.path.exists(page):
            os.replace(page, png)
        missing += not (os.path.exists(png) and os.path.exists(spos))
    if missing:
        tail = "\n".join(line for line in (r.stderr or "").splitlines() if "sandbox_extension" not in line)[-400:]
        return f"MuseScore wrote {len(jobs) - missing} of {len(jobs)} score snippets (exit {r.returncode}): {tail}"
    return None


def ink_box(png_path):
    """(x0, y0, x1, y1) of what is drawn on a page image, or None when the image cannot be read."""
    try:
        import av

        with av.open(png_path) as c:
            img = next(c.decode(video=0)).to_ndarray(format="rgba")
    except Exception:
        return None
    ink = (img[..., :3].min(-1) < 200) & (img[..., 3] > 0)
    ys, xs = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(xs) else None


def time_at(v, g):
    """g_time for a fractional sixteenth."""
    grid = v["grid"]
    return float(np.interp(v["origin"] + g, np.arange(len(grid)), grid))


def sheet_data(png, spos, v, bars, w0, marks=()):
    """What the page needs to run a cursor over a score snippet in step with the audio.

    `keys` are [seconds into the comparison's audio, x on the page image] for every note and rest, from the
    positions MuseScore reports for the engraved page; the cursor moves in a straight line between them. `crop`
    is the part of the page worth showing, `staff` the height the cursor spans, and `marks` the x of the notes
    written at the sixteenths in `marks` (the notes the other version writes differently)."""
    with open(png, "rb") as f:
        width, height = struct.unpack(">II", f.read(24)[16:24])
    root = ET.parse(spos).getroot()
    at = {e.get("id"): e for e in root.iter("element")}
    system = next(iter(at.values()))
    top = float(system.get("y")) / 12  # positions are in twelfths of a pixel at the export resolution
    bottom = top + float(system.get("sy")) / 12
    per_sixteenth = 60000 / float(f"{v['tempo']:.0f}") / 4  # the snippet plays at the tempo printed on it
    keys, where = [], {}
    for ev in root.iter("event"):
        x = float(at[ev.get("elid")].get("x")) / 12 + NOTEHEAD / 2
        g = 16 * (bars[0] - 1) + float(ev.get("position")) / per_sixteenth
        keys.append([round(time_at(v, g) - w0, 3), round(x, 1)])
        where[round(g)] = round(x, 1)
    keys.append([round(g_time(v, 16 * bars[1]) - w0, 3), SNIPPET_RIGHT - 3])
    box = ink_box(png) or (30, top - 115, width - 30, bottom + 22)
    crop = [max(0, box[0] - 8), max(0, int(box[1]) - 8), min(width, box[2] + 8), min(height, int(box[3]) + 8)]
    return dict(
        page=[width, height],
        crop=crop,
        staff=[round(top - 8, 1), round(bottom + 8, 1)],
        keys=keys,
        marks=sorted(where[g] for g in marks if g in where),
    )


def spectrum(mono, w0, w1):
    """What sounds in a bass track between w0 and w1, by pitch and time: rows from ROLL_HI down to ROLL_LO, three
    to a semitone, and ROLL_W columns. Linear amplitude. The lowest notes are blurred in time: telling E1 from
    F1 takes a few tenths of a second of sound."""
    a, b = max(0, int((w0 - 1.5) * SR)), min(len(mono), int((w1 + 1.5) * SR))
    y = librosa.resample(np.asarray(mono[a:b], dtype=np.float32), orig_sr=SR, target_sr=11025)
    c = np.abs(
        librosa.cqt(
            y,
            sr=11025,
            hop_length=64,
            fmin=librosa.midi_to_hz(ROLL_LO - 1 / 3),
            n_bins=3 * (ROLL_HI - ROLL_LO + 1),
            bins_per_octave=36,
        )
    )
    times = a / SR + np.arange(c.shape[1]) * 64 / 11025
    edges = np.searchsorted(times, np.linspace(w0, w1, ROLL_W + 1))
    c = np.stack([c[:, i : max(j, i + 1)].max(1) for i, j in zip(edges[:-1], edges[1:])], axis=1)
    c = np.pad(c, ((1, 1), (1, 1)), mode="edge")  # a light blur hides the filter bank's ripple
    c = 0.25 * c[:-2, 1:-1] + 0.5 * c[1:-1, 1:-1] + 0.25 * c[2:, 1:-1]
    return c[::-1]


def write_png(path, rgb):
    """An 8-bit RGB image, (height, width, 3), without an imaging library."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    head = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
        )


def write_picture(path, c, ref, other=None):
    """A spectrum as a picture: white where nothing sounds, dark blue at `ref` and above, over 42 dB. With
    `other`, the same passage of another track, what sounds here and hardly there (12 dB less) turns amber.
    Returns [width, height]."""
    level = lambda x: 20 * np.log10(np.maximum(x, 1e-12) / ref)
    db = level(c)
    v = np.clip(1 + db / 42, 0, 1)[..., None]
    ink = np.array([16.0, 42.0, 96.0]) * np.ones_like(v)
    if other is not None:
        ink = np.where(((db > -30) & (db - level(other) > 12))[..., None], np.array([214.0, 120.0, 0.0]), ink)
    rgb = (255 - v * (255 - ink)).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=0)  # nine pixels to a semitone
    write_png(path, rgb)
    return [rgb.shape[1], rgb.shape[0]]


def roll_view(src, size, length, notes=(), line=()):
    """A pitch picture for the page, in the shape of a score snippet (see sheet_data): the cursor crosses it at
    a steady pace. `notes` are [start, end, MIDI pitch, differs] in seconds into the audio; `line` is the
    tracked pitch."""
    w, h = size
    return dict(
        src=src,
        page=[w, h],
        crop=[0, 0, w, h],
        staff=[0, h],
        keys=[[0.0, 0.0], [round(length, 3), w]],
        marks=[],
        roll=dict(lo=ROLL_LO, hi=ROLL_HI, notes=list(notes), line=list(line)),
    )


def heard_boxes(v, other, w0, w1):
    """The version's notes as heard between w0 and w1, for drawing: [start, end, pitch, differs], where a note
    differs when the other version has none of that pitch starting within 60 ms of it. Notes played together
    are one box each."""
    every = lambda n: (n["pitch"], *n.get("also", []))
    theirs = [(n["start"], pitch) for n in other["notes"] for pitch in every(n)]
    out = []
    for n in v["notes"]:
        if n["end"] > w0 and n["start"] < w1:
            for pitch in every(n):
                alone = not any(p == pitch and abs(t - n["start"]) <= 0.06 for t, p in theirs)
                out.append([round(max(n["start"] - w0, 0), 3), round(min(n["end"], w1) - w0, 3), pitch, int(alone)])
    return out


def render_comparison(comp, va, vb, audio_for, workdir):
    """Write what the person hears for one comparison and, for the first two steps, the pitch pictures they see
    (render_sheets adds the score snippets of the later steps).

    Everything in a comparison covers the same stretch of the record, so the page can switch between its
    sources without losing its place. Files are named for what the person sees (A, B), never for the setting.
    `audio_for(version)` gives the SongAudio of the separator that version was made with."""
    folder = os.path.join(workdir, "media", comp["id"])
    os.makedirs(folder, exist_ok=True)
    kind, t0, t1 = comp["kind"], comp["t0"], comp["t1"]
    shown = dict(A=vb, B=va) if comp["flip"] else dict(A=va, B=vb)
    audio = audio_for(va)
    bar_len = (t1 - t0) / max(1, comp["bars_a"][1] - comp["bars_a"][0] + 1)
    lead = bar_len + 0.15 if kind == "bars" else 0.4
    w0, w1 = max(0.0, t0 - lead), min(audio.n / SR, t1 + 0.4)
    peak = lambda *xs: max(float(np.abs(x).max()) for x in xs) + 1e-9
    rms = lambda x: float(np.sqrt(np.mean(x**2))) + 1e-9
    rel = lambda name: f"media/{comp['id']}/{name}"
    record = clip(audio.mix, w0, w1)
    sources, alt, sheets = [["record", "Record"], ["A", "A"], ["B", "B"]], None, None
    if kind == "stem":  # the isolated bass tracks themselves, and what each left behind in the record
        solo = {name: clip(audio_for(v).bass, w0, w1) for name, v in shown.items()}
        rest = {name: clip(audio_for(v).rest, w0, w1) for name, v in shown.items()}
        loud, quiet = 0.9 / peak(record, *rest.values()), 0.85 / peak(*solo.values())
        out = dict(record=(record, loud), **{name: (x, quiet) for name, x in solo.items()})
        out.update({f"{name}_rest": (x, loud) for name, x in rest.items()})
        alt = dict(label="What was left behind", media=dict(A="A_rest", B="B_rest"))
        spectra = {name: spectrum(audio_for(v).bass_mono, w0, w1) for name, v in shown.items()}
        ref = max(float(c.max()) for c in spectra.values()) + 1e-9
        sheets = {}
        for name, c in spectra.items():
            other = spectra["B" if name == "A" else "A"]
            size = write_picture(os.path.join(folder, f"{name}_picture.png"), c, ref, other)
            sheets[name] = roll_view(rel(f"{name}_picture.png"), size, w1 - w0)
    elif kind == "heard":  # the isolated bass against the notes found in it, as heard
        real = clip(audio.bass, w0, w1)
        notes = {name: clip(audio.synth_heard(v), w0, w1) for name, v in shown.items()}
        level = 0.85 / peak(real, *notes.values())
        out = dict(record=(real, level), **{name: (x, level) for name, x in notes.items()})
        out.update({f"{name}_both": (np.stack([real.mean(0), x[0]]), level) for name, x in notes.items()})
        sources[0][1] = "Isolated bass"
        alt = dict(label="Real bass left, notes right", media=dict(A="A_both", B="B_both"))
        c = spectrum(audio.bass_mono, w0, w1)
        size = write_picture(os.path.join(folder, "picture.png"), c, float(c.max()) + 1e-9)
        line = audio.pitch_line(w0, w1)
        other = dict(A=shown["B"], B=shown["A"])
        sheets = {
            name: roll_view(rel("picture.png"), size, w1 - w0, heard_boxes(v, other[name], w0, w1), line)
            for name, v in shown.items()
        }
    elif kind == "bars":  # the record itself, counted in by each version
        level = 0.8 * np.abs(record).max()
        mixes = {name: record + level * clicks(v, w0, w1)[None, : record.shape[1]] for name, v in shown.items()}
        scale = 0.9 / peak(record, *mixes.values())
        out = dict(record=(record, scale), **{name: (x, scale) for name, x in mixes.items()})
    else:  # the record with the written bass in place of the real one, as loud as the record
        mixes = {name: clip(audio.rest + audio.synth(v)[None, :], w0, w1) for name, v in shown.items()}
        mixes = {name: x * (rms(record) / rms(x)) for name, x in mixes.items()}
        solos = dict(
            record=clip(audio.bass, w0, w1), **{name: clip(audio.synth(v), w0, w1) for name, v in shown.items()}
        )
        scale, quiet = 0.9 / peak(record, *mixes.values()), 0.8 / peak(*solos.values())
        out = dict(record=(record, scale), **{name: (x, scale) for name, x in mixes.items()})
        out.update({f"{name}_bass": (x, quiet) for name, x in solos.items()})
        alt = dict(label="Bass only", media=dict(record="record_bass", A="A_bass", B="B_bass"))
    same = kind == "tab"  # two fingerings of the same notes sound the same
    if same:
        out = {name: x for name, x in out.items() if not name.startswith("B")}
        sources = [["record", "Record"], ["A", "Transcription"]]
    for name, (x, scale) in out.items():
        write_audio(os.path.join(folder, f"{name}.flac"), x, scale)
    media = {name: rel(f"{name}.flac") for name in out}
    if alt:
        alt["media"] = {name: media[key] for name, key in alt["media"].items() if key in media}
    comp.update(
        media={name: path for name, path in media.items() if "_" not in name},
        sources=sources,
        alt=alt,
        same_audio=same,
        lead=round(t0 - w0, 3),
        length=round(w1 - w0, 3),
    )
    if sheets:
        comp.update(sheets=sheets, sheet_rev=SHEET_REV)


# ---------------------------------------------------------------------------------------------- the workspace


def read_json(workdir, name, default):
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_comparisons(workdir):
    return read_json(workdir, "comparisons.json", [])


def save_comparisons(workdir, comps):
    atomic_write_text(os.path.join(workdir, "comparisons.json"), json.dumps(comps, indent=1))


def load_judgments(workdir):
    """{comparison id: the latest pick}. Picks are appended, so a changed mind keeps its history."""
    path = os.path.join(workdir, "judgments.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    out[j["id"]] = j
    return out


def side_of(comp, pick):
    """Which version a pick names: 'a', 'b', or None for "no difference" and "neither". The page shows the two
    versions as A and B in an order drawn at random when the comparison was made (`flip`)."""
    if pick not in ("A", "B"):
        return None
    return ("b" if pick == "A" else "a") if comp["flip"] else ("a" if pick == "A" else "b")


def record_judgment(workdir, comp, pick, note=""):
    if pick not in PICKS:
        raise Bass2SheetError(f"unknown pick {pick!r}; expected one of {', '.join(PICKS)}")
    side = side_of(comp, pick)
    j = dict(
        id=comp["id"],
        pick=pick,
        side=side,
        value=comp[f"{side}_value"] if side else None,
        note=(note or "").strip(),
        at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(os.path.join(workdir, "judgments.jsonl"), "a") as f:
        f.write(json.dumps(j) + "\n")
    return j


def order(comps):
    """The order comparisons are shown in: changed passages first, then the pipeline's steps in order, and
    within a step one passage of every song before a second of any."""
    rank = {s["id"]: k for k, s in enumerate(STEPS)}
    return sorted(comps, key=lambda c: (c["choice"] != "change", rank.get(c["choice"], -1), c["round"], c["song"]))


def shown_labels(comp):
    """What the versions shown as A and B were, for the reveal after a pick."""
    return dict(A=comp["b_label"], B=comp["a_label"]) if comp["flip"] else dict(A=comp["a_label"], B=comp["b_label"])


def read_settings(workdir):
    """How this directory's comparisons are built: what `build` was last run with."""
    return read_json(workdir, "settings.json", {})


def steps_of(opts):
    wanted = opts.get("choices")
    return [s for s in STEPS if not wanted or s["id"] in wanted]


def upstream_key(base, steps, k):
    """The settings of the steps before step `k` that its comparisons are built on, as text."""
    return ",".join(f"{s['setting']}={base[s['setting']]}" for s in steps[:k])


def chain(song, comps, judged, opts):
    """Walk one song's steps in order: (the ids of its comparisons that count now, the settings its judged steps
    settle on, the next step to build as (index, settings, key) or None).

    A step's comparisons count once every step before it is judged, and only those built on what was picked
    there: change an earlier pick and the later comparisons built on the old one stop counting. A song settles
    on a step's alternative when it was picked more often than the default in that song's passages."""
    steps = steps_of(opts)
    base = dict(DEFAULTS, **(opts.get("defaults") or {}))
    live = []
    for k, step in enumerate(steps):
        key = upstream_key(base, steps, k)
        if key not in song.get("built", {}).get(step["id"], {}):
            return live, base, (k, dict(base), key)
        mine = [c for c in comps if (c["song"], c["choice"], c.get("base_key")) == (song["id"], step["id"], key)]
        live += [c["id"] for c in mine]
        if any(c["id"] not in judged for c in mine):
            return live, base, None
        default = base[step["setting"]]
        alt = next(v for v in ALTERNATIVES[step["setting"]] if v != default)
        votes = collections.Counter(judged[c["id"]]["value"] for c in mine)
        base[step["setting"]] = alt if votes[alt] > votes[default] else default
    return live, base, None


def live_comparisons(workdir, comps=None, judged=None):
    """The comparisons that count now (see chain), in the order they are shown."""
    comps = load_comparisons(workdir) if comps is None else comps
    judged = load_judgments(workdir) if judged is None else judged
    songs, opts = read_json(workdir, "songs.json", {}), read_settings(workdir)
    live = {cid for song in songs.values() for cid in chain(song, comps, judged, opts)[0]}
    return order([c for c in comps if c["choice"] == "change" or c["song"] not in songs or c["id"] in live])


def progress(workdir, live=None, judged=None):
    """Per step, in order: how many passages count now and how many of them are judged."""
    live = live_comparisons(workdir) if live is None else live
    judged = load_judgments(workdir) if judged is None else judged
    out = []
    for step in [CHANGE, *steps_of(read_settings(workdir))]:
        mine = [c for c in live if c["choice"] == step["id"]]
        if mine or step is not CHANGE:
            out.append(
                dict(id=step["id"], name=step["name"], total=len(mine), judged=sum(c["id"] in judged for c in mine))
            )
    return out


def tally(workdir):
    """Per choice, over the comparisons that count now: how often each alternative was picked, how often no
    difference was heard or neither was right, how many are still open, and the notes the person left."""
    judged = load_judgments(workdir)
    out = {}
    for c in live_comparisons(workdir, judged=judged):
        ch = choice_of(c["choice"])
        t = out.setdefault(
            c["choice"],
            dict(labels=ch["labels"], picks={v: 0 for v in ch["labels"]}, same=0, neither=0, open=0, notes=[]),
        )
        j = judged.get(c["id"])
        if not j:
            t["open"] += 1
            continue
        if j["value"] is None:
            t[j["pick"]] += 1
        else:
            t["picks"][j["value"]] = t["picks"].get(j["value"], 0) + 1
        if j["note"]:
            t["notes"].append(dict(id=c["id"], song=c["title"], pick=j["value"] or j["pick"], note=j["note"]))
    return out


def add_comparison(
    comps, song, choice, kind, va, vb, rel_a, rel_b, window, rng, values, default=None, key="", round_no=1
):
    """A new comparison for a passage of version A's bars. `values` names the two versions (a setting's two
    values); the one equal to `default` is marked in its label. `key` is what it was built on (upstream_key)."""
    t0, t1 = bar_time(va, window["bars"][0]), bar_time(va, window["bars"][1] + 1)
    taken = [int(c["id"].rsplit(".", 1)[1]) for c in comps if c["song"] == song["id"] and c["choice"] == choice]
    n = 1 + max(taken, default=0)
    label = lambda v: choice_of(choice)["labels"][v] + (" (the default)" if v == default else "")
    other = window["bars"] if same_grid(va, vb) else bars_covering(vb, t0, t1)
    comp = dict(
        id=f"{song['id']}.{choice}.{n}",
        song=song["id"],
        title=song["title"],
        artist=song["artist"],
        choice=choice,
        kind=kind,
        round=round_no,
        base_key=key,
        a=rel_a,
        b=rel_b,
        a_value=values[0],
        b_value=values[1],
        a_label=label(values[0]),
        b_label=label(values[1]),
        flip=rng.random() < 0.5,
        t0=round(t0, 3),
        t1=round(t1, 3),
        bars_a=list(window["bars"]),
        bars_b=list(other) if other else None,
        weight=window["weight"],
    )
    comps.append(comp)
    return comp


def render_sheets(workdir, comps, mscore, say=print, force=False):
    """Engrave the score snippets of every comparison that has none, or has them from an older layout, and work
    out where the cursor runs over each. Returns how many comparisons were engraved."""
    stale = [
        c
        for c in comps
        if c.get("media") and c["kind"] not in PICTURE_KINDS and (force or c.get("sheet_rev") != SHEET_REV)
    ]
    if stale and not mscore:
        say("MuseScore 4 not found: the comparisons have audio but no score snippets")
        stale = []
    versions = {}

    def version(rel):
        if rel not in versions:
            versions[rel] = load_version(workdir, rel)
        return versions[rel]

    plan = []
    for c in stale:
        va, vb = version(c["a"]), version(c["b"])
        only = dict(a=[], b=[])
        if c["kind"] in ("notes", "writing") and same_grid(va, vb):  # where the two differ, for the page to mark
            lo, hi = 16 * (c["bars_a"][0] - 1), 16 * c["bars_a"][1]
            only_a, only_b = ([e["g0"] for e in es if lo <= e["g0"] < hi] for es in unmatched_events(va, vb, c["kind"]))
            total = sum(lo <= e["g0"] < hi for e in events(va) + events(vb))
            if 2 * (len(only_a) + len(only_b)) <= total:  # when most notes differ, marking them all says nothing
                only = dict(a=only_a, b=only_b)
        folder = os.path.join(workdir, "media", c["id"])
        for name, side in (("A", "b"), ("B", "a")) if c["flip"] else (("A", "a"), ("B", "b")):
            bars = c[f"bars_{side}"]
            if bars:
                v = version(c[side])
                files = tuple(os.path.join(folder, f"{name}.{ext}") for ext in ("musicxml", "png", "spos"))
                atomic_write_text(files[0], snippet_xml(v, tuple(bars)))
                plan.append((c, name, v, bars, only[side], files))
    err = engrave_snippets([files for *_, files in plan], mscore) if plan else None
    if err:
        say(err)
    for c in stale:
        c["sheets"] = {}
        c["media"] = {k: rel for k, rel in c["media"].items() if not k.endswith("_sheet")}
    for c, name, v, bars, marks, (_, png, spos) in plan:
        if os.path.exists(png) and os.path.exists(spos):
            data = sheet_data(png, spos, v, bars, c["t0"] - c["lead"], marks)
            c["sheets"][name] = dict(src=f"media/{c['id']}/{name}.png", **data)
    for c in stale:
        if c["sheets"]:
            c["sheet_rev"] = SHEET_REV
    save_comparisons(workdir, comps)
    return len(stale)


def same_passage_of(c, what):
    """A comparison of the same passage of the same two versions, already there."""
    return (c["song"], c["choice"], c["a"], c["b"], c["bars_a"]) == what


def audio_source(song, opts):
    """audio_for(version): the song as the version's separator took it apart, loaded once."""
    made = {}

    def audio_for(v):
        separator = v["settings"]["separator"]
        if separator not in made:
            made[separator] = SongAudio(
                song["path"],
                opts.get("cache_dir"),
                opts.get("device"),
                song.get("stems_dir"),
                separator,
                opts.get("crepe_model", "full"),
            )
        return made[separator]

    return audio_for


def heard_under(workdir, song, opts, keep=True):
    """version(settings): how the tool writes this song under `settings`."""
    folder = os.path.join(workdir, "heard", song["id"]) if keep else None
    kw = dict(
        cache_dir=opts.get("cache_dir"),
        device=opts.get("device"),
        stems_dir=song.get("stems_dir"),
        crepe_model=opts.get("crepe_model", "full"),
    )
    return lambda settings: make_version(settings, hear(song["path"], settings, keep=folder, **kw))


def asked_already(workdir, comps, song, step_id, key, va, vb, bars):
    """True when a comparison of this step that still counts shows these bars as the two versions write them."""
    for c in comps:
        if (c["song"], c["choice"], c.get("base_key"), c["bars_a"]) != (song["id"], step_id, key, list(bars)):
            continue
        if not changed_how(load_version(workdir, c["a"]), va, c) and not changed_how(
            load_version(workdir, c["b"]), vb, c
        ):
            return True
    return False


def build_step(workdir, song, step, base, key, comps, opts, rng, say):
    """Make one step's comparisons for one song on `base`: the tool's defaults with what was picked at the
    earlier steps. Returns how many comparisons the step has on this base."""
    setting = step["setting"]
    if setting == "separator" and song.get("stems_dir"):
        say(f"  {step['name']}: the stems were given, so there is nothing to compare")
        return 0
    alt = next(v for v in ALTERNATIVES[setting] if v != base[setting])
    version = heard_under(workdir, song, opts)
    try:
        va, vb = version(base), version(dict(base, **{setting: alt}))
    except Bass2SheetError as e:
        say(f"  {step['name']}: skipped ({e})")
        return 0
    if step["kind"] == "bars" and grids_agree(va, vb):
        say(f"  {step['name']}: the two versions count the song the same way")
        return 0
    rel_a, rel_b = save_version(workdir, song["id"], va), save_version(workdir, song["id"], vb)
    audio_for = audio_source(song, opts)
    count = 0
    for w in pick_windows(va, vb, SELECT[step["kind"]], opts.get("per_song", 3)):
        count += 1
        what = (song["id"], step["id"], rel_a, rel_b, list(w["bars"]))
        old = next((c for c in comps if same_passage_of(c, what)), None)
        if old:  # reached before from other earlier picks: its judgment stands
            old["base_key"] = key
            continue
        if asked_already(workdir, comps, song, step["id"], key, va, vb, w["bars"]):
            continue  # the tool has changed since, but not what it writes in this passage
        values = (base[setting], alt)
        comp = add_comparison(
            comps, song, step["id"], step["kind"], va, vb, rel_a, rel_b, w, rng, values, values[0], key, count
        )
        render_comparison(comp, va, vb, audio_for, workdir)
    say(f"  {step['name']}: {count} passage{'' if count == 1 else 's'}")
    return count


def advance(workdir, say=None):
    """Build what can be built now: for every song, the first step that is not built yet and whose earlier steps
    are all judged. Returns the number of new comparisons."""
    say = say or (lambda text: print(text, flush=True))
    opts, songs, comps = read_settings(workdir), read_json(workdir, "songs.json", {}), load_comparisons(workdir)
    rng = random.Random(len(comps) + 1)
    before = len(comps)
    for song in songs.values():
        named = False
        while True:
            todo = chain(song, comps, load_judgments(workdir), opts)[2]
            if not todo:
                break
            if not named:
                say(f"{song['artist']} - {song['title']}" if song["artist"] else song["title"])
                named = True
            k, base, key = todo
            step = steps_of(opts)[k]
            count = build_step(workdir, song, step, base, key, comps, opts, rng, say)
            render_sheets(workdir, comps, score.find_mscore(opts.get("mscore")), say)  # saves the comparisons
            song.setdefault("built", {}).setdefault(step["id"], {})[key] = count  # only now does the step count
            atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
    return len(comps) - before


def build(
    audio_paths,
    workdir,
    choices=None,
    per_song=3,
    cache_dir=None,
    device=None,
    stems_dirs=None,
    mscore=None,
    crepe_model="full",
    defaults=None,
    say=None,
):
    """Add songs and prepare the first step for each: transcribe it under the tool's default and under that
    step's alternative, find the passages where the two differ, and render what the person will hear and see.
    The later steps are prepared as the earlier ones are judged (advance), each on what was picked before it.
    `defaults` overrides common.DEFAULTS (tests use it to stay off models that need a download)."""
    unknown = set(choices or []) - {s["id"] for s in STEPS}
    if unknown:
        known = ", ".join(s["id"] for s in STEPS)
        raise Bass2SheetError(f"unknown choice {', '.join(sorted(unknown))}; the choices are {known}")
    os.makedirs(workdir, exist_ok=True)
    opts = dict(
        choices=list(choices) if choices else None,
        per_song=per_song,
        cache_dir=cache_dir,
        device=device,
        mscore=mscore,
        crepe_model=crepe_model,
        defaults=defaults or {},
    )
    atomic_write_text(os.path.join(workdir, "settings.json"), json.dumps(opts, indent=1))
    songs = read_json(workdir, "songs.json", {})
    for path in audio_paths:
        entry = song_entry(path)
        songs[entry["id"]] = dict(songs.get(entry["id"], {}), **entry, stems_dir=(stems_dirs or {}).get(path))
    atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
    return advance(workdir, say)


def forget_unjudged_steps(song, comps, judged):
    """Drop the steps of a song that nobody has judged, with their comparisons, so that they are built again."""
    for step_id, built in song.get("built", {}).items():
        for key in list(built):
            mine = [c["id"] for c in comps if (c["song"], c["choice"], c.get("base_key")) == (song["id"], step_id, key)]
            if not any(cid in judged for cid in mine):
                comps[:] = [c for c in comps if c["id"] not in mine]
                del built[key]


def retire_unsettled(workdir, song, judged, live, version, now):
    """Set aside the comparisons of a song that are not settled (not judged yet, or judged "neither") and whose
    passage the tool now writes differently under either version's settings. Their step is built again, by the
    tool as it is now; a pick for A, B or "no difference" is not touched here (see update). Returns how many
    were set aside."""
    gone = 0
    for c in live:
        if c["song"] != song["id"] or c["kind"] == "stem" or c["choice"] == "change":
            continue
        if judged.get(c["id"], {}).get("pick") in ("A", "B", "same"):
            continue
        for side in ("a", "b"):
            was = load_version(workdir, c[side])
            name = json.dumps(was["settings"], sort_keys=True)
            if name not in now:
                now[name] = version(was["settings"])
            if changed_how(was, now[name], c):
                song.get("built", {}).get(c["choice"], {}).pop(c.get("base_key"), None)
                c["base_key"] = f"(set aside) {c.get('base_key')}"  # counts no more; its step is open again
                gone += 1
                break
    return gone


def changed_how(picked, now, c):
    """None when the tool still writes comparison `c`'s passage as the picked version did; otherwise what differs
    now, which decides how the two are compared: the count ("bars"), the notes heard, or only the writing."""
    kind = SELECT.get(c["kind"], "notes")
    same = lambda k: same_passage(passage(picked, c["t0"], c["t1"], k), passage(now, c["t0"], c["t1"], k))
    agree = grids_agree(picked, now)
    if agree and same("notes" if kind == "bars" else kind):
        return None
    return "bars" if not agree else "heard" if c["kind"] == "heard" else kind if same("notes") else "notes"


def update(workdir, say=None):
    """After the tool has changed: transcribe every song again and look at each judged passage.

    A passage the tool still writes as the picked version did needs nothing. One it now writes differently,
    under the same settings, comes back as a new comparison between the pick and the new writing. What is not
    settled is built again by the tool as it is now: steps nobody has judged yet, and passages that were not
    judged or where neither version was right, if they changed. Returns (passages kept, picks the tool's
    defaults do not follow, passages that changed and wait for a new pick). Isolated bass tracks are not looked
    at: they change only with the separation models."""
    say = say or (lambda text: print(text, flush=True))
    opts, songs = read_settings(workdir), read_json(workdir, "songs.json", {})
    comps, judged = load_comparisons(workdir), load_judgments(workdir)
    defaults = dict(DEFAULTS, **(opts.get("defaults") or {}))
    live = [c for c in live_comparisons(workdir, comps, judged) if c["choice"] != "change" and c["song"] in songs]
    stray = sum(
        judged[c["id"]]["value"] not in (None, defaults[choice_of(c["choice"])["setting"]])
        for c in live
        if c["id"] in judged
    )
    rng = random.Random(len(comps) + 1)
    kept = new = aside = 0
    for song in songs.values():
        forget_unjudged_steps(song, comps, judged)  # built again below, by the tool as it is now
        mine = [
            c for c in live if c["song"] == song["id"] and judged.get(c["id"], {}).get("pick") in ("A", "B", "same")
        ]
        version, audio_for, now = heard_under(workdir, song, opts), audio_source(song, opts), {}
        aside += retire_unsettled(workdir, song, judged, live, version, now)
        for c in mine:
            if c["kind"] == "stem":
                kept += 1
                continue
            side = judged[c["id"]]["side"] or "a"  # "no difference": either version stands for the pick
            picked = load_version(workdir, c[side])
            name = json.dumps(picked["settings"], sort_keys=True)
            if name not in now:
                now[name] = version(picked["settings"])
            how = changed_how(picked, now[name], c)
            if not how:
                kept += 1
                continue
            rel_now = save_version(workdir, song["id"], now[name])
            window = dict(bars=tuple(c[f"bars_{side}"]), weight=None)
            if any(same_passage_of(x, (song["id"], "change", c[side], rel_now, list(window["bars"]))) for x in comps):
                continue  # asked already
            comp = add_comparison(
                comps, song, "change", how, picked, now[name], c[side], rel_now, window, rng, ("picked", "current")
            )
            render_comparison(comp, picked, now[name], audio_for, workdir)
            new += 1
    render_sheets(workdir, comps, score.find_mscore(opts.get("mscore")), say)
    atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
    advance(workdir, say)
    say(
        f"{kept} judged passages are still written as you picked; {new} changed and wait for a new pick; "
        f"{stray} of your picks are for an alternative the tool does not use by default"
    )
    if aside:
        say(f"{aside} passages that were open or had no right version are written differently now: built again")
    return kept, stray, new


def report(workdir):
    """The picks so far, as Markdown: per step, and what each song settles on."""
    t = tally(workdir)
    if not t:
        return "# Picks so far\n\nNo comparisons yet: run `bass2sheet-judge build`.\n"
    lines = ["# Picks so far", ""]
    for cid, row in t.items():
        lines += [f"## {choice_of(cid)['name']}", "", "| | Picked |", "|---|---|"]
        lines += [f"| {row['labels'].get(v, v)} | {n} |" for v, n in row["picks"].items()]
        lines += [f"| No difference | {row['same']} |", f"| Neither is right | {row['neither']} |"]
        lines += [f"| Not judged yet | {row['open']} |", ""]
        lines += [f"- {n['song']} ({n['id']}), picked {n['pick']}: {n['note']}" for n in row["notes"]]
        lines += [""] if row["notes"] else []
    songs, opts = read_json(workdir, "songs.json", {}), read_settings(workdir)
    defaults = dict(DEFAULTS, **(opts.get("defaults") or {}))
    comps, judged = load_comparisons(workdir), load_judgments(workdir)
    moved = []
    for song in songs.values():
        base = chain(song, comps, judged, opts)[1]
        changed = [
            choice_of(s["id"])["labels"][base[s["setting"]]]
            for s in steps_of(opts)
            if base[s["setting"]] != defaults[s["setting"]]
        ]
        if changed:
            moved.append(f"- {song['title']}: {'; '.join(changed)}")
    if moved:
        lines += ["## Songs whose later steps are built on an alternative", "", *moved, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="bass2sheet-judge",
        description="Judge the tool's choices on your own songs, step by step, by ear and by eye.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="add songs and prepare the first step's comparisons for each")
    b.add_argument("songs", nargs="*", help="audio files you have the rights to use")
    b.add_argument("--per-song", type=int, default=3, help="passages per song and step (default 3)")
    b.add_argument("--choices", help="comma-separated: " + ", ".join(s["id"] for s in STEPS) + " (default all)")
    b.add_argument("--cache-dir", help="where models, stems and pitch tracks are cached")
    b.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    b.add_argument("--mscore", help="path to the MuseScore 4 binary")
    s = sub.add_parser("serve", help="open the judging page on this machine")
    s.add_argument("--port", type=int, default=8765)
    r = sub.add_parser("report", help="print the picks so far")
    u = sub.add_parser("update", help="after the tool changed: bring back judged passages it now writes differently")
    e = sub.add_parser("sheets", help="engrave every comparison's score snippets again; picks and audio are kept")
    for p in (b, s, r, u, e):
        p.add_argument("--dir", default="judging", help="working directory (default ./judging)")
    args = ap.parse_args(argv)
    try:
        if args.command == "build":
            missing = [p for p in args.songs if not os.path.isfile(p)]
            if missing:
                raise Bass2SheetError(f"no such file: {missing[0]}")
            if not args.songs and not os.path.exists(os.path.join(args.dir, "songs.json")):
                raise Bass2SheetError("name at least one song to judge")
            n = build(
                args.songs,
                args.dir,
                args.choices.split(",") if args.choices else None,
                args.per_song,
                args.cache_dir,
                args.device,
                mscore=args.mscore,
            )
            print(f"{n} new comparisons in {args.dir}. Judge them with: bass2sheet-judge serve --dir {args.dir}")
        elif args.command == "serve":
            from .judge_server import serve

            serve(args.dir, args.port)
        elif args.command == "report":
            print(report(args.dir), end="")
        elif args.command == "sheets":
            mscore = score.find_mscore(read_settings(args.dir).get("mscore"))
            n = render_sheets(args.dir, load_comparisons(args.dir), mscore, force=True)
            print(f"score snippets engraved for {n} comparisons")
        else:
            update(args.dir)
    except Bass2SheetError as e:
        print(f"bass2sheet-judge: {e}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
