"""Judging with a person in the loop.

Scores against reference transcriptions say whether the tool is exact. They cannot say whether its choices suit the
person who will play from the sheet. Here two versions of a transcription of the person's own song are compared
on short passages where they differ: by ear (the record with each version's written bass in place of the real
one) and by eye (the same bars of the two scores). The person picks without knowing which version is which.

Each comparison sets one of the tool's standing choices (common.DEFAULTS) against its alternative. A working
directory holds everything: the versions (snapshots that are never overwritten), the comparisons with their
audio and score snippets, and the picks. Every judged passage is kept, so when a later change to the tool alters
one, it comes back as a new comparison between what the person picked and what the tool writes now.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import time

import numpy as np
import soundfile as sf

from . import render, score
from .common import ALTERNATIVES, DEFAULTS, SR, Bass2SheetError, atomic_write_text, clean_name, sha1_file
from .rhythm import simplify

# What is put in front of the person. `kind` says what can differ between the two versions: which notes there
# are (notes), how the same line is written (writing), where it sits on the neck (tab), or where the beats and
# bar lines fall (bars).
CHOICES = [
    dict(
        id="separator",
        setting="separator",
        kind="notes",
        about="The two versions isolate the bass from the record with two different models, and so hear "
        "different notes.",
        ask="Which is closer to the bass on the record?",
        labels=dict(roformer="bass isolated by BS-RoFormer", demucs="bass isolated by Demucs"),
    ),
    dict(
        id="replucks",
        setting="replucks",
        kind="notes",
        about="When a note is plucked again while it rings, one version writes a new note at every pluck it "
        "hears. The other does so only at a clear, bright pluck, which leaves fewer and longer notes.",
        ask="Which is closer to what the bassist plays?",
        labels=dict(every="a new note at every re-pluck", clear="a new note only at a clear re-pluck"),
    ),
    dict(
        id="rhythm",
        setting="rhythm",
        kind="writing",
        about="One version writes the rhythm as played. The other moves stray sixteenths onto eighths and "
        "closes short rests. What you hear is each sheet played exactly as written.",
        ask="Which would you rather read and play from?",
        labels=dict(literal="rhythm written as played", simple="rhythm simplified"),
    ),
    dict(
        id="barlines",
        setting="beat_tracker",
        kind="bars",
        about="The two versions put the beats and bar lines in different places. Each plays the record with its "
        "own count: a high click on beat 1, lower clicks on the other beats.",
        ask="Which count and which bar lines fit the music?",
        labels=dict(beat_this="beats from the Beat This! model", librosa="beats from the built-in tracker"),
    ),
    dict(
        id="tab",
        setting="tab",
        kind="tab",
        about="The same notes with two fingerings: one stays low on the neck and uses open strings, the other "
        "keeps the hand in one position.",
        ask="Which tab would you rather play from?",
        labels=dict(low="tab low on the neck, with open strings", position="tab with the hand in one position"),
    ),
]
CHANGE = dict(
    id="change",
    about="The tool has changed since you judged this passage, and now writes it differently from the version "
    "you picked.",
    ask="Which is better?",
    labels=dict(picked="the version you picked before", current="what the tool writes now"),
)
WRITING = ("rhythm", "tab")  # settings that write the same notes differently: no new transcription needed
PICKS = ("A", "B", "same", "neither")
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
    return CHANGE if choice_id == "change" else next(c for c in CHOICES if c["id"] == choice_id)


def song_entry(audio_path):
    name = clean_name(audio_path)
    artist, _, title = name.partition(" - ") if " - " in name else ("", "", name)
    return dict(id=slug(name), path=os.path.abspath(audio_path), title=title.strip(), artist=artist.strip())


# ---------------------------------------------------------------------------------------------------- versions


def hear(audio_path, settings, cache_dir=None, device=None, stems_dir=None, crepe_model="full"):
    """Transcribe a song under `settings` (separator, replucks, beat_tracker): the notes as heard, on their beat
    grid. Heavy the first time; with stems and pitch tracks in the cache it takes seconds."""
    from .pipeline import seg_params, transcribe

    wav, stems, audio_id, cache_dir, device = load_song(audio_path, cache_dir, device, stems_dir, settings["separator"])
    c = transcribe(
        stems,
        cache_dir,
        seg_params(settings["replucks"]),
        "off",
        None,
        device,
        crepe_model,
        wav.mean(0),
        audio_id,
        settings["beat_tracker"],
    )
    return {k: c[k] for k in ("notes", "grid", "origin", "tempo", "key", "chords")}


def load_song(audio_path, cache_dir=None, device=None, stems_dir=None, separator="roformer"):
    """(wav, stems, audio id, cache dir, device) for one separator, from the cache where possible."""
    from .audio import decode, validate
    from .separate import load_stems, separate

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir or "~/.cache/bass2sheet"))
    os.makedirs(cache_dir, exist_ok=True)
    wav = decode(audio_path)
    validate(wav)
    audio_id = sha1_file(audio_path)
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if stems_dir:
        stems = load_stems(stems_dir, wav.shape[1])
    else:
        mix_path = os.path.join(cache_dir, f"mix-{audio_id[:16]}.wav")
        if not os.path.exists(mix_path):
            sf.write(mix_path, wav.T, SR, subtype="FLOAT")
        stems = separate(separator, wav, mix_path, audio_id, cache_dir, device)
    return wav, stems, audio_id, cache_dir, device


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
        [(n["grid_start"], n["grid_len"], n["pitch"], bool(n.get("flag"))) for n in v["notes"]],
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


KEYS = {
    "notes": lambda e: (e["pitch"],),
    "writing": lambda e: (e["pitch"], e["g1"] - e["g0"]),
    "tab": lambda e: (e["pitch"], e["string"], e["fret"]),
}


def unmatched(va, vb, kind):
    """Written notes that one version has and the other does not: [(A's bar, what, side)]. On one grid a note
    matches when it has the same place and the same `kind` key; across two grids, when it starts within 60 ms."""
    key = KEYS.get(kind, KEYS["notes"])
    ea, eb = events(va), events(vb)
    if same_grid(va, vb):
        a, b = {(e["g0"], *key(e)) for e in ea}, {(e["g0"], *key(e)) for e in eb}
        return [(k[0] // 16 + 1, (k[0] % 16, *k[1:]), "a") for k in a - b] + [
            (k[0] // 16 + 1, (k[0] % 16, *k[1:]), "b") for k in b - a
        ]
    starts = np.array(sorted(bar_times(va).items()))  # [bar, time]
    bar_at = lambda t: int(starts[max(0, np.searchsorted(starts[:, 1], t + 0.03) - 1), 0])
    pool = [(g_time(vb, e["g0"]), key(e)) for e in eb]
    used = [False] * len(pool)
    out = []
    for e in ea:
        t, k = g_time(va, e["g0"]), key(e)
        hit = next((i for i, (tb, kb) in enumerate(pool) if not used[i] and kb == k and abs(tb - t) <= 0.06), None)
        if hit is None:
            out.append((bar_at(t), (round(t - bar_time(va, bar_at(t)), 1), *k), "a"))
        else:
            used[hit] = True
    out += [
        (bar_at(tb), (round(tb - bar_time(va, bar_at(tb)), 1), *kb), "b")
        for i, (tb, kb) in enumerate(pool)
        if not used[i]
    ]
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
        dict(start=g_time(v, e["g0"]), end=g_time(v, e["g1"]), pitch=e["pitch"], flag=e.get("flag")) for e in events(v)
    ]


class SongAudio:
    """The record, its separated bass and everything else, for building what the person hears."""

    def __init__(self, audio_path, cache_dir=None, device=None, stems_dir=None, separator="roformer"):
        wav, stems, *_ = load_song(audio_path, cache_dir, device, stems_dir, separator)
        self.mix = wav.astype(np.float64)
        self.bass = stems["bass"].astype(np.float64)
        self.rest = sum(s for name, s in stems.items() if name != "bass").astype(np.float64)
        self.n = wav.shape[1]
        self._synth = {}

    def synth(self, v):
        """The version's written bass, at the level of the real one."""
        key = version_hash(v)
        if key not in self._synth:
            s = render.synth(written_notes(v), self.n)
            real = self.bass.mean(0)
            act = np.abs(real) > 1e-4
            gain = np.sqrt(np.mean(real[act] ** 2) / (np.mean(s[act] ** 2) + 1e-12)) if act.any() else 1.0
            self._synth[key] = s * gain
        return self._synth[key]


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
    """[(MusicXML path, PNG path)] through one MuseScore run. Returns an error string or None."""
    if not jobs:
        return None
    folder = os.path.dirname(jobs[0][0])
    style, job = os.path.join(folder, ".snippet.mss"), os.path.join(folder, ".snippets.json")
    with open(style, "w") as f:
        f.write(SNIPPET_STYLE)
    with open(job, "w") as f:
        json.dump([{"in": x, "out": p} for x, p in jobs], f)
    try:
        r = subprocess.run(
            [mscore, "-S", style, "-T", "14", "-r", "150", "-j", job], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"MuseScore timed out after {timeout} s"
    finally:
        for p in (style, job):
            if os.path.exists(p):
                os.remove(p)
    missing = 0
    for _, png in jobs:
        page = png[:-4] + "-1.png"  # MuseScore numbers the pages of an image export
        if os.path.exists(page):
            os.replace(page, png)
        elif not os.path.exists(png):
            missing += 1
    if missing:
        tail = "\n".join(line for line in (r.stderr or "").splitlines() if "sandbox_extension" not in line)[-400:]
        return f"MuseScore wrote {len(jobs) - missing} of {len(jobs)} score snippets (exit {r.returncode}): {tail}"
    return None


def render_comparison(comp, va, vb, audio, workdir):
    """Write the audio for one comparison and return the score-snippet jobs still to engrave.

    Everything in a comparison covers the same stretch of the record, so the page can switch between the record
    and the two versions without losing its place. Files are named for what the person sees (A, B), never for
    the setting."""
    folder = os.path.join(workdir, "media", comp["id"])
    os.makedirs(folder, exist_ok=True)
    kind, t0, t1 = comp["kind"], comp["t0"], comp["t1"]
    shown = dict(A=vb, B=va) if comp["flip"] else dict(A=va, B=vb)
    bars = dict(A=comp["bars_b"], B=comp["bars_a"]) if comp["flip"] else dict(A=comp["bars_a"], B=comp["bars_b"])
    bar_len = (t1 - t0) / max(1, comp["bars_a"][1] - comp["bars_a"][0] + 1)
    lead = bar_len + 0.15 if kind == "bars" else 0.4
    w0, w1 = max(0.0, t0 - lead), min(audio.n / SR, t1 + 0.4)
    record = clip(audio.mix, w0, w1)
    if kind == "bars":  # the record itself, counted in by each version
        level = 0.8 * np.abs(record).max()
        mixes = {name: record + level * clicks(v, w0, w1)[None, : record.shape[1]] for name, v in shown.items()}
        solos = {}
    else:  # the record with the written bass in place of the real one
        mixes = {name: clip(audio.rest + audio.synth(v)[None, :], w0, w1) for name, v in shown.items()}
        solos = {name: clip(audio.synth(v), w0, w1) for name, v in shown.items()}
        solos["record"] = clip(audio.bass, w0, w1)
    same = kind == "tab"  # two fingerings of the same notes sound the same
    media = {"record": f"media/{comp['id']}/record.flac"}
    scale = 0.9 / (max(np.abs(x).max() for x in (record, *mixes.values())) + 1e-9)
    write_audio(os.path.join(folder, "record.flac"), record, scale)
    for name, x in mixes.items():
        if not (same and name == "B"):
            write_audio(os.path.join(folder, f"{name}.flac"), x, scale)
            media[name] = f"media/{comp['id']}/{name}.flac"
    if solos:
        solo_scale = 0.8 / (max(np.abs(x).max() for x in solos.values()) + 1e-9)
        for name, x in solos.items():
            if not (same and name == "B"):
                write_audio(os.path.join(folder, f"{name}_bass.flac"), x, solo_scale)
                media[f"{name}_bass"] = f"media/{comp['id']}/{name}_bass.flac"
    comp.update(media=media, same_audio=same, lead=round(t0 - w0, 3), length=round(w1 - w0, 3))
    jobs = []
    for name, v in shown.items():
        if bars[name]:
            xml = os.path.join(folder, f"{name}.musicxml")
            atomic_write_text(xml, snippet_xml(v, tuple(bars[name])))
            jobs.append((xml, os.path.join(folder, f"{name}.png")))
    return jobs


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
    """The order comparisons are shown in: changed passages first, then one passage of every choice for every
    song before a second of any, so that a short sitting still touches everything."""
    rank = {c["id"]: k for k, c in enumerate(CHOICES)}
    return sorted(comps, key=lambda c: (c["choice"] != "change", c["round"], rank.get(c["choice"], -1), c["song"]))


def shown_labels(comp):
    """What the versions shown as A and B were, for the reveal after a pick."""
    return dict(A=comp["b_label"], B=comp["a_label"]) if comp["flip"] else dict(A=comp["a_label"], B=comp["b_label"])


def tally(workdir):
    """Per choice: how often each alternative was picked, how often no difference was heard or neither was
    right, how many comparisons are still open, and the notes the person left."""
    comps, judged = load_comparisons(workdir), load_judgments(workdir)
    out = {}
    for c in comps:
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


def add_comparison(comps, song, choice, kind, va, vb, rel_a, rel_b, window, rng, values, default=None):
    """A new comparison for a passage of version A's bars, unless the same one is already there. `values` names
    the two versions (a setting's two values); the one equal to `default` is marked in its label."""
    t0, t1 = bar_time(va, window["bars"][0]), bar_time(va, window["bars"][1] + 1)
    for c in comps:
        if (c["song"], c["a"], c["b"], c["bars_a"]) == (song["id"], rel_a, rel_b, list(window["bars"])):
            return None
    n = 1 + sum(1 for c in comps if c["song"] == song["id"] and c["choice"] == choice)
    label = lambda v: choice_of(choice)["labels"][v] + (" (the default)" if v == default else "")
    other = window["bars"] if same_grid(va, vb) else bars_covering(vb, t0, t1)
    comp = dict(
        id=f"{song['id']}.{choice}.{n}",
        song=song["id"],
        title=song["title"],
        artist=song["artist"],
        choice=choice,
        kind=kind,
        round=n,
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


def finish_sheets(workdir, comps, jobs, mscore, say=print):
    """Engrave the pending score snippets and note on each comparison which images exist."""
    if jobs and not mscore:
        say("MuseScore 4 not found: the comparisons have audio but no score snippets")
    elif jobs:
        err = engrave_snippets(jobs, mscore)
        if err:
            say(err)
    for c in comps:
        for name in ("A", "B"):
            rel = f"media/{c['id']}/{name}.png"
            if os.path.exists(os.path.join(workdir, rel)):
                c.setdefault("media", {})[f"{name}_sheet"] = rel
    save_comparisons(workdir, comps)


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
    say=print,
):
    """Transcribe each song under the tool's defaults and under each alternative, find the passages where the
    two differ, and render what the person will hear and see. Running it again adds songs or choices and keeps
    every pick. `defaults` overrides common.DEFAULTS (tests use it to stay off models that need a download)."""
    unknown = set(choices or []) - {c["id"] for c in CHOICES}
    if unknown:
        known = ", ".join(c["id"] for c in CHOICES)
        raise Bass2SheetError(f"unknown choice {', '.join(sorted(unknown))}; the choices are {known}")
    os.makedirs(workdir, exist_ok=True)
    wanted = [c for c in CHOICES if choices is None or c["id"] in choices]
    base_settings = dict(DEFAULTS, **(defaults or {}))
    songs, comps = read_json(workdir, "songs.json", {}), load_comparisons(workdir)
    rng = random.Random(len(comps) + 1)
    jobs, added = [], 0
    for path in audio_paths:
        song = song_entry(path)
        stems_dir = (stems_dirs or {}).get(path)
        songs[song["id"]] = dict(song, stems_dir=stems_dir)
        say(f"{song['artist']} - {song['title']}" if song["artist"] else song["title"])
        heard = {}

        def version(settings, path=path, stems_dir=stems_dir, heard=heard):
            key = tuple(settings[k] for k in ("separator", "replucks", "beat_tracker"))
            if key not in heard:
                heard[key] = hear(path, settings, cache_dir, device, stems_dir, crepe_model)
            return make_version(settings, heard[key])

        base = version(base_settings)
        rel_base = save_version(workdir, song["id"], base)
        audio = SongAudio(path, cache_dir, device, stems_dir, base_settings["separator"])
        for ch in wanted:
            setting = ch["setting"]
            alt = next(v for v in ALTERNATIVES[setting] if v != base_settings[setting])
            try:
                other = version(dict(base_settings, **{setting: alt}))
            except Bass2SheetError as e:
                say(f"  {ch['id']}: skipped ({e})")
                continue
            if ch["kind"] == "bars" and grids_agree(base, other):
                say(f"  {ch['id']}: the two versions count the song the same way")
                continue
            rel_other = save_version(workdir, song["id"], other)
            new = 0
            for w in pick_windows(base, other, ch["kind"], per_song):
                values = (base_settings[setting], alt)
                comp = add_comparison(
                    comps, song, ch["id"], ch["kind"], base, other, rel_base, rel_other, w, rng, values, values[0]
                )
                if comp:
                    jobs += render_comparison(comp, base, other, audio, workdir)
                    new += 1
            added += new
            say(f"  {ch['id']}: {new} new passage{'' if new == 1 else 's'}")
        atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
        save_comparisons(workdir, comps)
    finish_sheets(workdir, comps, jobs, score.find_mscore(mscore), say)
    return added


def update(workdir, cache_dir=None, device=None, mscore=None, crepe_model="full", defaults=None, say=print):
    """After the tool has changed: transcribe every song again and look at each judged passage.

    A passage the tool still writes as the person picked needs nothing. One it writes as the version they
    rejected is counted: the tool does not follow that pick (yet). One it now writes differently from both comes
    back as a new comparison between the pick and the new writing. Returns (kept, not followed, new)."""
    songs, comps, judged = read_json(workdir, "songs.json", {}), load_comparisons(workdir), load_judgments(workdir)
    base_settings = dict(DEFAULTS, **(defaults or {}))
    rng = random.Random(len(comps) + 1)
    kept = ignored = new = 0
    jobs = []
    for song in songs.values():
        mine = [
            c for c in comps if c["song"] == song["id"] and judged.get(c["id"], {}).get("pick") in ("A", "B", "same")
        ]
        if not mine:
            continue
        stems_dir = song.get("stems_dir")
        base = make_version(base_settings, hear(song["path"], base_settings, cache_dir, device, stems_dir, crepe_model))
        rel_base = save_version(workdir, song["id"], base)
        audio = None
        for c in mine:
            side = judged[c["id"]]["side"] or "a"  # "no difference": either version stands for the pick
            picked, rejected = load_version(workdir, c[side]), load_version(workdir, c["b" if side == "a" else "a"])
            kind = c["kind"] if c["kind"] in KEYS else "notes"
            now = passage(base, c["t0"], c["t1"], kind)
            if same_passage(passage(picked, c["t0"], c["t1"], kind), now):
                kept += 1
            elif judged[c["id"]]["side"] and same_passage(passage(rejected, c["t0"], c["t1"], kind), now):
                ignored += 1
            else:
                window = dict(bars=tuple(c[f"bars_{side}"]), weight=None)
                same_notes = same_passage(
                    passage(picked, c["t0"], c["t1"], "notes"), passage(base, c["t0"], c["t1"], "notes")
                )
                # what differs now decides how the two are compared: the count, the notes, or only the writing
                how = "bars" if not grids_agree(picked, base) else kind if same_notes else "notes"
                comp = add_comparison(
                    comps, song, "change", how, picked, base, c[side], rel_base, window, rng, ("picked", "current")
                )
                if comp:
                    audio = audio or SongAudio(song["path"], cache_dir, device, stems_dir, base_settings["separator"])
                    jobs += render_comparison(comp, picked, base, audio, workdir)
                    new += 1
        save_comparisons(workdir, comps)
    finish_sheets(workdir, comps, jobs, score.find_mscore(mscore), say)
    say(
        f"{kept} judged passages are still written as you picked; {ignored} are written the way you turned down; "
        f"{new} changed and wait for a new pick"
    )
    return kept, ignored, new


def report(workdir):
    """The picks so far, as Markdown."""
    lines = ["# Picks so far", ""]
    t = tally(workdir)
    if not t:
        return "# Picks so far\n\nNo comparisons yet: run `bass2sheet-judge build`.\n"
    for cid, row in t.items():
        lines += [f"## {cid}", "", "| | Picked |", "|---|---|"]
        lines += [f"| {row['labels'].get(v, v)} | {n} |" for v, n in row["picks"].items()]
        lines += [f"| No difference | {row['same']} |", f"| Neither is right | {row['neither']} |"]
        lines += [f"| Not judged yet | {row['open']} |", ""]
        lines += [f"- {n['song']} ({n['id']}), picked {n['pick']}: {n['note']}" for n in row["notes"]]
        lines += [""] if row["notes"] else []
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="bass2sheet-judge", description="Judge the tool's choices on your own songs, by ear and by eye."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="transcribe songs under each alternative and prepare the comparisons")
    b.add_argument("songs", nargs="+", help="audio files you have the rights to use")
    b.add_argument("--per-song", type=int, default=3, help="passages per song and choice (default 3)")
    b.add_argument("--choices", help="comma-separated: " + ", ".join(c["id"] for c in CHOICES) + " (default all)")
    s = sub.add_parser("serve", help="open the judging page on this machine")
    s.add_argument("--port", type=int, default=8765)
    r = sub.add_parser("report", help="print the picks so far")
    u = sub.add_parser("update", help="after the tool changed: bring back judged passages it now writes differently")
    for p in (b, s, r, u):
        p.add_argument("--dir", default="judging", help="working directory (default ./judging)")
    for p in (b, u):
        p.add_argument("--cache-dir", help="where models, stems and pitch tracks are cached")
        p.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
        p.add_argument("--mscore", help="path to the MuseScore 4 binary")
    args = ap.parse_args(argv)
    try:
        if args.command == "build":
            missing = [p for p in args.songs if not os.path.isfile(p)]
            if missing:
                raise Bass2SheetError(f"no such file: {missing[0]}")
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
        else:
            update(args.dir, args.cache_dir, args.device, args.mscore)
    except Bass2SheetError as e:
        print(f"bass2sheet-judge: {e}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
