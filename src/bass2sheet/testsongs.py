"""Test songs in the copilot: the songs the tool's scores are measured on, for the person to hear and to say
whether they are like the music they care about.

A score measured on songs unlike yours says little about yours. `add_set` adds the songs of an evaluation dataset
to a copilot working directory as ordinary songs, so each opens in the page with its isolated bass and the notes
the tool hears, and keeps what is known of each in tests.json: which set it is from, what kind of recording that
is, and the score each note finder got on it in `bass2sheet-eval` runs. The page at /tests plays them and takes one
of three answers for each: like my music, partly, not like it. `summary` recounts the scores by answer.

    tests.json      the sets and their songs, written when they are added
    ratings.json    the person's answers and notes, by song
    ratings.jsonl   every answer as it was given, appended
"""

import json
import os
import threading
import time

from . import copilot
from .common import Bass2SheetError, atomic_write_text, clock

PAGE_PATH = os.path.join(os.path.dirname(__file__), "testsongs_page.html")
RATINGS = dict(yes="Like my music", partly="Partly", no="Not like it")
# What each kind of dataset is, in words for someone who has not read about it, and which of bass2sheet-eval's
# scores is its headline (evaluate.py; docs/evaluation.md).
SETS = dict(
    rwc=dict(
        name="RWC Popular",
        about="Songs written and recorded for research around 2000 by studio musicians, in the style of Japanese "
        "chart pop of the 1990s and American chart pop of the 1980s. None was released commercially. The answer "
        "key is a MIDI file of each song, transcribed by ear and lined up with the recording by computer: it can "
        "miss a note or be a little off in time.",
        measure="f_100ms",
    ),
    slakh=dict(
        name="Slakh",
        about="Not recordings: MIDI files of pop songs played by sampled instruments. The answer key is exact, "
        "and the sound is cleaner than a record's.",
        measure="f",
    ),
    mdb=dict(
        name="MDB-bass-synth",
        about="Real multitrack songs whose bass was replaced by a synthesizer that follows its pitch. The answer "
        "key is that pitch curve, not a list of notes.",
        measure="note_acc",
    ),
)
PLAYED = dict(Gt="guitar", Bs="bass", Dr="drums", HH="hi-hat", Org="organ", Pf="piano", Key="keyboard")
_lock = threading.Lock()  # one person, but the page sends an answer and a note close together


def load(workdir):
    return copilot.read_json(os.path.join(workdir, "tests.json"), dict(sets={}, songs={}))


def ratings(workdir):
    return copilot.read_json(os.path.join(workdir, "ratings.json"), {})


def facts(kind, track):
    """What is known of one song that helps to place it, as a few words each."""
    about = track.get("about") or {}
    if kind == "rwc" and about:
        live = [x.strip() for x in about.get("live", "").split(",") if x.strip()]
        out = [about.get("genre", ""), f"{float(about['tempo']):.0f} bpm" if about.get("tempo") else ""]
        out.append("bass played live" if "Bs" in live else "bass not among the instruments played live")
        others = sorted({PLAYED.get(x, x) for x in live if x != "Bs"})
        out.append("played live: " + ", ".join(others) if others else "no instrument played live")
        out.append(about.get("drums", "").lower())
        return [x for x in out if x]
    if kind == "slakh":
        patch = (track["bass"][0].get("patch") or "").rsplit(".", 1)[0].replace("_", " ")
        return [f"sampled bass: {patch}"] if patch else []
    return []


def busy(notes, window=20.0):
    """Where to start listening: the start of the `window` seconds with the most notes in them."""
    starts = sorted(n["start"] for n in notes)
    best, at, j = 0, 0.0, 0
    for i, s in enumerate(starts):
        while starts[j] < s - window:
            j += 1
        if i - j + 1 > best:
            best, at = i - j + 1, starts[j]
    return round(max(0.0, at - 1.0), 1)


def scores_of(run_dir, measure):
    """{track: score} from a bass2sheet-eval run directory: its results from the mix, where there are several."""
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(path):
        raise Bass2SheetError(f"{run_dir} is not a bass2sheet-eval run directory (no results.jsonl)")
    out = {}
    with open(path) as f:
        for row in map(json.loads, filter(str.strip, f)):
            if row.get(measure) is not None and (row.get("condition") == "mix" or row["track"] not in out):
                out[row["track"]] = round(float(row[measure]), 3)
    return out


def add_set(dataset, workdir, annotations=None, runs=None, name=None, only=None, opts=None, stems_dirs=None, say=None):
    """Add a dataset's songs to `workdir` as test songs and prepare each.

    `runs` is {a note finder's name: a bass2sheet-eval run directory}: each song's score under that name is read
    from it. The songs added are `only` (track ids), else those of the first run, else every usable one (a single
    bass part). `name` names the set (by default what kind of dataset it is). `stems_dirs` is as in copilot.add."""
    from . import evaluate, judge

    kind, tracks = evaluate.load_dataset(dataset, annotations)
    info = SETS[kind]
    name = name or info["name"]
    measured = {label: scores_of(run, info["measure"]) for label, run in (runs or {}).items()}
    wanted = set(only) if only else set(next(iter(measured.values()))) if measured else None
    tracks = [t for t in tracks if len(t["bass"]) == 1 and (wanted is None or t["id"] in wanted)]
    if not tracks:
        raise Bass2SheetError(f"none of the songs asked for is in {dataset}")
    os.makedirs(workdir, exist_ok=True)
    opts = dict(copilot.read_json(os.path.join(workdir, "settings.json"), {}), **(opts or {}))
    atomic_write_text(os.path.join(workdir, "settings.json"), json.dumps(opts, indent=1))
    for t in tracks:
        about = t.get("about") or {}
        song_id = judge.slug(f"{name} {t['id']}")
        songs = copilot.load_songs(workdir)
        entry = dict(
            id=song_id,
            path=os.path.abspath(t["mix"]),
            title=about.get("title") or t["id"],
            artist=about.get("artist", ""),
            settings={},
            stems_dir=(stems_dirs or {}).get(t["mix"]),
            test=True,
        )
        songs[song_id] = dict(songs.get(song_id, {}), **entry)
        atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
        copilot.prepare(workdir, songs[song_id], opts, say)
        machine = copilot.read_json(os.path.join(copilot.song_dir(workdir, song_id), "machine.json"))
        with _lock:
            tests = load(workdir)
            tests["sets"][name] = dict(kind=kind, about=info["about"], measure=info["measure"])
            tests["songs"][song_id] = dict(
                set=name,
                track=t["id"],
                facts=facts(kind, t),
                length=machine["length"],
                busy=busy(machine["notes"]),
                scores={label: found.get(t["id"]) for label, found in measured.items()},
            )
            atomic_write_text(os.path.join(workdir, "tests.json"), json.dumps(tests, indent=1))
    return load(workdir)


def rate(workdir, song_id, **change):
    """Keep the person's answer for a test song: `rating` (one of RATINGS, or None to take it back) and `note`
    (their words on what is different). What is not given stays as it was."""
    if song_id not in load(workdir)["songs"]:
        raise Bass2SheetError(f"no test song {song_id}")
    if set(change) - {"rating", "note"} or not change:
        raise Bass2SheetError("an answer is a rating, a note, or both")
    if "rating" in change and change["rating"] is not None and change["rating"] not in RATINGS:
        raise Bass2SheetError(f"a rating is one of {', '.join(RATINGS)}, or nothing")
    if "note" in change and (not isinstance(change["note"], str) or len(change["note"]) > 1000):
        raise Bass2SheetError("a note is a few words")
    with _lock:
        kept = ratings(workdir)
        mine = dict(kept.get(song_id, {}), **change, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
        mine["note"] = (mine.get("note") or "").strip()
        if mine.get("rating") is None and not mine["note"]:
            kept.pop(song_id, None)
        else:
            kept[song_id] = mine
        atomic_write_text(os.path.join(workdir, "ratings.json"), json.dumps(kept, indent=1))
        with open(os.path.join(workdir, "ratings.jsonl"), "a") as f:
            f.write(json.dumps(dict(song=song_id, **change, when=mine["when"])) + "\n")
    return kept.get(song_id, {})


def state(workdir):
    """Everything the page shows: the sets, their songs with the person's answers, and the scores by answer."""
    tests, kept, songs = load(workdir), ratings(workdir), copilot.load_songs(workdir)
    rows, labels = {}, []
    for song_id, t in tests["songs"].items():
        if song_id not in songs:
            continue
        mine = kept.get(song_id, {})
        rows[song_id] = dict(
            t,
            id=song_id,
            title=songs[song_id].get("title", song_id),
            artist=songs[song_id].get("artist", ""),
            rating=mine.get("rating"),
            note=mine.get("note", ""),
        )
        labels += [label for label in t["scores"] if label not in labels]

    def group(of):
        scores = {}
        for label in labels:
            values = [r["scores"][label] for r in of if r["scores"].get(label) is not None]
            scores[label] = round(sum(values) / len(values), 3) if values else None
        return dict(songs=len(of), scores=scores)

    everything = list(rows.values())
    by_answer = [(key, words, [r for r in everything if r["rating"] == key]) for key, words in RATINGS.items()]
    by_answer.append((None, "Not rated yet", [r for r in everything if r["rating"] is None]))
    summary = [dict(group(of), rating=key, words=words) for key, words, of in by_answer]
    summary.append(dict(group(everything), rating="all", words="All of them"))
    sets = []
    for name, s in tests["sets"].items():  # what a kind of set is said to be is this code's, not the file's
        about = SETS[s["kind"]]["about"] if s.get("kind") in SETS else s["about"]
        sets.append(dict(s, name=name, about=about, songs=[i for i, r in rows.items() if r["set"] == name]))
    rated = sum(r["rating"] is not None for r in everything)
    return dict(sets=sets, songs=rows, labels=labels, summary=summary, rated=rated, total=len(everything))


def rate_request(workdir, data):
    """What the page sends when an answer is given; the new state comes back."""
    rate(workdir, data.get("song", ""), **{k: data[k] for k in ("rating", "note") if k in data})
    return state(workdir)


def report(workdir):
    """The scores by the person's answers, as Markdown; nothing where the directory holds no test songs."""
    s = state(workdir)
    if not s["total"]:
        return ""
    per_set = ", ".join(f"{len(x['songs'])} from {x['name']}" for x in s["sets"])
    lines = ["# The test songs, by how like your music they are", ""]
    lines += [f"{s['total']} test songs ({per_set}). You have rated {s['rated']}.", ""]
    lines += ["| | Songs | " + " | ".join(s["labels"]) + " |", "|---|---|" + "---|" * len(s["labels"])]
    for g in s["summary"]:
        cells = ["–" if g["scores"][label] is None else f"{g['scores'][label]:.2f}" for label in s["labels"]]
        lines.append(f"| {g['words']} | {g['songs']} | " + " | ".join(cells) + " |")
    lines.append("")
    noted = [r for r in s["songs"].values() if r["note"]]
    if noted:
        lines += ["What you noted:", ""]
        for r in noted:
            answer = RATINGS.get(r["rating"], "not rated").lower()
            lines.append(f"- {r['title']} ({r['track']}, {answer}, {clock(r['length'])}): {r['note']}")
        lines.append("")
    return "\n".join(lines) + "\n"
