"""bass2sheet's note finding against MuScriptor's (Rouard et al., 2026: a transformer by Kyutai and Mirelo that
writes every instrument's notes straight from audio), on the same recordings and scored the same way.

    python dev/muscriptor_eval.py [--size medium] [--ask any|bass] [--keep DIR]
        [--idmt DIR] [--dataset DIR [--annotations DIR]] ...

MuScriptor is run one of two ways, chosen by the environment:
  - MUSCRIPTOR_NOTES=<the muscriptor_notes binary> and MUSCRIPTOR_GGUF=<directory of muscriptor-SIZE-f16.gguf>:
    its C++ port, muscriptor.cpp, with the converted weights (dev/muscriptor_cpp);
  - MUSCRIPTOR_PYTHON=<interpreter of an environment with the muscriptor package>: the reference code, which
    fetches the original weights once their conditions are accepted on Hugging Face (dev/muscriptor_runner.py).
The weights are under CC BY-NC 4.0 either way. The tool itself does not use MuScriptor.

bass2sheet separates the bass from the mix and finds the notes in it; MuScriptor writes every instrument's notes
from the mix itself and says which instrument each belongs to. So it is asked three ways:
  - the mix, and the notes it calls bass are taken;
  - the bass as bass2sheet isolated it, and the notes it calls bass are taken;
  - the bass as bass2sheet isolated it, told to write the bass and nothing else (its instrument conditioning).
On the IDMT lines there is no mix: the bass alone goes in, with and without being told it is a bass.
(Told to write only the bass, a full mix comes out as thousands of notes: every sound in it written as bass.
The conditioning says which instruments are there, it does not pick one out. That way of asking is left out.)

Scores are those of `bass2sheet-eval` (evaluate.score: a note is right with its pitch and its start within 50 ms).
MuScriptor's times can all lag by a few hundredths of a second, which its own MIDI export takes out with the beat
grid; here each track is also scored with the median lag of its matched notes removed, for both systems.
Uses the cache only for bass2sheet's side: tracks without separated stems are skipped. MuScriptor's notes are kept
under --keep (default ~/.cache/bass2sheet/muscriptor), so a second run only scores.
"""

import csv
import glob
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf

from bass2sheet import evaluate as ev
from bass2sheet import judge
from bass2sheet.audio import decode
from bass2sheet.common import SR, sha1_file
from bass2sheet.notes import SegParams, segment_notes
from bass2sheet.pipeline import find_notes, seg_params
from bass2sheet.pitch import features

cache = os.path.expanduser("~/.cache/bass2sheet")
device = "mps" if sys.platform == "darwin" else "cpu"
BASS = ("electric_bass", "acoustic_bass")
args, size, keep, sets, ask = sys.argv[1:], "medium", os.path.join(cache, "muscriptor"), [], ("any", "bass")
while args:
    a = args.pop(0)
    if a == "--size":
        size = args.pop(0)
    elif a == "--ask":  # only one of the two ways of asking: every instrument (any) or the bass alone (bass)
        ask = (args.pop(0),)
    elif a == "--keep":
        keep = args.pop(0)
    elif a == "--idmt":
        sets.append(("idmt", args.pop(0), None))
    elif a == "--dataset":
        root, annotations = args.pop(0), None
        if args[:1] == ["--annotations"]:
            args.pop(0)
            annotations = args.pop(0)
        sets.append(("mix", root, annotations))
python, binary, weights = (os.environ.get(k) for k in ("MUSCRIPTOR_PYTHON", "MUSCRIPTOR_NOTES", "MUSCRIPTOR_GGUF"))
if not (binary and weights) and not python:
    sys.exit("set MUSCRIPTOR_NOTES and MUSCRIPTOR_GGUF, or MUSCRIPTOR_PYTHON: see the top of this file")


def as_the_port_reads_it(source, raw):
    """The recording as raw 32-bit floats, mono, 16 kHz: what muscriptor.cpp takes."""
    import librosa

    y, sr = sf.read(source, dtype="float32", always_2d=True)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    librosa.resample(y.mean(1), orig_sr=sr, target_sr=16000, res_type="soxr_hq").astype("<f4").tofile(raw)


def tracks_of(kind, root, annotations):
    """dict(id, ours, ref or f0, inputs={what goes into MuScriptor: path}) per track."""
    if kind == "idmt":
        for wav in sorted(glob.glob(os.path.join(root, "audio", "*.wav"))):
            name = os.path.basename(wav)[:3]
            with open(os.path.join(root, "misc", "notes_csv", f"{name}_note_parameters.csv")) as f:
                ref = [(float(r[0]), float(r[1]), int(r[2])) for r in csv.reader(f) if r]
            mono = decode(wav).mean(0)
            ours = [
                (n["start"], n["end"], n["pitch"])
                for n in segment_notes(features(mono, cache, device), mono, SegParams())[0]
            ]
            yield dict(id=name, ours=ours, ref=ref, f0=None, inputs=dict(alone=wav))
        return
    for t in ev.load_dataset(root, annotations)[1]:
        if len(t["bass"]) != 1 or not os.path.exists(
            os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        ):
            continue
        stem = os.path.join(keep, "stems", f"{t['id']}.wav")
        mine = os.path.join(keep, "bass2sheet", f"{t['id']}-{judge.code_id()}.json")  # what this code hears, kept
        if not (os.path.exists(stem) and os.path.exists(mine)):
            wav, stems, audio_id, cdir, dev = judge.load_song(t["mix"], cache, None, None, "roformer")
            c = find_notes(
                stems, cdir, seg_params("every", "write", lowest=0), dev, "full", wav.mean(0), audio_id, "beat_this"
            )
            os.makedirs(os.path.dirname(stem), exist_ok=True)
            os.makedirs(os.path.dirname(mine), exist_ok=True)
            sf.write(stem, stems["bass"].T, SR, subtype="PCM_16")
            with open(mine, "w") as f:
                json.dump([(n["start"], n["end"], q) for n in c["notes"] for q in (n["pitch"], *n.get("also", []))], f)
        with open(mine) as f:
            ours = [tuple(n) for n in json.load(f)]
        truth = ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))
        yield dict(id=t["id"], ours=ours, ref=truth["notes"], f0=truth["f0"], inputs=dict(mix=t["mix"], stem=stem))


# what is asked of MuScriptor: (label, which input, which notes it may write, which of its notes count)
ASKED = dict(
    idmt=[
        ("the bass alone, every note it writes", "alone", "any", None),
        ("the bass alone, told to write only the bass", "alone", ",".join(BASS), None),
    ],
    mix=[
        ("the mix, the notes it calls bass", "mix", "any", BASS),
        ("our isolated bass, the notes it calls bass", "stem", "any", BASS),
        ("our isolated bass, told to write only the bass", "stem", ",".join(BASS), None),
    ],
)


def one_line(notes):
    """A bass plays a pitch once at a time: where MuScriptor writes the same note for two instruments (the
    electric and the acoustic bass, which it does now and then when both are allowed), one is kept."""
    kept = []
    for s, e, p in sorted(notes):
        if not any(q == p and abs(s - a) <= 0.05 for a, _, q in kept[-8:]):
            kept.append((s, e, p))
    return kept


def csv_path(set_name, track, source, may):
    return os.path.join(keep, size, "any" if may == "any" else "bass", f"{set_name}-{track}-{source}.csv")


def measure(t, est):
    """dict(score, shifted, precision, recall, octave, lag) for one track; `score` is the set's headline."""
    if t["ref"] is None:
        s = ev.score_f0(t["f0"][0], t["f0"][1], est)
        return dict(
            score=s["note_acc"], shifted=s["note_acc"], precision=None, recall=None, octave=s["octave_err"], lag=0.0
        )
    s = ev.score(t["ref"], est)
    lag = s.get("onset_bias_ms", 0.0) or 0.0
    back = lambda x: max(x - lag / 1000, 0.0)
    moved = ev.score(t["ref"], [(back(a), max(back(b), back(a) + 0.01), p) for a, b, p in est])
    return dict(  # a transcription with no bass in it has none of these: it scores nothing
        score=s.get(t["key"], 0.0),
        shifted=moved.get(t["key"], 0.0),
        precision=s.get("precision", 0.0),
        recall=s.get("recall", 0.0),
        octave=(s.get("pitch_octave_up", 0) or 0) + (s.get("pitch_octave_down", 0) or 0),
        lag=lag,
    )


loaded = []
for kind, root, annotations in sets:
    name = "idmt" if kind == "idmt" else root.rstrip("/").split("/")[-1]
    tracks = list(tracks_of(kind, root, annotations))
    for t in tracks:
        t["key"] = "f_100ms" if annotations else "f"  # RWC's references are aligned by hand: 100 ms
    loaded.append((kind, name, tracks))
    print(f"{name}: {len(tracks)} tracks ready", flush=True)

for may in ("any", ",".join(BASS)):  # the model is loaded once for each way of asking
    if ("any" if may == "any" else "bass") not in ask:
        continue
    jobs = []
    for kind, name, tracks in loaded:
        for _, source, asked, _ in ASKED[kind]:
            if asked == may:
                jobs += [x for t in tracks for x in (t["inputs"][source], csv_path(name, t["id"], source, may))]
    todo = [b for b in jobs[1::2] if not os.path.exists(b)]
    if todo:
        for b in todo:
            os.makedirs(os.path.dirname(b), exist_ok=True)
        print(f"MuScriptor {size}, allowed to write {may}: {len(todo)} files to transcribe", flush=True)
        if binary and weights:
            pairs = []
            for source, out in zip(jobs[::2], jobs[1::2]):
                if not os.path.exists(out):
                    raw = os.path.join(keep, "f32", os.path.basename(out)[:-4] + ".f32")
                    if not os.path.exists(raw):
                        as_the_port_reads_it(source, raw)
                    pairs += [raw, out]
            subprocess.run([binary, os.path.join(weights, f"muscriptor-{size}-f16.gguf"), may, *pairs], check=True)
        else:
            runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muscriptor_runner.py")
            subprocess.run([python, runner, size, may, *jobs], check=True)

for kind, name, tracks in loaded:
    rows = {"bass2sheet": [measure(t, t["ours"]) for t in tracks]}
    for label, source, may, counted in ASKED[kind]:
        if ("any" if may == "any" else "bass") not in ask:
            continue
        got = []
        for t in tracks:
            with open(csv_path(name, t["id"], source, may)) as f:
                notes = [r for r in csv.DictReader(f) if r["instrument"] != "drums"]
            mine = [r for r in notes if not counted or r["instrument"] in counted]
            got.append(measure(t, one_line((float(r["start"]), float(r["end"]), int(r["pitch"])) for r in mine)))
        rows[f"MuScriptor {size}: {label}"] = got
    what = "note accuracy" if tracks[0]["ref"] is None else "F at 100 ms" if tracks[0]["key"] == "f_100ms" else "F"
    print(f"\n{name}: {len(tracks)} tracks, {what}")
    ours = np.array([m["score"] for m in rows["bass2sheet"]])
    for label, got in rows.items():
        s, moved = np.array([m["score"] for m in got]), np.array([m["shifted"] for m in got])
        line = f"  {label}\n      mean {s.mean():.3f}, median {np.median(s):.3f}, lowest {s.min():.3f}"
        if got[0]["precision"] is not None:
            line += (
                f" | precision {np.mean([m['precision'] for m in got]):.3f}, recall {np.mean([m['recall'] for m in got]):.3f}"
                f" | with the lag removed (median {np.median([m['lag'] for m in got]):+.0f} ms): {moved.mean():.3f}"
            )
        line += f" | an octave off: {np.mean([m['octave'] for m in got]):.1%}"
        if label != "bass2sheet":
            line += f" | better than bass2sheet by 0.02 or more on {int((s - ours >= 0.02).sum())} tracks, worse on {int((ours - s >= 0.02).sum())}"
        print(line)
    print("  per track (bass2sheet | " + " | ".join(label.split(": ")[-1] for label in list(rows)[1:]) + "):")
    for k, t in enumerate(tracks):
        print(f"    {t['id'][-28:]:28s} " + " | ".join(f"{got[k]['score']:.2f}" for got in rows.values()))
