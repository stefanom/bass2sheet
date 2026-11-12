"""Note finding on real bass guitar played alone: IDMT-SMT-Bass-Single-Track (17 lines, 948 notes with exact starts,
ends and pitches, by plucking style; https://zenodo.org/records/7544099, CC BY-NC-ND 4.0).

    python dev/idmt_eval.py IDMT_DIR [name=value,... ...] [--basic-pitch] [--yourmt3]

Each `name=value,...` argument is a set of SegParams overrides to compare with the defaults. `--basic-pitch` and
`--yourmt3` add those models, when their environments are there (see second_opinion.py and dev/mt3_runner.py:
YourMT3+ needs YOURMT3_PYTHON and YOURMT3_HOME). There is no mix here and no separation: this measures the note
finder alone, on what a clean bass track gives it.
"""

import collections
import csv
import glob
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from bass2sheet import evaluate as ev
from bass2sheet.audio import decode
from bass2sheet.notes import SegParams, segment_notes
from bass2sheet.pitch import features

STYLES = dict(FS="finger", MU="muted", PK="pick", ST="slap, thumb", SP="slap, pop")
cache = os.path.expanduser("~/.cache/bass2sheet")
args = sys.argv[1:]
root = args.pop(0)
with_bp, with_mt3 = "--basic-pitch" in args, "--yourmt3" in args
specs = [a for a in args if not a.startswith("--")]

tracks = []
for wav in sorted(glob.glob(os.path.join(root, "audio", "*.wav"))):
    name = os.path.basename(wav)[:3]
    with open(os.path.join(root, "misc", "notes_csv", f"{name}_note_parameters.csv")) as f:
        rows = [r for r in csv.reader(f) if r]  # onset, offset, pitch, string, fret, plucking style, expression, ...
    bass = decode(wav)
    tracks.append(dict(id=name, wav=wav, rows=rows, bass=bass, mono=bass.mean(0)))
    tracks[-1]["ref"] = [(float(r[0]), float(r[1]), int(r[2])) for r in rows]
    tracks[-1]["feat"] = features(tracks[-1]["mono"], cache, "mps" if sys.platform == "darwin" else "cpu")


def ours(params):
    def fn(t):
        return [(n["start"], n["end"], n["pitch"]) for n in segment_notes(t["feat"], t["mono"], params)[0]]

    return fn


methods = {"bass2sheet": ours(SegParams())}
for spec in specs:
    p = SegParams()
    for kv in spec.split(","):
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(eval(v)))
    methods[spec] = ours(p)
if with_bp:
    from bass2sheet.second_opinion import basic_pitch_notes

    methods["Basic Pitch"] = lambda t: [
        (s, e, p) for s, e, p, _ in basic_pitch_notes(t["bass"], t["feat"]["stem_id"], cache)[0]
    ]
if with_mt3:
    out_dir = tempfile.mkdtemp()
    jobs = [x for t in tracks for x in (t["wav"], os.path.join(out_dir, t["id"] + ".csv"))]
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mt3_runner.py")
    device = "mps" if sys.platform == "darwin" else "cpu"
    subprocess.run([os.environ["YOURMT3_PYTHON"], runner, device, *jobs], check=True, capture_output=True)

    def mt3(t):
        with open(os.path.join(out_dir, t["id"] + ".csv")) as f:
            return [
                (float(r["start"]), float(r["end"]), int(r["pitch"])) for r in csv.DictReader(f) if r["is_drum"] == "0"
            ]

    methods["YourMT3+"] = mt3


def found(ref_note, est, tol=0.05):
    return any(abs(s - ref_note[0]) <= tol and p == ref_note[2] for s, _, p in est)


print(f"{len(tracks)} lines, {sum(len(t['ref']) for t in tracks)} notes")
for label, fn in methods.items():
    scores, by_style, jump = [], collections.defaultdict(lambda: [0, 0]), [0, 0]
    for t in tracks:
        est = fn(t)
        scores.append(ev.score(t["ref"], est))
        before = None
        for note, row in zip(t["ref"], t["rows"]):
            if row[6] != "DN":  # dead notes have no pitch to find
                hit = found(note, est)
                by_style[row[5]][0] += hit
                by_style[row[5]][1] += 1
                if before is not None and note[2] - before >= 7:
                    jump[0] += hit
                    jump[1] += 1
            before = note[2]
    mean = lambda k: float(np.mean([s[k] for s in scores]))
    err = lambda side, k: float(np.mean([s[side][k] for s in scores]))
    print(
        f"\n{label}\n  F {mean('f'):.3f}  precision {mean('precision'):.3f}  recall {mean('recall'):.3f}  "
        f"F at 100 ms {mean('f_100ms'):.3f}\n"
        f"  reference notes: missed {err('ref_errors', 'missed'):.0%}, under another note "
        f"{err('ref_errors', 'covered_other'):.0%}, merged {err('ref_errors', 'merged'):.0%}, an octave off "
        f"{err('ref_errors', 'octave'):.0%}, other wrong pitch {err('ref_errors', 'wrong_pitch'):.0%}\n"
        f"  our extra notes: re-attacks {err('est_errors', 'split'):.0%}, other {err('est_errors', 'extra_other'):.0%}\n"
        f"  right note at the right time after a jump up of a fifth or more: {jump[0]}/{jump[1]} = {jump[0] / jump[1]:.0%}\n"
        "  by plucking style: "
        + ", ".join(f"{STYLES[k]} {h}/{n} = {h / n:.0%}" for k, (h, n) in sorted(by_style.items()))
    )
