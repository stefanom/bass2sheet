"""Notes played together, against ground truth.

    python dev/together_eval.py IDMT_DIR [--dataset DIR [--annotations DIR]] ...

Part one: lines made of real bass notes, cut from IDMT-SMT-Bass-Single-Track (see dev/idmt_eval.py) and played
alone, two together (a third to a tenth apart) or an octave apart. An octave must come out as one note. One line
has every note end where the next begins; in the other each rings on under the next, as an open string does.

Part two, for each `--dataset` (BabySlakh, RWC Popular: see docs/evaluation.md): the bass tracks already
separated in the cache. Their references have next to no notes played together, so nearly every pair written
there is a false one. Uses the cache only: tracks without separated stems and pitch probabilities are skipped.
"""

import collections
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from bass2sheet import evaluate as ev
from bass2sheet.audio import decode
from bass2sheet.common import SR, sha1_array, sha1_file
from bass2sheet.notes import SegParams, segment_notes
from bass2sheet.pitch import features

cache = os.path.expanduser("~/.cache/bass2sheet")
device = "mps" if sys.platform == "darwin" else "cpu"
INTERVALS = [3, 4, 5, 7, 8, 9, 10, 15, 16]


def clean_notes(root, shortest=0.28):
    """Finger- and pick-style notes without bends, slides or vibrato, cut out with short fades."""
    out = []
    for wav in sorted(glob.glob(os.path.join(root, "audio", "*.wav"))):
        name = os.path.basename(wav)[:3]
        with open(os.path.join(root, "misc", "notes_csv", f"{name}_note_parameters.csv")) as f:
            rows = [r for r in csv.reader(f) if r]
        y = decode(wav).mean(0)
        for r in rows:
            s, e, pitch = float(r[0]), float(r[1]), int(r[2])
            if r[5] in ("FS", "PK") and r[6] == "NO" and e - s >= shortest:
                seg = y[max(0, int((s - 0.012) * SR)) : int(e * SR)].astype(np.float64).copy()
                seg[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
                out.append(dict(pitch=pitch, audio=seg / (np.abs(seg).max() + 1e-9)))
    return out


def build(notes, seed, events=150, gap=0.75, ring=0.0):
    """(mono line, [(start, end, [pitches])]): an event every `gap` s, each lasting `ring` s into the next."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for n in notes:
        by[n["pitch"]].append(n)
    y, truth = np.zeros(int((events * gap + 2) * SR)), []
    for k in range(events):
        t = 1.0 + k * gap
        kind = rng.choice(["single", "pair", "octave"], p=[0.4, 0.45, 0.15])
        chosen = [notes[rng.integers(len(notes))]]
        for iv in [] if kind == "single" else [12] if kind == "octave" else list(rng.permutation(INTERVALS)):
            pool = by.get(chosen[0]["pitch"] + iv) or by.get(chosen[0]["pitch"] - iv)
            if pool:
                chosen.append(pool[rng.integers(len(pool))])
                break
        dur = min(min(len(c["audio"]) for c in chosen) / SR, gap - 0.08 + ring)
        for c in chosen:
            seg = c["audio"][: int(dur * SR)].copy()
            seg[-int(0.03 * SR) :] *= np.linspace(1, 0, int(0.03 * SR))
            i = int((t - 0.012) * SR)
            y[i : i + len(seg)] += 0.45 * 10 ** (rng.uniform(-3, 3) / 20) * seg
        truth.append((t, t + dur - 0.012, sorted(c["pitch"] for c in chosen)))
    return y.astype(np.float32), truth


args = sys.argv[1:]
idmt = args.pop(0)
datasets = []
while args:
    a = args.pop(0)
    if a == "--dataset":
        datasets.append([args.pop(0), None])
    elif a == "--annotations":
        datasets[-1][1] = args.pop(0)

notes = clean_notes(idmt)
print(f"{len(notes)} real notes to build from")
for name, kw in (
    ("each note ends where the next begins", dict(seed=1)),
    ("each rings on under the next", dict(seed=2, ring=0.25)),
):
    mono, truth = build(notes, **kw)
    feat = features(mono, cache, device)
    print(
        f"\n{name}: {sum(len(p) == 1 for _, _, p in truth)} single notes, "
        f"{sum(len(p) == 2 and p[1] - p[0] != 12 for _, _, p in truth)} pairs, "
        f"{sum(len(p) == 2 and p[1] - p[0] == 12 for _, _, p in truth)} octaves"
    )
    for label, params in (("one note at a time", SegParams(together=False)), ("notes together written", SegParams())):
        written, _, _ = segment_notes(feat, mono, params)
        res = collections.Counter()
        for s, e, ps in truth:
            sets = [sorted([n["pitch"], *n.get("also", [])]) for n in written if s - 0.06 <= n["start"] < e - 0.05]
            if len(ps) == 2 and ps[1] - ps[0] != 12:
                res[
                    "both notes, as one event"
                    if sets == [ps]
                    else "both notes, in pieces"
                    if ps in sets
                    else "a wrong second note"
                    if any(len(x) == 2 for x in sets)
                    else "one of the two"
                    if sets
                    else "nothing"
                ] += 1
            else:
                res["(a note alone or an octave) written as two"] += any(len(x) == 2 for x in sets)
        print(f"  {label}: " + "; ".join(f"{k}: {v}" for k, v in sorted(res.items())))

for root, annotations in datasets:
    pairs = total = 0
    intervals, worst = collections.Counter(), []
    for t in ev.load_dataset(root, annotations)[1]:
        path = os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        if len(t["bass"]) != 1 or not os.path.exists(path):
            continue
        stems = np.load(path)
        mono = stems["bass"].mean(0)
        if not os.path.exists(os.path.join(cache, f"crepe-{sha1_array(mono)[:16]}.npz")):
            continue
        rest = sum(stems[k].mean(0) for k in stems.files if k != "bass")
        written, _, _ = segment_notes(features(mono, cache, device), mono, SegParams(), rest)
        mine = [n for n in written if n.get("also")]
        pairs, total = pairs + len(mine), total + len(written)
        intervals.update(n["also"][0] - n["pitch"] for n in mine)
        worst.append((len(mine), t["id"]))
    print(
        f"\n{root}: {pairs} of {total} notes written as two ({pairs / max(1, total):.2%}); "
        f"intervals {dict(sorted(intervals.items()))}; most in one track: {sorted(worst)[-3:]}"
    )
