"""Fast segmentation experiments against ground truth, on stems and pitch tracks already in the cache.

    python dev/seg_eval.py DATASET [--annotations DIR] [--tracks ID,ID] [--cond mix|stem] [name=value ...]

Each `name=value` set overrides SegParams (comma-separated within one argument); several arguments are compared.
Scores are on as-played notes (no grid), so this needs no beat tracking: a few seconds per track.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from bass2sheet import evaluate as ev
from bass2sheet.audio import decode
from bass2sheet.common import sha1_file
from bass2sheet.notes import SegParams, segment_notes
from bass2sheet.pitch import features

args = sys.argv[1:]
dataset = args.pop(0)
tracks, cond, specs, annotations = None, "mix", [], None
while args:
    a = args.pop(0)
    if a == "--tracks":
        tracks = set(args.pop(0).split(","))
    elif a == "--cond":
        cond = args.pop(0)
    elif a == "--annotations":
        annotations = args.pop(0)
    else:
        specs.append(a)
cache = os.path.expanduser("~/.cache/bass2sheet")
rows = {}
feats = {}
for t in ev.load_dataset(dataset, annotations)[1]:
    if len(t["bass"]) != 1 or (tracks and t["id"] not in tracks):
        continue
    if cond == "mix":
        path = os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        if not os.path.exists(path):
            continue
        bass = np.load(path)["bass"]
    else:
        bass = decode(t["bass"][0]["audio"])
    mono = bass.mean(0)
    from bass2sheet.common import sha1_array

    if not os.path.exists(os.path.join(cache, f"crepe-{sha1_array(mono)[:16]}.npz")):
        continue  # pitch tracks not cached yet: skip rather than spend minutes
    ref = ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"]
    feats[t["id"]] = (features(mono, cache), mono, ref)
print(f"{len(feats)} tracks with cached stems and pitch tracks ({cond})")
for spec in specs or [""]:
    p = SegParams()
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(eval(v)))
    ms = []
    for tid, (feat, mono, ref) in feats.items():
        notes, _, _ = segment_notes(feat, mono, p)
        m = ev.score(ref, [(n["start"], n["end"], n["pitch"]) for n in notes])
        m["plucked"] = float(np.mean([n.get("plucked", False) for n in notes]))
        ms.append((tid, m))
    mean = lambda k: float(np.mean([m[k] for _, m in ms if m.get(k) is not None]))
    print(
        f"\n== {spec or 'defaults'}: mean F {mean('f'):.3f}  P {mean('precision'):.3f}  R {mean('recall'):.3f}  F100 {mean('f_100ms'):.3f}  "
        f"F+off {mean('f_with_offsets'):.3f}  bias {mean('onset_bias_ms'):+.0f} ms  plucked {mean('plucked'):.0%}"
    )
    for tid, m in ms:
        re, ee = m["ref_errors"], m["est_errors"]
        print(
            f"  {tid}  F {m['f']:.2f}  P {m['precision']:.2f} R {m['recall']:.2f}  F100 {m['f_100ms']:.2f}  bias {m.get('onset_bias_ms', 0):+4.0f} ±{m.get('onset_spread_ms', 0):3.0f}  "
            f"ref: merged {re['merged']:.0%} covered {re['covered_other']:.0%} timing {re['timing']:.0%} missed {re['missed']:.0%} oct {re['octave']:.0%} pitch {re['wrong_pitch']:.0%} | "
            f"est: split {ee['split']:.0%} ghost {ee['ghost']:.0%} other {ee['extra_other']:.0%}"
        )
