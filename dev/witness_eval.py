"""Three witnesses for each of MuScriptor's notes: does the original recording say which of them to doubt?

    python dev/witness_eval.py --muscriptor KEEP_DIR --dataset DIR [--annotations DIR] ...

With --notes muscriptor a note is flagged on the word of two witnesses, both reading the isolated bass:
MuScriptor wrote it and bass2sheet's own finder does not hear it (muscriptor.as_notes). The third would be the
original recording: the rise in level at the note's own partials around its start, asked for together with the
rise in the isolated bass (muscriptor.rises; the smaller of the two is the evidence, the measure that finds the
most re-plucks in dev/repluck_eval.py).

For every note MuScriptor writes this prints how well that rise tells the right notes from the wrong ones, by
what bass2sheet's finder says of the note and by the kind of mistake, and what becomes of the flag under each
rule that uses it: the flag taken off the notes the original supports, put on the notes it does not, and the
rise in place of the pluck where bass2sheet holds the pitch. A flag is measured by the share of the notes it
falls on, the share of the flagged notes that are wrong, and the share of the wrong notes it holds. The notes
themselves are the same under every rule, and so are the note scores.

Uses what dev/muscriptor_eval.py keeps under KEEP_DIR (bass2sheet's notes, MuScriptor's CSVs for the isolated
bass, the isolated bass): nothing is separated and no model runs.
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf

from bass2sheet import evaluate as ev
from bass2sheet import muscriptor
from bass2sheet.audio import decode
from bass2sheet.pipeline import seg_params

cache = os.path.expanduser("~/.cache/bass2sheet")
args, sets, keep = sys.argv[1:], [], None
while args:
    a = args.pop(0)
    if a == "--dataset":
        root, annotations = args.pop(0), None
        if args[:1] == ["--annotations"]:
            args.pop(0)
            annotations = args.pop(0)
        sets.append((root, annotations))
    elif a == "--muscriptor":
        keep = args.pop(0)
if not keep or not sets:
    sys.exit(__doc__)


def auc(pos, neg):
    """How often a value of `pos` is above one of `neg`."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mistake(note, ref, tol):
    """What the reference has where a wrong note is written."""
    t, q = note["start"], note["pitch"]
    near = [(s, e, p) for s, e, p in ref if s - 0.5 <= t <= e + 0.5]
    if any(p == q and s + tol < t < e - 0.02 for s, e, p in near):
        return "the pitch is held there (a re-pluck that was not played)"
    if any(p == q and abs(s - t) <= 0.25 for s, e, p in near):
        return "the note starts up to a quarter of a second away"
    if any((p - q) % 12 == 0 and abs(s - t) <= tol for s, e, p in near):
        return "the note is an octave away"
    if any(abs(s - t) <= tol for s, e, p in near):
        return "another pitch starts there"
    if any(s <= t < e for s, e, p in near):
        return "another pitch is held there"
    return "nothing is played there"


def flag_quality(flag, wrong):
    caught = (flag & wrong).sum()
    return (
        f"on {flag.mean():5.1%} of the notes | {caught / max(1, flag.sum()):5.1%} of them wrong | "
        f"holds {caught / max(1, wrong.sum()):5.1%} of the wrong notes"
    )


p = seg_params("every", "write", lowest=0)
for root, annotations in sets:
    name, tol = root.rstrip("/").split("/")[-1], 0.1 if annotations else 0.05  # RWC is aligned by hand
    rows = []  # per note: wrong, flagged, agree, plucked, other pitch under it, nothing under it, bass, original
    kinds = []
    songs = 0
    for t in ev.load_dataset(root, annotations)[1]:
        kept = sorted(f for f in os.listdir(os.path.join(keep, "bass2sheet")) if f.startswith(t["id"] + "-"))
        path = os.path.join(keep, "medium", "bass", f"{name}-{t['id']}-stem.csv")
        stem = os.path.join(keep, "stems", f"{t['id']}.wav")
        if not kept or not os.path.exists(path) or not os.path.exists(stem):
            continue
        with open(os.path.join(keep, "bass2sheet", kept[0])) as f:
            own = [dict(start=s, end=e, pitch=q) for s, e, q in json.load(f)]
        with open(path) as f:
            raw = [
                (float(r["start"]), float(r["end"]), int(r["pitch"]))
                for r in csv.DictReader(f)
                if r["instrument"] != "drums"
            ]
        bass = sf.read(stem, dtype="float32", always_2d=True)[0].mean(1)
        ref = sorted(ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"])
        notes = muscriptor.as_notes(raw, own, bass, p)
        songs += 1
        for nt, (in_bass, in_original) in zip(notes, muscriptor.rises(notes, bass, decode(t["mix"]).mean(0))):
            under = [o["pitch"] for o in own if o["start"] <= nt["start"] + 0.03 < o["end"]]
            right = any(q == nt["pitch"] and abs(s - nt["start"]) <= tol for s, _, q in ref)
            rows.append(
                (
                    not right,
                    nt["flag"] == muscriptor.ALONE,
                    nt["agree"],
                    nt["plucked"],
                    any(q != nt["pitch"] for q in under),
                    not under,
                    in_bass,
                    in_original,
                )
            )
            kinds.append("" if right else mistake(nt, ref, tol))
    if not rows:
        continue
    v, kinds = np.array(rows, dtype=float), np.array(kinds)
    wrong, flag, agree, plucked, other, nothing = (v[:, k] > 0 if k != 2 else v[:, k] for k in range(6))
    both = np.minimum(v[:, 6], v[:, 7])  # both must rise
    print(f"\n{name}: {songs} songs, {len(v)} notes of MuScriptor's, {wrong.mean():.1%} of them wrong")
    print(f"  the flag as it is: {flag_quality(flag, wrong)}")
    print("\n  How well the rise tells the right notes from the wrong (1.0: always, 0.5: not at all)")
    groups = {
        "bass2sheet's finder has the note too": agree == 1.0,
        "it holds that pitch, and a pluck is heard": (agree == 0.5) & plucked,
        "it holds that pitch, no pluck (flagged)": (agree == 0.5) & ~plucked,
        "it hears another pitch (flagged)": (agree == 0.0) & other,
        "it hears nothing (flagged)": (agree == 0.0) & nothing,
    }
    for label, on in groups.items():
        r, w = on & ~wrong, on & wrong
        print(
            f"    {label:42s} {on.sum():6d} notes, {w.sum() / max(1, on.sum()):5.1%} wrong | {auc(both[r], both[w]):.2f} | "
            f"median rise {np.median(both[r]) if r.any() else 0:4.1f} dB where right, "
            f"{np.median(both[w]) if w.any() else 0:4.1f} dB where wrong"
        )
    print("\n  The wrong notes by what the reference has there, each against all the right notes")
    for kind in sorted(set(kinds) - {""}, key=lambda k: -(kinds == k).sum()):
        on = kinds == kind
        print(
            f"    {kind:58s} {on.sum():5d} ({on.sum() / wrong.sum():4.0%} of the wrong), {flag[on].mean():4.0%} flagged | "
            f"{auc(both[~wrong], both[on]):.2f} | median rise {np.median(both[on]):4.1f} dB"
        )
    print(f"    {'the right notes':58s} {(~wrong).sum():5d} | median rise {np.median(both[~wrong]):4.1f} dB")
    print("\n  The flag under each rule")
    rules = {"as it is (two witnesses)": flag}
    for db in (6, 12, 16):
        rules[f"taken off where both rise by {db} dB or more"] = flag & (both < db)
    for db in (1, 2):
        rules[f"also put on where they rise by less than {db} dB"] = flag | (both < db)
    for db in (2, 4):
        rules[f"a held pitch: the rise (under {db} dB) for the pluck"] = np.where(agree == 0.5, both < db, flag)
    for db in (6, 12):
        rules[f"all bass2sheet lacks, unless both rise by {db} dB"] = (agree < 1.0) & (both < db)
    for label, on in rules.items():
        print(f"    {label:52s} {flag_quality(on, wrong)}")
        if label.startswith("as it is"):
            continue
        for change, moved in (("flag taken off", flag & ~on), ("flag put on", on & ~flag)):
            if moved.any():
                print(f"    {'':52s}   {change}: {moved.sum()} notes, {wrong[moved].mean():.0%} of them wrong")
