"""Can the original recording tell a string plucked again from a note held, where the isolated bass smears it?

    python dev/repluck_eval.py --dataset DIR [--annotations DIR] ... [--muscriptor KEEP_DIR]

In the reference, a re-pluck is a note that repeats the pitch of the note before with no gap; a held moment is a
point inside a note of half a second or more. Around each, the rise in level is measured (from the quietest
moment in the 100 ms before to the loudest in the 60 ms after), in the isolated bass and in the original
recording, over the whole band and at the note's own partials. The AUC and the share of re-plucks found at a
given share of false alarms say how well each rise tells the two apart. For comparison: how often bass2sheet's
notes have a note of that pitch starting there, and MuScriptor's (from the CSVs dev/muscriptor_eval.py keeps
under KEEP_DIR, for the isolated bass and for the original).

Uses the cache only: tracks without separated stems are skipped.
"""

import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import librosa
import numpy as np

from bass2sheet import evaluate as ev
from bass2sheet import judge
from bass2sheet.common import SR, sha1_file
from bass2sheet.pipeline import find_notes, seg_params

NFFT, HOP = 2048, 256
STEP = HOP / SR
cache = os.path.expanduser("~/.cache/bass2sheet")
args, sets, kept = sys.argv[1:], [], None
while args:
    a = args.pop(0)
    if a == "--dataset":
        root, annotations = args.pop(0), None
        if args[:1] == ["--annotations"]:
            args.pop(0)
            annotations = args.pop(0)
        sets.append((root, annotations))
    elif a == "--muscriptor":
        kept = args.pop(0)


def power(y):
    return np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, window="hann")) ** 2


def at_partials(spec, pitch, partials=(1, 2, 3, 4)):
    """The energy over time at a note's own partials."""
    f0, e = 440 * 2 ** ((pitch - 69) / 12), np.zeros(spec.shape[1])
    for h in partials:
        k = int(round(h * f0 * NFFT / SR))
        if k + 1 < spec.shape[0]:
            e += spec[max(1, k - 1) : k + 2].sum(0)
    return e


def rise(e, t, tol):
    """The largest rise in dB within `tol` of t: from the quietest of the 100 ms before to the loudest of the 60 ms
    after."""
    best = 0.0
    for d in np.arange(-tol, tol + 1e-9, 0.01):
        i = int(round((t + d) / STEP))
        before, after = e[max(0, i - int(0.10 / STEP)) : max(1, i - int(0.01 / STEP))], e[i : i + int(0.06 / STEP) + 1]
        if len(before) and len(after):
            best = max(best, 10 * np.log10((after.max() + 1e-12) / (before.min() + 1e-12)))
    return best


def auc(pos, neg):
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def found(pos, neg, alarms):
    return float(np.mean(np.asarray(pos) > np.quantile(neg, 1 - alarms)))


def starts(notes, t, pitch, tol):
    return any(abs(s - t) <= tol and q == pitch for s, q in notes)


def muscriptors(name, track, source, mode):
    path = os.path.join(kept or "", "medium", mode, f"{name}-{track}-{source}.csv")
    if not kept or not os.path.exists(path):
        return None
    with open(path) as f:
        return [(float(r["start"]), int(r["pitch"])) for r in csv.DictReader(f) if r["instrument"] != "drums"]


MEASURES = (
    "the isolated bass, its whole level",
    "the isolated bass, at the note's partials",
    "the original, under 300 Hz",
    "the original, at the note's partials",
    "both: the isolated bass and the original, at the note's partials",
)
for root, annotations in sets:
    name, tol = root.rstrip("/").split("/")[-1], 0.08 if annotations else 0.03  # RWC is aligned by hand
    rises = {m: ([], []) for m in MEASURES}
    wrote = {m: [0, 0, 0, 0] for m in ("bass2sheet", "MuScriptor on the isolated bass", "MuScriptor on the original")}
    for t in ev.load_dataset(root, annotations)[1]:
        if len(t["bass"]) != 1 or not os.path.exists(
            os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        ):
            continue
        wav, stems, audio_id, cdir, device = judge.load_song(t["mix"], cache, None, None, "roformer")
        p = seg_params("every", "write", lowest=0)
        c = find_notes(stems, cdir, p, device, "full", wav.mean(0), audio_id, "beat_this")
        notes = dict(bass2sheet=[(n["start"], q) for n in c["notes"] for q in (n["pitch"], *n.get("also", []))])
        notes["MuScriptor on the isolated bass"] = muscriptors(name, t["id"], "stem", "bass")
        notes["MuScriptor on the original"] = muscriptors(name, t["id"], "mix", "any")
        ref = sorted(ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"])
        original, bass = power(wav.mean(0)), power(stems["bass"].mean(0))
        again = [
            (b[0], b[2])
            for a, b in zip(ref, ref[1:])
            if b[2] == a[2] and -0.02 <= b[0] - a[1] <= 0.08 and a[1] - a[0] >= 0.1
        ]
        held = [(x, q) for s, e, q in ref if e - s >= 0.5 for x in np.arange(s + 0.2, e - 0.15, 0.125)]
        for k, moments in enumerate((again, held)):
            for x, q in moments:
                v = [
                    rise(bass[1:].sum(0), x, tol),
                    rise(at_partials(bass, q), x, tol),
                    rise(original[1 : int(300 * NFFT / SR)].sum(0), x, tol),
                    rise(at_partials(original, q), x, tol),
                ]
                for m, value in zip(MEASURES, [*v, min(v[1], v[3])]):
                    rises[m][k].append(value)
                for who, theirs in notes.items():
                    if theirs is not None:
                        wrote[who][2 * k] += starts(theirs, x, q, max(tol, 0.05) + 0.02)
                        wrote[who][2 * k + 1] += 1
    n_again, n_held = (len(x) for x in rises[MEASURES[0]])
    print(f"\n{name}: {n_again} notes plucked again on the pitch before, {n_held} moments inside held notes")
    for m, (pos, neg) in rises.items():
        print(
            f"  rise in {m:66s} AUC {auc(pos, neg):.2f} | re-plucks found at 5% false alarms "
            f"{found(pos, neg, 0.05):.0%}, at 3% {found(pos, neg, 0.03):.0%} | median rise {np.median(pos):.1f} dB "
            f"at a re-pluck, {np.median(neg):.1f} dB in a held note"
        )
    for who, (a, b, c_, d) in wrote.items():
        if b:
            print(
                f"  {who:34s} writes a note at {a / b:.0%} of the re-plucks and at {c_ / max(1, d):.1%} of the held moments"
            )
