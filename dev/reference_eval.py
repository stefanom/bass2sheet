"""How well a file is lined up with a recording (reference.align), measured where the true timing is known.

    python dev/reference_eval.py --dataset DIR [--annotations DIR] ...

Each test song has an answer key that is already in the recording's time. From it a file is made as someone would
have sequenced it, and the step under test has to find the way back:
  - RWC Popular: every note of the arrangement is put on a steady beat (through the annotated beats), 7% faster
    than the recording, after three seconds of nothing. The recording's own tempo wanders; the file's does not.
  - Slakh: the arrangement is slowed by 7% with a tempo that wanders by 3% either way.
Then the same with eight seconds of the song left out of the file, as when a tab has a verse less.

Reported: how many of the answer key's bass notes are placed; how far from their true time (the middle error of a
song, which is a constant offset, and the spread around it); and bass2sheet's score against the key lined up this
way beside its score against the dataset's own key. The numbers are in docs/evaluation.md. Uses the cache: tracks without separated stems are skipped.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pretty_midi

from bass2sheet import evaluate as ev
from bass2sheet import judge, reference
from bass2sheet.common import SR, sha1_file
from bass2sheet.pipeline import find_notes, seg_params

cache = os.path.expanduser("~/.cache/bass2sheet")
args, sets = sys.argv[1:], []
while args:
    a = args.pop(0)
    if a == "--dataset":
        root, annotations = args.pop(0), None
        if args[:1] == ["--annotations"]:
            args.pop(0)
            annotations = args.pop(0)
        sets.append((root, annotations))


def as_sequenced(track, cut=None):
    """(the file, the function from a time of the recording to its time in the file). `cut` is a span of the
    recording, in seconds, that the file leaves out: times in it have no place in the file (NaN)."""
    lead = 3.0
    if track.get("kind") == "rwc":
        beats = np.array(
            [float(row.split(";")[0]) for row in open(track["beats_csv"]) if row.strip() and row[0].isdigit()]
        )
        first, last, each = beats[1] - beats[0], beats[-1] - beats[-2], float(np.median(np.diff(beats))) / 1.07

        def steady(t):
            t = np.asarray(t, dtype=np.float64)
            pos = np.interp(t, beats, np.arange(len(beats)))
            pos = np.where(t < beats[0], (t - beats[0]) / first, pos)
            return np.where(t > beats[-1], len(beats) - 1 + (t - beats[-1]) / last, pos) * each

        source = pretty_midi.PrettyMIDI(track["midi"])
    else:
        steady = lambda t: (
            1.07 * (np.asarray(t, dtype=np.float64) + 0.03 * 40 / (2 * np.pi) * np.sin(2 * np.pi * np.asarray(t) / 40))
        )
        source = pretty_midi.PrettyMIDI(track["all_midi"])

    def to_file(t):
        t = np.asarray(t, dtype=np.float64)
        out = lead + steady(t)
        if cut:
            out = np.where(t >= cut[1], out - (steady(cut[1]) - steady(cut[0])), out)
            out = np.where((t >= cut[0]) & (t < cut[1]), np.nan, out)
        return out

    made = pretty_midi.PrettyMIDI(initial_tempo=120)
    for inst in source.instruments:
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        if inst.notes:
            starts, ends = to_file([n.start for n in inst.notes]), to_file([n.end for n in inst.notes])
            for n, a, b in zip(inst.notes, starts, ends):
                if np.isfinite(a):
                    new.notes.append(
                        pretty_midi.Note(
                            n.velocity, n.pitch, float(a), float(b) if np.isfinite(b) and b > a else float(a) + 0.1
                        )
                    )
        made.instruments.append(new)
    return made, to_file


def lined_up(track, mono, key, cut=None):
    """The answer key's notes as the step places them, with what it took: (notes or None where not placed)."""
    made, to_file = as_sequenced(track, cut)
    al = reference.align(made, mono, SR)
    in_file = to_file([n[0] for n in key])
    placed = np.full(len(key), np.nan)
    there = np.isfinite(in_file)
    placed[there] = al.at(in_file[there])
    return al, placed, there


for root, annotations in sets:
    name = root.rstrip("/").split("/")[-1]
    rows, cuts = [], []
    for t in ev.load_dataset(root, annotations)[1]:
        if len(t["bass"]) != 1 or not os.path.exists(
            os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        ):
            continue
        wav, stems, audio_id, cdir, device = judge.load_song(t["mix"], cache, None, None, "roformer")
        mono = wav.mean(0)
        c = find_notes(stems, cdir, seg_params("every", "write", lowest=0), device, "full", mono, audio_id, "beat_this")
        ours = [(n["start"], n["end"], q) for n in c["notes"] for q in (n["pitch"], *n.get("also", []))]
        key = sorted(ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"])
        true = np.array([n[0] for n in key])
        measure = "f_100ms" if annotations else "f"

        al, placed, there = lined_up(t, mono, key)
        ok = np.isfinite(placed)
        err = placed[ok] - true[ok]
        again = [(float(p), float(p) + (n[1] - n[0]), n[2]) for n, p in zip(key, placed) if np.isfinite(p)]
        rows.append(
            dict(
                placed=ok.mean(),
                middle=float(np.median(err)),
                spread=float(np.median(np.abs(err - np.median(err)))),
                in50=float(np.mean(np.abs(err) < 0.05)),
                in100=float(np.mean(np.abs(err) < 0.1)),
                theirs=ev.score(key, ours).get(measure, 0.0),
                mine=ev.score(again, ours).get(measure, 0.0),
                segments=len(al.segments),
            )
        )
        # the same with eight seconds left out of the file, from a third of the way into the song
        left_out = (true[len(true) // 3], true[len(true) // 3] + 8.0)
        al, placed, there = lined_up(t, mono, key, left_out)
        ok = np.isfinite(placed)
        err = placed[ok & there] - true[ok & there]
        gaps = [(a[1], b[0]) for a, b in zip(al.covered(), al.covered()[1:])]
        found = max((min(g[1], left_out[1]) - max(g[0], left_out[0]) for g in gaps), default=0.0) / 8.0
        cuts.append(
            dict(
                placed=float(ok[there].mean()),
                astray=int(np.sum(np.abs(err) >= 0.1)),
                notes=int(there.sum()),
                found=found,
                wrongly=float(ok[~there].mean()) if (~there).any() else 0.0,
            )
        )
        print(
            f"  {t['id'][-12:]:12s} placed {rows[-1]['placed']:.0%} | middle {rows[-1]['middle'] * 1000:+.0f} ms, spread {rows[-1]['spread'] * 1000:.0f} ms | score {rows[-1]['theirs']:.2f} -> {rows[-1]['mine']:.2f} | cut found {found:.0%}, {cuts[-1]['astray']} notes astray",
            flush=True,
        )
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    gap = np.array([r["mine"] - r["theirs"] for r in rows])
    print(
        f"\n{name}: {len(rows)} songs\n"
        f"  bass notes of the answer key placed: {mean(rows, 'placed'):.1%}\n"
        f"  a song's middle error: {np.median([r['middle'] for r in rows]) * 1000:+.0f} ms in the middle song, "
        f"{min(r['middle'] for r in rows) * 1000:+.0f} to {max(r['middle'] for r in rows) * 1000:+.0f} ms; "
        f"spread around it {mean(rows, 'spread') * 1000:.0f} ms\n"
        f"  notes within 50 ms of their true time {mean(rows, 'in50'):.1%}, within 100 ms {mean(rows, 'in100'):.1%}\n"
        f"  bass2sheet's score against the dataset's key {mean(rows, 'theirs'):.3f}, against the key lined up here "
        f"{mean(rows, 'mine'):.3f}; per song the two differ by {np.abs(gap).mean():.3f} on average, at most {np.abs(gap).max():.3f}\n"
        f"  with eight seconds left out of the file: the gap found in {sum(c['found'] >= 0.7 for c in cuts)} of {len(cuts)} songs; "
        f"of the other notes {mean(cuts, 'placed'):.1%} placed, {sum(c['astray'] for c in cuts)} of {sum(c['notes'] for c in cuts)} "
        f"more than 100 ms astray"
    )
