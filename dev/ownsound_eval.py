"""The check by the bass track's own sound (ownsound.py), against ground truth.

    python dev/ownsound_eval.py [--versions] [--idmt DIR] [--dataset DIR [--annotations DIR]] ...

For each set, the tool's notes are found on the bass track (IDMT: the recordings as they are; the others: the
bass separated from the mix, from the cache) and each note is checked against the song's other notes. The
reference then says which notes are wrong. Printed per set:
  - how well the notes of its own pitch explain a right note and a wrong one, and how well that tells them apart
    (AUC: the chance that a right note scores higher than a wrong one);
  - the notes flagged "another pitch explains it better": how many are wrong, how many name the reference's pitch;
  - across songs, the rank correlation of the share of doubtful notes with the song's true score;
  - with --versions: each song is also transcribed by five other settings of the note finder. Of two versions
    of a song whose true scores differ, does the one with fewer doubtful notes have the better score?
Uses the cache only: tracks without separated stems and pitch probabilities are skipped.
"""

import collections
import csv
import glob
import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr

from bass2sheet import evaluate as ev
from bass2sheet import ownsound
from bass2sheet.audio import decode
from bass2sheet.common import sha1_array, sha1_file
from bass2sheet.notes import SegParams, segment_notes
from bass2sheet.pitch import features

cache = os.path.expanduser("~/.cache/bass2sheet")
device = "mps" if sys.platform == "darwin" else "cpu"


def idmt_tracks(root):
    for wav in sorted(glob.glob(os.path.join(root, "audio", "*.wav"))):
        name = os.path.basename(wav)[:3]
        with open(os.path.join(root, "misc", "notes_csv", f"{name}_note_parameters.csv")) as f:
            rows = [r for r in csv.reader(f) if r]
        yield name, decode(wav).mean(0), [(float(r[0]), float(r[1]), int(r[2])) for r in rows]


def separated_tracks(root, annotations):
    for t in ev.load_dataset(root, annotations)[1]:
        path = os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        if len(t["bass"]) != 1 or not os.path.exists(path):
            continue
        mono = np.load(path)["bass"].mean(0)
        if os.path.exists(os.path.join(cache, f"crepe-{sha1_array(mono)[:16]}.npz")):
            yield t["id"], mono, ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"]


def says(note, ref):
    """What the reference has where our note sounds: (right | octave | other | nothing, its pitch)."""
    mid = (note["start"] + min(note["end"], note["start"] + 0.3)) / 2
    over = [r for r in ref if r[0] - 0.03 <= mid < r[1] + 0.03]
    if not over:
        return "nothing", None
    r = min(over, key=lambda r: abs(r[2] - note["pitch"]))
    d = note["pitch"] - r[2]
    return ("right" if d == 0 else "octave" if abs(d) == 12 else "other"), r[2]


def auc(high, low):
    """The chance that a value from `high` is larger than one from `low` (ties count half)."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    if not len(high) or not len(low):
        return float("nan")
    return float(np.mean([np.mean(h > low) + 0.5 * np.mean(h == low) for h in high]))


VERSIONS = dict(  # other settings of the note finder: what a change to the tool looks like
    clear=dict(strict_replucks=True, split_held=False),
    no_splitting=dict(split_held=False, flux_thr=0.0, flux_support=0.0),
    no_slide_joining=dict(glide_frames=0, thump_frames=0, tail_db=0.0),
    more_short_notes=dict(min_voiced_frames=4),
    stricter_pitch=dict(enter_per=0.6),
)

args, sets, versions = sys.argv[1:], [], False
while args:
    a = args.pop(0)
    if a == "--versions":
        versions = True
    elif a == "--idmt":
        sets.append(("real bass guitar alone (IDMT)", idmt_tracks(args.pop(0)), False))
    elif a == "--dataset":
        root = args.pop(0)
        annotations = args.pop(1) if args[:1] == ["--annotations"] else None
        if annotations:
            args.pop(0)
        sets.append((root.rstrip("/").split("/")[-1], separated_tracks(root, annotations), bool(annotations)))

for name, tracks, by_ear in sets:
    notes_seen, songs, pairs = [], [], []
    key = "f_100ms" if by_ear else "f"
    for _, mono, ref in tracks:
        feat = features(mono, cache, device)
        notes, _, _ = segment_notes(feat, mono, SegParams(lowest=0, together=False))
        mine = [(*says(n, ref), n["pitch"], h) for n, h in zip(notes, ownsound.check(mono, notes)) if h]
        notes_seen += mine
        score = ev.score(ref, [(n["start"], n["end"], n["pitch"]) for n in notes])
        songs.append((score[key], np.mean([ownsound.doubtful(m[3]) for m in mine])))
        if versions:
            same_song = [songs[-1]]
            for kw in VERSIONS.values():
                other, _, _ = segment_notes(feat, mono, SegParams(lowest=0, together=False, **kw))
                heard = [h for h in ownsound.check(mono, other) if h]
                true = ev.score(ref, [(n["start"], n["end"], n["pitch"]) for n in other])[key]
                same_song.append((true, np.mean([ownsound.doubtful(h) for h in heard]) if heard else 1.0))
            pairs += list(itertools.combinations(same_song, 2))
    kinds = collections.defaultdict(list)
    for m in notes_seen:
        kinds[m[0]].append(m)
    explained = lambda k: [m[3]["explained"] for m in kinds[k]]
    print(f"\n{name}: {len(songs)} songs, {len(notes_seen)} notes judged")
    print(f"  right notes ({len(kinds['right'])}): explained {np.median(explained('right')):.2f} (median)")
    for k, label in (("octave", "an octave off"), ("other", "another pitch"), ("nothing", "nothing there")):
        if kinds[k]:
            print(
                f"  {label} ({len(kinds[k])}): explained {np.median(explained(k)):.2f}; "
                f"told from right notes with AUC {auc(explained('right'), explained(k)):.2f}"
            )
    flagged = [m for m in notes_seen if m[3]["better"] is not None]
    wrong = [m for m in notes_seen if m[0] != "right"]
    named = sum(m[3]["better"] == m[1] for m in flagged if m[0] in ("octave", "other"))
    print(
        f"  'another pitch explains it better': {len(flagged)} notes, of which {np.mean([m[0] != 'right' for m in flagged]):.0%} "
        f"are wrong (of all notes: {len(wrong) / len(notes_seen):.0%}); {named} name the reference's pitch; "
        f"they are {np.mean([m[3]['better'] is not None for m in wrong]):.0%} of the wrong notes"
    )
    doubtful = [m for m in notes_seen if ownsound.doubtful(m[3])]
    print(
        f"  doubtful notes (those, and notes explained below {ownsound.UNLIKE}): {len(doubtful)}, of which "
        f"{np.mean([m[0] != 'right' for m in doubtful]):.0%} are wrong; they are "
        f"{np.mean([ownsound.doubtful(m[3]) for m in wrong]):.0%} of the wrong notes and "
        f"{np.mean([ownsound.doubtful(m[3]) for m in kinds['right']]):.0%} of the right ones"
    )
    for gap in (0.03, 0.06) if versions else ():
        apart = [(a, b) for a, b in pairs if abs(a[0] - b[0]) >= gap]
        fewer = sum((a[1] - b[1]) * (a[0] - b[0]) < 0 for a, b in apart)
        print(
            f"  two versions of a song whose true scores differ by {gap} or more: the one with fewer doubtful "
            f"notes is the better one in {fewer} of {len(apart)}"
        )
    s = np.array(songs)
    print(
        f"  per song, share of doubtful notes against the true score: rank correlation {spearmanr(s[:, 1], s[:, 0])[0]:+.2f}"
    )
