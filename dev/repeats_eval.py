"""Repeats (repeats.py) against ground truth: what a fix carried to the repeats is worth, and how often it is right.

    python dev/repeats_eval.py [--keep DIR] --dataset DIR [--annotations DIR] ...

For each song the tool's notes are found on the bass separated from the mix (from the cache), and the bars that
sound alike are found in that bass track. Then a person is played by the reference: they go through the song bar
by bar and make each bar right. Printed per set:
  - sixteenths to fix by hand without help, and when every fix (of one note at a time) is carried to the other
    bars that sound the same there and are written as the fixed bar was (`same_mistake`);
  - what the carried changes do, by kind: a wrong sixteenth made right, a right one broken;
  - differences (`differences_of`), a third and two thirds of the way through the song: of the sixteenths marked
    as written unlike their repeats, how many are wrong, and how often the repeats' version is the right one,
    by whose word the repeat is ("fixed", "passed" or "most": the tool against itself) and by kind of difference.
A sixteenth is right when the pitch at its middle is the reference's and a note starts there if and only if one
does in the reference. Uses the cache only: tracks without separated stems are skipped. With --keep DIR what was
measured in each song is kept there, and a second run takes seconds.
"""

import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from bass2sheet import evaluate as ev
from bass2sheet import judge
from bass2sheet.common import sha1_file
from bass2sheet.pipeline import find_notes, seg_params
from bass2sheet.repeats import Repeats, measure

cache = os.path.expanduser("~/.cache/bass2sheet")


def kind(old, new):
    if old[0] == new[0]:
        return "where a note starts"
    if not old[0]:
        return "silence to a note"
    if not new[0]:
        return "a note to silence"
    return "the octave" if (old[0] - new[0]) % 12 == 0 else "another pitch"


def only_lengths(old_p, old_s, new_p, new_s, k0, k1):
    """True when a fix in sixteenths k0..k1-1 of a bar only makes notes shorter or longer: the copilot does not
    offer that to the repeats, so the reference as the person does not carry it either."""
    for k in range(k0, k1):
        if (old_p[k], old_s[k]) == (new_p[k], new_s[k]):
            continue
        for was_p, was_s, now_p in ((old_p, old_s, new_p), (new_p, new_s, old_p)):  # cut short; held longer
            if now_p[k] == 0 and was_p[k] and not was_s[k]:
                a = k
                while a > 0 and not was_s[a] and was_p[a - 1] == was_p[k]:
                    a -= 1
                if (new_p[a], new_s[a]) == (old_p[a], old_s[a]):  # its note starts where it did, as it did
                    break
        else:
            return False
    return True


def play(rep, mine, ref):
    """The reference as the person: bar by bar, and in each bar note by note (a stretch of sixteenths where the
    reference has one note, or silence), they make what is wrong right, and every such fix is carried to the bars
    `same_mistake` names, unless it only made notes shorter or longer. Returns (by hand without help, by hand with fixes carried, carried changes by kind as
    [made right, broke a right one, neither], differences by (why, kind) as [marked, wrong, the repeats'
    version right])."""
    (mp, ms, _), (rp, rs, _) = rep.tokens(mine), rep.tokens(ref)
    plain = int(((mp != rp) | (ms != rs)).sum())
    cur_p, cur_s, level = mp.copy(), ms.copy(), np.zeros_like(mp)
    carried, marks = collections.defaultdict(lambda: [0, 0, 0]), collections.defaultdict(lambda: [0, 0, 0])
    by_hand = 0
    for i in range(rep.bars):
        if i in (rep.bars // 3, 2 * rep.bars // 3):
            for d in rep.differences_of(cur_p, cur_s, level):
                for k in range(d["k0"], d["k1"] + 1):
                    b, s = d["bar"], d["source"]
                    here, there, true = (cur_p[b, k], cur_s[b, k]), (cur_p[s, k], cur_s[s, k]), (rp[b, k], rs[b, k])
                    if here == there:
                        continue
                    where = "a bar already passed" if level[b, k] else "a bar not reached"
                    m = marks[(d["why"], where, kind(here, there))]
                    m[0] += 1
                    m[1] += here != true
                    m[2] += there == true
        edges = [0, *(k for k in range(1, 16) if rp[i, k] != rp[i, k - 1] or rs[i, k]), 16]
        for k0, k1 in zip(edges[:-1], edges[1:]):  # one note of the reference, or one silence
            ks = slice(k0, k1)
            wrong = (cur_p[i, ks] != rp[i, ks]) | (cur_s[i, ks] != rs[i, ks])
            level[i, ks] = np.maximum(level[i, ks], np.where(wrong, 2, 1))
            if not wrong.any():
                continue
            by_hand += int(wrong.sum())
            before = (cur_p.copy(), cur_s.copy())
            cur_p[i, ks], cur_s[i, ks] = rp[i, ks], rs[i, ks]
            if only_lengths(before[0][i], before[1][i], cur_p[i], cur_s[i], k0, k1):
                continue
            for item in rep.same_mistake(before, (cur_p, cur_s), level):
                j = item["bar"]
                for k in range(item["k0"], item["k1"] + 1):
                    old, new, true = (cur_p[j, k], cur_s[j, k]), (cur_p[i, k], cur_s[i, k]), (rp[j, k], rs[j, k])
                    if old == new:
                        continue
                    tally = carried[kind(old, new)]
                    tally[0 if new == true else 1 if old == true else 2] += 1
                    cur_p[j, k], cur_s[j, k] = new
                    level[j, k] = 2
    return plain, by_hand, carried, marks


def inputs(t, keep):
    """(what the bars sound like, grid, origin, the tool's notes, the reference's) for a track; kept in `keep`."""
    path = os.path.join(keep, t["id"] + ".npz") if keep else None
    if path and os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            measured = {k: z[k] for k in ("alike", "sounding", "rise")}
            return measured, z["grid"], int(z["origin"]), z["mine"].tolist(), z["ref"].tolist()
    wav, stems, audio_id, cdir, device = judge.load_song(t["mix"], cache, None, None, "roformer")
    c = find_notes(
        stems, cdir, seg_params("every", "write", lowest=0), device, "full", wav.mean(0), audio_id, "beat_this"
    )
    measured = measure(stems["bass"].mean(0), c["grid"], c["origin"])
    mine = [(n["start"], n["end"], n["pitch"]) for n in c["notes"]]
    truth = ev.reference(t, os.path.join(cache, f"refoct-{t['id']}.json"))["notes"]
    if path:
        os.makedirs(keep, exist_ok=True)
        np.savez_compressed(
            path, grid=np.asarray(c["grid"]), origin=c["origin"], mine=np.array(mine), ref=np.array(truth), **measured
        )
    return measured, c["grid"], c["origin"], mine, truth


args, sets, keep = sys.argv[1:], [], None
while args:
    a = args.pop(0)
    if a == "--keep":
        keep = args.pop(0)
    elif a == "--dataset":
        root, annotations = args.pop(0), None
        if args[:1] == ["--annotations"]:
            args.pop(0)
            annotations = args.pop(0)
        sets.append((root, annotations))

for root, annotations in sets:
    name = root.rstrip("/").split("/")[-1]
    total = dict(plain=0, hand=0, bars=0, repeated=0)
    carried, marks, per_song = (
        collections.defaultdict(lambda: [0, 0, 0]),
        collections.defaultdict(lambda: [0, 0, 0]),
        [],
    )
    for t in ev.load_dataset(root, annotations)[1]:
        if len(t["bass"]) != 1 or not os.path.exists(
            os.path.join(cache, f"stems-{sha1_file(t['mix'])[:16]}-roformer.npz")
        ):
            continue
        measured, grid, origin, mine, truth = inputs(t, keep and os.path.join(keep, name))
        rep = Repeats(measured, grid, origin)
        as_notes = lambda rows: [dict(start=float(s), end=float(e), pitch=int(p)) for s, e, p in rows]
        plain, hand, c_song, m_song = play(rep, as_notes(mine), as_notes(truth))
        for k, v in c_song.items():
            carried[k] = [x + y for x, y in zip(carried[k], v)]
        for k, v in m_song.items():
            marks[k] = [x + y for x, y in zip(marks[k], v)]
        total["plain"] += plain
        total["hand"] += hand
        total["bars"] += rep.bars
        total["repeated"] += int(rep.related.any(1).sum())
        per_song.append(1 - hand / max(1, plain))
        print(
            f"  {t['id']:12s} {rep.bars:4d} bars, {int(rep.family.max()) + 1:3d} patterns | to fix by hand {plain:5d} -> {hand:5d}",
            flush=True,
        )
    print(
        f"\n{name}: {len(per_song)} songs, {total['bars']} bars, {total['repeated'] / max(1, total['bars']):.0%} of them with a repeat"
    )
    print(
        f"  sixteenths to fix by hand: {total['plain']} without help, {total['hand']} with every fix carried to the "
        f"repeats ({1 - total['hand'] / max(1, total['plain']):.0%} fewer); per song the saving is "
        f"{np.percentile(per_song, 10):.0%} / {np.median(per_song):.0%} / {np.percentile(per_song, 90):.0%} (10th pct / median / 90th)"
    )
    for k, (good, bad, neither) in sorted(carried.items(), key=lambda x: -sum(x[1])):
        print(
            f"  carried, {k:22s}: {good + bad + neither:5d} sixteenths; made right {good}, broke a right one {bad}, wrong before and after {neither}"
            f" -> right {good / max(1, good + bad):.0%}"
        )
    for (why, where, k), (n, wrong, right) in sorted(marks.items()):
        print(
            f"  marked unlike its repeats ({why:6s}, {where:20s}, {k:20s}): {n:5d} sixteenths; {wrong / max(1, n):.0%} are wrong; the repeats' version is right for {right / max(1, n):.0%}"
        )
