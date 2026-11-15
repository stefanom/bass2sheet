"""How right the chords and the key are (harmony.find), against RWC Popular's chord annotations; and where the
finder's weights come from.

    python dev/chords_eval.py --audio RWC_DIR --annotations RWC_ANNOTATIONS [--songs check|held-out|RWC_P001,...]
                              [--slakh BABYSLAKH_DIR] [--ingredients] [--cv] [--by-hand] [--weights FILE.npz]
                              [--beats librosa] [--separate] [--prepare-only] [--keep DIR]
    python dev/chords_eval.py --slakh BABYSLAKH_DIR --fit

**Measuring.** For every song the finder's chords are scored with mir_eval.chord, beside the chord per bar the
tool wrote before (rhythm.chords_per_bar, one triad a bar) on the same bar times: the tool's own beat grid, not
a constant tempo. Reported per song and overall: root, major/minor, sevenths (root and quality right over maj,
min, 7, maj7, min7), the bass note under the chord, and how well the chord changes fall where the reference's
do. Then how a chord's score relates to its being right, and the key against keys read by hand from the
annotated chords (see KEYS). The 24 `check` songs are the ones with separated stems in the cache; nothing the
finder ships with was fitted to them. `held-out` names 25 others (every third of the other 76), for a
measurement on songs no choice was made on.

**Fitting.** `--slakh DIR --fit` fits the weights of the match on BabySlakh and prints them for harmony.py, with
the table that turns a chord's likelihood into its score. BabySlakh has no chord annotations but it has every
instrument's MIDI: the chord of each eighth is read from the notes themselves (`midi_classes`), and the weights
are fitted so that the audio says the same. The audio is what the tool makes of the mix: its separated stems,
its bass notes, its beat grid.

**What each part is worth.** `--ingredients` (with `--slakh`) fits and measures the finder without each of its
parts in turn. `--cv` fits the weights on the RWC songs themselves, four folds, each song scored by weights
fitted on the others: what annotated real pop would buy. `--by-hand` uses weights set without any data.
`--weights` measures weights from a file (as `np.savez(weights=, biases=)`). `--beats librosa` measures on the
grid of the built-in beat tracker, in place of the Beat This! model's.

A song needs its separated stems, pitch track and beats in the cache. `--separate` makes them for the songs
that lack them (a few minutes a song on a GPU); `--prepare-only` stops there. What was found is kept under
`--keep` (default: chords-eval in the cache), so a second run does not find the notes again and a run that was
stopped goes on where it was. The numbers are in docs/evaluation.md.
"""

import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import mir_eval
import numpy as np

from bass2sheet import evaluate, harmony, judge, rhythm
from bass2sheet.common import FLATS, cache_home, sha1_file
from bass2sheet.pipeline import find_notes, seg_params

CHECK = [
    f"RWC_P{n:03d}"
    for n in (1, 7, 11, 17, 20, 24, 27, 33, 36, 39, 44, 47, 51, 55, 58, 61, 64, 68, 81, 84, 88, 94, 97, 100)
]
HELD_OUT = [f"RWC_P{n:03d}" for n in range(1, 101) if f"RWC_P{n:03d}" not in CHECK][1::3]
MEASURES = ("root", "majmin", "sevenths", "thirds", "triads", "tetrads", "mirex")
DEGREE = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}
HARTE = {"": "maj", "m": "min", "7": "7", "maj7": "maj7", "m7": "min7"}  # our qualities as the annotations name them

# The main key of each check song, (tonic, is major), read by hand from its annotated chords: which chords take
# the most time in each part of the song, and how it starts and ends. Where the MIDI file of the song has a key
# signature other than the default C major (10 of the 24), it agrees. They were written down before any key
# the finder gave was looked at.
KEYS = {
    "RWC_P001": (8, False),
    "RWC_P007": (1, False),
    "RWC_P011": (2, False),
    "RWC_P017": (0, True),
    "RWC_P020": (5, True),
    "RWC_P024": (3, True),
    "RWC_P027": (9, False),
    "RWC_P033": (9, False),
    "RWC_P036": (2, False),
    "RWC_P039": (2, True),
    "RWC_P044": (3, True),
    "RWC_P047": (0, True),
    "RWC_P051": (2, True),
    "RWC_P055": (0, True),
    "RWC_P058": (6, False),
    "RWC_P061": (4, False),
    "RWC_P064": (0, True),
    "RWC_P068": (2, True),
    "RWC_P081": (5, True),
    "RWC_P084": (4, True),
    "RWC_P088": (0, True),
    "RWC_P094": (1, True),
    "RWC_P097": (9, False),
    "RWC_P100": (0, True),
}
STEPS = (4, 16)  # the other frame lengths --ingredients tries, in sixteenths: a beat, a bar


# --------------------------------------------------------------------------------------------------- the songs


def reference(annotations, rid):
    """(intervals, labels) of a song's annotated chords."""
    path = os.path.join(annotations, "01_annotations_preprocessed", "chords", "RWC-P", rid + ".csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return np.array([[float(r["t_start"]), float(r["t_end"])] for r in rows]), [r["chord"] for r in rows]


def prepare(rid, path, cache, keep, separate, device, beats="beat_this"):
    """What the tool hears in a song (`path` is its mix): its notes, its grid by the beat tracker `beats`, and
    the chord per bar and the key it wrote before there was harmony.py. From `keep` where it was found before.
    None when the song's stems are not in the cache and `separate` is off."""
    kept = os.path.join(keep, rid + ("" if beats == "beat_this" else "-" + beats) + ".json")
    if os.path.exists(kept):
        with open(kept) as f:
            return dict(json.load(f), id=rid, path=path)
    if not separate and not os.path.exists(os.path.join(cache, f"stems-{sha1_file(path)[:16]}-roformer.npz")):
        return None
    began = time.time()
    wav, stems, audio_id, cdir, device = judge.load_song(path, cache, device, None, "roformer")
    c = find_notes(stems, cdir, seg_params("every", "write"), device, "full", wav.mean(0), audio_id, beats)
    found = dict(
        notes=[dict(start=n["start"], end=n["end"], pitch=n["pitch"], also=n.get("also", [])) for n in c["notes"]],
        grid=[float(t) for t in c["grid"]],
        origin=int(c["origin"]),
        beat_tracker=c["beat_tracker"],
        seconds=round(time.time() - began, 1),
    )
    old, chroma = rhythm.chords_per_bar(stems["harmony"], c["notes"], c["grid"], c["origin"])
    found["old_chords"] = [dict(bar=o["bar"], root=o["root"], quality=o["quality"]) for o in old]
    found["old_key"] = list(rhythm.estimate_key(chroma))
    os.makedirs(keep, exist_ok=True)
    with open(kept, "w") as f:
        json.dump(json.loads(json.dumps(found, default=float)), f)
    return dict(found, id=rid, path=path)


def listen(song, cache, more=False, steps=()):
    """What harmony.hear makes of a prepared song, and the same in frames of other lengths (`by_step`). With
    `more`, two things more to match a chord on, which --ingredients tries: `sung`, the notes of the voice, and
    `notes_half_bar`, the instruments' notes over the half bar each frame is in."""
    stems = judge.load_song(song["path"], cache, "cpu", None, "roformer")[1]
    song["length"] = stems["harmony"].shape[-1] / harmony.SR
    song["heard"] = heard = harmony.hear(stems, song["notes"], song["grid"], song["origin"])
    if more:
        per_frame = lambda mono: harmony._per_frame(harmony.chroma(harmony.spectrum(mono)), heard["edges"])
        notes, half_bar = per_frame(stems["harmony"].mean(0)), (heard["index"] - song["origin"]) // 8
        pooled = np.array([notes[half_bar == h].mean(0) for h in half_bar])
        heard["groups"].update(
            sung=harmony._unit(per_frame(stems["vocals"].mean(0))), notes_half_bar=harmony._unit(pooled)
        )
    song["by_step"] = {}
    for step in steps:
        with changed(STEP=step):
            song["by_step"][step] = harmony.hear(stems, song["notes"], song["grid"], song["origin"])
    judge._song.cache_clear()  # the stems are large: one song in memory at a time
    return song


def rwc_songs(ids, audio, annotations, cache, keep, separate, device, prepare_only=False, beats="beat_this", **heard):
    songs = []
    for rid in ids:
        song = prepare(rid, os.path.join(audio, rid + ".wav"), cache, keep, separate, device, beats)
        if song is None:
            print(f"{rid}: no stems in the cache (run with --separate): skipped", flush=True)
            continue
        if prepare_only:
            print(
                f"{rid}: {len(song['notes'])} notes, beats by {song['beat_tracker']}, {song['seconds']} s", flush=True
            )
            continue
        song["ref"] = reference(annotations, rid)
        songs.append(listen(song, cache, **heard))
    return songs


# ----------------------------------------------------------------------------------------------------- scoring


def label(chord):
    """A chord of harmony.find as the annotations write it."""
    under = "" if chord.get("bass") is None else "/" + DEGREE[(chord["bass"] - chord["root"]) % 12]
    return f"{FLATS[chord['root']]}:{HARTE[chord['quality']]}{under}"


def as_intervals(spans, end):
    """[(start, end, label)] in time order as intervals without gaps: what is between chords is no chord."""
    ints, labels, at = [], [], 0.0
    for t0, t1, name in spans:
        t0 = max(t0, at)
        if t1 <= t0:
            continue
        if t0 > at + 1e-6:
            ints.append([at, t0])
            labels.append("N")
        ints.append([t0, t1])
        labels.append(name)
        at = t1
    if at < end:
        ints.append([at, end])
        labels.append("N")
    return np.array(ints), labels


def merged(ref, spans):
    """The reference and an estimate cut at each other's boundaries: (intervals, durations, the reference's
    labels, the estimate's labels, the estimate's intervals)."""
    ref_i, ref_l = ref
    est_i, est_l = as_intervals(spans, max(float(ref_i.max()), spans[-1][1] if spans else 0.0))
    est_i, est_l = mir_eval.util.adjust_intervals(est_i, est_l, ref_i.min(), ref_i.max(), "N", "N")
    ints, rl, el = mir_eval.util.merge_labeled_intervals(ref_i, ref_l, est_i, est_l)
    return ints, mir_eval.util.intervals_to_durations(ints), rl, el, est_i


def score(ref, spans):
    """mir_eval's weighted accuracies; `bass`: the share of the time with a chord in which the bass note is the
    reference's; `bass_slash`: the same where the reference's bass is not the root; `seg`: mir_eval's measure
    of how well the chord changes fall where the reference's do."""
    ints, dur, rl, el, est_i = merged(ref, spans)
    out = {m: float(mir_eval.chord.weighted_accuracy(getattr(mir_eval.chord, m)(rl, el), dur)) for m in MEASURES}
    ref_root, _, ref_bass = mir_eval.chord.encode_many(rl, False)
    est_root, _, est_bass = mir_eval.chord.encode_many(el, False)
    there = ref_root >= 0
    same = (ref_root + ref_bass) % 12 == np.where(est_root >= 0, (est_root + est_bass) % 12, -1)
    slash = there & (ref_bass != 0)
    out["bass"] = float((dur * there * same).sum() / max(1e-9, (dur * there).sum()))
    out["bass_slash"] = float((dur * slash * same).sum() / (dur * slash).sum()) if slash.any() else float("nan")
    out["slash_root"] = (
        float((dur * slash * (ref_root == est_root)).sum() / (dur * slash).sum()) if slash.any() else float("nan")
    )
    out["seg"] = float(mir_eval.chord.seg(ref[0], est_i))
    return out


def old_spans(song):
    """The chord per bar the tool wrote before, on the tool's own bar times."""
    bars = np.asarray(song["grid"])[song["origin"] :: 16]
    return [
        (float(bars[c["bar"] - 1]), float(bars[c["bar"]]), FLATS[c["root"]] + (":min" if c["quality"] else ":maj"))
        for c in song["old_chords"]
        if c["bar"] < len(bars)
    ]


def new_spans(chords):
    return [(c["start"], c["end"], label(c)) for c in chords]


def right_share(ref, chords):
    """For each chord found, the share of its time in which the reference has its root and third (None where
    the reference has nothing to compare)."""
    ints, dur, rl, el, _ = merged(ref, new_spans(chords))
    ok = mir_eval.chord.thirds(rl, el)
    out = []
    for c in chords:
        inside = (ints[:, 0] >= c["start"] - 1e-6) & (ints[:, 1] <= c["end"] + 1e-6) & (ok >= 0)
        out.append(float((dur[inside] * ok[inside]).sum() / dur[inside].sum()) if dur[inside].sum() > 0 else None)
    return out


# ------------------------------------------------------------------------------------------- fitting the weights


def chord_class(name, qualities):
    """(root, place in `qualities`) of a reference label, or (-1, -1) where it is no chord or none of ours:
    the reference's chord is taken as the triad or seventh chord of ours it contains."""
    root, bits, _ = mir_eval.chord.encode(name)
    if root < 0 or not bits[7] or bits[3] == bits[4]:
        return -1, -1
    quality = ("7" if bits[10] else "maj7" if bits[11] else "") if bits[4] else ("m7" if bits[10] else "m")
    if quality not in qualities:
        quality = harmony.TRIAD[quality]
    return int(root), qualities.index(quality)


def frame_classes(ref, heard, qualities):
    """The reference's chord at the middle of each frame, as (root, quality) rows; -1 where it has none."""
    ref_i, ref_l = ref
    middle = 0.5 * (heard["edges"][:-1] + heard["edges"][1:])
    row = np.searchsorted(ref_i[:, 0], middle, side="right") - 1
    known = {}
    out = np.full((len(middle), 2), -1)
    for k, i in enumerate(row):
        if 0 <= i < len(ref_l) and middle[k] < ref_i[i, 1]:
            if ref_l[i] not in known:
                known[ref_l[i]] = chord_class(ref_l[i], qualities)
            out[k] = known[ref_l[i]]
    return out


def fit(songs, smooth=1e-5):
    """Weights and biases of the match (harmony.chord_fit) that make each frame's chord (`classes`) most likely
    in the frame taken alone: a softmax over root and quality, the same weights for all twelve roots."""
    from scipy.optimize import minimize

    x = np.concatenate([harmony.from_each_root(s["heard"]["groups"]) for s in songs]).astype(np.float32)
    y = np.concatenate([s["classes"] for s in songs])
    x, y = x[y[:, 0] >= 0], y[y[:, 0] >= 0]
    n, _, d = x.shape
    q = len(harmony.QUALITIES)
    target = np.zeros((n, 12, q), np.float32)
    target[np.arange(n), y[:, 0], y[:, 1]] = 1

    def loss(theta):
        w, b = theta[: q * d].reshape(q, d), theta[q * d :]
        s = x @ w.T.astype(np.float32) + b.astype(np.float32)
        s = s - s.max(axis=(1, 2), keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=(1, 2), keepdims=True)
        value = -np.log(p[np.arange(n), y[:, 0], y[:, 1]] + 1e-12).mean() + smooth * (w**2).sum()
        g = (p - target) / n
        grad_w = np.einsum("frq,frd->qd", g, x) + 2 * smooth * w
        return float(value), np.concatenate([grad_w.ravel(), g.sum(axis=(0, 1))]).astype(np.float64)

    best = minimize(loss, np.zeros(q * d + q), jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    return best.x[: q * d].reshape(q, d), best.x[q * d :]


def by_hand(match=25.0, bass=3.0, seventh=-1.5, overtones=0.6):
    """Weights set without data: for the instruments each chord's own notes, with their first overtones; for
    the bass the root, then the chord's other notes, then anything else. A seventh chord has to earn its place
    over the triad (`seventh`)."""
    w = np.zeros((len(harmony.QUALITIES), 12 * len(harmony.GROUPS)))
    b = np.zeros(len(harmony.QUALITIES))
    for q, quality in enumerate(harmony.QUALITIES):
        tones = harmony.TONES[quality]
        for tone in tones:
            for overtone, up in enumerate((0, 0, 7, 0, 4, 7)):
                w[q, (tone + up) % 12] += overtones**overtone
        w[q, :12] *= match / np.linalg.norm(w[q, :12])
        if "bass" in harmony.GROUPS:
            at = 12 * harmony.GROUPS.index("bass")
            w[q, at : at + 12] = bass * np.array([1.0 if k == 0 else 0.25 if k in tones else -0.5 for k in range(12)])
        b[q] = seventh if len(tones) > 3 else 0.0
    return w, b


# ------------------------------------------------------------------------------------------------- BabySlakh

NOT_CHORDS = {"Bass", "Drums", "Sound Effects", "Percussive"}  # Slakh's instrument classes that carry no chords
SEVENTH = 0.07  # a chord is a seventh chord when its seventh takes this share of the time its notes are held


def midi_notes(track):
    """(the notes of the instruments that carry the chords, the bass's notes), as dicts with start, end, pitch,
    from the MIDI file of each of a Slakh track's stems."""
    import pretty_midi
    import yaml

    with open(os.path.join(track["dir"], "metadata.yaml")) as f:
        stems = yaml.safe_load(f).get("stems") or {}
    played, bass = [], []
    for sid, about in stems.items():
        midi = os.path.join(track["dir"], "MIDI", sid + ".mid")
        if about.get("is_drum") or not os.path.exists(midi):
            continue
        notes = [
            dict(start=n.start, end=n.end, pitch=n.pitch)
            for inst in pretty_midi.PrettyMIDI(midi).instruments
            if not inst.is_drum
            for n in inst.notes
            if n.end - n.start >= 0.01
        ]
        if about.get("inst_class") == "Bass":
            bass += notes
        elif about.get("inst_class") not in NOT_CHORDS:
            played += notes
    return played, bass


def midi_classes(heard, played, bass, cost=None):
    """The chord of each frame as the song's own notes have it, (root, quality) rows, -1 where fewer than two
    pitch classes sound; and the line they were read from (a column of harmony.chord_fit per frame).

    The notes are matched as the audio is, with weights set by hand and no overtones (these are notes, not
    sound), as one line through the song. Then, once for each stretch on one root and triad, whether the chord
    has a seventh is decided again from all the notes held during it. `cost` is what a change costs in every
    frame (default: by its place in the bar, as for the audio)."""
    held = harmony.held_per_frame(played, heard["edges"])
    notes = dict(heard, groups=dict(notes=harmony._unit(held), bass=harmony.held_per_frame(bass, heard["edges"])))
    with changed(GROUPS=("notes", "bass")):
        fit_ = harmony.chord_fit(notes, *by_hand(overtones=0.0))
    line = harmony.best_line(fit_, harmony.change_costs(heard["place"]) if cost is None else np.full(len(fit_), cost))
    chords = 12 * len(harmony.QUALITIES)
    triad = lambda state: (
        (int(state) % 12, harmony.TRIAD[harmony.QUALITIES[int(state) // 12]]) if state < chords else None
    )
    out = np.full((len(line), 2), -1)
    k = 0
    while k < len(line):
        j = k
        while j + 1 < len(line) and triad(line[j + 1]) == triad(line[k]):
            j += 1
        if line[k] < chords:
            root, quality = triad(line[k])
            if "7" in harmony.QUALITIES:
                share = held[k : j + 1].sum(0) / max(held[k : j + 1].sum(), 1e-9)
                minor, major = share[(root + 10) % 12], share[(root + 11) % 12]
                if max(minor, major) >= SEVENTH:
                    quality = (
                        ("7" if minor >= major else "maj7") if quality == "" else ("m7" if minor >= major else "m")
                    )
            line[k : j + 1] = 12 * harmony.QUALITIES.index(quality) + root
            sounding = (held[k : j + 1] > 0.05).sum(1) >= 2
            out[k : j + 1][sounding] = (root, harmony.QUALITIES.index(quality))
        k = j + 1
    return out, line


def slakh_songs(root, cache, keep, separate, device, steps=()):
    """BabySlakh's songs as the tool hears them, each with its own notes from its MIDI (`midi`)."""
    songs = []
    for track in evaluate.load_dataset(root)[1]:
        song = prepare(track["id"], track["mix"], cache, keep, separate, device)
        if song is None:
            print(f"{track['id']}: no stems in the cache (run with --separate): skipped", flush=True)
            continue
        song["midi"], song["track"] = midi_notes(track), track
        songs.append(listen(song, cache, steps=steps))
    return songs


def fit_on(songs, step=None):
    """Weights fitted on BabySlakh songs under harmony's settings as they are now (`step`: frames of another
    length, heard before)."""
    heard = [s["heard"] if step is None else s["by_step"][step] for s in songs]
    return fit([dict(heard=h, classes=midi_classes(h, *s["midi"])[0]) for s, h in zip(songs, heard)])


def score_table(songs, folds=4):
    """A chord's likelihood against how often it has the root and third the MIDI's notes have: for
    harmony.SHARE_RIGHT. Each song's chords are found with weights fitted on the songs of the other folds."""
    likely, right, weight = [], [], []
    with changed(SHARE_RIGHT=((0.0, 0.0), (1.0, 1.0))):
        for fold in range(folds):
            w, b = fit_on([s for k, s in enumerate(songs) if k % folds != fold])
            for s in [s for k, s in enumerate(songs) if k % folds == fold]:
                classes = midi_classes(s["heard"], *s["midi"])[0]
                starts = list(s["heard"]["index"])
                for c in harmony.read(s["heard"], s["length"], w, b)["chords"]:
                    inside = classes[starts.index(c["at"]) : starts.index(c["until"] - harmony.STEP) + 1]
                    inside = inside[inside[:, 0] >= 0]
                    if len(inside):
                        triad = [harmony.TRIAD[harmony.QUALITIES[q]] for q in inside[:, 1]]
                        same = (inside[:, 0] == c["root"]) & (np.array(triad) == harmony.TRIAD[c["quality"]])
                        likely.append(c["score"])
                        right.append(same.mean())
                        weight.append(len(inside))
    likely, right, weight = np.array(likely), np.array(right), np.array(weight, dtype=float)
    edges = (0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001)
    rows = []  # (mean likelihood, share right, share of the time) of each band that has chords
    print("likelihood against root and third as the MIDI has them (BabySlakh, four folds):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (likely >= lo) & (likely < hi)
        if m.any():
            rows.append(
                (np.average(likely[m], weights=weight[m]), np.average(right[m], weights=weight[m]), weight[m].sum())
            )
            print(
                f"  {lo:.3f}-{min(hi, 1.0):.3f}: {rows[-1][2] / weight.sum():5.1%} of the time, right {rows[-1][1]:.2f}"
            )
    shares = never_falling([r[1] for r in rows], [r[2] for r in rows])
    pairs = [(0.0, 0.0)] + [(round(float(r[0]), 3), round(float(y), 2)) for r, y in zip(rows, shares)]
    print("SHARE_RIGHT = (" + ", ".join(f"({x}, {y})" for x, y in pairs) + ")")


def never_falling(values, weights):
    """The values nearest to `values` that never fall from one to the next: neighbours that do are pooled into
    their weighted mean (a band with few chords in it says little)."""
    blocks = [[v, w, 1] for v, w in zip(values, weights)]
    k = 0
    while k + 1 < len(blocks):
        if blocks[k][0] > blocks[k + 1][0]:
            (v0, w0, n0), (v1, w1, n1) = blocks[k], blocks[k + 1]
            blocks[k : k + 2] = [[(v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1, n0 + n1]]
            k = max(k - 1, 0)
        else:
            k += 1
    return [v for v, _, n in blocks for _ in range(n)]


def change_places(songs, cache):
    """Where the MIDI's chords change in the bar, on the MIDI's own beats and bar lines (the songs in 4/4), with
    a change costing the same anywhere: changes per bar at each eighth."""
    place, bars = np.zeros(16), 0
    for s in songs:
        grid = evaluate.reference(s["track"], os.path.join(cache, f"refoct-{s['id']}.json"))["grid"]
        if not grid or not grid["four_four"]:
            continue
        times, origin = rhythm.beat_grid(np.asarray(grid["beats"]), 1, grid["first_downbeat_beat"] % 4, s["length"])
        index = harmony.frame_starts(times, origin, s["length"])
        frames = dict(
            index=index, edges=np.r_[times[index], times[index[-1] + harmony.STEP]], place=(index - origin) % 16
        )
        line = midi_classes(dict(frames, level=np.ones(len(index))), *s["midi"], cost=2.5)[1]
        chords = 12 * len(harmony.QUALITIES)
        for k in np.flatnonzero((np.diff(line) != 0) & (line[1:] < chords) & (line[:-1] < chords)) + 1:
            place[frames["place"][k]] += 1
        bars += len(index) // 8
    print(
        f"chord changes per bar at each eighth of the bar, in the MIDI ({bars} bars): "
        + " ".join(f"{v:.2f}" for v in place[::2] / bars)
    )


# ------------------------------------------------------------------------------------------- running the finder


class changed:
    """harmony's settings changed for a while: `with changed(STEP=4): ...`."""

    def __init__(self, **settings):
        self.settings = settings

    def __enter__(self):
        self.before = {k: getattr(harmony, k) for k in self.settings}
        for k, v in self.settings.items():
            setattr(harmony, k, v)

    def __exit__(self, *exc):
        for k, v in self.before.items():
            setattr(harmony, k, v)


def found_in(songs, weights=None, biases=None, step=None):
    """harmony.find on what was already heard in each song, with the given weights (default: harmony's own)."""
    return [
        harmony.read(s["heard"] if step is None else s["by_step"][step], s["length"], weights, biases) for s in songs
    ]


def cross_validated(songs, folds=4):
    """Each RWC song's harmony with weights fitted on the annotated chords of the songs of the other folds."""
    out = [None] * len(songs)
    for s in songs:
        s["classes"] = frame_classes(s["ref"], s["heard"], harmony.QUALITIES)
    for fold in range(folds):
        w, b = fit([s for k, s in enumerate(songs) if k % folds != fold])
        for k, s in enumerate(songs):
            if k % folds == fold:
                out[k] = harmony.read(s["heard"], s["length"], w, b)
    return out


# ---------------------------------------------------------------------------------------------------- reporting


def mean_line(rows, names=("root", "majmin", "sevenths", "bass", "slash_root", "seg")):
    return "  ".join(f"{m} {np.nanmean([r[m] for r in rows]):.3f}" for m in names)


def report(songs, found, title):
    """Per song and overall, the chord per bar of before beside the finder; then the score and the key."""
    print(f"\n== {title}: {len(songs)} songs")
    old = [score(s["ref"], old_spans(s)) for s in songs]
    new = [score(s["ref"], new_spans(f["chords"])) for s, f in zip(songs, found)]
    for s, o, n in zip(songs, old, new):
        print(
            f"{s['id']}  before: root {o['root']:.2f} maj/min {o['majmin']:.2f} sevenths {o['sevenths']:.2f} | "
            f"finder: root {n['root']:.2f} maj/min {n['majmin']:.2f} sevenths {n['sevenths']:.2f} "
            f"bass {n['bass']:.2f} changes {n['seg']:.2f}"
        )
    for name, rows in (("before", old), ("finder", new)):
        print(f"{name} mean: {mean_line(rows, ('root', 'majmin', 'sevenths', 'thirds', 'triads', 'tetrads', 'mirex'))}")
        print(f"            {mean_line(rows, ('bass', 'bass_slash', 'slash_root', 'seg'))}")
        for m in ("root", "majmin", "sevenths"):
            v = np.array([r[m] for r in rows])
            print(f"       {m}: median {np.median(v):.2f}, worst {v.min():.2f}, songs under 0.70: {(v < 0.7).sum()}")
    worst = sorted(zip(songs, old, new), key=lambda t: t[2]["root"])[:5]
    print(
        "worst five by root: " + "; ".join(f"{s['id']} {n['root']:.2f} (before {o['root']:.2f})" for s, o, n in worst)
    )
    better = sum(n["root"] > o["root"] + 0.02 for o, n in zip(old, new))
    worse = sum(n["root"] < o["root"] - 0.02 for o, n in zip(old, new))
    print(f"root better than before by 0.02 or more on {better} songs, worse on {worse}")
    scores_against_right(songs, found)
    key_table(songs, found)


def scores_against_right(songs, found, edges=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    given, right, weight = [], [], []
    for s, f in zip(songs, found):
        for c, share in zip(f["chords"], right_share(s["ref"], f["chords"])):
            if share is not None:
                given.append(c["score"])
                right.append(share)
                weight.append(c["end"] - c["start"])
    given, right, weight = np.array(given), np.array(right), np.array(weight)
    print("a chord's score against its root and third being right (shares of the chords' time):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (given >= lo) & (given < hi)
        if m.any():
            print(
                f"  {lo:.1f}-{min(hi, 1.0):.1f}: {weight[m].sum() / weight.sum():5.1%} of the time, "
                f"right {np.average(right[m], weights=weight[m]):.2f} (mean score {np.average(given[m], weights=weight[m]):.2f})"
            )


def relation(key, ref):
    """How a key found relates to the reference's: same, relative, parallel, fifth or other."""
    (root, major), (r, m) = key, ref
    if (root, major) == (r, m):
        return "same"
    if major != m and (root - r) % 12 == (9 if m else 3):
        return "relative"
    if major != m and root == r:
        return "parallel"
    if major == m and (root - r) % 12 in (5, 7):
        return "fifth"
    return "other"


def key_table(songs, found):
    rows = []
    for s, f in zip(songs, found):
        if s["id"] not in KEYS or not f["keys"]:
            continue
        main = max(f["keys"], key=lambda k: k["end"] - k["start"])
        name, root, major = harmony.main_key(f)
        other = main["other"]
        rows.append(
            dict(
                id=s["id"],
                main=relation((root, major), KEYS[s["id"]]),
                other=relation((other["root"], other["major"]), KEYS[s["id"]]),
                before=relation((s["old_key"][1], s["old_key"][2]), KEYS[s["id"]]),
                score=main["score"],
                spans=len(f["keys"]),
                name=name,
            )
        )
    if not rows:
        return
    count = lambda field: {
        v: sum(r[field] == v for r in rows) for v in ("same", "relative", "parallel", "fifth", "other")
    }
    print(f"main key against the key read by hand, {len(rows)} songs:")
    print(
        f"  finder: {count('main')}; right, or its other key right: {sum('same' in (r['main'], r['other']) for r in rows)}"
    )
    print(f"  before: {count('before')}")
    ok = [r["score"] for r in rows if r["main"] == "same"]
    bad = [r["score"] for r in rows if r["main"] != "same"]
    print(
        f"  score of the main key: right {np.mean(ok) if ok else float('nan'):.2f}, wrong {np.mean(bad) if bad else float('nan'):.2f}"
    )
    print(f"  keys per song: median {np.median([r['spans'] for r in rows]):.0f}, most {max(r['spans'] for r in rows)}")
    for r in rows:
        if r["main"] != "same":
            print(f"  {r['id']}: found {r['name']} ({r['main']}; its other key: {r['other']}; score {r['score']:.2f})")


def print_weights(w, b):
    print("WEIGHTS = np.array(\n    [")
    for row in w:
        print("        [" + ", ".join(f"{v:.2f}" for v in row) + "],")
    print("    ]\n)")
    print("BIASES = np.array([" + ", ".join(f"{v:.2f}" for v in b) + "])")


def ingredients(songs, slakh):
    """The finder without each of its parts in turn: weights fitted on BabySlakh as the part asks, measured on
    the RWC songs."""
    print(f"\n== what each part is worth, on {len(songs)} RWC songs")

    def line(name, found):
        rows = [score(s["ref"], new_spans(f["chords"])) for s, f in zip(songs, found)]
        print(f"{name:66s} {mean_line(rows)}", flush=True)

    print(f"{'before: one triad a bar':66s} {mean_line([score(s['ref'], old_spans(s)) for s in songs])}")
    line("the finder, as it ships", found_in(songs))
    line("the finder, its weights fitted again", found_in(songs, *fit_on(slakh)))
    with changed(GROUPS=("notes",)):
        line("without the bass notes", found_in(songs, *fit_on(slakh)))
    with changed(QUALITIES=("", "m")):
        line("triads only", found_in(songs, *fit_on(slakh)))
    for flat in (2.0, 3.0, 4.0):
        with changed(CHANGE_COST={p: flat for p in range(0, 16, 2)}):
            line(f"a change costs {flat} anywhere in the bar", found_in(songs, *fit_on(slakh)))
    for step, name in ((4, "frames of a beat"), (16, "frames of a bar")):
        with changed(STEP=step):
            line(name, found_in(songs, *fit_on(slakh, step), step))
    line("weights set by hand, no data", found_in(songs, *by_hand()))
    line("weights fitted on these RWC songs' annotations, four folds", cross_validated(songs))
    with changed(GROUPS=harmony.GROUPS + ("sung",)):
        line("the same, with what is sung", cross_validated(songs))
    with changed(GROUPS=harmony.GROUPS + ("notes_half_bar",)):
        line("the same, with the notes of the half bar around each eighth", cross_validated(songs))


# -------------------------------------------------------------------------------------------------------- main


def main(argv):
    opts, flags, args = dict(songs="check"), set(), list(argv)
    while args:
        a = args.pop(0)
        if a in ("--audio", "--annotations", "--songs", "--slakh", "--keep", "--device", "--weights", "--beats"):
            opts[a[2:]] = args.pop(0)
        else:
            flags.add(a)
    cache = cache_home()
    keep = opts.get("keep") or os.path.join(cache, "chords-eval")
    separate, device = "--separate" in flags, opts.get("device")
    every = "--ingredients" in flags
    slakh = slakh_songs(opts["slakh"], cache, keep, separate, device, STEPS if every else ()) if "slakh" in opts else []
    if "--fit" in flags:
        print(f"fitted on {len(slakh)} songs of BabySlakh:")
        print_weights(*fit_on(slakh))
        score_table(slakh)
        change_places(slakh, cache)
        return
    ids = dict(check=CHECK, tuning=CHECK).get(opts["songs"]) or (
        HELD_OUT if opts["songs"] == "held-out" else opts["songs"].split(",")
    )
    songs = rwc_songs(
        ids,
        opts["audio"],
        opts["annotations"],
        cache,
        keep,
        separate,
        device,
        "--prepare-only" in flags,
        opts.get("beats", "beat_this"),
        more=every,
        steps=STEPS if every else (),
    )
    if "--prepare-only" in flags:
        return
    if "weights" in opts:
        w = np.load(opts["weights"])
        report(songs, found_in(songs, w["weights"], w["biases"]), f"{opts['songs']}, weights of {opts['weights']}")
    elif "--by-hand" in flags:
        report(songs, found_in(songs, *by_hand()), opts["songs"] + ", weights set by hand")
    elif "--cv" in flags:
        report(
            songs,
            cross_validated(songs),
            opts["songs"] + ", weights fitted on the other songs' annotations (four folds)",
        )
    else:
        report(songs, found_in(songs), opts["songs"])
    if every:
        ingredients(songs, slakh)


if __name__ == "__main__":
    main(sys.argv[1:])
