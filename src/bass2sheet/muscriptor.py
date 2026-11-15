"""MuScriptor's notes in place of bass2sheet's own. Optional: nothing here runs unless it is asked for.

MuScriptor (Rouard et al., 2026, by Kyutai and Mirelo) is a transformer that writes notes straight from audio.
Given the bass track bass2sheet isolates, and told that it is a bass, it finds the notes better than notes.py does
(docs/evaluation.md: 0.83 against 0.76 on RWC Popular, 0.91 against 0.79 on a bass played alone). Given the whole
mix it does worse than separating first, so the separation stays.

It is not part of bass2sheet. Its weights are under CC BY-NC 4.0 with conditions of use of their authors, and it
runs through a program of its own (dev/muscriptor_cpp builds one over muscriptor.cpp). This module asks that
program for the notes and brings them into the shape the rest of the pipeline works on. bass2sheet's own notes
become the second opinion: a note MuScriptor writes and bass2sheet does not hear is flagged.

    MUSCRIPTOR_NOTES   the muscriptor_notes program
    MUSCRIPTOR_GGUF    the directory holding muscriptor-SIZE-f16.gguf
    MUSCRIPTOR_SIZE    small, medium (the default) or large
"""

import csv
import os
import subprocess
import tempfile

import numpy as np

from .common import SR, Bass2SheetError

BASS = "electric_bass,acoustic_bass"  # the instrument groups it may write
SHORTEST = 0.03  # s: what it closes at once (10 ms) is not a note
SAME = 0.05  # s: one pitch twice within this is one note (it now and then writes a note for both basses)
TOGETHER = 0.04  # s: notes that start within this of each other are played together
HEARD_TOO = 0.1  # s: bass2sheet has the note too if it starts one of that pitch within this
STEP = 2  # semitones: two notes this close are not played together on a bass; one of them is the note
ALONE = "alone"  # the flag of a note bass2sheet's own finder does not hear
LOW = (
    8  # the level around a note's start is read from the audio at an eighth of the rate: its partials are below 1.4 kHz
)


def program():
    """(the program, the weights file, the size), or an error that says what to set."""
    binary, weights = os.environ.get("MUSCRIPTOR_NOTES", ""), os.environ.get("MUSCRIPTOR_GGUF", "")
    size = os.environ.get("MUSCRIPTOR_SIZE", "medium")
    model = os.path.join(weights, f"muscriptor-{size}-f16.gguf")
    if not os.path.isfile(binary) or not os.path.isfile(model):
        raise Bass2SheetError(
            "MuScriptor's notes were asked for, and it is not set up: set MUSCRIPTOR_NOTES to the muscriptor_notes "
            "program (dev/muscriptor_cpp builds it) and MUSCRIPTOR_GGUF to the directory holding "
            + os.path.basename(model)
        )
    return binary, model, size


def heard(bass_mono, stem_id, cache_dir):
    """The notes MuScriptor writes for a bass track, as (start, end, pitch) in seconds and MIDI numbers. Kept in
    `cache_dir` by the track's id and the model's size."""
    binary, model, size = program()
    out = os.path.join(cache_dir, f"muscriptor-{stem_id}-{size}.csv")
    if not os.path.exists(out):
        import librosa

        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "bass.f32")  # what the program reads: mono, 16 kHz, 32-bit floats
            y = librosa.resample(
                np.asarray(bass_mono, dtype=np.float32), orig_sr=SR, target_sr=16000, res_type="soxr_hq"
            )
            y.astype("<f4").tofile(raw)
            done = subprocess.run([binary, model, BASS, raw, out], capture_output=True, text=True)
        if done.returncode or not os.path.exists(out):
            raise Bass2SheetError(f"MuScriptor did not write the notes: {done.stderr.strip()[-400:] or 'no message'}")
    with open(out) as f:
        return [
            (float(r["start"]), float(r["end"]), int(r["pitch"]))
            for r in csv.DictReader(f)
            if r["instrument"] != "drums"
        ]


def _partials(mono, start, end, pitch):
    """How strongly a pitch's second to fourth partials sound in the bass track over a note. A step apart, two
    fundamentals are too close to tell in a short note; their partials are not."""
    if mono is None:
        return 0.0
    seg = np.asarray(mono[int((start + 0.03) * SR) : int(max(end, start + 0.2) * SR)], dtype=np.float64)
    if len(seg) < 256:
        return 0.0
    window, t = np.hanning(len(seg)), np.arange(len(seg)) / SR
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    at = lambda hz: abs(np.sum(seg * window * np.exp(-2j * np.pi * hz * t)))
    return float(sum(at(k * f0) for k in (2, 3, 4)) / window.sum())


def _apart(tones, start, ends, theirs, bass_mono):
    """The pitches that start together, without those a step from another. A bass does not play two notes a
    semitone or a tone apart at once: that low they grate. Where MuScriptor writes two that close, one is the note
    and the other its shadow: the one bass2sheet's finder hears too is kept, or else the one whose partials are
    the stronger in the bass track."""
    tones = sorted(tones)
    while True:
        close = [(a, b) for a, b in zip(tones, tones[1:]) if b - a <= STEP]
        if not close:
            return tones
        a, b = close[0]
        mine = [any(q == t and abs(s - start) <= HEARD_TOO for s, q in theirs) for t in (a, b)]
        if mine[0] != mine[1]:
            tones.remove(b if mine[0] else a)
        else:
            level = [_partials(bass_mono, start, ends[t], t) for t in (a, b)]
            tones.remove(a if level[1] > level[0] else b)


def _both(start, low, high, own):
    """Whether bass2sheet's own finder wrote the same two notes together there."""
    return any(
        abs(o["start"] - start) <= HEARD_TOO and o.get("also") and {low, high} <= {o["pitch"], *o["also"]} for o in own
    )


def as_notes(raw, own, bass_mono, p):
    """MuScriptor's notes as the pipeline's (see notes.segment_notes for the shape).

    `raw` is (start, end, pitch); `own` the notes bass2sheet's finder heard in the same track; `p` the SegParams.
    A bass plays one line: a pitch written twice at once is one note, and a note ends where the next begins.
    Of the notes MuScriptor starts together one is the note: the one bass2sheet hears too, or else the lowest (of
    two a step apart, see _apart). The others are written with it, in `also`, only where bass2sheet's own finder
    wrote the same two together (and `p.together` is on): on the test songs two or three in a hundred of
    MuScriptor's second notes are in the answer key (docs/evaluation.md). Pitches under the instrument's lowest
    note go up by octaves. Starts move
    from MuScriptor's 10 ms steps to where the pluck begins to sound; a note bass2sheet hears too ends where
    bass2sheet ends it.

    The second opinion: `agree` is 1 where bass2sheet's finder heard the note too. Where it did not, the note is
    flagged ALONE if bass2sheet hears another pitch there, or nothing, or holds that pitch from before and no
    pluck is heard at the note's start. A pitch bass2sheet holds and MuScriptor plucks again, with a pluck to be
    heard, is not flagged (agree 0.5): on real recordings MuScriptor is right about those nearly nine times in
    ten, and of the flagged notes more than half are wrong (docs/evaluation.md)."""
    from .notes import refine_onsets

    line = []
    for start, end, pitch in sorted(raw):
        while p.lowest and pitch < p.lowest:
            pitch += 12
        if end - start >= SHORTEST and not any(q == pitch and abs(start - a) <= SAME for a, _, q in line[-8:]):
            line.append((start, end, pitch))
    groups = []  # what starts together: (start, {pitch: end})
    for start, end, pitch in line:
        if groups and start - groups[-1][0] <= TOGETHER:
            groups[-1][1][pitch] = max(end, groups[-1][1].get(pitch, 0.0))
        else:
            groups.append((start, {pitch: end}))
    theirs = [(o["start"], q) for o in own for q in (o["pitch"], *o.get("also", []))]
    in_order = sorted(own, key=lambda o: o["start"])
    notes = []
    for start, ends in groups:
        tones = _apart(ends, start, ends, theirs, bass_mono)
        heard = [q for q in tones if any(pq == q and abs(s - start) <= HEARD_TOO for s, pq in theirs)]
        note = (heard or tones)[0]  # the one bass2sheet hears too, or else the lowest
        with_it = [q for q in tones if q != note and (q - note) % 12 and _both(start, *sorted((note, q)), own)]
        kept = sorted([note, *with_it]) if p.together else [note]
        nt = dict(pitch=kept[0], start=start, end=ends[kept[0]])  # the lowest is the note; it keeps its own end
        if kept[1:]:
            nt["also"] = kept[1:]
        notes.append(nt)
        under = [o for o in own if o["start"] <= nt["start"] + 0.03 < o["end"]]  # what bass2sheet has sounding there
        if any(q == nt["pitch"] and abs(s - nt["start"]) <= HEARD_TOO for s, q in theirs):
            nt["agree"] = 1.0
        elif under and all(o["pitch"] == nt["pitch"] for o in under):
            nt["agree"] = 0.5  # the same pitch, held: the question is only whether it is plucked again
        else:
            nt["agree"] = 0.0
        nt["plucked"] = True
        # Where bass2sheet hears the note too it ends where bass2sheet ends it: at the fall of the track's level,
        # which is closer than MuScriptor's ends (docs/evaluation.md). Notes of that pitch that follow without a
        # gap, while MuScriptor's note lasts, carry it on.
        end = None
        for o in in_order:
            if o["pitch"] != nt["pitch"]:
                continue
            if end is None and (
                abs(o["start"] - nt["start"]) <= HEARD_TOO or o["start"] <= nt["start"] + 0.03 < o["end"]
            ):
                end = o["end"]
            elif end is not None and 0 <= o["start"] - end <= 0.05 and o["start"] < nt["end"]:
                end = o["end"]
        if end is not None:
            nt["end"] = end
    for nt, after in zip(notes, notes[1:]):
        nt["end"] = min(nt["end"], after["start"])
    for nt in notes:
        nt["end"] = max(nt["end"], nt["start"] + 0.02)
    if p.refine_onsets and len(notes):
        refine_onsets(notes, bass_mono, p)
    for nt in notes:
        nt["flag"] = ALONE if nt["agree"] == 0.0 or (nt["agree"] == 0.5 and not nt["plucked"]) else None
    return notes


def rises(notes, bass_mono, mix, within=0.03):
    """The third witness, measured and not used: how far the level rises around each note's start at the note's
    own partials, in dB, in the isolated bass and in the original recording. One (bass, original) per note.

    The rise is from the quietest moment of the 100 ms before to the loudest of the 60 ms after, the largest
    within `within` seconds of the start. A string plucked again raises both; a held note raises neither, and
    another instrument that comes in on the same pitch raises the original alone, so the smaller of the two is
    the evidence: it finds 36% of the re-plucks of RWC Popular at 3% false alarms in held notes, against 14% for
    the isolated bass alone (dev/repluck_eval.py).

    It does not decide which of MuScriptor's notes are flagged. Where MuScriptor has written a note the rise says
    little more about whether the note is right: taking the flag off the notes the original supports, or putting
    it on the ones it does not, made the flag worse on RWC Popular at every threshold (dev/witness_eval.py,
    docs/evaluation.md)."""
    import librosa
    from scipy.signal import resample_poly

    n_fft, hop = 256, 32  # at SR / LOW: a 46 ms window every 5.8 ms, 21.5 Hz to the bin
    step, out = hop * LOW / SR, []
    specs = [
        np.abs(
            librosa.stft(
                resample_poly(np.asarray(y, dtype=np.float64), 1, LOW).astype(np.float32),
                n_fft=n_fft,
                hop_length=hop,
                window="hann",
            )
        )
        ** 2
        for y in (bass_mono, mix)
    ]
    before, gap, after = int(0.10 / step), int(0.01 / step), int(0.06 / step)
    for nt in notes:
        f0, pair = 440 * 2 ** ((nt["pitch"] - 69) / 12), []
        for spec in specs:
            level = np.zeros(spec.shape[1])
            for partial in (1, 2, 3, 4):
                k = int(round(partial * f0 * n_fft * LOW / SR))
                if k + 1 < spec.shape[0]:
                    level += spec[max(1, k - 1) : k + 2].sum(0)
            best = 0.0
            for d in np.arange(-within, within + 1e-9, 0.01):
                i = int(round((nt["start"] + d) / step))
                was, now = level[max(0, i - before) : max(1, i - gap)], level[max(0, i) : max(0, i) + after + 1]
                if len(was) and len(now):
                    best = max(best, 10 * np.log10((now.max() + 1e-12) / (was.min() + 1e-12)))
            pair.append(float(best))
        out.append(tuple(pair))
    return out
