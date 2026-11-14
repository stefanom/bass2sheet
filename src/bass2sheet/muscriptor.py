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
ALONE = "alone"  # the flag of a note bass2sheet's own finder does not hear


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


def as_notes(raw, own, bass_mono, p):
    """MuScriptor's notes as the pipeline's (see notes.segment_notes for the shape).

    `raw` is (start, end, pitch); `own` the notes bass2sheet's finder heard in the same track; `p` the SegParams.
    A bass plays one line: a pitch written twice at once is one note, notes that start together are one note with
    the others in `also` (octaves of the lowest are left out, and all of them when `p.together` is off), and a
    note ends where the next begins. Pitches under the instrument's lowest note go up by octaves. Starts move
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
    notes = []
    for start, end, pitch in line:
        last = notes[-1] if notes else None
        if last and start - last["start"] <= TOGETHER:
            low, high = sorted((last["pitch"], pitch))
            if pitch < last["pitch"]:  # the lower one is the note; it keeps its own end
                last["end"] = end
            last["pitch"] = low
            last["also"] = sorted(set(last["also"]) | {high})
        else:
            notes.append(dict(pitch=pitch, start=start, end=end, also=[]))
    theirs = [(o["start"], q) for o in own for q in (o["pitch"], *o.get("also", []))]
    in_order = sorted(own, key=lambda o: o["start"])
    for nt in notes:
        nt["also"] = [q for q in nt["also"] if (q - nt["pitch"]) % 12] if p.together else []
        if not nt["also"]:
            del nt["also"]
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
