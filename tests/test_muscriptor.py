"""MuScriptor's notes in place of bass2sheet's own (muscriptor.py). The model is never run here: a stand-in
program writes the notes a test hands it."""

import json
import os
import stat
import sys

import numpy as np
import pytest

from bass2sheet import copilot, evaluate, muscriptor, pipeline
from bass2sheet.common import SR, Bass2SheetError
from bass2sheet.notes import SegParams
from bass2sheet.pipeline import seg_params

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


def _line(spec, seconds):
    """A bass line as audio: (start, length, pitch) plucked notes."""
    import librosa

    from bass2sheet.render import pluck

    y = np.zeros(int(seconds * SR), dtype=np.float32)
    for start, dur, pitch in spec:
        w = 0.4 * pluck(librosa.midi_to_hz(pitch), dur, 100)
        y[int(start * SR) : int(start * SR) + len(w)] += w
    return y


def _stand_in(tmp_path, monkeypatch, rows, fails=False):
    """Sets up a program that writes `rows` (start, end, pitch, instrument) as MuScriptor would, and the weights
    file it is pointed at. Returns the path of the log of its calls."""
    log, program = str(tmp_path / "calls.txt"), str(tmp_path / "muscriptor_notes")
    lines = "".join(f"{s:.4f},{e:.4f},{p},{i}\n" for s, e, p, i in rows)
    with open(program, "w") as f:
        f.write(
            f"#!{sys.executable}\nimport os, sys\n"
            f"samples = os.path.getsize(sys.argv[3]) // 4\n"
            f"open({log!r}, 'a').write(' '.join(sys.argv[1:3]) + ' ' + str(samples) + '\\n')\n"
            + ("sys.stderr.write('the model would not load\\n'); sys.exit(1)\n" if fails else "")
            + f"open(sys.argv[4], 'w').write('start,end,pitch,instrument\\n' + {lines!r})\n"
        )
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "muscriptor-medium-f16.gguf").write_bytes(b"")
    monkeypatch.setenv("MUSCRIPTOR_NOTES", program)
    monkeypatch.setenv("MUSCRIPTOR_GGUF", str(weights))
    monkeypatch.delenv("MUSCRIPTOR_SIZE", raising=False)
    return log


def test_asked_for_and_not_set_up_says_what_to_set(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSCRIPTOR_NOTES", raising=False)
    monkeypatch.delenv("MUSCRIPTOR_GGUF", raising=False)
    with pytest.raises(Bass2SheetError, match="MUSCRIPTOR_NOTES.*MUSCRIPTOR_GGUF.*muscriptor-medium-f16.gguf"):
        muscriptor.heard(np.zeros(SR, np.float32), "abc", str(tmp_path))
    monkeypatch.setenv("MUSCRIPTOR_SIZE", "large")
    with pytest.raises(Bass2SheetError, match="muscriptor-large-f16.gguf"):
        muscriptor.program()
    with pytest.raises(Bass2SheetError, match="unknown note finder 'basic'"):
        seg_params("every", notes="basic")
    assert seg_params("every").finder == "own" and seg_params("every", notes="muscriptor").finder == "muscriptor"


def test_the_program_is_asked_once_for_a_track_and_its_notes_are_kept(tmp_path, monkeypatch):
    rows = [(0.5, 1.0, 40, "electric_bass"), (0.5, 0.51, 36, "drums"), (1.0, 1.5, 43, "acoustic_bass")]
    log = _stand_in(tmp_path, monkeypatch, rows)
    mono = np.zeros(2 * SR, np.float32)
    assert muscriptor.heard(mono, "abc", str(tmp_path)) == [
        (0.5, 1.0, 40),
        (1.0, 1.5, 43),
    ]  # the drum hit is not a note
    assert muscriptor.heard(mono, "abc", str(tmp_path)) == [(0.5, 1.0, 40), (1.0, 1.5, 43)]
    calls = open(log).read().splitlines()
    assert len(calls) == 1  # the second time from the cache
    model, may, samples = calls[0].split(" ")
    assert model.endswith("muscriptor-medium-f16.gguf") and may == "electric_bass,acoustic_bass"
    assert abs(int(samples) - 2 * 16000) <= 2  # it was handed the track at 16 kHz
    muscriptor.heard(mono, "another", str(tmp_path))
    assert len(open(log).read().splitlines()) == 2


def test_a_program_that_fails_is_an_error_with_its_words(tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch, [], fails=True)
    with pytest.raises(Bass2SheetError, match="MuScriptor did not write the notes: the model would not load"):
        muscriptor.heard(np.zeros(SR, np.float32), "abc", str(tmp_path))


def test_what_it_writes_becomes_one_line():
    p = SegParams(refine_onsets=False)
    raw = [
        (0.50, 1.00, 40),
        (0.51, 1.00, 40),  # the same note written for the other bass
        (1.00, 1.60, 47),  # played together with the note under it
        (1.01, 1.50, 43),
        (1.50, 2.10, 45),  # runs over the start of the next
        (2.00, 2.01, 50),  # closed at once: not a note
        (2.00, 2.50, 52),  # an octave pair
        (2.02, 2.50, 40),
        (3.00, 3.40, 27),  # under the low E
    ]
    notes = muscriptor.as_notes(raw, [], np.zeros(4 * SR, np.float32), p)
    assert [(n["start"], n["end"], n["pitch"], n.get("also")) for n in notes] == [
        (0.50, 1.00, 40, None),
        (1.00, 1.50, 43, [47]),
        (1.50, 2.00, 45, None),
        (2.00, 2.50, 40, None),  # the octave above the note is not written
        (3.00, 3.40, 39, None),
    ]
    assert all(n["flag"] == muscriptor.ALONE and n["agree"] == 0.0 for n in notes)  # nobody else heard any of it
    single = muscriptor.as_notes(raw, [], np.zeros(4 * SR, np.float32), SegParams(refine_onsets=False, together=False))
    assert not any("also" in n for n in single) and [n["pitch"] for n in single] == [40, 43, 45, 40, 39]
    assert (
        muscriptor.as_notes(raw, [], np.zeros(4 * SR, np.float32), SegParams(refine_onsets=False, lowest=0))[-1][
            "pitch"
        ]
        == 27
    )
    assert muscriptor.as_notes([], [], np.zeros(SR, np.float32), p) == []


def test_a_note_both_hear_ends_where_bass2sheet_ends_it():
    p = SegParams(refine_onsets=False)
    quiet = np.zeros(6 * SR, np.float32)
    own = [
        dict(start=0.5, end=0.9, pitch=40),  # MuScriptor holds this one too long
        dict(start=1.5, end=1.9, pitch=43),  # ... and lets go of this one too soon
        dict(start=2.5, end=2.9, pitch=45),  # bass2sheet wrote this note in two; MuScriptor in one
        dict(start=2.92, end=3.3, pitch=45),
        dict(start=3.5, end=4.4, pitch=47),  # bass2sheet holds it; MuScriptor plucks it twice
    ]
    raw = [(0.52, 1.4, 40), (1.52, 1.6, 43), (2.52, 3.4, 45), (3.52, 3.8, 47), (4.0, 4.6, 47), (5.0, 5.5, 50)]
    notes = muscriptor.as_notes(raw, own, quiet, p)
    assert [n["end"] for n in notes] == [0.9, 1.9, 3.3, 4.0, 4.4, 5.5]  # the last is MuScriptor's alone: its own end
    assert [n["agree"] for n in notes] == [1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
    # a note never runs over the next, whoever ends it
    crowded = muscriptor.as_notes([(0.52, 0.6, 40), (0.7, 1.0, 43)], [dict(start=0.5, end=2.0, pitch=40)], quiet, p)
    assert [(n["start"], n["end"]) for n in crowded] == [(0.52, 0.7), (0.7, 1.0)]


def test_bass2sheets_own_notes_are_the_second_opinion():
    own = [
        dict(start=0.5, end=1.0, pitch=40),
        dict(start=1.0, end=1.5, pitch=42),
        dict(start=1.5, end=2.5, pitch=45),
        dict(start=2.5, end=3.5, pitch=47),
    ]
    raw = [
        (0.52, 1.0, 40),  # heard by both
        (1.02, 1.5, 43),  # bass2sheet hears another pitch
        (1.52, 2.0, 45),  # heard by both
        (2.02, 2.5, 45),  # bass2sheet holds the note; no pluck is in the sound
        (2.52, 3.0, 47),  # heard by both
        (3.02, 3.5, 47),  # bass2sheet holds the note; it is plucked again in the sound
        (4.02, 4.3, 50),  # bass2sheet hears nothing: and there is nothing
    ]
    sound = _line([(0.5, 0.5, 40), (1.0, 0.5, 43), (1.5, 1.0, 45), (2.5, 0.5, 47), (3.0, 0.5, 47)], 5.0)
    notes = muscriptor.as_notes(raw, own, sound, SegParams())
    assert [n["agree"] for n in notes] == [1.0, 0.0, 1.0, 0.5, 1.0, 0.5, 0.0]
    assert [n["flag"] for n in notes] == [None, "alone", None, "alone", None, None, "alone"]
    assert [n["plucked"] for n in notes] == [True, True, True, False, True, True, False]
    for n, (start, _, _) in zip(notes, raw):
        if n["plucked"]:  # the start has moved from MuScriptor's late 10 ms step to the pluck
            assert abs(n["start"] - (start - 0.02)) < 0.008, (n["start"], start)
        else:
            assert n["start"] == start
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(notes, notes[1:]))


def test_the_pipeline_writes_its_notes_when_asked_to(song, song_files, cache_dir, tmp_path, monkeypatch):
    mix, stems = song_files
    late = lambda t: round(t + 0.015, 2)  # as MuScriptor's are: on 10 ms steps, a little late
    rows = [(late(s), late(e), p, "electric_bass") for s, e, p in song["notes"]]
    odd = max(e for _, e, _ in song["notes"]) + 0.5  # and one note that is not in the song
    log = _stand_in(tmp_path, monkeypatch, [*rows, (odd, odd + 0.3, 60, "electric_bass")])
    out = str(tmp_path / "out")
    kw = dict(stems_dir=stems, second_mode="off", crepe_model=CREPE, beat_tracker="librosa", device="cpu")
    assert pipeline.run(mix, out, notes="muscriptor", cache_dir=cache_dir, pdf=False, renders=False, **kw) in (0, 2)
    data = json.load(open(os.path.join(out, "notes.json")))
    assert data["params"]["finder"] == "muscriptor" and len(open(log).read().splitlines()) == 1
    est = [(n["start"], n["end"], n["pitch"]) for n in data["notes"]]
    m = evaluate.score(song["notes"], est)
    assert m["f"] >= 0.97 and abs(m["onset_bias_ms"]) < 8, m  # its notes, with the starts back on the plucks
    extra = [n for n in data["notes"] if n["pitch"] == 60]
    assert len(extra) == 1 and extra[0]["flag"] == "alone"
    assert sum(n["flag"] == "alone" for n in data["notes"]) <= 3  # bass2sheet's finder heard the rest too
    report = open(os.path.join(out, "report.json")).read()
    assert "MuScriptor's alone" in report


def test_the_copilot_starts_from_its_notes_and_doubts_the_ones_it_alone_hears(song, song_files, tmp_path, monkeypatch):
    mix, stems = song_files
    rows = [(round(s + 0.015, 2), round(e + 0.015, 2), p, "electric_bass") for s, e, p in song["notes"]]
    odd = max(e for _, e, _ in song["notes"]) + 0.5
    _stand_in(tmp_path, monkeypatch, [*rows, (odd, odd + 0.3, 60, "electric_bass")])
    work = str(tmp_path / "scores")
    opts = dict(
        defaults=dict(beat_tracker="librosa"), crepe_model=CREPE, device="cpu", cache_dir=str(tmp_path / "cache")
    )
    copilot.add([mix], work, opts, settings=dict(notes="muscriptor"), stems_dirs={mix: stems})
    d = copilot.Desk(work, "mix")
    state = d.public()
    assert state["finder"] == "muscriptor" and d.machine["settings"]["notes"] == "muscriptor"
    assert abs(len(state["notes"]) - len(rows) - 1) <= 1
    extra = [n for n in state["notes"] if n["pitch"] == 60]
    assert len(extra) == 1 and extra[0]["doubt"]["alone"] is True
    assert sum(bool(n.get("doubt", {}).get("alone")) for n in state["notes"]) <= 3
    d.act(dict(kind="delete", id=extra[0]["id"], offer=False))
    m = copilot.mistakes(work, "mix")
    assert m["finder"] == "muscriptor" and m["kinds"] == {"extra note": 1}
    assert "Of MuScriptor's" in copilot.report(work)
    # the same song from bass2sheet's own notes stands beside it, in another working directory
    own = str(tmp_path / "own")
    copilot.add([mix], own, opts, stems_dirs={mix: stems})
    assert copilot.Desk(own, "mix").public()["finder"] == "own" and "Of the tool's" in copilot.report(own)


def _record(bass, seed=0):
    """The original recording of a bass line: the bass with a held chord above it and a little noise."""
    t = np.arange(len(bass)) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (523.3, 659.3, 784.0))
    noise = np.random.default_rng(seed).standard_normal(len(bass))
    return (bass + 0.05 * chord + 0.003 * noise).astype(np.float32)


def test_a_string_plucked_again_raises_the_level_in_the_bass_and_in_the_original_and_a_held_note_in_neither():
    # E plucked, plucked again while it rings, then a G that is held
    bass = _line([(0.5, 2.5, 40), (1.5, 1.5, 40), (3.5, 2.5, 43)], 7.0)
    claimed = [
        dict(start=0.5, pitch=40),
        dict(start=1.5, pitch=40),
        dict(start=3.5, pitch=43),
        dict(start=4.5, pitch=43),
    ]
    rises = muscriptor.rises(claimed, bass, _record(bass))
    assert len(rises) == 4 and all(len(r) == 2 for r in rises)
    for in_bass, in_original in rises[:3]:  # the three plucks
        assert in_bass > 6 and in_original > 6, rises
    assert max(rises[3]) < 2, rises  # the note written inside the held G: nothing rises
    # MuScriptor's starts are up to 20 ms late, and the measure looks that far around them
    late = muscriptor.rises([dict(n, start=n["start"] + 0.02) for n in claimed], bass, _record(bass))
    assert all(min(r) > 6 for r in late[:3]) and max(late[3]) < 2, late


def test_what_rises_in_one_of_the_two_alone_is_seen_as_such():
    bass = _line([(0.5, 4.0, 40)], 5.0)  # one held E
    t = np.arange(len(bass)) / SR
    held = [dict(start=2.0, pitch=40)]
    # another instrument comes in on the E an octave up, the note's second partial: only the original rises
    organ = 0.3 * np.sin(2 * np.pi * 164.8 * t) * (t >= 2.0)
    in_bass, in_original = muscriptor.rises(held, bass, _record(bass) + organ.astype(np.float32))[0]
    assert in_bass < 2 and in_original > 6, (in_bass, in_original)
    # the separation loses the note for a tenth of a second: only the isolated bass rises
    torn = bass * np.where((t >= 1.9) & (t < 2.0), 0.1, 1.0).astype(np.float32)
    in_bass, in_original = muscriptor.rises(held, torn, _record(bass))[0]
    assert in_bass > 6 and in_original < 2, (in_bass, in_original)


def test_the_rise_is_measured_at_the_edges_of_the_recording_and_in_silence():
    bass = _line([(0.0, 1.0, 40), (1.85, 0.1, 45)], 2.0)
    notes = [
        dict(start=0.0, pitch=40),
        dict(start=1.85, pitch=45),
        dict(start=1.99, pitch=45),
        dict(start=2.5, pitch=45),
    ]
    for mix in (
        _record(bass),
        _record(bass)[:SR],
        np.zeros(3 * SR, np.float32),
    ):  # the original shorter, longer, silent
        rises = muscriptor.rises(notes, bass, mix)
        assert len(rises) == 4 and all(np.isfinite(v) and v >= 0 for r in rises for v in r), rises
    assert muscriptor.rises(notes, np.zeros(2 * SR, np.float32), np.zeros(2 * SR, np.float32)) == [(0.0, 0.0)] * 4
    assert muscriptor.rises([], bass, _record(bass)) == []


def test_the_flag_is_decided_by_the_two_finders_and_the_original_has_no_say():
    # Measured (dev/witness_eval.py): the rise in the original does not say which of MuScriptor's notes are wrong
    # on real records. A note bass2sheet's finder does not hear stays flagged however clearly it is plucked.
    sound = _line([(0.5, 0.5, 40), (1.0, 0.5, 43), (1.5, 1.0, 45)], 3.0)
    own = [dict(start=0.5, end=1.0, pitch=40), dict(start=1.0, end=1.5, pitch=42), dict(start=1.5, end=2.5, pitch=45)]
    notes = muscriptor.as_notes([(0.52, 1.0, 40), (1.02, 1.5, 43), (1.52, 2.5, 45)], own, sound, SegParams())
    assert [n["flag"] for n in notes] == [None, "alone", None]
    assert min(muscriptor.rises(notes, sound, _record(sound))[1]) > 6  # plucked, in the bass and in the original
