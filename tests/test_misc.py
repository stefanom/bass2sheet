import json
import os

import numpy as np
import pretty_midi
import pytest
from conftest import note

from bass2sheet import cli, common, pipeline, render, report
from bass2sheet.second_opinion import monophonic


def test_monophonic_keeps_one_note_at_a_time():
    notes = [(0.0, 1.0, 40, 0.5), (0.5, 1.5, 47, 0.9), (0.6, 0.9, 52, 0.2), (2.0, 2.01, 40, 0.5)]
    out = monophonic(notes)
    assert out == [(0.0, 0.5, 40, 0.5), (0.5, 1.5, 47, 0.9)]  # louder note cuts in; the quiet inner and the blip go
    for (_, e0, *_), (s1, *_rest) in zip(out, out[1:]):
        assert e0 <= s1


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/x/Some Band - A Song Title [abc123XYZ_-].mp4", ("A Song Title", "Some Band")),
        ("/x/CAKE - I Will Survive (Official HD Video) [f9rCUQjmkxU].opus", ("I Will Survive", "CAKE")),
        ("Muse - Hysteria [Official Music Video] [3dm_5qWWDV8].opus", ("Hysteria", "Muse")),
        ("The Roots - You Got Me ft. Erykah Badu [MJCHeEQV454].opus", ("You Got Me ft. Erykah Badu", "The Roots")),
        ("Band - Song (Live at Wembley) (Remastered 2011).flac", ("Song (Live at Wembley)", "Band")),
        ("song.wav", ("song", "")),
        ("A - B - C.mp3", ("B - C", "A")),
    ],
)
def test_guess_title(path, expected):
    assert pipeline.guess_title(path) == expected


def test_spelling_and_note_names():
    assert common.spelling(11, False) == common.SHARPS  # B minor -> D major: sharps
    assert common.spelling(5, True) == common.FLATS  # F major: flats
    assert common.note_name(28) == "E1" and common.note_name(46, common.FLATS) == "Bb2"
    assert common.clock(75.04) == "1:15.0" and common.clock(-3) == "0:00.0"


def test_atomic_writes_use_normal_permissions(tmp_path):
    p = tmp_path / "x.json"
    common.atomic_write_text(str(p), "{}")
    assert p.read_text() == "{}" and (p.stat().st_mode & 0o600) == 0o600 and (p.stat().st_mode & 0o044)
    common.atomic_savez(str(tmp_path / "a.npz"), a=np.arange(3))
    assert np.load(tmp_path / "a.npz")["a"].tolist() == [0, 1, 2]
    assert [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")] == []


def test_write_midi_as_played_and_steady(tmp_path):
    notes = [
        note(1.02, 1.5, 40, grid_start=0, grid_len=4),
        note(1.51, 2.0, 43, grid_start=4, grid_len=4, flag="uncertain"),
    ]
    render.write_midi(notes, 120.0, str(tmp_path / "a.mid"), steady=False)
    render.write_midi(notes, 120.0, str(tmp_path / "s.mid"), steady=True)
    played = pretty_midi.PrettyMIDI(str(tmp_path / "a.mid")).instruments[0].notes
    steady = pretty_midi.PrettyMIDI(str(tmp_path / "s.mid")).instruments[0].notes
    assert [round(n.start, 2) for n in played] == [1.02, 1.51] and [n.pitch for n in played] == [40, 43]
    assert [round(n.start, 3) for n in steady] == [0.0, 0.5] and round(steady[0].end, 3) == 0.5
    assert played[1].velocity < played[0].velocity  # flagged notes are quieter


def test_pluck_sounds_at_the_written_pitch():
    w = render.pluck(110.0, 0.5, 100)
    spectrum = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1 / common.SR)
    assert abs(freqs[spectrum.argmax()] - 110.0) < 3


def test_report_renders_verdict_gates_and_checks(tmp_path):
    gate = dict(
        name="coverage",
        label="Sounding bass covered by notes",
        value=0.91,
        warn=0.95,
        fail=0.85,
        higher_is_better=True,
        meaning="m",
        status="warn",
        note="",
    )
    data = dict(
        version="0",
        title="T <x>",
        artist="A",
        input="in.wav",
        verdict="WARN",
        gates=[gate],
        ear_checks=[dict(bar=5, time="0:09.0", severity=4, reasons=["notes missing: 60%"])],
        attempts=[dict(method="roformer", outcome="transcribed: WARN")],
        warnings=["w"],
        files={"bass.pdf": "score"},
        check_severity=3,
        timings=dict(total=12.0),
        summary=dict(duration="0:26.0", notes=72, bars=12, tempo=120.0, key="E minor", separator="roformer"),
    )
    report.write(str(tmp_path / "r.html"), str(tmp_path / "r.json"), data)
    html = (tmp_path / "r.html").read_text()
    assert "WARN" in html and "T &lt;x&gt;" in html and "91%" in html and "notes missing: 60%" in html
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "WARN"
    from bass2sheet.report import threshold

    assert threshold(gate) == "pass ≥ 95%; fail beyond 85%"
    assert threshold(dict(gate, fail=None)) == "pass ≥ 95%; warns only"


def test_cli_reports_a_missing_file(capsys):
    assert cli.main(["/nonexistent/song.mp3"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_reports_unreadable_input(tmp_path, capsys):
    p = tmp_path / "fake.wav"
    p.write_text("nope")
    assert cli.main([str(p), "-o", str(tmp_path / "out")]) == 1
    assert "bass2sheet:" in capsys.readouterr().err


def test_candidates_rank_by_failures_then_confirmed_note_starts_then_coverage():
    from bass2sheet.pipeline import rank_candidate
    from bass2sheet.quality import Gate

    def gates(fails=0, warns=0, onsets=None, coverage=0.9):
        out = [Gate(f"f{k}", "F", 0.0, 0, 0, True, "", status="fail") for k in range(fails)]
        out += [Gate(f"w{k}", "W", 0.0, 0, 0, True, "", status="warn") for k in range(warns)]
        out.append(Gate("second_model_onsets", "S", onsets, 0, 0, True, "", status="pass" if onsets else "skipped"))
        out.append(Gate("coverage", "C", coverage, 0, 0, True, "", status="pass"))
        return out

    best = lambda *cands: min(range(len(cands)), key=lambda k: rank_candidate(cands[k]))
    assert best(gates(fails=1, onsets=0.9), gates(fails=0, onsets=0.5)) == 1  # a failed gate outweighs the rest
    assert best(gates(warns=5, onsets=0.70), gates(warns=3, onsets=0.58)) == 0  # warnings are not counted
    assert best(gates(coverage=0.90), gates(coverage=0.95)) == 1  # no second model: coverage decides


def test_notes_played_together_sound_and_are_exported_together(tmp_path):
    import pretty_midi

    from bass2sheet import render

    notes = [
        dict(start=0.5, end=1.0, pitch=33, also=[40], flag=None, grid_start=0, grid_len=4),
        dict(start=1.0, end=1.5, pitch=36, flag=None, grid_start=4, grid_len=4),
    ]
    for steady in (False, True):
        path = str(tmp_path / f"{steady}.mid")
        render.write_midi(notes, 120.0, path, steady)
        got = sorted((round(n.start, 2), n.pitch) for n in pretty_midi.PrettyMIDI(path).instruments[0].notes)
        t0 = 0.0 if steady else 0.5
        assert got == [(t0, 33), (t0, 40), (t0 + 0.5, 36)]
    both, alone = render.synth(notes, 2 * 44100), render.synth([dict(notes[0], also=[]), notes[1]], 2 * 44100)
    first = slice(int(0.5 * 44100), int(1.0 * 44100))
    assert (both[first] ** 2).sum() > 1.5 * (alone[first] ** 2).sum()
    assert (both[int(1.1 * 44100) :] == alone[int(1.1 * 44100) :]).all()


def test_the_choice_for_notes_played_together_reaches_the_note_finder():
    import pytest

    from bass2sheet.common import ALTERNATIVES, DEFAULTS, Bass2SheetError
    from bass2sheet.pipeline import seg_params

    assert DEFAULTS["together"] == "write" and ALTERNATIVES["together"] == ("write", "single")
    assert seg_params("every").together and not seg_params("every", "single").together
    assert seg_params("clear", "write").strict_replucks
    with pytest.raises(Bass2SheetError):
        seg_params("every", "both")


def test_note_names_parse_to_midi_numbers():
    import pytest

    from bass2sheet.common import LOWEST_NOTE, Bass2SheetError, note_name, note_number
    from bass2sheet.pipeline import seg_params

    assert [note_number(x) for x in ("E1", "B0", "D1", "F#2", "Bb0", "C4", "e1")] == [28, 23, 26, 42, 22, 60, 28]
    assert all(note_number(note_name(m)) == m for m in range(20, 70))
    for bad in ("H2", "E", "1E", ""):
        with pytest.raises(Bass2SheetError):
            note_number(bad)
    assert LOWEST_NOTE == 28 and seg_params("every").lowest == 28
    assert seg_params("every", lowest=23).lowest == 23 and seg_params("every", lowest=0).lowest == 0


def test_cli_hands_the_choices_to_the_pipeline_and_its_exit_code_back(tmp_path, monkeypatch, capsys):
    song = tmp_path / "Some Band - A Song [abc123].wav"
    song.write_bytes(b"not read here")
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: calls.append((a, kw)) or pipeline.EXIT_FAILED_GATES)
    asked = [str(song), "--lowest", "B0", "--rhythm", "simple", "--tab", "position", "--replucks", "clear"]
    asked += ["--together", "single", "--notes", "muscriptor", "--strict", "--no-audio", "--no-pdf", "-v"]
    assert cli.main(asked) == 2  # the gates failed: the command says so
    (audio_path, out), kw = calls[0]
    assert audio_path == str(song) and out == str(tmp_path / "Some Band - A Song bass")  # beside the song
    chosen = dict(lowest=23, rhythm="simple", tab="position", replucks="clear", together="single", notes="muscriptor")
    assert {k: kw[k] for k in chosen} == chosen
    assert (kw["strict"], kw["renders"], kw["pdf"]) == (True, False, False)
    assert cli.main([str(song), "-o", str(tmp_path / "out"), "--lowest", "none"]) == 2
    (_, out), kw = calls[1]
    assert out == str(tmp_path / "out") and kw["lowest"] == 0  # the notes as heard, however low
    left = dict(replucks=None, together=None, rhythm=None, tab=None, notes=None, stems_dir=None, bass_stem=None)
    assert {k: kw[k] for k in left} == left  # what is not said is left to the standing choices
    assert (kw["strict"], kw["renders"], kw["pdf"], kw["separator"]) == (False, True, True, "auto")
    assert cli.main([str(song), "--lowest", "H9"]) == 1 and len(calls) == 2
    assert "bass2sheet: 'H9' is not a note name (examples: E1, Bb0, F#2)" in capsys.readouterr().err

    def stopped(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run", stopped)
    assert cli.main([str(song)]) == 130  # Ctrl-C ends it without a traceback


def test_giving_gpu_memory_back_never_breaks_a_run(monkeypatch):
    import torch

    freed = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append("cuda"))
    common.release_gpu_memory()
    assert freed == ["cuda"]

    def broken():
        raise RuntimeError("the driver is gone")

    monkeypatch.setattr(torch.cuda, "empty_cache", broken)
    common.release_gpu_memory()  # housekeeping: what goes wrong in it is not the run's problem
