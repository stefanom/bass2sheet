import json
import os

import numpy as np
import pretty_midi
import pytest

from bass2sheet import cli, common, pipeline, render, report
from bass2sheet.second_opinion import monophonic
from conftest import note


def test_monophonic_keeps_one_note_at_a_time():
    notes = [(0.0, 1.0, 40, 0.5), (0.5, 1.5, 47, 0.9), (0.6, 0.9, 52, 0.2), (2.0, 2.01, 40, 0.5)]
    out = monophonic(notes)
    assert out == [(0.0, 0.5, 40, 0.5), (0.5, 1.5, 47, 0.9)]  # louder note cuts in; the quiet inner and the blip go
    for (s0, e0, *_), (s1, *_rest) in zip(out, out[1:]):
        assert e0 <= s1


@pytest.mark.parametrize("path,expected", [
    ("/x/Wilco - How To Fight Loneliness [7CAYFIpi89k].mp4", ("How To Fight Loneliness", "Wilco")),
    ("song.wav", ("song", "")),
    ("A - B - C.mp3", ("B - C", "A")),
])
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
    notes = [note(1.02, 1.5, 40, grid_start=0, grid_len=4), note(1.51, 2.0, 43, grid_start=4, grid_len=4, flag="uncertain")]
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
    gate = dict(name="coverage", label="Sounding bass covered by notes", value=0.91, warn=0.95, fail=0.85,
                higher_is_better=True, meaning="m", status="warn", note="")
    data = dict(version="0", title="T <x>", artist="A", input="in.wav", verdict="WARN", gates=[gate],
                ear_checks=[dict(bar=5, time="0:09.0", severity=4, reasons=["notes missing: 60%"])],
                attempts=[dict(method="roformer", outcome="transcribed: WARN")], warnings=["w"], files={"bass.pdf": "score"},
                check_severity=3, timings=dict(total=12.0),
                summary=dict(duration="0:26.0", notes=72, bars=12, tempo=120.0, key="E minor", separator="roformer"))
    report.write(str(tmp_path / "r.html"), str(tmp_path / "r.json"), data)
    html = (tmp_path / "r.html").read_text()
    assert "WARN" in html and "T &lt;x&gt;" in html and "91%" in html and "notes missing: 60%" in html
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "WARN"


def test_cli_reports_a_missing_file(capsys):
    assert cli.main(["/nonexistent/song.mp3"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_reports_unreadable_input(tmp_path, capsys):
    p = tmp_path / "fake.wav"
    p.write_text("nope")
    assert cli.main([str(p), "-o", str(tmp_path / "out")]) == 1
    assert "bass2sheet:" in capsys.readouterr().err
