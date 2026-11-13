"""End to end on the synthetic song, with its stems given (no separation model, no network): the whole pipeline
from audio files to MusicXML, MIDI, renders and the report, scored against the song's known notes.

CREPE's `tiny` model keeps this to seconds; set BASS2SHEET_TEST_CREPE=full to use the production model."""

import json
import os
import xml.dom.minidom

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from bass2sheet import cli, evaluate, pipeline
from bass2sheet.score import find_mscore

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


@pytest.fixture(scope="module")
def result(song, song_files, cache_dir, tmp_path_factory):
    mix, stems = song_files
    out = str(tmp_path_factory.mktemp("out"))
    code = pipeline.run(
        mix,
        out,
        title="Synthetic Song",
        artist="bass2sheet tests",
        stems_dir=stems,
        second_mode="off",
        crepe_model=CREPE,
        beat_tracker="librosa",  # the Beat This! model needs a download: the full tier covers it
        cache_dir=cache_dir,
        device="cpu",
        pdf=False,
    )
    data = json.load(open(os.path.join(out, "notes.json")))
    rep = json.load(open(os.path.join(out, "report.json")))
    est = [(n["start"], n["end"], n["pitch"]) for n in data["notes"]]
    grid = dict(beats=song["beats"], first_downbeat_beat=0, four_four=True)
    metrics = evaluate.score(song["notes"], est, [(n["grid_start"], n["bar"]) for n in data["notes"]], grid, set())
    return dict(code=code, out=out, data=data, report=rep, metrics=metrics)


def test_exit_code_and_verdict(result):
    assert result["code"] == 0 and result["report"]["verdict"] in ("PASS", "WARN")
    assert not [g["name"] for g in result["report"]["gates"] if g["status"] == "fail"]


def test_notes_match_the_known_bass_line(result):
    m = result["metrics"]
    assert m["f"] >= 0.93, m
    assert m["f_note_name"] >= 0.95 and m["pitch_octave_up"] + m["pitch_octave_down"] <= 0.03
    assert abs(m["onset_bias_ms"]) < 15  # starts sit on the plucks, not a frame or two late


def test_rhythm_grid_matches(result, song):
    m, data = result["metrics"], result["data"]
    assert abs(data["tempo_bpm"] - song["tempo"]) / song["tempo"] < 0.02 and data["tracked_pulse"] == "quarter"
    assert m["bar_position_acc"] >= 0.95 and m["grid_shift_mode"] == 0
    assert data["key"] in ("E minor", "G major")


def test_outputs_are_complete_and_valid(result, song):
    out = result["out"]
    for name in (
        "bass.musicxml",
        "bass.mid",
        "bass_as_played.mid",
        "notes.json",
        "report.html",
        "report.json",
        "audio/bass_stem.wav",
        "audio/synth_bass.wav",
        "audio/song_resynth.wav",
        "audio/bass_ab_lr.wav",
        "audio/bass_rebuilt.wav",
    ):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    doc = xml.dom.minidom.parse(os.path.join(out, "bass.musicxml"))
    assert doc.getElementsByTagName("movement-title")[0].firstChild.data == "Synthetic Song"
    for measure in doc.getElementsByTagName("measure"):
        total = {"1": 0, "2": 0}
        for n in measure.getElementsByTagName("note"):
            if not n.getElementsByTagName("grace"):
                total[n.getElementsByTagName("staff")[0].firstChild.data] += int(
                    n.getElementsByTagName("duration")[0].firstChild.data
                )
        assert total == {"1": 16, "2": 16}
    played = pretty_midi.PrettyMIDI(os.path.join(out, "bass_as_played.mid")).instruments[0].notes
    assert len(played) == len(result["data"]["notes"])
    info = sf.info(os.path.join(out, "audio/song_resynth.wav"))
    assert abs(info.duration - song["mix"].shape[1] / song["sr"]) < 0.1 and info.channels == 2
    rebuilt = sf.info(os.path.join(out, "audio/bass_rebuilt.wav"))  # the notes played with the bass's own sound
    assert abs(rebuilt.duration - info.duration) < 0.1 and rebuilt.channels == 1
    own = next(g for g in result["report"]["gates"] if g["name"] == "own_sound")
    assert own["status"] == "pass" and own["value"] < 0.05
    assert all(n["own_sound"] > 0.6 and "sounds_like" not in n for n in result["data"]["notes"])
    assert "Synthetic Song" in open(os.path.join(out, "report.html")).read()
    assert set(result["report"]["files"]) >= {"bass.musicxml", "bass.mid", "notes.json", "report.html"}


@pytest.mark.skipif(find_mscore() is None, reason="MuseScore 4 is not installed")
def test_pdf_is_engraved(result, tmp_path):
    from bass2sheet.score import engrave

    pdf = str(tmp_path / "bass.pdf")
    assert engrave(os.path.join(result["out"], "bass.musicxml"), pdf, find_mscore()) is None
    assert open(pdf, "rb").read(5) == b"%PDF-"


def test_cli_runs_end_to_end_from_the_cache(result, song_files, cache_dir, tmp_path):
    mix, stems = song_files
    out = str(tmp_path / "cli-out")
    code = cli.main(
        [
            mix,
            "-o",
            out,
            "--stems",
            stems,
            "--second-opinion",
            "off",
            "--beat-tracker",
            "librosa",
            "--crepe-model",
            CREPE,
            "--cache-dir",
            cache_dir,
            "--device",
            "cpu",
            "--no-audio",
            "--no-pdf",
            "--title",
            "CLI run",
        ]
    )
    assert code == 0
    assert json.load(open(os.path.join(out, "notes.json")))["notes"] == result["data"]["notes"]  # deterministic
    assert not os.path.exists(os.path.join(out, "audio"))


def _stems_with(tmp_path, song, **replace):
    d = tmp_path / "stems"
    d.mkdir()
    for name, y in {**song["stems"], **replace}.items():
        sf.write(str(d / f"{name}.wav"), y.T, song["sr"], subtype="PCM_16")
    return str(d)


def test_drums_in_place_of_bass_do_not_pass(song, song_files, cache_dir, tmp_path, capsys):
    """Garbage in the bass stem must end in an error or a FAIL verdict, never a sheet that claims to be fine."""
    stems = _stems_with(tmp_path, song, bass=song["stems"]["drums"])
    out = str(tmp_path / "out")
    code = cli.main(
        [
            song_files[0],
            "-o",
            out,
            "--stems",
            stems,
            "--second-opinion",
            "off",
            "--beat-tracker",
            "librosa",
            "--crepe-model",
            CREPE,
            "--cache-dir",
            cache_dir,
            "--device",
            "cpu",
            "--no-audio",
            "--no-pdf",
            "--strict",
        ]
    )
    assert code in (1, 2)
    assert not os.path.exists(os.path.join(out, "bass.musicxml"))  # strict mode writes no sheet on FAIL
    if code == 2:
        assert json.load(open(os.path.join(out, "report.json")))["verdict"] == "FAIL"


def test_silent_bass_is_an_error(song, song_files, cache_dir, tmp_path, capsys):
    stems = _stems_with(tmp_path, song, bass=np.zeros_like(song["stems"]["bass"]))
    code = cli.main(
        [
            song_files[0],
            "-o",
            str(tmp_path / "out"),
            "--stems",
            stems,
            "--second-opinion",
            "off",
            "--cache-dir",
            cache_dir,
            "--device",
            "cpu",
        ]
    )
    assert code == 1 and "no bass found" in capsys.readouterr().err


def test_missing_stems_are_an_error(song_files, cache_dir, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code = cli.main([song_files[0], "-o", str(tmp_path / "out"), "--stems", str(empty), "--cache-dir", cache_dir])
    assert code == 1 and "no bass stem" in capsys.readouterr().err


def test_notes_played_together_come_out_as_chords(cache_dir, tmp_path):
    """A song with three double stops: the sheet, the MIDI and the notes have both notes of each, and no others;
    with --together single the tracker's one note is written (here the upper, which is not the bass note)."""
    from songsynth import make_song, write_song

    song = make_song(double_stops=True)
    mix, stems = write_song(song, str(tmp_path / "song"))
    kw = dict(stems_dir=stems, second_mode="off", crepe_model=CREPE, beat_tracker="librosa", device="cpu")
    kw.update(cache_dir=cache_dir, pdf=False, renders=False)
    out = {}
    for choice in ("write", "single"):
        out[choice] = str(tmp_path / choice)
        assert pipeline.run(mix, out[choice], together=choice, **kw) == 0
    data = json.load(open(os.path.join(out["write"], "notes.json")))
    pairs = [(n["start"], [n["pitch"], *n["also"]]) for n in data["notes"] if n.get("also")]
    assert [p for _, p in pairs] == [p for _, p in song["together"]]
    assert all(abs(a - b) < 0.03 for (a, _), (b, _) in zip(pairs, song["together"]))
    assert data["written"]["together"] == "write" and len(data["notes"]) == len(song["notes"])
    doc = xml.dom.minidom.parse(os.path.join(out["write"], "bass.musicxml"))
    chords = [n for n in doc.getElementsByTagName("note") if n.getElementsByTagName("chord")]
    assert len(chords) == 2 * len(song["together"])  # on the staff and in the tab
    played = pretty_midi.PrettyMIDI(os.path.join(out["write"], "bass_as_played.mid")).instruments[0].notes
    assert len(played) == len(song["notes"]) + len(song["together"])
    single = json.load(open(os.path.join(out["single"], "notes.json")))
    assert not any(n.get("also") for n in single["notes"]) and single["written"]["together"] == "single"
    at = lambda t: [n["pitch"] for n in single["notes"] if abs(n["start"] - t) < 0.05]
    assert all(at(t) == [p[1]] for t, p in song["together"])
    assert "<chord/>" not in open(os.path.join(out["single"], "bass.musicxml")).read()
