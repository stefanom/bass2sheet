"""The pipeline's own steps, around the models: which separator is tried and which result is kept, what a failed
step leaves behind, the beat tracker's fallback, the second opinion, the PDF. A short synthetic song goes in; where
a model or a program would run, a stand-in does (the opt-in full tier runs the real ones: tests/test_e2e_full.py)."""

import json
import os
import shutil
import stat
import sys

import numpy as np
import pytest
import soundfile as sf

from bass2sheet import cli, pipeline, score
from bass2sheet.common import SR, Bass2SheetError
from bass2sheet.quality import Gate
from bass2sheet.separate import load_stems

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")
BARS = 6


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """A song of six bars and its files: (the song, its mix, the folder of its stems)."""
    from songsynth import make_song, write_song

    song = make_song(bars=BARS)
    return (song, *write_song(song, str(tmp_path_factory.mktemp("short"))))


@pytest.fixture
def run(short, cache_dir, tmp_path):
    """Runs the pipeline on the short song with the quick settings and no model: the stems are given unless a
    separator is named. Returns (exit code, the report, the output folder)."""
    count = iter(range(100))

    def go(**kw):
        out = str(tmp_path / f"out-{next(count)}")
        settings = dict(second_mode="off", crepe_model=CREPE, beat_tracker="librosa", cache_dir=cache_dir)
        settings.update(device="cpu", pdf=False, renders=False)
        if "separator" not in kw:
            settings["stems_dir"] = short[2]
        code = pipeline.run(short[1], out, **dict(settings, **kw))
        return code, json.load(open(os.path.join(out, "report.json"))), out

    return go


def _separators(monkeypatch, short, **by_method):
    """Stand-ins for the separation models: each gives the song's own stems, changed by its function, or fails
    with its exception. Returns the list of the methods the pipeline asks for, as it asks."""
    asked = []

    def separate(method, wav, mix_path, audio_id, cache_dir, device):
        asked.append(method)
        if isinstance(by_method[method], Exception):
            raise by_method[method]
        stems = load_stems(short[2], wav.shape[1])
        return dict(stems, **by_method[method](stems))

    monkeypatch.setattr(pipeline, "separate", separate)
    return asked


good = lambda stems: {}
no_bass = lambda stems: dict(bass=np.zeros_like(stems["bass"]))


def _failing(monkeypatch, *counts):
    """Makes the transcription of the k-th separator tried fail counts[k] gates more than it does by itself."""
    real, calls = pipeline.transcribe, iter(counts)

    def transcribe(*args, **kw):
        c = real(*args, **kw)
        extra = [Gate(f"made_to_fail_{k}", "A gate made to fail", 0.0, 1, 1, True, "") for k in range(next(calls))]
        return dict(c, gates=c["gates"] + [g.judge() for g in extra])

    monkeypatch.setattr(pipeline, "transcribe", transcribe)


def _outcomes(report):
    return [(a["method"], a["outcome"].split(":")[0].split(" (")[0]) for a in report["attempts"]]


def _program(tmp_path, name, body):
    path = str(tmp_path / name)
    with open(path, "w") as f:
        f.write(f"#!{sys.executable}\nimport sys\n{body}")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


# ------------------------------------------------------------------------------------------ which separator


def test_the_short_song_comes_out_right_with_its_stems_given(run, short):
    code, report, out = run()
    data = json.load(open(os.path.join(out, "notes.json")))
    assert code == 0 and report["verdict"] in ("PASS", "WARN") and report["summary"]["separator"] == "given stems"
    assert abs(len(data["notes"]) - len(short[0]["notes"])) <= 1
    assert report["summary"]["bars"] == BARS + 1  # the bar the record starts in, which has no note, and the song's
    assert _outcomes(report) == [("given stems", "transcribed")]


def test_when_the_first_separator_cannot_run_the_other_one_is_used(run, monkeypatch, short):
    broken = RuntimeError("its model could not be fetched")
    asked = _separators(monkeypatch, short, roformer=broken, demucs=good)
    code, report, _ = run(separator="auto")
    assert code == 0 and asked == ["roformer", "demucs"] and report["summary"]["separator"] == "demucs"
    assert _outcomes(report) == [("roformer", "separation failed"), ("demucs", "transcribed")]
    assert "its model could not be fetched" in report["attempts"][0]["outcome"]


def test_a_separator_that_is_asked_for_by_name_is_the_only_one_tried(run, monkeypatch, short):
    asked = _separators(monkeypatch, short, roformer=good, demucs=RuntimeError("no such model"))
    code, report, _ = run(separator="roformer")
    assert code == 0 and asked == ["roformer"] and report["summary"]["separator"] == "roformer"
    with pytest.raises(Bass2SheetError, match="no usable transcription: demucs: separation failed: no such model"):
        run(separator="demucs")
    assert asked == ["roformer", "demucs"]  # and roformer was not tried in its place


def test_a_separator_that_finds_no_bass_gives_way_to_the_next(run, monkeypatch, short, tmp_path, capsys):
    asked = _separators(monkeypatch, short, roformer=no_bass, demucs=good)
    code, report, _ = run(separator="auto")
    assert code == 0 and asked == ["roformer", "demucs"] and report["summary"]["separator"] == "demucs"
    first = report["attempts"][0]["outcome"]
    assert first.startswith("no bass found (Bass stem level vs the mix: -") and first.endswith("next separator")
    _separators(monkeypatch, short, roformer=no_bass, demucs=no_bass)
    with pytest.raises(Bass2SheetError) as e:
        run(separator="auto")
    said = str(e.value)  # each attempt by name; the last one has no next separator to speak of
    assert said.startswith("no usable transcription: roformer: no bass found (") and "; demucs: no bass" in said
    assert said.count("trying the next separator") == 1 and said.endswith("dB)")
    out = str(tmp_path / "from the command line")
    assert cli.main([short[1], "-o", out, "--cache-dir", str(tmp_path / "cache"), "--device", "cpu"]) == 1
    assert "bass2sheet: no usable transcription" in capsys.readouterr().err
    assert not os.path.exists(os.path.join(out, "notes.json"))


def test_a_result_that_fails_its_gates_sends_the_song_to_the_other_separator(run, monkeypatch, short):
    asked = _separators(monkeypatch, short, roformer=good, demucs=good)
    _failing(monkeypatch, 1, 0)
    code, report, out = run(separator="auto")
    assert code == 0 and asked == ["roformer", "demucs"] and report["summary"]["separator"] == "demucs"
    assert _outcomes(report) == [("roformer", "transcribed"), ("demucs", "transcribed"), ("demucs", "chosen")]
    assert report["attempts"][0]["outcome"].startswith("transcribed: FAIL: ")
    assert "A gate made to fail (fail)" in report["attempts"][0]["outcome"]
    assert "made_to_fail_0" not in {g["name"] for g in report["gates"]}  # the report is the kept result's
    assert "DRAFT" not in open(os.path.join(out, "bass.musicxml")).read()


@pytest.mark.parametrize("failed,kept", [((2, 1), "demucs"), ((1, 2), "roformer")])
def test_of_two_failed_results_the_one_with_fewer_failed_gates_is_kept_as_a_draft(
    run, monkeypatch, short, failed, kept
):
    _separators(monkeypatch, short, roformer=good, demucs=good)
    _failing(monkeypatch, *failed)
    code, report, out = run(separator="auto")
    assert code == pipeline.EXIT_FAILED_GATES and report["verdict"] == "FAIL"
    assert report["summary"]["separator"] == kept and report["attempts"][-1]["method"] == kept
    assert sum(g["status"] == "fail" for g in report["gates"]) == min(failed)
    assert "DRAFT: failed quality gates" in open(os.path.join(out, "bass.musicxml")).read()


def test_strict_writes_no_sheet_for_a_failed_result_and_says_so(run, monkeypatch):
    _failing(monkeypatch, 1)
    code, report, out = run(strict=True)
    assert code == pipeline.EXIT_FAILED_GATES and report["verdict"] == "FAIL"
    assert "strict mode: the result failed its quality gates, so no sheet was written" in report["warnings"]
    assert sorted(os.listdir(out)) == ["notes.json", "report.html", "report.json"]  # what was heard, and why not
    assert not {"bass.musicxml", "bass.mid", "bass.pdf"} & set(report["files"])


# ------------------------------------------------------------------------------------- what goes in, what is refused


def test_choices_the_tool_does_not_have_are_refused_by_name(run):
    for choice, said in (
        (dict(rhythm="swung"), "unknown rhythm 'swung'; expected one of literal, simple"),
        (dict(tab="high"), "unknown tab 'high'; expected one of low, position"),
        (dict(replucks="some"), "unknown re-pluck policy 'some'"),
        (dict(notes="basic"), "unknown note finder 'basic'"),
    ):
        with pytest.raises(Bass2SheetError, match=said):
            run(**choice)


def test_a_bass_track_given_for_evaluation_takes_the_place_of_the_separated_one(run, short, tmp_path):
    song, _, stems = short
    mine = tmp_path / "stems"
    mine.mkdir()
    for name in ("drums.wav", "harmony.wav"):
        shutil.copy(os.path.join(stems, name), mine / name)
    drums = song["stems"]["drums"]
    sf.write(str(mine / "bass.wav"), drums.T, SR, subtype="PCM_16")  # what a separator might give for a bass
    given = str(tmp_path / "clean bass.flac")
    sf.write(given, song["stems"]["bass"].T[:-SR], SR)  # a second shorter than the song: fitted to it
    code, report, out = run(stems_dir=str(mine), bass_stem=given)
    assert code == 0 and report["summary"]["separator"] == "given stems + given bass stem"
    notes = json.load(open(os.path.join(out, "notes.json")))["notes"]
    assert abs(len(notes) - len(song["notes"])) <= 1
    truth = sorted(song["notes"])
    assert sum(abs(n["start"] - t[0]) < 0.05 and n["pitch"] == t[2] for n, t in zip(notes, truth)) >= len(truth) - 2


def test_a_bass_track_with_next_to_no_notes_is_an_error_and_not_a_sheet(run, short, tmp_path):
    song, _, stems = short
    mine = tmp_path / "stems"
    shutil.copytree(stems, mine)
    bass = song["stems"]["bass"].copy()
    bass[:, int(song["downbeats"][1] * SR) :] = 0  # one bar of bass and then nothing
    sf.write(str(mine / "bass.wav"), bass.T, SR, subtype="PCM_16")
    said = r"no usable transcription: given stems: only \d bass notes found; the stem has no clearly pitched bass"
    with pytest.raises(Bass2SheetError, match=said):
        run(stems_dir=str(mine))


# ----------------------------------------------------------------------------------------------- beats and bars


def test_the_models_beats_are_used_when_it_runs_and_the_built_in_tracker_when_it_does_not(short, tmp_path, monkeypatch):
    song = short[0]
    stems, mix = load_stems(short[2], song["mix"].shape[1]), song["mix"].mean(0)
    asked = []

    def model(mix_mono, device, cache_path):
        asked.append((len(mix_mono), device, cache_path))
        return song["beats"], song["downbeats"]

    grid = lambda tracker: pipeline.track_grid(stems, mix, "0123456789abcdef-and-more", str(tmp_path), "cpu", tracker)
    monkeypatch.setattr(pipeline, "model_beats", model)
    tempo, pulses, per_quarter, phase, fit, tracker = grid("auto")
    assert tracker == "beat_this" and tempo == pytest.approx(120, abs=0.1) and (per_quarter, phase, fit) == (1, 0, 1.0)
    assert (pulses == song["beats"]).all()
    assert asked == [(len(mix), "cpu", str(tmp_path / "beats-0123456789abcdef.npz"))]  # kept by the song

    monkeypatch.setattr(pipeline, "model_beats", lambda *a: (song["beats"][:10], song["downbeats"][:3]))
    tempo, _, _, _, fit, tracker = grid("auto")  # too few beats to go by: the built-in tracker instead
    assert tracker == "librosa" and fit is None and tempo == pytest.approx(120, rel=0.02)

    def broken(*a):
        raise OSError("no network for the first download")

    monkeypatch.setattr(pipeline, "model_beats", broken)
    assert grid("auto")[-1] == "librosa"
    with pytest.raises(Bass2SheetError, match="the Beat This! beat tracker failed: no network for the first download"):
        grid("beat_this")  # asked for by name, it is not replaced quietly
    monkeypatch.setattr(pipeline, "model_beats", lambda *a: pytest.fail("the model was asked"))
    assert grid("librosa")[-1] == "librosa"
    few = {k: v[:, : 5 * SR] for k, v in stems.items()}  # five seconds: ten beats
    with pytest.raises(Bass2SheetError, match="could not find a steady beat"):
        pipeline.track_grid(few, mix[: 5 * SR], "x", str(tmp_path), "cpu", "librosa")


def test_with_the_models_beats_the_meter_gate_is_judged_and_the_standing_choice_decides_auto(run, short, monkeypatch):
    song = short[0]
    monkeypatch.setattr(pipeline, "model_beats", lambda *a: (song["beats"], song["downbeats"]))
    code, report, out = run(beat_tracker="auto")
    meter = next(g for g in report["gates"] if g["name"] == "meter_fit")
    assert code == 0 and (meter["status"], meter["value"]) == ("pass", 1.0)
    assert json.load(open(os.path.join(out, "notes.json")))["beat_tracker"] == "beat_this"
    monkeypatch.setitem(pipeline.DEFAULTS, "beat_tracker", "librosa")  # as picks in bass2sheet-judge can settle it
    monkeypatch.setattr(pipeline, "model_beats", lambda *a: pytest.fail("the model was asked"))
    code, report, out = run(beat_tracker="auto")
    assert json.load(open(os.path.join(out, "notes.json")))["beat_tracker"] == "librosa"
    assert next(g for g in report["gates"] if g["name"] == "meter_fit")["status"] == "skipped"


# ----------------------------------------------------------------------------------- the second opinion and the PDF


def test_the_second_opinion_is_asked_and_its_gates_are_judged(run, short, tmp_path, cache_dir):
    rows = "".join(f"{s:.3f},{e:.3f},{p},0.500\\n" for s, e, p in short[0]["notes"])
    asked = str(tmp_path / "asked.txt")
    program = _program(
        tmp_path,
        "python-with-basic-pitch",
        f"open({asked!r}, 'w').write(' '.join(sys.argv[1:]))\n"
        f"open(sys.argv[3], 'w').write('start,end,pitch,amplitude\\n{rows}')\n",
    )
    try:
        code, report, _ = run(second_mode="auto", bp_python=program)
    finally:
        for name in os.listdir(cache_dir):
            if name.startswith("basicpitch-"):
                os.remove(os.path.join(cache_dir, name))  # kept by the track: not for the tests after this one
    gates = {g["name"]: g for g in report["gates"]}
    assert code == 0
    assert gates["second_model_pitch"]["status"] == "pass" and gates["second_model_pitch"]["value"] > 0.97
    assert gates["second_model_onsets"]["status"] == "pass" and gates["second_model_onsets"]["value"] > 0.9
    runner, track, _ = open(asked).read().split(" ")
    assert runner.endswith("bp_runner.py") and os.path.dirname(track) == cache_dir
    code, report, _ = run()
    off = {g["name"]: g for g in report["gates"]}["second_model_pitch"]
    assert (off["status"], off["note"]) == ("skipped", "turned off (--second-opinion off)")


def _mscore(tmp_path, fails=False):
    """A program that takes MuseScore's place for `-S STYLE -o PDF MUSICXML`."""
    if fails:
        body = "sys.stderr.write('sandbox_extension: noise\\nno such font\\n')\nsys.exit(3)\n"
    else:
        body = (
            "assert sys.argv[1] == '-S' and sys.argv[3] == '-o', sys.argv\n"
            "assert '<pageWidth>' in open(sys.argv[2]).read() and '<score-partwise' in open(sys.argv[5]).read()\n"
            "open(sys.argv[4], 'wb').write(b'%PDF-1.4 stand-in')\n"
        )
    return _program(tmp_path, "mscore", body)


def test_the_pdf_is_engraved_by_the_musescore_that_is_named(run, tmp_path):
    code, report, out = run(pdf=True, mscore=_mscore(tmp_path), rhythm="simple", tab="position")
    assert code == 0 and open(os.path.join(out, "bass.pdf"), "rb").read(5) == b"%PDF-"
    assert report["files"]["bass.pdf"] == "printable score: bass clef + tab"
    assert ".bass2sheet.mss" not in os.listdir(out)  # the style it was given is not left behind
    written = json.load(open(os.path.join(out, "notes.json")))["written"]
    assert (written["rhythm"], written["tab"]) == ("simple", "position")


def test_without_a_pdf_the_run_still_succeeds_and_the_report_says_why(run, tmp_path, monkeypatch):
    code, report, out = run(pdf=True, mscore=_mscore(tmp_path, fails=True))
    assert code == 0 and "bass.pdf" not in report["files"] and "bass.pdf" not in os.listdir(out)
    assert "PDF not written: MuseScore exited 3: no such font" in report["warnings"]
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", [])  # a machine without MuseScore
    code, report, out = run(pdf=True)
    assert code == 0 and "bass.pdf" not in report["files"] and "bass.musicxml" in report["files"]
    assert any(w.startswith("PDF not written: MuseScore 4 not found (install it, or pass") for w in report["warnings"])
