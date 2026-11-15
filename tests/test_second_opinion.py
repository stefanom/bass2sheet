"""The second opinion: Basic Pitch in an interpreter of its own, asked through bp_runner.py. A stand-in takes the
place of the basic-pitch package (which is not installed where the tests run), so what is tested is everything
around the model: what the runner asks of it, the file it writes, and how that file is read back."""

import json
import os
import stat
import sys

import numpy as np
import pytest
import soundfile as sf

from bass2sheet import second_opinion
from bass2sheet.common import SR

STEM_ID = "0123456789abcdef0123"


def _basic_pitch(tmp_path, monkeypatch, events):
    """Puts a package named basic_pitch where an interpreter started from here finds it. Its `predict` answers
    with `events` and writes down what it was asked. Returns the file it writes that to."""
    package, asked = tmp_path / "stand-in" / "basic_pitch", str(tmp_path / "asked.json")
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "inference.py").write_text(
        "import json\n"
        "import soundfile as sf\n"
        "class Model:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "def predict(stem, model, **kw):\n"
        "    info = sf.info(stem)\n"
        "    said = dict(kw, model=model.path, rate=info.samplerate, channels=info.channels, frames=info.frames)\n"
        f"    json.dump(said, open({asked!r}, 'w'))\n"
        f"    return None, None, {events!r}\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(package.parent), os.environ.get("PYTHONPATH", "")]))
    return asked


def _program(tmp_path, body):
    path = str(tmp_path / "python-with-basic-pitch")
    with open(path, "w") as f:
        f.write(f"#!{sys.executable}\nimport sys, time\n{body}")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def test_what_basic_pitch_hears_is_written_down_and_read_back_one_note_at_a_time(tmp_path, monkeypatch):
    events = [(1.0, 1.5, 43, 0.6, None), (0.5, 1.0, 40, 0.8, None), (0.7, 0.9, 52, 0.3, None)]
    asked = _basic_pitch(tmp_path, monkeypatch, events)
    cache = tmp_path / "cache"
    cache.mkdir()
    bass = np.zeros((2, 2 * SR), np.float32)
    notes, why_not = second_opinion.basic_pitch_notes(bass, STEM_ID, str(cache), python=sys.executable)
    assert why_not is None
    assert notes == [(0.5, 1.0, 40, 0.8), (1.0, 1.5, 43, 0.6)]  # in order; the quiet one inside another is gone
    said = json.load(open(asked))
    assert (said["rate"], said["channels"], said["frames"]) == (SR, 2, 2 * SR)  # it was given the bass track
    assert said["minimum_frequency"] < 41.2 and said["maximum_frequency"] > 392  # the low E to the G two octaves up
    assert said["model"].endswith("nmp.onnx")
    kept = cache / f"basicpitch-{STEM_ID[:16]}.csv"
    assert kept.read_text().splitlines()[:2] == ["start,end,pitch,amplitude", "0.500,1.000,40,0.800"]
    assert os.listdir(cache) == [kept.name]  # the track it was given is not left behind
    os.remove(asked)
    again, _ = second_opinion.basic_pitch_notes(bass, STEM_ID, str(cache), python="/nonexistent/python")
    assert again == notes and not os.path.exists(asked)  # kept for the track: Basic Pitch is not asked twice


def test_without_basic_pitch_the_reason_says_what_to_set(tmp_path, monkeypatch):
    monkeypatch.setattr(second_opinion, "find_python", lambda explicit=None: None)
    notes, why_not = second_opinion.basic_pitch_notes(np.zeros((2, SR), np.float32), STEM_ID, str(tmp_path))
    assert notes is None and "BASS2SHEET_BASIC_PITCH_PYTHON" in why_not and "Basic Pitch not found" in why_not
    assert os.listdir(tmp_path) == []


def test_a_basic_pitch_that_fails_or_hangs_is_a_reason_and_not_an_error(tmp_path):
    bass = np.zeros((2, SR), np.float32)
    broken = _program(tmp_path, "sys.stderr.write('Traceback\\nImportError: no onnxruntime here\\n'); sys.exit(1)\n")
    notes, why_not = second_opinion.basic_pitch_notes(bass, STEM_ID, str(tmp_path), python=broken)
    assert notes is None and why_not == "Basic Pitch failed: ImportError: no onnxruntime here"
    silent = _program(tmp_path, "sys.exit(0)\n")  # it says it is done and wrote nothing
    assert second_opinion.basic_pitch_notes(bass, STEM_ID, str(tmp_path), python=silent) == (None, "Basic Pitch failed")
    stuck = _program(tmp_path, "time.sleep(30)\n")
    notes, why_not = second_opinion.basic_pitch_notes(bass, STEM_ID, str(tmp_path), python=stuck, timeout=0.5)
    assert notes is None and why_not == "Basic Pitch timed out after 0.5 s"
    assert not [name for name in os.listdir(tmp_path) if name.endswith((".wav", ".csv"))]  # nothing half made


def test_the_interpreter_is_the_one_named_then_the_environments_then_a_venv_bp(tmp_path, monkeypatch):
    named, in_env = _program(tmp_path, ""), str(tmp_path / "from-the-environment")
    open(in_env, "w").close()
    here = tmp_path / "work" / ".venv-bp" / "bin"
    here.mkdir(parents=True)
    (here / "python").write_text("")
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.delenv(second_opinion.ENV_VAR, raising=False)
    assert second_opinion.find_python() == str(here / "python")  # a .venv-bp where the tool is run from
    monkeypatch.setenv(second_opinion.ENV_VAR, in_env)
    assert second_opinion.find_python() == in_env
    assert second_opinion.find_python(named) == named
    assert second_opinion.find_python("/nonexistent/python") == in_env  # one that is not there is passed over


def test_the_runner_writes_no_file_until_all_of_it_is_written(tmp_path, monkeypatch):
    """bp_runner.py as the other interpreter runs it: by its path, with the track and the file to write."""
    import subprocess

    _basic_pitch(tmp_path, monkeypatch, [(0.25, 0.5, 33, 0.5, None)])
    stem, out = str(tmp_path / "bass.wav"), str(tmp_path / "out.csv")
    sf.write(stem, np.zeros((SR, 2), np.float32), SR, subtype="FLOAT")
    runner = os.path.join(os.path.dirname(second_opinion.__file__), "bp_runner.py")
    done = subprocess.run([sys.executable, runner, stem, out], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "1", done.stderr[-2000:]  # it says how many notes
    assert open(out).read() == "start,end,pitch,amplitude\n0.250,0.500,33,0.500\n"
    assert sorted(os.listdir(tmp_path)) == ["asked.json", "bass.wav", "out.csv", "stand-in"]  # no out.csv.tmp
    with pytest.raises(subprocess.CalledProcessError):  # with no file names it fails and writes nothing
        subprocess.run([sys.executable, runner], check=True, capture_output=True, timeout=120)
