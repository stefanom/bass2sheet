"""Shared fixtures. Tests use synthesized audio only (tests/songsynth.py): no recordings are needed.

Tiers:
  default      unit tests plus the end-to-end run on pre-separated synthetic stems (no model download)
  -m full      also the tests that need the separation models and, optionally, an evaluation dataset;
               enable with BASS2SHEET_FULL_E2E=1 (and BASS2SHEET_EVAL_DATA=<BabySlakh dir> for the regression)
  -m browser   the pages in a real browser (tests/test_browser_*.py); skipped where Playwright or a browser to
               drive is missing

BASS2SHEET_TEST_BARE=1 runs the tests as a machine with nothing but `uv sync` on it would: the MuseScore and the
Rubber Band that are installed here are not found. That is what CI has, and what its coverage floor is set from.
"""

import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TQDM_DISABLE", "1")

if os.environ.get("BASS2SHEET_TEST_BARE") == "1":  # before any test module asks whether MuseScore is installed
    import imageio_ffmpeg

    from bass2sheet import copilot, score

    score.MSCORE_CANDIDATES = []
    copilot.stretcher = lambda: ("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())


def mscore_stand_in(directory):
    """Writes a program that takes MuseScore's place (tests/mscore_standin.py) into `directory` and returns its
    path. A test puts it where the tool looks for MuseScore: monkeypatch.setattr(score, "MSCORE_CANDIDATES", [path])."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mscore_standin.py")
    program = os.path.join(str(directory), "mscore")
    with open(program, "w") as f:
        f.write(f"#!{sys.executable}\nimport runpy, sys\nsys.argv[0] = {script!r}\n")
        f.write(f"runpy.run_path({script!r}, run_name='__main__')\n")
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    return program


@pytest.fixture(scope="session")
def song():
    from songsynth import make_song

    return make_song()


@pytest.fixture(scope="session")
def song_files(song, tmp_path_factory):
    """(mix path, stems dir) of the synthetic song on disk."""
    from songsynth import write_song

    return write_song(song, str(tmp_path_factory.mktemp("song")))


@pytest.fixture(scope="session")
def taste_song_files(tmp_path_factory):
    """(mix path, stems dir) of a synthetic song with passages that can be written either way (see make_song)."""
    from songsynth import make_song, write_song

    return write_song(make_song(taste=True), str(tmp_path_factory.mktemp("taste_song")))


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    """A private cache so tests never read or write ~/.cache/bass2sheet."""
    return str(tmp_path_factory.mktemp("cache"))


def note(start, end, pitch, **kw):
    """A note dict as the pipeline carries it after quantization."""
    d = dict(start=start, end=end, pitch=pitch, agree=1.0, flag=None)
    d.update(kw)
    return d
