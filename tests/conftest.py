"""Shared fixtures. Tests use synthesized audio only (tests/songsynth.py): no recordings are needed.

Tiers:
  default      unit tests plus the end-to-end run on pre-separated synthetic stems (no model download)
  -m full      also the tests that need the separation models and, optionally, an evaluation dataset;
               enable with BASS2SHEET_FULL_E2E=1 (and BASS2SHEET_EVAL_DATA=<BabySlakh dir> for the regression)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TQDM_DISABLE", "1")


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
