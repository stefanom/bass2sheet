"""The slow tier: real source separation, the second-opinion model, and the dataset regression.

Off by default. Enable with BASS2SHEET_FULL_E2E=1; the first run downloads the separation models (about 1 GB) into
the cache. The dataset regression also needs BASS2SHEET_EVAL_DATA=<BabySlakh directory>.
"""

import json
import os

import pytest

from bass2sheet import evaluate, pipeline
from bass2sheet.second_opinion import find_python

pytestmark = [pytest.mark.full,
              pytest.mark.skipif(os.environ.get("BASS2SHEET_FULL_E2E") != "1", reason="set BASS2SHEET_FULL_E2E=1 to run")]
HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "..", "eval", "baselines", "babyslakh.json")
QUICK_TRACKS = "Track00002,Track00008,Track00010,Track00016"


def _score(song, out):
    data = json.load(open(os.path.join(out, "notes.json")))
    est = [(n["start"], n["end"], n["pitch"]) for n in data["notes"]]
    grid = dict(beats=song["beats"], first_downbeat_beat=0, four_four=True)
    return evaluate.score(song["notes"], est, [(n["grid_start"], n["bar"]) for n in data["notes"]], grid, set())


@pytest.mark.parametrize("separator", ["roformer", "demucs"])
def test_full_pipeline_from_the_mix(separator, song, song_files, tmp_path):
    """The mix alone goes in: separation, the production pitch model and (when installed) Basic Pitch all run."""
    out = str(tmp_path / "out")
    code = pipeline.run(song_files[0], out, separator=separator, renders=False, pdf=False)
    assert code in (0, 2)
    m = _score(song, out)
    assert m["f"] >= 0.85 and m["bar_position_acc"] >= 0.9, m
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["summary"]["separator"] == separator
    if find_python():
        assert {g["name"]: g["status"] for g in rep["gates"]}["second_model_pitch"] != "skipped"


@pytest.mark.skipif(not os.environ.get("BASS2SHEET_EVAL_DATA"), reason="set BASS2SHEET_EVAL_DATA to a BabySlakh directory")
@pytest.mark.skipif(not os.path.exists(BASELINE), reason="no committed baseline")
def test_no_regression_on_babyslakh(tmp_path):
    code = evaluate.main([os.environ["BASS2SHEET_EVAL_DATA"], "--out", str(tmp_path / "run"), "--conditions", "mix",
                          "--tracks", QUICK_TRACKS, "--baseline", BASELINE])
    assert code == 0
