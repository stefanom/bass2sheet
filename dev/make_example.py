"""Regenerate docs/example-score.png: the score bass2sheet writes for the synthesized test song.

python dev/make_example.py            (needs MuseScore 4; uses the production pitch model)
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from songsynth import make_song, write_song

from bass2sheet import pipeline
from bass2sheet.score import find_mscore

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with tempfile.TemporaryDirectory() as d:
    mix, stems = write_song(make_song(bars=16), d)
    out = os.path.join(d, "out")
    code = pipeline.run(
        mix,
        out,
        title="Synthetic Song",
        artist="bass2sheet",
        stems_dir=stems,
        second_mode="off",
        renders=False,
        pdf=False,
        cache_dir=os.path.join(d, "cache"),
    )
    print("exit code", code)
    png = os.path.join(d, "score.png")
    subprocess.run(
        [find_mscore(), "-r", "110", "-T", "30", "-o", png, os.path.join(out, "bass.musicxml")],
        check=True,
        capture_output=True,
    )
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    os.replace(os.path.join(d, "score-1.png"), os.path.join(root, "docs", "example-score.png"))
    print("wrote docs/example-score.png")
