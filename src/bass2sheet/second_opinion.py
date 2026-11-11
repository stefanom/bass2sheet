"""Second opinion: Basic Pitch (Spotify) on the same bass stem, run as a subprocess in its own environment."""

import csv
import os
import subprocess
from pathlib import Path

import soundfile as sf

from .common import SR, log

ENV_VAR = "BASS2SHEET_BASIC_PITCH_PYTHON"


def find_python(explicit=None):
    """The Python interpreter that has Basic Pitch: `explicit`, then $BASS2SHEET_BASIC_PITCH_PYTHON, then a
    `.venv-bp` in the working directory, in the source checkout or beside it, then the cache directory."""
    here = Path(__file__).resolve()
    candidates = [explicit, os.environ.get(ENV_VAR)]
    for base in (Path.cwd(), here.parents[2], here.parents[3], Path.home() / ".cache" / "bass2sheet"):
        candidates.append(str(base / ".venv-bp" / "bin" / "python"))
    return next((c for c in candidates if c and os.path.exists(c)), None)


def basic_pitch_notes(bass, stem_id, cache_dir, python=None, timeout=1800):
    """Monophonic Basic Pitch notes [(start, end, pitch, amplitude)], or (None, reason) when unavailable."""
    out_csv = os.path.join(cache_dir, f"basicpitch-{stem_id[:16]}.csv")
    if not os.path.exists(out_csv):
        py = find_python(python)
        if not py:
            return None, (
                f"Basic Pitch not found: set {ENV_VAR} to a Python with basic-pitch and onnxruntime (see README)"
            )
        wav = os.path.join(cache_dir, f"bass-{stem_id[:16]}.wav")
        sf.write(wav, bass.T, SR, subtype="FLOAT")
        log.info("second opinion: Basic Pitch")
        try:
            r = subprocess.run(
                [py, str(Path(__file__).with_name("bp_runner.py")), wav, out_csv],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"Basic Pitch timed out after {timeout} s"
        finally:
            os.remove(wav)
        if r.returncode != 0 or not os.path.exists(out_csv):
            last = (r.stderr or "").strip().splitlines()
            return None, "Basic Pitch failed" + (f": {last[-1]}" if last else "")
    with open(out_csv) as f:
        notes = [
            (float(x["start"]), float(x["end"]), int(x["pitch"]), float(x["amplitude"])) for x in csv.DictReader(f)
        ]
    return monophonic(notes), None


def monophonic(notes):
    """One note at a time: a later-starting note cuts the previous one short if it is louder (or, equally loud,
    lower); otherwise it starts when the previous one ends."""
    out = []
    for s, e, p, a in sorted(notes):
        if out and s < out[-1][1]:
            ps, pe, pp, pa = out[-1]
            if a > pa or (a == pa and p < pp):
                out[-1] = (ps, s, pp, pa)
            elif e > pe:
                s = pe
            else:
                continue
        if e - s >= 0.03:
            out.append((s, e, p, a))
    return [n for n in out if n[1] - n[0] >= 0.03]
