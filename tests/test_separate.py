import numpy as np
import pytest
import soundfile as sf

from bass2sheet import separate
from bass2sheet.common import SR, Bass2SheetError


def _write(path, value, seconds=1.0):
    sf.write(str(path), np.full((int(seconds * SR), 2), value, np.float32), SR, subtype="FLOAT")


def test_load_stems_sums_harmony_parts_and_fits_length(tmp_path):
    _write(tmp_path / "bass.wav", 0.1)
    _write(tmp_path / "drums.wav", 0.2, seconds=0.5)  # shorter than the mix: padded
    _write(tmp_path / "other.wav", 0.01)
    _write(tmp_path / "Guitar.wav", 0.02)  # names are case-insensitive
    stems = separate.load_stems(str(tmp_path), SR)
    assert set(stems) == {"bass", "drums", "harmony", "vocals"}
    assert all(v.shape == (2, SR) for v in stems.values())
    assert np.allclose(stems["harmony"], 0.03) and np.allclose(stems["vocals"], 0)
    assert np.allclose(stems["drums"][:, : SR // 2], 0.2) and np.allclose(stems["drums"][:, SR // 2 + 10 :], 0)


def test_load_stems_requires_bass_and_drums(tmp_path):
    _write(tmp_path / "bass.wav", 0.1)
    with pytest.raises(Bass2SheetError, match="no drums stem"):
        separate.load_stems(str(tmp_path), SR)


def test_unknown_separator_is_refused(tmp_path):
    with pytest.raises(ValueError):
        separate.separate("spleeter", np.zeros((2, 10), np.float32), "x", "id", str(tmp_path), "cpu")
