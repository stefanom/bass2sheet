import numpy as np
import pytest
import soundfile as sf

from bass2sheet import audio
from bass2sheet.common import SR, Bass2SheetError


def test_decode_resamples_to_stereo_44k(tmp_path):
    p = str(tmp_path / "mono16k.wav")
    t = np.arange(16000 * 2) / 16000
    sf.write(p, 0.5 * np.sin(2 * np.pi * 220 * t), 16000, subtype="PCM_16")
    wav = audio.decode(p)
    assert wav.shape[0] == 2 and wav.dtype == np.float32
    assert abs(wav.shape[1] - 2 * SR) < SR * 0.02
    assert 0.3 < np.abs(wav).max() < 0.6


def test_decode_rejects_non_audio(tmp_path):
    p = tmp_path / "fake.mp3"
    p.write_text("this is not audio\n" * 50)
    with pytest.raises(Bass2SheetError, match="could not open|decod"):
        audio.decode(str(p))


def test_validate_refuses_short_long_and_silent_audio():
    tone = lambda seconds: np.tile(0.2 * np.sin(np.arange(int(seconds * SR)) * 0.05).astype(np.float32), (2, 1))
    with pytest.raises(Bass2SheetError, match="at least"):
        audio.validate(tone(5))
    with pytest.raises(Bass2SheetError, match="silent"):
        audio.validate(np.zeros((2, 30 * SR), np.float32))
    with pytest.raises(Bass2SheetError, match="at most"):
        audio.validate(np.zeros((2, int(21 * 60 * SR)), np.float32))


def test_validate_warns_about_mono_and_clipping():
    y = 0.2 * np.sin(np.arange(30 * SR) * 0.05).astype(np.float32)
    seconds, warnings = audio.validate(np.stack([y, y]))
    assert seconds == 30 and any("mono" in w for w in warnings)
    stereo = np.stack([y, np.roll(y, 100)])
    assert audio.validate(stereo)[1] == []
    clipped = np.clip(stereo * 20, -1, 1)
    assert any("clipped" in w for w in audio.validate(clipped)[1])
