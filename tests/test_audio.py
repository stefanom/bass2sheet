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


def test_a_file_with_no_sound_in_it_is_refused_by_what_is_wrong_with_it(tmp_path):
    import av

    film = str(tmp_path / "silent film.mp4")
    with av.open(film, "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(Bass2SheetError, match="silent film.mp4' has no audio track"):
        audio.decode(film)
    empty = str(tmp_path / "empty.wav")  # a file that says it is audio and holds none
    sf.write(empty, np.zeros((0, 2), np.float32), SR)
    with pytest.raises(Bass2SheetError, match="empty.wav' decoded to no audio"):
        audio.decode(empty)
    cut = str(tmp_path / "cut off.flac")  # half of a file: it opens, and breaks on the way
    sf.write(cut, 0.3 * np.sin(np.arange(3 * SR) * 0.03), SR)
    whole = open(cut, "rb").read()
    open(cut, "wb").write(whole[: len(whole) // 2])
    with pytest.raises(Bass2SheetError, match=r"decoding '.*cut off.flac' failed \(.*Invalid data"):
        audio.decode(cut)
