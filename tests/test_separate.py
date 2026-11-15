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


def _level(value, n):
    return np.full((2, n), value, np.float32)


def test_a_separation_is_kept_and_not_run_twice(tmp_path, monkeypatch):
    runs = []

    def roformer(mix_path, n, cache_dir):
        runs.append(mix_path)
        return dict(bass=_level(0.1, n - 100), drums=_level(0.2, n + 100), harmony=_level(0.3, n), vocals=_level(0, n))

    monkeypatch.setattr(separate, "_roformer", roformer)
    wav = _level(0.5, SR)
    stems = separate.separate("roformer", wav, "mix.wav", "0123456789abcdef-and-more", str(tmp_path), "cpu")
    assert all(v.shape == (2, SR) for v in stems.values())  # each as long as the song, to the sample
    assert np.allclose(stems["bass"][:, : SR - 100], 0.1) and np.allclose(stems["bass"][:, SR - 100 :], 0)
    assert (tmp_path / "stems-0123456789abcdef-roformer.npz").exists()
    again = separate.separate("roformer", wav, "mix.wav", "0123456789abcdef-and-more", str(tmp_path), "cpu")
    assert runs == ["mix.wav"] and all((again[k] == stems[k]).all() for k in stems)
    separate.separate("roformer", wav, "other.wav", "another-song-entirely", str(tmp_path), "cpu")
    assert runs == ["mix.wav", "other.wav"]  # kept by the song, not for every song


def test_demucs_is_tried_again_on_the_cpu_when_the_gpu_cannot_run_it(tmp_path, monkeypatch):
    tried = []

    def demucs(wav, device):
        tried.append(device)
        if device != "cpu":
            raise RuntimeError("an operation this GPU does not have")
        return dict(bass=wav * 0.5, drums=wav * 0.25, harmony=wav * 0.125, vocals=wav * 0)

    monkeypatch.setattr(separate, "_demucs", demucs)
    wav = _level(0.8, 1000)
    stems = separate.separate("demucs", wav, "mix.wav", "a" * 20, str(tmp_path), "mps")
    assert tried == ["mps", "cpu"] and np.allclose(stems["bass"], 0.4)

    def broken(wav, device):
        raise RuntimeError("no such model")

    monkeypatch.setattr(separate, "_demucs", broken)
    with pytest.raises(RuntimeError, match="no such model"):  # on the CPU there is nothing left to try
        separate.separate("demucs", wav, "mix.wav", "b" * 20, str(tmp_path), "cpu")
    assert not (tmp_path / f"stems-{'b' * 16}-demucs.npz").exists()  # and nothing is kept of a failure


def _stand_in_separator(monkeypatch, files):
    """Puts a stand-in where audio-separator's Separator is imported from: it writes `files` ({name: (level, sample
    rate)}) into its output folder, as the real one does, and says what it wrote. Returns what it was asked."""
    import sys
    import types

    asked = {}

    class Separator:
        def __init__(self, output_dir, model_file_dir, output_format, log_level):
            asked.update(output_dir=output_dir, model_file_dir=model_file_dir, output_format=output_format)

        def load_model(self, model_filename):
            asked["model"] = model_filename

        def separate(self, mix_path):
            asked["mix"] = mix_path
            out = []
            for k, (name, (value, rate)) in enumerate(files.items()):
                path = f"{asked['output_dir']}/{name}"
                sf.write(path, np.full((rate // 2, 2), value, np.float32), rate, subtype="FLOAT")
                out.append(path if k % 2 else name)  # it names some by their whole path and some without
            return out

    package, module = types.ModuleType("audio_separator"), types.ModuleType("audio_separator.separator")
    module.Separator = Separator
    package.separator = module
    monkeypatch.setitem(sys.modules, "audio_separator", package)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    return asked


def test_roformer_stems_are_read_by_the_names_the_separator_gives_them(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("PATH", os.environ["PATH"])  # _ensure_ffmpeg may add to it: put back after the test
    asked = _stand_in_separator(
        monkeypatch,
        {
            "mix_(Bass)_BS-Roformer-SW.wav": (0.1, SR),
            "mix_(Drums)_BS-Roformer-SW.wav": (0.2, SR),
            "mix_(Other)_BS-Roformer-SW.wav": (0.01, SR),
            "mix_(Guitar)_BS-Roformer-SW.wav": (0.02, SR // 2),  # at another sample rate: brought to ours
            "mix_(Piano)_BS-Roformer-SW.wav": (0.04, SR),
            "mix_(Vocals)_BS-Roformer-SW.wav": (0.3, SR),
            "mix_BS-Roformer-SW.wav": (0.9, SR),  # no stem's name in it: left alone
        },
    )
    n = SR  # the song lasts a second; the separator's files half of one
    stems = separate._roformer("/somewhere/mix.wav", n, str(tmp_path))
    assert set(stems) == {"bass", "drums", "harmony", "vocals"} and all(v.shape == (2, n) for v in stems.values())
    first = slice(100, SR // 2 - 100)
    assert np.allclose(stems["bass"][:, first], 0.1) and np.allclose(stems["drums"][:, first], 0.2)
    assert np.allclose(stems["harmony"][:, first], 0.07, atol=2e-3)  # other, guitar and piano together
    assert np.allclose(stems["vocals"][:, first], 0.3) and np.allclose(stems["bass"][:, SR // 2 + 100 :], 0)
    assert asked["model"] == separate.ROFORMER_MODEL and asked["mix"] == "/somewhere/mix.wav"
    assert asked["model_file_dir"] == str(tmp_path / "models") and asked["output_format"] == "WAV"
    assert not [p for p in os.listdir(tmp_path) if p.startswith("sep-")]  # its folder is gone again


def test_roformer_without_a_bass_stem_is_an_error(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("PATH", os.environ["PATH"])
    _stand_in_separator(monkeypatch, {"mix_(Drums)_x.wav": (0.2, SR), "mix_(Vocals)_x.wav": (0.3, SR)})
    with pytest.raises(Bass2SheetError, match="produced no bass stem"):
        separate._roformer("mix.wav", SR, str(tmp_path))
    assert not [p for p in os.listdir(tmp_path) if p.startswith("sep-")]


def test_demucs_hears_the_song_at_one_level_and_gives_the_stems_back_at_the_songs_own(monkeypatch):
    import sys
    import types

    import torch

    heard = {}

    class Model:
        sources = ["drums", "bass", "other", "vocals"]

        def eval(self):
            heard["eval"] = True

    def apply_model(model, x, device, **kw):
        heard.update(device=device, mean=float(x.mean()), std=float(x.mean(1).std()), shape=tuple(x.shape))
        return torch.stack([x[0] * (k + 1) for k in range(4)])[None]  # each source the song, k + 1 times as loud

    apply, pretrained = types.ModuleType("demucs.apply"), types.ModuleType("demucs.pretrained")
    apply.apply_model, pretrained.get_model = apply_model, lambda name: heard.update(name=name) or Model()
    for name, module in (("demucs.apply", apply), ("demucs.pretrained", pretrained)):
        monkeypatch.setitem(sys.modules, name, module)
    rng = np.random.default_rng(0)
    wav = (0.3 + 0.05 * rng.standard_normal((2, 4000))).astype(np.float32)  # a song with an offset and a level
    stems = separate._demucs(wav, "cpu")
    assert heard["name"] == "htdemucs_ft" and heard["eval"] and heard["shape"] == (1, 2, 4000)
    assert abs(heard["mean"]) < 1e-3 and heard["std"] == pytest.approx(1.0, abs=0.05)  # what the model was given
    mean = wav.mean()
    assert set(stems) == {"bass", "drums", "harmony", "vocals"} and stems["bass"].dtype == np.float32
    assert np.allclose(stems["drums"], wav, atol=1e-5)  # back at the level the song has
    assert np.allclose(stems["bass"], (wav - mean) * 2 + mean, atol=1e-4)  # each source under its own name
    assert np.allclose(stems["harmony"], (wav - mean) * 3 + mean, atol=1e-4)
    assert np.allclose(stems["vocals"], (wav - mean) * 4 + mean, atol=1e-4)


def test_the_separator_gets_an_ffmpeg_where_none_is_installed(tmp_path, monkeypatch):
    import os
    import shutil

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    separate._ensure_ffmpeg(str(tmp_path))
    assert os.listdir(tmp_path) == [] and os.environ["PATH"] == "/usr/bin:/bin"  # one is there: nothing to do
    monkeypatch.setattr(shutil, "which", lambda name: None)
    for _ in range(2):  # the second time the link is already there
        separate._ensure_ffmpeg(str(tmp_path))
    link = tmp_path / "bin" / "ffmpeg"
    assert os.path.islink(link) and os.access(link, os.X_OK)  # the one that comes with imageio-ffmpeg
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
