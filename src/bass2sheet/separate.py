"""Source separation: BS-RoFormer SW (primary) or Demucs htdemucs_ft (fallback), cached by audio hash."""

import os
import re
import shutil
import tempfile
import logging

import numpy as np
import soundfile as sf

from .common import SR, Bass2SheetError, atomic_savez, log

ROFORMER_MODEL = "BS-Roformer-SW.ckpt"
METHODS = ("roformer", "demucs")


def _fit(a, n):
    a = a[:, :n]
    return np.pad(a, ((0, 0), (0, n - a.shape[1]))) if a.shape[1] < n else a


def _ensure_ffmpeg(cache_dir):
    """audio-separator shells out to ffmpeg; use imageio-ffmpeg's static binary when none is installed."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    bindir = os.path.join(cache_dir, "bin")
    os.makedirs(bindir, exist_ok=True)
    link = os.path.join(bindir, "ffmpeg")
    if not os.path.exists(link):
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), link)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


def _roformer(mix_path, n, cache_dir):
    from audio_separator.separator import Separator

    _ensure_ffmpeg(cache_dir)
    out_dir = tempfile.mkdtemp(prefix="sep-", dir=cache_dir)
    try:
        sep = Separator(output_dir=out_dir, model_file_dir=os.path.join(cache_dir, "models"), output_format="WAV",
                        log_level=logging.WARNING)
        sep.load_model(model_filename=ROFORMER_MODEL)
        files = sep.separate(mix_path)
        stems = {}
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(out_dir, os.path.basename(f))
            m = re.search(r"_\(([^)]+)\)_", os.path.basename(p))
            if not m:
                continue
            y, sr = sf.read(p, always_2d=True, dtype="float32")
            if sr != SR:
                import librosa

                y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
            stems[m.group(1).lower()] = _fit(y.T, n)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    missing = {"bass", "drums"} - set(stems)
    if missing:
        raise Bass2SheetError(f"RoFormer separation produced no {', '.join(sorted(missing))} stem")
    zeros = np.zeros((2, n), np.float32)
    harmony = sum(stems.get(k, zeros) for k in ("other", "guitar", "piano"))
    return {"bass": stems["bass"], "drums": stems["drums"], "harmony": harmony, "vocals": stems.get("vocals", zeros)}


def _demucs(wav, device):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_ft")
    model.eval()
    x = torch.from_numpy(wav)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.no_grad():
        out = apply_model(model, ((x - mean) / std)[None], device=device, shifts=1, split=True, overlap=0.25,
                          progress=False)[0]
    out = out * std + mean
    s = {name: out[k].cpu().numpy().astype(np.float32) for k, name in enumerate(model.sources)}
    return {"bass": s["bass"], "drums": s["drums"], "harmony": s["other"], "vocals": s["vocals"]}


STEM_FILES = {"bass": ("bass",), "drums": ("drums",), "harmony": ("harmony", "other", "guitar", "piano"),
              "vocals": ("vocals",)}


def load_stems(stems_dir, n):
    """Stems the user already has: a directory with bass.* and drums.* and, optionally, harmony.* (or other.*,
    guitar.*, piano.*, which are summed) and vocals.*. Any audio format; each is fitted to `n` samples."""
    from .audio import decode

    def find(name):
        hits = sorted(p for p in os.listdir(stems_dir) if os.path.splitext(p)[0].lower() == name and not p.startswith("."))
        return os.path.join(stems_dir, hits[0]) if hits else None

    zeros = np.zeros((2, n), np.float32)
    out = {}
    for stem, names in STEM_FILES.items():
        parts = [_fit(decode(p), n) for p in map(find, names) if p]
        if not parts and stem in ("bass", "drums"):
            raise Bass2SheetError(f"no {stem} stem in {stems_dir!r} (expected {stem}.wav or another audio format)")
        out[stem] = sum(parts) if parts else zeros
    return out


def separate(method, wav, mix_path, audio_id, cache_dir, device):
    """Stems {bass, drums, harmony, vocals}, each (2, n) float32 aligned with `wav`."""
    if method not in METHODS:
        raise ValueError(method)
    path = os.path.join(cache_dir, f"stems-{audio_id[:16]}-{method}.npz")
    if os.path.exists(path):
        log.info("separation (%s): cached", method)
        return dict(np.load(path))
    n = wav.shape[1]
    log.info("separation (%s): running", method)
    if method == "roformer":
        stems = _roformer(mix_path, n, cache_dir)
    else:
        try:
            stems = _demucs(wav, device)
        except Exception as e:  # some Demucs ops are unsupported on MPS
            if device == "cpu":
                raise
            log.warning("Demucs on %s failed (%s); retrying on CPU", device, e)
            stems = _demucs(wav, "cpu")
    stems = {k: _fit(v, n) for k, v in stems.items()}
    atomic_savez(path, **stems)
    return stems
