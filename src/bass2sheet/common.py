"""Shared constants, errors, hashing and atomic file writes."""

import hashlib
import logging
import os
import re
import tempfile

import numpy as np

SR = 44100  # working sample rate
HOP_S = 0.01  # analysis frame: 10 ms
HOP = int(SR * HOP_S)
FMIN, FMAX = 30.0, 500.0  # bass range with headroom (B0 to B4)
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

log = logging.getLogger("bass2sheet")


# The tool's standing choices. Each has one alternative; `bass2sheet-judge` puts the two in front of the person
# who plays from the sheet, and their picks decide what stands here (docs/judging.md).
DEFAULTS = dict(
    separator="roformer",  # or "demucs": which model isolates the bass first
    replucks="every",  # or "clear": a repeated pitch is a new note at every pluck heard, or only at a bright one
    together="write",  # or "single": two notes where two are played together, or one note at a time
    rhythm="literal",  # or "simple": the rhythm as played, or stray sixteenths on eighths and short rests closed
    beat_tracker="beat_this",  # or "librosa": where the beats and bar lines come from
    tab="low",  # or "position": the tab low on the neck with open strings, or with the hand in one position
)
ALTERNATIVES = dict(
    separator=("roformer", "demucs"),
    replucks=("every", "clear"),
    together=("write", "single"),
    rhythm=("literal", "simple"),
    beat_tracker=("beat_this", "librosa"),
    tab=("low", "position"),
)


class Bass2SheetError(Exception):
    """A failure the user can act on; the message says what went wrong and what to do."""


def release_gpu_memory():
    """Give cached GPU memory back to the system. PyTorch keeps what a model used (several GB for the separator on
    Apple GPUs) until the process ends unless told to let go; long batch runs and parallel runs need it back."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # never let housekeeping break a run
        log.debug("could not release GPU memory: %s", e)


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha1_array(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _publish(tmp, path):
    """mkstemp files are 0600; give the result the permissions a plain open() would, then move it into place."""
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    _publish(tmp, path)


def atomic_savez(path, compressed=False, **arrays):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".npz")
    os.close(fd)
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    _publish(tmp, path)


def note_name(midi, names=NOTE_NAMES):
    return f"{names[midi % 12]}{midi // 12 - 1}"


def spelling(key_root, key_major):
    """Sharps for sharp keys (G, D, A, E, B, F# major and their relative minors), flats otherwise."""
    relative_major = key_root if key_major else (key_root + 3) % 12
    return SHARPS if relative_major in (7, 2, 9, 4, 11, 6) else FLATS


def clock(s):
    return f"{int(max(s, 0) // 60)}:{max(s, 0) % 60:04.1f}"


# "(Official HD Video)", "[Official Music Video]", "(Lyric Video)", "(Remastered 2011)": decoration, not the title
_DECORATION = re.compile(
    r"\s*[\(\[][^\)\]]*\b(official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|remaster(ed)?)\b[^\)\]]*[\)\]]", re.I
)


def clean_name(path):
    """The file name without its extension, a trailing '[videoid]' and video-title decoration."""
    base = re.sub(r"\s*\[[^\]]+\]$", "", os.path.splitext(os.path.basename(path))[0])
    return re.sub(r"\s{2,}", " ", _DECORATION.sub("", base)).strip()
