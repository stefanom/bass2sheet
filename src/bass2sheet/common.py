"""Shared constants, errors, hashing and atomic file writes."""

import hashlib
import logging
import os
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


class Bass2SheetError(Exception):
    """A failure the user can act on; the message says what went wrong and what to do."""


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


def atomic_savez(path, **arrays):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".npz")
    os.close(fd)
    np.savez(tmp, **arrays)
    _publish(tmp, path)


def note_name(midi, names=NOTE_NAMES):
    return f"{names[midi % 12]}{midi // 12 - 1}"


def spelling(key_root, key_major):
    """Sharps for sharp keys (G, D, A, E, B, F# major and their relative minors), flats otherwise."""
    relative_major = key_root if key_major else (key_root + 3) % 12
    return SHARPS if relative_major in (7, 2, 9, 4, 11, 6) else FLATS


def clock(s):
    return f"{int(max(s, 0) // 60)}:{max(s, 0) % 60:04.1f}"
