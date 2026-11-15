"""Shared constants, errors, hashing and atomic file writes."""

import contextlib
import contextvars
import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from http.server import ThreadingHTTPServer

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


CACHE = "~/.cache/bass2sheet"
CACHE_ENV = "BASS2SHEET_CACHE"


def cache_home(given=None):
    """Where the models, the separated tracks and the pitch tracks are kept: `given` (a command's --cache-dir),
    else $BASS2SHEET_CACHE, else ~/.cache/bass2sheet."""
    return os.path.abspath(os.path.expanduser(given or os.environ.get(CACHE_ENV) or CACHE))


_listener = contextvars.ContextVar("bass2sheet_listener", default=None)


@contextlib.contextmanager
def listening(tell):
    """While this is open, `tell` is called with every step the work goes through (see `step`)."""
    token = _listener.set(tell)
    try:
        yield
    finally:
        _listener.reset(token)


@contextlib.contextmanager
def step(name):
    """One step of hearing a song, for whoever is listening: told when it starts, as dict(step=name,
    state="running"), and when it is done, as dict(step=name, state="done", seconds=...) with what it found, which
    the code inside puts in the dict this yields. A step that fails is not told as done: the error says the rest.
    With no one listening this does nothing."""
    tell, found, began = _listener.get(), {}, time.monotonic()
    if tell:
        tell(dict(step=name, state="running"))
    yield found
    if tell:
        tell(dict(found, step=name, state="done", seconds=round(time.monotonic() - began, 2)))


class PageServer(ThreadingHTTPServer):
    """The server behind the pages (the copilot's and the judging page's)."""

    # A page asks for many things at once (the first page: a picture of every song), and a browser opens several
    # connections to do it. With the standard five waiting to be answered, one more is turned away now and then.
    request_queue_size = 64

    def handle_error(self, request, client_address):
        """A browser that goes away before its answer is written (a tab closed, a sound cut off when another
        starts) is nothing to report. Anything else is, as always."""
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


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


LOWEST_NOTE = 28  # E1, the lowest string of a four-string bass in standard tuning


def note_number(name):
    """MIDI number of a note name such as E1, Bb0 or F#2 (C4 = 60)."""
    import re

    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", name.strip())
    if not m:
        raise Bass2SheetError(f"{name!r} is not a note name (examples: E1, Bb0, F#2)")
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()]
    return base + {"": 0, "#": 1, "b": -1}[m.group(2)] + 12 * (int(m.group(3)) + 1)


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
