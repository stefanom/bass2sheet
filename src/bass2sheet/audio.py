"""Decoding and input validation."""

import numpy as np

from .common import SR, Bass2SheetError, log

MIN_SECONDS = 20.0
MAX_SECONDS = 20 * 60.0


def decode(path):
    """Any audio or video file -> float32 stereo array (2, n) at 44.1 kHz."""
    import av

    try:
        container = av.open(path)
    except Exception as e:  # PyAV raises several error types for unreadable input
        raise Bass2SheetError(f"could not open {path!r} as audio or video ({e})") from e
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise Bass2SheetError(f"{path!r} has no audio track")
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            chunks += [f.to_ndarray() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray() for f in resampler.resample(None)]
    except Bass2SheetError:
        raise
    except Exception as e:
        raise Bass2SheetError(f"decoding {path!r} failed ({e})") from e
    finally:
        container.close()
    if not chunks:
        raise Bass2SheetError(f"{path!r} decoded to no audio")
    return np.concatenate(chunks, axis=1).astype(np.float32)


def validate(wav):
    """Refuse input the pipeline cannot transcribe meaningfully; returns warnings for borderline input."""
    seconds = wav.shape[1] / SR
    if seconds < MIN_SECONDS:
        raise Bass2SheetError(f"only {seconds:.1f} s of audio; bass2sheet needs at least {MIN_SECONDS:.0f} s")
    if seconds > MAX_SECONDS:
        raise Bass2SheetError(f"{seconds / 60:.1f} min of audio; bass2sheet handles at most {MAX_SECONDS / 60:.0f} min")
    peak = float(np.abs(wav).max())
    if peak < 1e-4:
        raise Bass2SheetError("the audio is silent")
    warnings = []
    clipped = float(np.mean(np.abs(wav) > 0.999))
    if clipped > 1e-3:
        warnings.append(f"{clipped:.1%} of samples are clipped; separation and pitch tracking may suffer")
    if np.allclose(wav[0], wav[1], atol=1e-6):
        warnings.append("the audio is mono; stereo cues that help separation are missing")
    for w in warnings:
        log.warning(w)
    return seconds, warnings
