"""Two notes played together: found in the spectrum, written as one event (synthesized plucks, no models)."""

import librosa
import numpy as np
import pytest

from bass2sheet import notes as N
from bass2sheet import together as T
from bass2sheet.common import HOP, SR
from bass2sheet.render import pluck


def _line(spec, seconds):
    """Mono audio from (start s, duration s, MIDI pitch) plucks; notes that share a start sound together."""
    y = np.zeros(int(seconds * SR))
    for start, dur, pitch in spec:
        w = 0.4 * pluck(librosa.midi_to_hz(pitch), dur, 100)
        i = int(start * SR)
        y[i : i + len(w)] += w[: len(y) - i]
    return y.astype(np.float32)


def _peaks(*notes):
    """Peaks of notes given as (pitch, [dB of each partial, loudest first...])."""
    return [(float(T.hz(p)) * (k + 1), db) for p, levels in notes for k, db in enumerate(levels)]


def test_the_sieve_finds_one_note_or_two():
    one = _peaks((33, [-6, 0, -8, -12]))
    assert T.sieve(one) == [33]
    assert T.sieve(_peaks((33, [-6, 0, -8, -12]), (40, [-4, -7, -10]))) == [33, 40]  # a fifth above
    assert T.sieve(_peaks((36, [0, -5, -9]), (52, [-3, -8, -12]))) == [36, 52]  # a tenth
    assert T.sieve(_peaks((33, [-6, 0, -8, -12]), (45, [-2, -6, -10]))) == [33]  # an octave: one note
    assert T.sieve(one + [(float(T.hz(40)), -3)]) == [33]  # one stray peak is not a note
    assert T.sieve(_peaks((33, [-6, 0, -8, -12]), (40, [-20, -22, -25]))) == [33]  # too faint to be played
    assert T.sieve([]) == [] and T.sieve([(700.0, 0.0)]) == []


def test_the_sieve_takes_the_tracked_note_first_when_the_peaks_bear_it_out():
    peaks = _peaks((33, [-6, 0, -8, -12]), (40, [-4, -7, -10]))
    assert T.sieve(peaks, given=40) == [33, 40] and T.sieve(peaks, given=33) == [33, 40]
    assert T.sieve(peaks, given=50) == [33, 40]  # nothing sounds at D3: the tracker's note is set aside


@pytest.mark.parametrize(
    "spec,expected",
    [
        ([(0.2, 0.5, 33)], [33]),
        ([(0.2, 0.5, 33), (0.2, 0.5, 40)], [33, 40]),  # a fifth
        ([(0.2, 0.5, 36), (0.2, 0.5, 52)], [36, 52]),  # a tenth
        ([(0.2, 0.5, 38), (0.2, 0.5, 43)], [38, 43]),  # a fourth
        ([(0.2, 0.5, 33), (0.2, 0.5, 45)], [33]),  # an octave is written as its lower note
        ([(0.2, 0.5, 36), (0.2, 0.5, 38)], [36]),  # a step apart is not a pair
    ],
)
def test_notes_in_a_stretch(spec, expected):
    assert T.notes_in(_line(spec, 1.0), 0.2, 0.7, given=expected[0]) == expected


def test_a_note_still_ringing_or_gone_early_is_not_played_together():
    ringing = _line([(0.1, 1.2, 33), (0.7, 0.5, 40)], 1.5)  # the A rings on under the E that starts later
    assert T.notes_in(ringing, 0.7, 1.2, given=40) == [40]
    assert T.notes_in(ringing, 0.1, 0.6, given=33) == [33]
    cut = _line([(0.2, 0.6, 33), (0.2, 0.08, 40)], 1.0)  # the E stops after 80 ms: a slip, not a double stop
    assert T.notes_in(cut, 0.2, 0.8, given=33) == [33]
    assert T.rose(ringing, 0.7, 40) and not T.rose(ringing, 0.7, 33, beside=40)


def _features(y, tracked, lag_frames=2):
    """Features as the pitch stage would give them, with the pitch the tracker followed: (start, dur, pitch)."""
    n = len(y) // HOP + 1
    f0, per = np.full(n, 50.0), np.full(n, 0.02)
    for start, dur, pitch in tracked:
        a, b = int(round(start * 100)) + lag_frames, int(round((start + dur) * 100))
        f0[a:b], per[a:b] = librosa.midi_to_hz(pitch), 0.9
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0] + 1e-9)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=1000)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True)
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72, bins_per_octave=12))
    return dict(f0_c=f0, per_c=per, f0_p=f0.copy(), rms_db=rms_db, onsets=onsets, cqt=cqt)


PLAYED = [(0.5, 0.45, 33), (1.0, 0.45, 33), (1.0, 0.45, 40), (1.5, 0.45, 36), (2.0, 0.45, 38), (2.0, 0.45, 45)]


def _written(tracked, **params):
    y = _line(PLAYED, 3.0)
    out, _, _ = N.segment_notes(_features(y, tracked), y, N.SegParams(**params))
    return [(round(n["start"], 1), n["pitch"], n.get("also")) for n in out]


def test_a_pair_is_one_event_whichever_note_the_tracker_followed():
    both = [(0.5, 33, None), (1.0, 33, [40]), (1.5, 36, None), (2.0, 38, [45])]
    upper = [(0.5, 0.45, 33), (1.0, 0.45, 40), (1.5, 0.45, 36), (2.0, 0.45, 45)]
    lower = [(0.5, 0.45, 33), (1.0, 0.45, 33), (1.5, 0.45, 36), (2.0, 0.45, 38)]
    assert _written(upper) == both and _written(lower) == both
    assert _written(upper, together=False) == [(0.5, 33, None), (1.0, 40, None), (1.5, 36, None), (2.0, 45, None)]


def test_a_tracker_that_flips_between_the_two_notes_or_loses_them_still_gives_one_event():
    both = [(0.5, 33, None), (1.0, 33, [40]), (1.5, 36, None), (2.0, 38, [45])]
    flips = [(0.5, 0.45, 33), (1.0, 0.2, 40), (1.2, 0.25, 33), (1.5, 0.45, 36), (2.0, 0.45, 38)]
    assert _written(flips) == both
    lost = [(0.5, 0.45, 33), (1.5, 0.45, 36), (2.0, 0.45, 45)]  # no pitch at all while the first pair sounds
    assert _written(lost) == both
    late = [(0.5, 0.45, 33), (1.25, 0.2, 40), (1.5, 0.45, 36), (2.0, 0.45, 38)]  # ...or only for its second half
    assert _written(late) == both
