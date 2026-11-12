"""Decoding CREPE's bin probabilities into a pitch path (no model needed: the probabilities are made up)."""

import numpy as np

from bass2sheet import pitch


def _probs(frames, peaks, level=0.02):
    """(frames, bins) probabilities: `peaks` is a list of (first frame, last frame, MIDI pitch, probability)."""
    p = np.full((frames, pitch.PITCH_BINS), level)
    bins = np.arange(pitch.PITCH_BINS)
    for a, b, midi, prob in peaks:
        centre = pitch.bin_of_hz(440 * 2 ** ((midi - 69) / 12))
        p[a:b] = np.maximum(p[a:b], prob * np.exp(-0.5 * ((bins - centre) / 1.5) ** 2))
    return p


def _midi(f0):
    return 69 + 12 * np.log2(f0 / 440)


def test_bins_and_frequencies_convert_both_ways():
    hz = np.array([41.2, 82.4, 110.0, 440.0])
    assert np.allclose(pitch.hz_of_bin(pitch.bin_of_hz(hz)), hz)
    assert abs(pitch.bin_of_hz(82.4) - pitch.bin_of_hz(41.2) - 60) < 1e-6  # an octave is 60 bins of 20 cents


def test_the_path_follows_an_octave_jump_at_once():
    p = _probs(60, [(0, 30, 28, 0.9), (30, 60, 40, 0.9), (30, 60, 28, 0.3)])  # the low note still shows faintly
    f0, per = pitch.decode_pitch(p)
    m = _midi(f0)
    assert np.all(np.abs(m[:30] - 28) < 0.2) and np.all(np.abs(m[31:] - 40) < 0.2)
    assert per[:30].min() > 0.8 and per[31:].min() > 0.8


def test_a_short_high_note_between_low_ones_is_kept():
    p = _probs(90, [(0, 30, 31, 0.9), (30, 42, 43, 0.9), (42, 90, 31, 0.9)])  # a 120 ms pop an octave up
    m = _midi(pitch.decode_pitch(p)[0])
    assert np.all(np.abs(m[31:41] - 43) < 0.2) and np.all(np.abs(m[44:] - 31) < 0.2)


def test_a_stray_frame_does_not_move_the_path():
    p = _probs(60, [(0, 60, 33, 0.9)])
    p[25:27] = _probs(2, [(0, 2, 52, 0.95)])  # 20 ms of something else
    f0, per = pitch.decode_pitch(p)
    assert np.all(np.abs(_midi(f0) - 33) < 0.2)
    assert per[25] < 0.1 < per[24]  # ...and the path's confidence drops there instead


def test_a_glide_is_followed_smoothly_and_out_of_range_bins_are_ignored():
    frames = 40
    glide = [(t, t + 1, 40 + 2 * t / frames, 0.9) for t in range(frames)]
    m = _midi(pitch.decode_pitch(_probs(frames, glide))[0])
    assert np.all(np.diff(m) > -0.05) and abs(m[0] - 40) < 0.2 and abs(m[-1] - 42) < 0.2
    p = _probs(30, [(0, 30, 33, 0.5), (0, 30, 12, 0.99), (0, 30, 90, 0.99)])  # 16 Hz and 1480 Hz: not a bass
    assert np.all(np.abs(_midi(pitch.decode_pitch(p)[0]) - 33) < 0.2)
