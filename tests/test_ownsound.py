"""A transcription checked by the bass track's own sound (synthesized plucks, no models)."""

import librosa
import numpy as np

from bass2sheet import ownsound as O
from bass2sheet.common import SR
from bass2sheet.render import pluck


def _track(spec, seconds):
    """(mono audio, notes) from (start s, duration s, MIDI pitch) plucks."""
    y = np.zeros(int(seconds * SR))
    for start, dur, pitch in spec:
        w = 0.4 * pluck(librosa.midi_to_hz(pitch), dur, 100)
        i = int(start * SR)
        y[i : i + len(w)] += w[: len(y) - i]
    return y.astype(np.float32), [dict(start=s, end=s + d, pitch=p) for s, d, p in spec]


# a line over four pitches, each played several times
LINE = [(0.5 * k, 0.45, p) for k, p in enumerate([33, 40, 36, 38, 33, 38, 40, 36, 33, 36, 40, 38, 33, 40, 38, 36])]


def test_the_same_sound_overlaps_and_a_shift_in_pitch_is_a_shift_along_the_axis():
    y, _ = _track([(0.2, 0.5, 40), (1.0, 0.5, 40), (1.8, 0.5, 41), (2.6, 0.5, 47)], 3.5)
    a, same, up1, fifth = (O.spectrum(y, t, 0.3) for t in (0.2, 1.0, 1.8, 2.6))
    assert abs(a.sum() - 1) < 1e-9 and O.overlap(a, same) > 0.95
    assert O.overlap(a, up1) < 0.4 and O.overlap(a, fifth) < 0.4
    assert O.overlap(a, O.shifted(up1, -1)) > 0.85 and O.overlap(a, O.shifted(fifth, -7)) > 0.7
    assert O.spectrum(y, 3.49, 0.3) is None  # nothing left to take a spectrum of


def test_right_notes_are_explained_by_the_others():
    y, notes = _track(LINE, 8.5)
    heard = O.check(y, notes)
    assert all(h and h["explained"] > 0.8 and h["better"] is None for h in heard)


def test_a_wrong_note_among_right_ones_is_found_and_named():
    y, notes = _track(LINE, 8.5)
    notes[5] = dict(notes[5], pitch=40)  # it sounds a D2 and is written E2
    notes[8] = dict(notes[8], pitch=45)  # it sounds an A1 and is written an octave up
    heard = O.check(y, notes)
    assert heard[5]["better"] == 38 and heard[5]["explained"] < 0.5 and heard[5]["by"] > 0.3
    assert heard[8] == dict(explained=0.0, better=33, by=heard[8]["by"]) and heard[8]["by"] > 0.8  # no other A2
    others = [h for k, h in enumerate(heard) if k not in (5, 8)]
    assert all(h["better"] is None and h["explained"] > 0.8 for h in others)


def test_a_mistake_made_every_time_is_not_seen():
    y, notes = _track(LINE, 8.5)
    wrong = [dict(n, pitch=45) if n["pitch"] == 33 else n for n in notes]  # every A1 written an octave up
    heard = O.check(y, wrong)
    assert all(h["better"] is None for h, n in zip(heard, wrong) if n["pitch"] == 45)  # they explain each other


def test_a_note_is_never_compared_with_itself():
    y, notes = _track([(0.5, 0.45, 33), (1.0, 0.45, 40), (1.5, 0.45, 47)], 2.5)
    assert O.check(y, notes) == [None, None, None]  # no two within reach of each other: nothing to compare with
    y, notes = _track([(0.5, 0.45, 33), (1.0, 0.45, 40), (1.5, 0.45, 33)], 2.5)
    heard = O.check(y, [notes[0], notes[1], dict(notes[2], pitch=45)])  # the only "A2" sounds like the only A1
    assert heard[2]["better"] == 33 and heard[0]["better"] == 45 and heard[1] is None  # one of the two is wrong
    y, notes = _track([(0.5, 0.45, 33), (1.0, 0.45, 35), (1.5, 0.45, 47)], 2.5)
    heard = O.check(y, notes)
    assert heard[0]["explained"] > 0.8 and heard[1]["explained"] > 0.8 and heard[2] is None  # a neighbour, shifted


def test_chords_and_very_short_notes_are_left_out():
    y, notes = _track(LINE, 8.5)
    notes[3] = dict(notes[3], also=[45])
    notes[6] = dict(notes[6], end=notes[6]["start"] + 0.06)
    heard = O.check(y, notes)
    assert heard[3] is None and heard[6] is None and all(h for k, h in enumerate(heard) if k not in (3, 6))
    assert 3 not in O.Bank(y, notes).by_pitch[38] and 6 not in O.Bank(y, notes).by_pitch[40]


def _alike(a, b):
    """Overlap of two tracks' spectra over time (what a listener would compare)."""
    A, B = (np.abs(librosa.stft(x, n_fft=4096, hop_length=441)[3:140]) for x in (a, b))
    return float(np.minimum(A, B).sum() / np.maximum(A, B).sum())


def test_the_rebuilt_track_matches_the_bass_where_the_notes_are_right():
    y, notes = _track(LINE, 8.5)
    rebuilt = O.rebuild(y, notes)
    assert rebuilt.dtype == np.float32 and len(rebuilt) == len(y) and _alike(y, rebuilt) > 0.8
    wrong = [dict(n, pitch=n["pitch"] + {1: 1, 9: -1}.get(k, 0)) for k, n in enumerate(notes)]  # two wrong notes
    assert _alike(y, O.rebuild(y, wrong)) < _alike(y, rebuilt) - 0.05
    fewer = [n for k, n in enumerate(notes) if k % 4 != 1]  # four notes left out
    assert _alike(y, O.rebuild(y, fewer)) < _alike(y, rebuilt) - 0.1


def test_each_note_is_rebuilt_from_another_note():
    y, notes = _track([(0.5, 0.45, 33)], 1.5)
    assert not O.rebuild(y, notes).any()  # the only note: nothing to play it with
    y, notes = _track([(0.5, 0.45, 33), (1.0, 0.45, 33)], 2.0)
    rebuilt = O.rebuild(y, notes)
    first, second = rebuilt[int(0.5 * SR) : int(0.9 * SR)], rebuilt[int(1.0 * SR) : int(1.4 * SR)]
    assert np.abs(first).max() > 0.05 and np.abs(second).max() > 0.05
    y, notes = _track([(0.5, 0.45, 33), (1.0, 0.45, 35)], 2.0)  # a neighbour, played two semitones lower
    low = O.rebuild(y, notes)[int(0.55 * SR) : int(0.9 * SR)]
    spec = np.abs(np.fft.rfft(low * np.hanning(len(low)), 1 << 16))
    peak = np.fft.rfftfreq(1 << 16, 1 / SR)[spec.argmax()]
    assert min(abs(peak / (k * librosa.midi_to_hz(33)) - 1) for k in (1, 2, 3)) < 0.02  # it sounds an A1


def test_a_note_longer_than_every_sample_still_lasts_and_a_quiet_spot_is_still_played():
    y, notes = _track([(0.5, 0.3, 33), (1.0, 0.3, 33), (1.5, 1.2, 33)], 3.0)
    rebuilt = O.rebuild(y, notes)
    assert np.abs(rebuilt[int(2.3 * SR) : int(2.6 * SR)]).max() > 0.005  # the short samples were stretched
    ghost = notes + [dict(start=2.8, end=2.95, pitch=33)]  # written where nothing sounds
    assert np.abs(O.rebuild(y, ghost)[int(2.82 * SR) : int(2.9 * SR)]).max() > 0.02
