import numpy as np
from conftest import note

from bass2sheet import rhythm


def test_beat_grid_is_sixteenths_with_origin_on_a_downbeat():
    pulses = 2.0 + 0.5 * np.arange(40)  # quarter notes at 120 BPM; the first tracked pulse is beat 3 of a bar
    grid, origin = rhythm.beat_grid(pulses, per_quarter=1, phase=2, end=24.0)
    assert np.allclose(np.diff(grid), 0.125)
    assert grid[0] <= 0 and grid[-1] >= 24.0
    assert np.isclose((grid[origin] - 3.0) % 2.0, 0)  # beat 1 is two pulses after the first tracked pulse
    assert grid[origin] < 2.0  # the earliest downbeat, extended backward


def test_beat_grid_with_eighth_note_pulses():
    pulses = 1.0 + 0.25 * np.arange(80)
    grid, origin = rhythm.beat_grid(pulses, per_quarter=2, phase=0, end=22.0)
    assert np.allclose(np.diff(grid), 0.125)
    assert np.isclose((grid[origin] - 1.0) % 2.0, 0)


def test_quantize_places_notes_in_bars_and_merges_same_sixteenth():
    grid = np.arange(0, 20, 0.125)
    origin = 8  # bar 1 starts at 1.0 s
    notes = [
        note(1.01, 1.49, 40),
        note(1.52, 1.54, 40),
        note(1.55, 1.80, 40),
        note(3.02, 3.9, 43),
        note(0.76, 0.99, 38),
    ]
    notes.sort(key=lambda n: n["start"])
    out = rhythm.quantize(notes, grid, origin)
    by_start = {round(n["start"], 2): n for n in out}
    assert by_start[0.76]["bar"] == 0 and by_start[0.76]["grid_start"] == -2  # pickup
    first = by_start[1.01]
    assert (first["bar"], first["beat"], first["sixteenth"], first["grid_len"]) == (1, 1, 1, 4)
    assert (by_start[1.52]["beat"], by_start[1.52]["sixteenth"]) == (2, 1)
    assert 1.55 not in by_start and by_start[1.52]["end"] == 1.80  # same pitch, same 16th: one note
    assert (by_start[3.02]["bar"], by_start[3.02]["beat"]) == (2, 1)
    assert all(n["grid_len"] >= 1 for n in out)


def test_estimate_key_recognises_major_and_minor_profiles():
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    assert rhythm.estimate_key(np.roll(major, 7)[:, None])[:3] == ("G major", 7, True)
    name, root, is_major = rhythm.estimate_key(np.roll(minor, 11)[:, None])
    assert (name, root, is_major) == ("B minor", 11, False)
    assert rhythm.estimate_key(np.roll(major, 3)[:, None])[0] == "Eb major"  # flat keys are spelled with flats


def test_beats_downbeats_and_chords_on_the_synthetic_song(song):
    st = song["stems"]
    tempo, pulses, per_quarter, phase = rhythm.beats_and_downbeats(st["drums"], st["bass"], st["harmony"])
    assert abs(tempo - song["tempo"]) / song["tempo"] < 0.02 and per_quarter == 1
    grid, origin = rhythm.beat_grid(pulses, per_quarter, phase, song["mix"].shape[1] / song["sr"])
    # the grid's downbeats coincide with the song's (bar numbering may start earlier: the grid extends backward)
    offset = (song["downbeats"][0] - grid[origin]) / (4 * 60 / tempo)
    assert abs(offset - round(offset)) < 0.05
    notes = [note(s, e, p) for s, e, p in song["notes"]]
    notes = rhythm.quantize(notes, grid, origin)
    chords, chroma = rhythm.chords_per_bar(st["harmony"], notes, grid, origin)
    first = int(round(offset)) + 1  # bar number of the song's first bar
    names = [c["chord"] for c in chords if first <= c["bar"] < first + 4]
    assert names == ["Em", "C", "G", "D"]
    assert chroma.shape[0] == 12
