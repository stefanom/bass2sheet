"""The song's harmony: its chords, where they change, the bass under them, and its key (synthesized audio only)."""

import json
import xml.dom.minidom

import numpy as np
import pytest
from conftest import note
from songsynth import make_song

from bass2sheet import harmony, rhythm, score
from bass2sheet.common import FLATS, SHARPS, SR

# (root for the bass, the pad's notes): the chords the synthesized songs below are made of
E_MINOR, C, G, D = (28, (52, 55, 59)), (36, (48, 52, 55)), (31, (55, 59, 62)), (38, (50, 54, 57))
F, A_MINOR, A, B_FLAT = (29, (53, 57, 60)), (33, (57, 60, 64)), (33, (57, 61, 64)), (34, (58, 62, 65))


def heard_in(song):
    """(stems, notes, grid, origin) of a synthesized song, as the pipeline hands them to harmony.find: the
    song's own bass notes on a grid of sixteenths made from its own beats. Bar 1 is the song's first bar."""
    grid, origin = rhythm.beat_grid(song["beats"], 1, 0, song["mix"].shape[1] / song["sr"])
    notes = rhythm.quantize([note(s, e, p) for s, e, p in song["notes"]], grid, origin)
    return song["stems"], notes, grid, origin


def names(chords):
    return [FLATS[c["root"]] + c["quality"] + ("/" + FLATS[c["bass"]] if c["bass"] is not None else "") for c in chords]


@pytest.fixture(scope="module")
def found(song):
    return harmony.find(*heard_in(song)), heard_in(song)


# ------------------------------------------------------------------------------------------------- the chords


def test_the_chords_of_a_song_are_found_each_with_its_bar(found):
    (_, _, grid, origin), chords = found[1], found[0]["chords"]
    assert names(chords) == ["Em", "C", "G", "D"] * 3
    for k, c in enumerate(chords):  # each starts on its bar line and lasts the bar
        assert (c["at"], c["until"]) == (origin + 16 * k, origin + 16 * (k + 1))
        assert abs(c["start"] - grid[c["at"]]) < 1e-3 and abs(c["end"] - grid[c["until"]]) < 1e-3
        assert c["bass"] is None and 0.8 <= c["score"] <= 1.0


def test_what_is_found_is_plain_data_that_can_be_stored(found):
    again = json.loads(json.dumps(found[0]))
    assert again == found[0] and set(again) == {"keys", "chords"}
    assert set(again["chords"][0]) == {"start", "end", "at", "until", "root", "quality", "bass", "score"}
    assert set(again["keys"][0]) == {"start", "end", "at", "until", "root", "major", "score", "other"}


def test_sevenths_and_a_chord_that_changes_in_the_middle_of_a_bar_are_found():
    e_minor7, g_major7, d7 = (28, (52, 55, 59, 62)), (31, (55, 59, 62, 66)), (38, (50, 54, 57, 60))
    song = make_song(bars=8, chords=[e_minor7, [C, D], g_major7, d7])
    stems, notes, grid, origin = heard_in(song)
    chords = harmony.find(stems, notes, grid, origin)["chords"]
    assert names(chords) == ["Em7", "C", "D", "Gmaj7", "D7"] * 2
    assert chords[2]["at"] == origin + 16 + 8  # the D comes in on beat 3 of the second bar


def test_a_bass_on_the_third_is_written_under_the_chord_and_not_taken_for_its_root():
    c_over_e = (28, (48, 52, 55))  # the pad plays C major, the bass plays from E
    song = make_song(bars=8, chords=[c_over_e, F, c_over_e, G])
    chords = harmony.find(*heard_in(song))["chords"]
    assert names(chords) == ["C/E", "F", "C/E", "G"] * 2


def test_where_only_bass_and_drums_play_there_is_no_chord(song):
    stems, notes, grid, origin = heard_in(song)
    quiet = dict(stems, harmony=stems["harmony"].copy())
    half = quiet["harmony"].shape[1] // 2
    quiet["harmony"][:, :half] = 0  # the instruments come in halfway through the song
    chords = harmony.find(quiet, notes, grid, origin)["chords"]
    assert chords and chords[0]["start"] >= half / SR - 0.3
    nothing = dict(stems, harmony=np.zeros_like(stems["harmony"]))
    assert harmony.find(nothing, notes, grid, origin) == dict(keys=[], chords=[])
    assert harmony.find(stems, notes, grid[:2], 0) == dict(keys=[], chords=[])  # nor is there in no time at all


def test_a_record_tuned_a_little_sharp_is_heard_in_tune():
    t = np.arange(2 * SR) / SR
    for cents in (0, 30):
        spec = harmony.spectrum(np.sin(2 * np.pi * 220 * 2 ** (cents / 1200) * t).astype(np.float32))
        assert spec.shape[0] == 60 and int(spec.mean(1).argmax()) == 21  # A3, counted from C2
        assert int(harmony.chroma(spec).mean(1).argmax()) == 9  # A
    assert not harmony.spectrum(np.zeros(SR, np.float32)).any()


def test_of_bass_notes_played_together_the_lowest_counts():
    edges = np.array([0.0, 1.0, 2.0])
    held = harmony.held_per_frame([note(0.0, 1.5, 43, also=[36]), note(1.5, 2.0, 40)], edges)
    assert held[0, 0] == 1.0 and held[0, 7] == 0.0  # C under the G: the C is the bass
    assert np.allclose(held[1, [0, 4]], 0.5)  # half the second frame each


def test_the_line_of_chords_changes_where_a_change_is_cheap():
    fit = np.zeros((16, 3))
    fit[:9, 0], fit[9:, 1] = 1.0, 1.0  # the notes change a frame late: an eighth after the bar line
    cost = np.full(16, 5.0)
    cost[8] = 1.0  # the bar line
    assert list(harmony.best_line(fit, cost)) == [0] * 8 + [1] * 8
    assert list(harmony.best_line(fit, np.full(16, 5.0))) == [0] * 9 + [1] * 7  # where the notes change, otherwise
    likely = harmony.likelihoods(fit, cost)
    assert np.allclose(likely.sum(1), 1) and likely[3, 0] > 0.6 and likely[12, 1] > 0.6
    assert likely[8].max() < likely[3].max()  # less sure at the change than inside a chord


def test_a_chords_score_rises_with_its_likelihood_and_stays_between_nothing_and_sure():
    scores = [harmony.share_right(x) for x in np.linspace(0, 1, 101)]
    assert scores[0] == 0.0 and 0.9 < scores[-1] <= 1.0 and all(a <= b for a, b in zip(scores, scores[1:]))


# --------------------------------------------------------------------------------------------------- the key


def likely_of(*chords):
    """Frames in which each of the given (root, quality) chords in turn is certain, eight frames (a bar) each."""
    out = np.zeros((8 * len(chords), 12 * len(harmony.QUALITIES) + 1))
    for k, (root, quality) in enumerate(chords):
        out[8 * k : 8 * k + 8, 12 * harmony.QUALITIES.index(quality) + root] = 1.0
    return out


def keys_of(likely):
    line, fit = harmony.key_line(likely)
    heard = dict(index=2 * np.arange(len(line)), edges=0.25 * np.arange(len(line) + 1))
    return harmony._keys(line, fit, heard, 0.25 * len(line))


def test_the_key_is_read_from_the_chords_and_its_other_key_is_the_relative():
    major = keys_of(likely_of((0, ""), (5, ""), (7, ""), (0, ""), (9, "m"), (5, ""), (7, "7"), (0, "")))
    assert [(k["root"], k["major"]) for k in major] == [(0, True)]
    assert major[0]["other"] == dict(root=9, major=False) and 0.5 < major[0]["score"] < 1
    minor = keys_of(likely_of((9, "m"), (5, ""), (7, ""), (9, "m"), (2, "m"), (4, "7"), (9, "m"), (9, "m")))
    assert [(k["root"], k["major"]) for k in minor] == [(9, False)] and minor[0]["other"] == dict(root=0, major=True)
    assert (major[0]["at"], major[0]["until"], major[0]["start"], major[0]["end"]) == (0, 128, 0.0, 16.0)


def test_a_song_that_moves_up_a_tone_changes_key_and_a_borrowed_chord_does_not():
    in_c, in_d = [(0, ""), (5, ""), (7, ""), (0, "")], [(2, ""), (7, ""), (9, ""), (2, "")]
    moved = keys_of(likely_of(*in_c * 4, *in_d * 4))
    assert [(k["root"], k["major"]) for k in moved] == [(0, True), (2, True)]
    assert moved[1]["at"] == 2 * 8 * 16 and moved[0]["until"] == moved[1]["at"]  # where the D major chords begin
    borrowed = keys_of(likely_of(*in_c * 4, (10, ""), (8, ""), *in_c * 4))  # two bars of Bb and Ab
    assert [(k["root"], k["major"]) for k in borrowed] == [(0, True)]


def test_the_key_of_the_synthesized_song_is_g_major_or_its_relative_e_minor(found):
    keys = found[0]["keys"]
    assert len(keys) == 1 and keys[0]["start"] == 0.0 and keys[0]["until"] >= found[0]["chords"][-1]["until"]
    pair = {(keys[0]["root"], keys[0]["major"]), (keys[0]["other"]["root"], keys[0]["other"]["major"])}
    assert pair == {(7, True), (4, False)}
    assert harmony.main_key(found[0]) in (("G major", 7, True), ("E minor", 4, False))


def test_without_chords_the_key_is_the_one_the_bass_notes_fit(song):
    nothing = dict(keys=[], chords=[])
    notes = [note(s, e, p) for s, e, p in song["notes"]]
    assert harmony.main_key(nothing, notes)[0] in ("E minor", "G major")
    assert harmony.main_key(nothing) == ("C major", 0, True)  # and with no notes either, the plainest key


def test_the_key_a_song_is_in_for_the_longest_is_its_main_key():
    span = lambda start, end, root, major: dict(start=start, end=end, root=root, major=major)
    keys = [span(0, 40, 2, True), span(40, 70, 3, True), span(70, 100, 2, True)]
    assert harmony.main_key(dict(keys=keys, chords=[1])) == ("D major", 2, True)
    assert harmony.key_name(1, False) == "C# minor" and harmony.key_name(1, True) == "Db major"


# ------------------------------------------------------------------------- the stored data of before, and the sheet


def chord(at, until, root, quality="", bass=None):
    return dict(start=0.0, end=0.0, at=at, until=until, root=root, quality=quality, bass=bass, score=0.9)


def key(root, major, at=0, until=10_000):
    return dict(start=0.0, end=0.0, at=at, until=until, root=root, major=major, score=0.9, other=None)


def test_the_chord_per_bar_of_older_data_is_the_triad_that_holds_most_of_the_bar():
    origin = 12  # bar 1 starts at grid index 12
    chords = [chord(6, 12, 9), chord(12, 22, 4, "m7"), chord(22, 44, 0, "maj7", bass=4), chord(60, 76, 2, "7")]
    found = dict(keys=[key(4, False)], chords=chords)  # the A before bar 1 is a pickup: older data has no bar 0
    bars = harmony.per_bar(found, origin, ("E minor", 4, False))
    assert [(b["bar"], b["chord"], b["quality"]) for b in bars] == [(1, "Em", "m"), (2, "C", ""), (4, "D", "")]
    assert bars[0]["names"] == SHARPS and bars[0]["score"] == 0.9  # bar 3 has no chord: no entry


def test_the_sheet_gets_one_chord_a_bar_and_a_second_where_the_chord_changes_on_beat_three():
    found = dict(
        keys=[key(0, True)],
        chords=[chord(0, 16, 0), chord(16, 24, 5), chord(24, 32, 7, "7"), chord(32, 46, 9, "m"), chord(46, 48, 4)],
    )
    sheet = harmony.for_sheet(dict(harmony=found, origin=0, chords=[]))
    assert {bar: [(at, FLATS[c["root"]] + c["quality"]) for at, c in entries] for bar, entries in sheet.items()} == {
        1: [(0, "C")],
        2: [(0, "F"), (8, "G7")],
        3: [(0, "Am")],  # the E takes an eighth of the bar's second half: not written
    }


def test_data_stored_before_there_was_harmony_still_gives_the_sheet_its_chords():
    old = dict(
        chords=[
            dict(bar=1, root=4, quality="m", names=SHARPS, chord="Em"),
            dict(bar=2, root=0, quality="", names=FLATS, chord="C"),
        ],
        key=["E minor", 4, False],
        origin=12,
    )
    for data in (old, dict(old, harmony=None), dict(old, harmony=dict(keys=[], chords=[]))):
        sheet = harmony.for_sheet(data)
        assert [(bar, at, c["root"], c["quality"], c["bass"]) for bar, ((at, c),) in sheet.items()] == [
            (1, 0, 4, "m", None),
            (2, 0, 0, "", None),
        ]


def test_chords_are_spelled_by_the_key_they_are_in():
    name = lambda root, key_root, major, quality="": (lambda c: c["names"][c["root"]] + c["quality"])(
        harmony.for_sheet(
            dict(harmony=dict(keys=[key(key_root, major)], chords=[chord(0, 16, root, quality)]), origin=0)
        )[1][0][1]
    )
    assert name(8, 4, True, "m") == "G#m"  # the third step of E major
    assert name(1, 6, False) == "C#"  # the fifth step of F# minor, not Db
    assert name(10, 2, True) == "Bb"  # the flat sixth borrowed in D major, not A#
    assert name(3, 7, True) == "Eb" and name(6, 0, True) == "F#"
    assert harmony.note_name(8, "E", SHARPS) == "G#" and harmony.note_name(8, "F", FLATS) == "Ab"
    assert harmony.note_name(4, "Ab", FLATS) == "E"  # Fb is not written
    # and a note of the chord is spelled as the chord has it: in F major, the third of an E major chord is G#
    sheet = harmony.for_sheet(dict(harmony=dict(keys=[key(5, True)], chords=[chord(0, 16, 4)]), origin=0))
    assert sheet[1][0][1]["names"][8] == "G#" and sheet[1][0][1]["names"][10] == "Bb"


def test_the_score_writes_both_chords_of_a_bar_with_their_kind_and_the_bass_under_them():
    chords = [chord(0, 8, 0, "maj7"), chord(8, 16, 9, "m7"), chord(16, 32, 0, "", bass=4), chord(32, 48, 10, "7")]
    found = dict(keys=[key(0, True)], chords=chords)
    quantized = [
        dict(note(0.5 * g, 0.5 * g + 0.4, 36), grid_start=4 * g, grid_len=4, bar=g // 4 + 1) for g in range(12)
    ]
    sheet = harmony.for_sheet(dict(harmony=found, origin=0))
    doc = xml.dom.minidom.parseString(
        score.musicxml(quantized, sheet, ("C major", 0, True), 120.0, 0.0, "T", "A", {}, False)
    )
    symbols = doc.getElementsByTagName("harmony")
    text = lambda el, tag: el.getElementsByTagName(tag)[0].firstChild.data if el.getElementsByTagName(tag) else None
    fields = ("root-step", "root-alter", "kind", "bass-step", "offset")
    assert [tuple(text(s, f) for f in fields) for s in symbols] == [
        ("C", None, "major-seventh", None, None),
        ("A", None, "minor-seventh", None, "8"),  # on beat 3: eight sixteenths into the bar
        ("C", None, "major", "E", None),
        ("B", "-1", "dominant", None, None),  # B flat, the flat seventh step of C
    ]
    assert [s.parentNode.getAttribute("number") for s in symbols] == ["1", "1", "2", "3"]


def test_a_chord_that_goes_on_is_named_once_and_again_after_a_bar_of_rest():
    found = dict(keys=[key(0, True)], chords=[chord(0, 80, 0)])
    quantized = [
        dict(note(0.5 * g, 0.5 * g + 0.4, 36), grid_start=4 * g, grid_len=4, bar=g // 4 + 1)
        for g in (*range(8), *range(12, 20))
    ]
    xml_text = score.musicxml(
        quantized,
        harmony.for_sheet(dict(harmony=found, origin=0)),
        ("C major", 0, True),
        120.0,
        0.0,
        "T",
        "A",
        {},
        False,
    )
    doc = xml.dom.minidom.parseString(xml_text)
    assert [s.parentNode.getAttribute("number") for s in doc.getElementsByTagName("harmony")] == ["1", "4"]
