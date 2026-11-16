"""The song's harmony for the copilot's page: the parts its bass line goes round, the chords the person writes on
its beats (named against a home), the chords the tool offers for a beat, and what the bass plays."""

import json
import os
import shutil
import threading
import types
import urllib.error
import urllib.request

import pytest

from bass2sheet import copilot, copilot_harmony, copilot_server
from bass2sheet.common import Bass2SheetError

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")
C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11


@pytest.mark.parametrize(
    "root, quality, home, written",
    [
        (C, "", (C, True), "I"),
        (F, "", (C, True), "IV"),
        (G, "7", (C, True), "V7"),
        (A, "m", (C, True), "vi"),
        (D, "m7", (C, True), "ii7"),
        (C, "maj7", (C, True), "Imaj7"),
        (B, "dim", (C, True), "vii°"),
        (B, "dim7", (C, True), "vii°7"),
        (D, "m7b5", (C, True), "iiø7"),
        (10, "", (C, True), "bVII"),
        (6, "", (C, True), "#IV"),
        (G, "sus4", (C, True), "Vsus4"),
        # a minor key is named on the major scale of its home, as the people who play these songs write it
        (A, "m", (A, False), "i"),
        (C, "", (A, False), "bIII"),
        (D, "m", (A, False), "iv"),
        (F, "", (A, False), "bVI"),
        (G, "", (A, False), "bVII"),
        (E, "7", (A, False), "V7"),
    ],
)
def test_a_chord_is_named_as_a_step_above_home(root, quality, home, written):
    assert copilot_harmony.numeral(root, quality, home) == written


def test_small_letters_are_chords_with_a_minor_third():
    assert [q for q in ("", "m", "7", "m7", "maj7", "dim", "sus4") if copilot_harmony.minor_like(q)] == [
        "m",
        "m7",
        "dim",
    ]


def test_the_relative_key_has_the_same_notes_and_the_other_mode():
    assert copilot_harmony.relative((C, True)) == (A, False) and copilot_harmony.relative((A, False)) == (C, True)
    assert copilot_harmony.relative((11, False)) == (D, True)
    assert copilot_harmony.key_name((11, False)) == "B minor" and copilot_harmony.key_name((3, True)) == "Eb major"


def test_a_bass_note_is_a_part_of_the_chord_or_a_passing_note():
    parts = {40: "root", 52: "root", 47: "fifth", 43: "third", 44: "third", 50: "seventh", 51: "seventh", 45: "other"}
    assert {pitch: copilot_harmony.role(pitch, E) for pitch in parts} == parts


@pytest.mark.parametrize(
    "root, quality, home, does",
    [
        (C, "", (C, True), "rest"),
        (A, "m", (C, True), "rest"),
        (E, "m", (C, True), "rest"),
        (C, "m", (C, True), "rest"),  # borrowed from the minor on the same home
        (3, "", (C, True), "rest"),
        (C, "sus4", (C, True), "rest"),
        (F, "", (C, True), "drift"),
        (D, "m", (C, True), "drift"),
        (F, "m", (C, True), "drift"),
        (8, "", (C, True), "drift"),
        (10, "", (C, True), "drift"),
        (1, "", (C, True), "drift"),
        (G, "m", (C, True), "drift"),  # a minor v does not pull
        (D, "dim", (C, True), "drift"),
        (D, "m7b5", (C, True), "drift"),
        (G, "", (C, True), "pull"),
        (G, "7", (C, True), "pull"),
        (B, "dim", (C, True), "pull"),
        (6, "dim", (C, True), "pull"),
        (C, "7", (C, True), "pull"),  # the dominant of IV
        (D, "", (C, True), "pull"),  # II, III, VI and VII as major chords lead to other chords of the key
        (E, "", (C, True), "pull"),
        (A, "", (C, True), "pull"),
        (B, "", (C, True), "pull"),
        (A, "m", (A, False), "rest"),
        (C, "", (A, False), "rest"),
        (A, "", (A, False), "rest"),
        (D, "m", (A, False), "drift"),
        (D, "", (A, False), "drift"),
        (E, "m", (A, False), "drift"),
        (F, "", (A, False), "drift"),
        (G, "", (A, False), "drift"),
        (E, "", (A, False), "pull"),
        (B, "", (A, False), "pull"),
        (G, "7", (A, False), "pull"),
    ],
)
def test_a_chord_has_a_job_in_a_key(root, quality, home, does):
    assert copilot_harmony.job(root, quality, home) == does


@pytest.mark.parametrize(
    "root, quality, home, out",
    [
        (C, "", (C, True), 0),
        (G, "7", (C, True), 0),
        (B, "dim", (C, True), 0),
        (F, "m", (C, True), 1),  # its notes are in C minor
        (10, "", (C, True), 1),
        (C, "7", (C, True), 1),  # the dominant of a chord of the key
        (A, "7", (C, True), 1),
        (6, "dim", (C, True), 1),  # a diminished chord a semitone under a note of the key
        (1, "", (C, True), 2),
        (1, "7", (C, True), 2),
        (A, "m", (A, False), 0),
        (E, "", (A, False), 0),  # the V of a minor key, with its raised seventh
        (G, "", (A, False), 0),
        (A, "", (A, False), 1),
        (D, "", (A, False), 1),
        (10, "", (A, False), 2),
    ],
)
def test_a_chord_is_in_the_key_one_step_outside_it_or_far_out(root, quality, home, out):
    assert copilot_harmony.ring(root, quality, home) == out


def test_brightness_and_distance_from_rest():
    H, home = copilot_harmony, (C, True)
    assert H.bright(C, "", home) == pytest.approx(5 / 3) and H.bright(A, "m", home) == pytest.approx(3 - 2 / 3)
    assert H.bright(F, "", home) < H.bright(C, "", home) < H.bright(G, "", home)  # flatter is darker
    assert H.bright(1, "dim", home) == pytest.approx(7 - 3)  # a passing diminished chord is heard as a sharp
    assert H.bright(B, "dim", home) == pytest.approx(5 - 3)
    far = [
        H.far(r, q, home)
        for r, q in ((C, ""), (A, "m"), (F, ""), (G, ""), (G, "7"), (B, "dim"), (E, "7"), (1, "7"), (8, ""))
    ]
    assert far == [0.0, 0.8, 2.2, 3.6, 4.2, 4.2, 5.0, 5.8, 3.0]
    assert H.kind("7") == H.kind("sus4") == "dom" and H.kind("m7b5") == H.kind("dim7") == "dim"
    assert H.kind("m6") == "min" and H.kind("maj7") == H.kind("") == "maj" and H.tones("m9") == (0, 3, 7)


def _spots(home, spent=None):
    spots, where = copilot_harmony.roads(home, spent or {})
    named = {copilot_harmony.numeral(s["root"], s["quality"], home): s for s in spots}
    return spots, where, named


def test_the_map_of_jobs_puts_each_regular_chord_in_its_place():
    spots, _, named = _spots((C, True))
    assert len(spots) == 21 and all(s["regular"] and s["seconds"] == 0 for s in spots)
    at = {name: (round(s["x"]), s["y"]) for name, s in named.items()}
    assert at["I"] == (380, 356) and at["vi"] == (420, 356) and at["iii"] == (480, 356)  # home, in the key
    assert at["bIII"] == (280, 414) and at["i"] == (320, 414)  # home, borrowed: a row out
    assert at["IV"] == (271, 182) and at["ii"] == (311, 182)  # away: the darker chord to the left
    assert [at[n] for n in ("bVI", "iv", "ii°", "bVII", "v")] == [
        (91, 124),
        (131, 124),
        (171, 124),
        (211, 124),
        (251, 124),
    ]
    assert at["bII"] == (31, 66) and at["bII7"][1] == 66  # the rim
    assert at["V"][1] == at["vii°"][1] == 182 and at["vii°"][0] < at["V"][0]
    assert [at[n][0] for n in ("I7", "II7", "VI7", "III7", "VII7")] == [438, 544, 604, 664, 724]
    for s in spots:  # every chord is inside the region of its job, in the row of its distance from the key
        x0, y0, w, h = copilot_harmony.REGIONS[s["job"]]
        assert x0 + 24 <= s["x"] <= x0 + w - 24 and y0 < s["y"] < y0 + h
        assert s["y"] == copilot_harmony.ROWS[s["job"]][s["ring"]]
    minor = _spots((A, False))[2]
    assert (
        set(minor) >= {"i", "bIII", "I", "iv", "bVI", "bVII", "v", "V", "vii°", "bVII7"} and minor["i"]["job"] == "rest"
    )


def test_a_songs_chords_land_on_the_map():
    home = (C, True)
    spent = {(C, ""): 30.0, (G, "7"): 10.0, (G, ""): 5.0, (A, "m7"): 4.0, (6, "m7"): 2.0, (8, "7"): 3.0}
    spots, where, named = _spots(home, spent)
    # V and V7 are one place, named as the song has it most; A minor seventh sits where vi does
    assert where[(G, "7")] == where[(G, "")] and spots[where[(G, "7")]]["seconds"] == 15.0 and "V7" in named
    assert spots[where[(A, "m7")]]["regular"] and named["vi7"]["seconds"] == 4.0 and "vi" not in named
    assert named["I"]["seconds"] == 30.0 and named["IV"]["seconds"] == 0
    # a chord the map does not have gets a place of its own, clear of the others
    for name in ("#iv7", "bVI7"):
        own = named[name]
        assert not own["regular"] and own["seconds"] > 0
        others = [s for s in spots if s is not own]
        assert all(((s["x"] - own["x"]) ** 2 + (s["y"] - own["y"]) ** 2) ** 0.5 >= 39 for s in others)
    assert len(spots) == 23 and set(where) == set(spent)


def test_the_wheel_has_home_at_the_top():
    H = copilot_harmony
    spots, where = H.wheel((C, True), {(C, ""): 30.0, (G, "7"): 10.0, (A, "m"): 3.0})
    at = {(s["root"], H.kind(s["quality"])): (s["x"], s["y"]) for s in spots}
    assert len(spots) == 36
    assert spots[where[(C, "")]]["x"] == 300 and spots[where[(C, "")]]["y"] == 35  # C at the top
    assert at[(A, "min")] == (300.0, 85.0) and at[(B, "dim")] == (300.0, 129.0)  # the chords with its notes under it
    g = spots[where[(G, "7")]]  # a fifth up is the next one clockwise; a dominant goes with the major chords
    assert g["root"] == G and g["quality"] == "7" and g["job"] == "pull" and g["x"] > 300 and g["turn"] == 1
    assert [s["turn"] for s in spots if s["root"] == F and s["quality"] == ""] == [11]
    # in a minor key the home chord is at the top of its ring, under the major chord with the same notes
    minor, where = H.wheel((A, False), {(A, "m"): 9.0})
    assert (minor[where[(A, "m")]]["x"], minor[where[(A, "m")]]["y"]) == (300.0, 85.0)
    assert [(s["x"], s["y"]) for s in minor if s["root"] == C and s["quality"] == ""] == [(300.0, 35.0)]
    # by semitones the three chords on a note share a spoke, and the next note up is the next spoke
    semis, _ = H.wheel((C, True), {}, "semitones")
    assert {s["turn"] for s in semis if s["root"] == C} == {0} and {s["turn"] for s in semis if s["root"] == 1} == {1}
    assert {s["turn"] for s in semis if s["root"] == B} == {11}


def _shape(found):
    return [(p["first"], p["bars"], p["letter"], "".join(p["loop"])) for p in found]


def test_a_song_is_cut_into_the_loops_it_goes_round():
    parts = copilot_harmony.parts
    # two bars that lead in, a loop of four bars three and a half times, a loop of two, a passage, the first loop again
    found = _shape(parts(list("xxABCDABCDABCDABEFEFEFEFqrsABCDABCDAB")))
    assert found == [
        (0, 2, "A", "xx"),
        (2, 14, "B", "ABCD"),
        (16, 8, "C", "EF"),
        (24, 3, "D", "qrs"),
        (27, 10, "B", "ABCD"),
    ]
    assert _shape(parts(list("EEEEEEEEABABABAB"))) == [
        (0, 8, "A", "E"),
        (8, 8, "B", "AB"),
    ]  # one bar held is a loop of a bar
    assert _shape(parts(list("ABCDEFG"))) == [(0, 7, "A", "ABCDEFG")]  # nothing repeats: one passage
    assert _shape(parts(list("ABCDABCEABCDABCEABCDABCE"))) == [(0, 24, "A", "ABCDABCE")]  # eight bars, not four
    # one bar heard differently in one round does not break the loop, and the loop is what most rounds have
    assert _shape(parts(list("ABCDABXDABCDABCD"))) == [(0, 16, "A", "ABCD")]
    assert _shape(parts(list("AXCDABCDABCDABCD"))) == [(0, 16, "A", "ABCD")]
    assert parts([]) == [] and _shape(parts(list("AB"))) == [(0, 2, "A", "AB")]
    assert _shape(parts(list("AAB"))) == [(0, 3, "A", "AAB")]  # a bar held for two bars is not a loop yet
    # a loop that comes back begun part of the way round is the same part, and says how far round it begins
    found = parts(list("ABCDABCDABCD" + "xyzD" + "ABCDABCDABCD"))
    assert _shape(found) == [(0, 12, "A", "ABCD"), (12, 3, "B", "xyz"), (15, 13, "A", "ABCD")]
    assert [p["skip"] for p in found] == [0, 0, 3]  # it comes back in on the loop's last bar
    found = parts(list("ABCDABCDABCDxyCDABCDABCDAB"))
    assert _shape(found) == [(0, 12, "A", "ABCD"), (12, 2, "B", "xy"), (14, 12, "A", "ABCD")]
    assert [p["skip"] for p in found] == [0, 0, 2]
    assert [p["skip"] for p in parts(list("EEEEEEEEqEEEEEEEE"))] == [0, 0, 0]  # one bar held has no way round
    # a loop that differs from an earlier one in one bar of four is that part; in two bars it is another
    assert [p["letter"] for p in parts(list("ABCDABCDABCD" + "xy" + "ABCEABCEABCE"))] == ["A", "B", "A"]
    assert [p["letter"] for p in parts(list("ABCDABCDABCD" + "xy" + "ABEFABEFABEF"))] == ["A", "B", "C"]


def test_a_loop_is_taken_in_whole_rounds_and_then_from_home():
    parts = copilot_harmony.parts
    # seven bars of nothing, a bar that leads in, eight bars twice, four bars twice
    song = list("-------F" + "BBAAGGFF" * 2 + "EEAAEEAA")
    found = parts(song, "B")
    assert _shape(found) == [(0, 7, "A", "-"), (7, 17, "B", "BBAAGGFF"), (24, 8, "C", "EEAA")]
    assert [p["skip"] for p in found] == [0, 7, 0]  # the bar that leads in is the loop's last bar
    assert _shape(parts(song))[1] == (7, 17, "B", "FBBAAGGF")  # with no home given: from the bar it starts on
    # three whole rounds are not turned round to begin at home
    assert _shape(parts(list("xECGDECGDECGDyy"), "G"))[1] == (1, 12, "B", "ECGD")
    assert parts(song, "B") == found and parts(song, "B") is not found  # kept, and given out as a copy


def test_a_bass_note_is_named_as_a_step_of_its_chord():
    degree = copilot_harmony.degree
    low_e = 40  # the line stands on this E
    assert [degree(p, E, low_e) for p in (40, 52, 47, 43, 50, 38, 55, 64, 28)] == [
        "R",
        "8",
        "5",
        "b3",
        "b7",
        "b7↓",
        "b3↑",
        "R↑",
        "R↓",
    ]


def test_a_bass_note_is_named_as_a_step_of_the_key():
    step = copilot_harmony.step
    assert [step(p, (E, False)) for p in (40, 52, 47, 43, 50, 45, 41)] == ["1", "1", "5", "b3", "b7", "4", "b2"]
    assert step(55, (G, True)) == "1" and step(54, (G, True)) == "7" and step(49, (G, True)) == "b5"


def _note(start, end, pitch):
    return dict(start=start, end=end, pitch=pitch)


def test_the_note_the_bass_stands_on_in_each_bar():
    starts_on, main_notes = copilot_harmony.starts_on, copilot_harmony.main_notes
    notes = [
        _note(0.0, 0.5, 40),  # E starts bar 1
        _note(0.5, 2.05, 47),  # B is held longer, and rings on a hair over the bar line
        _note(1.95, 2.6, 45),  # A comes in a touch early: it is bar 2's first note
        _note(2.6, 4.0, 52),
        _note(5.0, 5.5, 43),  # bar 3 has one note, in its middle
        _note(5.8, 7.5, 38),  # held into bar 4 from before
    ]
    bars, ends = [0.0, 2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0, 10.0]
    assert [n and n["pitch"] for n in (starts_on(notes, a, b) for a, b in zip(bars, ends))] == [40, 45, None, 38, None]
    # the note a bar starts on counts for half a bar more than the time it is held
    assert main_notes(notes, bars, ends) == [E, A, G, D, None]
    assert main_notes([_note(0.0, 0.4, 40), _note(0.4, 2.0, 47)], [0.0], [2.0]) == [B]  # held four times as long
    assert main_notes([_note(1.97, 3.0, 40)], [0.0, 2.0], [2.0, 4.0]) == [None, E]  # a hair before the bar line


def _machine(bars=4, length=None, **more):
    grid = [round(-0.5 + 0.125 * k, 3) for k in range(4 + 16 * bars + 1)]  # bar 1 starts at 0, a bar lasts 2 s
    return dict(grid=grid, origin=4, length=2.0 * bars if length is None else length, **more)


def test_the_persons_chords_are_marks_on_beats(tmp_path):
    folder = str(tmp_path)
    read, put = copilot_harmony.read_marks, copilot_harmony.set_marks
    assert read(folder) == []
    put(
        folder,
        [dict(at=32, chord=[A, "m"]), dict(at=0, chord=[E, "m7"]), dict(at=40, chord=None), dict(at=64, chord="none")],
    )
    first = [
        dict(at=0, chord=(E, "m7")),
        dict(at=32, chord=(A, "m")),
        dict(at=40, chord=None),
        dict(at=64, chord="none"),
    ]
    assert read(folder) == first  # in order: a chord, a change still to choose, no chord from here
    # a mark is changed where it is, and taken away; taking away what is not there is nothing
    put(folder, [dict(at=40, chord=[G, ""]), dict(at=64, chord="remove"), dict(at=400, chord="remove")])
    assert read(folder) == [dict(at=0, chord=(E, "m7")), dict(at=32, chord=(A, "m")), dict(at=40, chord=(G, ""))]
    # they are kept beside the home and the fingering the person chose
    copilot_harmony.set_home(folder, (G, True))
    copilot_harmony.set_tab(folder, "low")
    put(folder, [dict(at=0, chord="remove")])
    assert copilot_harmony.read_home(folder) == (G, True) and copilot.tab_style(folder) == "low"
    kept = [dict(at=32, chord=(A, "m")), dict(at=40, chord=(G, ""))]
    assert read(folder) == kept
    for bad in (
        [],
        None,
        ["bar 3"],
        [dict(at=2, chord=[A, "m"])],  # not on a beat
        [dict(at=-4, chord=[A, "m"])],
        [dict(at=True, chord=[A, "m"])],
        [dict(at=4.0, chord=None)],
        [dict(at=4, chord=[12, ""])],
        [dict(at=4, chord=[A, "minor"])],
        [dict(at=4, chord=[A])],
        [dict(at=4, chord="maybe")],
        [dict(at=8, chord=[C, ""]), dict(at=9, chord=[C, ""])],  # one bad change: nothing of it is kept
    ):
        with pytest.raises(Bass2SheetError):
            put(folder, bad)
    assert read(folder) == kept


def test_a_chord_lasts_until_the_next_mark_or_the_end_of_its_part():
    stretches, sounding = copilot_harmony.stretches, copilot_harmony.sounding
    marks = [
        dict(at=0, chord=(E, "m")),
        dict(at=8, chord=None),
        dict(at=16, chord=(C, "")),
        dict(at=40, chord="none"),
        dict(at=48, chord=(G, "")),
        dict(at=64, chord=(D, "")),  # beyond the end of the song
    ]
    spans = stretches(_machine(length=7.5), marks, [32, 64])  # a part of two bars, and one of two
    assert [(s["at"], s["until"], s["start"], s["end"], s["chord"]) for s in spans] == [
        (0, 8, 0.0, 1.0, (E, "m")),
        (8, 16, 1.0, 2.0, None),
        (16, 32, 2.0, 4.0, (C, "")),  # to the end of its part, not on into the next
        (40, 48, 5.0, 6.0, "none"),
        (48, 64, 6.0, 7.5, (G, "")),  # the last one ends with the song
    ]
    assert [sounding(spans, at) for at in (0, 4, 10, 20, 36, 44, 60, 80)] == [
        (E, "m"),
        (E, "m"),
        None,  # a change is marked: its chord is to choose
        (C, ""),
        None,  # nothing was written from the start of the second part
        None,  # no chord from here
        (G, ""),
        None,
    ]
    assert stretches(_machine(), [], [32]) == []
    assert stretches(_machine(), [dict(at=16, chord=(C, ""))], [])[0]["until"] == 64  # no parts: to the end


def test_the_same_place_in_the_other_rounds():
    cut = copilot_harmony.parts(list("xECGDECGDECGDyyGDECGDEC"), "E")
    assert _shape(cut) == [(0, 1, "A", "x"), (1, 12, "B", "ECGD"), (13, 2, "C", "yy"), (15, 8, "B", "ECGD")]
    part, bar, round_ = copilot_harmony.place_of(cut, 16 * 6 + 4)  # bar 7, beat 2
    assert (part["first"], bar, round_) == (1, 1, 1)  # the loop's second bar, the second time round
    assert copilot_harmony.place_of(cut, 16 * 16)[1:] == (3, 0)  # the last part comes in on the loop's third bar
    assert copilot_harmony.place_of(cut, 16 * 40) is None
    # beat 3 of the bar on C: the same beat of every other bar that is the loop's second
    assert copilot_harmony.same_place(cut, 16 * 2 + 8) == [16 * b + 8 for b in (6, 10, 18, 22)]
    assert copilot_harmony.same_place(cut, 8) == [] and copilot_harmony.same_place(cut, 16 * 40) == []


def test_the_chords_the_tool_heard_are_kept_for_what_it_offers():
    heard = copilot_harmony.heard_chords
    found = [
        dict(bar=1, root=E, quality="m", score=0.9),
        dict(bar=2, root=C, quality="", score=0.0),  # a bar where no harmony was heard has no chord
        dict(bar=3, root=G, quality="", score=0.7),
        dict(bar=9, root=D, quality="", score=0.6),  # beyond the song
    ]
    old = heard(_machine(chords=found))  # a song prepared long ago: a chord a bar
    assert [(c["start"], c["end"], c["root"], c["quality"]) for c in old] == [(0.0, 2.0, E, "m"), (4.0, 6.0, G, "")]
    assert heard(_machine()) == [] and heard(_machine(chords=None)) == []
    fine = dict(chords=[dict(start=0.0, end=1.0, at=4, until=12, root=A, quality="m7", bass=None, score=0.8)])
    now = heard(_machine(chords=found, harmony=fine))  # heard eighth by eighth: those, as they are
    assert now == [dict(start=0.0, end=1.0, root=A, quality="m7", score=0.8)]


def test_the_chords_offered_for_a_stretch_of_the_bass_line():
    suggest = copilot_harmony.suggest
    line = [_note(0.0, 1.0, 40), _note(1.0, 1.5, 47), _note(1.5, 2.0, 43)]  # E, its fifth, its minor third
    got = suggest(line, 0.0, 2.0, (E, False))
    assert got[0] == (E, "m") and len(got) == 8 == len(set(got)) and all(quality in ("", "m") for _, quality in got)
    assert suggest([_note(0.0, 1.0, 40), _note(1.0, 2.0, 44)], 0.0, 2.0, (E, True))[0] == (E, "")  # a major third
    assert suggest([], 0.0, 2.0, (G, True))[0] == (G, "")  # nothing is played: the chords of the key, home first
    assert len(suggest(line, 0.0, 2.0, (E, False), most=3)) == 3
    # what the tool heard there is tried early, as its major or minor chord
    heard = [dict(start=0.0, end=2.0, root=C, quality="maj7", score=0.9)]
    assert (C, "") in suggest(line, 0.0, 2.0, (E, False), heard=heard)[:2]
    assert (C, "") not in suggest(line, 0.0, 2.0, (E, False))[:2]
    assert (C, "") not in suggest(line, 0.0, 2.0, (E, False), heard=[dict(heard[0], start=2.0, end=4.0)])[:2]
    # and what the person put at the same place in another round comes first
    assert suggest(line, 0.0, 2.0, (E, False), picked=[(A, "m7")])[0] == (A, "m")


@pytest.fixture(scope="module")
def prepared(song_files, tmp_path_factory):
    mix, stems = song_files
    work = str(tmp_path_factory.mktemp("scores"))
    opts = dict(defaults=dict(beat_tracker="librosa"), crepe_model=CREPE, device="cpu", cache_dir=work + "-cache")
    copilot.add([mix], work, opts, stems_dirs={mix: stems}, say=lambda text: None)
    return work


@pytest.fixture
def workdir(prepared, tmp_path):
    work = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, work)
    return work


# The song the tests prepare is a loop of E minor, C, G and D, a bar each, three times round, from bar 2 on; over
# each chord the bass plays root, root, fifth, octave, fifth, fourth. The tool takes it to be in G major.
ROUND = [
    dict(at=16, chord=[E, "m"]),
    dict(at=32, chord=[C, ""]),
    dict(at=48, chord=[G, ""]),
    dict(at=64, chord=[D, ""]),
]


def test_the_page_gets_the_song_with_no_chords_until_the_person_writes_some(workdir):
    desk = copilot.Desk(workdir, "mix")
    seen = copilot_harmony.view(desk)
    # home is the tool's guess, with the key that has the same notes to choose instead
    assert seen["home"] == dict(root=G, major=True, name="G major", yours=False)
    assert [(h["name"], h["guess"]) for h in seen["homes"]] == [("G major", True), ("E minor", False)]
    # no chord anywhere: none in the song, none on the maps, nothing said about them
    assert (
        seen["chords"] == []
        and seen["marks"] == []
        and (seen["written"], seen["to_choose"], seen["covered"]) == (0, 0, 0)
    )
    assert seen["places"] == [] and seen["moves"] == [] and seen["motions"] == [] and seen["roles"] is None
    assert set(seen["maps"]) == {"roads", "wheel", "semitones"} and tuple(seen["maps"]["roads"]["box"]) == (760, 534)
    assert all(
        spot["seconds"] == 0 and spot["name"] and spot["numeral"] for m in seen["maps"].values() for spot in m["spots"]
    )
    assert set(seen["regions"]) == set(copilot_harmony.JOBS)
    # the bars: when each starts and each of its four beats, and the note the bass stands on
    bars = seen["bars"]
    assert len(bars) >= 14 and bars[0]["bar"] == 1 and "pattern" in bars[0]
    assert [bar["stands_on"] for bar in bars[:6]] == [None, "E", "C", "G", "D", "E"]
    for bar in bars[1:13]:
        assert bar["beats"][0] == bar["time"] and bar["beats"] == sorted(bar["beats"]) and bar["beats"][3] < bar["end"]
        assert bar["holds"] == [] and bar["comes_in"] is None
    # the parts come from the bass line: a bar without bass, the loop of four bars three times round, the end
    parts = seen["parts"]
    assert parts[0]["bar"] == 1 and sum(p["bars"] for p in parts) == len(bars)
    assert all(a["bar"] + a["bars"] == b["bar"] and a["end"] <= b["start"] + 1e-6 for a, b in zip(parts, parts[1:]))
    loop = max(parts, key=lambda p: p["bars"])
    assert (loop["bar"], loop["bars"], loop["size"], loop["skip"]) == (2, 12, 4, 0)
    assert loop["stands"] == ["E", "C", "G", "D"] and loop["loop"] == [[], [], [], []]
    said = {entry["letter"]: entry for entry in seen["loops"]}
    mine = said[loop["letter"]]
    assert (mine["size"], mine["bars"], mine["rounds"], mine["times"]) == (4, 12, 3.0, 1) and len(mine["bass"]) == 4
    assert mine["stands"] == loop["stands"] and not mine["same"]  # the same shape, but on other steps of the key
    # what the bass plays, as steps of the key: over E in G major the line is 6 6 3 6 3 2
    assert [n["said"] for n in mine["bass"][0]["notes"]] == ["6", "6", "3", "6", "3", "2"]
    assert [n["said"] for n in mine["bass"][1]["notes"]] == ["4", "4", "1", "4", "1", "b7"]  # and over C
    assert all(line["of"] == "key" and line["share"] == 1.0 for line in mine["bass"])
    assert all(n["role"] is None and 0 <= n["slot"] <= 15 for line in mine["bass"] for n in line["notes"])
    notes = seen["bass"]
    assert len(notes) == len(desk.notes) and seen["strings"] == [28, 33, 38, 43]
    assert all(seen["strings"][n["string"]] + n["fret"] == n["pitch"] and 0 <= n["slot"] <= 15 for n in notes)
    assert all(n["step"] and n["chord"] is None and n["degree"] is None and n["role"] is None for n in notes)
    assert [s["step"] for s in seen["steps"]][:2] == ["1", "5"] and abs(
        sum(s["share"] for s in seen["steps"]) - 1
    ) < 0.01
    # every chord the person may put on a beat, with its notes
    assert len(seen["palette"]) == 12 * len(seen["qualities"]) == 144
    assert seen["palette"]["9,m7"]["tones"] == [9, 0, 4, 7] and seen["palette"]["9,m7"]["name"] == "Am7"
    for name, address in seen["audio"].items():
        assert os.path.isfile(os.path.join(workdir, "mix", "audio", name + ".flac")) and address.endswith(
            name + ".flac"
        )
    assert not os.path.exists(os.path.join(workdir, "mix", "harmony.json"))  # looking writes nothing


def test_a_chord_the_person_writes_is_the_songs_from_its_beat_on(workdir):
    desk = copilot.Desk(workdir, "mix")
    # the first round: Em, C, a change still to choose, and D, which then lasts to the end of the part
    copilot_harmony.set_marks(desk.folder, ROUND[:2] + [dict(at=48, chord=None), ROUND[3]])
    seen = copilot_harmony.view(desk)
    assert [(m["bar"], m["beat"], m["chord"] and m["chord"]["name"], m["open"]) for m in seen["marks"]] == [
        (2, 1, "Em", False),
        (3, 1, "C", False),
        (4, 1, None, True),
        (5, 1, "D", False),
    ]
    chords = seen["chords"]
    assert [(c["name"], c["bar"], c["at"], c["until"]) for c in chords] == [
        ("Em", 2, 16, 32),
        ("C", 3, 32, 48),
        ("D", 5, 64, 208),
    ]
    assert [(c["numeral"], c["job"]) for c in chords] == [("vi", "rest"), ("IV", "drift"), ("V", "pull")]
    assert all(c["start"] == seen["bars"][c["bar"] - 1]["time"] and c["tones"][0] == c["root"] for c in chords)
    assert chords[2]["end"] == seen["parts"][1]["end"]  # the end of the loop's part
    assert (seen["written"], seen["to_choose"]) == (3, 1) and 0.7 < seen["covered"] < 0.9
    # the bars say what they hold, and the chord that comes into each from before
    assert [[c["name"] for c in bar["holds"]] for bar in seen["bars"][:6]] == [[], ["Em"], ["C"], [], ["D"], ["D"]]
    assert [bar["comes_in"] and bar["comes_in"]["name"] for bar in seen["bars"][:7]] == [
        None,
        "Em",
        "C",
        None,
        "D",
        "D",
        "D",
    ]
    # the bass under a chord is named by the chord; where there is none, by the key
    line = lambda bar: [(n["degree"], n["role"]) for n in seen["bass"] if n["bar"] == bar]  # noqa: E731
    assert [d for d, _ in line(2)] == ["R", "R", "5", "8", "5", "4"] == [d for d, _ in line(3)]
    assert [r for _, r in line(2)] == ["root", "root", "fifth", "root", "fifth", "other"]
    assert set(line(4)) == {(None, None)}
    assert [n["step"] for n in seen["bass"] if n["bar"] == 4][:6] == ["1", "1", "5", "1", "5", "4"]
    assert all(seen["chords"][n["chord"]]["start"] - 0.05 <= n["start"] for n in seen["bass"] if n["chord"] is not None)
    assert seen["on_root"] == 1.0 and abs(sum(seen["roles"].values()) - 1) < 0.01  # each chord starts on its root
    # where the song spends its time, how it moves, and each chord's place on the three maps
    assert {p["name"] for p in seen["places"]} == {"Em", "C", "D"} and abs(
        sum(p["share"] for p in seen["places"]) - 1
    ) < 0.02
    assert [(m["a"]["name"], m["b"]["name"], m["times"]) for m in seen["moves"]] == [("Em", "C", 1), ("C", "D", 1)]
    assert [m["name"] for m in seen["motions"]] == [copilot_harmony.MOTIONS[m["up"]] for m in seen["motions"]]
    for name, drawn in seen["maps"].items():
        for c in chords:
            spot = drawn["spots"][c["on"][name]]
            assert spot["root"] == c["root"] and spot["seconds"] > 0
    assert desk.notes == copilot.Desk(workdir, "mix").notes  # and nothing of the notes


def test_a_round_is_copied_to_the_rounds_that_have_no_chords(workdir):
    desk = copilot.Desk(workdir, "mix")
    copilot_harmony.set_marks(desk.folder, ROUND)
    assert copilot_harmony.copy_round(desk, 16 * 3 + 8) == 2  # from any beat of the round
    seen = copilot_harmony.view(desk)
    assert [m["at"] for m in seen["marks"]] == [16 * bar for bar in range(1, 13)] and seen["written"] == 12
    assert [m["chord"]["name"] for m in seen["marks"]] == ["Em", "C", "G", "D"] * 3
    loop = max(seen["parts"], key=lambda p: p["bars"])
    assert [[c["name"] for c in bar] for bar in loop["loop"]] == [["Em"], ["C"], ["G"], ["D"]]
    mine = {entry["letter"]: entry for entry in seen["loops"]}[loop["letter"]]
    for line in mine["bass"][:2]:  # over a chord: root, root, fifth, octave, fifth, fourth
        assert [n["said"] for n in line["notes"]] == ["R", "R", "5", "8", "5", "4"] and line["share"] == 1.0
        assert [n["role"] for n in line["notes"]] == ["root", "root", "fifth", "root", "fifth", "other"]
    assert all(line["of"] == "chord" for line in mine["bass"])
    assert seen["roles"]["root"] > 0.5 and seen["roles"]["root"] + seen["roles"]["fifth"] > 0.8
    # a round the person wrote is never written over: one chord changed in the second, and nothing is copied
    copilot_harmony.set_marks(desk.folder, [dict(at=96, chord=[A, "m"])])
    before = copilot_harmony.read_marks(desk.folder)
    assert copilot_harmony.copy_round(desk, 16) == 0 and copilot_harmony.read_marks(desk.folder) == before
    with pytest.raises(Bass2SheetError, match="no chords to copy"):
        copilot_harmony.copy_round(desk, 16 * 13)  # the bars the song dies away in
    with pytest.raises(Bass2SheetError, match="no part of the song"):
        copilot_harmony.copy_round(desk, 16 * 400)


def test_a_round_come_into_part_of_the_way_gets_the_chords_from_there_on(tmp_path):
    # four bars three times, two other bars, and the loop again from its third bar: C D, twice round, A B
    pitch = dict(A=45, B=47, C=48, D=50, x=41, y=43)
    labels = "ABCDABCDABCDxyCDABCDABCDAB"
    notes = [_note(2.0 * k, 2.0 * k + 1.9, pitch[label]) for k, label in enumerate(labels)]
    machine = _machine(bars=len(labels), key=["A minor", A, False])
    desk = types.SimpleNamespace(machine=machine, notes=notes, folder=str(tmp_path))
    put, read = copilot_harmony.set_marks, copilot_harmony.read_marks
    put(desk.folder, [dict(at=0, chord=[A, "m"]), dict(at=48, chord=[G, ""])])  # Am for three bars, G in the fourth
    assert copilot_harmony.copy_round(desk, 0) == 6
    assert [(m["at"] // 16, m["chord"]) for m in read(desk.folder)] == [
        (0, (A, "m")),
        (3, (G, "")),
        (4, (A, "m")),
        (7, (G, "")),
        (8, (A, "m")),
        (11, (G, "")),
        (14, (A, "m")),  # the round that starts on its third bar starts in the chord that sounds there
        (15, (G, "")),
        (16, (A, "m")),
        (19, (G, "")),
        (20, (A, "m")),
        (23, (G, "")),
        (24, (A, "m")),  # and the last, cut after two bars, has what fits
    ]
    # a chord that comes into the round from before it is copied as the round's first
    for m in read(desk.folder):
        put(desk.folder, [dict(at=m["at"], chord="remove")])
    put(desk.folder, [dict(at=0, chord=[A, "m"]), dict(at=80, chord=[G, ""])])  # Am from bar 1 on into the second round
    assert copilot_harmony.copy_round(desk, 16 * 5) == 5  # the first round has a mark of its own: it is left
    assert [(m["at"] // 16, m["chord"]) for m in read(desk.folder)][2:5] == [(8, (A, "m")), (9, (G, "")), (14, (G, ""))]


def test_the_tool_offers_chords_for_a_beat(workdir):
    desk = copilot.Desk(workdir, "mix")
    got = copilot_harmony.offers(desk, 16)  # bar 2, beat 1: the bass is on E
    names = [o["name"] for o in got["offers"]]
    assert got["at"] == 16 and names[0] == "Em" and len(names) == 6 == len(set(names))
    first = got["offers"][0]
    assert (first["root"], first["quality"], first["numeral"], first["job"], first["yours"]) == (
        E,
        "m",
        "vi",
        "rest",
        False,
    )
    assert all(o["quality"] in ("", "m") and not o["yours"] for o in got["offers"])
    # the same place in the other rounds, where the chord may be wanted too
    assert got["rounds"] == [dict(at=80, bar=6, chord=None), dict(at=144, bar=10, chord=None)]
    assert [copilot_harmony.offers(desk, 16 * bar)["offers"][0]["name"] for bar in (2, 3, 4)] == ["C", "G", "D"]
    assert not os.path.exists(os.path.join(desk.folder, "harmony.json"))  # offering writes nothing
    # what the person put at the same place in another round is offered first, and marked as theirs
    copilot_harmony.set_marks(desk.folder, [dict(at=80, chord=[E, "7"])])
    got = copilot_harmony.offers(desk, 16)
    assert (got["offers"][0]["name"], got["offers"][0]["yours"]) == ("E", True) and got["rounds"][0]["chord"][
        "name"
    ] == "E7"
    assert sum(o["yours"] for o in got["offers"]) == 1
    # the offers are for the stretch to the next mark: on beat 3 of a bar, for its second half
    assert copilot_harmony.offers(desk, 16 + 8)["offers"][0]["name"] in ("Em", "E")
    for bad in (6, -4, 16 * 400, "16", None):
        with pytest.raises(Bass2SheetError, match="beat of a bar"):
            copilot_harmony.offers(desk, bad)


def test_the_score_carries_the_persons_chords_and_no_others(workdir):
    desk = copilot.Desk(workdir, "mix")
    folder, machine = desk.folder, desk.machine
    assert copilot.sheet_chords(folder, machine, desk.notes) == {}
    assert copilot.sheet_key(folder, machine) == ("G major", G, True)
    with open(copilot.sheet(workdir, "mix", mscore=False)["musicxml"]) as f:
        assert "<harmony>" not in f.read()
    # Em; C and, on beat 3, A minor seventh; no chord in bar 4; G from bar 5 to the end of the part
    marks = [ROUND[0], ROUND[1], dict(at=40, chord=[A, "m7"]), dict(at=48, chord="none"), dict(at=64, chord=[G, ""])]
    copilot_harmony.set_marks(folder, marks)
    got = copilot.sheet_chords(folder, machine, desk.notes)
    assert sorted(got) == [2, 3] + list(range(5, 14))
    assert [(at, c["root"], c["quality"], c["bass"]) for at, c in got[3]] == [(0, C, "", None), (8, A, "m7", None)]
    assert [(at, c["root"]) for at, c in got[9]] == [(0, G)] and len(got[2][0][1]["names"]) == 12
    with open(copilot.sheet(workdir, "mix", mscore=False)["musicxml"]) as f:
        xml = f.read()
    assert xml.count("<harmony>") == 4 and "<kind>minor-seventh</kind>" in xml  # a chord is named where it changes
    assert "<offset>8</offset>" in xml and "Chord symbols approximate" not in xml
    # the key the score is written in is the home the person chose
    copilot_harmony.set_home(folder, (E, False))
    assert copilot.sheet_key(folder, machine) == ("E minor", E, False)


def test_the_song_is_put_on_the_neck_for_playing_unless_the_person_chooses_another_way(workdir):
    from bass2sheet import score

    desk = copilot.Desk(workdir, "mix")
    copilot_harmony.set_marks(desk.folder, ROUND)
    seen = copilot_harmony.view(desk)
    assert seen["tab"] == "easy" == copilot.tab_style(desk.folder)
    # what each way asks of the hand over the whole song, for the page to say
    assert [t["id"] for t in seen["tabs"]] == list(score.TAB_STYLES) and all(
        t["notes"] == len(seen["bass"]) for t in seen["tabs"]
    )
    ways = {t["id"]: t for t in seen["tabs"]}
    assert ways["closed"]["open"] < ways["low"]["open"] and ways["easy"]["moves"] <= ways["position"]["moves"]
    easy = [(n["string"], n["fret"]) for n in seen["bass"]]
    assert all(seen["strings"][s] + f == n["pitch"] for (s, f), n in zip(easy, seen["bass"]))
    assert [n["hand"] for n in seen["bass"]] == score.hand(easy)  # where the hand is for each note

    copilot_harmony.set_tab(desk.folder, "low")
    low = copilot_harmony.view(desk)
    assert low["tab"] == "low" == copilot.tab_style(desk.folder)
    assert [(n["string"], n["fret"]) for n in low["bass"]] != easy and any(n["fret"] == 0 for n in low["bass"])
    assert low["home"] == seen["home"] and low["chords"] == seen["chords"]  # nothing else changes
    copilot_harmony.set_tab(desk.folder, None)
    assert copilot_harmony.view(desk)["tab"] == "easy"
    with pytest.raises(Bass2SheetError, match="easy, closed, low, position"):
        copilot_harmony.set_tab(desk.folder, "high")


def _long_ago(workdir):
    """The prepared song as one prepared before the tool heard chords eighth by eighth: no harmony, a key and a
    chord a bar of its own. Returns (the machine data as written, the harmony it had)."""
    path = os.path.join(workdir, "mix", "machine.json")
    machine = copilot.read_json(path)
    fresh = machine.pop("harmony")
    names = machine["chords"][0]["names"]
    machine.update(
        key=["C major", 0, True], chords=[dict(bar=2, root=C, quality="", score=0.5, names=names, chord="C")]
    )
    with open(path, "w") as f:
        json.dump(machine, f)
    copilot_harmony._ASKED.discard(os.path.join(workdir, "mix"))
    return machine, fresh


def test_a_song_prepared_before_has_its_harmony_heard_when_its_page_is_opened(workdir):
    folder = os.path.join(workdir, "mix")
    path = os.path.join(folder, "machine.json")
    machine, fresh = _long_ago(workdir)
    with open(os.path.join(folder, "notes.json"), "rb") as f:
        mine = f.read()
    desk = copilot.Desk(workdir, "mix")
    seen = copilot_harmony.view(desk)
    now = copilot.read_json(path)
    # the same chords as on preparing the song anew (the notes are kept rounded, so a score may differ a hair)
    place = lambda found: [(c["at"], c["until"], c["root"], c["quality"], c["bass"]) for c in found["chords"]]  # noqa: E731
    assert place(now["harmony"]) == place(fresh) and desk.machine == now
    assert [c["score"] for c in now["harmony"]["chords"]] == pytest.approx(
        [c["score"] for c in fresh["chords"]], abs=0.03
    )
    # it is added: everything the song had is as it was, the key and the chord a bar too, and the person's work
    assert {k: v for k, v in now.items() if k != "harmony"} == machine
    with open(os.path.join(folder, "notes.json"), "rb") as f:
        assert f.read() == mine
    # it is used for the tool's guess of the key and for what it offers; it puts no chord in the song
    assert seen["home"]["name"] == "G major" and seen["chords"] == [] and seen["written"] == 0
    assert [h["name"] for h in seen["homes"]] == ["G major", "E minor", "C major"]  # the key it had is there to pick
    assert copilot_harmony.offers(desk, 16)["offers"][0]["name"] == "Em"
    assert copilot.sheet_chords(folder, now, desk.notes) == {}


def test_a_song_whose_separated_tracks_are_gone_keeps_the_key_it_had(workdir):
    folder = os.path.join(workdir, "mix")
    path = os.path.join(folder, "machine.json")
    machine, _ = _long_ago(workdir)
    songs = copilot.load_songs(workdir)
    songs["mix"]["stems_dir"] = os.path.join(workdir, "gone")
    with open(os.path.join(workdir, "songs.json"), "w") as f:
        json.dump(songs, f)
    assert copilot.hear_harmony(workdir, "mix") is None
    desk = copilot.Desk(workdir, "mix")
    seen = copilot_harmony.view(desk)
    assert copilot.read_json(path) == machine and "harmony" not in desk.machine  # nothing could be heard
    assert folder in copilot_harmony._ASKED  # and it is not asked for again at every look
    assert [h["name"] for h in seen["homes"]] == ["C major", "A minor"] and seen["home"]["name"] == "C major"
    assert seen["chords"] == [] and [bar["stands_on"] for bar in seen["bars"][1:5]] == ["E", "C", "G", "D"]
    # the chord a bar it had still weighs in what is offered: C for bar 2, where the bass is on E
    assert "C" in [o["name"] for o in copilot_harmony.offers(desk, 16)["offers"]][:2]
    copilot_harmony._ASKED.discard(folder)


def test_the_person_chooses_home_and_the_names_follow(workdir):
    desk = copilot.Desk(workdir, "mix")
    copilot_harmony.set_marks(desk.folder, ROUND)
    before = copilot_harmony.view(desk)
    other = copilot_harmony.relative((before["home"]["root"], before["home"]["major"]))
    copilot_harmony.set_home(desk.folder, other)
    after = copilot_harmony.view(desk)
    assert (after["home"]["root"], after["home"]["major"]) == other and after["home"]["yours"]
    assert [c["numeral"] for c in before["chords"]] == ["vi", "IV", "I", "V"]
    assert [c["numeral"] for c in after["chords"]] == ["i", "bVI", "bIII", "bVII"]
    assert [c["root"] for c in after["chords"]] == [c["root"] for c in before["chords"]]  # the chords are the same
    assert [n["step"] for n in after["bass"]][:6] == ["1", "1", "5", "1", "5", "4"]  # and the steps are of E now
    assert len(after["homes"]) == 2 and desk.notes == copilot.Desk(workdir, "mix").notes  # nothing of the notes
    # a home that is neither the tool's guess nor its relative is offered too, so that it can be seen as chosen
    copilot_harmony.set_home(desk.folder, (1, True))
    assert [h["name"] for h in copilot_harmony.view(desk)["homes"]][-1] == "Db major"
    copilot_harmony.set_home(desk.folder, None)  # back to the tool's guess
    assert copilot_harmony.view(desk)["home"] == before["home"]
    for bad in ((12, True), (0, "major"), ("C", True)):
        with pytest.raises(Bass2SheetError):
            copilot_harmony.set_home(desk.folder, bad)


@pytest.fixture
def served(workdir):
    srv = copilot_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.RequestHandlerClass.adding.close()
    srv.shutdown()
    srv.server_close()


def _json(url, data=None):
    req = urllib.request.Request(
        url, None if data is None else json.dumps(data).encode(), method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_page_is_served_and_the_other_pages_lead_to_it(served):
    with urllib.request.urlopen(served + "/harmony", timeout=30) as r:
        page = r.read().decode()
    assert r.status == 200 and '<script id="logic">' in page and "/api/harmony" in page and "/api/offers" in page
    for address, link in (("/", "/harmony#${id}"), ("/song", 'id="harmonylink"')):
        with urllib.request.urlopen(served + address, timeout=30) as r:
            assert link in r.read().decode()


def test_the_pages_own_logic():
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node is not installed")
    script = os.path.join(os.path.dirname(__file__), "copilot_harmony_logic.js")
    out = subprocess.run(["node", script, copilot_harmony.PAGE_PATH], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_the_server_gives_the_harmony_and_keeps_what_the_person_sets(served, workdir):
    folder = os.path.join(workdir, "mix")
    status, seen = _json(served + "/api/harmony?id=mix")
    assert status == 200 and seen["id"] == "mix" and seen["chords"] == [] and not seen["home"]["yours"]
    assert _json(served + "/api/harmony?id=nothing")[0] == 404
    other = seen["homes"][1]
    status, after = _json(served + "/api/harmony", dict(song="mix", home=[other["root"], other["major"]]))
    assert status == 200 and after["home"]["name"] == other["name"] and after["home"]["yours"]
    assert copilot_harmony.read_home(folder) == (other["root"], other["major"])
    assert _json(served + "/api/harmony?id=mix")[1]["home"] == after["home"]  # kept
    status, said = _json(served + "/api/harmony", dict(song="mix", home=[12, True]))
    assert status == 400 and "twelve notes" in said["error"]
    # what the tool offers for a beat
    status, got = _json(served + "/api/offers?id=mix&at=16")
    assert status == 200 and got["offers"][0]["name"] == "Em" and [r["bar"] for r in got["rounds"]] == [6, 10]
    for bad in ("id=mix&at=6", "id=mix", "id=mix&at=two", "id=nothing&at=16"):  # no such beat, no such song
        assert _json(served + "/api/offers?" + bad)[0] == 404
    # the person's chords: kept, logged, and the home is left as it is
    status, marked = _json(served + "/api/harmony", dict(song="mix", marks=ROUND))
    assert status == 200 and marked["written"] == 4 and marked["home"]["yours"]
    assert [m["chord"]["name"] for m in marked["marks"]] == ["Em", "C", "G", "D"]
    assert [m["chord"] for m in copilot_harmony.read_marks(folder)] == [(E, "m"), (C, ""), (G, ""), (D, "")]
    status, said = _json(served + "/api/harmony", dict(song="mix", marks=[dict(at=18, chord=[A, "m"])]))
    assert status == 400 and "on a beat" in said["error"]
    assert _json(served + "/api/harmony", dict(song="mix", marks=[]))[0] == 400
    status, fewer = _json(served + "/api/harmony", dict(song="mix", marks=[dict(at=64, chord="remove")]))
    assert status == 200 and fewer["written"] == 3
    # the round copied to the rounds that have none
    status, copied = _json(served + "/api/harmony", dict(song="mix", copy=16))
    assert status == 200 and copied["written"] == 9 and copied["home"]["yours"]
    assert _json(served + "/api/harmony", dict(song="mix", copy=16 * 400))[0] == 400
    with open(os.path.join(folder, "log.jsonl")) as f:
        logged = [json.loads(line) for line in f if '"chord' in line]
    assert [entry["kind"] for entry in logged] == ["chord", "chord", "chords copied"]
    assert logged[0]["marks"] == ROUND and (logged[2]["at"], logged[2]["rounds"]) == (16, 2)
    status, back = _json(served + "/api/harmony", dict(song="mix", home=None))
    assert status == 200 and back["home"] == seen["home"] and back["written"] == 9
    # how the song is put on the neck: kept, and the chords and the home are left as they are
    status, closed = _json(served + "/api/harmony", dict(song="mix", tab="closed"))
    assert status == 200 and closed["tab"] == "closed" and closed["written"] == 9
    assert copilot.tab_style(folder) == "closed"
    assert _json(served + "/api/harmony", dict(song="mix", tab="high"))[0] == 400
    assert _json(served + "/api/harmony?id=mix")[1]["tab"] == "closed"
