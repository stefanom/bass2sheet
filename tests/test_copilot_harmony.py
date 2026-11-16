"""The song's harmony for the copilot: the parts its bass line goes round, a chord for every bar (the tool's best
guess, or the person's where they chose on the song's page), the chords the tool offers in place of one, and what
the bass plays. The harmony page shows it and changes nothing."""

import json
import os
import shutil
import threading
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


def _heard(*chords):
    """What the finder leaves in a song's data, for chords given as (first sixteenth, one past the last, root,
    quality), counted from the start of bar 1, in C major."""
    found = [
        dict(at=4 + a, until=4 + b, start=a / 8, end=b / 8, root=r, quality=q, bass=None, score=0.9)
        for a, b, r, q in chords
    ]
    return dict(
        chords=found,
        keys=[
            dict(
                at=0,
                until=10**6,
                start=0.0,
                end=1e6,
                root=C,
                major=True,
                name="C major",
                other=dict(root=A, major=False),
            )
        ],
    )


def test_the_tools_best_guess_is_a_triad_for_each_bar_and_one_more_where_it_changes_on_beat_three():
    guessed = copilot_harmony.guessed
    heard = _heard(
        (0, 16, C, "maj7"), (16, 24, A, "m"), (24, 32, F, ""), (32, 44, G, ""), (44, 48, G, "7"), (56, 64, D, "m7")
    )
    assert guessed(_machine(harmony=heard)) == {
        0: [(0, C, "")],  # a seventh is offered as its triad
        1: [(0, A, "m"), (2, F, "")],  # the second half of the bar has another chord: it starts on beat 3
        2: [(0, G, "")],  # the seventh after its own triad is the same chord
        3: [(2, D, "m")],  # nothing is heard in the first half
    }
    # a song prepared long ago has the chord a bar it was given, where one was heard
    old = [
        dict(bar=1, root=E, quality="m", score=0.9),
        dict(bar=2, root=C, quality="", score=0.0),
        dict(bar=3, root=G, quality=""),
    ]
    assert guessed(_machine(chords=old)) == {0: [(0, E, "m")], 2: [(0, G, "")]}
    assert guessed(_machine()) == {} and guessed(_machine(chords=None, harmony=dict(chords=[], keys=[]))) == {}


def test_the_persons_chords_are_kept_by_bar(tmp_path):
    folder = str(tmp_path)
    read, put = copilot_harmony.read_bars, copilot_harmony.set_bars
    assert read(folder) == {}
    put(
        folder,
        [dict(bar=2, chords=[[0, A, "m"], [2, E, "7"]]), dict(bar=0, chords=[[0, E, "m7"]]), dict(bar=5, chords=[])],
    )
    first = {0: [(0, E, "m7")], 2: [(0, A, "m"), (2, E, "7")], 5: []}  # a chord, two in a bar, a bar with no chord
    assert read(folder) == first
    put(folder, [dict(bar=2, chords=[[0, G, ""]]), dict(bar=5, chords=None), dict(bar=40, chords=None)])
    assert read(folder) == {
        0: [(0, E, "m7")],
        2: [(0, G, "")],
    }  # a bar is changed whole; None gives it back to the tool
    # they are kept beside the home and the fingering the person chose
    copilot_harmony.set_home(folder, (G, True))
    copilot_harmony.set_tab(folder, "low")
    put(folder, [dict(bar=0, chords=None)])
    assert copilot_harmony.read_home(folder) == (G, True) and copilot.tab_style(folder) == "low"
    kept = {2: [(0, G, "")]}
    assert read(folder) == kept
    for bad in (
        [],
        None,
        ["bar 3"],
        [dict(bar=-1, chords=[])],
        [dict(bar=True, chords=[])],
        [dict(bar=1.0, chords=[])],
        [dict(bar=1, chords="Am")],
        [dict(bar=1, chords=[[0, A]])],
        [dict(bar=1, chords=[[4, A, "m"]])],  # a bar has four beats
        [dict(bar=1, chords=[[0, 12, ""]])],
        [dict(bar=1, chords=[[0, A, "minor"]])],
        [dict(bar=1, chords=[[2, A, "m"], [0, C, ""]])],  # out of order
        [dict(bar=1, chords=[[0, A, "m"], [0, C, ""]])],  # two on one beat
        [dict(bar=3, chords=[[0, C, ""]]), dict(bar=4, chords=[[0, C, "major"]])],  # one bad change: nothing is kept
    ):
        with pytest.raises(Bass2SheetError):
            put(folder, bad)
    assert read(folder) == kept


def test_chords_written_as_marks_on_beats_are_read_as_what_each_bar_holds(tmp_path):
    folder = str(tmp_path)
    marks = [
        dict(at=0, chord=[B, "m"]),  # for two bars
        dict(at=32, chord=[A, ""]),  # for a bar and a half
        dict(at=56, chord=[E, "7"]),  # from beat 3 of the fourth bar
        dict(at=64, chord=None),  # a change that was marked and never given a chord
        dict(at=96, chord="none"),
        dict(at=120, chord=[G, ""]),  # the last: to the end of its bar
    ]
    with open(os.path.join(folder, "harmony.json"), "w") as f:
        json.dump(dict(marks=marks, home=[B, False]), f)
    read = {0: [(0, B, "m")], 1: [(0, B, "m")], 2: [(0, A, "")], 3: [(0, A, ""), (2, E, "7")], 7: [(2, G, "")]}
    assert copilot_harmony.read_bars(folder) == read
    # the first change writes them by bar; the marks are kept as they were, and so is the rest
    copilot_harmony.set_bars(folder, [dict(bar=1, chords=[[0, D, ""]])])
    saved = copilot.read_json(os.path.join(folder, "harmony.json"))
    assert saved["marks_before"] == marks and "marks" not in saved and saved["home"] == [B, False]
    assert copilot_harmony.read_bars(folder) == {**read, 1: [(0, D, "")]}


def test_each_bar_holds_the_persons_chords_or_the_tools_guess(tmp_path):
    heard = _heard((0, 16, C, ""), (16, 32, A, "m"), (48, 64, G, "7"))
    folder, machine = str(tmp_path), _machine(harmony=heard, key=["C major", C, True], notes=[])
    guess = [dict(chords=[(0, C, "")], yours=False), dict(chords=[(0, A, "m")], yours=False)]
    guess += [dict(chords=[], yours=False), dict(chords=[(0, G, "")], yours=False)]
    assert copilot_harmony.bar_chords(folder, machine)[:4] == guess and copilot_harmony.chosen_share(
        folder, machine
    ) == (0, 3)
    copilot_harmony.set_bars(folder, [dict(bar=1, chords=[[0, F, "maj7"]]), dict(bar=3, chords=[])])
    now = copilot_harmony.bar_chords(folder, machine)
    assert (
        now[0] == guess[0]
        and now[1] == dict(chords=[(0, F, "maj7")], yours=True)
        and now[3] == dict(chords=[], yours=True)
    )
    assert copilot_harmony.chosen_share(folder, machine) == (2, 2)
    # the score's chords, as they stand: each named by its bar from 1 and the sixteenth it starts on
    sheet = copilot_harmony.for_sheet(folder, machine)
    assert sorted(sheet) == [1, 2] and [(at, c["root"], c["quality"], c["bass"]) for at, c in sheet[2]] == [
        (0, F, "maj7", None)
    ]
    assert len(sheet[1][0][1]["names"]) == 12
    assert copilot.chord_note(folder, machine) == ""  # every bar that has a chord has the person's
    copilot_harmony.set_bars(folder, [dict(bar=3, chords=None)])
    assert (
        copilot.chord_note(folder, machine)
        == "Chord symbols: the tool's guess, except in the 1 bars you chose them for"
    )
    copilot_harmony.set_bars(folder, [dict(bar=1, chords=None)])
    assert copilot.chord_note(folder, machine) == "Chord symbols: the tool's guess"


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
    # and what the person put in other bars like it comes first
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
LOOP = ["Em", "C", "G", "D"]


def test_the_harmony_page_gets_the_song_with_the_tools_guess_at_its_chords(workdir):
    desk = copilot.Desk(workdir, "mix")
    seen = copilot_harmony.view(desk)
    # home is the tool's guess, with the key that has the same notes beside it
    assert seen["home"] == dict(root=G, major=True, name="G major", yours=False)
    assert [(h["name"], h["guess"]) for h in seen["homes"]] == [("G major", True), ("E minor", False)]
    # the chords are the tool's best guess, a bar each: none is the person's
    chords = seen["chords"]
    assert [c["name"] for c in chords] == LOOP * 3 and not any(c["yours"] for c in chords)
    assert [(c["bar"], c["beat"]) for c in chords] == [(bar, 1) for bar in range(2, 14)]
    assert [(c["numeral"], c["job"]) for c in chords[:4]] == [
        ("vi", "rest"),
        ("IV", "drift"),
        ("I", "rest"),
        ("V", "pull"),
    ]
    assert all(
        c["start"] == seen["bars"][c["bar"] - 1]["time"] and c["end"] == seen["bars"][c["bar"] - 1]["end"]
        for c in chords
    )
    assert all(
        c["tones"][0] == c["root"] and len(c["tones"]) == 3 and c["far"] >= 0 and c["role"] == "root" for c in chords
    )
    assert (seen["yours"], seen["guessed"]) == (0, 12) and 0.85 < seen["covered"] < 0.95
    # the bars: when each starts and each of its four beats, the note the bass stands on, and what it holds
    bars = seen["bars"]
    assert len(bars) >= 14 and bars[0]["bar"] == 1 and "pattern" in bars[0]
    assert [bar["stands_on"] for bar in bars[:6]] == [None, "E", "C", "G", "D", "E"]
    assert (
        bars[0]["chords"] == []
        and [c["name"] for c in bars[1]["chords"]] == ["Em"]
        and bars[1]["chords"][0]["beat"] == 1
    )
    for bar in bars[1:13]:
        assert bar["beats"][0] == bar["time"] and bar["beats"] == sorted(bar["beats"]) and bar["beats"][3] < bar["end"]
        assert not bar["yours"] and [c["name"] for c in bar["holds"]] == [c["name"] for c in bar["chords"]]
    # the parts come from the bass line: a bar without bass, the loop of four bars three times round, the end
    parts = seen["parts"]
    assert parts[0]["bar"] == 1 and sum(p["bars"] for p in parts) == len(bars)
    assert all(a["bar"] + a["bars"] == b["bar"] and a["end"] <= b["start"] + 1e-6 for a, b in zip(parts, parts[1:]))
    loop = max(parts, key=lambda p: p["bars"])
    assert (loop["bar"], loop["bars"], loop["size"], loop["skip"]) == (2, 12, 4, 0)
    assert loop["stands"] == ["E", "C", "G", "D"] and [[c["name"] for c in bar] for bar in loop["loop"]] == [
        [n] for n in LOOP
    ]
    mine = {entry["letter"]: entry for entry in seen["loops"]}[loop["letter"]]
    assert (mine["size"], mine["bars"], mine["rounds"], mine["times"]) == (4, 12, 3.0, 1) and len(mine["bass"]) == 4
    # a guess is not agreed to: the bass is named by the key, not by the chord: over E in G major 6 6 3 6 3 2
    assert [n["said"] for n in mine["bass"][0]["notes"]] == ["6", "6", "3", "6", "3", "2"]
    assert [n["said"] for n in mine["bass"][1]["notes"]] == ["4", "4", "1", "4", "1", "b7"]
    assert all(line["of"] == "key" and line["share"] == 1.0 for line in mine["bass"])
    notes = seen["bass"]
    assert len(notes) == len(desk.notes) and seen["strings"] == [28, 33, 38, 43]
    assert all(seen["strings"][n["string"]] + n["fret"] == n["pitch"] and 0 <= n["slot"] <= 15 for n in notes)
    assert all(n["step"] and n["degree"] is None and n["role"] is None for n in notes)
    assert seen["roles"] is None and seen["on_root"] == 0.0  # said only of chords the person chose
    assert [s["step"] for s in seen["steps"]][:2] == ["1", "5"] and abs(
        sum(s["share"] for s in seen["steps"]) - 1
    ) < 0.01
    # where the song spends its time, how it moves, and each chord's place on the three maps
    assert {p["name"] for p in seen["places"]} == set(LOOP) and abs(sum(p["share"] for p in seen["places"]) - 1) < 0.02
    assert [(m["a"]["name"], m["b"]["name"], m["times"]) for m in seen["moves"]][:3] == [
        ("Em", "C", 3),
        ("C", "G", 3),
        ("G", "D", 3),
    ]
    assert seen["motions"] and all(m["name"] == copilot_harmony.MOTIONS[m["up"]] for m in seen["motions"])
    assert set(seen["maps"]) == {"roads", "wheel", "semitones"} and tuple(seen["maps"]["roads"]["box"]) == (760, 534)
    for name, drawn in seen["maps"].items():
        for c in chords:
            spot = drawn["spots"][c["on"][name]]
            assert spot["root"] == c["root"] and spot["seconds"] > 0 and spot["name"] and spot["numeral"]
    assert set(seen["regions"]) == set(copilot_harmony.JOBS)
    for name, address in seen["audio"].items():
        assert os.path.isfile(os.path.join(workdir, "mix", "audio", name + ".flac")) and address.endswith(
            name + ".flac"
        )
    assert not os.path.exists(os.path.join(workdir, "mix", "harmony.json"))  # looking writes nothing


def test_the_chords_the_person_chose_are_the_songs_and_the_bass_is_named_by_them(workdir):
    desk = copilot.Desk(workdir, "mix")
    # bar 2: E minor seventh; bar 3: C and, from beat 3, A minor; bar 4: no chord
    copilot_harmony.set_bars(
        desk.folder,
        [dict(bar=1, chords=[[0, E, "m7"]]), dict(bar=2, chords=[[0, C, ""], [2, A, "m"]]), dict(bar=3, chords=[])],
    )
    seen = copilot_harmony.view(desk)
    chords = seen["chords"]
    assert [(c["name"], c["bar"], c["beat"], c["yours"]) for c in chords[:5]] == [
        ("Em7", 2, 1, True),
        ("C", 3, 1, True),
        ("Am", 3, 3, True),
        ("D", 5, 1, False),  # bar 4 has none; from bar 5 on the tool's guess stands
        ("Em", 6, 1, False),
    ]
    assert (
        chords[1]["end"] == chords[2]["start"] == seen["bars"][2]["beats"][2]
        and chords[2]["end"] == seen["bars"][2]["end"]
    )
    assert (seen["yours"], seen["guessed"]) == (3, 9)
    bars = seen["bars"]
    assert [bar["yours"] for bar in bars[:6]] == [False, True, True, True, False, False]
    assert [(c["name"], c["beat"]) for c in bars[2]["chords"]] == [("C", 1), ("Am", 3)] and bars[3]["chords"] == []
    assert [c["name"] for c in bars[2]["holds"]] == ["C", "Am"] and bars[3]["holds"] == []
    # under a chord the person chose, a bass note is named by the chord; elsewhere by the key
    said = lambda bar: [n["degree"] or n["step"] for n in seen["bass"] if n["bar"] == bar]  # noqa: E731
    assert said(2) == ["R", "R", "5", "8", "5", "4"] and said(3) == ["R", "R", "5", "b3", "b7↓", "b6↓"]
    assert said(4)[:6] == ["1", "1", "5", "1", "5", "4"] and said(6) == ["6", "6", "3", "6", "3", "2"]
    assert [n["role"] for n in seen["bass"] if n["bar"] == 2] == ["root", "root", "fifth", "root", "fifth", "other"]
    assert abs(sum(seen["roles"].values()) - 1) < 0.01 and seen["roles"]["root"] > 0.4 and 0.6 < seen["on_root"] < 0.7
    # everything that is said of the song follows
    assert {"Em7", "Am"} <= {p["name"] for p in seen["places"]}
    assert any(s["name"] == "Am" and s["seconds"] > 0 for s in seen["maps"]["roads"]["spots"])
    assert desk.notes == copilot.Desk(workdir, "mix").notes  # and nothing of the notes


def test_the_tool_offers_chords_in_place_of_the_one_a_bar_has(workdir):
    desk = copilot.Desk(workdir, "mix")
    got = copilot_harmony.offers(desk, 1)  # bar 2: the bass is on E, and the tool's guess is E minor
    assert (got["bar"], got["beat"]) == (1, 0) and got["chord"]["name"] == "Em" and not got["chord"]["yours"]
    names = [o["name"] for o in got["offers"]]
    assert (
        len(names) == 6 == len(set(names)) and "Em" not in names and {"E", "C", "G"} <= set(names)
    )  # not what is there
    first = got["offers"][0]
    assert (first["name"], first["numeral"], first["job"], first["tones"]) == ("E", "VI", "pull", [E, 8, B])
    assert (got["start"], got["end"]) == tuple(round(desk.machine["grid"][g], 3) for g in (16, 32))
    # the bars like it: where the bass stands on the same note and the same chord is written
    assert got["alike"] == dict(part=[5, 9], song=[5, 9])
    assert not os.path.exists(os.path.join(desk.folder, "harmony.json"))  # asking writes nothing
    # a bar changed is no longer like the others; and one of two chords in a bar is offered for its own stretch
    copilot_harmony.set_bars(
        desk.folder, [dict(bar=5, chords=[[0, E, ""]]), dict(bar=2, chords=[[0, C, ""], [2, A, "m"]])]
    )
    assert copilot_harmony.offers(desk, 1)["alike"] == dict(part=[9], song=[9])
    assert copilot_harmony.offers(desk, 5)["chord"] == dict(got["offers"][0], yours=True)
    half = copilot_harmony.offers(desk, 2, 2)
    assert half["chord"]["name"] == "Am"
    assert (half["start"], half["end"]) == tuple(round(desk.machine["grid"][g], 3) for g in (40, 48))
    assert "Am" not in [o["name"] for o in half["offers"]] and half["alike"] == dict(part=[], song=[])
    # a bar without a chord is asked about from its first beat; the bars like it are the other bars without one
    none = copilot_harmony.offers(desk, 0)
    assert none["chord"] is None and len(none["offers"]) == 6 and none["alike"]["song"] == [13, 14]
    for bar, beat in (
        (-1, 0),
        (400, 0),
        (1, 4),
        (1, 2),
        (0, 2),
        ("1", 0),
        (1, None),
    ):  # no such bar, beat, or chord there
        with pytest.raises(Bass2SheetError):
            copilot_harmony.offers(desk, bar, beat)


def test_the_songs_page_gets_the_chords_the_key_the_fingering_and_the_neck(workdir):
    from bass2sheet import score

    desk = copilot.Desk(workdir, "mix")
    got = copilot_harmony.for_notes(desk)
    assert got["home"] == dict(root=G, major=True, name="G major", yours=False)
    assert [(h["name"], h["guess"]) for h in got["homes"]] == [("G major", True), ("E minor", False)]
    assert (
        got["spelled"][6] == "F#" and len(got["spelled"]) == 12 and got["qualities"][:5] == ["", "m", "7", "maj7", "m7"]
    )
    assert len(got["qualities"]) == 12 and got["yours"] == []
    # the chords as stretches, bars and beats counted from 0, each with its notes for the keyboard sound
    assert [c["name"] for c in got["chords"]] == LOOP * 3 and [c["bar"] for c in got["chords"]] == list(range(1, 13))
    first = got["chords"][0]
    assert (first["beat"], first["yours"], first["tones"], first["numeral"]) == (0, False, [E, G, B], "vi")
    assert (first["start"], first["end"]) == (round(desk.machine["grid"][16], 3), round(desk.machine["grid"][32], 3))
    # how the notes are put on the neck, what each way asks of the hand, and where each note is played
    assert (
        got["tab"] == "easy"
        and [t["id"] for t in got["tabs"]] == list(score.TAB_STYLES)
        and got["strings"] == [28, 33, 38, 43]
    )
    assert all(t["notes"] == len(desk.notes) for t in got["tabs"])
    ways = {t["id"]: t for t in got["tabs"]}
    assert ways["closed"]["open"] < ways["low"]["open"] and ways["easy"]["moves"] <= ways["position"]["moves"]
    neck = got["neck"]
    assert set(neck) == {n["id"] for n in desk.notes}
    by_id = {n["id"]: n for n in desk.notes}
    assert all(got["strings"][string] + fret == by_id[i]["pitch"] for i, (string, fret, *_) in neck.items())
    in_order = sorted(desk.notes, key=lambda n: n["start"])
    assert [neck[n["id"]][3] for n in in_order[:6]] == ["6", "6", "3", "6", "3", "2"]  # steps of the key
    assert [neck[n["id"]][2] for n in in_order] == score.hand([tuple(neck[n["id"]][:2]) for n in in_order])
    assert all(neck[n["id"]][4] is None for n in in_order)
    assert got["neck"] == copilot_harmony.on_the_neck(desk)

    # the person's choices: a chord, the key, the fingering; each is seen at once
    copilot_harmony.set_bars(desk.folder, [dict(bar=1, chords=[[0, E, "m"]])])
    copilot_harmony.set_home(desk.folder, (E, False))
    copilot_harmony.set_tab(desk.folder, "low")
    now = copilot_harmony.for_notes(desk)
    assert (
        now["home"] == dict(root=E, major=False, name="E minor", yours=True)
        and now["yours"] == [1]
        and now["tab"] == "low"
    )
    assert now["chords"][0]["yours"] and now["chords"][0]["numeral"] == "i" and not now["chords"][1]["yours"]
    assert [now["neck"][n["id"]][3] for n in in_order[:6]] == ["R", "R", "5", "8", "5", "4"]  # by the chord they chose
    assert [now["neck"][n["id"]][4] for n in in_order[:3]] == ["root", "root", "fifth"]
    assert [now["neck"][n["id"]][3] for n in in_order[6:9]] == ["b6", "b6", "b3"]  # the next bar: steps of E minor
    assert [tuple(now["neck"][n["id"]][:2]) for n in in_order] != [tuple(neck[n["id"]][:2]) for n in in_order]
    assert any(now["neck"][n["id"]][1] == 0 for n in in_order)  # open strings
    copilot_harmony.set_tab(desk.folder, None)
    assert copilot_harmony.for_notes(desk)["tab"] == "easy" == copilot.tab_style(desk.folder)
    with pytest.raises(Bass2SheetError, match="easy, closed, low, position"):
        copilot_harmony.set_tab(desk.folder, "high")


def test_the_score_carries_the_chords_as_they_stand_and_says_whose_they_are(workdir):
    desk = copilot.Desk(workdir, "mix")
    folder, machine = desk.folder, desk.machine
    guess = copilot.sheet_chords(folder, machine)
    assert sorted(guess) == list(range(2, 14)) and [(at, c["root"], c["quality"]) for at, c in guess[2]] == [
        (0, E, "m")
    ]
    assert copilot.sheet_key(folder, machine) == ("G major", G, True)
    with open(copilot.sheet(workdir, "mix", mscore=False)["musicxml"]) as f:
        xml = f.read()
    assert xml.count("<harmony>") == 12 and "Chord symbols: the tool's guess" in xml
    # C and, on beat 3, A minor seventh in bar 3; no chord in bar 4
    copilot_harmony.set_bars(folder, [dict(bar=2, chords=[[0, C, ""], [2, A, "m7"]]), dict(bar=3, chords=[])])
    got = copilot.sheet_chords(folder, machine)
    assert sorted(got) == [2, 3] + list(range(5, 14))
    assert [(at, c["root"], c["quality"], c["bass"]) for at, c in got[3]] == [(0, C, "", None), (8, A, "m7", None)]
    with open(copilot.sheet(workdir, "mix", mscore=False)["musicxml"]) as f:
        xml = f.read()
    assert xml.count("<harmony>") == 12 and "<kind>minor-seventh</kind>" in xml and "<offset>8</offset>" in xml
    assert "Chord symbols: the tool's guess, except in the 2 bars you chose them for" in xml
    # the key the score is written in is the home the person chose
    copilot_harmony.set_home(folder, (E, False))
    assert copilot.sheet_key(folder, machine) == ("E minor", E, False)


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
    seen = copilot_harmony.for_notes(desk)  # the song's own page, or its harmony page: whichever is opened first
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
    # the guess at the key and at the chords comes from it
    assert seen["home"]["name"] == "G major" and [c["name"] for c in seen["chords"]] == LOOP * 3
    assert [h["name"] for h in copilot_harmony.view(desk)["homes"]] == ["G major", "E minor", "C major"]


def test_a_song_whose_separated_tracks_are_gone_keeps_the_key_and_the_chords_it_had(workdir):
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
    assert [(c["name"], c["bar"]) for c in seen["chords"]] == [("C", 2)]  # the chord a bar it was given
    assert [bar["stands_on"] for bar in seen["bars"][1:5]] == ["E", "C", "G", "D"]
    assert copilot_harmony.offers(desk, 1)["chord"]["name"] == "C" and copilot_harmony.offers(desk, 2)["chord"] is None
    copilot_harmony._ASKED.discard(folder)


def test_the_person_chooses_home_and_the_names_follow(workdir):
    desk = copilot.Desk(workdir, "mix")
    before = copilot_harmony.view(desk)
    other = copilot_harmony.relative((before["home"]["root"], before["home"]["major"]))
    copilot_harmony.set_home(desk.folder, other)
    after = copilot_harmony.view(desk)
    assert (after["home"]["root"], after["home"]["major"]) == other and after["home"]["yours"]
    assert [c["numeral"] for c in before["chords"][:4]] == ["vi", "IV", "I", "V"]
    assert [c["numeral"] for c in after["chords"][:4]] == ["i", "bVI", "bIII", "bVII"]
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


def test_the_pages_are_served_and_lead_to_each_other(served):
    with urllib.request.urlopen(served + "/harmony", timeout=30) as r:
        page = r.read().decode()
    assert r.status == 200 and '<script id="logic">' in page and "/api/harmony" in page
    assert (
        "/api/chords" not in page and "/api/offers" not in page and 'method: "POST"' not in page
    )  # it changes nothing
    with urllib.request.urlopen(served + "/song", timeout=30) as r:
        song = r.read().decode()
    assert 'id="harmonylink"' in song and "/api/chords" in song and "/api/offers" in song and 'id="neck"' in song
    with urllib.request.urlopen(served + "/", timeout=30) as r:
        assert "/harmony#${id}" in r.read().decode()


def test_the_harmony_pages_own_logic():
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node is not installed")
    script = os.path.join(os.path.dirname(__file__), "copilot_harmony_logic.js")
    out = subprocess.run(["node", script, copilot_harmony.PAGE_PATH], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_the_server_gives_the_harmony_and_takes_no_change_to_it_there(served, workdir):
    status, seen = _json(served + "/api/harmony?id=mix")
    assert status == 200 and seen["id"] == "mix" and [c["name"] for c in seen["chords"]] == LOOP * 3
    assert _json(served + "/api/harmony?id=nothing")[0] == 404
    assert _json(served + "/api/harmony", dict(song="mix", home=[E, False]))[0] == 404  # no such way in
    assert copilot_harmony.read_home(os.path.join(workdir, "mix")) is None


def test_the_server_keeps_the_chords_the_key_and_the_fingering_set_on_the_songs_page(served, workdir):
    folder = os.path.join(workdir, "mix")
    status, song = _json(served + "/api/song?id=mix")
    assert status == 200 and song["notes"] and [c["name"] for c in song["chords"]] == LOOP * 3
    assert (
        song["home"]["name"] == "G major"
        and song["tab"] == "easy"
        and set(song["neck"]) == {n["id"] for n in song["notes"]}
    )
    assert song["names"]["40"] == "E2" and song["spelled"][4] == "E"  # the notes' names, and the key's
    # what the tool offers in place of a chord
    status, got = _json(served + "/api/offers?id=mix&bar=1")
    assert status == 200 and got["chord"]["name"] == "Em" and got["alike"]["part"] == [5, 9]
    assert _json(served + "/api/offers?id=mix&bar=2&beat=0")[1]["chord"]["name"] == "C"
    for bad in ("id=mix&bar=400", "id=mix", "id=mix&bar=two", "id=mix&bar=1&beat=2", "id=nothing&bar=1"):
        assert _json(served + "/api/offers?" + bad)[0] == 404
    # the person's chords for some bars: kept, logged, and given back with everything that follows from them
    change = [dict(bar=1, chords=[[0, E, "m7"]]), dict(bar=5, chords=[[0, E, "m7"]])]
    status, now = _json(served + "/api/chords", dict(song="mix", bars=change))
    assert status == 200 and now["yours"] == [1, 5] and now["chords"][0]["name"] == "Em7" and now["chords"][0]["yours"]
    assert copilot_harmony.read_bars(folder) == {1: [(0, E, "m7")], 5: [(0, E, "m7")]}
    status, said = _json(served + "/api/chords", dict(song="mix", bars=[dict(bar=1, chords=[[5, E, ""]])]))
    assert status == 400 and "twelve notes" in said["error"]
    assert _json(served + "/api/chords", dict(song="mix", bars=[]))[0] == 400
    status, fewer = _json(served + "/api/chords", dict(song="mix", bars=[dict(bar=5, chords=None)]))
    assert status == 200 and fewer["yours"] == [1]
    with open(os.path.join(folder, "log.jsonl")) as f:
        logged = [json.loads(line) for line in f if '"chord"' in line]
    assert [entry["kind"] for entry in logged] == ["chord", "chord"] and logged[0]["bars"] == change
    # the key: kept, and the names follow; None is the tool's guess again
    status, minor = _json(served + "/api/chords", dict(song="mix", home=[E, False]))
    assert (
        status == 200
        and minor["home"] == dict(root=E, major=False, name="E minor", yours=True)
        and minor["yours"] == [1]
    )
    assert minor["chords"][0]["numeral"] == "i7" and copilot_harmony.read_home(folder) == (E, False)
    assert _json(served + "/api/chords", dict(song="mix", home=[12, True]))[0] == 400
    status, back = _json(served + "/api/chords", dict(song="mix", home=None))
    assert status == 200 and back["home"] == song["home"]
    # the fingering: kept; the neck follows
    status, low = _json(served + "/api/chords", dict(song="mix", tab="low"))
    assert status == 200 and low["tab"] == "low" and low["neck"] != song["neck"] and copilot.tab_style(folder) == "low"
    assert _json(served + "/api/chords", dict(song="mix", tab="high"))[0] == 400
    assert _json(served + "/api/song?id=mix")[1]["tab"] == "low"
    # a fix to a note comes back with where the notes are played now
    note = song["notes"][3]
    status, fixed = _json(
        served + "/api/act",
        dict(song="mix", action=dict(kind="pitch", id=note["id"], pitch=note["pitch"] + 12, how="key")),
    )
    assert status == 200 and set(fixed["neck"]) == {n["id"] for n in fixed["notes"]}
    assert fixed["neck"][note["id"]][:2] != low["neck"][note["id"]][:2]
