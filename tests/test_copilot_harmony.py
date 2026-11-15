"""The song's harmony for the copilot's page: chords named against a home, how they follow each other, and what
the bass plays under each."""

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
    ]  # one chord held is a loop of a bar
    assert _shape(parts(list("ABCDEFG"))) == [(0, 7, "A", "ABCDEFG")]  # nothing repeats: one passage
    assert _shape(parts(list("ABCDABCEABCDABCEABCDABCE"))) == [(0, 24, "A", "ABCDABCE")]  # eight bars, not four
    # one bar heard differently in one round does not break the loop, and the loop is what most rounds have
    assert _shape(parts(list("ABCDABXDABCDABCD"))) == [(0, 16, "A", "ABCD")]
    assert _shape(parts(list("AXCDABCDABCDABCD"))) == [(0, 16, "A", "ABCD")]
    assert parts([]) == [] and _shape(parts(list("AB"))) == [(0, 2, "A", "AB")]
    assert _shape(parts(list("AAB"))) == [(0, 3, "A", "AAB")]  # a chord held for two bars is not a loop yet
    # a loop that comes back begun part of the way round is the same part, and says how far round it begins
    found = parts(list("ABCDABCDABCD" + "xyzD" + "ABCDABCDABCD"))
    assert _shape(found) == [(0, 12, "A", "ABCD"), (12, 3, "B", "xyz"), (15, 13, "A", "ABCD")]
    assert [p["skip"] for p in found] == [0, 0, 3]  # it comes back in on the loop's last bar
    found = parts(list("ABCDABCDABCDxyCDABCDABCDAB"))
    assert _shape(found) == [(0, 12, "A", "ABCD"), (12, 2, "B", "xy"), (14, 12, "A", "ABCD")]
    assert [p["skip"] for p in found] == [0, 0, 2]
    assert [p["skip"] for p in parts(list("EEEEEEEEqEEEEEEEE"))] == [0, 0, 0]  # one chord held has no way round


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


def _machine(chords, bars=4, length=None):
    grid = [round(-0.5 + 0.125 * k, 3) for k in range(4 + 16 * bars + 1)]  # bar 1 starts at 0, a bar lasts 2 s
    return dict(grid=grid, origin=4, length=2.0 * bars if length is None else length, chords=chords)


def test_the_tools_chords_become_stretches_of_time():
    found = [
        dict(bar=1, root=E, quality="m", score=0.9),
        dict(bar=2, root=C, quality="", score=0.0),  # a bar where no harmony was heard has no chord
        dict(bar=3, root=G, quality="", score=0.7),
        dict(bar=4, root=D, quality="", score=0.6),
        dict(bar=9, root=D, quality="", score=0.6),  # beyond the song
    ]
    chords = copilot_harmony.chords_of(_machine(found, length=7.5))
    assert [(c["start"], c["end"], c["root"], c["quality"]) for c in chords] == [
        (0.0, 2.0, E, "m"),
        (4.0, 6.0, G, ""),
        (6.0, 7.5, D, ""),  # the last one ends with the song
    ]
    assert copilot_harmony.chords_of(_machine([])) == [] and copilot_harmony.chords_of(_machine(None)) == []


def test_the_persons_chord_takes_the_place_of_what_the_tool_heard_in_its_bar():
    H = copilot_harmony
    found = [
        dict(bar=k, root=root, quality=q, score=0.8)
        for k, (root, q) in enumerate([(E, "m"), (C, ""), (G, ""), (D, "")], 1)
    ]
    machine = _machine(found)
    heard = H.chords_of(machine)
    said = lambda chords: [(c["start"], c["end"], c["root"], c["quality"], c["yours"]) for c in chords]  # noqa: E731
    assert said(H.with_fixes(machine, heard, {})) == [
        (0.0, 2.0, E, "m", False),
        (2.0, 4.0, C, "", False),
        (4.0, 6.0, G, "", False),
        (6.0, 8.0, D, "", False),
    ]
    fixed = H.with_fixes(machine, heard, {2: [(A, "m")]})
    assert said(fixed)[1] == (2.0, 4.0, A, "m", True) and [c["yours"] for c in fixed] == [False, True, False, False]
    assert fixed[1]["score"] == 1.0 and fixed[0]["score"] == 0.8
    # two chords share their bar half and half; a bar past the end of the song is left out
    two = H.with_fixes(machine, heard, {3: [(G, ""), (B, "dim")], 9: [(C, "")]})
    assert said(two)[2:4] == [(4.0, 5.0, G, "", True), (5.0, 6.0, B, "dim", True)] and len(two) == 5
    # a chord the tool held across the bar line is cut where the person's begins, and a bar it had nothing in is filled
    long = [
        dict(start=0.0, end=4.0, root=E, quality="m", score=0.7),
        dict(start=6.0, end=8.0, root=D, quality="", score=0.7),
    ]
    assert said(H.with_fixes(machine, long, {2: [(C, "")], 3: [(G, "7")]})) == [
        (0.0, 2.0, E, "m", False),
        (2.0, 4.0, C, "", True),
        (4.0, 6.0, G, "7", True),
        (6.0, 8.0, D, "", False),
    ]


def test_the_persons_chords_are_kept_beside_the_song(tmp_path):
    H, folder = copilot_harmony, str(tmp_path)
    assert H.read_fixes(folder) == {}
    H.set_chords(folder, [3, 7], (A, "m7"))
    H.set_chords(folder, [4], (D, ""))
    H.set_home(folder, (C, True))  # the home and the chords are kept apart in the one file
    assert H.read_fixes(folder) == {3: [(A, "m7")], 7: [(A, "m7")], 4: [(D, "")]} and H.read_home(folder) == (C, True)
    H.set_chords(folder, [7, 9], None)  # taken away again; a bar that had none is no matter
    assert H.read_fixes(folder) == {3: [(A, "m7")], 4: [(D, "")]} and H.read_home(folder) == (C, True)
    for bars, chord in (
        ([], (C, "")),
        ([0], (C, "")),
        ([True], (C, "")),
        ("3", (C, "")),
        ([3], (12, "")),
        ([3], (C, "minor")),
        ([3], (True, "")),
    ):
        with pytest.raises(Bass2SheetError):
            H.set_chords(folder, bars, chord)
    assert H.read_fixes(folder) == {3: [(A, "m7")], 4: [(D, "")]}


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


def test_the_page_gets_the_song_as_chords_against_a_home(workdir):
    desk = copilot.Desk(workdir, "mix")
    seen = copilot_harmony.view(desk)
    found = (desk.machine["key"][1], desk.machine["key"][2])
    assert (seen["home"]["root"], seen["home"]["major"]) == found and not seen["home"]["yours"]
    assert seen["home"]["name"] == copilot_harmony.key_name(found)
    # the choices: the key the tool found, and the key with the same notes
    assert [(h["root"], h["major"], h["found"]) for h in seen["homes"]] == [
        (*found, True),
        (*copilot_harmony.relative(found), False),
    ]
    # the song is a loop of E minor, C, G and D, a bar each: the chords are those, in time order, a bar long
    chords = seen["chords"]
    assert len(chords) >= 8 and {c["name"] for c in chords} <= {"Em", "C", "G", "D", "E", "Cm", "Gm", "Dm"}
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(chords, chords[1:]))
    assert all(1.5 < c["end"] - c["start"] < 2.5 for c in chords[1:-1])  # the first and last bars are cut short
    assert all(c["numeral"] == copilot_harmony.numeral(c["root"], c["quality"], found) for c in chords)
    assert [c["bar"] for c in chords] == sorted(c["bar"] for c in chords) and chords[0]["bar"] >= 1
    assert len(seen["bars"]) >= 12 and seen["bars"][0]["bar"] == 1 and "pattern" in seen["bars"][0]
    # the bass starts each bar on the chord's root, and spends most of its time on roots and fifths
    assert sum(c["role"] == "root" for c in chords) >= 0.8 * len(chords)
    assert abs(sum(seen["roles"].values()) - 1) < 0.01 and seen["roles"]["root"] > 0.5
    assert seen["roles"]["root"] + seen["roles"]["fifth"] > 0.8
    # where the song spends its time, and how it moves
    assert abs(sum(p["share"] for p in seen["places"]) - 1) < 0.02
    assert seen["places"][0]["seconds"] >= seen["places"][-1]["seconds"]
    assert seen["moves"] and all(m["a"]["name"] != m["b"]["name"] and m["times"] >= 1 for m in seen["moves"])
    assert seen["motions"] and all(m["name"] == copilot_harmony.MOTIONS[m["up"]] for m in seen["motions"])
    assert not seen["fine"]
    # each chord with its job, its distance from rest, its notes, and its place on each of the three maps
    assert {c["job"] for c in chords} <= set(copilot_harmony.JOBS) and all(c["far"] >= 0 for c in chords)
    assert all(c["tones"][0] == c["root"] and len(c["tones"]) == 3 for c in chords)
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


def test_the_song_is_given_in_parts_with_what_the_bass_plays_over_each(workdir):
    seen = copilot_harmony.view(copilot.Desk(workdir, "mix"))
    parts, bars = seen["parts"], seen["bars"]
    # the parts follow each other and cover every bar
    assert parts[0]["bar"] == 1 and sum(p["bars"] for p in parts) == len(bars)
    assert all(a["bar"] + a["bars"] == b["bar"] and a["end"] <= b["start"] + 1e-6 for a, b in zip(parts, parts[1:]))
    # the song is one loop of four bars, three times round, between a bar that leads in and the bars it dies away in
    loop = max(parts, key=lambda p: p["bars"])
    assert (loop["bars"], loop["size"]) == (12, 4) and [len(bar) for bar in loop["loop"]] == [1, 1, 1, 1]
    assert [bar[0]["name"] for bar in loop["loop"]] == ["Em", "C", "G", "D"]
    assert {bar[0]["job"] for bar in loop["loop"]} <= set(copilot_harmony.JOBS)
    said = {entry["letter"]: entry for entry in seen["loops"]}
    mine = said[loop["letter"]]
    assert (mine["size"], mine["bars"], mine["rounds"], mine["times"]) == (4, 12, 3.0, 1) and len(mine["bass"]) == 4
    # over each chord the bass plays root, root, fifth, octave, fifth, fourth: the first bars are heard just so
    for line in mine["bass"][:2]:
        assert [n["degree"] for n in line["notes"]] == ["R", "R", "5", "8", "5", "4"] and line["share"] == 1.0
        assert [n["slot"] for n in line["notes"]] == [0, 4, 6, 8, 12, 14]
        assert [n["role"] for n in line["notes"]] == ["root", "root", "fifth", "root", "fifth", "other"]
    assert seen["on_root"] == 1.0  # every chord starts on its root
    # each bass note: where it is played, the chord it sounds under, and what it is to that chord
    notes = seen["bass"]
    assert len(notes) == len(copilot.Desk(workdir, "mix").notes) and seen["strings"] == [28, 33, 38, 43]
    assert all(
        0 <= n["string"] <= 3 and 0 <= n["fret"] <= 24 and seen["strings"][n["string"]] + n["fret"] == n["pitch"]
        for n in notes
    )
    assert all(0 <= n["slot"] <= 15 and n["bar"] >= 1 for n in notes)
    under = [n for n in notes if n["chord"] is not None]
    assert len(under) >= 0.9 * len(notes) and all(n["degree"] and n["role"] for n in under)
    assert all(
        seen["chords"][n["chord"]]["start"] - 0.05 <= n["start"] < seen["chords"][n["chord"]]["end"] for n in under
    )


def test_a_chord_the_person_changed_is_in_the_song_from_then_on(workdir):
    desk = copilot.Desk(workdir, "mix")
    before = copilot_harmony.view(desk)
    loop = max(before["parts"], key=lambda p: p["bars"])
    bars = [loop["bar"] + 1 + 4 * k for k in range(3)]  # the second bar of the loop, in each of its three rounds
    assert all(before["bars"][b - 1]["holds"][0]["name"] == "C" and not before["bars"][b - 1]["yours"] for b in bars)
    assert before["fixed"] == 0 and len(before["palette"]) == 12 * len(before["qualities"]) == 144
    assert before["palette"]["9,m7"]["tones"] == [9, 0, 4, 7] and before["palette"]["9,m7"]["name"] == "Am7"

    copilot_harmony.set_chords(desk.folder, bars, (A, "m"))
    after = copilot_harmony.view(desk)
    assert after["fixed"] == 3
    for b in bars:
        bar = after["bars"][b - 1]
        assert bar["yours"] and bar["holds"][0]["name"] == "Am" and bar["heard"][0]["name"] == "C"
    assert sum(c["yours"] for c in after["chords"]) == 3 and "yours" in after["chords"][0]
    # everything that is said of the song follows: its loop, where it spends its time, the map
    now = max(after["parts"], key=lambda p: p["bars"])
    assert [bar[0]["name"] for bar in now["loop"]] == ["Em", "Am", "G", "D"] and now["bars"] == 12
    assert "Am" in {p["name"] for p in after["places"]} and "C" not in {p["name"] for p in after["places"]}
    assert any(s["name"] == "Am" and s["seconds"] > 0 for s in after["maps"]["roads"]["spots"])
    assert after["home"] == before["home"] and desk.notes == copilot.Desk(workdir, "mix").notes  # and nothing else

    copilot_harmony.set_chords(desk.folder, bars[:1], None)  # one bar back to what the tool heard
    back = copilot_harmony.view(desk)
    assert (
        back["fixed"] == 2
        and back["bars"][bars[0] - 1]["holds"][0]["name"] == "C"
        and not back["bars"][bars[0] - 1]["yours"]
    )


def test_the_song_is_put_on_the_neck_for_playing_unless_the_person_chooses_another_way(workdir):
    from bass2sheet import score

    desk = copilot.Desk(workdir, "mix")
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


def test_the_person_chooses_home_and_the_names_follow(workdir):
    desk = copilot.Desk(workdir, "mix")
    before = copilot_harmony.view(desk)
    other = copilot_harmony.relative((before["home"]["root"], before["home"]["major"]))
    copilot_harmony.set_home(desk.folder, other)
    after = copilot_harmony.view(desk)
    assert (after["home"]["root"], after["home"]["major"]) == other and after["home"]["yours"]
    assert [c["numeral"] for c in after["chords"]] != [c["numeral"] for c in before["chords"]]
    assert [c["root"] for c in after["chords"]] == [c["root"] for c in before["chords"]]  # the chords are the same
    assert len(after["homes"]) == 2 and desk.notes == copilot.Desk(workdir, "mix").notes  # nothing of the notes
    # a home that is neither the found key nor its relative is offered too, so that it can be seen as chosen
    copilot_harmony.set_home(desk.folder, (1, True))
    assert [h["name"] for h in copilot_harmony.view(desk)["homes"]][-1] == "Db major"
    copilot_harmony.set_home(desk.folder, None)  # back to the key the tool found
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
    assert r.status == 200 and '<script id="logic">' in page and "/api/harmony" in page
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


def test_the_server_gives_the_harmony_and_keeps_the_choice_of_home(served, workdir):
    status, seen = _json(served + "/api/harmony?id=mix")
    assert status == 200 and seen["id"] == "mix" and seen["chords"] and not seen["home"]["yours"]
    assert _json(served + "/api/harmony?id=nothing")[0] == 404
    other = seen["homes"][1]
    status, after = _json(served + "/api/harmony", dict(song="mix", home=[other["root"], other["major"]]))
    assert status == 200 and after["home"]["name"] == other["name"] and after["home"]["yours"]
    assert copilot_harmony.read_home(os.path.join(workdir, "mix")) == (other["root"], other["major"])
    assert _json(served + "/api/harmony?id=mix")[1]["home"] == after["home"]  # kept
    status, said = _json(served + "/api/harmony", dict(song="mix", home=[12, True]))
    assert status == 400 and "twelve notes" in said["error"]
    # a chord of the person's for some bars: kept, logged, and the home is left as it is
    _json(served + "/api/harmony", dict(song="mix", home=[other["root"], other["major"]]))
    status, fixed = _json(served + "/api/harmony", dict(song="mix", chords=dict(bars=[3, 7], chord=[A, "m"])))
    assert status == 200 and fixed["fixed"] == 2 and fixed["bars"][2]["yours"] and fixed["home"]["yours"]
    assert copilot_harmony.read_fixes(os.path.join(workdir, "mix")) == {3: [(A, "m")], 7: [(A, "m")]}
    with open(os.path.join(workdir, "mix", "log.jsonl")) as f:
        logged = [json.loads(line) for line in f if '"chord"' in line]
    assert logged[-1]["kind"] == "chord" and logged[-1]["bars"] == [3, 7] and logged[-1]["chord"] == [A, "m"]
    status, said = _json(served + "/api/harmony", dict(song="mix", chords=dict(bars=[3], chord=[A, "minor"])))
    assert status == 400 and "twelve notes" in said["error"]
    assert _json(served + "/api/harmony", dict(song="mix", chords=dict(bars=[], chord=[A, "m"])))[0] == 400
    status, fewer = _json(served + "/api/harmony", dict(song="mix", chords=dict(bars=[3], chord=None)))
    assert status == 200 and fewer["fixed"] == 1 and not fewer["bars"][2]["yours"] and fewer["home"]["yours"]
    status, back = _json(served + "/api/harmony", dict(song="mix", home=None))
    assert status == 200 and back["home"] == seen["home"] and back["fixed"] == 1
    # how the song is put on the neck: kept, and the chords and the home are left as they are
    status, closed = _json(served + "/api/harmony", dict(song="mix", tab="closed"))
    assert status == 200 and closed["tab"] == "closed" and closed["fixed"] == 1
    assert copilot.tab_style(os.path.join(workdir, "mix")) == "closed"
    assert _json(served + "/api/harmony", dict(song="mix", tab="high"))[0] == 400
    assert _json(served + "/api/harmony?id=mix")[1]["tab"] == "closed"
