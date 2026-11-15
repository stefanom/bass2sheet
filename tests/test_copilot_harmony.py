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
    for name, address in seen["audio"].items():
        assert os.path.isfile(os.path.join(workdir, "mix", "audio", name + ".flac")) and address.endswith(
            name + ".flac"
        )


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
    status, back = _json(served + "/api/harmony", dict(song="mix", home=None))
    assert status == 200 and back["home"] == seen["home"]
