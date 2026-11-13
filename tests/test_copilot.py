"""The copilot: the person fixes the notes, the tool helps and keeps the record (synthetic song, no models to fetch)."""

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from bass2sheet import copilot, copilot_server
from bass2sheet.common import Bass2SheetError

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


@pytest.fixture(scope="module")
def prepared(taste_song_files, cache_dir, tmp_path_factory):
    """A working directory with the synthetic song prepared, made once."""
    mix, stems = taste_song_files
    workdir = str(tmp_path_factory.mktemp("scores"))
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    said = []
    copilot.add([mix], workdir, opts, stems_dirs={mix: stems}, say=said.append)
    return workdir, said


@pytest.fixture
def workdir(prepared, tmp_path):
    """A fresh copy of it, for a test that changes things."""
    out = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared[0], out)
    return out


def _desk(workdir):
    return copilot.Desk(workdir, "mix")


# ------------------------------------------------------------------------------------------------- preparing


def test_a_prepared_song_has_what_the_page_and_the_copilot_need(prepared):
    workdir, said = prepared
    folder = os.path.join(workdir, "mix")
    machine, mine = (json.load(open(os.path.join(folder, name))) for name in ("machine.json", "notes.json"))
    assert len(machine["notes"]) >= 60 and machine["notes"][0]["id"] == "m0" and 25 < machine["length"] < 28
    assert [dict(n, state="machine") for n in machine["notes"]] == mine["notes"] and mine["checked"] == []
    p = machine["picture"]
    assert p["rows"] == copilot.PER_SEMI * (copilot.HIGH - copilot.LOW + 1) and p["low"] == copilot.LOW
    assert os.path.getsize(os.path.join(folder, "picture.bin")) == p["rows"] * p["columns"]
    assert abs(p["columns"] * p["step"] - machine["length"]) < 0.1
    assert os.path.getsize(os.path.join(folder, "pitch.bin")) // 4 == pytest.approx(machine["length"] * 100, abs=5)
    extra = np.load(os.path.join(folder, "desk.npz"))
    assert extra["semis"].shape[1] == copilot.SEMI_HI - copilot.SEMI_LO + 1 and extra["pyin"].ndim == 1
    for name in ("record.flac", "bass.flac"):
        assert os.path.getsize(os.path.join(folder, "audio", name)) > 100_000
    assert "notes" in said[-1] and said[0].endswith("hearing it")


def test_preparing_a_song_again_keeps_the_persons_work(workdir, prepared, cache_dir):
    d = _desk(workdir)
    first = d.notes[5]
    d.act(dict(kind="pitch", id=first["id"], pitch=first["pitch"] + 2, how="key"))
    song = copilot.load_songs(workdir)["mix"]
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    copilot.prepare(workdir, song, opts, say=lambda _: 0)
    again = _desk(workdir)
    assert again.note(first["id"])["state"] == "fixed" and again.public()["fixes"] == 1


def test_the_picture_puts_a_note_on_its_own_rows():
    import librosa

    from bass2sheet.common import SR
    from bass2sheet.render import pluck

    y = np.zeros(2 * SR, dtype=np.float32)
    w = 0.4 * pluck(librosa.midi_to_hz(40), 1.0, 100)
    y[SR // 2 : SR // 2 + len(w)] = w
    pic = copilot.picture(y)
    assert pic.dtype == np.uint8 and pic.shape[0] == copilot.PER_SEMI * (copilot.HIGH - copilot.LOW + 1)
    column = pic[:, int(1.0 / (copilot.PICTURE_HOP / copilot.PICTURE_SR))]
    rows = copilot.PER_SEMI * (40 - copilot.LOW)
    assert column[rows : rows + copilot.PER_SEMI].max() > 200  # the E2 sounds on the E2 rows
    assert column[rows - 2 * copilot.PER_SEMI : rows - copilot.PER_SEMI].max() < 120  # and not two rows under
    assert pic[:, 5].max() < 40  # nothing sounds before it


def test_intervals_are_joined_where_they_touch():
    assert copilot.merged([[5, 7], [0, 2], [1.98, 3], [3.04, 4], [9, 9]]) == [[0.0, 4.0], [5.0, 7.0]]
    assert copilot.inside(3.5, [[0, 4]]) and not copilot.inside(4.5, [[0, 4]])


# --------------------------------------------------------------------------------------------------- fixing


def test_a_change_of_pitch_remembers_what_the_tool_had_and_forgets_it_when_put_back(workdir):
    d = _desk(workdir)
    n = dict(d.notes[10])
    had = n["pitch"]
    d.act(dict(kind="pitch", id=n["id"], pitch=had + 12, how="key", offer=False))
    assert d.note(n["id"])["state"] == "fixed" and d.note(n["id"])["was"]["pitch"] == had
    d.act(dict(kind="pitch", id=n["id"], pitch=had, how="key"))
    assert d.note(n["id"]) == dict(n, state="machine") and "was" not in d.note(n["id"])
    with pytest.raises(Bass2SheetError, match="not a pitch"):
        d.act(dict(kind="pitch", id=n["id"], pitch=5))
    with pytest.raises(Bass2SheetError, match="no note"):
        d.act(dict(kind="pitch", id="nope", pitch=40))
    again = _desk(workdir)  # every fix is on disk
    assert [x["pitch"] for x in again.notes] == [x["pitch"] for x in d.notes]


def test_notes_are_deleted_added_moved_split_and_joined(workdir):
    d = _desk(workdir)
    count = len(d.notes)
    a, b, c = (dict(x) for x in d.notes[20:23])
    d.act(dict(kind="delete", id=b["id"]))
    assert len(d.notes) == count - 1 and all(x["id"] != b["id"] for x in d.notes)
    # a note added where another sounds: that one gives way, and the new one ends where the next begins
    mid = round((a["start"] + a["end"]) / 2, 3)
    new = d.act(dict(kind="add", start=mid, end=mid + 5, pitch=a["pitch"] + 7))["id"]
    assert d.note(new)["state"] == "added" and d.note(new)["start"] == mid and d.note(new)["end"] == c["start"]
    assert d.note(a["id"])["end"] == mid and d.note(a["id"])["was"]["end"] == a["end"]
    with pytest.raises(Bass2SheetError, match="starts right here"):
        d.act(dict(kind="add", start=c["start"] + 0.01, end=c["start"] + 0.2, pitch=50))
    # a start moved earlier: the note before gives way; an end cannot pass the next note
    d.act(dict(kind="move", id=c["id"], start=c["start"] - 0.05, end=c["end"] + 9))
    after_c = sorted(d.notes, key=lambda x: x["start"])
    k = next(i for i, x in enumerate(after_c) if x["id"] == c["id"])
    assert d.note(c["id"])["start"] == pytest.approx(c["start"] - 0.05, abs=1e-3)
    assert d.note(new)["end"] == d.note(c["id"])["start"] and d.note(c["id"])["end"] == after_c[k + 1]["start"]
    # split and join
    e = dict(d.notes[30])
    second = d.act(dict(kind="split", id=e["id"], at=(e["start"] + e["end"]) / 2))["id"]
    assert d.note(second)["split_from"] == e["id"] and d.note(e["id"])["end"] == d.note(second)["start"]
    assert d.note(second)["end"] == e["end"] and d.note(second)["pitch"] == e["pitch"]
    d.act(dict(kind="merge", id=e["id"]))
    assert d.note(e["id"])["end"] == e["end"] and d.note(e["id"])["merged"] == [second]
    assert all(x["id"] != second for x in d.notes)
    with pytest.raises(Bass2SheetError, match="inside it"):
        d.act(dict(kind="split", id=e["id"], at=e["start"]))
    with pytest.raises(Bass2SheetError, match="unknown kind"):
        d.act(dict(kind="dance"))


def test_several_notes_are_moved_deleted_and_joined_together(workdir):
    d = _desk(workdir)
    count = len(d.notes)
    run = [dict(x) for x in d.notes[10:14]]
    ids = [x["id"] for x in run]
    d.act(dict(kind="shift", ids=ids, by=12))
    assert [d.note(i)["pitch"] for i in ids] == [x["pitch"] + 12 for x in run]
    assert all(d.note(i)["state"] == "fixed" and d.note(i)["was"]["pitch"] == x["pitch"] for i, x in zip(ids, run))
    d.act(dict(kind="shift", ids=ids, by=-12))
    assert all(d.note(i) == dict(x, state="machine") for i, x in zip(ids, run))  # back where the tool had them
    with pytest.raises(Bass2SheetError, match="not a pitch"):
        d.act(dict(kind="shift", ids=ids, by=-40))
    assert [d.note(i)["pitch"] for i in ids] == [x["pitch"] for x in run]  # all or none
    # joined into the first: it lasts to the end of the last
    assert d.act(dict(kind="merge", ids=ids[::-1]))["joined"] == 4
    assert len(d.notes) == count - 3 and d.note(ids[0])["end"] == run[-1]["end"] and d.note(ids[0])["merged"] == ids[1:]
    assert d.act(dict(kind="undo"))["undone"] == "merge" and len(d.notes) == count
    with pytest.raises(Bass2SheetError, match="follow each other"):
        d.act(dict(kind="merge", ids=[ids[0], ids[2]]))
    d.act(dict(kind="delete", ids=ids[1:3]))
    assert len(d.notes) == count - 2 and d.act(dict(kind="undo"))["undone"] == "delete" and len(d.notes) == count
    m = copilot.mistakes(workdir, "mix")
    assert m["kinds"] == {}


def test_a_note_is_joined_with_every_repeat_that_follows(workdir):
    d = _desk(workdir)
    order = sorted(d.notes, key=lambda x: x["start"])
    k = next(i for i in range(len(order) - 2) if order[i]["pitch"] != order[i + 1]["pitch"])
    with pytest.raises(Bass2SheetError, match="not a repeat"):
        d.act(dict(kind="merge", id=order[k]["id"], run=True))
    # chop one note into four, as the tool does to a held note, and join the pieces again with one fix
    n = dict(order[k])
    step = (n["end"] - n["start"]) / 4
    for _ in range(3):
        last = max(
            (x for x in d.notes if x["pitch"] == n["pitch"] and n["start"] <= x["start"] < n["end"]),
            key=lambda x: x["start"],
        )
        d.act(dict(kind="split", id=last["id"], at=last["start"] + step))
    assert len(d.notes) == len(order) + 3
    assert d.act(dict(kind="merge", id=n["id"], run=True))["joined"] == 4
    assert len(d.notes) == len(order) and d.note(n["id"])["end"] == n["end"]
    assert sorted(d.notes, key=lambda x: x["start"])[k + 1]["id"] == order[k + 1]["id"]  # it stopped at the next pitch


def test_every_fix_can_be_undone_in_turn(workdir):
    d = _desk(workdir)
    before = json.dumps(sorted(d.notes, key=lambda n: n["id"]))
    n, m = dict(d.notes[12]), dict(d.notes[40])
    d.act(dict(kind="pitch", id=n["id"], pitch=n["pitch"] + 3, how="key", offer=False))
    d.act(dict(kind="delete", id=m["id"]))
    new = d.act(dict(kind="split", id=d.notes[3]["id"], at=(d.notes[3]["start"] + d.notes[3]["end"]) / 2))["id"]
    d.act(dict(kind="merge", id=d.notes[3]["id"]))
    d.act(dict(kind="add", start=0.2, end=0.5, pitch=45))
    assert d.public()["undo"] == 5 and all(x["id"] != new for x in d.notes)
    kinds = [d.act(dict(kind="undo"))["undone"] for _ in range(5)]
    assert kinds == ["add", "merge", "split", "delete", "pitch"]
    assert json.dumps(sorted(d.notes, key=lambda n: n["id"])) == before
    assert d.act(dict(kind="undo")) == dict(undone=None)
    log = [json.loads(line) for line in open(os.path.join(workdir, "mix", "log.jsonl"))]
    assert [e["kind"] for e in log] == ["pitch", "delete", "split", "merge", "add"] + ["undo"] * 5
    assert log[0]["before"][0]["pitch"] == n["pitch"] and log[0]["how"] == "key" and "when" in log[0]


# --------------------------------------------------------------------------------------- the copilot's help


def test_the_right_pitch_is_among_the_first_guesses_for_a_wrong_note(workdir):
    d = _desk(workdir)
    hits = []
    for k, shift in ((8, 12), (17, -12), (26, 5), (35, 2), (44, 7)):
        n = d.notes[k]
        right = n["pitch"]
        d.act(dict(kind="pitch", id=n["id"], pitch=right + shift, how="key", offer=False))
        guesses = d.guesses(n["id"])
        assert len(guesses) <= 6 and right + shift not in guesses and len(set(guesses)) == len(guesses)
        hits.append(guesses.index(right) if right in guesses else 99)
        d.act(dict(kind="undo"))
    assert max(hits) <= 2 and hits.count(0) >= 3, hits


def _same_mistake_everywhere(workdir, shift=5):
    """As if the tool had written every note of one pitch a fourth too high. Returns (the label they carry now,
    the pitch they sound, their ids)."""
    folder = os.path.join(workdir, "mix")
    machine = json.load(open(os.path.join(folder, "machine.json")))
    pitches = [n["pitch"] for n in machine["notes"]]
    sounds = max(set(pitches), key=pitches.count)
    ids = [n["id"] for n in machine["notes"] if n["pitch"] == sounds]
    for name in ("machine.json", "notes.json"):
        data = json.load(open(os.path.join(folder, name)))
        for n in data["notes"]:
            if n["id"] in ids:
                n["pitch"] = sounds + shift
        json.dump(data, open(os.path.join(folder, name), "w"))
    return sounds + shift, sounds, ids


def test_after_a_fix_the_notes_that_sound_the_same_are_offered_and_the_person_decides(workdir):
    label, sounds, ids = _same_mistake_everywhere(workdir)
    d = _desk(workdir)
    result = d.act(dict(kind="pitch", id=ids[0], pitch=sounds, how="guess", rank=1))
    offer = result["offer"]
    assert (offer["seed"], offer["label"], offer["pitch"]) == (ids[0], label, sounds)
    assert set(offer["ids"]) == set(ids[1:])  # the others with the same wrong label, and nothing else
    d.act(dict(kind="pitches", ids=offer["ids"], pitch=sounds, how="offer", seed=ids[0]))
    assert all(d.note(i)["pitch"] == sounds and d.note(i)["state"] == "fixed" for i in ids)
    assert d.act(dict(kind="undo"))["undone"] == "pitches" and d.note(ids[1])["pitch"] == label
    d.act(dict(kind="refuse", ids=offer["ids"], pitch=sounds, seed=ids[0]))  # logged, nothing changes
    log = [json.loads(line) for line in open(os.path.join(workdir, "mix", "log.jsonl"))]
    assert log[-1]["kind"] == "refuse" and d.note(ids[1])["pitch"] == label


def test_what_the_person_played_past_is_not_offered_and_tells_right_from_wrong(workdir):
    d = _desk(workdir)
    pitches = [n["pitch"] for n in d.notes]
    common = max(set(pitches), key=pitches.count)
    same = [n for n in d.notes if n["pitch"] == common]
    assert len(same) >= 6
    d.mark_checked(0, same[3]["start"] + 0.01)  # four of them played past: found right
    odd = same[4]
    result = d.act(dict(kind="pitch", id=odd["id"], pitch=common + 7, how="key"))  # one "fixed" all the same
    # the rest sound like the ones found right at least as much as like this one: nothing to offer
    assert "offer" not in result
    assert d.alike(same[5]["id"]) == (None, [])  # and a note that was not changed has no look-alikes to fix


# ------------------------------------------------------------------------------------------- the record


def test_the_record_of_mistakes_names_each_kind(workdir):
    label, sounds, ids = _same_mistake_everywhere(workdir)
    d = _desk(workdir)
    free = [n for n in d.notes if n["id"] not in ids]
    d.act(dict(kind="pitch", id=ids[0], pitch=sounds, how="guess", rank=1, offer=False))  # wrong pitch
    d.act(dict(kind="pitch", id=free[0]["id"], pitch=free[0]["pitch"] + 12, how="key", offer=False))  # wrong octave
    d.act(dict(kind="delete", id=free[1]["id"]))  # extra note
    d.act(dict(kind="split", id=free[2]["id"], at=(free[2]["start"] + free[2]["end"]) / 2))  # two written as one
    d.act(dict(kind="merge", id=free[4]["id"]))  # one written as two
    d.act(dict(kind="add", start=0.1, end=0.4, pitch=40))  # missed note
    d.act(dict(kind="move", id=free[8]["id"], start=free[8]["start"] + 0.08, end=free[8]["end"]))  # start
    d.mark_checked(0, free[10]["start"])
    m = copilot.mistakes(workdir, "mix")
    assert m["kinds"] == {
        "wrong pitch": 1,
        "wrong octave": 1,
        "extra note": 1,
        "two notes written as one": 1,
        "one note written as two": 1,
        "missed note": 1,
        "start": 1,
    }
    wrong = next(r for r in m["rows"] if r["kind"] == "wrong pitch")
    assert wrong["had"] != wrong["now"] and m["notes"] == len(d.machine["notes"])
    assert m["right"] > 0 and m["unchecked"] > 0 and m["right"] + m["unchecked"] < m["notes"]
    text = copilot.report(workdir)
    assert "## mix" in text and "| wrong octave | 1 |" in text and "are not checked yet" in text


def test_the_sheet_is_written_from_the_persons_notes(workdir):
    d = _desk(workdir)
    n = dict(d.notes[6])
    d.act(dict(kind="pitch", id=n["id"], pitch=n["pitch"] + 12, how="key", offer=False))
    d.act(dict(kind="delete", id=d.notes[9]["id"]))
    d.mark_checked(0, d.machine["length"] / 2)
    files = copilot.sheet(workdir, "mix", mscore="/nonexistent/mscore")
    xml = open(files["musicxml"]).read()
    assert "checked by ear against the record; the rest is the machine" in xml and "DRAFT" not in xml
    import pretty_midi

    played = pretty_midi.PrettyMIDI(files["midi"]).instruments[0].notes
    assert len(played) == len(d.notes) and n["pitch"] + 12 in {x.pitch for x in played}
    d.mark_checked(0, d.machine["length"])
    assert (
        "Bass: checked by ear against the record<"
        in open(copilot.sheet(workdir, "mix", mscore="/x")["musicxml"]).read()
    )


# --------------------------------------------------------------------------------------------------- the page


@pytest.fixture
def server(workdir):
    srv = copilot_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_the_server_gives_the_song_takes_fixes_and_serves_its_files(server):
    url, workdir = server
    assert b"bass2sheet" in _get(url + "/")[2]
    songs = json.loads(_get(url + "/api/songs")[2])
    assert [s["id"] for s in songs] == ["mix"] and songs[0]["notes"] >= 60
    song = json.loads(_get(url + "/api/song?id=mix")[2])
    assert song["bars"][0]["bar"] == 1 and song["low"] < song["high"] and song["names"]["40"] == "E2"
    n = song["notes"][10]
    r = _post(
        url + "/api/act", dict(song="mix", action=dict(kind="pitch", id=n["id"], pitch=n["pitch"] + 12, how="key"))
    )
    assert r["fixes"] == 1 and r["undo"] == 1 and next(x for x in r["notes"] if x["id"] == n["id"])["state"] == "fixed"
    guesses = json.loads(_get(url + f"/api/guesses?song=mix&id={n['id']}")[2])["guesses"]
    assert n["pitch"] in guesses[:3]
    assert _post(url + "/api/checked", dict(song="mix", start=0, end=5.5))["checked"] == [[0.0, 5.5]]
    assert _post(url + "/api/act", dict(song="mix", action=dict(kind="undo")))["fixes"] == 0
    for bad, code in (
        (dict(song="mix", action=dict(kind="pitch", id="nope", pitch=40)), 400),
        (dict(song="other", action=dict(kind="undo")), 400),
    ):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(url + "/api/act", bad)
        assert e.value.code == code
    status, headers, body = _get(url + "/data/mix/audio/bass.flac", {"Range": "bytes=0-3"})
    assert (status, body, headers["Content-Type"]) == (206, b"fLaC", "audio/flac")
    assert len(_get(url + "/data/mix/picture.bin")[2]) == song["picture"]["rows"] * song["picture"]["columns"]
    for path in ("/data/mix/notes.json", "/data/mix/../songs.json", "/data/other/picture.bin", "/api/song?id=other"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(url + path)
        assert e.value.code == 404, path
    links = _post(url + "/api/sheet", dict(song="mix"))
    assert (
        links["musicxml"] == "/data/mix/sheet/bass.musicxml" and b"<score-partwise" in _get(url + links["musicxml"])[2]
    )


def test_serving_an_empty_directory_says_what_to_do(tmp_path):
    with pytest.raises(Bass2SheetError, match="bass2sheet-copilot add"):
        copilot_server.make_server(str(tmp_path), 0)


def test_the_page_has_every_element_its_script_reaches_for():
    import re

    page = open(copilot_server.PAGE_PATH).read()
    html, script = page.split('<script id="app">')
    made = set(re.findall(r'id="([^"]+)"', page))
    missing = set(re.findall(r'\$\("([^"]+)"\)', script)) - made
    assert not missing, missing
    assert {"hearRecord", "hearBass", "hearNotes", "hearSplit"} <= made  # reached by a built name


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's logic")
def test_the_page_logic_holds():
    script = os.path.join(os.path.dirname(__file__), "copilot_page_logic.js")
    r = subprocess.run(["node", script, copilot_server.PAGE_PATH], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def test_the_command_line_reports_and_refuses_politely(workdir, capsys):
    assert copilot.main(["report", "--dir", workdir]) == 0
    assert "# What the tool got wrong" in capsys.readouterr().out
    assert copilot.main(["add", "/nonexistent/song.mp3", "--dir", workdir]) == 1
    assert "no such file" in capsys.readouterr().out
    assert copilot.main(["sheet", "nope", "--dir", workdir]) == 1
    assert "is not prepared" in capsys.readouterr().out
