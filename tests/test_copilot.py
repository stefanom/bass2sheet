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


def _plucks(spec, seconds):
    import librosa

    from bass2sheet.common import SR
    from bass2sheet.render import pluck

    y = np.zeros(int(seconds * SR), dtype=np.float32)
    for start, dur, pitch in spec:
        w = 0.4 * pluck(librosa.midi_to_hz(pitch), dur, 100)
        y[int(start * SR) : int(start * SR) + len(w)] += w
    return y


def test_the_picture_puts_a_note_on_its_own_rows_from_where_it_starts_to_where_it_ends():
    pic = copilot.picture(_plucks([(0.5, 1.0, 40)], 2.0))
    assert pic.dtype == np.uint8 and pic.shape[0] == copilot.PER_SEMI * (copilot.HIGH - copilot.LOW + 1)
    at = lambda t: pic[:, int(t / (copilot.PICTURE_HOP / copilot.PICTURE_SR))]
    rows = lambda p: slice(copilot.PER_SEMI * (p - copilot.LOW), copilot.PER_SEMI * (p - copilot.LOW + 1))
    assert at(1.0)[rows(40)].max() > 200  # the E2 sounds on the E2 rows
    assert max(at(1.0)[rows(p)].max() for p in (38, 39, 41, 42)) < 40  # and on no other near it
    assert at(1.0)[rows(52)].max() > 60  # its octave partial shows above it, fainter
    assert pic[:, 5].max() < 40  # nothing sounds before it
    # it is there a few hundredths of a second after it starts, and not before
    assert at(0.42).max() < 40 and at(0.58)[rows(40)].max() > 150
    assert at(1.40)[rows(40)].max() > 100 and at(1.60).max() < 40


def test_the_picture_keeps_quick_notes_a_semitone_apart_each_on_its_own_rows():
    line = [(0.2 + 0.2 * k, 0.19, p) for k, p in enumerate([28, 29, 28, 29, 40, 41, 40, 41])]  # down to the low E
    pic = copilot.picture(_plucks(line, 2.2))
    rows = lambda p: slice(copilot.PER_SEMI * (p - copilot.LOW), copilot.PER_SEMI * (p - copilot.LOW + 1))
    for start, dur, pitch in line:
        middle = pic[:, int((start + dur / 2) / (copilot.PICTURE_HOP / copilot.PICTURE_SR))]
        other = pitch + 1 if pitch in (28, 40) else pitch - 1
        assert middle[rows(pitch)].max() > 150 and middle[rows(other)].max() < 60, (start, pitch)
    assert (
        copilot.picture(np.zeros(100, np.float32)).max() == 0
        and copilot.picture(np.zeros(44100, np.float32)).max() == 0
    )


def test_a_song_prepared_with_another_kind_of_picture_gets_the_new_one_when_it_is_opened(workdir):
    folder = os.path.join(workdir, "mix")
    machine = json.load(open(os.path.join(folder, "machine.json")))
    was = dict(machine["picture"])
    assert was["kind"] == copilot.PICTURE_KIND and was["step"] == 0.01
    machine["picture"] = dict(rows=141, columns=7, low=copilot.LOW, per_semitone=3, step=128 / 11025)  # as it was
    json.dump(machine, open(os.path.join(folder, "machine.json"), "w"))
    open(os.path.join(folder, "picture.bin"), "wb").write(bytes(141 * 7))
    notes = open(os.path.join(folder, "notes.json")).read()
    d = _desk(workdir)
    assert d.public()["picture"] == was and json.load(open(os.path.join(folder, "machine.json")))["picture"] == was
    assert os.path.getsize(os.path.join(folder, "picture.bin")) == was["rows"] * was["columns"]
    assert open(os.path.join(folder, "notes.json")).read() == notes and len(d.machine["notes"]) == len(machine["notes"])


def test_intervals_are_joined_where_they_touch():
    assert copilot.merged([[5, 7], [0, 2], [1.98, 3], [3.04, 4], [9, 9]]) == [[0.0, 4.0], [5.0, 7.0]]
    assert copilot.inside(3.5, [[0, 4]]) and not copilot.inside(4.5, [[0, 4]])


def _stretchers():
    import imageio_ffmpeg

    found = [("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())]
    if copilot.stretcher()[0] == "rubberband":
        found.append(copilot.stretcher())
    return found


@pytest.mark.parametrize("program", _stretchers(), ids=lambda p: p[0])
def test_a_song_is_slowed_down_at_the_same_pitch(workdir, monkeypatch, program):
    import soundfile as sf

    monkeypatch.setattr(copilot, "stretcher", lambda: program)
    folder = os.path.join(workdir, "mix", "audio")
    full, sr = sf.read(os.path.join(folder, "bass.flac"), dtype="float32")
    path = copilot.slowed(workdir, "mix", "bass", 80)
    slow, sr2 = sf.read(path, dtype="float32")
    assert path.endswith("bass-80.flac") and sr2 == sr and len(slow) == round(len(full) / 0.8)  # to the sample
    d = _desk(workdir)
    n = max(d.notes, key=lambda x: x["end"] - x["start"])

    def loudest_hz(y, t0, t1):
        seg = y[int(t0 * sr) : int(t1 * sr)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 17))
        return np.fft.rfftfreq(1 << 17, 1 / sr)[spec.argmax()]

    a, b = n["start"] + 0.05, n["end"] - 0.05
    assert loudest_hz(slow, a / 0.8, b / 0.8) == pytest.approx(loudest_hz(full, a, b), rel=0.01)  # the same note
    at = int((n["start"] + 0.06) / 0.8 * sr)
    assert float(np.abs(slow[at : at + sr // 20]).max()) > 0.02  # and it sounds where the song has it
    made = os.path.getmtime(path)
    assert copilot.slowed(workdir, "mix", "bass", 80) == path and os.path.getmtime(path) == made  # kept
    record = sf.info(copilot.slowed(workdir, "mix", "record", 50))
    assert record.channels == 2 and record.frames == 2 * sf.info(os.path.join(folder, "record.flac")).frames
    for name, percent in (("bass", 85), ("drums", 80), ("bass", 100)):
        with pytest.raises(Bass2SheetError, match="the speeds are"):
            copilot.slowed(workdir, "mix", name, percent)
    assert sorted(os.listdir(folder)) == [
        "bass-80.flac",
        "bass.flac",
        "record-50.flac",
        "record.flac",
    ]  # nothing left over


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


@pytest.fixture
def no_repeats(monkeypatch):
    """As for a song with no two bars alike: after a fix, the notes that sound the same are what is offered."""
    monkeypatch.setattr(copilot.Desk, "repeats", property(lambda self: None))


def test_after_a_fix_the_notes_that_sound_the_same_are_offered_and_the_person_decides(workdir, no_repeats):
    label, sounds, ids = _same_mistake_everywhere(workdir)
    d = _desk(workdir)
    result = d.act(dict(kind="pitch", id=ids[0], pitch=sounds, how="guess", rank=1))
    offer = result["offer"]
    assert (offer["type"], offer["seed"], offer["label"], offer["pitch"]) == ("alike", ids[0], label, sounds)
    assert set(offer["ids"]) == set(ids[1:])  # the others with the same wrong label, and nothing else
    d.act(dict(kind="pitches", ids=offer["ids"], pitch=sounds, how="offer", seed=ids[0]))
    assert all(d.note(i)["pitch"] == sounds and d.note(i)["state"] == "fixed" for i in ids)
    assert d.act(dict(kind="undo"))["undone"] == "pitches" and d.note(ids[1])["pitch"] == label
    d.act(dict(kind="refuse", ids=offer["ids"], pitch=sounds, seed=ids[0]))  # logged, nothing changes
    log = [json.loads(line) for line in open(os.path.join(workdir, "mix", "log.jsonl"))]
    assert log[-1]["kind"] == "refuse" and d.note(ids[1])["pitch"] == label


def test_what_the_person_played_past_is_not_offered_and_tells_right_from_wrong(workdir, no_repeats):
    d = _desk(workdir)
    pitches = [n["pitch"] for n in d.notes]
    common = max(set(pitches), key=pitches.count)
    same = [n for n in d.notes if n["pitch"] == common]
    assert len(same) >= 6
    last = max(d.notes, key=lambda n: n["start"])
    d.act(dict(kind="pitch", id=last["id"], pitch=last["pitch"] + 1, offer=False))  # they are at work on the song
    d.mark_checked(0, same[3]["start"] + 0.01)  # four of them played past: found right
    odd = same[4]
    result = d.act(dict(kind="pitch", id=odd["id"], pitch=common + 7, how="key"))  # one "fixed" all the same
    # the rest sound like the ones found right at least as much as like this one: nothing to offer
    assert "offer" not in result
    assert d.alike(same[5]["id"]) == (None, [])  # and a note that was not changed has no look-alikes to fix


# -------------------------------------------------------------------------------------------------- repeats


def _bars_of_a_pattern(d):
    """The bars of the first pattern with three bars or more, and the notes that cover their ninth sixteenth."""
    rep = d.repeats
    family = next(f for f in range(int(rep.family.max()) + 1) if (rep.family == f).sum() >= 3)
    bars = [int(b) for b in np.flatnonzero(rep.family == family)]
    owner = rep.tokens(d.notes)[2]
    return bars, [d.notes[owner[b, 8]]["id"] for b in bars]


def _write_wrong(workdir, ids, shift=7):
    """As if the tool had written those notes a fifth too high."""
    for name in ("machine.json", "notes.json"):
        path = os.path.join(workdir, "mix", name)
        data = json.load(open(path))
        for n in data["notes"]:
            if n["id"] in ids:
                n["pitch"] += shift
        json.dump(data, open(path, "w"))


def test_the_songs_bars_are_sorted_into_patterns_from_the_sound(workdir):
    d = _desk(workdir)
    rep = d.repeats
    assert os.path.exists(os.path.join(workdir, "mix", "repeats.npz"))  # measured once, then kept
    assert rep.bars >= 12 and (rep.family == rep.family.max()).sum() >= 3 and rep.family.max() == 3  # four chords
    shown = d.public()["repeats"]
    assert len(shown["bars"]) == rep.bars and shown["differences"] == []
    bars, _ = _bars_of_a_pattern(d)
    assert shown["bars"][bars[0]] == dict(pattern="A", of=bars[1:]) and shown["bars"][0] == dict(pattern=None, of=[])
    again = _desk(workdir).repeats
    assert (again.family == rep.family).all() and (again.related == rep.related).all()


def test_a_fix_is_offered_to_the_bars_that_sound_the_same_and_are_written_as_this_one_was(workdir):
    bars, ids = _bars_of_a_pattern(_desk(workdir))
    _write_wrong(workdir, ids)
    d = _desk(workdir)
    start = json.dumps(sorted(d.notes, key=lambda n: n["id"]))
    right = d.note(ids[0])["pitch"] - 7
    offer = d.act(dict(kind="pitch", id=ids[0], pitch=right, how="drag"))["offer"]
    assert offer["type"] == "repeat" and offer["bars"] == len(bars) - 1
    assert offer["items"] == [dict(source=bars[0], bar=b, k0=8, k1=11) for b in bars[1:]]
    result = d.act(dict(kind="repeats", items=offer["items"], how="offer"))
    assert "offer" not in result  # what was carried is not carried again
    assert all(d.note(i)["pitch"] == right and d.note(i)["state"] == "fixed" for i in ids)
    assert all(d.note(i)["was"]["pitch"] == right + 7 for i in ids) and len(d.notes) == len(d.machine["notes"])
    assert copilot.mistakes(workdir, "mix")["kinds"] == {"wrong pitch": len(bars)}
    assert d.differences() == []
    assert [d.act(dict(kind="undo"))["undone"] for _ in range(2)] == ["repeats", "pitch"]
    assert json.dumps(sorted(d.notes, key=lambda n: n["id"])) == start
    # a bar the person has been through is theirs: it is neither in the offer nor marked
    d.mark_checked(0, d.repeats.slot_time(bars[0] * 16 + 16) + 0.01)
    offer = d.act(dict(kind="pitch", id=ids[-1], pitch=right, how="drag"))["offer"]  # the last bar fixed first
    assert [i["bar"] for i in offer["items"]] == bars[1:-1]
    marked = d.differences()  # as long as the offer is not taken, the bars in it are marked
    assert [(m["bar"], m["k0"], m["k1"], m["source"], m["why"]) for m in marked] == [
        (b, 8, 11, bars[-1], "fixed") for b in bars[1:-1]
    ]
    ghost = marked[0]["notes"]  # what the repeat has there, laid where it would be in this bar
    assert [n["pitch"] for n in ghost] == [right] and abs(ghost[0]["start"] - d.note(ids[1])["start"]) < 0.03
    d.act(dict(kind="refuse", items=offer["items"]))  # an offer turned down is worth knowing, and changes nothing
    assert json.loads(open(os.path.join(workdir, "mix", "log.jsonl")).read().splitlines()[-1])["kind"] == "refuse"


def test_a_difference_is_taken_from_the_repeat_or_left_as_written(workdir):
    bars, ids = _bars_of_a_pattern(_desk(workdir))
    _write_wrong(workdir, ids[:2])  # two bars wrong, the third right
    d = _desk(workdir)
    assert [(m["bar"], m["why"]) for m in d.differences()] == [(bars[2], "most")]  # two against one, and wrong
    right = d.note(ids[2])["pitch"]
    d.act(dict(kind="pitch", id=ids[0], pitch=right, how="drag", offer=False))
    marked = d.differences()  # what the person fixed is the word now: the other wrong bar is marked, the right one not
    assert [(m["bar"], m["source"], m["why"]) for m in marked] == [(bars[1], bars[0], "fixed")]
    m = marked[0]
    kept = d.act(dict(kind="keep", start=m["start"], end=m["end"]))
    assert kept == {} and d.differences() == [] and d.state["kept"] == [[m["start"], m["end"]]]
    assert d.act(dict(kind="undo"))["undone"] == "keep" and d.state["kept"] == [] and len(d.differences()) == 1
    d.act(
        dict(kind="repeats", items=[dict(source=m["source"], bar=m["bar"], k0=m["k0"], k1=m["k1"])], how="difference")
    )
    assert d.note(ids[1])["pitch"] == right and d.note(ids[1])["state"] == "fixed" and d.differences() == []
    for bad in (
        dict(source=0, bar=0, k0=0, k1=3),
        dict(source=bars[0], bar=99, k0=0, k1=3),
        dict(source=1, bar=2, k0=5, k1=2),
    ):
        with pytest.raises(Bass2SheetError, match="no such stretch"):
            d.act(dict(kind="repeats", items=[bad]))
    with pytest.raises(Bass2SheetError, match="nothing to take"):
        d.act(dict(kind="repeats", items=[]))


def test_one_bar_is_made_to_read_as_another_over_any_stretch_and_the_rest_is_left_alone(workdir):
    d = _desk(workdir)
    rep = d.repeats
    rng = np.random.default_rng(7)
    pairs = [(int(i), int(j)) for i in range(rep.bars) for j in np.flatnonzero(rep.related[i])]
    for trial in range(40):
        made = 0
        for _ in range(int(rng.integers(0, 5))):  # first some fixes of the person's, anywhere
            n = d.notes[int(rng.integers(len(d.notes)))]
            what = int(rng.integers(4))
            try:
                if what == 0:
                    d.act(dict(kind="pitch", id=n["id"], pitch=n["pitch"] + int(rng.integers(1, 8)), offer=False))
                elif what == 1:
                    d.act(dict(kind="delete", id=n["id"], offer=False))
                elif what == 2:
                    d.act(dict(kind="split", id=n["id"], at=(n["start"] + n["end"]) / 2, offer=False))
                else:
                    d.act(dict(kind="merge", id=n["id"], offer=False))
                made += 1
            except Bass2SheetError:
                pass
        i, j = pairs[int(rng.integers(len(pairs)))]
        k0 = int(rng.integers(16))
        k1 = int(rng.integers(k0, 16))
        before_json = json.dumps(sorted(d.notes, key=lambda n: n["id"]), sort_keys=True)
        p0, s0, _ = rep.tokens(d.notes)
        d.act(dict(kind="repeats", items=[dict(source=i, bar=j, k0=k0, k1=k1)]))
        p1, s1, _ = rep.tokens(d.notes)
        ks = slice(k0, k1 + 1)
        where = f"trial {trial}: bar {i} to bar {j}, sixteenths {k0} to {k1}"
        assert p1[j, ks].tolist() == p0[i, ks].tolist(), where  # the stretch reads as the repeat does
        assert s1[j, k0 + 1 : k1 + 1].tolist() == s0[i, k0 + 1 : k1 + 1].tolist(), where
        assert s1[j, k0] or not s0[i, k0] or not p0[i, k0], where  # a note that starts there starts here too
        outside = np.ones(p0.shape, bool)
        outside[j, ks] = False
        assert (p1[outside] == p0[outside]).all(), where  # nothing else moved
        after = (j * 16 + k1 + 1) if k1 < 15 or j + 1 < rep.bars else None
        changed = np.flatnonzero((s1 != s0).reshape(-1) & outside.reshape(-1)).tolist()
        assert changed in ([], [after]), where  # but a note cut at the end of the stretch starts there now
        order = sorted(d.notes, key=lambda n: n["start"])
        assert all(n["end"] - n["start"] >= copilot.SHORTEST - 1e-6 for n in order), where
        assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(order, order[1:])), where  # no two at once
        assert len({n["id"] for n in order}) == len(order), where
        assert d.act(dict(kind="undo"))["undone"] == "repeats"
        assert json.dumps(sorted(d.notes, key=lambda n: n["id"]), sort_keys=True) == before_json, where
        for _ in range(made):
            d.act(dict(kind="undo"))
    assert [n["state"] for n in d.notes] == ["machine"] * len(d.machine["notes"])


def test_a_fix_that_fails_leaves_nothing_half_done(workdir):
    bars, ids = _bars_of_a_pattern(_desk(workdir))
    d = _desk(workdir)
    start = json.dumps(d.notes)
    good = dict(source=bars[0], bar=bars[1], k0=0, k1=15)
    d.act(dict(kind="pitch", id=ids[0], pitch=60, offer=False))
    middle = json.dumps(d.notes)
    with pytest.raises(Bass2SheetError):
        d.act(dict(kind="repeats", items=[good, dict(source=bars[0], bar=99, k0=0, k1=3)]))  # the second cannot be
    assert json.dumps(d.notes) == middle and d.public()["undo"] == 1
    assert json.dumps(_desk(workdir).notes) == middle
    d.act(dict(kind="undo"))
    assert json.dumps(d.notes) == start


def test_listening_ahead_is_not_working_through(workdir):
    d = _desk(workdir)
    length = d.machine["length"]
    d.mark_checked(0, length)  # the whole song heard, and nothing fixed
    assert d.reached() is None and d.worked() == [] and d.public()["checked"] == [[0.0, length]]
    assert copilot.mistakes(workdir, "mix")["right"] == 0
    early = next(n for n in d.notes if n["start"] > 5)
    d.act(dict(kind="pitch", id=early["id"], pitch=early["pitch"] + 12, offer=False))
    assert d.reached() == early["start"] and d.worked() == [[0.0, early["start"]]]  # as far as the fix, no further
    m = copilot.mistakes(workdir, "mix")
    assert (
        m["kinds"] == {"wrong octave": 1} and 0 < m["right"] < 20 and m["checked_seconds"] == round(early["start"], 1)
    )
    later = next(n for n in d.notes if n["start"] > 12)
    d.act(dict(kind="delete", id=later["id"], offer=False))  # a note taken out further on: they have worked to there
    assert d.reached() == later["start"]
    d.act(dict(kind="upto", time=20.0))  # "I have checked it up to here"
    assert d.reached() == 20.0 and d.worked() == [[0.0, 20.0]] and d.public()["worked"] == [[0.0, 20.0]]
    assert copilot.mistakes(workdir, "mix")["right"] > m["right"]
    assert d.act(dict(kind="undo"))["undone"] == "upto" and d.reached() == later["start"]
    assert d.act(dict(kind="undo"))["undone"] == "delete" and d.reached() == early["start"]
    with pytest.raises(Bass2SheetError, match="not a moment"):
        d.act(dict(kind="upto", time=-3))


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
    written = lambda: open(copilot.sheet(workdir, "mix", mscore="/x")["musicxml"]).read()
    assert "the rest is the machine" in written()  # played to the end, but worked through only as far as the fixes
    d.act(dict(kind="upto", time=d.machine["length"]))
    assert "Bass: checked by ear against the record<" in written()


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
    assert song["repeats"]["bars"][1]["pattern"] == "A" and song["repeats"]["differences"] == []
    assert [d["why"] for d in r["differences"]] == ["fixed", "fixed"]  # its two repeats are written differently now
    assert r["differences"][0]["notes"][0]["pitch"] == n["pitch"] + 12
    assert _post(url + "/api/checked", dict(song="mix", start=0, end=5.5)) == dict(
        checked=[[0.0, 5.5]], worked=[[0.0, n["start"]]], differences=r["differences"]
    )  # played to 5.5 s, worked through as far as the fix
    assert r["worked"] == [] and song["worked"] == []
    undone = _post(url + "/api/act", dict(song="mix", action=dict(kind="undo")))
    assert undone["fixes"] == 0 and undone["differences"] == []
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
    status, headers, body = _get(url + "/data/mix/audio/bass-70.flac", {"Range": "bytes=0-3"})  # made when asked for
    assert (status, body) == (206, b"fLaC") and os.path.exists(os.path.join(workdir, "mix", "audio", "bass-70.flac"))
    for path in (
        "/data/mix/notes.json",
        "/data/mix/../songs.json",
        "/data/other/picture.bin",
        "/api/song?id=other",
        "/data/mix/audio/bass-75.flac",
        "/data/mix/audio/drums-80.flac",
    ):
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
