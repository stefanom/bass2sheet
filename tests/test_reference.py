"""A transcription lined up with a recording (reference.py), on the synthetic song: the file is made from the
song's own notes, at another tempo, in another key, after a lead-in, and has to be found back."""

import json
import os

import numpy as np
import pretty_midi
import pytest

from bass2sheet import copilot, reference
from bass2sheet.common import SR, Bass2SheetError
from bass2sheet.score import find_mscore

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


def _file(song, faster=1.0, lead=0.0, up=0, leave_out=None, drums=True):
    """The song as a file: its bass line (and a drum part on its beats), `faster` times the song's tempo, after
    `lead` seconds, `up` semitones higher; `leave_out` is a span of the song, in seconds, that the file skips."""
    gone = (leave_out[1] - leave_out[0]) if leave_out else 0.0

    def when(t):
        return lead + (t - (gone if leave_out and t >= leave_out[1] else 0.0)) / faster

    pm = pretty_midi.PrettyMIDI(initial_tempo=song["tempo"] * faster)
    bass = pretty_midi.Instrument(program=33, name="Bass")
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    kept = []
    for s, e, p in song["notes"]:
        if leave_out and leave_out[0] <= s < leave_out[1]:
            continue
        bass.notes.append(pretty_midi.Note(100, p + up, when(s), when(e)))
        kept.append((s, e, p))
    for b in song["beats"]:
        if not (leave_out and leave_out[0] <= b < leave_out[1]):
            kit.notes.append(pretty_midi.Note(100, 36, when(b), when(b) + 0.05))
    pm.instruments += [kit, bass] if drums else [bass]
    return pm, kept


def _mono(song):
    return song["mix"].mean(0)


def test_a_file_at_another_tempo_in_another_key_is_found_back(song):
    pm, kept = _file(song, faster=1.12, lead=2.5, up=3)
    al = reference.align(pm, _mono(song), song["sr"])
    assert al.shift == -3 and al.ratio == pytest.approx(1.12, rel=0.02)
    assert len(al.segments) == 1 and al.segments[0]["like"] > 0.3
    notes = al.notes(pm, 1)
    assert len(notes) >= 0.95 * len(kept)
    true = {round(s, 3): p for s, _, p in kept}
    starts = np.array(sorted(true))
    errors = []
    for s, e, p in notes:
        nearest = starts[np.argmin(np.abs(starts - s))]
        errors.append(s - nearest)
        assert p == true[round(float(nearest), 3)] and e > s  # the song's own pitches again
    assert np.median(np.abs(errors)) < 0.03 and np.percentile(np.abs(errors), 95) < 0.06, np.round(errors, 3)
    covered = al.covered()
    assert covered[0][0] < kept[0][0] + 0.5 and covered[-1][1] > kept[-1][0] - 0.5
    assert np.isnan(al.at([0.0, 1000.0])).all()  # before the file's first note and after its end: nowhere


def _wandering(seconds=44.0, seed=3):
    """A bass line that never comes back to what it played before, as sound (at SR) and as notes."""
    import librosa

    from bass2sheet.render import pluck

    rng = np.random.default_rng(seed)
    notes, t, p = [], 0.5, 40
    while t < seconds - 1:
        p = int(np.clip(p + rng.choice([-7, -5, -3, -2, 2, 3, 5, 7]), 30, 52))
        notes.append((t, t + 0.36, p))
        t += 0.4
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    for start, end, pitch in notes:
        w = 0.4 * pluck(librosa.midi_to_hz(pitch), end - start, 100)
        y[int(start * SR) : int(start * SR) + len(w)] += w
    return y, notes


def test_what_the_file_leaves_out_is_jumped_over_not_stretched():
    y, played = _wandering()
    out = (16.0, 24.0)  # eight seconds of the recording are not in the file
    line = dict(notes=played, beats=np.array([]), tempo=150.0)
    pm, kept = _file(line, faster=0.95, lead=1.0, leave_out=out, drums=False)
    al = reference.align(pm, y, SR)
    assert len(al.segments) == 2 and al.ratio == pytest.approx(0.95, rel=0.02)
    (a, b), (c, d) = al.covered()
    assert abs(b - out[0]) < 1.0 and abs(c - out[1]) < 1.0  # the recording between them is not spoken of
    notes = al.notes(pm, 0)
    assert not [n for n in notes if out[0] + 0.3 < n[0] < out[1] - 0.3]
    starts = np.array([s for s, _, _ in kept])
    errors = np.array([np.min(np.abs(starts - s)) for s, _, _ in notes])
    assert len(notes) >= 0.9 * len(kept) and np.median(errors) < 0.03 and np.mean(errors < 0.1) > 0.95
    # the other way round: the file has eight seconds the recording lacks
    cut = np.concatenate([y[: int(out[0] * SR)], y[int(out[1] * SR) :]])
    whole, _ = _file(line, faster=1.0, lead=0.0, drums=False)
    al = reference.align(whole, cut, SR)
    assert len(al.segments) == 2
    placed = np.isfinite(al.at([s for s, _, _ in played]))
    lost = [s for (s, _, _), ok in zip(played, placed) if not ok]
    assert all(out[0] - 1.0 < s < out[1] + 1.0 for s in lost[1:-1]) and 15 <= len(lost) <= 26  # those notes: nowhere


def test_two_things_with_nothing_in_common_are_not_lined_up(song):
    rng = np.random.default_rng(1)
    pm = pretty_midi.PrettyMIDI(initial_tempo=90)
    other = pretty_midi.Instrument(program=33, name="Bass")
    for k in range(120):  # a line that wanders at random
        other.notes.append(pretty_midi.Note(100, int(rng.integers(28, 52)), 0.37 * k, 0.37 * k + 0.3))
    pm.instruments.append(other)
    try:
        al = reference.align(pm, _mono(song), song["sr"])
    except Bass2SheetError as e:
        assert "nothing in common" in str(e)
    else:  # or only a short stretch that happens to fit
        assert sum(b - a for a, b in al.covered()) < 0.5 * len(_mono(song)) / song["sr"]
    with pytest.raises(Bass2SheetError, match="too short"):
        reference.align(pm, _mono(song)[: 5 * song["sr"]], song["sr"])


def test_the_alignment_jumps_where_one_has_a_stretch_the_other_lacks():
    s = np.full((60, 80), -0.3, np.float32)
    for k in range(25):
        s[k, k + 5] = 0.6  # the first 25 frames run side by side
    for k in range(25, 60):
        s[k, k + 20] = 0.6  # ... then the second sequence has 15 frames the first lacks
    pairs, score = reference.side_by_side(s)
    assert pairs[0].tolist() == [0, 5] and pairs[-1].tolist() == [59, 79] and len(pairs) == 60
    assert np.array_equal(pairs[:25, 1] - pairs[:25, 0], np.full(25, 5))
    assert np.array_equal(pairs[25:, 1] - pairs[25:, 0], np.full(35, 20))
    assert score == pytest.approx(60 * 0.6 - reference.JUMP, abs=0.01)
    drift = np.full((40, 44), -0.3, np.float32)
    for k in range(40):
        drift[k, k + k // 10] = 0.6  # the second runs a frame further every ten
    pairs, score = reference.side_by_side(drift)
    assert len(pairs) == 40 and score == pytest.approx(40 * 0.6 - 3 * reference.SLIP, abs=0.01)


def test_the_bass_is_the_part_called_so_or_the_lowest_line():
    def part(program, name, pitches, drum=False):
        inst = pretty_midi.Instrument(program=program, is_drum=drum, name=name)
        inst.notes = [pretty_midi.Note(90, p, 0.5 * k, 0.5 * k + 0.4) for k, p in enumerate(pitches)]
        return inst

    pm = pretty_midi.PrettyMIDI()
    pm.instruments = [
        part(0, "Drums", [36] * 40, drum=True),
        part(87, "BASS&LEAD", [70] * 16),  # called a bass, and far too high to be one
        part(35, "", [30, 37, 40, 42] * 5),
        part(33, "Bass 2", [31, 38] * 4),
        part(25, "Guitar", [52, 55, 59] * 10),
    ]
    assert reference.bass_track(pm) == (2, [3])
    pm.instruments = [part(0, "Drums", [36] * 40, drum=True), part(25, "A", [52, 55] * 10), part(4, "B", [40, 43] * 10)]
    assert reference.bass_track(pm) == (2, [1])  # nobody is called a bass: the lowest line
    pm.instruments = pm.instruments[:1]
    with pytest.raises(Bass2SheetError, match="no pitched part"):
        reference.bass_track(pm)


def test_reading_a_file(tmp_path, monkeypatch):
    with pytest.raises(Bass2SheetError, match="no such file"):
        reference.read(str(tmp_path / "missing.mid"))
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"this is not a MIDI file")
    with pytest.raises(Bass2SheetError, match="not a MIDI file that can be read"):
        reference.read(str(broken))
    tab = tmp_path / "song.gp5"
    tab.write_bytes(b"\x00")
    monkeypatch.setattr("bass2sheet.score.find_mscore", lambda explicit=None: None)
    with pytest.raises(Bass2SheetError, match="MuseScore.*is not installed"):
        reference.read(str(tab))


XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Bass</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type>
</time><clef><sign>F</sign><line>4</line></clef></attributes>{notes}</measure></part></score-partwise>"""


@pytest.mark.skipif(find_mscore() is None, reason="needs MuseScore to turn a score into MIDI")
def test_a_score_is_read_through_musescore(tmp_path):
    steps = [("E", 2), ("G", 2), ("A", 2), ("B", 2)]
    notes = "".join(
        f"<note><pitch><step>{s}</step><octave>{o}</octave></pitch><duration>1</duration><type>quarter</type></note>"
        for s, o in steps
    )
    path = tmp_path / "line.musicxml"
    path.write_text(XML.format(notes=notes))
    pm = reference.read(str(path))
    pitches = [n.pitch for inst in pm.instruments for n in sorted(inst.notes, key=lambda n: n.start)]
    assert pitches == [40, 43, 45, 47]


def test_a_version_against_a_key():
    key = [(1.0, 1.4, 30), (2.0, 2.4, 37), (3.0, 3.4, 40), (4.0, 4.4, 42), (9.0, 9.4, 30)]
    version = [(1.02, 1.4, 30), (2.0, 2.4, 49), (3.05, 3.4, 28), (4.0, 4.4, 43), (6.0, 6.3, 35)]
    a = reference.against(key, version, [[0.0, 5.0]])
    assert a["key_notes"] == 4 and a["notes"] == 4  # the key's note at 9 s and the version's at 6 s are outside
    assert (a["same"], a["higher"], a["lower"], a["other"], a["none"]) == (0.25, 0.25, 0.25, 0.25, 0.0)
    assert a["f_100ms"] == 0.25 and a["f_note_name"] == 0.75
    everywhere = reference.against(key, version, [[0.0, 10.0]])
    assert everywhere["key_notes"] == 5 and everywhere["none"] == 0.2
    assert reference.against(key, [], [[0.0, 5.0]])["none"] == 1.0
    assert reference._common([[0, 5], [8, 12]], [[4, 9]]) == [[4, 5], [8, 9]]


# ------------------------------------------------------------------------------------------ in the copilot


@pytest.fixture(scope="module")
def prepared(song_files, tmp_path_factory):
    mix, stems = song_files
    work = str(tmp_path_factory.mktemp("scores"))
    opts = dict(defaults=dict(beat_tracker="librosa"), crepe_model=CREPE, device="cpu", cache_dir=work + "-cache")
    copilot.add([mix], work, opts, stems_dirs={mix: stems}, say=lambda text: None)
    return work


def test_a_file_becomes_an_answer_key_of_a_song(prepared, song, tmp_path, capsys):
    pm, kept = _file(song, faster=1.05, lead=2.0, up=12)  # written an octave above its sound, as many are
    path = str(tmp_path / "Some Tab.mid")
    pm.write(path)
    name, key = reference.add(prepared, "mix", path)
    assert (
        name == "some-tab"
        and key["file"] == "Some Tab.mid"
        and key["part"] == dict(index=1, name="Bass", notes=len(kept))
    )
    assert key["shift"] == 0 and key["octave"] == -12 and key["octave_agreement"] > 0.8  # brought down to the sound
    assert len(key["notes"]) >= 0.95 * len(kept) and {p for _, _, p in key["notes"]} <= {p for _, _, p in kept}
    assert list(reference.keys(prepared, "mix")) == ["some-tab"]
    (row,) = reference.compare(prepared, "mix")
    assert row["name"] == "some-tab" and row["worked"] is None  # nothing worked through yet
    assert (
        row["tool"]["f_100ms"] >= 0.9 and row["tool"]["same"] >= 0.9 and row["tool"]["key_notes"] == len(key["notes"])
    )
    d = copilot.Desk(prepared, "mix")  # the person works on the first ten seconds and gets a note wrong on purpose
    early = next(n for n in d.notes if 4 < n["start"] < 9)
    d.act(dict(kind="pitch", id=early["id"], pitch=early["pitch"] + 5, offer=False))
    d.mark_checked(0, 10)
    (row,) = reference.compare(prepared, "mix")
    w = row["worked"]
    assert w["spans"][0][1] == pytest.approx(early["start"], abs=0.01)  # as far as they have worked
    assert w["yours"]["key_notes"] == w["tool"]["key_notes"] > 3 and w["yours"]["same"] < w["tool"]["same"]
    text = reference.report(prepared)
    assert "# Against the answer keys you gave" in text and '**Some Tab.mid**: its part "Bass"' in text
    assert "moved -1 octave to where the bass sounds" in text and "| Your version, where you have worked |" in text
    assert copilot.main(["reference", "mix", path, "--dir", prepared]) == 0
    out = capsys.readouterr().out
    assert 'Took part 1 "Bass"' in out and "# Against the answer keys you gave" in out
    assert copilot.main(["report", "--dir", prepared]) == 0 and "Some Tab.mid" in capsys.readouterr().out
    assert copilot.main(["reference", "nobody", path, "--dir", prepared]) == 1
    with pytest.raises(Bass2SheetError, match="no part 7"):
        reference.add(prepared, "mix", path, part=7)
    saved = json.load(open(os.path.join(prepared, "mix", "reference", "some-tab.json")))
    assert saved["covered"] == key["covered"] and "when" in saved


def _table(n, m, alike):
    """How alike frame i of a file and frame j of a recording are: 1 for the pairs in `alike`, -0.5 for the rest."""
    s = np.full((n, m), -0.5, np.float32)
    for i, j in alike:
        s[i, j] = 1.0
    return s


def test_the_alignment_read_as_plain_python_does_what_the_compiled_one_does():
    """The alignment is compiled (numba) when it is first used, and what is compiled cannot be followed line by
    line. Here the same function runs as the Python it is written in, on small tables, beside the compiled one."""
    run = reference._side_by_side()
    cases = dict(
        same=_table(30, 30, [(k, k) for k in range(30)]),
        recording_has_more=_table(30, 38, [(i, i if i < 10 else i + 8) for i in range(30)]),
        file_has_more=_table(38, 30, [(i if i < 10 else i + 8, i) for i in range(30)]),
        slower=_table(30, 40, [(i, i + i // 3) for i in range(30)]),  # a frame behind at every third step
        part=_table(30, 60, [(i, i + 25) for i in range(5, 25)]),
        nothing=_table(20, 20, []),
    )
    found = {}
    for name, s in cases.items():
        for jump in (3.0, 10.0):
            pairs, score = run.py_func(s, np.float32(0.75), np.float32(jump))
            compiled, compiled_score = reference.side_by_side(s, 0.75, jump)
            assert pairs.tolist() == compiled.tolist() and float(score) == pytest.approx(compiled_score), (name, jump)
            found[name, jump] = (pairs.tolist(), float(score))
    assert found["same", 10.0] == ([[k, k] for k in range(30)], 30.0)
    # a stretch one of the two lacks is jumped over when the jump is cheap: every frame placed, the jump paid once
    pairs, score = found["recording_has_more", 3.0]
    assert pairs == [[i, i if i < 10 else i + 8] for i in range(30)] and score == 27.0
    pairs, score = found["file_has_more", 3.0]
    assert pairs == [[i if i < 10 else i + 8, i] for i in range(30)] and score == 27.0
    # ... and when it costs more than the short side is worth, only the longer side is kept
    assert found["recording_has_more", 10.0] == ([[i, i + 8] for i in range(10, 30)], 20.0)
    assert found["file_has_more", 10.0] == ([[i + 8, i] for i in range(10, 30)], 20.0)
    pairs, score = found["slower", 10.0]  # a recording that runs slower is followed, at a cost for each slip
    assert pairs == [[i, i + i // 3] for i in range(30)] and score == 30 - 9 * 0.75
    assert found["part", 10.0] == ([[i, i + 25] for i in range(5, 25)], 20.0)  # a part is found where it lies
    assert found["nothing", 10.0][1] == 0.0 and len(found["nothing", 10.0][0]) == 1


def test_anything_but_midi_goes_through_musescore_and_what_it_cannot_read_is_refused(tmp_path, monkeypatch):
    import stat
    import sys

    from bass2sheet import score

    line = pretty_midi.PrettyMIDI(initial_tempo=100)
    bass = pretty_midi.Instrument(program=33, name="Bass")
    bass.notes = [pretty_midi.Note(100, 40 + k % 4, 0.6 * k, 0.6 * k + 0.5) for k in range(8)]
    line.instruments.append(bass)
    as_midi = str(tmp_path / "what musescore makes of it.mid")
    line.write(as_midi)
    program = str(tmp_path / "mscore")
    with open(program, "w") as f:  # a stand-in: `mscore -o OUT.mid FILE`
        f.write(
            f"#!{sys.executable}\nimport shutil, sys\n"
            "assert sys.argv[1] == '-o' and sys.argv[2].endswith('.mid'), sys.argv\n"
            "if open(sys.argv[3], 'rb').read(4) != b'FICH':\n"
            "    sys.exit('this is not a Guitar Pro file')\n"
            f"shutil.copy({as_midi!r}, sys.argv[2])\n"
        )
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", [program])
    tab = tmp_path / "song.gp5"
    tab.write_bytes(b"FICHIER GUITAR PRO v5.00")
    pm = reference.read(str(tab))
    assert [n.pitch for n in pm.instruments[0].notes] == [40, 41, 42, 43] * 2 and reference.bass_track(pm)[0] == 0
    other = tmp_path / "notes.gp5"
    other.write_bytes(b"something else")
    with pytest.raises(Bass2SheetError, match="MuseScore could not read notes.gp5: this is not a Guitar Pro file"):
        reference.read(str(other))


def test_a_song_without_an_answer_key_has_nothing_to_compare_or_report(prepared, tmp_path):
    import shutil

    work = str(tmp_path / "scores")
    shutil.copytree(prepared, work)
    shutil.rmtree(os.path.join(work, "mix", "reference"), ignore_errors=True)
    assert reference.keys(work, "mix") == {} and reference.compare(work, "mix") == [] and reference.report(work) == ""
    os.makedirs(os.path.join(work, "mix", "reference"))  # the folder is there and holds no key
    open(os.path.join(work, "mix", "reference", "notes.txt"), "w").write("not a key")
    assert reference.keys(work, "mix") == {} and reference.compare(work, "mix") == [] and reference.report(work) == ""
