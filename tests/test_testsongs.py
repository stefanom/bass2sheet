"""The test songs in the copilot (testsongs.py): added from a dataset with their scores, heard and rated on the
page, and the scores recounted by the person's answers."""

import json
import os
import threading
import urllib.error
import urllib.request

import pretty_midi
import pytest
import soundfile as sf

from bass2sheet import copilot, copilot_server, testsongs
from bass2sheet.common import Bass2SheetError

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


@pytest.fixture(scope="module")
def dataset(song, song_files, tmp_path_factory):
    """A dataset in Slakh's layout holding the synthetic song twice, the runs of two note finders on it, and the
    song's stems (so nothing has to be separated)."""
    root = tmp_path_factory.mktemp("dataset")
    mix, stems = song_files
    y, sr = sf.read(mix, dtype="float32", always_2d=True)
    for name, patch in (("Track00001", "classic_bass.nkm"), ("Track00002", "scarbee_rickenbacker_bass.nkm")):
        track = root / name
        (track / "MIDI").mkdir(parents=True)
        (track / "stems").mkdir()
        sf.write(str(track / "mix.wav"), y, sr)
        sf.write(str(track / "stems" / "S00.wav"), y.mean(1), sr)
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=33)
        inst.notes = [pretty_midi.Note(100, p, s, e) for s, e, p in song["notes"]]
        pm.instruments.append(inst)
        pm.write(str(track / "MIDI" / "S00.mid"))
        pm.write(str(track / "all_src.mid"))
        (track / "metadata.yaml").write_text(
            f"stems:\n  S00:\n    inst_class: Bass\n    is_drum: false\n    plugin_name: {patch}\n    program_num: 33\n"
        )
    runs = {}
    for label, scores in (("bass2sheet", (0.9, 0.5)), ("MuScriptor", (0.95, 0.7))):
        run = tmp_path_factory.mktemp(label)
        rows = [dict(track=f"Track0000{k + 1}", condition="mix", f=f, f_100ms=1.0) for k, f in enumerate(scores)]
        rows.append(dict(track="Track00001", condition="stem", f=0.99))  # the clean stem's score is not the mix's
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        runs[label] = str(run)
    wavs = {str(root / n / "mix.wav"): stems for n in ("Track00001", "Track00002")}
    return str(root), runs, wavs


@pytest.fixture(scope="module")
def added(dataset, tmp_path_factory):
    root, runs, stems = dataset
    work = str(tmp_path_factory.mktemp("tests"))
    opts = dict(defaults=dict(beat_tracker="librosa"), crepe_model=CREPE, device="cpu", cache_dir=work + "-cache")
    testsongs.add_set(root, work, runs=runs, name="Tiny", opts=opts, stems_dirs=stems, say=lambda text: None)
    return work


@pytest.fixture
def workdir(added, tmp_path):
    import shutil

    out = str(tmp_path / "tests")
    shutil.copytree(added, out)
    return out


def test_a_datasets_songs_are_added_with_what_is_known_of_them(added, song):
    tests = testsongs.load(added)
    assert list(tests["sets"]) == ["Tiny"] and tests["sets"]["Tiny"]["measure"] == "f"
    assert "sampled instruments" in tests["sets"]["Tiny"]["about"]
    assert list(tests["songs"]) == ["tiny-track00001", "tiny-track00002"]
    one = tests["songs"]["tiny-track00001"]
    assert one["set"] == "Tiny" and one["track"] == "Track00001" and one["facts"] == ["sampled bass: classic bass"]
    assert one["scores"] == dict(bass2sheet=0.9, MuScriptor=0.95)  # from the mix, not from the clean stem
    assert tests["songs"]["tiny-track00002"]["scores"] == dict(bass2sheet=0.5, MuScriptor=0.7)
    assert one["length"] == pytest.approx(song["mix"].shape[1] / song["sr"], abs=0.01)
    assert 0 <= one["busy"] < one["length"] - 15
    # each is an ordinary song of the working directory: it opens in the copilot
    songs = copilot.load_songs(added)
    assert songs["tiny-track00001"]["test"] is True and songs["tiny-track00001"]["title"] == "Track00001"
    assert len(copilot.Desk(added, "tiny-track00002").notes) > 20
    assert [s["id"] for s in copilot_server.songs(added)] == ["tiny-track00001", "tiny-track00002"]


def test_where_listening_starts_is_the_busiest_stretch():
    notes = [dict(start=s) for s in (1, 2, 50, 51, 52, 53, 54, 90)]
    assert testsongs.busy(notes) == 49.0 and testsongs.busy([]) == 0.0 and testsongs.busy([dict(start=0.2)]) == 0.0


def test_what_a_database_says_of_a_song_is_put_in_a_few_words():
    about = dict(genre="J-pop", tempo="135.0", live="Gt, Bs, Dr", drums="Live drums")
    assert testsongs.facts("rwc", dict(about=about)) == [
        "J-pop",
        "135 bpm",
        "bass played live",
        "played live: drums, guitar",
        "live drums",
    ]
    sequenced = testsongs.facts("rwc", dict(about=dict(about, live="", drums="Drum sequences")))
    assert sequenced[2:] == [
        "bass not among the instruments played live",
        "no instrument played live",
        "drum sequences",
    ]
    assert testsongs.facts("slakh", dict(bass=[dict(patch=None)])) == [] and testsongs.facts("mdb", dict()) == []


def test_the_scores_are_recounted_by_the_persons_answers(workdir):
    s = testsongs.state(workdir)
    assert s["labels"] == ["bass2sheet", "MuScriptor"] and (s["rated"], s["total"]) == (0, 2)
    by = lambda s: {
        g["rating"]: (g["songs"], g["scores"]["bass2sheet"], g["scores"]["MuScriptor"]) for g in s["summary"]
    }
    assert by(s) == {
        "yes": (0, None, None),
        "partly": (0, None, None),
        "no": (0, None, None),
        None: (2, 0.7, 0.825),
        "all": (2, 0.7, 0.825),
    }
    assert [g["words"] for g in s["summary"]] == [
        "Like my music",
        "Partly",
        "Not like it",
        "Not rated yet",
        "All of them",
    ]
    assert testsongs.rate(workdir, "tiny-track00002", rating="yes")["rating"] == "yes"
    testsongs.rate(workdir, "tiny-track00001", rating="no", note="  too clean, no band  ")
    s = testsongs.state(workdir)
    assert by(s) == {
        "yes": (1, 0.5, 0.7),
        "partly": (0, None, None),
        "no": (1, 0.9, 0.95),
        None: (0, None, None),
        "all": (2, 0.7, 0.825),
    }
    assert s["rated"] == 2 and s["songs"]["tiny-track00001"]["note"] == "too clean, no band"
    testsongs.rate(workdir, "tiny-track00001", note="")  # the note taken back: the answer stays
    assert testsongs.ratings(workdir)["tiny-track00001"]["rating"] == "no"
    testsongs.rate(workdir, "tiny-track00001", rating=None)  # and the answer: nothing is left of it
    assert "tiny-track00001" not in testsongs.ratings(workdir) and testsongs.state(workdir)["rated"] == 1
    log = [json.loads(line) for line in open(os.path.join(workdir, "ratings.jsonl"))]
    assert [(e["song"][-1], e.get("rating", "-"), e.get("note", "-")) for e in log] == [
        ("2", "yes", "-"),
        ("1", "no", "  too clean, no band  "),
        ("1", "-", ""),
        ("1", None, "-"),
    ]
    for bad in (dict(rating="maybe"), dict(note=7), dict(stars=3), dict()):
        with pytest.raises(Bass2SheetError):
            testsongs.rate(workdir, "tiny-track00001", **bad)
    with pytest.raises(Bass2SheetError, match="no test song"):
        testsongs.rate(workdir, "mix", rating="yes")


def test_the_report_counts_the_scores_on_the_songs_that_are_like_the_persons(workdir, capsys):
    testsongs.rate(workdir, "tiny-track00002", rating="yes", note="this is my kind of bass")
    text = testsongs.report(workdir)
    assert "2 test songs (2 from Tiny). You have rated 1." in text
    assert "| | Songs | bass2sheet | MuScriptor |" in text and "| Like my music | 1 | 0.50 | 0.70 |" in text
    assert "| Not like it | 0 | – | – |" in text and "| All of them | 2 | 0.70 | 0.82 |" in text
    assert "- Track00002 (Track00002, like my music, " in text and "this is my kind of bass" in text
    assert copilot.main(["report", "--dir", workdir]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# The test songs") and "# What the tool got wrong" in out
    assert "## Track00001" not in out  # a test song nobody has worked on is not listed among the mistakes
    d = copilot.Desk(workdir, "tiny-track00001")
    d.act(dict(kind="delete", id=d.notes[3]["id"], offer=False))
    assert "## Track00001" in copilot.report(workdir)
    assert testsongs.report(str(os.path.dirname(workdir))) == ""  # a directory with no test songs: nothing


def test_adding_takes_the_first_runs_songs_or_the_ones_named(dataset, tmp_path):
    root, runs, stems = dataset
    with pytest.raises(Bass2SheetError, match="none of the songs asked for"):
        testsongs.add_set(root, str(tmp_path / "a"), runs=runs, only=["Track00009"])
    with pytest.raises(Bass2SheetError, match="not a bass2sheet-eval run directory"):
        testsongs.add_set(root, str(tmp_path / "b"), runs=dict(x=str(tmp_path)))
    assert copilot.main(["testset", root, "--scores", "nothing", "--dir", str(tmp_path / "c")]) == 1


# --------------------------------------------------------------------------------------------------- the page


@pytest.fixture
def server(workdir):
    srv = copilot_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_the_page_lists_the_songs_plays_them_and_takes_answers(server):
    url, workdir = server
    status, headers, page = _get(url + "/tests")
    assert (
        status == 200 and b"Like the music you care about?" in page and headers["Content-Type"].startswith("text/html")
    )
    s = json.loads(_get(url + "/api/tests")[2])
    assert [x["name"] for x in s["sets"]] == ["Tiny"] and s["sets"][0]["songs"] == [
        "tiny-track00001",
        "tiny-track00002",
    ]
    row = s["songs"]["tiny-track00001"]
    assert (
        row["title"] == "Track00001"
        and row["rating"] is None
        and row["note"] == ""
        and row["scores"]["MuScriptor"] == 0.95
    )
    for kind in ("record", "bass"):  # what the page plays
        status, headers, body = _get(f"{url}/data/tiny-track00001/audio/{kind}.flac")
        assert status == 200 and headers["Content-Type"] == "audio/flac" and body[:4] == b"fLaC"
    s = _post(url + "/api/rate", dict(song="tiny-track00001", rating="partly"))
    assert s["songs"]["tiny-track00001"]["rating"] == "partly" and s["rated"] == 1
    s = _post(url + "/api/rate", dict(song="tiny-track00001", note="synth bass"))
    assert s["songs"]["tiny-track00001"] == dict(row, rating="partly", note="synth bass")
    assert testsongs.ratings(workdir)["tiny-track00001"]["note"] == "synth bass"
    for bad in (dict(song="nobody", rating="yes"), dict(song="tiny-track00001", rating="five")):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(url + "/api/rate", bad)
        assert e.value.code == 400 and "error" in json.loads(e.value.read())
    assert b"<html" in _get(url + "/song")[2]  # where the page's "Open" link leads


def test_the_pages_own_logic():
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node is not installed")
    script = os.path.join(os.path.dirname(__file__), "testsongs_page_logic.js")
    out = subprocess.run(["node", script, testsongs.PAGE_PATH], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
