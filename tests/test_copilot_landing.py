"""The copilot's first page: the songs and how far along each is, and a new song coming in from the browser, kept
under the working directory and prepared in the background (synthetic song, no models to fetch)."""

import http.client
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from bass2sheet import copilot, copilot_adding, copilot_server
from bass2sheet.common import SR, Bass2SheetError, listening, step

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


@pytest.fixture(scope="module")
def prepared(taste_song_files, cache_dir, tmp_path_factory):
    """A working directory with the synthetic song prepared, made once. Its settings.json keeps what a song
    added later is prepared with: the tests' cache, the CPU, the small pitch model, the built-in beat tracker."""
    mix, stems = taste_song_files
    workdir = str(tmp_path_factory.mktemp("scores"))
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    copilot.add([mix], workdir, opts, stems_dirs={mix: stems}, say=lambda _: 0)
    return workdir


@pytest.fixture
def served(prepared, taste_song_files, tmp_path):
    """(address, working directory, the song's audio file) of a server on a fresh copy of the prepared directory.
    Every song added there is given the synthetic song's stems, so nothing has to be separated."""
    workdir = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, workdir)
    srv = copilot_server.make_server(workdir, 0, stems_dir=taste_song_files[1])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir, taste_song_files[0]
    srv.RequestHandlerClass.adding.close()
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return dict(r.headers), r.read()


def _json(url):
    return json.loads(_get(url)[1])


def _post(url, data=None, body=None, headers=None):
    """(status, what the server answered) of a POST: JSON `data`, or the bytes `body`."""
    sent = json.dumps(data).encode() if body is None else body
    req = urllib.request.Request(url, sent, dict(headers or {}), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _upload(url, name, body, notes="own", headers=None):
    query = urllib.parse.urlencode(dict(name=name, notes=notes))
    return _post(f"{url}/api/upload?{query}", body=body, headers=headers)


def _wav(y):
    out = io.BytesIO()
    sf.write(out, y, SR, format="WAV", subtype="PCM_16")
    return out.getvalue()


def _settled(url, song_id, timeout=180):
    """Watch a song being added until it is ready or has failed. Returns (its job, the steps seen running)."""
    seen, until = [], time.time() + timeout
    while time.time() < until:
        job = next((j for j in _json(url + "/api/adding")["jobs"] if song_id in (j["id"], j["key"])), None)
        assert job is not None, "the song is not in the row"
        seen += [s["id"] for s in job["steps"] if s["state"] == "running" and s["id"] not in seen]
        if job["state"] in ("done", "failed"):
            return job, seen
        time.sleep(0.1)
    raise AssertionError(f"{song_id} was not prepared in {timeout} s: {job}")


# ----------------------------------------------------------------------------------------------- the two pages


def test_the_songs_are_at_the_root_and_one_song_has_its_own_page(served):
    url, workdir, _ = served
    landing, page = _get(url + "/")[1].decode(), _get(url + "/song")[1].decode()
    assert 'id="songs"' in landing and 'id="drop"' in landing and 'id="roll"' not in landing
    assert 'location.replace("/song" + location.hash)' in landing  # an old link to /#song still opens the song
    assert "rights to use" in landing and "goes nowhere else" in landing
    assert 'id="roll"' in page and 'href="/"' in page
    assert _get(url + "/song")[1] == open(copilot_server.PAGE_PATH, "rb").read()


def test_the_list_says_how_far_the_person_is_with_each_song(served):
    url, workdir, _ = served
    (song,) = _json(url + "/api/songs")
    assert (song["id"], song["finder"], song["fixes"], song["worked"], song["checked"]) == ("mix", "own", 0, 0, 0)
    assert song["notes"] >= 60 and 25 < song["length"] < 28 and song["spans"] == dict(heard=[], worked=[])
    assert song["score"] is None
    opened = _json(url + "/api/song?id=mix")
    n = opened["notes"][10]
    fix = dict(song="mix", action=dict(kind="pitch", id=n["id"], pitch=n["pitch"] + 12, how="key"))
    assert _post(url + "/api/act", fix)[0] == 200
    assert _post(url + "/api/checked", dict(song="mix", start=0, end=9.0))[0] == 200
    (song,) = _json(url + "/api/songs")
    assert song["fixes"] == 1 and song["checked"] == 9.0 and song["at"] == 9.0
    assert song["worked"] == pytest.approx(n["start"], abs=0.06)  # heard to 9 s, worked through as far as the fix
    assert song["spans"] == dict(heard=[[0.0, 9.0]], worked=[[0.0, n["start"]]])
    desk = copilot.Desk(workdir, "mix")  # the list reads two files; the open song agrees with it
    assert (desk.fixes(), desk.worked()) == (song["fixes"], song["spans"]["worked"])
    assert copilot.progress(desk.machine, desk.state)["reached"] == desk.reached() == n["start"]
    copilot.sheet(workdir, "mix", mscore="/nonexistent/mscore")  # the score is made: the list says when and from what
    made = _json(url + "/api/songs")[0]["score"]
    assert made["fixes"] == 1 and 0 < made["checked"] < 1 and made["when"][:4].isdigit()


def test_the_whole_song_is_shown_in_a_strip_of_its_picture(served):
    url, workdir, _ = served
    headers, body = _get(url + "/api/strip?song=mix&columns=200")
    rows = copilot.PER_SEMI * (copilot.HIGH - copilot.LOW + 1)
    assert (int(headers["X-Rows"]), int(headers["X-Columns"])) == (rows, 200) and len(body) == rows * 200
    whole = np.fromfile(os.path.join(workdir, "mix", "picture.bin"), np.uint8).reshape(rows, -1)
    small = np.frombuffer(body, np.uint8).reshape(rows, 200)
    assert small.max() == whole.max() and small[:, 0].max() == whole[:, : whole.shape[1] // 200].max()
    assert copilot_adding.strip(workdir, "mix", 10**6).shape == whole.shape  # never more than there is
    for bad in ("/api/strip?song=nope&columns=200", "/api/strip?song=mix&columns=many", "/api/strip?song=../x"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(url + bad)
        assert e.value.code == 404, bad


def test_an_empty_directory_is_served_so_that_the_first_song_can_be_added(tmp_path):
    srv = copilot_server.make_server(str(tmp_path / "new"), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert _json(url + "/api/songs") == [] and b'id="drop"' in _get(url + "/")[1]
        way = _json(url + "/api/adding")
        assert way["jobs"] == [] and way["kinds"] == list(copilot_adding.AUDIO) and way["finders"][0]["ready"]
        assert not os.path.exists(tmp_path / "new")  # nothing is made until a song comes
    finally:
        srv.RequestHandlerClass.adding.close()
        srv.shutdown()
        srv.server_close()
    (tmp_path / "file").write_text("")
    with pytest.raises(Bass2SheetError, match="not a directory"):
        copilot_server.make_server(str(tmp_path / "file"), 0)


# -------------------------------------------------------------------------------------------- what is refused


def test_what_is_not_a_song_is_refused_and_nothing_of_it_is_kept(served, monkeypatch):
    url, workdir, mix = served
    monkeypatch.delenv("MUSCRIPTOR_NOTES", raising=False)
    monkeypatch.delenv("MUSCRIPTOR_GGUF", raising=False)
    song = open(mix, "rb").read()
    quiet = _wav(np.zeros((3 * SR, 2), np.float32))
    opera = io.BytesIO()
    sf.write(opera, np.zeros(21 * 60 * 2000, np.float32), 2000, format="WAV", subtype="PCM_U8")
    for name, body, notes, status, words in (
        ("opera.wav", opera.getvalue(), "own", 413, "lasts 21:00; the tool takes songs up to 20 minutes"),
        ("notes.txt", b"hello", "own", 415, "not an audio file"),
        ("song.pdf", song, "own", 415, "not an audio file"),
        ("", song, "own", 415, "not an audio file"),
        ("words.mp3", b"this is not music\n" * 2000, "own", 415, "not an audio file this tool can read"),
        ("picture.wav", b"\x89PNG\r\n\x1a\n" + bytes(5000), "own", 415, "has no sound in it"),
        ("short.wav", quiet, "own", 400, "at least 10 s"),
        ("empty.wav", b"", "own", 400, "is empty"),
        ("mix.wav", song, "own", 409, "already among your songs"),
        ("new.wav", song, "muscriptor", 400, "MuScriptor is not set up"),
        ("new.wav", song, "someone", 400, "unknown note finder"),
    ):
        answer = _upload(url, name, body, notes)
        assert answer[0] == status and words in answer[1]["error"], (name, answer)
    assert _json(url + "/api/adding")["jobs"] == [] and list(copilot.load_songs(workdir)) == ["mix"]
    assert not os.path.exists(os.path.join(workdir, "uploads")) or os.listdir(os.path.join(workdir, "uploads")) == []


def test_a_file_that_is_unreasonably_large_is_refused_before_it_is_read(served, monkeypatch):
    url, workdir, mix = served
    monkeypatch.setattr(copilot_adding, "MAX_BYTES", 100_000)
    line = http.client.HTTPConnection(url.split("//")[1], timeout=30)
    line.putrequest("POST", "/api/upload?name=huge.wav")
    line.putheader("Content-Length", str(10**9))  # it says how large it is, and is answered before it is sent
    line.endheaders()
    answer = line.getresponse()
    assert answer.status == 413 and "954 MB: the copilot takes files up to 0 MB" in json.loads(answer.read())["error"]
    line.close()
    assert _json(url + "/api/adding")["limit"] == 100_000
    assert not os.path.exists(os.path.join(workdir, "uploads"))


def test_only_the_copilots_own_pages_can_change_things(served):
    url, workdir, mix = served
    there = {"Origin": "https://elsewhere.example"}
    assert _upload(url, "new.wav", open(mix, "rb").read(2000), headers=there)[0] == 403
    assert _post(url + "/api/act", dict(song="mix", action=dict(kind="undo")), headers=there)[0] == 403
    assert _post(url + "/api/adding", dict(id="mix", do="remove"), headers=there)[0] == 403
    here = {"Origin": "http://localhost:8770"}
    assert _post(url + "/api/act", dict(song="mix", action=dict(kind="undo")), headers=here)[0] == 200
    assert _upload(url, "notes.txt", b"hello", headers=here)[0] == 415  # let in, and refused for what it is


def test_a_file_is_kept_under_a_name_that_is_only_a_name():
    assert copilot_adding.safe_name("../../etc/Artist - Song.WAV") == "Artist - Song.wav"
    assert copilot_adding.safe_name("C:\\Music\\A <b>.Mp3") == "A b.mp3"
    assert copilot_adding.safe_name(".hidden.flac") == "hidden.flac" and copilot_adding.safe_name("...") == ""
    assert len(copilot_adding.safe_name("x" * 400 + ".ogg")) == 154


def test_a_file_without_its_size_or_cut_off_on_the_way_is_refused(tmp_path):
    adding = copilot_adding.Adding(str(tmp_path))
    with pytest.raises(copilot_adding.Refused, match="the file came without its size") as e:
        adding.accept("song.wav", None, io.BytesIO(b""))
    assert e.value.status == 411
    for _ in range(2):  # the second time it is not "already being prepared": the first left nothing behind
        with pytest.raises(copilot_adding.Refused, match="song.wav did not arrive whole") as e:
            adding.accept("song.wav", 5000, io.BytesIO(b"RIFF" + bytes(996)))  # a fifth of what was announced
        assert e.value.status == 400
    assert os.listdir(tmp_path / "uploads") == [] and adding.listing() == []
    adding.close()


def test_why_nothing_was_fetched_is_read_from_what_the_downloader_said():
    why = copilot_adding.why_not
    assert why("[download] An Opera does not pass filter (!is_live), skipping ..") == (
        "it is a live stream, or lasts more than 20 minutes"
    )
    assert why("[download] File is larger than max-filesize (600 > 500). Aborting.") == (
        "its audio is larger than 500 MB"
    )
    said = "[youtube] Extracting URL\nERROR: [youtube] one: Sign in\n\nERROR: [youtube] abc: Video unavailable\n"
    assert why(said) == "[youtube] abc: Video unavailable"  # the last thing it called an error
    assert why("  something odd happened \n\n") == "something odd happened"
    assert why("") == "the downloader did not say why"


# ---------------------------------------------------------------------------------------- a song is prepared


def test_a_new_song_is_kept_prepared_step_by_step_and_can_be_opened(served):
    url, workdir, mix = served
    body = open(mix, "rb").read()
    assert _upload(url, "Test Band - New Song.wav", body) == (200, dict(id="test-band-new-song", title="New Song"))
    again = _upload(url, "Test Band - New Song.wav", body)
    assert again[0] == 409 and "already being prepared" in again[1]["error"]
    job, seen = _settled(url, "test-band-new-song")
    assert job["state"] == "done" and job["error"] is None, job
    assert (job["title"], job["artist"], job["finder"]) == ("New Song", "Test Band", "own")
    order = ["reading", "isolating", "pitch", "notes", "beats", "checks", "picture", "ready"]
    assert [s["id"] for s in job["steps"]] == order
    assert all(s["state"] == "done" and s["seconds"] is not None for s in job["steps"]), job["steps"]
    assert seen == [s for s in order if s in seen] and len(seen) >= 2  # in order; the quick ones may go unseen
    assert [s["id"] for s in job["steps"] if s["check"]] == ["checks"] and all(s["about"] for s in job["steps"])
    found = {s["id"]: s["found"] for s in job["steps"]}
    assert 25 < found["reading"]["length"] < 28 and found["isolating"] == dict(by="given")
    assert 0.3 < found["pitch"]["pitched"] <= 1 and found["notes"]["notes"] >= 60 and found["notes"]["low"] == "E1"
    assert found["beats"]["tempo"] == pytest.approx(120, abs=2) and found["beats"]["bars"] >= 12
    assert "major" in found["beats"]["key"] or "minor" in found["beats"]["key"]
    assert found["checks"]["doubted"] >= 0 and found["picture"]["columns"] > 2000
    kept = os.path.join(workdir, "uploads", "Test Band - New Song.wav")
    assert open(kept, "rb").read() == body and os.listdir(os.path.join(workdir, "uploads")) == [os.path.basename(kept)]
    songs = {s["id"]: s for s in _json(url + "/api/songs")}
    new = songs["test-band-new-song"]
    assert list(songs) == ["mix", "test-band-new-song"] and (new["title"], new["artist"]) == ("New Song", "Test Band")
    assert new["notes"] == found["notes"]["notes"] and new["fixes"] == 0 and new["finder"] == "own"
    assert copilot.load_songs(workdir)["test-band-new-song"]["path"] == kept
    opened = _json(url + "/api/song?id=test-band-new-song")  # the existing page opens it like any other
    assert len(opened["notes"]) == new["notes"] and opened["title"] == "New Song"
    assert int(_get(url + "/api/strip?song=test-band-new-song&columns=64")[0]["X-Columns"]) == 64
    # ready: it leaves the row when the person hides it, and stays among the songs
    assert _post(url + "/api/adding", dict(id="test-band-new-song", do="remove"))[1]["jobs"] == []
    assert len(_json(url + "/api/songs")) == 2 and os.path.exists(kept)
    assert _upload(url, "Test Band - New Song.wav", body)[0] == 409


def test_a_song_that_cannot_be_prepared_says_why_and_is_tried_again_or_removed(served):
    url, workdir, _ = served
    silence = _wav(np.zeros((12 * SR, 2), np.float32))
    assert _upload(url, "Nothing.wav", silence)[0] == 200
    job, _ = _settled(url, "nothing")
    assert job["state"] == "failed" and job["error"] == "the audio is silent"
    assert [s["state"] for s in job["steps"]][:2] == ["failed", "waiting"]
    kept = os.path.join(workdir, "uploads", "Nothing.wav")
    assert os.path.exists(kept) and [s["id"] for s in _json(url + "/api/songs")] == ["mix"]
    assert _post(url + "/api/adding", dict(id="nothing", do="again"))[0] == 200
    job, _ = _settled(url, "nothing")
    assert job["state"] == "failed" and job["error"] == "the audio is silent"
    assert _post(url + "/api/adding", dict(id="nothing", do="remove"))[1]["jobs"] == []
    until = time.time() + 30
    while os.path.exists(kept) and time.time() < until:
        time.sleep(0.1)
    assert os.listdir(os.path.join(workdir, "uploads")) == [] and list(copilot.load_songs(workdir)) == ["mix"]
    assert not os.path.exists(os.path.join(workdir, "nothing"))
    for data in (dict(id="nothing", do="again"), dict(id="mix", do="remove"), dict(id="mix", do="rename")):
        assert _post(url + "/api/adding", data)[0] == 400, data


def test_muscriptors_notes_and_the_second_opinion_are_steps_when_it_is_chosen(served, song, tmp_path, monkeypatch):
    from test_muscriptor import _stand_in

    url, workdir, mix = served
    rows = [(round(s + 0.015, 2), round(e + 0.015, 2), p, "electric_bass") for s, e, p in song["notes"]]
    _stand_in(tmp_path, monkeypatch, [*rows, (25.0, 25.3, 60, "electric_bass")])
    ready = {f["id"]: f["ready"] for f in _json(url + "/api/adding")["finders"]}
    assert ready == dict(own=True, muscriptor=True)
    assert _upload(url, "Other Notes.wav", open(mix, "rb").read(), notes="muscriptor")[0] == 200
    job, _ = _settled(url, "other-notes")
    assert job["state"] == "done" and job["finder"] == "muscriptor", job
    order = ["reading", "isolating", "pitch", "notes", "muscriptor", "second", "beats", "checks", "picture", "ready"]
    assert [s["id"] for s in job["steps"]] == order and all(s["state"] == "done" for s in job["steps"])
    assert [s["id"] for s in job["steps"] if s["check"]] == ["second", "checks"]  # the two double checks
    found = {s["id"]: s["found"] for s in job["steps"]}
    assert found["muscriptor"] == dict(notes=len(rows) + 1) and found["second"]["alone"] >= 1
    assert {s["id"]: s["finder"] for s in _json(url + "/api/songs")} == {"mix": "own", "other-notes": "muscriptor"}


def _downloader(tmp_path, monkeypatch, audio):
    """A program that takes yt-dlp's place: it writes `audio` under the name a video would give it into the
    folder it is told, or fails as yt-dlp does for a video that is gone, too long, or not audio at all. Returns
    the log of what it was asked."""
    program, log = str(tmp_path / "yt-dlp"), str(tmp_path / "asked.txt")
    with open(program, "w") as f:
        f.write(
            f"#!{sys.executable}\nimport os, shutil, sys\n"
            "args = sys.argv[1:]\n"
            f"open({log!r}, 'a').write(' '.join(args) + '\\n')\n"
            "url, folder = args[-1], args[args.index('-P') + 1]\n"
            "assert args[-2] == '--' and '--no-playlist' in args and '--no-config' in args, args\n"
            "if 'gone' in url:\n"
            "    print('[youtube] Extracting URL: ' + url)\n"
            "    sys.exit('ERROR: [youtube] gone: Video unavailable')\n"
            "if 'long' in url:\n"
            "    print('[download] An Opera does not pass filter (!is_live & duration<=?1200), skipping ..')\n"
            "    sys.exit(0)\n"
            "if 'words' in url:\n"
            "    open(os.path.join(folder, 'Words [abc].webm'), 'w').write('this is not music' * 500)\n"
            "    sys.exit(0)\n"
            f"name = 'Stand In - Linked Song (Official Video) [dQw4w9WgXcQ].wav'\n"
            f"shutil.copy({audio!r}, os.path.join(folder, name))\n"
        )
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    monkeypatch.setenv("BASS2SHEET_DOWNLOADER", program)
    return log


def test_links_are_taken_only_where_a_downloader_is_installed(served, monkeypatch):
    url, workdir, _ = served
    monkeypatch.setenv("BASS2SHEET_DOWNLOADER", "none")
    assert _json(url + "/api/adding")["links"] is False
    status, answer = _post(url + "/api/link", dict(url="https://youtu.be/dQw4w9WgXcQ"))
    assert status == 400 and "no downloader (yt-dlp) is installed" in answer["error"]
    monkeypatch.setenv("BASS2SHEET_DOWNLOADER", "/nonexistent/yt-dlp")
    assert copilot_adding.downloader() is None
    monkeypatch.delenv("BASS2SHEET_DOWNLOADER")
    monkeypatch.setattr(copilot_adding.shutil, "which", lambda name: None)
    assert copilot_adding.downloader() in (None, "/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp")


def test_a_link_is_fetched_as_the_first_step_and_then_prepared_like_a_file(served, tmp_path, monkeypatch):
    url, workdir, mix = served
    log = _downloader(tmp_path, monkeypatch, mix)
    assert _json(url + "/api/adding")["links"] is True
    for bad in ("", "youtu.be/x", "ftp://www.youtube.com/watch?v=x", "https://example.com/watch?v=x", "/etc/passwd"):
        status, answer = _post(url + "/api/link", dict(url=bad))
        assert status == 400 and "not a link to YouTube" in answer["error"], bad
    assert _post(url + "/api/link", dict(url="https://youtu.be/x", notes="someone"))[0] == 400
    link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert _post(url + "/api/link", dict(url=f"  {link} ")) == (200, dict(key="link-1"))
    job, _ = _settled(url, "link-1")
    assert job["state"] == "done" and job["error"] is None, job
    assert (job["id"], job["title"], job["artist"], job["url"]) == (
        "stand-in-linked-song",
        "Linked Song",
        "Stand In",
        link,
    )
    order = ["fetching", "reading", "isolating", "pitch", "notes", "beats", "checks", "picture", "ready"]
    assert [s["id"] for s in job["steps"]] == order and all(s["state"] == "done" for s in job["steps"])
    assert job["steps"][0]["found"]["megabytes"] > 1 and job["steps"][0]["seconds"] is not None
    kept = "Stand In - Linked Song (Official Video) [dQw4w9WgXcQ].wav"
    assert os.listdir(os.path.join(workdir, "uploads")) == [kept]  # the file, and nothing of the fetching
    assert open(os.path.join(workdir, "uploads", kept), "rb").read() == open(mix, "rb").read()
    songs = {s["id"]: s for s in _json(url + "/api/songs")}
    assert songs["stand-in-linked-song"]["title"] == "Linked Song" and len(songs) == 2
    asked = open(log).read().split("\n")[0]
    assert asked.endswith("-- " + link) and "--max-filesize 500M" in asked and "-f bestaudio/best" in asked
    assert "--match-filters !is_live & duration<=?1200" in asked
    # the same link again: fetched, and found to be a song that is already there
    assert _post(url + "/api/link", dict(url=link))[0] == 200
    job, _ = _settled(url, "link-2")
    assert job["state"] == "failed" and job["error"] == "Linked Song is already among your songs"
    assert _post(url + "/api/adding", dict(id="link-2", do="remove"))[0] == 200
    assert os.listdir(os.path.join(workdir, "uploads")) == [kept] and len(_json(url + "/api/songs")) == 2


def test_a_link_that_cannot_be_fetched_says_why(served, tmp_path, monkeypatch):
    url, workdir, mix = served
    _downloader(tmp_path, monkeypatch, mix)
    for k, (link, words) in enumerate(
        (
            ("https://youtu.be/gone", "The audio could not be fetched: [youtube] gone: Video unavailable"),
            (
                "https://youtu.be/long",
                "The audio could not be fetched: it is a live stream, or lasts more than 20 minutes",
            ),
            ("https://youtu.be/words", "Words [abc].webm is not an audio file this tool can read"),
        )
    ):
        assert _post(url + "/api/link", dict(url=link))[0] == 200
        job, _ = _settled(url, f"link-{k + 1}")
        assert job["state"] == "failed" and job["error"] == words, job
        assert [s["state"] for s in job["steps"]][:2] == ["failed", "waiting"] and job["id"] is None
    assert _post(url + "/api/adding", dict(id="link-1", do="again"))[0] == 200
    assert _settled(url, "link-1")[0]["state"] == "failed"
    for k in (1, 2, 3):
        assert _post(url + "/api/adding", dict(id=f"link-{k}", do="remove"))[0] == 200
    assert _json(url + "/api/adding")["jobs"] == [] and os.listdir(os.path.join(workdir, "uploads")) == []
    assert list(copilot.load_songs(workdir)) == ["mix"]


def _slow_downloader(tmp_path, monkeypatch, audio):
    """A program in yt-dlp's place that can be made to wait: a link with `wait` in it is fetched only once the
    file this returns exists, one with `slow` never, and one with `slides` comes as a file that is not audio."""
    program, go = str(tmp_path / "yt-dlp"), str(tmp_path / "go")
    with open(program, "w") as f:
        f.write(
            f"#!{sys.executable}\nimport os, shutil, sys, time\n"
            "url, folder = sys.argv[-1], sys.argv[sys.argv.index('-P') + 1]\n"
            "if 'slides' in url:\n"
            "    open(os.path.join(folder, 'Slides [abc].pdf'), 'w').write('%PDF-1.4')\n"
            "    sys.exit(0)\n"
            f"while 'slow' in url or not os.path.exists({go!r}):\n"
            "    time.sleep(0.05)\n"
            f"shutil.copy({audio!r}, os.path.join(folder, 'Waited For [abc].wav'))\n"
        )
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    monkeypatch.setenv("BASS2SHEET_DOWNLOADER", program)
    return go


def _until(what, timeout=60):
    until = time.time() + timeout
    while time.time() < until:
        if what():
            return
        time.sleep(0.05)
    raise AssertionError("it did not happen")


def test_a_link_being_fetched_is_not_taken_twice_and_can_be_given_up(served, tmp_path, monkeypatch):
    url, workdir, mix = served
    _slow_downloader(tmp_path, monkeypatch, mix)
    link = "https://youtu.be/wait"
    assert _post(url + "/api/link", dict(url=link)) == (200, dict(key="link-1"))
    status, answer = _post(url + "/api/link", dict(url=link))
    assert (status, answer["error"]) == (409, "that link is already being fetched")
    _until(lambda: _json(url + "/api/adding")["jobs"][0]["steps"][0]["state"] == "running")
    assert os.listdir(os.path.join(workdir, "uploads")) == [".link-1"]  # where it is fetched to
    assert _post(url + "/api/adding", dict(id="link-1", do="remove"))[0] == 200  # "Cancel", while it is fetched
    _until(lambda: _json(url + "/api/adding")["jobs"] == [] and os.listdir(os.path.join(workdir, "uploads")) == [])
    assert list(copilot.load_songs(workdir)) == ["mix"]
    assert _post(url + "/api/link", dict(url=link)) == (200, dict(key="link-2"))  # given up, it can be asked again
    assert _post(url + "/api/adding", dict(id="link-2", do="remove"))[0] == 200


def test_a_link_that_gives_no_audio_or_takes_too_long_says_so(served, tmp_path, monkeypatch):
    url, workdir, mix = served
    _slow_downloader(tmp_path, monkeypatch, mix)
    assert _post(url + "/api/link", dict(url="https://youtu.be/slides"))[0] == 200
    job, _ = _settled(url, "link-1")
    assert job["state"] == "failed", job
    assert job["error"] == "The audio came as Slides [abc].pdf, a kind of file the copilot does not take"
    monkeypatch.setattr(copilot_adding, "FETCH_TIMEOUT", 1)
    assert _post(url + "/api/link", dict(url="https://youtu.be/slow"))[0] == 200
    job, _ = _settled(url, "link-2")
    assert job["state"] == "failed" and job["error"].startswith("The audio could not be fetched: it took more than")
    assert os.listdir(os.path.join(workdir, "uploads")) == []  # and nothing of either is kept


def test_a_song_being_prepared_is_cancelled_and_nothing_of_it_is_left(served):
    url, workdir, mix = served
    assert _upload(url, "Given Up.wav", open(mix, "rb").read())[0] == 200
    _until(lambda: _json(url + "/api/adding")["jobs"][0]["state"] == "running")
    assert _post(url + "/api/adding", dict(id="given-up", do="remove"))[0] == 200  # "Cancel"
    gone = lambda: (
        not os.path.exists(os.path.join(workdir, "given-up")) and "given-up" not in copilot.load_songs(workdir)
    )
    _until(lambda: _json(url + "/api/adding")["jobs"] == [] and os.listdir(os.path.join(workdir, "uploads")) == [])
    assert gone() and [s["id"] for s in _json(url + "/api/songs")] == ["mix"]
    assert _upload(url, "Given Up.wav", open(mix, "rb").read(2000))[0] != 409  # it is not "already being prepared"


def _program_for(adding, monkeypatch, tmp_path, body):
    """Makes `adding` prepare its songs with a program that does `body` in place of the real preparation."""
    program = str(tmp_path / "prepare")
    say = f"say = lambda text: print('{copilot_adding.MARK}' + text, flush=True)"
    with open(program, "w") as f:
        f.write(f"#!{sys.executable}\nimport sys\n{say}\n{body}")
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    monkeypatch.setattr(adding, "command", lambda job: [program])


def _sent(adding, name):
    song = _wav(np.zeros((11 * SR, 2), np.float32))
    return adding.accept(name, len(song), io.BytesIO(song))


def test_what_the_preparation_says_is_what_the_page_shows(tmp_path, monkeypatch):
    adding = copilot_adding.Adding(str(tmp_path))
    _program_for(
        adding,
        monkeypatch,
        tmp_path,
        'say(\'{"step": "reading", "state": "running"}\')\n'
        "say('{this is not what was agreed')\n"  # a line it cannot read is passed over
        "print('a line for the log, not for the page')\n"
        'say(\'{"step": "reading", "state": "done", "seconds": 0.5, "length": 11.0}\')\n'
        'say(\'{"step": "teleporting", "state": "running"}\')\n'  # a step the page does not know
        'say(\'{"step": "pitch", "state": "running"}\')\n'
        'say(\'{"step": "pitch", "state": "done", "pitched": 0.8}\')\n'  # no time: the page's own
        'say(\'{"say": "Quiet: 31 notes"}\')\n',
    )
    job = _sent(adding, "Quiet.wav")
    _until(lambda: job["state"] in ("done", "failed"))
    shown = adding.listing()[0]
    steps = {s["id"]: s for s in shown["steps"]}
    assert shown["state"] == "done" and all(s["state"] == "done" for s in steps.values())
    assert (steps["reading"]["seconds"], steps["reading"]["found"]) == (0.5, dict(length=11.0))
    assert steps["pitch"]["found"] == dict(pitched=0.8) and steps["pitch"]["seconds"] is not None
    assert steps["notes"]["seconds"] is None and "teleporting" not in steps  # never ran: kept from an earlier time
    assert os.listdir(tmp_path / "uploads") == ["Quiet.wav"]  # its log is not kept when it went well
    adding.close()


def test_a_preparation_that_stops_leaves_its_last_words_and_the_next_song_is_prepared(tmp_path, monkeypatch):
    adding = copilot_adding.Adding(str(tmp_path))
    _program_for(
        adding,
        monkeypatch,
        tmp_path,
        'say(\'{"step": "isolating", "state": "running"}\')\n'
        "sys.stderr.write('Traceback (most recent call last):\\nMemoryError: the model did not fit\\n')\n"
        "sys.exit(3)\n",
    )
    job = _sent(adding, "Too Big.wav")
    _until(lambda: job["state"] == "failed")
    assert job["error"] == "MemoryError: the model did not fit"
    states = {s["id"]: s["state"] for s in adding.listing()[0]["steps"]}
    assert (states["reading"], states["isolating"], states["pitch"]) == ("waiting", "failed", "waiting")
    assert sorted(os.listdir(tmp_path / "uploads")) == ["Too Big.wav", "too-big.log"]  # the log says where
    _program_for(adding, monkeypatch, tmp_path, "sys.exit(9)\n")  # one that stops without a word
    other = _sent(adding, "Silent End.wav")
    _until(lambda: other["state"] == "failed")
    assert other["error"] == "the preparation stopped (exit 9)"

    def breaks(job):
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(adding, "_prepare", breaks)  # whatever a song does to the worker, it stays up
    adding.again("too-big")
    _until(lambda: job["state"] == "failed" and job["error"] == "RuntimeError: the disk is full")
    monkeypatch.undo()
    _program_for(adding, monkeypatch, tmp_path, "sys.exit(0)\n")
    adding.again("too-big")
    _until(lambda: job["state"] == "done")
    assert [j["state"] for j in adding.listing()] == ["done", "failed"]
    adding.close()


def test_the_preparing_program_says_how_it_went_whatever_happens(tmp_path, monkeypatch, capsys):
    asked = []

    def add(paths, workdir, **kw):
        asked.append((paths, workdir, {k: v for k, v in kw.items() if k != "say"}))
        with step("picture") as found:
            found["columns"] = 2650
        kw["say"]("Song: 61 notes, 2 of them doubted")

    told = lambda: [json.loads(line[len(copilot_adding.MARK) :]) for line in capsys.readouterr().out.splitlines()]
    monkeypatch.setattr(copilot, "add", add)
    assert copilot_adding.main(["/scores", "/uploads/Song.wav", "own"]) == 0
    assert asked == [(["/uploads/Song.wav"], "/scores", dict(settings=None, stems_dirs=None))]
    events = told()
    assert [(e.get("step"), e.get("state")) for e in events] == [
        ("picture", "running"),
        ("picture", "done"),
        ("ready", "running"),  # what is left after the picture: writing it all down
        (None, None),
        ("ready", "done"),
    ]
    assert events[1]["columns"] == 2650 and events[3] == dict(say="Song: 61 notes, 2 of them doubted")
    assert copilot_adding.main(["/scores", "/uploads/Song.wav", "muscriptor", "/stems"]) == 0
    chosen = dict(settings=dict(notes="muscriptor"), stems_dirs={"/uploads/Song.wav": "/stems"})
    assert asked[1] == (["/uploads/Song.wav"], "/scores", chosen)
    told()

    def refuses(*args, **kw):
        raise Bass2SheetError("the audio is silent")

    monkeypatch.setattr(copilot, "add", refuses)
    assert copilot_adding.main(["/scores", "/uploads/Song.wav", "own"]) == 1
    assert told() == [dict(error="the audio is silent")]

    def breaks(*args, **kw):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(copilot, "add", breaks)
    assert copilot_adding.main(["/scores", "/uploads/Song.wav", "own"]) == 1
    said = capsys.readouterr()
    assert said.out == copilot_adding.MARK + '{"error": "ZeroDivisionError: division by zero"}\n'  # for the page
    assert "Traceback" in said.err and "breaks" in said.err  # for the log: where it happened


def test_a_song_left_half_prepared_by_a_server_that_stopped_is_shown_to_try_again(prepared, tmp_path):
    workdir = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, workdir)
    songs = copilot.load_songs(workdir)
    path = os.path.join(workdir, "uploads", "Half.wav")
    songs["half"] = dict(id="half", path=path, title="Half", artist="", settings=dict(notes="muscriptor"))
    songs["by-hand"] = dict(id="by-hand", path="/somewhere/By Hand.wav", title="By Hand", artist="", settings={})
    json.dump(songs, open(os.path.join(workdir, "songs.json"), "w"))
    adding = copilot_adding.Adding(workdir)
    (job,) = adding.listing()  # only what came in from the page; what `add` is doing in a terminal is left alone
    assert (job["id"], job["state"], job["finder"]) == ("half", "failed", "muscriptor")
    assert "stopped" in job["error"] and "muscriptor" in [s["id"] for s in job["steps"]]
    with pytest.raises(Bass2SheetError, match="not kept any more"):
        adding.again("half")
    adding.close()


def test_the_steps_of_the_work_are_told_to_whoever_listens():
    told = []
    with listening(told.append):
        with step("one") as found:
            found["notes"] = 3
        with pytest.raises(ValueError), step("two"):
            raise ValueError("no")
    with step("three") as found:  # no one listens: nothing happens
        found["x"] = 1
    assert [(e["step"], e["state"]) for e in told] == [("one", "running"), ("one", "done"), ("two", "running")]
    assert told[1]["notes"] == 3 and 0 <= told[1]["seconds"] < 1


# ---------------------------------------------------------------------------------------------------- the page


def test_the_landing_page_has_every_element_its_script_reaches_for():
    import re

    page = open(copilot_server.LANDING_PATH).read()
    script = page.split('<script id="app">')[1]
    made = set(re.findall(r'id="([^"]+)"', page))
    missing = set(re.findall(r'\$\("([^"]+)"\)', script)) - made
    assert not missing, missing
    labels = {sid for sid, *_ in copilot_adding.STEPS}
    said = set(re.findall(r'case "([a-z]+)":', page))
    assert labels - {"ready"} <= said, labels - said  # every step has its few words on the page
    stages = re.findall(r'<li data-stage="(\w+)"><a href="#(\w+)"><i>(\d)</i><b>([^<]+)</b>', page)
    assert [(k, name) for _, _, k, name in stages] == [
        ("1", "Add a song"),
        ("2", "Watch it being prepared"),
        ("3", "Fix the notes"),
        ("4", "Make the score"),
    ]  # the path, in its order; each stage leads to a part of the page that is there
    assert {anchor for _, anchor, _, _ in stages} <= made


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's logic")
def test_the_landing_page_logic_holds():
    script = os.path.join(os.path.dirname(__file__), "copilot_landing_logic.js")
    r = subprocess.run(["node", script, copilot_server.LANDING_PATH], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
