"""The copilot's score, seen on its own pages: drawn page by page when it is made, and shown with its files
(synthetic song; a stand-in for MuseScore, and MuseScore itself where it is installed)."""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from bass2sheet import copilot, copilot_server, score

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")


@pytest.fixture(scope="module")
def prepared(taste_song_files, cache_dir, tmp_path_factory):
    mix, stems = taste_song_files
    workdir = str(tmp_path_factory.mktemp("scores"))
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    copilot.add([mix], workdir, opts, stems_dirs={mix: stems}, say=lambda _: 0)
    return workdir


@pytest.fixture
def served(prepared, tmp_path):
    workdir = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, workdir)
    srv = copilot_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.RequestHandlerClass.adding.close()
    srv.shutdown()
    srv.server_close()


def _stand_in(tmp_path, monkeypatch, pages=3):
    """A program that takes MuseScore's place: for `-S STYLE -j JOB` it writes each PDF asked for and `pages`
    pictures beside it, as MuseScore names them. Returns the file that says how many pages the next score has."""
    program, count = str(tmp_path / "mscore"), str(tmp_path / "pages.txt")
    with open(count, "w") as f:
        f.write(str(pages))
    with open(program, "w") as f:
        f.write(
            f"#!{sys.executable}\nimport json, sys\n"
            "assert sys.argv[1] == '-S' and sys.argv[3] == '-j', sys.argv\n"
            "for job in json.load(open(sys.argv[4])):\n"
            "    assert '<score-partwise' in open(job['in']).read()\n"
            "    for out in job['out']:\n"
            "        if out.endswith('.pdf'):\n"
            "            open(out, 'wb').write(b'%PDF-1.4 stand-in')\n"
            "        else:\n"
            f"            for k in range(int(open({count!r}).read())):\n"
            "                open(f'{out[:-4]}-{k + 1}.svg', 'w').write(f'<svg xmlns=\"http://www.w3.org/2000/svg\">"
            "<title>page {k + 1}</title></svg>')\n"
        )
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", [program])
    return count


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return dict(r.headers), r.read()


def _json(url):
    return json.loads(_get(url)[1])


def _post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_the_score_is_drawn_page_by_page_and_shown_with_its_files(served, tmp_path, monkeypatch):
    url, workdir = served
    count = _stand_in(tmp_path, monkeypatch)
    before = _json(url + "/api/score?song=mix")
    assert before == dict(id="mix", title="mix", artist="", made=None, mscore=True)
    desk = copilot.Desk(workdir, "mix")
    n = desk.notes[6]
    fix = dict(song="mix", action=dict(kind="pitch", id=n["id"], pitch=n["pitch"] + 12, how="key", offer=False))
    _post(url + "/api/act", fix)
    _post(url + "/api/checked", dict(song="mix", start=0, end=desk.machine["length"]))
    links = _post(url + "/api/sheet", dict(song="mix"))  # what the song's page gets is as it was
    assert links == {
        k: f"/data/mix/sheet/{v}"
        for k, v in dict(musicxml="bass.musicxml", midi="bass_as_played.mid", pdf="bass.pdf").items()
    }
    view = _json(url + "/api/score?song=mix")
    made = view["made"]
    assert (
        made["fixes"] == 1
        and made["notes"] >= 60
        and 0 < made["checked"] < 1
        and made["worked"] == pytest.approx(n["start"], abs=0.06)
    )
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", made["when"])
    assert view["files"] == links and view["pages"] == [f"/data/mix/sheet/bass-{k}.svg" for k in (1, 2, 3)]
    for k, page in enumerate(view["pages"]):
        headers, body = _get(url + page)
        assert headers["Content-Type"] == "image/svg+xml" and f"<title>page {k + 1}</title>".encode() in body
    assert _get(url + view["files"]["pdf"])[1] == b"%PDF-1.4 stand-in"
    assert b"<score-partwise" in _get(url + view["files"]["musicxml"])[1]
    with open(count, "w") as f:  # made again, shorter: the page that is no more is not left behind
        f.write("2")
    _post(url + "/api/sheet", dict(song="mix"))
    assert _json(url + "/api/score?song=mix")["pages"] == ["/data/mix/sheet/bass-1.svg", "/data/mix/sheet/bass-2.svg"]
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(url + "/data/mix/sheet/bass-3.svg")
    assert e.value.code == 404
    for bad in ("/api/score?song=nope", "/api/score", "/data/mix/sheet/sheet.json", "/data/mix/sheet/bass-x.svg"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(url + bad)
        assert e.value.code == 404, bad


def test_without_musescore_the_score_is_written_and_the_page_is_told_it_cannot_be_drawn(served, tmp_path, monkeypatch):
    url, workdir = served
    _stand_in(tmp_path, monkeypatch)
    _post(url + "/api/sheet", dict(song="mix"))
    assert len(_json(url + "/api/score?song=mix")["pages"]) == 3
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", [])  # MuseScore is gone
    links = _post(url + "/api/sheet", dict(song="mix"))
    view = _json(url + "/api/score?song=mix")
    assert sorted(links) == ["midi", "musicxml"] and view["files"] == links
    assert view["pages"] == [] and view["mscore"] is False and view["made"]["notes"] >= 60
    assert not os.path.exists(os.path.join(workdir, "mix", "sheet", "bass.pdf"))  # not the PDF of another version


@pytest.mark.skipif(score.find_mscore() is None, reason="MuseScore 4 is not installed")
def test_musescore_draws_the_pages(prepared, tmp_path):
    workdir = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, workdir)
    files = copilot.sheet(workdir, "mix")
    made = copilot.score_made(workdir, "mix")
    assert made["files"] == {k: os.path.basename(v) for k, v in files.items()} and "pdf" in files
    assert made["pages"] and made["pages"][0] == "bass-1.svg"
    for name in made["pages"]:
        with open(os.path.join(workdir, "mix", "sheet", name)) as f:
            assert "<svg" in f.read(400)
    assert not [f for f in os.listdir(os.path.join(workdir, "mix", "sheet")) if f.startswith(".")]


def test_the_score_has_a_page_and_the_songs_page_opens_it_in_a_panel(served):
    url, _ = served
    headers, page = _get(url + "/score")
    assert headers["Content-Type"].startswith("text/html") and b'id="pages"' in page
    headers, panel = _get(url + "/score-panel.js")
    assert headers["Content-Type"].startswith("text/javascript") and b"window.showScore = showScore" in panel
    song = _get(url + "/song")[1].decode()
    assert '<script src="/score-panel.js"></script>' in song and "showScore(S.id);" in song
    assert "copilot-score-close" in page.decode() and "copilot-score-close" in panel.decode()  # how the panel is closed


def test_the_score_page_has_every_element_its_script_reaches_for():
    page = open(copilot_server.SCORE_PATH).read()
    script = page.split('<script id="app">')[1]
    made = set(re.findall(r'id="([^"]+)"', page))
    missing = set(re.findall(r'\$\("([^"]+)"\)', script)) - made
    assert not missing, missing


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's logic")
def test_the_score_page_logic_holds():
    script = os.path.join(os.path.dirname(__file__), "copilot_score_logic.js")
    r = subprocess.run(["node", script, copilot_server.SCORE_PATH], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
