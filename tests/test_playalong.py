"""Play along: the record without its bass, the score cut into lines with the time of every bar and note, the
page, and the home network (synthetic song, no models to fetch; MuseScore where it is installed)."""

import http.client
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from bass2sheet import copilot, copilot_server, playalong, score
from bass2sheet.common import Bass2SheetError

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")
needs_mscore = pytest.mark.skipif(score.find_mscore() is None, reason="needs MuseScore 4 to engrave the score")


@pytest.fixture(params=["a stand-in", "MuseScore"])
def engraver(request, tmp_path, monkeypatch):
    """What draws the score: a stand-in that lays the bars out evenly (tests/mscore_standin.py), so that the test
    runs anywhere, and MuseScore itself where it is installed."""
    from conftest import mscore_stand_in

    if request.param == "MuseScore":
        if score.find_mscore() is None:
            pytest.skip("MuseScore 4 is not installed")
    else:
        monkeypatch.setattr(score, "MSCORE_CANDIDATES", [mscore_stand_in(tmp_path)])
    return request.param


@pytest.fixture(scope="module")
def prepared(taste_song_files, cache_dir, tmp_path_factory):
    mix, stems = taste_song_files
    workdir = str(tmp_path_factory.mktemp("play"))
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    copilot.add([mix], workdir, opts, stems_dirs={mix: stems}, say=lambda _: 0)
    return workdir


@pytest.fixture
def workdir(prepared, tmp_path):
    out = os.path.join(str(tmp_path), "scores")
    shutil.copytree(prepared, out)
    return out


def _audio(workdir, name):
    return sf.read(os.path.join(workdir, "mix", "audio", name + ".flac"), dtype="float64", always_2d=True)[0].T


def _point_at(workdir, stems_dir):
    songs = copilot.load_songs(workdir)
    songs["mix"]["stems_dir"] = stems_dir
    with open(os.path.join(workdir, "songs.json"), "w") as f:
        json.dump(songs, f)


# --------------------------------------------------------------------------------------- the song without the bass


def test_a_prepared_song_has_the_record_without_its_bass(prepared):
    record, bass, rest = (_audio(prepared, name) for name in ("record", "bass", "nobass"))
    assert rest.shape == record.shape and bass.shape[0] == 1
    left = record - rest - bass  # the stems of the synthetic song add up to it: what is taken out is the bass
    assert float(np.sum(left**2)) < 0.01 * float(np.sum(bass**2)) * 2
    assert float(np.sum(rest**2)) > 0.2 * float(np.sum(record**2))  # and the rest of the band is still there


def test_the_record_without_its_bass_is_made_when_it_is_missing(workdir, tmp_path):
    first = _audio(workdir, "nobass")
    out = os.path.join(workdir, "mix", "audio", "nobass.flac")
    made = os.path.getmtime(out)
    assert playalong.backing(workdir, "mix") == out and os.path.getmtime(out) == made  # there already: kept
    os.remove(out)
    assert playalong.backing(workdir, "mix") == out  # from the stems the song was prepared from
    assert float(np.abs(_audio(workdir, "nobass") - first).max()) < 2e-3

    # a separator turns a loud record down before taking it apart: its stems at 0.6 of their level give the same
    stems = copilot.load_songs(workdir)["mix"]["stems_dir"]
    quiet = str(tmp_path / "quiet")
    os.makedirs(quiet)
    for name in os.listdir(stems):
        y, sr = sf.read(os.path.join(stems, name), dtype="float64")
        sf.write(os.path.join(quiet, name), 0.6 * y, sr, subtype="PCM_16")
    _point_at(workdir, quiet)
    os.remove(out)
    playalong.backing(workdir, "mix")
    assert float(np.abs(_audio(workdir, "nobass") - first).max()) < 5e-3

    # without the stems, or with another song's, the isolated bass is taken out of the record
    def closeness():
        again = _audio(workdir, "nobass")
        return float(np.corrcoef(again.ravel(), first.ravel())[0, 1])

    _point_at(workdir, str(tmp_path / "gone"))
    os.remove(out)
    playalong.backing(workdir, "mix")
    assert closeness() > 0.99
    other = str(tmp_path / "other")
    os.makedirs(other)
    rng = np.random.default_rng(0)
    for name in os.listdir(stems):
        sf.write(os.path.join(other, name), 0.1 * rng.standard_normal((len(first[0]), 2)), 44100, subtype="PCM_16")
    _point_at(workdir, other)
    os.remove(out)
    playalong.backing(workdir, "mix")
    assert closeness() > 0.99
    with pytest.raises(Bass2SheetError, match="not prepared"):
        playalong.backing(workdir, "nope")


def test_it_is_slowed_down_like_the_record_and_made_in_the_background(workdir):
    path = playalong.sound(workdir, "mix", "nobass", 80)
    whole = sf.info(os.path.join(workdir, "mix", "audio", "nobass.flac"))
    slow = sf.info(path)
    assert path.endswith("nobass-80.flac") and slow.channels == 2 and slow.frames == round(whole.frames / 0.8)
    assert playalong.sound(workdir, "mix", "record") == os.path.join(workdir, "mix", "audio", "record.flac")
    for name, percent in (("drums", 100), ("nobass", 85)):
        with pytest.raises(Bass2SheetError):
            playalong.sound(workdir, "mix", name, percent)
    os.remove(os.path.join(workdir, "mix", "audio", "nobass.flac"))
    maker = playalong.Sounds(workdir)
    assert maker.ask("mix", "bass", 100) == dict(ready=True, url="/play/mix/audio/bass.flac")
    assert maker.ask("mix", "nobass", 70) == dict(ready=False)  # started: the page asks again
    until = time.time() + 120
    while not (answer := maker.ask("mix", "nobass", 70))["ready"]:
        assert time.time() < until
        time.sleep(0.2)
    assert answer["url"] == "/play/mix/audio/nobass-70.flac"
    assert os.path.exists(os.path.join(workdir, "mix", "audio", "nobass-70.flac"))
    for bad in (("other", "bass", 100), ("mix", "drums", 100), ("mix", "bass", 75)):
        with pytest.raises(Bass2SheetError):
            maker.ask(*bad)


# ------------------------------------------------------------------------------------------- when things are played

POSITIONS = """<?xml version="1.0" encoding="UTF-8"?>
<score><elements>{elements}</elements><events>{events}</events></score>"""


def _positions(items):
    """A positions file as MuseScore writes it: items are (x, y, width, height, page, ms), in picture points."""
    elements = "".join(
        f'<element id="{i}" x="{x * 12}" y="{y * 12}" sx="{w * 12}" sy="{h * 12}" page="{page}"></element>'
        for i, (x, y, w, h, page, _) in enumerate(items)
    )
    events = "".join(f'<event elid="{i}" position="{ms}"></event>' for i, (*_, ms) in enumerate(items))
    return POSITIONS.format(elements=elements, events=events)


def test_a_bar_and_a_note_take_their_time_from_the_songs_own_grid():
    # three bars at the score's steady 120 (2000 ms each), two on the first line and one on the next page;
    # the record itself slows down: every sixteenth of its grid is 2% longer than the one before
    steps = 0.125 * 1.02 ** np.arange(80)
    grid = np.concatenate([[1.0], 1.0 + np.cumsum(steps)])
    origin = 4  # bar 1 starts on the fifth sixteenth of the grid
    measures = _positions([(100, 50, 400, 120, 1, 0), (500, 50, 300, 120, 1, 2000), (40, 50, 700, 120, 2, 4000)])
    marks = [(120, 0), (220, 500), (320, 1000), (520, 2000), (700, 3500), (60, 4000), (400, 5000)]
    segments = _positions([(x, 50, 20, 120, 1 if ms < 4000 else 2, ms) for x, ms in marks])
    starts, sounding = {0, 4, 16, 28, 40}, {0, 1, 2, 3, 4, 5, 16, 17, 28, 29, 30, 31, 32, 33, 40}
    lines = playalong.lines_of(
        playalong.positions(measures), playalong.positions(segments), 1, grid, origin, starts, sounding
    )
    assert [[b["bar"] for b in line["bars"]] for line in lines] == [[1, 2], [3]]
    assert [(b["x0"], b["x1"]) for b in lines[0]["bars"]] == [(100, 500), (500, 800)]
    for line in lines:
        for b in line["bars"]:
            assert b["t"] == pytest.approx(grid[origin + 16 * (b["bar"] - 1)])
    assert lines[0]["end"] == pytest.approx(grid[origin + 32]) and lines[1]["end"] == pytest.approx(grid[origin + 48])
    got = [(m["x"], m["kind"], round(m["t"], 6)) for line in lines for m in line["marks"]]
    sixteenths = [0, 4, 8, 16, 28, 32, 40]  # where the score's clock puts them
    assert got == [
        (x, kind, round(float(grid[origin + g]), 6))
        for (x, _), kind, g in zip(marks, ["n", "n", "r", "n", "n", "t", "n"], sixteenths)
    ]
    assert lines[0]["bars"][1]["t"] - lines[0]["bars"][0]["t"] < lines[1]["bars"][0]["t"] - lines[0]["bars"][1]["t"]
    # the grid goes on at its own pace where the score reaches past it
    assert playalong.grid_time(grid, len(grid)) == pytest.approx(grid[-1] + (grid[-1] - grid[-2]))
    assert playalong.grid_time(grid, -2) == pytest.approx(1.0 - 2 * 0.125)
    with pytest.raises(Bass2SheetError, match="sixteenths"):  # a note between two sixteenths: not this score
        off = _positions([(120, 50, 20, 120, 1, 60)])
        playalong.lines_of(playalong.positions(measures), playalong.positions(off), 1, grid, origin)


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_the_score_is_engraved_line_by_line_with_the_time_of_everything_in_it(workdir, engraver):
    made = playalong.lines(workdir, "mix", 3)
    folder = os.path.join(workdir, "mix", "play", "b3")
    machine = copilot.read_json(os.path.join(workdir, "mix", "machine.json"))
    grid, origin = machine["grid"], machine["origin"]
    assert made["bars"] == 3 and len(made["lines"]) >= 4
    for line in made["lines"]:  # every line a picture, all of one size
        assert _png_size(os.path.join(folder, line["file"])) == (made["width"], made["height"])
    assert 3 < made["width"] / made["height"] < 6 and made["width"] > 1500  # wide, and sharp on a phone
    bars = [b for line in made["lines"] for b in line["bars"]]
    assert [b[0] for b in bars] == list(range(1, len(bars) + 1)) and max(
        len(line["bars"]) for line in made["lines"]
    ) == 3
    for number, x0, x1, t in bars:
        assert t == pytest.approx(playalong.grid_time(grid, origin + 16 * (number - 1)), abs=1e-3) and 0 <= x0 < x1 <= 1
    for line in made["lines"]:
        xs, ts = [m[0] for m in line["marks"]], [m[2] for m in line["marks"]]
        assert xs == sorted(xs) and ts == sorted(ts) and len(set(ts)) == len(ts)
        assert line["bars"][0][3] <= ts[0] and ts[-1] < line["end"]
        assert line["bars"][0][1] < xs[0] and xs[-1] < line["bars"][-1][2]
    # every note the score writes is a mark, at the time of the sixteenth it is written on
    written, tab = playalong.written(machine, copilot.read_json(os.path.join(workdir, "mix", "notes.json"))["notes"])
    events = [e for e in score.written_events(written, tab) if not e["grace"]]
    marks = [m for line in made["lines"] for m in line["marks"]]
    assert sorted(m[2] for m in marks if m[3] == "n") == pytest.approx(
        sorted(playalong.grid_time(grid, origin + e["g0"]) for e in events), abs=1e-3
    )
    heard = sorted(n["start"] for n in copilot.read_json(os.path.join(workdir, "mix", "notes.json"))["notes"])
    near = [min(abs(m[2] - s) for s in heard) for m in marks if m[3] == "n"]
    assert float(np.median(near)) < 0.04  # which is when the record plays it
    assert {m[3] for m in marks} >= {"n", "r"}

    # kept until the notes change; then engraved again, and the old pictures go
    stamp = os.path.getmtime(os.path.join(folder, made["lines"][0]["file"]))
    assert playalong.lines(workdir, "mix", 3) == made
    assert os.path.getmtime(os.path.join(folder, made["lines"][0]["file"])) == stamp
    desk = copilot.Desk(workdir, "mix")
    desk.act(dict(kind="pitch", id=desk.notes[8]["id"], pitch=desk.notes[8]["pitch"] + 5, how="key", offer=False))
    again = playalong.lines(workdir, "mix", 3)
    assert again["key"] != made["key"] and len(again["lines"]) == len(made["lines"])
    assert sorted(os.listdir(folder)) == sorted([line["file"] for line in again["lines"]] + ["score.json"])
    with pytest.raises(Bass2SheetError, match="lines of 7 bars"):
        playalong.lines(workdir, "mix", 7)


def test_without_musescore_the_page_is_told_what_is_missing(workdir, monkeypatch):
    monkeypatch.setattr(score, "find_mscore", lambda explicit=None: None)
    with pytest.raises(Bass2SheetError, match="MuseScore 4, which is not installed"):
        playalong.lines(workdir, "mix", 3)
    empty = copilot.read_json(os.path.join(workdir, "mix", "notes.json"))
    empty["notes"] = []
    with open(os.path.join(workdir, "mix", "notes.json"), "w") as f:
        json.dump(empty, f)
    with pytest.raises(Bass2SheetError, match="no notes to play along with"):
        playalong.lines(workdir, "mix", 3)


def test_the_highlight_follows_a_record_that_does_not_keep_time(workdir, engraver):
    """The same song played by a band that drifts: every moment t of it happens at warp(t) instead. The score is
    the same; every bar and note in it is played when the drifting record plays it."""
    steady = playalong.lines(workdir, "mix", 3)
    warp = lambda t: t + 0.6 * np.sin(t / 4.0) + 0.02 * t
    folder = os.path.join(workdir, "mix")
    machine, mine = (copilot.read_json(os.path.join(folder, name)) for name in ("machine.json", "notes.json"))
    began = machine["grid"][0]  # before its first sixteenth the grid is only guessed at: not compared
    machine["grid"] = [float(warp(t)) for t in machine["grid"]]
    for n in mine["notes"]:
        n["start"], n["end"] = float(warp(n["start"])), float(warp(n["end"]))
    for name, data in (("machine.json", machine), ("notes.json", mine)):
        with open(os.path.join(folder, name), "w") as f:
            json.dump(data, f)
    drifting = playalong.lines(workdir, "mix", 3)
    assert len(drifting["lines"]) == len(steady["lines"])
    for a, b in zip(steady["lines"], drifting["lines"]):
        assert [x[:3] for x in a["bars"]] == [x[:3] for x in b["bars"]]  # the same picture
        for was, now in zip(
            [x[3] for x in a["bars"]] + [m[2] for m in a["marks"]],
            [x[3] for x in b["bars"]] + [m[2] for m in b["marks"]],
        ):
            assert was < began or now == pytest.approx(warp(was), abs=2e-3)
    lengths = np.diff([x[3] for line in drifting["lines"] for x in line["bars"]])
    assert lengths.max() - lengths.min() > 0.2  # bars of different lengths, as the record has them


@needs_mscore
def test_a_song_too_busy_for_the_line_gets_fewer_bars_on_it(workdir, monkeypatch):
    monkeypatch.setattr(playalong, "LINE", 42.0)  # a narrow line: three of these bars do not fit across it
    made = playalong.lines(workdir, "mix", 3)
    counts = [len(line["bars"]) for line in made["lines"]]
    assert made["bars"] < 3 and max(counts) <= made["bars"] and counts.count(made["bars"]) >= 0.8 * len(counts)
    assert [b[0] for line in made["lines"] for b in line["bars"]] == list(range(1, sum(counts) + 1))


# ------------------------------------------------------------------------------------------------------- the page


@pytest.fixture
def server(workdir):
    srv = copilot_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as r:
        return r.status, dict(r.headers), r.read()


def _fetch(port, path, headers=None):
    """(status, headers, body) of a GET, redirects not followed."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


def _status(url, data=None):
    req = urllib.request.Request(url, None if data is None else json.dumps(data).encode())
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_the_server_gives_the_page_what_a_phone_needs(server):
    url, workdir = server
    page = _get(url + "/play")[2].decode()
    for tag in (
        '<meta name="apple-mobile-web-app-capable" content="yes">',
        '<meta name="mobile-web-app-capable" content="yes">',
        "viewport-fit=cover",
        '<link rel="manifest" href="/play/manifest.webmanifest">',
        '<link rel="apple-touch-icon" href="/play/icon-180.png">',
    ):
        assert tag in page, tag
    manifest = json.loads(_get(url + "/play/manifest.webmanifest")[2])
    assert (
        manifest["display"] == "fullscreen"
        and manifest["start_url"] == "/play"
        and manifest["orientation"] == "landscape"
    )
    for icon in manifest["icons"]:
        body = _get(url + icon["src"])[2]
        assert body[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", body[16:24]) == tuple(
            int(v) for v in icon["sizes"].split("x")
        )
    (listed,) = json.loads(_get(url + "/api/play/songs")[2])
    assert (listed["id"], listed["title"]) == ("mix", "mix") and 25 < listed["length"] < 28 and listed["tempo"] > 60
    # the sound: asked for until it is there, then played in ranges
    until = time.time() + 60
    while not (answer := json.loads(_get(url + "/api/play/sound?id=mix&name=nobass&speed=100")[2]))["ready"]:
        assert time.time() < until
        time.sleep(0.1)
    status, headers, body = _get(url + answer["url"], {"Range": "bytes=0-3"})
    assert (status, body, headers["Content-Type"], headers["Accept-Ranges"]) == (206, b"fLaC", "audio/flac", "bytes")
    assert _get(url + "/play/mix/audio/record.flac", {"Range": "bytes=4-"})[0] == 206
    film = _get(url + "/play/awake.mp4")
    assert film[1]["Content-Type"] == "video/mp4" and b"ftyp" in film[2][:32]
    for path, code in (
        ("/play/mix/audio/notes.flac", 404),
        ("/play/mix/audio/bass-75.flac", 404),  # not made: the page asks /api/play/sound first
        ("/play/other/audio/bass.flac", 404),
        ("/play/mix/b3/../../notes.json", 404),
        ("/play/mix/b3/line-nope.png", 404),
        ("/play/nothing", 404),
        ("/api/play/song?id=other", 404),
        ("/api/play/song?id=mix&bars=7", 404),
        ("/api/play/song?id=mix&bars=x", 400),
        ("/api/play/sound?id=mix&name=drums", 404),
    ):
        assert _status(url + path) == code, path
    assert b"bass2sheet" in _get(url + "/")[2]  # the copilot's own page is still where it was


def test_the_server_gives_a_song_with_its_lines(server, engraver):
    url, workdir = server
    song = json.loads(_get(url + "/api/play/song?id=mix&bars=3")[2])
    assert song["sounds"] == ["nobass", "record", "bass"] and song["speeds"] == [100, 90, 80, 70, 60, 50]
    assert song["title"] == "mix" and song["bars"] == 3 and song["images"] == "/play/mix/b3/"
    status, headers, body = _get(url + song["images"] + song["lines"][1]["file"])
    assert body[:8] == b"\x89PNG\r\n\x1a\n" and "immutable" in headers["Cache-Control"]
    assert struct.unpack(">II", body[16:24]) == (song["width"], song["height"])


def test_the_page_has_every_element_its_script_reaches_for():
    import re

    page = open(playalong.PAGE_PATH).read()
    html, script = page.split('<script id="app">')
    made = set(re.findall(r'id="([^"]+)"', html))
    missing = set(re.findall(r'\$\("([^"]+)"\)', script)) - made
    assert not missing, missing
    assert set(re.findall(r'data-sound="(\w+)"', html)) == set(playalong.SOUNDS)
    assert {int(b) for b in re.findall(r'data-bars="(\d)"', html)} == set(playalong.BARS)
    speeds = [int(v) for v in re.search(r"const SPEEDS = \[([\d, ]+)\]", script).group(1).split(",")]
    assert speeds == sorted([100, *copilot.SPEEDS])


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's logic")
def test_the_page_logic_holds():
    script = os.path.join(os.path.dirname(__file__), "playalong_page_logic.js")
    r = subprocess.run(["node", script, playalong.PAGE_PATH], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


# ------------------------------------------------------------------------------------------------ the home network


def test_only_the_machine_itself_may_change_a_score(workdir, monkeypatch):
    for address, ours in (
        ("127.0.0.1", True),
        ("127.8.9.10", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("192.168.1.20", False),
        ("10.0.0.7", False),
        ("::ffff:192.168.1.20", False),
        ("fe80::1%en0", False),
        ("203.0.113.9", False),
        ("localhost", False),
        ("", False),
    ):
        assert playalong.from_this_machine(address) is ours, address
    for address, near in (
        ("127.0.0.1", True),
        ("::1", True),
        ("192.168.4.25", True),
        ("10.1.2.3", True),
        ("172.20.10.2", True),  # a phone's hotspot
        ("::ffff:192.168.1.20", True),
        ("169.254.7.7", True),
        ("192.0.0.2", False),  # what an IPv6-only network gives a machine for its own use
        ("172.32.0.1", False),
        ("100.64.0.9", False),
        ("203.0.113.9", False),
        ("2607:fb90::1", False),
        ("nonsense", False),
    ):
        assert playalong.on_this_network(address) is near, address
    home = copilot_server.make_server(workdir, 0)
    assert home.server_address[0] == "127.0.0.1"  # this machine only, unless asked
    home.server_close()
    srv = copilot_server.make_server(workdir, 0, lan=True)
    assert srv.server_address[0] == "0.0.0.0"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        note = copilot.Desk(workdir, "mix").notes[3]
        fix = dict(song="mix", action=dict(kind="pitch", id=note["id"], pitch=note["pitch"] + 12, how="key"))
        seen = []
        monkeypatch.setattr(playalong, "from_this_machine", lambda address: seen.append(address) or False)
        for path, data in (
            ("/api/act", fix),
            ("/api/checked", dict(song="mix", start=0, end=5)),
            ("/api/sheet", dict(song="mix")),
            ("/api/anything", {}),
        ):
            assert _status(url + path, data) == 403, path  # as from another machine: refused
        state = copilot.read_json(os.path.join(workdir, "mix", "notes.json"))
        assert all(n["state"] == "machine" for n in state["notes"]) and state["checked"] == []
        assert not os.path.exists(os.path.join(workdir, "mix", "log.jsonl")) and seen == ["127.0.0.1"] * 4
        # As from another device on the network: nothing of a song without the key.
        port, key = srv.server_address[1], playalong.phone_key(workdir)
        assert re.fullmatch(r"\d{8}", key) and srv.RequestHandlerClass.door.key == key  # made once, and kept
        assert os.stat(os.path.join(workdir, "phone_key")).st_mode & 0o777 == 0o600
        songs_of = ("/api/play/songs", "/api/song?id=mix", "/data/mix/audio/record.flac", "/api/play/song?id=mix")
        for path in ("/play", "/", "/song", "/api/phone", *songs_of):
            assert _fetch(port, path)[0] == 403, path
        assert b"needs the link from your computer" in _fetch(port, "/play")[2]
        status, _, body = _fetch(port, "/play/manifest.webmanifest")  # what a phone fetches without its cookie
        assert status == 200 and "start_url" not in json.loads(body)  # the app starts where it was added: with the key
        assert _fetch(port, "/play/icon-180.png")[0] == 200
        # With the key in the link it is let in, and given a cookie to come back with.
        status, headers, body = _fetch(port, f"/play?key={key}")
        assert status == 200 and b"<title>" in body and "HttpOnly" in headers["Set-Cookie"]
        held = {"Cookie": headers["Set-Cookie"].split(";")[0]}
        assert held["Cookie"] == f"{playalong.COOKIE}={key}"
        for path in songs_of:
            assert _fetch(port, path, held)[0] == 200, path  # looking and listening
        status, headers, _ = _fetch(port, "/play?song=mix", held)  # the key goes back into the address
        assert status == 302 and headers["Location"] == f"/play?song=mix&key={key}"
        assert _fetch(port, "/api/phone", held)[0] == 404  # the link itself is told to this machine only
        assert _fetch(port, "/api/play/songs", {"Cookie": f"{playalong.COOKIE}=nonsense"})[0] == 403
        assert _status(url + "/api/act", fix) == 403  # and a device that is let in still changes nothing
        # Nobody finds the key by trying them all.
        door = srv.RequestHandlerClass.door
        for k in range(playalong.WRONG):
            assert _fetch(port, f"/play?key=x{k}")[0] == 403
        assert _fetch(port, f"/play?key={key}")[0] == 403 and door.rests_until > 0  # resting: not the right one either
        assert _fetch(port, "/api/play/songs", held)[0] == 200  # a device already let in is not shut out
        door.rests_until = 0.0
        assert _fetch(port, f"/play?key={key}")[0] == 200
        monkeypatch.setattr(playalong, "on_this_network", lambda address: False)
        for path in ("/play", f"/play?key={key}", "/api/play/songs", "/", "/api/song?id=mix", "/play/icon-180.png"):
            assert _fetch(port, path, held)[0] == 403, path  # from further away than a home network: nothing at all
        monkeypatch.undo()
        assert json.loads(_fetch(port, "/api/phone")[2])["on"] is True  # this machine: no key asked, and the link told
        assert _status(url + "/api/act", fix) == 200  # from the machine itself: as before
        assert copilot.Desk(workdir, "mix").note(note["id"])["state"] == "fixed"
    finally:
        srv.shutdown()
        srv.server_close()


def test_serving_on_the_home_network_says_where_to_open_it(workdir, monkeypatch, capsys):
    for a in playalong.addresses():  # whatever this machine has: a home network's, never the loopback
        assert playalong.on_this_network(a) and not playalong.from_this_machine(a) and not a.startswith("192.0.0.")
    monkeypatch.setattr(playalong, "addresses", lambda: ["192.168.1.20", "10.0.0.7"])
    said = playalong.announce(8770, "12345678")
    assert said[0] == "On your phone, on the same Wi-Fi, open http://192.168.1.20:8770/play?key=12345678"
    assert "http://10.0.0.7:8770/play?key=12345678" in said[1] and "only this machine can change a score" in said[-1]
    assert "any other device on the network is refused" in said[-2]
    door = playalong.Door("12345678")
    assert playalong.phone(8770, door) == dict(on=True, links=[said[0].split()[-1], said[1].split()[-1].rstrip(")")])
    assert playalong.phone(8770, None) == dict(on=False, links=[])
    assert door.knows("12345678") and not door.knows("1234567") and not door.knows("") and door.knows("12345678")
    monkeypatch.setattr(playalong, "addresses", lambda: [])
    assert "no home-network address" in playalong.announce(8770, "12345678")[0]
    assert playalong.phone(8770, door) == dict(on=True, links=[])
    asked = []
    monkeypatch.setattr(copilot_server, "serve", lambda *args: asked.append(args))
    assert copilot.main(["serve", "--dir", workdir, "--port", "8799", "--lan"]) == 0
    assert copilot.main(["serve", "--dir", workdir]) == 0
    assert asked == [(workdir, 8799, True), (workdir, 8770, False)]


# ------------------------------------------------------------------------------- what goes wrong, and what is said


def _program(tmp_path, name, body):
    import stat
    import sys

    path = str(tmp_path / name)
    with open(path, "w") as f:
        f.write(f"#!{sys.executable}\nimport json, os, sys, time\n{body}")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def test_a_musescore_that_fails_says_so_and_nothing_half_made_is_left(workdir, tmp_path, monkeypatch):
    tests = os.path.dirname(os.path.abspath(__file__))
    without_pictures = (  # the positions of everything, and no picture of any page
        f"sys.path.insert(0, {tests!r})\nimport mscore_standin\n"
        "for job in json.load(open(sys.argv[sys.argv.index('-j') + 1])):\n"
        "    outs = [out for out in job['out'] if not out.endswith('.png')]\n"
        "    mscore_standin.engrave(job['in'], outs, open(sys.argv[2]).read(), 360.0)\n"
    )
    folder = os.path.join(workdir, "mix", "play", "b3")
    for body, said in (
        (
            "sys.stderr.write('sandbox_extension: noise\\nno display to draw on\\n')\nsys.exit(1)\n",
            r"exit 1\): no display",
        ),
        ("sys.exit(0)\n", r"MuseScore did not engrave the score \(exit 0\)"),  # it says it is done and wrote nothing
        (without_pictures, "MuseScore wrote no picture of one of the score's pages"),
    ):
        monkeypatch.setattr(score, "MSCORE_CANDIDATES", [_program(tmp_path, "mscore", body)])
        with pytest.raises(Bass2SheetError, match=said):
            playalong.lines(workdir, "mix", 3)
        assert os.listdir(folder) == []  # no picture, no score.json, and its work folder is gone
    machine, mine = (copilot.read_json(os.path.join(workdir, "mix", name)) for name in ("machine.json", "notes.json"))
    xml, first_bar, starts, sounding = playalong.score_xml(machine, mine["notes"], 3)
    stuck = _program(tmp_path, "mscore", "time.sleep(30)\n")
    with pytest.raises(Bass2SheetError, match="MuseScore took more than 0.5 s over the score"):
        playalong.engrave(
            xml, first_bar, machine["grid"], machine["origin"], starts, sounding, folder, stuck, timeout=0.5
        )
    assert os.listdir(folder) == []
    with pytest.raises(Bass2SheetError, match="nope is not prepared"):
        playalong.lines(workdir, "nope", 3)


def test_where_the_grid_or_the_layout_has_nothing_to_go_by():
    assert playalong.grid_time([], 5) == 0.0  # no grid at all: the start
    assert playalong.grid_time([1.5], 7) == 1.5 and playalong.grid_time([1.5], -3) == 1.5  # one moment: that one
    with pytest.raises(Bass2SheetError, match="MuseScore laid out no measures"):
        playalong.lines_of([], [], 1, [0.0, 0.125], 0)


def test_a_sound_that_cannot_be_made_is_said_once_and_then_tried_again(workdir, monkeypatch):
    def cannot(*args):
        raise Bass2SheetError("could not slow the song down (ffmpeg): no such filter")

    monkeypatch.setattr(copilot, "slowed", cannot)
    maker = playalong.Sounds(workdir)
    assert maker.ask("mix", "record", 60) == dict(ready=False)  # it is being made, as far as the page knows
    with pytest.raises(Bass2SheetError, match="no such filter"):
        for _ in range(500):
            maker.ask("mix", "record", 60)
            time.sleep(0.01)
    assert maker.ask("mix", "record", 60) == dict(ready=False)  # asked again, it is tried again
    with pytest.raises(Bass2SheetError, match="nope is not prepared"):
        playalong.sound(workdir, "nope", "record")
    with pytest.raises(Bass2SheetError, match="no record at 85%"):
        playalong.sound(workdir, "mix", "record", 85)


def test_the_record_without_its_bass_is_made_from_the_separators_stems_in_the_cache(workdir, tmp_path):
    from bass2sheet.common import sha1_file
    from bass2sheet.separate import load_stems

    first = _audio(workdir, "nobass")
    out = os.path.join(workdir, "mix", "audio", "nobass.flac")
    songs = copilot.load_songs(workdir)
    stems = load_stems(songs["mix"]["stems_dir"], first.shape[1])
    songs["mix"]["stems_dir"] = None  # a song the tool separated itself: its stems are in the cache, by the song
    with open(os.path.join(workdir, "songs.json"), "w") as f:
        json.dump(songs, f)
    cache = tmp_path / "cache"
    cache.mkdir()
    with open(os.path.join(workdir, "settings.json"), "w") as f:
        json.dump(dict(cache_dir=str(cache)), f)
    kept = cache / f"stems-{sha1_file(songs['mix']['path'])[:16]}-roformer.npz"
    np.savez(kept, **stems)
    os.remove(out)
    playalong.backing(workdir, "mix")
    assert float(np.abs(_audio(workdir, "nobass") - first).max()) < 2e-3  # the same sum a new song gets
    np.savez(kept, bass=stems["bass"])  # a cache that holds the bass and nothing else: nothing to add up
    os.remove(out)
    playalong.backing(workdir, "mix")  # the isolated bass is taken out of the record instead
    assert float(np.corrcoef(_audio(workdir, "nobass").ravel(), first.ravel())[0, 1]) > 0.99
    kept.write_bytes(b"not what was kept")  # a file that cannot be read is not the end of it either
    os.remove(out)
    playalong.backing(workdir, "mix")
    assert float(np.corrcoef(_audio(workdir, "nobass").ravel(), first.ravel())[0, 1]) > 0.99
    assert playalong._fitted(np.zeros(10), np.ones(10)) == (0.0, 1.0)  # silence holds nothing of anything


def test_sound_is_served_in_the_ranges_a_phone_asks_for(server, monkeypatch):
    url, workdir = server
    record = url + "/play/mix/audio/record.flac"
    size = os.path.getsize(os.path.join(workdir, "mix", "audio", "record.flac"))
    status, headers, body = _get(record, {"Range": "bytes=-4"})  # its last four bytes
    assert (status, len(body), headers["Content-Range"]) == (206, 4, f"bytes {size - 4}-{size - 1}/{size}")
    request = urllib.request.Request(record, headers={"Range": f"bytes={size + 5}-"})
    with pytest.raises(urllib.error.HTTPError) as e:  # from beyond its end
        urllib.request.urlopen(request, timeout=30)
    assert e.value.code == 416 and e.value.headers["Content-Range"] == f"bytes */{size}"
    with urllib.request.urlopen(urllib.request.Request(record, method="HEAD"), timeout=30) as r:
        assert (r.status, r.read(), int(r.headers["Content-Length"])) == (200, b"", size)
    # the silent film that keeps the screen lit: where it cannot be made, the page is told it is not there
    import imageio_ffmpeg

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/nonexistent/ffmpeg")
    assert playalong.awake(workdir) is None and _status(url + "/play/awake.mp4") == 404
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/usr/bin/false")
    assert playalong.awake(workdir) is None and not os.path.exists(os.path.join(workdir, ".play-awake.mp4"))


def test_the_addresses_a_phone_can_reach_are_the_home_networks(tmp_path, monkeypatch):
    import socket

    listed = (
        "lo0: flags=8049<UP,LOOPBACK>\\n\\tinet 127.0.0.1 netmask 0xff000000\\n"
        "en0: flags=8863<UP>\\n\\tinet 10.0.0.7 netmask 0xffffff00\\n\\tinet addr:192.168.1.20  Bcast:192.168.1.255\\n"
        "\\tinet 192.0.0.2 netmask 0xffffffff\\n\\tinet 8.8.8.8 netmask 0xffffff00\\n\\tinet6 fe80::1%lo0\\n"
    )
    program = _program(tmp_path, "ifconfig", f"print('{listed}')\n")

    def no_socket(*args, **kw):
        raise OSError("no network")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "gethostbyname_ex", no_socket)
    monkeypatch.setattr(playalong.shutil, "which", lambda name: program)
    # the home network's own first; not the loopback, a hotspot's 192.0.0.x or an address the internet reaches
    assert playalong.addresses() == ["192.168.1.20", "10.0.0.7"]
    assert playalong.links(8770, "12345678") == [
        "http://192.168.1.20:8770/play?key=12345678",
        "http://10.0.0.7:8770/play?key=12345678",
    ]
    monkeypatch.setattr(playalong.shutil, "which", lambda name: "/nonexistent/ifconfig")
    assert playalong.addresses() == []  # nothing to ask: no address, and no failure
    assert playalong.announce(8770, "12345678")[0].startswith("Listening on the network, but this machine has no")


def test_a_phone_key_that_is_not_one_is_made_again(tmp_path):
    work = str(tmp_path / "scores")
    key = playalong.phone_key(work)  # the folder is made for it
    assert re.fullmatch(r"\d{8}", key) and playalong.phone_key(work) == key  # made once
    with open(os.path.join(work, "phone_key"), "w") as f:
        f.write("let me in\n")
    again = playalong.phone_key(work)
    assert re.fullmatch(r"\d{8}", again) and open(os.path.join(work, "phone_key")).read() == again + "\n"
    assert os.stat(os.path.join(work, "phone_key")).st_mode & 0o777 == 0o600
