"""Judging with a person in the loop: versions, passages, picks, the page's server and the loop after a change."""

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from bass2sheet import judge, judge_server
from bass2sheet.common import DEFAULTS, SR, Bass2SheetError
from bass2sheet.score import find_mscore

CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")
KEY = ("E minor", 4, False)


def _note(g, length, pitch):
    return dict(start=g * 0.125, end=(g + length) * 0.125, pitch=pitch, grid_start=g, grid_len=length, flag=None)


def _heard(notes, bars=16, offset=0.0, origin=0):
    grid = offset + 0.125 * np.arange(16 * bars + origin + 1)
    return dict(notes=notes, grid=grid, origin=origin, tempo=120.0, key=KEY, chords=[])


def _version(notes, **settings):
    return judge.make_version(dict(DEFAULTS, **settings), _heard(notes))


def _riff(bars=12, change=()):
    """Quarter notes E G A B in every bar; in the bars named in `change`, beat 3 is a C instead of an A."""
    out = []
    for bar in range(bars):
        for beat, pitch in enumerate((40, 43, 45, 47)):
            out.append(_note(16 * bar + 4 * beat, 4, 48 if bar + 1 in change and beat == 2 else pitch))
    return out


# ---------------------------------------------------------------------------------------------------- versions


def test_a_version_is_plain_data_written_under_its_settings():
    notes = [_note(0, 1, 40), _note(2, 1, 43), _note(4, 12, 45)]
    literal, simple = _version(notes), _version(notes, rhythm="simple", tab="position")
    assert json.loads(json.dumps(literal)) == literal
    assert [n["grid_len"] for n in literal["notes"]] == [1, 1, 12]
    assert [n["grid_len"] for n in simple["notes"]] == [2, 2, 12]  # short rests closed
    assert (literal["tab"], simple["tab"]) == ("low", "position")
    assert literal["key"] == ["E minor", 4, False] and literal["grid"][8] == 1.0


def test_snapshots_are_named_by_what_they_write_and_never_overwritten(tmp_path):
    a, b = _version(_riff()), _version(_riff(change={3}))
    assert judge.version_hash(a) == judge.version_hash(_version(_riff())) != judge.version_hash(b)
    assert judge.version_hash(a) != judge.version_hash(_version(_riff(), tab="position"))
    rel = judge.save_version(str(tmp_path), "song", a)
    stamp = os.path.getmtime(tmp_path / rel)
    assert judge.save_version(str(tmp_path), "song", a) == rel and os.path.getmtime(tmp_path / rel) == stamp
    assert judge.load_version(str(tmp_path), rel) == a


def test_song_entry_reads_artist_and_title_from_the_file_name(tmp_path):
    e = judge.song_entry(str(tmp_path / "Some Band - A Song (Official Video) [abc123XYZ_-].opus"))
    assert (e["id"], e["artist"], e["title"]) == ("some-band-a-song", "Some Band", "A Song")
    assert judge.song_entry("untitled.wav")["artist"] == ""


# ---------------------------------------------------------------------------------------------------- passages


def test_passages_are_where_the_versions_differ_and_a_repeated_figure_is_judged_once():
    a, b = _version(_riff()), _version(_riff(change={2, 6, 10}))  # the same change in three four-bar phrases
    assert sorted(bar for bar, _, _ in judge.unmatched(a, b, "notes")) == [2, 2, 6, 6, 10, 10]
    assert [w["bars"] for w in judge.pick_windows(a, b, "notes", 3)] == [(1, 4)]
    c = _version(_riff(change={2, 7}))  # two different places in the phrase: two figures
    assert [w["bars"] for w in judge.pick_windows(a, c, "notes", 3)] == [(1, 4), (5, 8)]
    assert judge.pick_windows(a, a, "notes", 3) == []
    assert judge.pick_windows(a, c, "notes", 1) == [dict(bars=(1, 4), weight=2)]


def test_a_passage_where_one_version_has_almost_nothing_is_judged_last():
    full = _riff()
    thin = [n for n in full if not 0 <= n["grid_start"] < 64]  # bars 1-4 missing altogether
    thin[4] = dict(thin[4], pitch=50)  # and one wrong note in bars 5-8
    a, b = _version(full), _version(thin)
    assert [w["bars"] for w in judge.pick_windows(a, b, "notes", 1)] == [(5, 8)]
    assert [w["bars"] for w in judge.pick_windows(a, b, "notes", 2)] == [(1, 4), (5, 8)]


def test_writing_and_tab_differences_are_found_on_the_same_notes():
    notes = [_note(16 * bar + g, 2, 33 + (g // 4) % 4) for bar in range(4) for g in range(0, 16, 4)]  # eighth, rest
    a = _version(notes)
    assert judge.unmatched(a, _version(notes, rhythm="simple"), "notes") == []
    assert judge.pick_windows(a, _version(notes, rhythm="simple"), "writing", 2)[0]["bars"] == (1, 4)
    assert judge.pick_windows(a, _version(notes, tab="position"), "tab", 2)[0]["bars"] == (1, 4)


def test_versions_on_two_beat_grids_are_compared_by_time():
    a = _version(_riff())
    late = judge.make_version(DEFAULTS, _heard(_riff(), offset=0.02))  # the same notes, 20 ms later
    assert not judge.same_grid(a, late) and judge.grids_agree(a, late)
    assert judge.unmatched(a, late, "notes") == []
    shifted = judge.make_version(DEFAULTS, _heard(_riff(), origin=4))  # bar lines a beat later
    assert not judge.grids_agree(a, shifted)
    spread = judge.pick_windows(a, shifted, "bars", 3)
    assert [w["bars"] for w in spread] == [(1, 4), (5, 8), (9, 12)]
    assert judge.bars_covering(shifted, judge.bar_time(a, 5), judge.bar_time(a, 9)) == (5, 8)


def test_a_passage_is_the_same_when_its_notes_are_within_a_few_hundredths():
    a, b = _version(_riff()), judge.make_version(DEFAULTS, _heard(_riff(), offset=0.03))
    p = judge.passage(a, 2.0, 4.0, "notes")
    assert p == [[0.0, 40], [0.5, 43], [1.0, 45], [1.5, 47]]
    assert judge.same_passage(p, judge.passage(b, 2.0, 4.0, "notes"))
    assert not judge.same_passage(p, judge.passage(_version(_riff(change={2})), 2.0, 4.0, "notes"))
    assert not judge.same_passage(p, p[:-1])
    tab = judge.passage(a, 2.0, 4.0, "tab")
    assert not judge.same_passage(tab, judge.passage(_version(_riff(), tab="position"), 2.0, 4.0, "tab"))
    assert judge.passage(a, 2.0, 4.0, "writing")[0] == [0.0, 40, 0.5]


def test_the_cursor_follows_the_engraved_notes_through_the_tracked_beats(tmp_path):
    """Positions as MuseScore reports them (twelfths of a pixel; milliseconds at the printed tempo) become
    [seconds into the audio, x] on the tracked beat grid, which here runs 10% slow."""
    import struct

    heard = _heard(_riff())
    heard["grid"] = heard["grid"] * 1.1
    v = judge.make_version(DEFAULTS, heard)
    xs, ms = [2400, 4800, 7200, 9600, 12000], [0, 500, 1000, 1500, 2000]  # quarter notes at 120, bar 2 beat 1
    spos = "<score><elements>" + "".join(
        f'<element id="{k}" x="{x}" y="3000" sx="161" sy="1860" page="0"/>' for k, x in enumerate(xs)
    )
    spos += "</elements><events>" + "".join(f'<event elid="{k}" position="{m}"/>' for k, m in enumerate(ms))
    (tmp_path / "A.spos").write_text(spos + "</events></score>")
    (tmp_path / "A.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 1650, 600))
    sheet = judge.sheet_data(str(tmp_path / "A.png"), str(tmp_path / "A.spos"), v, (2, 2), 2.0, marks=[20, 99])
    half = judge.NOTEHEAD / 2
    assert sheet["keys"][:5] == [[round(0.2 + 0.55 * k, 3), round(x / 12 + half, 1)] for k, x in enumerate(xs)]
    assert sheet["keys"][5] == [
        round(4.4 - 2.0, 3),
        judge.SNIPPET_RIGHT - 3,
    ]  # the end of the bar, at the last bar line
    assert sheet["page"] == [1650, 600] and sheet["staff"] == [242.0, 413.0]
    assert sheet["marks"] == [round(4800 / 12 + half, 1)]  # sixteenth 20 is beat 2 of bar 2; 99 is not on the page
    assert sheet["crop"] == [22, 127, 1628, 435]  # the image cannot be read here: a generous box round the staves


def test_differing_notes_are_listed_for_each_version():
    a, b = _version(_riff()), _version(_riff(change={2}))
    only_a, only_b = judge.unmatched_events(a, b, "notes")
    assert [(e["g0"], e["pitch"]) for e in only_a] == [(24, 45)] and [(e["g0"], e["pitch"]) for e in only_b] == [
        (24, 48)
    ]
    late = judge.make_version(DEFAULTS, _heard(_riff(change={2}), offset=0.02))
    only_a, only_b = judge.unmatched_events(a, late, "notes")
    assert [e["pitch"] for e in only_a] == [45] and [e["pitch"] for e in only_b] == [48]


def test_the_count_marks_beat_one_higher_than_the_other_beats():
    v = _version(_riff(bars=2))
    y = judge.clicks(v, 0.0, 2.0)
    pitch = lambda t: np.argmax(np.abs(np.fft.rfft(y[int(t * SR) : int(t * SR) + 1024]))) * SR / 1024
    assert abs(pitch(0.0) - 1760) < 50 and abs(pitch(0.5) - 1100) < 50
    assert np.abs(y[int(0.25 * SR) : int(0.45 * SR)]).max() < 1e-6  # silence between beats


# ------------------------------------------------------------------------------------------------------- picks


def _comp(cid, choice="separator", flip=False, **kw):
    values = dict(separator=("roformer", "demucs"), replucks=("every", "clear"), tab=("low", "position"))[choice]
    labels = judge.choice_of(choice)["labels"]
    c = dict(
        id=cid,
        song="song-one",
        title="Song One",
        artist="The Band",
        choice=choice,
        kind=judge.choice_of(choice)["kind"],
        round=int(cid.rsplit(".", 1)[1]),
        a="versions/a.json",
        b="versions/b.json",
        a_value=values[0],
        b_value=values[1],
        a_label=labels[values[0]] + " (the default)",
        b_label=labels[values[1]],
        flip=flip,
        t0=10.0,
        t1=18.0,
        bars_a=[5, 8],
        bars_b=[5, 8],
        weight=3,
        lead=0.4,
        length=8.8,
        same_audio=choice == "tab",
        media={k: f"media/{cid}/{k}.flac" for k in ("record", "A", "B", "record_bass", "A_bass", "B_bass")},
        sheets={
            name: dict(
                src=f"media/{cid}/{name}.png",
                page=[1650, 600],
                crop=[36, 161, 1612, 425],
                staff=[247.0, 418.0],
                keys=[[0.4, 216.0], [2.4, 520.0], [8.4, 1602.0]],
                marks=[300.0] if name == "A" and choice == "separator" else [],
            )
            for name in "AB"
        },
    )
    if choice == "tab":
        del c["media"]["B"], c["media"]["B_bass"]
    if choice == "replucks":  # as the second step is made: a pitch picture with the notes and the pitch line on it
        c["media"] = {k: f"media/{cid}/{k}.flac" for k in ("record", "A", "B")}
        c["sources"] = [["record", "Isolated bass"], ["A", "A"], ["B", "B"]]
        c["alt"] = dict(label="Real bass left, notes right", media={k: f"media/{cid}/{k}_both.flac" for k in "AB"})
        roll = dict(lo=26, hi=62, notes=[[0.5, 1.0, 40, 0], [1.2, 1.6, 43, 1]])
        roll["line"] = [[0.5, 40.0], [0.52, 40.1], None, [1.2, 43.0], [1.22, 43.0], [1.24, 43.1]]
        view = dict(page=[1200, 333], crop=[0, 0, 1200, 333], staff=[0, 333], keys=[[0.0, 0.0], [8.8, 1200]], marks=[])
        c["sheets"] = {name: dict(view, src=f"media/{cid}/picture.png", roll=roll) for name in "AB"}
    c.update(kw)
    return c


@pytest.fixture
def workdir(tmp_path):
    comps = [
        _comp("song-one.separator.1", flip=True),
        _comp("song-one.replucks.1", "replucks"),
        _comp("song-one.tab.1", "tab"),
    ]
    judge.save_comparisons(str(tmp_path), comps)
    for c in comps:
        os.makedirs(tmp_path / "media" / c["id"])
        (tmp_path / "media" / c["id"] / "A.flac").write_bytes(bytes(range(256)) * 4)
    return str(tmp_path)


def test_a_pick_names_a_version_through_the_random_order(workdir):
    flipped, straight, _ = judge.load_comparisons(workdir)
    assert judge.side_of(flipped, "A") == "b" and judge.side_of(flipped, "B") == "a"
    assert judge.side_of(straight, "A") == "a" and judge.side_of(straight, "same") is None
    assert judge.shown_labels(flipped)["A"] == "bass isolated by Demucs"
    assert judge.shown_labels(straight)["A"].endswith("(the default)")
    assert judge.record_judgment(workdir, flipped, "A", " muddy ")["value"] == "demucs"
    with pytest.raises(Bass2SheetError, match="unknown pick"):
        judge.record_judgment(workdir, flipped, "C")


def test_the_latest_pick_counts_and_the_tally_is_by_alternative(workdir):
    sep, rep, tab = judge.load_comparisons(workdir)
    judge.record_judgment(workdir, sep, "A", "muddy")  # Demucs
    judge.record_judgment(workdir, sep, "B")  # changed mind: BS-RoFormer
    judge.record_judgment(workdir, tab, "same")
    t = judge.tally(workdir)
    assert t["separator"]["picks"] == dict(roformer=1, demucs=0) and t["separator"]["notes"] == []
    assert (t["tab"]["same"], t["tab"]["open"], t["replucks"]["open"]) == (1, 0, 1)
    text = judge.report(workdir)
    assert "| bass isolated by BS-RoFormer | 1 |" in text and "| Not judged yet | 1 |" in text
    assert "No comparisons yet" in judge.report(os.path.join(workdir, "empty"))


def test_comparisons_are_shown_step_by_step_with_changed_passages_first():
    comps = [_comp("s.tab.2", "tab"), _comp("s.separator.2"), _comp("s.tab.1", "tab"), _comp("s.separator.1")]
    comps.append(dict(_comp("s.change.1"), choice="change"))
    shown = [c["id"] for c in judge.order(comps)]
    assert shown == ["s.change.1", "s.separator.1", "s.separator.2", "s.tab.1", "s.tab.2"]


# ------------------------------------------------------------------------------------------------- the server


@pytest.fixture
def server(workdir):
    srv = judge_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_the_page_is_told_nothing_that_tells_the_versions_apart(server):
    url, _ = server
    s = json.loads(_get(url + "/api/next")[2])
    c = s["comparison"]
    assert (s["total"], s["judged"], c["id"]) == (3, 0, "song-one.separator.1")
    assert c["where"] == "bars 5–8 · 0:10–0:18"
    text = json.dumps(c)
    for secret in ("flip", "a_value", "b_label", "roformer", "demucs", "default", "versions/"):
        assert secret not in text, secret
    assert c["sheets"]["A"]["src"] == "/media/song-one.separator.1/A.png" and c["sheets"]["B"]["keys"][0] == [
        0.4,
        216.0,
    ]
    assert b"<title>bass2sheet" in _get(url + "/")[2]


def test_a_pick_is_saved_and_answered_with_what_the_versions_were(server):
    url, workdir = server
    s = _post(url + "/api/judge", dict(id="song-one.separator.1", pick="A", note="muddy"))
    assert s["reveal"] == dict(A="bass isolated by Demucs", B="bass isolated by BS-RoFormer (the default)")
    assert s["judged"] == 1 and s["comparison"]["pick"] == "A" and s["comparison"]["note"] == "muddy"
    assert judge.load_judgments(workdir)["song-one.separator.1"]["value"] == "demucs"
    nxt = json.loads(_get(url + "/api/next?after=song-one.separator.1")[2])
    assert nxt["comparison"]["id"] == "song-one.replucks.1" and nxt["previous"] == "song-one.separator.1"
    wrap = json.loads(_get(url + "/api/next?after=song-one.tab.1")[2])  # past the end: back to what was skipped
    assert wrap["comparison"]["id"] == "song-one.replucks.1"
    again = json.loads(_get(url + "/api/next?id=song-one.separator.1")[2])["comparison"]
    assert again["reveal"]["A"] == "bass isolated by Demucs"
    for bad, code in ((dict(id="nope", pick="A"), 404), (dict(id="song-one.tab.1", pick="C"), 400)):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(url + "/api/judge", bad)
        assert e.value.code == code
    assert json.loads(_get(url + "/api/results")[2])["separator"]["picks"]["demucs"] == 1


def test_media_is_served_in_byte_ranges_and_nothing_else_is(server):
    url, _ = server
    media = url + "/media/song-one.separator.1/A.flac"
    status, headers, body = _get(media)
    assert (status, len(body), headers["Content-Type"], headers["Accept-Ranges"]) == (200, 1024, "audio/flac", "bytes")
    status, headers, body = _get(media, {"Range": "bytes=256-259"})
    assert (status, body, headers["Content-Range"]) == (206, bytes([0, 1, 2, 3]), "bytes 256-259/1024")
    assert _get(media, {"Range": "bytes=-2"})[2] == bytes([254, 255])
    for path in ("/media/../comparisons.json", "/media/song-one.separator.1/missing.flac", "/comparisons.json"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(url + path)
        assert e.value.code == 404


def test_the_page_has_every_element_its_script_reaches_for():
    import re

    html, script = judge_server.PAGE.split("<script>")
    missing = set(re.findall(r'\$\("([^"]+)"\)', script)) - set(re.findall(r'id="([^"]+)"', html))
    assert not missing, missing


def test_serving_an_empty_directory_says_what_to_do(tmp_path):
    with pytest.raises(Bass2SheetError, match="bass2sheet-judge build"):
        judge_server.make_server(str(tmp_path), 0)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's script")
def test_the_page_script_works_against_the_server(server):
    url, workdir = server
    script = os.path.join(os.path.dirname(__file__), "judge_page_smoke.js")
    r = subprocess.run(["node", script, url], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
    picks = judge.load_judgments(workdir)
    assert picks["song-one.separator.1"]["note"] == "too busy" and picks["song-one.tab.1"]["pick"] == "same"


# ------------------------------------------------------------------------------------ pictures and stem media


def test_a_png_is_written_without_an_imaging_library(tmp_path):
    import struct
    import zlib

    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[1, 2] = (255, 128, 0)
    judge.write_png(str(tmp_path / "x.png"), rgb)
    data = (tmp_path / "x.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (5, 4)
    start = data.index(b"IDAT") + 4
    raw = zlib.decompress(data[start : data.index(b"IEND") - 8])
    rows = [raw[k * 16 + 1 : k * 16 + 16] for k in range(4)]  # each row: a filter byte, then 5 pixels
    assert rows[1][6:9] == bytes([255, 128, 0]) and rows[0] == bytes(15)


def _tone(pitch, start, end, total=3.0):
    t = np.arange(int(total * SR)) / SR
    y = 0.4 * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * t)
    return (y * ((t >= start) & (t < end))).astype(np.float32)


def test_the_pitch_picture_shows_a_note_where_and_when_it_sounds(tmp_path):
    c = judge.spectrum(_tone(40, 1.0, 2.0), 0.5, 2.5)
    assert c.shape == (3 * (judge.ROLL_HI - judge.ROLL_LO + 1), judge.ROLL_W)
    row = 3 * (judge.ROLL_HI - 40) + 1  # the middle of the three rows of E2
    inside, before = c[:, 450:750], c[:, :150]  # 1.25-1.75 s and 0.5-0.75 s
    assert abs(int(np.argmax(inside.mean(1))) - row) <= 1 and inside[row].mean() > 30 * before[row].mean()
    size = judge.write_picture(str(tmp_path / "p.png"), c, float(c.max()), other=np.zeros_like(c))
    assert size == [judge.ROLL_W, 3 * c.shape[0]]
    view = judge.roll_view("media/x/p.png", size, 2.0, [[0.5, 1.5, 40, 1]], [[0.5, 40.0], None])
    assert view["keys"] == [[0.0, 0.0], [2.0, judge.ROLL_W]] and view["crop"] == [0, 0, *size] and view["marks"] == []
    assert view["roll"] == dict(lo=judge.ROLL_LO, hi=judge.ROLL_HI, notes=[[0.5, 1.5, 40, 1]], line=[[0.5, 40.0], None])


def test_heard_boxes_are_the_notes_as_played_and_flag_the_ones_only_one_version_has():
    held = [dict(start=1.0, end=2.0, pitch=40), dict(start=2.5, end=3.0, pitch=43)]
    split = [
        dict(start=1.0, end=1.5, pitch=40),
        dict(start=1.52, end=2.0, pitch=40),
        dict(start=2.53, end=3.5, pitch=43),
    ]
    a, b = dict(notes=held), dict(notes=split)
    assert judge.heard_boxes(b, a, 0.5, 3.2) == [[0.5, 1.0, 40, 0], [1.02, 1.5, 40, 1], [2.03, 2.7, 43, 0]]
    assert judge.heard_boxes(a, b, 0.5, 3.2) == [[0.5, 1.5, 40, 0], [2.0, 2.5, 43, 0]]
    assert judge.heard_boxes(a, b, 2.2, 2.4) == []


class _Audio:
    """A song as one separator took it apart, made up for the test."""

    def __init__(self, bass, rest):
        self.bass_mono = bass
        self.bass, self.rest = np.stack([bass, bass]).astype(float), np.stack([rest, rest]).astype(float)
        self.mix, self.n = self.bass + self.rest, len(bass)


def test_the_first_step_is_judged_on_the_isolated_bass_tracks_themselves(tmp_path):
    import random

    import soundfile as sf

    drums = (0.2 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32)
    whole, part = _tone(40, 0.2, 2.8), _tone(40, 0.2, 1.4)  # one separator loses the second half of the note
    audio = dict(roformer=_Audio(whole, drums), demucs=_Audio(part, drums + (whole - part)))
    notes = [_note(k * 4, 4, 40) for k in range(6)]
    va = judge.make_version(DEFAULTS, _heard(notes, bars=2))
    vb = judge.make_version(dict(DEFAULTS, separator="demucs"), _heard(notes[:3], bars=2))
    song, comps = dict(id="s", title="S", artist=""), []
    window = dict(bars=(1, 1), weight=3)
    comp = judge.add_comparison(
        comps,
        song,
        "separator",
        "stem",
        va,
        vb,
        "a.json",
        "b.json",
        window,
        random.Random(1),
        ("roformer", "demucs"),
        "roformer",
    )
    judge.render_comparison(comp, va, vb, lambda v: audio[v["settings"]["separator"]], str(tmp_path))
    assert comp["id"] == "s.separator.1" and comp["sources"] == [["record", "Record"], ["A", "A"], ["B", "B"]]
    assert sorted(comp["media"]) == ["A", "B", "record"] and comp["alt"]["label"] == "What was left behind"
    assert sorted(comp["alt"]["media"]) == ["A", "B"] and comp["sheet_rev"] == judge.SHEET_REV
    read = lambda rel: sf.read(os.path.join(tmp_path, rel))[0]
    whole_name, part_name = ("B", "A") if comp["flip"] else ("A", "B")
    rms = lambda y, a, b: float(np.sqrt(np.mean(y[int(a * SR) : int(b * SR)] ** 2)))
    late = (1.6, 2.2)  # seconds into the clip (the clip starts where the passage does: time 0)
    assert rms(read(comp["media"][part_name]), *late) < 0.01 * rms(read(comp["media"][whole_name]), *late)
    assert rms(read(comp["alt"]["media"][part_name]), *late) > 1.5 * rms(read(comp["alt"]["media"][whole_name]), *late)
    assert comp["sheets"]["A"]["src"].endswith("A_picture.png") and comp["sheets"]["A"]["roll"]["notes"] == []
    for sheet in comp["sheets"].values():
        assert os.path.getsize(os.path.join(tmp_path, sheet["src"])) > 500 and sheet["page"] == [judge.ROLL_W, 333]


# ------------------------------------------------------------------------------- the steps of a song, in order


@pytest.fixture(scope="module")
def built(taste_song_files, cache_dir, tmp_path_factory):
    """A synthetic song with its stems given (so there is no separator to judge) and three later steps that
    need no model download. The tests below walk through it in order."""
    mix, stems = taste_song_files
    workdir = str(tmp_path_factory.mktemp("judging"))
    said = []
    kw = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    n = judge.build(
        [mix], workdir, ["replucks", "rhythm", "tab"], per_song=2, stems_dirs={mix: stems}, say=said.append, **kw
    )
    return dict(workdir=workdir, added=n, said=said, mix=mix, stems=stems, kw=kw)


def _pick(workdir, comp, value):
    """Judge a comparison for the version with this value, through the random order it is shown in."""
    judge.record_judgment(workdir, comp, "A" if (comp["a_value"] == value) != comp["flip"] else "B")


def _live(workdir):
    return [c for c in judge.live_comparisons(workdir)]


def test_build_opens_the_first_step_only_and_shows_the_notes_as_heard(built):
    workdir = built["workdir"]
    comps = judge.load_comparisons(workdir)
    assert built["added"] == len(comps) > 0, built["said"]
    assert {(c["choice"], c["kind"], c["base_key"]) for c in comps} == {("replucks", "heard", "")}
    assert [p["total"] for p in judge.progress(workdir)] == [len(comps), 0, 0]
    assert judge.read_json(workdir, "songs.json", {})["mix"]["built"] == {"replucks": {"": len(comps)}}
    assert judge.read_settings(workdir)["choices"] == ["replucks", "rhythm", "tab"]
    import soundfile as sf

    for c in comps:
        assert c["a_label"].endswith("(the default)") and not c["b_label"].endswith("(the default)")
        assert c["sources"][0] == ["record", "Isolated bass"] and c["alt"]["label"] == "Real bass left, notes right"
        for rel in [*c["media"].values(), *c["alt"]["media"].values()]:
            y, sr = sf.read(os.path.join(workdir, rel))
            assert sr == SR and abs(len(y) / sr - c["length"]) < 0.01 and 0.05 < np.abs(y).max() <= 1.0
        both = sf.read(os.path.join(workdir, c["alt"]["media"]["A"]))[0]
        assert not np.allclose(both[:, 0], both[:, 1])  # the real bass in one ear, the notes in the other
        a, b = c["sheets"]["A"], c["sheets"]["B"]
        assert a["src"] == b["src"] and os.path.getsize(os.path.join(workdir, a["src"])) > 1000
        assert a["roll"]["line"] == b["roll"]["line"] and sum(p is not None for p in a["roll"]["line"]) > 50
        assert a["roll"]["notes"] != b["roll"]["notes"]
        assert any(n[3] for n in a["roll"]["notes"] + b["roll"]["notes"])  # what the two versions disagree on
        for t0, t1, pitch, _ in a["roll"]["notes"] + b["roll"]["notes"]:
            assert 0 <= t0 < t1 <= c["length"] + 1e-6 and judge.ROLL_LO <= pitch <= judge.ROLL_HI


def test_judging_a_step_opens_the_next_one_built_on_that_pick(built):
    workdir = built["workdir"]
    first = _live(workdir)
    for c in first:
        _pick(workdir, c, "clear")
    assert judge.advance(workdir, say=lambda _: 0) > 0
    live = _live(workdir)
    rhythm = [c for c in live if c["choice"] == "rhythm"]
    assert rhythm and {c["base_key"] for c in rhythm} == {"replucks=clear"} and len(live) == len(first) + len(rhythm)
    for c in rhythm:
        assert (c["kind"], c["a_value"], c["b_value"]) == ("writing", "literal", "simple")
        assert judge.load_version(workdir, c["a"])["settings"]["replucks"] == "clear"
        assert c["alt"]["label"] == "Bass only" and sorted(c["alt"]["media"]) == ["A", "B", "record"]
    assert judge.advance(workdir, say=lambda _: 0) == 0  # the tab waits for the rhythm


def test_changing_an_earlier_pick_retires_what_was_built_on_it(built):
    workdir = built["workdir"]
    rhythm = lambda: [c for c in _live(workdir) if c["choice"] == "rhythm"]
    first = [c for c in _live(workdir) if c["choice"] == "replucks"]
    on_clear = {c["id"] for c in rhythm()}
    for c in first:
        _pick(workdir, c, "every")
    assert rhythm() == []  # at once, before anything is rebuilt
    judge.advance(workdir, say=lambda _: 0)
    on_every = {c["id"] for c in rhythm()}
    assert on_every and not on_every & on_clear and {c["base_key"] for c in rhythm()} == {"replucks=every"}
    _pick(workdir, rhythm()[0], "simple")
    assert judge.tally(workdir)["rhythm"]["picks"]["simple"] == 1
    for c in first:  # back again: the comparisons made before count again and nothing is rebuilt
        _pick(workdir, c, "clear")
    assert {c["id"] for c in rhythm()} == on_clear and judge.advance(workdir, say=lambda _: 0) == 0
    row = judge.tally(workdir)["rhythm"]
    assert (row["picks"]["simple"], row["open"]) == (0, len(on_clear))  # a pick on a retired passage does not count
    assert "- mix: a new note only at a clear re-pluck" in judge.report(workdir)


def test_the_server_prepares_the_step_a_pick_opens(built):
    import time

    workdir = built["workdir"]
    srv = judge_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        s = json.loads(_get(url + "/api/next")[2])
        assert [(p["id"], p["total"] > 0) for p in s["steps"]] == [("replucks", True), ("rhythm", True), ("tab", False)]
        records = {c["id"]: c for c in judge.load_comparisons(workdir)}
        while s["comparison"] and s["comparison"]["choice"] == "rhythm":  # judge what is open: the rhythm step
            c = s["comparison"]
            assert c["step"] == "Step 2 of 3 · Write the rhythm" and c["legend"]
            simple = "A" if records[c["id"]]["flip"] else "B"  # the alternative is the second version
            assert _post(url + "/api/judge", dict(id=c["id"], pick=simple))["reveal"][simple] == "rhythm simplified"
            s = json.loads(_get(url + "/api/next?after=" + c["id"])[2])
        for _ in range(600):  # nothing to show yet: the tab is being prepared
            if s["comparison"] or not s["preparing"]:
                break
            time.sleep(0.1)
            s = json.loads(_get(url + "/api/next")[2])
        assert s["error"] is None and s["comparison"] and s["comparison"]["choice"] == "tab"
        assert s["comparison"]["sources"] == [["record", "Record", "r"], ["A", "Transcription", "a"]]
        assert [p["total"] > 0 for p in s["steps"]] == [True, True, True]
    finally:
        srv.shutdown()
        srv.server_close()


def test_score_snippets_follow_the_audio_and_are_engraved_once(built):
    if not find_mscore():
        pytest.skip("needs MuseScore")
    workdir = built["workdir"]
    comps = judge.load_comparisons(workdir)
    scores = [c for c in comps if c["kind"] in ("writing", "tab")]
    assert scores and all(set(c["sheets"]) == {"A", "B"} and c["sheet_rev"] == judge.SHEET_REV for c in scores)
    for c in scores:
        for sheet in c["sheets"].values():
            x0, y0, x1, y1 = sheet["crop"]
            assert os.path.getsize(os.path.join(workdir, sheet["src"])) > 1000
            assert 0 <= x0 < x1 <= sheet["page"][0] and 0 <= y0 < y1 <= sheet["page"][1]
            assert y1 - y0 < 0.75 * sheet["page"][1]  # cropped to what is drawn, not the whole page
            times, xs = [k[0] for k in sheet["keys"]], [k[1] for k in sheet["keys"]]
            assert times == sorted(times) and xs == sorted(xs) and x0 < xs[0] and xs[-1] < x1
            assert abs(times[0] - c["lead"]) < 0.05 and abs(times[-1] - (c["lead"] + c["t1"] - c["t0"])) < 0.05
            assert y0 < sheet["staff"][0] < sheet["staff"][1] < y1 and all(x0 < m < x1 for m in sheet["marks"])
        if c["kind"] == "tab":
            assert not c["sheets"]["A"]["marks"]  # every fingering differs: nothing worth marking
    quiet = lambda _: 0
    assert judge.render_sheets(workdir, comps, find_mscore(), quiet) == 0
    scores[0]["sheet_rev"] = 1  # from an older layout
    assert judge.render_sheets(workdir, comps, find_mscore(), quiet) == 1
    assert judge.render_sheets(workdir, comps, find_mscore(), quiet, force=True) == len(scores)  # pictures are kept
    said = []
    scores[0]["sheet_rev"] = 1
    assert judge.render_sheets(workdir, comps, None, said.append) == 0 and "MuseScore 4 not found" in said[0]
    judge.render_sheets(workdir, comps, find_mscore(), quiet)  # leave the directory as it was


def test_building_again_adds_nothing_and_keeps_the_picks(built):
    workdir = built["workdir"]
    before = [c["id"] for c in judge.load_comparisons(workdir)]
    picks = judge.load_judgments(workdir)
    again = judge.build(
        [built["mix"]],
        workdir,
        ["replucks", "rhythm", "tab"],
        2,
        stems_dirs={built["mix"]: built["stems"]},
        say=lambda _: 0,
        **built["kw"],
    )
    assert again == 0 and [c["id"] for c in judge.load_comparisons(workdir)] == before
    assert judge.load_judgments(workdir) == picks
    with pytest.raises(Bass2SheetError, match="unknown choice"):
        judge.build([built["mix"]], workdir, ["tempo"])


def test_after_a_change_only_passages_written_differently_come_back(built, monkeypatch):
    workdir = built["workdir"]
    quiet = lambda _: 0
    tabs = [c for c in _live(workdir) if c["choice"] == "tab"]
    _pick(workdir, tabs[0], "position")  # one tab judged; the rest of that step stays open
    judged = [c for c in _live(workdir) if c["id"] in judge.load_judgments(workdir)]
    kept, stray, new = judge.update(workdir, say=quiet)  # the tool has not changed
    assert (kept, new) == (len(judged), 0) and stray == len(judged)  # every pick here was for an alternative
    assert [c["id"] for c in _live(workdir)] == [c["id"] for c in judged] + [c["id"] for c in tabs[1:]]

    real = judge.hear

    def changed(*args, **kw):  # a later version of the tool hears every C as a D
        heard = real(*args, **kw)
        return dict(
            heard, notes=[dict(n, pitch=n["pitch"] + 2 if n["pitch"] % 12 == 0 else n["pitch"]) for n in heard["notes"]]
        )

    monkeypatch.setattr(judge, "hear", changed)
    kept, stray, new = judge.update(workdir, say=quiet)
    after = judge.load_comparisons(workdir)
    fresh = [c for c in after if c["choice"] == "change"]
    assert new == len(fresh) > 0 and kept + new <= len(judged)  # one passage may answer two picks
    assert fresh[0]["a_label"] == "the version you picked before" and fresh[0]["b_label"] == "what the tool writes now"
    assert {c["kind"] for c in fresh} <= {"heard", "notes"}  # the notes differ now, so both versions are heard
    assert all(not c["same_audio"] and os.path.exists(os.path.join(workdir, c["media"]["A"])) for c in fresh)
    live = _live(workdir)
    assert live[0]["choice"] == "change" and judge.progress(workdir)[0]["id"] == "change"
    assert judge.update(workdir, say=quiet)[2] == 0  # asked once, not again

    def changed_again(*args, **kw):  # a still later version hears every C as an E
        heard = real(*args, **kw)
        return dict(
            heard, notes=[dict(n, pitch=n["pitch"] + 4 if n["pitch"] % 12 == 0 else n["pitch"]) for n in heard["notes"]]
        )

    monkeypatch.setattr(judge, "hear", changed_again)
    judge.update(workdir, say=quiet)
    newer = [c for c in judge.load_comparisons(workdir) if c["choice"] == "change"]
    assert len(newer) == len(fresh) and not {c["id"] for c in newer} & {c["id"] for c in fresh}  # the old ones went
    assert {c["a"] for c in newer} == {c["a"] for c in fresh} and not {c["b"] for c in newer} & {c["b"] for c in fresh}
    assert not any(os.path.exists(os.path.join(workdir, "media", c["id"])) for c in fresh)
    judge.record_judgment(workdir, newer[0], "A")  # a judged one stays, whatever comes later
    monkeypatch.setattr(judge, "hear", real)  # ...even when the tool writes the passage as picked again
    judge.update(workdir, say=quiet)
    left = [c for c in judge.load_comparisons(workdir) if c["choice"] == "change"]
    assert [c["id"] for c in left] == [newer[0]["id"]]


def test_after_a_change_what_was_not_settled_is_built_again(taste_song_files, cache_dir, tmp_path, monkeypatch):
    mix, stems = taste_song_files
    workdir, quiet = str(tmp_path), lambda _: 0
    kw = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    judge.build([mix], workdir, ["replucks"], per_song=2, stems_dirs={mix: stems}, say=quiet, **kw)
    first = _live(workdir)
    assert len(first) == 2
    judge.record_judgment(workdir, first[0], "neither")  # both versions wrong there; the other passage is open
    judge.update(workdir, say=quiet)  # the tool has not changed: nothing moves
    assert [c["id"] for c in _live(workdir)] == [c["id"] for c in first]

    real = judge.hear
    kept = lambda n: first[1]["t0"] - 1 <= n["start"] < first[1]["t1"] + 1

    def changed_in_part(*args, **kw):  # a later version of the tool hears other notes, but not in the open passage
        heard = real(*args, **kw)
        return dict(heard, notes=[n if kept(n) else dict(n, pitch=n["pitch"] + 5) for n in heard["notes"]])

    monkeypatch.setattr(judge, "hear", changed_in_part)
    said = []
    assert judge.update(workdir, say=said.append) == (0, 0, 0)  # no pick for a version: nothing comes back
    assert any("built again" in line for line in said)
    live = _live(workdir)  # the passage that had no right version is built again; the open one is asked once
    assert len(live) == 2 and first[1] in live and first[0]["id"] not in [c["id"] for c in live]
    again = next(c for c in live if c != first[1])
    assert again["bars_a"] == first[0]["bars_a"] and again["a"] != first[0]["a"]  # the same bars, as written now
    assert again["id"] not in judge.load_judgments(workdir)
    assert judge.progress(workdir) == [dict(id="replucks", name="Find the notes", total=2, judged=0)]
    aside = next(c for c in judge.load_comparisons(workdir) if c["id"] == first[0]["id"])
    assert aside["base_key"].startswith("(set aside)")  # kept with its judgment, but it counts no more
    bars = lambda cs: sorted(c["bars_a"] for c in cs)
    assert judge.update(workdir, say=quiet) == (0, 0, 0) and bars(_live(workdir)) == bars(live)  # the same passages


def test_the_step_for_notes_played_together_shows_where_two_were_heard(cache_dir, tmp_path):
    from songsynth import make_song, write_song

    assert [s["id"] for s in judge.STEPS] == ["separator", "replucks", "together", "barlines", "rhythm", "tab"]
    song = make_song(double_stops=True)
    mix, stems = write_song(song, str(tmp_path / "song"))
    workdir = str(tmp_path / "judging")
    kw = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    judge.build([mix], workdir, ["together"], per_song=2, stems_dirs={mix: stems}, say=lambda _: 0, **kw)
    comps = _live(workdir)
    assert comps and {(c["choice"], c["kind"], c["a_value"], c["b_value"]) for c in comps} == {
        ("together", "heard", "write", "single")
    }
    for c in comps:
        va, vb = judge.load_version(workdir, c["a"]), judge.load_version(workdir, c["b"])
        assert any(n.get("also") for n in va["notes"]) and not any(n.get("also") for n in vb["notes"])
        a, b = c["sheets"]["A"]["roll"]["notes"], c["sheets"]["B"]["roll"]["notes"]
        both, one = (b, a) if c["flip"] else (a, b)
        assert len(both) == len(one) + 1  # the pair is two boxes, and the lower one is what the other lacks
        assert sorted(p for _, _, p, differs in both if differs) == [48]
        assert sorted(p for _, _, p, differs in one if differs) == []
    events = judge.events(judge.load_version(workdir, comps[0]["a"]))
    assert [e["also_tab"] for e in events if e.get("also")][0] == [[55, 1, 12]]  # the G over the C, one string up


def test_check_measures_the_songs_without_anyone_listening(taste_song_files, cache_dir, tmp_path, monkeypatch):
    mix, stems = taste_song_files
    workdir = str(tmp_path)
    kw = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    judge.build([mix], workdir, ["replucks"], per_song=1, stems_dirs={mix: stems}, say=lambda _: 0, **kw)
    said = []
    rows = judge.check(workdir, say=said.append)
    assert len(rows) == 1 and rows[0]["song"] == "mix" and rows[0]["was"] is None
    assert rows[0]["judged"] > 50 and rows[0]["unlike"] < 0.05 and rows[0]["another_pitch"] == 0
    assert rows[0]["settings"] == dict(separator="roformer", replucks="every", together="write")
    assert "do not sound like" in said[0] and "(was" not in said[0]

    real = judge.hear

    def worse(*args, **kw):  # a later version of the tool hears every fifth note a fourth too high
        heard = real(*args, **kw)
        notes = [dict(n, pitch=n["pitch"] + 5) if k % 5 == 2 else n for k, n in enumerate(heard["notes"])]
        return dict(heard, notes=notes)

    monkeypatch.setattr(judge, "hear", worse)
    monkeypatch.setattr(judge, "code_id", lambda: "later000")
    again = judge.check(workdir, say=said.append)
    assert again[0]["unlike"] > rows[0]["unlike"] + 0.1 and again[0]["was"] == rows[0]["unlike"]
    assert again[0]["another_pitch"] >= 8 and "(was " in said[1]
    with open(os.path.join(workdir, "checks.jsonl")) as f:
        assert [json.loads(line)["code"] for line in f] == [rows[0]["code"], "later000"]
