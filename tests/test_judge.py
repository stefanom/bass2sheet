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
    )
    if choice == "tab":
        del c["media"]["B"], c["media"]["B_bass"]
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


def test_comparisons_are_shown_one_of_each_choice_before_a_second_of_any():
    comps = [_comp("s.tab.2", "tab"), _comp("s.separator.2"), _comp("s.tab.1", "tab"), _comp("s.separator.1")]
    comps.append(dict(_comp("s.change.1"), choice="change"))
    assert [c["id"] for c in judge.order(comps)] == [
        "s.change.1",
        "s.separator.1",
        "s.tab.1",
        "s.separator.2",
        "s.tab.2",
    ]


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


# ---------------------------------------------------------------------------------- building and the later loop


@pytest.fixture(scope="module")
def built(song_files, cache_dir, tmp_path_factory):
    """The synthetic song under three choices that need no model download."""
    mix, stems = song_files
    workdir = str(tmp_path_factory.mktemp("judging"))
    said = []
    kw = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    n = judge.build(
        [mix], workdir, ["replucks", "rhythm", "tab"], per_song=2, stems_dirs={mix: stems}, say=said.append, **kw
    )
    return dict(workdir=workdir, added=n, said=said, mix=mix, kw=kw)


def test_build_makes_comparisons_with_audio_and_scores(built):
    comps = judge.load_comparisons(built["workdir"])
    assert built["added"] == len(comps) > 0, built["said"]
    tab = [c for c in comps if c["choice"] == "tab"]
    assert tab and tab[0]["same_audio"] and "B" not in tab[0]["media"]
    for c in comps:
        assert c["a_label"].endswith("(the default)") and not c["b_label"].endswith("(the default)")
        assert 2.0 < c["t1"] - c["t0"] < 12.0 and c["bars_b"] == c["bars_a"]
        for name, rel in c["media"].items():
            assert os.path.getsize(os.path.join(built["workdir"], rel)) > 1000, (c["id"], name)
        if find_mscore():
            assert {"A_sheet", "B_sheet"} <= set(c["media"])
        assert os.path.exists(os.path.join(built["workdir"], c["a"])) and c["a"] != c["b"]
    import soundfile as sf

    y, sr = sf.read(os.path.join(built["workdir"], comps[0]["media"]["record"]))
    assert sr == SR and abs(len(y) / sr - comps[0]["length"]) < 0.01 and 0.5 < np.abs(y).max() <= 1.0


def test_building_again_adds_nothing_and_keeps_the_picks(built):
    workdir = built["workdir"]
    comps = judge.load_comparisons(workdir)
    judge.record_judgment(workdir, comps[0], "A", "first")
    stems = judge.read_json(workdir, "songs.json", {})[comps[0]["song"]]["stems_dir"]
    again = judge.build(
        [built["mix"]],
        workdir,
        ["replucks", "rhythm", "tab"],
        2,
        stems_dirs={built["mix"]: stems},
        say=lambda _: 0,
        **built["kw"],
    )
    assert again == 0 and [c["id"] for c in judge.load_comparisons(workdir)] == [c["id"] for c in comps]
    assert judge.load_judgments(workdir)[comps[0]["id"]]["note"] == "first"
    with pytest.raises(Bass2SheetError, match="unknown choice"):
        judge.build([built["mix"]], workdir, ["tempo"])


def test_after_a_change_only_passages_written_differently_come_back(built, monkeypatch):
    workdir = built["workdir"]
    comps = judge.load_comparisons(workdir)
    tab = next(c for c in comps if c["choice"] == "tab")
    for c in comps:  # the default wins everywhere, except that the other tab is picked once
        judge.record_judgment(workdir, c, "A" if (c is tab) == c["flip"] else "B")
    assert judge.load_judgments(workdir)[tab["id"]]["value"] == "position"
    quiet = dict(say=lambda _: 0, **built["kw"])

    kept, ignored, new = judge.update(workdir, **quiet)  # nothing changed: only the tab pick is not followed
    assert (kept, ignored, new) == (len(comps) - 1, 1, 0)

    followed = dict(quiet, defaults=dict(beat_tracker="librosa", tab="position"))  # the default now follows it
    kept, ignored, new = judge.update(workdir, **followed)
    others = [c for c in comps if c["choice"] == "tab" and c is not tab]
    assert (ignored, new) == (len(others), 0) and kept == len(comps) - len(others)

    real = judge.hear

    def changed(*args, **kw):  # a later version of the tool hears every C as a D
        heard = real(*args, **kw)
        return dict(
            heard, notes=[dict(n, pitch=n["pitch"] + 2 if n["pitch"] % 12 == 0 else n["pitch"]) for n in heard["notes"]]
        )

    monkeypatch.setattr(judge, "hear", changed)
    kept, ignored, new = judge.update(workdir, **quiet)
    after = judge.load_comparisons(workdir)
    fresh = [c for c in after if c["choice"] == "change"]
    assert new == len(fresh) > 0 and kept + ignored + new <= len(comps)  # one passage may answer two picks
    assert not any(c["same_audio"] for c in fresh)  # the notes differ now, so both versions are heard
    assert fresh[0]["a_label"] == "the version you picked before" and fresh[0]["b_label"] == "what the tool writes now"
    assert judge.order(after)[0]["choice"] == "change" and os.path.exists(os.path.join(workdir, fresh[0]["media"]["A"]))
    assert judge.update(workdir, **quiet)[2] == 0  # asked once, not again
