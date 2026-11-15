"""The copilot's pages in a real browser: each page is opened from the real server, on the synthetic song, and its
main things are done as a person does them, with clicks and keys. What the page's own logic tests cannot see is
seen here: the script against the page it is in, against the server, and against a browser.

Skipped where Playwright or a browser to drive is missing (see the `browser` fixture in conftest.py). MuseScore's
place is taken by a stand-in (mscore_standin.py), so the score is drawn the same way everywhere, in no time."""

import json
import os
import re
import shutil
import threading
import urllib.request

import pytest
from conftest import mscore_stand_in

from bass2sheet import copilot, copilot_harmony, copilot_server, score, testsongs
from bass2sheet.common import atomic_write_text

pytestmark = pytest.mark.browser
CREPE = os.environ.get("BASS2SHEET_TEST_CREPE", "tiny")
SONG_READY = "typeof S !== 'undefined' && S && S.id === %s && tiles.length > 0 && !!A.bass"


class Site:
    def __init__(self, url, workdir, mix):
        self.url, self.workdir, self.mix = url, workdir, mix

    def desk(self, song_id):
        """The song as it is on the disk now."""
        return copilot.Desk(self.workdir, song_id)

    def post(self, path, data):
        req = urllib.request.Request(self.url + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


@pytest.fixture(scope="module")
def site(browser, cache_dir, tmp_path_factory):
    """The copilot's server on a working directory with one prepared song, `mix`: the synthetic song with three
    places where two notes are played together. `mix` is also the one test song of a set called Tiny."""
    from songsynth import make_song, write_song

    mix, stems = write_song(make_song(double_stops=True), str(tmp_path_factory.mktemp("song")))
    workdir = str(tmp_path_factory.mktemp("scores"))
    opts = dict(cache_dir=cache_dir, device="cpu", crepe_model=CREPE, defaults=dict(beat_tracker="librosa"))
    copilot.add([mix], workdir, opts, stems_dirs={mix: stems}, say=lambda _: 0)
    machine = copilot.read_json(os.path.join(workdir, "mix", "machine.json"))
    tiny = dict(set="Tiny", track="Track00001", facts=["sampled bass: a plucked string"], length=machine["length"])
    tiny.update(busy=testsongs.busy(machine["notes"]), scores=dict(bass2sheet=0.9, MuScriptor=0.95))
    about = {k: testsongs.SETS["slakh"][k] for k in ("about", "measure")}
    tests = dict(sets=dict(Tiny=dict(kind="slakh", **about)), songs=dict(mix=tiny))
    atomic_write_text(os.path.join(workdir, "tests.json"), json.dumps(tests))
    patch = pytest.MonkeyPatch()
    patch.setattr(score, "MSCORE_CANDIDATES", [mscore_stand_in(tmp_path_factory.mktemp("bin"))])
    patch.setenv("BASS2SHEET_DOWNLOADER", "none")  # no box for links, whatever is installed here
    patch.delenv("MUSCRIPTOR_NOTES", raising=False)
    patch.delenv("MUSCRIPTOR_GGUF", raising=False)
    srv = copilot_server.make_server(workdir, 0, stems_dir=stems)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield Site(f"http://127.0.0.1:{srv.server_address[1]}", workdir, mix)
    srv.RequestHandlerClass.adding.close()
    srv.shutdown()
    srv.server_close()
    patch.undo()


@pytest.fixture
def song(site, request):
    """A song of the test's own: a copy of the prepared one under the test's name, so that what one test fixes
    is not what the next one finds."""
    song_id = re.sub(r"[^a-z0-9]+", "-", request.node.name.lower()).strip("-")[5:45].strip("-")
    shutil.copytree(os.path.join(site.workdir, "mix"), os.path.join(site.workdir, song_id))
    songs = copilot.load_songs(site.workdir)
    songs[song_id] = dict(songs["mix"], id=song_id, title="A song of " + song_id.replace("-", " "))
    atomic_write_text(os.path.join(site.workdir, "songs.json"), json.dumps(songs))
    return song_id


# ---- on the song's page: where things are, in the page's own words

BOXES = """id => {  // where each box of a note is in the window: the note itself first, then what is played with it
  const r = document.getElementById("roll").getBoundingClientRect();
  return boxes(note(id)).map(([q, x, y, w]) => ({pitch: q, x: r.left + x + w / 2, y: r.top + y + row / 2}));
}"""
IN_VIEW = """want => {  // the first note in the window, well inside it, that is as wanted
  const r = document.getElementById("roll").getBoundingClientRect(), ok = new Function("n", "return " + want);
  for (const n of Logic.sorted(S.notes)) {
    const [q, x, y, w] = boxes(n)[0];
    if (x > GUTTER + 60 && x + w < r.width - 60 && ok(n)) return {id: n.id, pitch: n.pitch, start: n.start, end: n.end};
  }
  return null;
}"""
MAP_AT = """t => {  // where a moment of the song is on the map of the whole song
  const r = document.getElementById("map").getBoundingClientRect();
  return {x: r.left + mapX(t), y: r.top + r.height / 2};
}"""


def _song_page(browse, site, song_id):
    return browse(f"{site.url}/song#{song_id}", ready=SONG_READY % json.dumps(song_id))


def _click_note(page, note_id, box=0):
    """Click a box of a note; returns the pitch of the box. The page is let come to rest first: while it plays
    the timeline moves under the pointer, and an answer from the server can open the list of repeats under it."""
    page.wait_for_function("__quiet(200) && !A.playing")
    at = page.evaluate(BOXES, note_id)[box]
    page.mouse.click(at["x"], at["y"])
    page.wait_for_function("id => sel === id", arg=note_id)
    return at["pitch"]


def _panel(page):
    return page.inner_text("#panel")


# ------------------------------------------------------------------------------------------------ the song's page


def test_a_note_is_selected_its_pitch_is_changed_and_the_change_is_taken_back(browse, site, song):
    page = _song_page(browse, site, song)
    assert page.inner_text("#progress") == "worked through 0:00.0 of 0:26.5 · 0 fixes"
    assert page.is_disabled("#undo") and "Click a note to select it" in _panel(page)
    n = page.evaluate(IN_VIEW, "!(n.also || []).length && n.end - n.start > 0.2")
    _click_note(page, n["id"])
    name = page.evaluate("p => name(p)", n["pitch"])
    page.wait_for_function("document.querySelectorAll('#panel button.guess').length > 0")  # the tool's guesses came
    assert _panel(page).startswith(name) and "Or is it" in _panel(page)

    page.keyboard.press("ArrowUp")  # a semitone up
    page.wait_for_function("([id, p]) => note(id).pitch === p && S.fixes === 1", arg=[n["id"], n["pitch"] + 1])
    assert "the tool had " + name in _panel(page) and page.inner_text("#progress").endswith("· 1 fix")
    assert page.is_enabled("#undo")
    fixed = site.desk(song).note(n["id"])  # and the server has it, on the disk
    assert (fixed["pitch"], fixed["state"], fixed["was"]["pitch"]) == (n["pitch"] + 1, "fixed", n["pitch"])

    page.click("#undo")
    page.wait_for_function("([id, p]) => note(id).pitch === p && S.fixes === 0", arg=[n["id"], n["pitch"]])
    assert site.desk(song).note(n["id"])["state"] == "machine" and page.is_disabled("#undo")

    page.keyboard.press("Shift+ArrowUp")  # an octave, and one of the tool's guesses takes it back
    page.wait_for_function("([id, p]) => note(id).pitch === p", arg=[n["id"], n["pitch"] + 12])
    page.wait_for_function("p => guessesFor === sel + ':' + (p + 12) && guesses.includes(p)", arg=n["pitch"])
    rank = page.evaluate("p => guesses.indexOf(p) + 1", n["pitch"])
    page.click(f"#panel button.guess[data-guess='{rank - 1}']")
    page.wait_for_function("([id, p]) => note(id).pitch === p && S.fixes === 0", arg=[n["id"], n["pitch"]])
    page.keyboard.press("z")  # the key takes the last change back: the octave is there again
    page.wait_for_function("([id, p]) => note(id).pitch === p", arg=[n["id"], n["pitch"] + 12])
    assert site.desk(song).note(n["id"])["pitch"] == n["pitch"] + 12
    assert page.inner_text("#status") == ""


def test_one_of_two_notes_played_together_is_deleted_by_itself(browse, site, song):
    page = _song_page(browse, site, song)
    pairs = page.evaluate("Logic.sorted(S.notes).filter(n => (n.also || []).length).map(n => [n.id, n.start])")
    assert len(pairs) == 3, pairs
    (first, t1), (second, t2) = pairs[1], pairs[2]
    there = page.evaluate(MAP_AT, t1)
    page.mouse.click(there["x"], there["y"])  # on the map of the whole song: the timeline goes there
    page.wait_for_function("t => Math.abs(view.t0 + ($('roll').clientWidth - GUTTER) / view.pps / 2 - t) < 1", arg=t1)

    low, high = (b["pitch"] for b in page.evaluate(BOXES, first))
    assert high == low + 7  # the note and the fifth above it
    assert _click_note(page, first, box=1) == high  # the upper of the two
    page.wait_for_function("p => tone === p", arg=high)
    names = page.evaluate("ps => ps.map(name)", [low, high])
    assert _panel(page).startswith(f"{names[0]} + {names[1]}")
    buttons = page.eval_on_selector_all(
        "#panel button[data-tone], #panel button[data-act='delete']", "bs => bs.map(b => b.innerText)"
    )
    assert buttons == [f"Delete {names[0]}", f"Delete {names[1]}", "Delete both"]
    assert page.get_attribute(f"#panel button[data-tone='{high}']", "class") == "picked"  # the one that was clicked
    page.click(f"#panel button[data-tone='{high}']")
    page.wait_for_function("id => (note(id).also || []).length === 0", arg=first)
    left = site.desk(song).note(first)
    assert (left["pitch"], left["also"], left["state"]) == (low, [], "fixed")  # the lower one stays, by itself
    assert page.locator("#panel button[data-tone]").count() == 0 and _panel(page).startswith(names[0] + " ")

    # the next pair: the lower one is clicked and taken out with the key; the other one becomes the note
    if not page.evaluate(
        "id => { const x = boxes(note(id))[0][1]; return x > GUTTER + 40 && x < $('roll').clientWidth - 80; }", second
    ):
        there = page.evaluate(MAP_AT, t2)
        page.mouse.click(there["x"], there["y"])
    low, high = (b["pitch"] for b in page.evaluate(BOXES, second))
    assert _click_note(page, second, box=0) == low
    page.wait_for_function("p => tone === p", arg=low)
    page.keyboard.press("Backspace")
    page.wait_for_function(
        "([id, p]) => note(id) && note(id).pitch === p && !(note(id).also || []).length", arg=[second, high]
    )
    became = site.desk(song).note(second)
    assert (became["pitch"], became["also"], became["was"]["pitch"]) == (high, [], low)
    assert (
        len(site.desk(song).notes) == len(site.desk("mix").notes) and page.evaluate("S.fixes") == 2
    )  # no note is gone


def test_a_note_is_split_in_two_and_the_two_are_joined_again(browse, site, song):
    page = _song_page(browse, site, song)
    count = page.evaluate("S.notes.length")
    n = page.evaluate(IN_VIEW, "!(n.also || []).length && n.end - n.start > 0.3")
    _click_note(page, n["id"])
    page.click("#panel button[data-act='split']")
    page.wait_for_function("c => S.notes.length === c + 1 && note(sel) && note(sel).split_from", arg=count)
    second = page.evaluate("sel")  # the new note is the one selected now
    middle = round((n["start"] + n["end"]) / 2, 3)
    parts = site.desk(song)
    assert parts.note(n["id"])["end"] == pytest.approx(middle, abs=2e-3) and parts.note(second)["split_from"] == n["id"]
    assert parts.note(second)["start"] == parts.note(n["id"])["end"] and parts.note(second)["end"] == n["end"]

    _click_note(page, n["id"])  # the first of the two, and M joins it with the next
    page.keyboard.press("m")
    page.wait_for_function("c => S.notes.length === c", arg=count)
    whole = site.desk(song).note(n["id"])
    assert whole["end"] == n["end"] and whole["merged"] == [second] and page.evaluate("id => !note(id)", second)
    assert copilot.mistakes(site.workdir, song)["kinds"] == {}  # split and joined again: nothing the tool got wrong

    # two neighbours joined with the button, and taken apart again with Undo
    after = page.evaluate("id => Logic.neighbour(S.notes, id, 1, 0)", n["id"])
    page.click("#panel button[data-act='join']")
    page.wait_for_function("c => S.notes.length === c - 1", arg=count)
    assert site.desk(song).note(n["id"])["merged"] == [second, after]
    page.click("#undo")
    page.wait_for_function("c => S.notes.length === c", arg=count)
    assert page.evaluate("id => !!note(id)", after)


def test_what_was_played_and_left_alone_counts_as_checked_up_to_the_playhead(browse, site, song):
    page = _song_page(browse, site, song)
    page.keyboard.press("Space")  # play from the start
    page.wait_for_function("A.playing && position() > 2.5")
    assert page.inner_text("#play") == "■ Stop"
    page.keyboard.press("Space")
    page.wait_for_function("!A.playing && S.checked.length === 1")  # what was played is told to the server
    stopped = page.evaluate("playhead")
    assert 2.5 < stopped < 15 and page.inner_text("#play") == "▶ Play"
    assert page.evaluate("S.worked") == [] and page.inner_text("#progress").startswith("worked through 0:00.0")
    page.click("#upto")  # "Checked up to here"
    page.wait_for_function("S.worked.length === 1")
    heard, worked = page.evaluate("[S.checked, S.worked]")
    assert worked == [[0.0, pytest.approx(stopped, abs=0.02)]] and heard[0][1] == pytest.approx(stopped, abs=0.02)
    assert page.inner_text("#progress").startswith("worked through 0:0" + str(int(stopped)))
    desk = site.desk(song)
    assert desk.reached() == pytest.approx(stopped, abs=0.02) and desk.fixes() == 0
    right = copilot.mistakes(site.workdir, song)["right"]
    assert right == sum(n["start"] <= desk.reached() for n in desk.notes) > 3  # the notes before it count as right
    assert page.evaluate("S.at") is None and site.desk(song).state["at"] == pytest.approx(stopped, abs=0.02)


def test_the_sheet_is_made_and_shown_over_the_songs_page(browse, site, song):
    page = _song_page(browse, site, song)
    n = page.evaluate(IN_VIEW, "true")
    _click_note(page, n["id"])
    page.keyboard.press("Shift+ArrowUp")
    page.wait_for_function("S.fixes === 1")
    page.click("#sheet")  # "Make the sheet"
    page.wait_for_function("document.getElementById('status').textContent === 'the sheet is written'")
    links = page.eval_on_selector_all(
        "#sheetlinks a", "as => as.map(a => [a.innerText, a.getAttribute('href').split('?')[0]])"
    )
    assert links == [
        ["See the score", f"/score#{song}"],
        ["PDF", f"/data/{song}/sheet/bass.pdf"],
        ["MusicXML", f"/data/{song}/sheet/bass.musicxml"],
        ["MIDI", f"/data/{song}/sheet/bass_as_played.mid"],
    ]
    # the score is shown in a panel over the page: its pages, as pictures, and the files to download
    panel = page.frame_locator("#scorePanel iframe")
    panel.locator("#pages img").first.wait_for()
    page.wait_for_function("document.getElementById('scorePanel').style.display === 'block'")
    frame = next(f for f in page.frames if "/score" in f.url)
    frame.wait_for_function("[...document.querySelectorAll('#pages img')].every(i => i.complete && i.naturalWidth > 0)")
    assert frame.inner_text("#made").startswith("Made today at ") and "1 fix" in frame.inner_text("#made")
    assert frame.eval_on_selector_all("#files a", "as => as.map(a => a.innerText)") == ["PDF", "MusicXML", "MIDI"]
    assert frame.is_visible("#close") and not frame.is_visible("#notes")
    made = copilot.score_made(site.workdir, song)
    assert made["version"] == 1 and made["fixes"] == 1 and len(made["pages"]) >= 1
    xml = open(os.path.join(site.workdir, song, "sheet", "bass.musicxml")).read()
    assert "<score-partwise" in xml and "the rest is the machine" in xml
    # while it is shown the keys are not the song page's; Esc closes it, and the notes are as they were
    page.wait_for_function("!A.playing")  # the note that was changed has been heard to its end
    page.keyboard.press("Space")
    assert page.evaluate("A.playing") is False
    page.keyboard.press("Escape")
    page.wait_for_function("document.getElementById('scorePanel').style.display === 'none'")
    assert page.evaluate("[S.fixes, sel]") == [1, n["id"]]
    page.click("#sheetlinks a >> text=See the score")
    page.wait_for_function("document.getElementById('scorePanel').style.display === 'block'")
    frame = next(f for f in page.frames if "/score" in f.url)
    frame.click("#close")
    page.wait_for_function("document.getElementById('scorePanel').style.display === 'none'")


# ------------------------------------------------------------------------------------------------- the first page


def test_the_first_page_lists_the_songs_and_opens_one(browse, site, song):
    page = browse(site.url + "/", ready="document.querySelectorAll('#songs .song').length > 0")
    listed = page.eval_on_selector_all("#songs .song", "cards => cards.map(c => c.dataset.id)")
    assert "mix" in listed and song in listed and page.inner_text("#count") == f"{len(listed)} songs"
    card = page.locator(f"#songs .song[data-id='{song}']")
    assert card.locator(".title").inner_text() == "A song of " + song.replace("-", " ")
    assert card.locator(".facts").inner_text() == "Not started"
    assert card.locator(".foot a").all_inner_texts() == ["Fix the notes ›", "Make the score", "Harmony"]
    assert card.locator(".foot a").nth(2).get_attribute("href") == f"/harmony#{song}"
    # every song has its picture, the whole song across (asked for all at once: none may be turned away)
    page.wait_for_function("[...document.querySelectorAll('#songs .song canvas')].every(c => c.dataset.drawn)")
    assert page.is_hidden("#link") and page.is_hidden("#empty") and page.is_hidden("#jobsTitle")
    assert page.locator("#path li.now").count() >= 1 and page.inner_text("#phone") == "Play along"
    assert page.eval_on_selector_all("#finders input", "is => is.map(i => [i.value, i.checked, i.disabled])") == [
        ["own", True, False],
        ["muscriptor", False, True],  # not set up here: shown, and greyed out
    ]
    card.locator(".foot a.main").click()  # "Fix the notes"
    page.wait_for_function(SONG_READY % json.dumps(song))
    assert page.url.endswith(f"/song#{song}") and page.input_value("#song") == song
    page.click("header a[href='/']")  # "‹ Songs"
    page.wait_for_function("document.querySelectorAll('#songs .song').length > 0")


def test_a_song_is_added_on_the_first_page_and_what_is_not_a_song_is_refused(browse, site, tmp_path):
    page = browse(site.url + "/", ready="document.querySelectorAll('#songs .song').length > 0 && way.kinds.length > 0")
    before = page.locator("#songs .song").count()
    words = tmp_path / "words.txt"
    words.write_text("these are not notes")
    page.set_input_files("#file", str(words))
    page.wait_for_function("document.getElementById('addStatus').textContent.includes('words.txt')")
    assert "bad" in page.get_attribute("#addStatus", "class") and page.locator("#jobs .job").count() == 0

    new = tmp_path / "Browser Band - A New Song.wav"
    shutil.copy(site.mix, new)
    page.set_input_files("#file", str(new))
    job = page.locator("#jobs .job")
    job.wait_for()
    assert job.locator(".head b").inner_text() == "A New Song" and "Browser Band" in job.locator(".about").inner_text()
    steps = job.locator("ol.steps li .label").all_inner_texts()
    assert steps[0] == "Reading the file" and steps[-1] == "Ready" and "Isolating the bass" in steps
    assert page.is_visible("#jobsTitle") and page.inner_text("#addStatus") == ""
    page.wait_for_function("document.querySelector('#jobs .job .state').textContent === 'Ready'", timeout=180000)
    facts = job.locator("ol.steps li .fact").all_inner_texts()
    assert any("notes" in f for f in facts) and job.locator(".error").is_hidden()
    assert job.locator(".look canvas").is_visible()  # the picture of the bass, under the steps
    card = page.locator("#songs .song[data-id='browser-band-a-new-song']")
    card.wait_for()
    assert "fresh" in card.get_attribute("class") and page.locator("#songs .song").count() == before + 1
    assert card.locator(".title").inner_text() == "A New Song"
    assert os.path.exists(os.path.join(site.workdir, "browser-band-a-new-song", "machine.json"))
    job.locator(".buttons button").click()  # it is ready: out of the row, and still among the songs
    page.wait_for_function("document.querySelectorAll('#jobs .job').length === 0")
    assert card.is_visible()


# --------------------------------------------------------------------------------------------------- the score


def test_the_score_page_makes_the_score_shows_it_and_keeps_every_version(browse, site, song):
    page = browse(f"{site.url}/score#{song}", ready="view !== null")
    assert "No score has been made of this song yet." in page.inner_text("#notice")
    assert page.inner_text("#make") == "Make the score" and page.locator("#pages img").count() == 0
    loaded = (
        "(p => p.length > 0 && p.every(i => i.complete && i.naturalWidth > 0))([...$('pages').querySelectorAll('img')])"
    )
    page.click("#make")
    page.wait_for_function("view.made !== null && " + loaded)
    assert page.is_hidden("#notice") and page.inner_text("#make") == "Make it again" and page.is_hidden("#version")
    assert page.inner_text("#made").startswith(
        "Made today at "
    ) and "0 fixes · not checked by ear yet" in page.inner_text("#made")
    files = page.eval_on_selector_all(
        "#files a", "as => as.map(a => [a.innerText, a.getAttribute('href').split('?')[0], a.download])"
    )
    title = "A song of " + song.replace("-", " ")
    assert files == [
        ["PDF", f"/data/{song}/sheet/v1/bass.pdf", title + ".pdf"],
        ["MusicXML", f"/data/{song}/sheet/v1/bass.musicxml", title + ".musicxml"],
        ["MIDI", f"/data/{song}/sheet/v1/bass_as_played.mid", title + ".mid"],
    ]
    assert page.inner_text("#title") == title and page.get_attribute("#notes", "href") == f"/song#{song}"

    note = site.desk(song).notes[4]  # a fix, as the song's page sends it, and the score made again: a second version
    site.post(
        "/api/act", dict(song=song, action=dict(kind="pitch", id=note["id"], pitch=note["pitch"] + 12, offer=False))
    )
    page.click("#make")
    page.wait_for_function("view.versions.length === 2 && " + loaded)
    assert page.is_visible("#version") and page.inner_text("#made") == ""
    options = page.eval_on_selector_all("#version option", "os => os.map(o => [o.value, o.selected, o.innerText])")
    assert [o[:2] for o in options] == [["2", True], ["1", False]]
    assert (
        options[0][2].startswith("Latest: today at ")
        and "1 fix" in options[0][2]
        and options[1][2].startswith("Version 1: ")
    )
    page.select_option("#version", "1")  # the score as it was first made
    page.wait_for_function("view.made.version === 1 && " + loaded)
    assert page.inner_text("#made") == "an earlier version"
    assert page.get_attribute("#files a >> nth=0", "href").startswith(f"/data/{song}/sheet/v1/bass.pdf")
    assert [v["version"] for v in copilot.score_versions(site.workdir, song)] == [2, 1]


def test_a_score_page_without_its_song_says_so(browse, site):
    page = browse(site.url + "/score", ready="!document.getElementById('notice').classList.contains('hidden')")
    assert "No song is named in the address." in page.inner_text("#notice") and page.is_hidden("#make")
    page.click("#notice a")  # "Choose one of your songs"
    page.wait_for_function("location.pathname === '/' && document.querySelectorAll('#songs .song').length > 0")


# --------------------------------------------------------------------------------------------------- playing along


def test_the_play_along_page_loads_a_song_and_moves_from_line_to_line(browse, site, song):
    page = browse(
        f"{site.url}/play?song={song}",
        ready="typeof S !== 'undefined' && S && S.lines.length > 2 && !!loaded && loaded.ready",
        size=(844, 390),  # a phone held sideways
    )
    drawn = "(i => !!i && i.complete && i.naturalWidth > 0)(document.querySelector('#lines .line.cur img'))"
    page.wait_for_function(drawn)
    assert page.is_hidden("#msg") and page.is_hidden("#menu") and page.title().endswith("· play along")
    lines = page.evaluate("S.lines.map(l => l.bars.map(b => b[0]))")
    assert lines[0][0] == 1 and all(len(bars) <= 3 for bars in lines) and page.evaluate("[S.bars, shown.cur]") == [3, 0]
    assert (
        page.get_attribute("#lines .line.cur", "data-line") == "0"
        and page.get_attribute("#lines .line.next", "data-line") == "1"
    )
    assert page.inner_text("#clock").split("\n") == ["0:00", "bar 1"]
    assert page.get_attribute("#sounds button.on", "data-sound") == "nobass"  # the record without its bass

    page.click("#on")  # on to the next line
    page.wait_for_function("shown.cur === 1 && " + drawn)
    assert page.get_attribute("#lines .line.cur", "data-line") == "1"
    assert page.inner_text("#clock").split("\n")[1] == f"bar {lines[1][0]}"
    assert page.evaluate("now()") == pytest.approx(page.evaluate("S.lines[1].start"), abs=0.01)
    page.keyboard.press("ArrowRight")
    page.wait_for_function("shown.cur === 2")
    page.keyboard.press("ArrowLeft")  # back: to the line before
    page.wait_for_function("shown.cur === 1")
    page.click("#back")
    page.wait_for_function("shown.cur === 0")

    page.click("#play")
    page.wait_for_function("playing() && now() > 0.5")
    assert page.get_attribute("#play", "aria-label") == "Pause"
    page.click("#play")
    page.wait_for_function("!playing()")
    stopped = page.evaluate("now()")
    assert stopped > 0.5 and page.get_attribute("#play", "aria-label") == "Play"
    assert json.loads(page.evaluate("localStorage.getItem('playalong')"))["at"][song] == pytest.approx(
        stopped, abs=0.15
    )

    page.keyboard.press("l")  # a loop of four bars from the bar being played
    page.wait_for_function("loop !== null")
    assert page.is_visible("#loopSet") and page.inner_text("#loopBars") == "4 bars"
    page.click("#loopLess")
    assert page.inner_text("#loopBars") == "3 bars" and page.locator("#lines .tint").count() >= 1
    page.click("#open")  # songs and settings: two bars to a line
    page.wait_for_function(
        "!$('menu').classList.contains('hidden') && $('songs').querySelectorAll('button').length > 0"
    )
    page.click("#barsSet button[data-bars='2']")
    page.wait_for_function("S && S.bars === 2 && S.lines.every(l => l.bars.length <= 2) && " + drawn)
    assert page.is_hidden("#menu") and os.path.isdir(os.path.join(site.workdir, song, "play", "b2"))


# --------------------------------------------------------------------------------------------------- the test songs


def test_the_harmony_page_follows_the_song_on_its_maps_and_keeps_the_choice_of_home(browse, site, song):
    page = browse(f"{site.url}/harmony#{song}", ready="H !== null && document.querySelectorAll('#map .st').length > 20")
    assert "A song of" in page.inner_text("#title") and page.locator("#loops .verb").count() >= 1
    # the song in short: its one loop, said as steps above home
    assert page.locator("#short .line").count() == 1 and "a bar each" in page.inner_text("#short")
    loop = page.evaluate("H.loops.find(l => l.size === 4)")
    chips = page.locator(f"#loops .verb[data-letter='{loop['letter']}'] .chip")
    assert chips.count() == 4 and [chips.nth(k).locator("b").inner_text() for k in range(4)] == ["Em", "C", "G", "D"]
    assert chips.nth(0).locator("u").inner_text() == "R R 5 8 5 4"  # what the bass plays over the first chord

    # a click on a chord of a loop goes there and plays: the chord, its job, the bar, the neck and the map follow
    chips.nth(1).click()
    page.wait_for_function("!document.getElementById('au').paused && S.chord >= 0 && H.chords[S.chord].name === 'C'")
    page.wait_for_function("document.getElementById('play').textContent === '■ Stop'")
    assert page.inner_text("#here-name") == "C"
    assert page.inner_text("#here-num") == page.evaluate("Harmony.flat(H.chords[S.chord].numeral)")
    assert "bar" in page.inner_text("#where").lower() and page.locator("#path .chip.playing b").inner_text() == "C"
    assert page.locator("#map .st.cur").count() == 1
    assert page.locator("#map .st.cur").get_attribute("data-spot") == str(page.evaluate("H.chords[S.chord].on.roads"))
    page.wait_for_function("S.note >= 0 && document.querySelectorAll('#neck .sound').length === 1")
    # the neck shows what is played and nothing else: this bar's four places, joined, and the note that sounds
    places = page.evaluate("new Set(H.bass.filter(b => b.bar === S.bar + 1).map(b => b.string + ':' + b.fret)).size")
    assert page.locator("#neck .tone").count() == places >= 4 and page.locator("#neck .shape").count() == 1
    assert set(page.locator("#neck .tone text").all_text_contents()) == {"4", "5", "8", "R"}
    assert "The bass plays" in page.inner_text("#lead") and "string" in page.inner_text("#lead")
    page.wait_for_function("H.chords[S.chord].name === 'G'", timeout=20000)  # the song moves on, and the page with it
    assert page.inner_text("#here-name") == "G" and float(page.get_attribute("#strip-cursor", "x1")) > 0
    page.click("#play")
    page.wait_for_function(
        "document.getElementById('au').paused && document.getElementById('play').textContent === '▶ Play'"
    )

    # the other two maps show the same chord
    for view, spots in (
        ("wheel", 36),
        ("semitones", 36),
        ("roads", 21),
    ):  # the map's own chords; all of the song's are among them
        page.click(f"#lens-tabs [data-view='{view}']")
        assert page.locator("#map .st").count() == spots and page.locator("#map .st.cur").count() == 1
        assert page.locator("#map .st.cur").get_attribute("data-spot") == str(
            page.evaluate(f"H.chords[S.chord].on.{view}")
        )
    # a click on a chord of the map goes to where the song is on it next
    spot = page.evaluate("H.maps.roads.spots.findIndex(s => s.name === 'D')")
    page.click(f"#map .st[data-spot='{spot}']")
    page.wait_for_function("S.chord >= 0 && H.chords[S.chord].name === 'D'")
    page.click("#play")
    page.wait_for_function("document.getElementById('au').paused")

    # the fingering: for playing unless another way is picked; the neck follows, with the hand drawn on it
    assert page.get_attribute("#tabs [data-tab='easy']", "aria-pressed") == "true" and "hand move" in page.inner_text(
        "#effort"
    )
    assert page.locator("#neck .handbox").count() == 1
    before = page.evaluate("H.bass.map(n => [n.string, n.fret])")
    page.click("#tabs [data-tab='low']")
    page.wait_for_function("H.tab === 'low'")
    assert page.evaluate("H.bass.map(n => [n.string, n.fret])") != before
    assert page.get_attribute("#tabs [data-tab='low']", "aria-pressed") == "true"
    assert copilot.tab_style(os.path.join(site.workdir, song)) == "low"
    page.click("#tabs [data-tab='easy']")
    page.wait_for_function("H.tab === 'easy'")

    # home: the key with the same notes; the names follow, and the choice is kept
    before = page.evaluate("[H.home.name, H.chords[S.chord].numeral]")
    page.locator("#homes button[aria-pressed='false']").first.click()
    page.wait_for_function(f"H.home.name !== {json.dumps(before[0])}")
    after = page.evaluate("[H.home.name, H.chords[S.chord].numeral, H.home.yours]")
    assert after[1] != before[1] and after[2] is True
    assert page.inner_text("#here-num") == page.evaluate("Harmony.flat(H.chords[S.chord].numeral)")  # with a flat sign
    assert copilot_harmony.read_home(os.path.join(site.workdir, song)) is not None
    page.reload()
    page.wait_for_function("H !== null && document.querySelectorAll('#map .st').length > 10")
    assert page.evaluate("H.home.name") == after[0]
    assert page.locator("#profile circle").count() == page.evaluate("H.chords.length")
    assert page.locator("#roles .role").count() == 5 and "on their root" in page.inner_text("#bass-note")


def test_a_chord_is_heard_changed_in_every_round_and_given_back_on_the_harmony_page(browse, site, song):
    page = browse(f"{site.url}/harmony#{song}", ready="H !== null && document.querySelectorAll('#map .st').length > 20")
    loop = page.evaluate("H.parts.find(p => p.size === 4)")
    bar = loop["bar"] + 1  # the second bar of the first round: C
    page.evaluate(f"document.getElementById('au').currentTime = H.bars[{bar - 1}].time + 0.3")
    page.wait_for_function(f"S.bar === {bar - 1}")
    page.click("#change")
    assert page.inner_text("#edit-where").lower() == f"bar {bar} · part {loop['letter']}".lower()
    assert page.inner_text("#edit-heard") == "The tool heard C." and page.is_disabled("#edit-here")
    offered = page.locator("#edit-offers .chip b").all_inner_texts()
    assert offered[0] == "C" and "Am" in offered and len(offered) == len(set(offered))

    # a click on a chord plays the bar with it, in the keyboard sound, and changes nothing
    page.locator("#edit-offers .chip", has_text="Am").click()
    page.wait_for_function("S.trying !== null && !document.getElementById('au').paused && Keys.now === 'try'")
    assert page.evaluate("Keys.voices.length") == 3 and page.evaluate("H.fixed") == 0
    page.wait_for_function(
        "S.trying === null && document.getElementById('au').paused && Keys.now === null", timeout=20000
    )
    assert page.inner_text("#edit-here") == "Change this bar to Am" and page.is_enabled("#edit-here")
    assert page.inner_text("#edit-all") == f"Change all 3 bars like it in part {loop['letter']}"

    page.click("#edit-all")
    page.wait_for_function("H.fixed === 3")
    assert (
        page.inner_text("#edit-heard") == "You put Am here. The tool heard C." and page.inner_text("#here-name") == "Am"
    )
    assert copilot_harmony.read_fixes(os.path.join(site.workdir, song)) == {bar + 4 * k: [(9, "m")] for k in range(3)}
    chips = page.locator(
        f"#loops .verb[data-letter='{page.evaluate('H.parts.find(p => p.size === 4).letter')}'] .chip b"
    )
    assert chips.all_inner_texts() == ["Em", "Am", "G", "D"] and page.locator("#path .chip.yours").count() == 1
    assert page.is_visible("#edit-back") and page.is_hidden("#edit-all")

    page.click("#edit-back")  # this bar only: the two other rounds keep the person's chord
    page.wait_for_function("H.fixed === 2")
    assert page.inner_text("#edit-heard") == "The tool heard C." and page.inner_text("#here-name") == "C"
    page.click("#edit-close")
    assert page.is_hidden("#edit")

    # the chords of the page in the keyboard sound, over the record, for as long as it plays
    page.click("#chords")
    page.click("#play")
    page.wait_for_function("Keys.now !== null && Keys.now === 'c' + S.chord")
    page.click("#play")
    page.wait_for_function("document.getElementById('au').paused && Keys.now === null")


def test_a_harmony_page_without_its_song_says_so(browse, site):
    page = browse(site.url + "/harmony", ready="!document.getElementById('error').hidden")
    assert "No song was named" in page.inner_text("#error")


def test_the_test_songs_page_plays_a_song_and_keeps_the_persons_answer(browse, site):
    page = browse(site.url + "/tests", ready="T !== null && document.querySelectorAll('#sets .song').length === 1")
    row = page.locator("#sets .song[data-song='mix']")
    assert page.inner_text("#count") == "0 of 1 rated" and "Tiny" in page.inner_text("#sets h2")
    assert "sampled bass: a plucked string" in row.locator(".facts").inner_text()
    assert row.locator(".scores").inner_text().replace("\xa0", " ").split() == [
        "bass2sheet",
        "0.90",
        "MuScriptor",
        "0.95",
    ]
    assert "Nothing is rated yet" in page.inner_text("#summary")

    row.locator("button[data-rate='yes']").click()  # "Like my music"
    page.wait_for_function("T.songs.mix.rating === 'yes'")
    assert (
        "sel" in row.locator("button[data-rate='yes']").get_attribute("class")
        and page.inner_text("#count") == "1 of 1 rated"
    )
    assert testsongs.ratings(site.workdir)["mix"]["rating"] == "yes"
    assert "Nothing is rated yet" not in page.inner_text("#summary")
    row.locator("input[data-note]").fill("the bass is a synth")
    row.locator("input[data-note]").press("Tab")
    page.wait_for_function("T.songs.mix.note === 'the bass is a synth'")
    assert testsongs.ratings(site.workdir)["mix"] == dict(
        testsongs.ratings(site.workdir)["mix"], note="the bass is a synth"
    )

    row.locator("button[data-hear='record']").click()  # "Listen": from where the bass is busiest
    page.wait_for_function("!audio.paused && audio.currentTime > 0")
    assert page.is_visible("#player") and "mix" in page.inner_text("#what") and "on" in row.get_attribute("class")
    assert "sel" in page.get_attribute("#asRecord", "class")
    page.click("#asBass")  # the isolated bass, from the same moment
    page.wait_for_function("now.kind === 'bass' && !audio.paused && audio.src.endsWith('/audio/bass.flac')")
    page.click("#pause")
    page.wait_for_function("audio.paused")

    page.check("#hideRated")
    page.wait_for_function("document.querySelectorAll('#sets .song').length === 0")
    assert "1 rated and hidden" in page.inner_text("#sets h2")
    page.uncheck("#hideRated")
    page.locator("#sets .song button[data-rate='yes']").click()  # pressed again, the answer is taken back
    page.wait_for_function("T.songs.mix.rating === null")
    assert page.inner_text("#count") == "0 of 1 rated"
    kept = testsongs.ratings(site.workdir)["mix"]  # the note they wrote stays
    assert (kept["rating"], kept["note"]) == (None, "the bass is a synth")
    assert page.inner_text("#error") == ""
