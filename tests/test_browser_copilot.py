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


def _score_frame(page):
    """The score shown over the song's page, once its frame has the score page in it."""
    for _ in range(200):
        for frame in page.frames:
            if "/score" in frame.url:
                return frame
        page.wait_for_timeout(50)
    raise AssertionError("the score did not come up over the page")


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


def test_the_note_just_fixed_stays_in_sight_when_the_repeats_come_up_in_a_low_window(browse, site, song):
    page = _song_page(browse, site, song)
    page.set_viewport_size(dict(width=1440, height=720))
    page.wait_for_function("__quiet(200)")
    n = page.evaluate(IN_VIEW, "!(n.also || []).length && n.end - n.start > 0.3")
    _click_note(page, n["id"])
    tall = page.evaluate("document.getElementById('rollwrap').clientHeight")
    assert page.evaluate("rowsOf() * row <= rollHeight()")  # every pitch has its row in the window
    page.click("#panel button[data-act='split']")
    page.wait_for_function("note(sel) && note(sel).split_from && dock !== null && dock.loaded")
    page.wait_for_function("__quiet(300) && !A.playing")
    # the repeats took their room from the timeline: it is lower, and has fewer rows than there are pitches
    assert page.evaluate("document.getElementById('rollwrap').clientHeight") < tall
    assert page.evaluate("rowsOf() * row > rollHeight()")
    # the two halves of the note are on the timeline still, where a click reaches them, and not under the repeats
    for half in (n["id"], page.evaluate("sel")):
        at = page.evaluate(BOXES, half)[0]
        assert page.evaluate("([x, y]) => document.elementFromPoint(x, y).id", [at["x"], at["y"]]) == "roll", half
    _click_note(page, n["id"])
    # the neck gave way to the repeats, and the sliders and the pickers stayed
    assert page.is_hidden("#neck") and page.is_visible("#volRecord") and page.is_visible("#tabSel")


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
    frame = _score_frame(page)
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
    frame = _score_frame(page)
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


CHORD_AT = """bar => {  // where a bar's chord is in the row of chords, the timeline moved so that the bar is in sight
  const r = document.getElementById("roll").getBoundingClientRect();
  view.t0 = Math.max(-0.2, S.bars[bar].time - 1); view.free = true; draw();
  const [a, z] = barSpan(bar);
  return {x: r.left + xOf((a + z) / 2), y: r.top + RULER + LANE + CHORDS / 2};
}"""
STILL = "!A.playing && __quiet(200)"  # nothing sounds and nothing is on its way


def _chord(page, what):
    """A button of the chord's panel under the timeline."""
    return page.locator(f"#panel button[data-chord='{what}']")


def test_a_chord_is_heard_tried_on_and_used_on_the_songs_page(browse, site, song):
    page = _song_page(browse, site, song)
    folder = os.path.join(site.workdir, song)
    # the row of chords has the tool's best guess for every bar of the loop: none is the person's yet
    assert page.evaluate("S.chords.map(c => c.name)") == ["Em", "C", "G", "D"] * 3
    assert page.evaluate("S.yours") == [] and not page.evaluate("S.chords.some(c => c.yours)")

    # a click on the chord of bar 3: the bar is heard with it in the keyboard sound, and other chords are offered
    page.wait_for_function(STILL)
    at = page.evaluate(CHORD_AT, 2)
    page.mouse.click(at["x"], at["y"])
    page.wait_for_function("selChord !== null && selChord.bar === 2 && chordOffers !== null")
    assert page.evaluate("A.playing && A.chords !== null && A.chorded >= 1")
    assert page.inner_text("#panel b") == "C" and "bar 3 · the tool's guess" in _panel(page)
    offered = _chord(page, "try").all_inner_texts()
    assert len(offered) == 6 == len(set(offered)) and "C" not in offered and "Am" in offered
    assert _chord(page, "use").count() == 0 and _chord(page, "back").count() == 0
    page.wait_for_function(STILL, timeout=15000)

    # a chord is tried on: the bar plays with it in place of the one written, and nothing changes
    _chord(page, "try").filter(has_text=re.compile("^Am$")).click()
    page.wait_for_function("tryOn !== null && A.playing && A.chords.alt && A.chords.alt.name === 'Am'")
    assert page.evaluate("[A.chords.alt.bar, A.chords.alt.beat, S.chords.find(c => c.bar === 2).name]") == [2, 0, "C"]
    assert not os.path.exists(os.path.join(folder, "harmony.json"))
    assert _chord(page, "use").inner_text() == "Use Am here"
    assert _chord(page, "usePart").inner_text() == "…and in the 2 bars like it in this part"
    assert _chord(page, "useSong").count() == 0  # there are no others in the song
    page.wait_for_function(STILL, timeout=15000)
    _chord(page, "use").click()  # in this bar only
    page.wait_for_function(
        "S.yours.length === 1 && tryOn === null && chordOffers !== null && chordOffers.chord.name === 'Am'"
    )
    assert copilot_harmony.read_bars(folder) == {2: [(0, 9, "m")]}
    assert page.inner_text("#panel b") == "Am" and "chosen by you" in _panel(page)
    assert page.evaluate("S.chords.filter(c => c.yours).map(c => c.bar)") == [2]

    # back to the tool's guess; then the same chord for the three bars like it at once
    _chord(page, "back").click()
    page.wait_for_function("S.yours.length === 0 && chordOffers !== null && chordOffers.chord.name === 'C'")
    assert copilot_harmony.read_bars(folder) == {}
    _chord(page, "try").filter(has_text=re.compile("^Am$")).click()
    page.wait_for_function("tryOn !== null && A.playing")
    _chord(page, "usePart").click()
    page.wait_for_function("S.yours.length === 3 && tryOn === null && chordOffers !== null")
    assert copilot_harmony.read_bars(folder) == {bar: [(0, 9, "m")] for bar in (2, 6, 10)}

    # any other chord from the two boxes; a second chord from beat 3; one chord again; no chord
    page.wait_for_function(STILL, timeout=15000)
    page.select_option("#chordRoot", "9")
    page.select_option("#chordKind", "m7")
    _chord(page, "tryOther").click()
    page.wait_for_function("tryOn !== null && tryOn.name === 'Am7' && A.playing && A.chords.alt.tones.length === 4")
    _chord(page, "use").click()
    page.wait_for_function("chordOffers !== null && chordOffers.chord.name === 'Am7'")
    _chord(page, "split").click()
    page.wait_for_function(
        "selChord.beat === 2 && S.chords.filter(c => c.bar === 2).length === 2 && chordOffers !== null"
    )
    assert copilot_harmony.read_bars(folder)[2] == [(0, 9, "m7"), (2, 9, "m7")] and "from beat 3" in _panel(page)
    _chord(page, "join").click()
    page.wait_for_function(
        "selChord.beat === 0 && S.chords.filter(c => c.bar === 2).length === 1 && chordOffers !== null"
    )
    _chord(page, "none").click()
    page.wait_for_function(
        "S.chords.filter(c => c.bar === 2).length === 0 && chordOffers !== null && chordOffers.chord === null"
    )
    assert copilot_harmony.read_bars(folder)[2] == [] and page.inner_text("#panel b") == "No chord"
    assert "you said: no chord" in _panel(page)
    _chord(page, "back").click()
    page.wait_for_function("S.chords.filter(c => c.bar === 2).length === 1 && !S.yours.includes(2)")
    assert sorted(copilot_harmony.read_bars(folder)) == [6, 10]
    page.wait_for_function(STILL, timeout=15000)
    _chord(page, "close").click()
    assert page.evaluate("selChord") is None and "Click a note to select it" in _panel(page)

    # the chords' slider: up, they sound while the song plays, as loud as it says
    before = page.evaluate("A.chorded")
    _slide(page, "volChords", 50)
    page.evaluate("playhead = S.bars[5].time - 0.2")
    page.click("#play")
    page.wait_for_function(f"A.playing && A.chorded > {before} && Math.abs(A.bus.chords.gain.value - 0.25) < 0.01")
    page.click("#play")
    page.wait_for_function("!A.playing")
    _slide(page, "volChords", 0)
    before = page.evaluate("A.chorded")
    page.evaluate("playhead = S.bars[5].time - 0.2")
    page.click("#play")
    page.wait_for_function("A.playing && position() > S.bars[5].time + 0.3")
    page.click("#play")
    assert page.evaluate("A.chorded") == before  # and not when it is all the way down
    # all the way down, a chord that is clicked is still heard, at the level a click has
    page.wait_for_function(STILL)
    at = page.evaluate(CHORD_AT, 3)
    page.mouse.click(at["x"], at["y"])
    page.wait_for_function(f"A.playing && A.chords !== null && A.chorded > {before}")
    page.wait_for_function("Math.abs(A.bus.chords.gain.value - Logic.gainAt(Logic.CHORDS_ASKED)) < 0.01")
    assert page.evaluate("[hear.chords, +document.getElementById('volChords').value]") == [0, 0]  # the slider stays


def _slide(page, slider, to):
    """Put a slider somewhere, as dragging it does."""
    page.evaluate(
        """([id, to]) => {
          const e = document.getElementById(id);
          e.value = to;
          e.dispatchEvent(new Event("input", {bubbles: true}));
          e.dispatchEvent(new Event("change", {bubbles: true}));
        }""",
        [slider, to],
    )


# What comes out of the page's sound, worked out without playing it: the page's own chain of sounds (buildAudio) and
# its own chord (chordVoice) in a context that renders to numbers. Levels are in dB below full scale.
RENDERED = """async () => {
  const sr = 44100, db = v => 20 * Math.log10(Math.max(v, 1e-9));
  const level = (buf, t0, t1) => {
    let ss = 0, n = 0, peak = 0;
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = Math.floor(t0 * sr); i < Math.min(d.length, Math.floor(t1 * sr)); i++) {
        ss += d[i] * d[i]; n++; peak = Math.max(peak, Math.abs(d[i]));
      }
    }
    return {rms: db(Math.sqrt(ss / Math.max(1, n))), peak: db(peak)};
  };
  const render = async (len, fill) => {
    const off = new OfflineAudioContext(2, Math.ceil(sr * len), sr), G = buildAudio(off);
    fill(off, G);
    return off.startRendering();
  };
  const from = (G, off, k, gain) => {
    const src = off.createBufferSource();
    src.buffer = A[k]; G.bus[k].gain.value = gain; src.connect(G.bus[k]); src.start(0, 2);
  };
  const chord = S.chords[0], out = {};
  const alone = await render(2.4, (off, G) => chordVoice(chord, 0.1, 2, G));
  out.struck = level(alone, 0.12, 0.3).rms; out.held = level(alone, 1.0, 1.6).rms;
  out.at_its_end = level(alone, 1.7, 2.0).rms;
  out.after = level(alone, 2.2, 2.4).rms; out.chord_peak = level(alone, 0, 2.4).peak;
  const asked = await render(2.4, (off, G) => {
    G.bus.chords.gain.value = Logic.gainAt(Logic.CHORDS_ASKED);
    chordVoice(chord, 0.1, 2, G);
  });
  out.asked = level(asked, 1.0, 1.6).rms;
  out.record = level(A.record, 2, 10);
  out.record_through = level(await render(8, (off, G) => from(G, off, "record", 1)), 0, 8);
  const all = await render(8, (off, G) => {
    for (const k of ["record", "bass"]) from(G, off, k, 1);
    for (let t = 0; t < 8; t += 2) chordVoice({root: 9, tones: [9, 0, 4, 7]}, t, 2, G);
  });
  out.everything = level(all, 0, 8);
  // a note in the synthesized bass, an A of a second and a half: how loud it is when plucked, later, and after its
  // end; how much of it eight harmonics of the A explain, against eight of the note a semitone above; and how loud
  // each of two notes played together is, against one alone
  const a1 = 33, hz = p => 440 * Math.pow(2, (p - 69) / 12);
  const note = await render(2.2, (off, G) => synthVoice({start: 0, end: 1.5}, a1, 0.1, 1, G));
  const fits = f => {
    const d = note.getChannelData(0);
    let e = 0;
    for (let k = 1; k <= 8; k++) {
      let re = 0, im = 0;
      for (let i = Math.floor(0.2 * sr); i < Math.floor(1.4 * sr); i++) {
        const ph = 2 * Math.PI * k * f * i / sr;
        re += d[i] * Math.cos(ph); im += d[i] * Math.sin(ph);
      }
      e += re * re + im * im;
    }
    return e;
  };
  const pair = await render(1, (off, G) => synthVoice({start: 0, end: 0.8}, a1, 0.1, 2, G));
  out.synth = {plucked: level(note, 0.105, 0.4).rms, later: level(note, 1.2, 1.5).rms, after: level(note, 1.7, 2.1).rms,
    peak: level(note, 0, 2.2).peak, a_semitone_off: 10 * Math.log10(fits(hz(a1 + 1)) / fits(hz(a1))),
    of_two: level(pair, 0.105, 0.4).rms - level(note, 0.105, 0.4).rms};
  return out;
}"""


def test_each_sound_has_a_slider_beside_the_neck_and_the_chords_are_loud_enough_to_hear(browse, site, song):
    page = _song_page(browse, site, song)
    # a slider for each sound at the bottom right, beside the neck; no boxes to tick for them
    for gone in ("#hearRecord", "#hearBass", "#hearNotes", "#hearChords"):
        assert page.locator(gone).count() == 0
    assert page.locator("#mixer label span.soft").all_inner_texts() == [
        "Record",
        "Isolated bass",
        "Notes: sampled",
        "Notes: synth",
        "Chords",
    ]
    where = page.evaluate("""() => {
      const box = id => document.getElementById(id).getBoundingClientRect(), neck = box("neck"), mixer = box("mixer");
      return {right_of_the_neck: mixer.left >= neck.right,
        beside_it: mixer.top < neck.bottom && mixer.bottom > neck.top,
        at_the_bottom: innerHeight - box("neckbar").bottom < 2, in_the_window: mixer.right <= innerWidth,
        sliders: [...document.querySelectorAll("#mixer input[type=range]")].map(e => [e.id, +e.value])};
    }""")
    assert where == dict(
        right_of_the_neck=True,
        beside_it=True,
        at_the_bottom=True,
        in_the_window=True,
        # the two recordings, the notes in two sounds, the chords: the record and the notes in the song's sound are up
        sliders=[["volRecord", 100], ["volBass", 0], ["volNotes", 100], ["volSynth", 0], ["volChords", 0]],
    )
    assert page.is_visible("#hearSplit") and not page.is_checked("#hearSplit")
    # a slider sets how loud its sound is, at once and for the next visit
    page.evaluate("playhead = S.bars[2].time")
    page.click("#play")
    page.wait_for_function("A.playing && Math.abs(A.bus.record.gain.value - 1) < 0.01 && A.bus.bass.gain.value < 0.01")
    _slide(page, "volRecord", 50)
    _slide(page, "volBass", 100)
    _slide(page, "volNotes", 0)
    page.wait_for_function(
        "Math.abs(A.bus.record.gain.value - 0.25) < 0.01 && Math.abs(A.bus.bass.gain.value - 1) < 0.01"
        " && A.bus.notes.gain.value < 0.01"
    )
    page.click("#play")
    page.wait_for_function("!A.playing")
    assert json.loads(page.evaluate("localStorage.getItem('copilot.hear')")) == dict(
        record=50, bass=100, notes=0, synth=0, chords=0, split=False
    )
    # the notes have two sounds, each with its slider: the song's own bass, and a synthesized one. The synthesized
    # one is made only while its slider is up
    assert page.evaluate("A.synthed || 0") == 0
    _slide(page, "volSynth", 80)
    page.evaluate("playhead = S.bars[2].time")
    page.click("#play")
    page.wait_for_function("A.playing && A.synthed >= 3 && Math.abs(A.bus.synth.gain.value - 0.64) < 0.01")
    assert page.evaluate("A.pan.synth.pan.value") == 0
    page.check("#hearSplit")  # the bass in one ear, the notes in the other, in whichever sound
    page.wait_for_function(
        "A.pan.synth.pan.value > 0.99 && A.pan.notes.pan.value > 0.99 && A.pan.bass.pan.value < -0.99"
    )
    page.click("#play")
    page.wait_for_function("!A.playing")
    assert page.evaluate("hear.split") is True and page.evaluate("hear.synth") == 80
    page.reload()
    page.wait_for_function(SONG_READY % json.dumps(song))
    assert page.evaluate("[...SOUNDS.map(k => +volOf(k).value), hearSplit.checked]") == [50, 100, 0, 80, 0, True]
    # what was kept from the time each sound was a box to tick is taken as sliders up or down
    once = dict(record=True, bass=False, notes=True, split=False, chords=False)
    page.evaluate("kept => localStorage.setItem('copilot.hear', JSON.stringify(kept))", once)
    page.reload()
    page.wait_for_function(SONG_READY % json.dumps(song))
    assert page.evaluate("hear") == dict(record=100, bass=0, notes=100, synth=0, chords=0, split=False)

    # the neck put away: the sliders and the pickers stay, in a row
    page.click("#neckBtn")
    assert page.is_hidden("#neck") and page.is_visible("#volChords") and page.is_visible("#homeSel")
    page.click("#neckBtn")
    assert page.is_visible("#neck")

    # what comes out. A chord holds to its end: it is struck a little louder, then stays, and is gone after
    got = page.evaluate(RENDERED)
    assert got["struck"] - 3 < got["held"] < got["struck"] and abs(got["at_its_end"] - got["held"]) < 0.5, got
    assert got["after"] < got["held"] - 40, got
    # with its slider at 100 a chord is about as loud as a loud record, and alone it fits the output; a chord that is
    # clicked with the slider down is 4 dB under that
    assert -13 < got["held"] < -9 and got["chord_peak"] < 0 and 3.5 < got["held"] - got["asked"] < 4.5, got
    # the record alone comes out as it is; everything at once, at 100, is held under what the output can carry
    assert abs(got["record_through"]["rms"] - got["record"]["rms"]) < 0.3, got
    assert got["record_through"]["peak"] <= got["record"]["peak"] + 0.1, got
    assert got["everything"]["peak"] < 1.0 and got["everything"]["rms"] > got["record"]["rms"], got
    # a note in the synthesized bass is plucked and dies away slowly, is gone after its end, and is the note asked for
    synth = got["synth"]
    assert -14 < synth["plucked"] < -10 and synth["peak"] < -3, synth  # about as loud as a chord, with room over it
    assert synth["plucked"] - 12 < synth["later"] < synth["plucked"] - 3 and synth["after"] < -100, synth
    assert synth["a_semitone_off"] < -15, synth
    assert -3.3 < synth["of_two"] < -2.7, synth  # each of two notes played together: 3 dB under one alone


# The playhead is put on every note of the song and after it: every place the neck is found in, and every height of
# the bar that holds it. One of each means the neck stays put while the song plays.
NECK_STAYS_ON_THE_SONGS_PAGE = """() => {
  const seen = new Set();
  for (const n of Logic.sorted(S.notes)) for (const t of [n.start + 0.01, n.end + 0.005]) {
    playhead = t; refresh();
    const r = document.getElementById("neck").getBoundingClientRect();
    seen.add([r.top.toFixed(1), r.height.toFixed(1), document.getElementById("neckbar").offsetHeight].join(" "));
  }
  return [...seen];
}"""


def test_the_key_the_fingering_and_the_neck_are_on_the_songs_page(browse, site, song):
    page = _song_page(browse, site, song)
    folder = os.path.join(site.workdir, song)
    # the neck does not move as the song goes from note to note, in a wide window or a narrow one
    for width in (1280, 1000, 820):
        page.set_viewport_size(dict(width=width, height=800))
        page.wait_for_function("__quiet(100)")
        assert len(page.evaluate(NECK_STAYS_ON_THE_SONGS_PAGE)) == 1, width
    page.set_viewport_size(dict(width=1280, height=720))
    page.wait_for_function("__quiet(100)")
    # the neck under the timeline: the notes of the bar at the playhead, and the one that sounds there
    page.evaluate("playhead = S.bars[1].time + 0.1; refresh()")
    in_bar = "S.notes.filter(n => n.start + 0.03 >= S.bars[1].time && n.start + 0.03 < S.bars[2].time)"
    places = page.evaluate(f"new Set({in_bar}.map(n => S.neck[n.id].slice(0, 2).join(':'))).size")
    assert page.is_visible("#neck") and page.locator("#neck .tone").count() == places >= 3
    assert page.locator("#neck .sound").count() == 1 and page.locator("#neck polyline").count() == 1
    # a circle has the note's name; where the note is played and what it is to the key are under the pointer, and
    # no words stand beside the neck
    assert set(page.locator("#neck .tone text").all_text_contents()) == {"E", "B", "A"}
    assert page.locator("#neck .sound text").text_content() == "E"
    assert page.locator("#neck .sound title").text_content() == "E1: the open E string · step 6 of G major"
    assert "E1: the open E string · step 6 of G major" in page.locator("#neck .tone title").all_text_contents()
    assert page.locator("#neckSays").count() == 0
    assert page.evaluate(
        "[...document.getElementById('sheetside').children].map(e => e.id || e.htmlFor || e.tagName)"
    ) == [
        "mixer",  # the sliders, then the two pickers
        "LABEL",
        "LABEL",
    ]
    # the neck is put away and taken out again; the timeline has the room meanwhile
    tall = page.evaluate("document.getElementById('rollwrap').clientHeight")
    page.click("#neckBtn")
    assert page.is_hidden("#neck") and page.evaluate("document.getElementById('rollwrap').clientHeight") > tall + 60
    page.click("#neckBtn")
    assert page.is_visible("#neck") and page.evaluate("document.getElementById('rollwrap').clientHeight") == tall
    n = page.evaluate(IN_VIEW, "!(n.also || []).length && n.end - n.start > 0.2")
    _click_note(page, n["id"])  # a note that is chosen is the one filled in
    assert page.locator("#neck .sound").count() == 1
    assert (
        page.locator("#neck .sound title").text_content().startswith(page.evaluate("p => name(p)", n["pitch"]) + ": ")
    )

    # the key: the tool's guess until another is picked; the steps and the chords are named by it, and it is kept
    assert "the tool's guess" in page.locator("#homeSel option:checked").inner_text()
    page.keyboard.press("Escape")
    page.evaluate("playhead = S.bars[1].time + 0.1; refresh()")
    page.select_option("#homeSel", "4,0")
    page.wait_for_function("S.home.name === 'E minor' && S.home.yours")
    assert copilot_harmony.read_home(folder) == (4, False) and page.evaluate("S.chords[0].numeral") == "i"
    assert set(page.locator("#neck .tone text").all_text_contents()) == {"E", "B", "A"}  # the names stay
    assert page.locator("#neck .sound title").text_content() == "E1: the open E string · step 1 of E minor"
    # the notes are named as the sheet spells them in the key: with flats in a key that has flats
    assert page.evaluate("[name(42), name(46), name(40)]") == ["F#2", "A#2", "E2"]
    page.select_option("#homeSel", "5,1")
    page.wait_for_function("S.home.name === 'F major'")
    assert page.evaluate("[name(42), name(46), name(40)]") == ["Gb2", "Bb2", "E2"]
    page.select_option("#homeSel", "7,1")  # the tool's guess again
    page.wait_for_function("S.home.name === 'G major' && !S.home.yours")
    assert copilot_harmony.read_home(folder) is None

    # the fingering: for playing unless another way is picked; the neck follows, and the sheet's tab will
    assert page.locator("#tabSel option:checked").inner_text().startswith("Fewest hand moves · ")
    before = page.evaluate("JSON.stringify(S.neck)")
    page.select_option("#tabSel", "low")
    page.wait_for_function("S.tab === 'low'")
    assert page.evaluate("JSON.stringify(S.neck)") != before and copilot.tab_style(folder) == "low"
    assert page.evaluate("Object.values(S.neck).some(p => p[1] === 0)")  # open strings
    page.select_option("#tabSel", "easy")
    page.wait_for_function("S.tab === 'easy'")
    assert page.evaluate("JSON.stringify(S.neck)") == before

    # a note that is fixed is put on the neck anew
    _click_note(page, n["id"])
    was = page.evaluate("id => S.neck[id].slice(0, 2)", n["id"])
    page.keyboard.press("Shift+ArrowUp")  # an octave up
    page.wait_for_function("([id, p]) => note(id).pitch === p", arg=[n["id"], n["pitch"] + 12])
    assert page.evaluate("id => S.neck[id].slice(0, 2)", n["id"]) != was
    assert page.evaluate("id => S.strings[S.neck[id][0]] + S.neck[id][1] === note(id).pitch", n["id"])


HARMONY_READY = "H !== null && document.querySelectorAll('#map .st').length > 20"
# The song is gone through a tenth of a second at a time, the bars without bass or chord too: every place the neck
# is found in, and every height of the bar that holds it. One of each means the neck stays put while the song plays.
NECK_STAYS = """() => {
  const au = document.getElementById("au"), seen = new Set();
  for (let t = 0; t < H.length - 0.2; t += 0.11) {
    au.currentTime = t; tick(true);
    const r = document.getElementById("neck").getBoundingClientRect();
    seen.add([r.top.toFixed(1), r.height.toFixed(1), document.getElementById("now").offsetHeight].join(" "));
  }
  au.currentTime = 0; tick(true);
  return [...seen];
}"""


def test_the_harmony_page_shows_how_the_song_moves_and_changes_nothing(browse, site, song):
    folder = os.path.join(site.workdir, song)
    copilot_harmony.set_bars(folder, [dict(bar=2, chords=[[0, 9, "m"]])])  # the person's chord for bar 3: A minor
    with open(os.path.join(folder, "harmony.json"), "rb") as f:
        kept = f.read()
    page = browse(f"{site.url}/harmony#{song}", ready=HARMONY_READY)
    assert "A song of" in page.inner_text("#title")
    # no way to change anything: what is set is said, with the way to the page where it is changed
    for gone in ("#edit", "#tapmode", "#tabs", "#homes button", "#other-home"):
        assert page.locator(gone).count() == 0
    there = f"/song#{song}"
    assert page.get_attribute("#to-change", "href") == there and page.get_attribute("#homes a", "href") == there
    assert page.inner_text("#homes").startswith("G major · the tool's guess")
    assert "the fewest hand moves" in page.inner_text("#effort") and page.get_attribute("#effort a", "href") == there
    assert page.evaluate("document.getElementById('effort').closest('.card') !== null")  # in a card, not by the neck
    assert "You chose the chords of 1 of 12 bars" in page.inner_text("#short")
    assert "the tool's best guess" in page.inner_text("#map-hint")
    loop = page.evaluate("H.parts.find(p => p.size === 4)")
    chips = page.locator(f"#loops .verb[data-letter='{loop['letter']}'] .chip")
    assert chips.locator("b").all_inner_texts() == ["Em", "C", "G", "D"]  # what most rounds have in each bar
    # the neck does not move as the song goes from note to note, in a wide window or a narrow one
    for width in (1280, 1000, 820):
        page.set_viewport_size(dict(width=width, height=800))
        assert len(page.evaluate(NECK_STAYS)) == 1, width
    page.set_viewport_size(dict(width=1280, height=720))

    # the part is played: the chord, its job, the bar, the neck and the map follow the song
    page.click(f"#loops [data-play='{loop['letter']}']")
    page.wait_for_function("!document.getElementById('au').paused && S.chord >= 0 && H.chords[S.chord].name === 'Am'")
    assert page.inner_text("#play") == "■ Stop" and page.inner_text("#here-name") == "Am"
    assert page.inner_text("#here-num") == page.evaluate("Harmony.flat(H.chords[S.chord].numeral)")
    assert page.inner_text("#where") == "1 of 3" and page.locator("#path .barbox").count() == 4  # the round, of three
    assert page.get_attribute("#path .partmark", "title") == f"Part {loop['letter']}, round 1 of 3"
    playing = page.locator("#path .barbox.playing .cell")
    assert playing.inner_text() == "Am" and "guess" not in playing.get_attribute("class")  # the person's: not a guess
    assert (
        page.locator("#path .cell.guess").count() == 3 and page.locator("#path .barbox.playing .beat.cur").count() == 1
    )
    assert page.locator("#map .st.cur").count() == 1
    assert page.locator("#map .st.cur").get_attribute("data-spot") == str(page.evaluate("H.chords[S.chord].on.roads"))
    # in the strip of parts the one that plays has a line round it: drawn over every part, so that the part after
    # it covers none of its sides, and inside the part's own box, so that it covers nothing of its neighbours
    line = page.evaluate("""() => {
      const on = document.querySelectorAll("#strip .part.on"), o = document.getElementById("strip-on").getBBox();
      const r = on[0].getBBox(), rects = [...document.querySelectorAll("#strip rect")];
      return {parts: on.length, mine: on[0] === document.querySelectorAll("#strip .part")[S.part],
        seen: document.getElementById("strip-on").getAttribute("visibility"), last: rects[rects.length - 1].id,
        inside: o.x >= r.x && o.x + o.width <= r.x + r.width && o.y >= r.y && o.y + o.height <= r.y + r.height,
        covered: (b => document.elementsFromPoint(b.right - 1, b.top + b.height / 2))(on[0].getBoundingClientRect())
          .filter(e => e.classList.contains("part")).length};
    }""")
    assert line == dict(parts=1, mine=True, seen="visible", last="strip-on", inside=True, covered=1)
    page.wait_for_function("S.note >= 0 && document.querySelectorAll('#neck .sound').length === 1")
    # the neck shows what is played and nothing else: a circle for each place, with the note's name. What the note
    # is (under the person's chord, its part of the chord) and where it is played are under the pointer; no words
    # stand beside the neck
    places = page.evaluate("new Set(H.bass.filter(b => b.bar === S.bar + 1).map(b => b.string + ':' + b.fret)).size")
    assert page.locator("#neck .tone").count() == places >= 4 and page.locator("#neck .shape").count() == 1
    assert set(page.locator("#neck .tone text").all_text_contents()) == {"C", "G", "F"}
    tips = page.locator("#neck .tone title").all_text_contents()
    assert {tip.split(" · ")[0] for tip in tips} == {
        "C: the third of Am",
        "G: the seventh of Am",
        "F: a passing note over Am",
    }
    assert all(tip.endswith(" string") for tip in tips)
    for gone in ("#lead", "#circles", ".kside"):
        assert page.locator(gone).count() == 0
    assert page.evaluate("document.querySelector('.kbd').children.length") == 1  # the neck, and nothing else
    page.wait_for_function("H.chords[S.chord].name === 'G'", timeout=20000)  # the song moves on, and the page with it
    assert page.inner_text("#here-name") == "G" and float(page.get_attribute("#strip-cursor", "x1")) > 0
    assert set(page.locator("#neck .tone text").all_text_contents()) <= {"G", "D", "C"}
    assert page.locator("#neck .sound title").text_content().split(" · ")[0] in (  # a guess: steps of the key
        "G: step 1 of G major",
        "D: step 5 of G major",
        "C: step 4 of G major",
    )
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
    # a click on a chord of the map goes to where the song is on it next; a click on a bar of the row goes to it
    spot = page.evaluate("H.maps.roads.spots.findIndex(s => s.name === 'D')")
    page.click(f"#map .st[data-spot='{spot}']")
    page.wait_for_function("S.chord >= 0 && H.chords[S.chord].name === 'D'")
    # the chords in the keyboard sound, over the record, for as long as it plays
    page.click("#chords")
    page.wait_for_function("Keys.now !== null && Keys.now.startsWith('c' + S.chord + ':')")
    page.click("#play")
    page.wait_for_function("document.getElementById('au').paused && Keys.now === null")
    page.locator("#path .barbox").first.click()
    page.wait_for_function("S.bar === +document.querySelector('#path .barbox').dataset.bar")
    assert page.is_visible("#profile-card") and page.locator("#profile circle").count() == 12
    assert page.locator("#roles .role").count() == 11 and "on their root" in page.inner_text("#bass-note")
    with open(os.path.join(folder, "harmony.json"), "rb") as f:
        assert f.read() == kept  # nothing was changed


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
