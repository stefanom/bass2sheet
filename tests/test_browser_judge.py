"""The judging page in a real browser: opened from the real server on three comparisons with sound and pictures
of their own, and judged as a person does it, with clicks and keys.

Skipped where Playwright or a browser to drive is missing (see the `browser` fixture in conftest.py)."""

import os
import threading

import numpy as np
import pytest
import soundfile as sf
from test_judge import _comp

from bass2sheet import judge, judge_server
from bass2sheet.common import SR

pytestmark = pytest.mark.browser
READY = "typeof current !== 'undefined' && !!current && document.querySelectorAll('#picks button').length === 4"


@pytest.fixture(scope="module")
def judging(browser, tmp_path_factory):
    """(address, working directory) of the judging server on three comparisons of one song: which model isolates
    the bass, which notes are found, which tab. Every sound they name is nine seconds of a tone of its own, and
    every picture a page of the size the comparison says it has."""
    workdir = str(tmp_path_factory.mktemp("judging"))
    comps = [
        _comp("song-one.separator.1", flip=True),
        _comp("song-one.replucks.1", "replucks"),
        _comp("song-one.tab.1", "tab"),
    ]
    judge.save_comparisons(workdir, comps)
    t = np.arange(9 * SR) / SR
    for c in comps:
        os.makedirs(os.path.join(workdir, "media", c["id"]))
        sounds = {*c["media"].values(), *(c.get("alt") or {}).get("media", {}).values()}
        for k, rel in enumerate(sorted(sounds)):
            sf.write(os.path.join(workdir, rel), 0.2 * np.sin(2 * np.pi * (110 + 55 * k) * t), SR)
        for sheet in c["sheets"].values():
            width, height = sheet["page"]
            page = np.full((height, width, 3), 255, np.uint8)
            page[int(sheet["staff"][0]) : int(sheet["staff"][1]) : 20, 40:-40] = 0  # something like staff lines
            judge.write_png(os.path.join(workdir, sheet["src"]), page)
    srv = judge_server.make_server(workdir, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", workdir
    srv.shutdown()
    srv.server_close()


def test_a_passage_is_heard_both_ways_judged_and_only_then_told_what_the_versions_were(browse, judging):
    url, workdir = judging
    page = browse(url + "/", ready=READY)
    assert page.inner_text("#title") == "The Band – Song One" and page.inner_text("#progress") == "0 of 3 judged"
    assert page.inner_text("#stepname") == "Step 1 of 6 · Isolate the bass" and page.inner_text("#where").startswith(
        "bars 5–8"
    )
    assert page.locator("#players button").all_inner_texts() == ["▶ Record", "▶ A", "▶ B"]
    assert page.locator("#picks button").all_inner_texts() == [
        "A is better",
        "B is better",
        "No difference",
        "Neither is right",
    ]
    seen = page.inner_text("#judging")
    assert not any(secret in seen for secret in ("RoFormer", "Demucs", "default"))  # nothing says which is which
    assert (
        page.is_hidden("#reveal") and page.is_hidden("#next") and page.is_visible("#skip") and page.is_disabled("#back")
    )
    page.wait_for_function("[...document.querySelectorAll('#sheets img')].every(i => i.complete && i.naturalWidth > 0)")
    assert page.locator("#sheets .score").count() == 2 and page.locator("#sheets .mark").count() == 1
    steps = page.locator("#steps button")
    assert steps.count() == 6 and "current" in steps.nth(0).get_attribute("class") and steps.nth(3).is_disabled()

    # A, then B from the same moment, then nothing
    cursor = lambda: float(page.evaluate("parseFloat(document.getElementById('cursor-A').style.left)"))
    start = cursor()
    page.click("#players button[data-name='A']")
    page.wait_for_function("playing === 'A' && !audio.A.paused && audio.A.currentTime > 0.5")
    assert "on" in page.get_attribute("#players button[data-name='A']", "class")
    assert "active" in page.get_attribute("#score-A", "class") and cursor() > start  # the cursor runs over the score
    page.keyboard.press("b")
    page.wait_for_function("playing === 'B' && !audio.B.paused && audio.A.paused && audio.B.currentTime > 0.5")
    page.keyboard.press("Space")
    page.wait_for_function("playing === null && audio.B.paused")
    assert page.inner_text("#error") == ""

    page.fill("#note", "muddy")
    page.click("#picks button[data-pick='A']")
    page.wait_for_function("!document.getElementById('reveal').classList.contains('hidden')")
    said = page.inner_text("#reveal")
    assert "A was bass isolated by Demucs" in said and "B was bass isolated by BS-RoFormer (the default)" in said
    assert page.inner_text("#progress") == "1 of 3 judged" and page.is_visible("#next") and page.is_hidden("#skip")
    assert "chosen" in page.get_attribute("#picks button[data-pick='A']", "class")
    kept = judge.load_judgments(workdir)["song-one.separator.1"]
    assert (kept["pick"], kept["value"], kept["note"]) == ("A", "demucs", "muddy")

    page.click("#next")  # the next passage: the notes found, on a picture of the bass track
    page.wait_for_function("current.id === 'song-one.replucks.1'")
    assert page.inner_text("#stepname") == "Step 2 of 6 · Find the notes" and page.input_value("#note") == ""
    assert page.locator("#sheets svg rect.n").count() == 4 and page.locator("#sheets svg polyline.line").count() == 4
    assert page.locator("#players button").all_inner_texts() == ["▶ Isolated bass", "▶ A", "▶ B"]
    assert page.is_visible("#soloOpt") and page.inner_text("#soloLabel") == "Real bass left, notes right"
    page.click("#picks button[data-pick='neither']")
    page.wait_for_function("!document.getElementById('why').classList.contains('hidden')")
    page.click("#why button[data-why='too low']")  # what is wrong, in one click: a phrase in the note
    page.wait_for_function("document.getElementById('note').value === 'too low' && current.note === 'too low'")
    kept = judge.load_judgments(workdir)["song-one.replucks.1"]
    assert (kept["pick"], kept["value"], kept["note"]) == ("neither", None, "too low")

    page.click("#back")  # the one before, with its pick and what the versions were
    page.wait_for_function("current.id === 'song-one.separator.1' && current.pick === 'A'")
    assert page.input_value("#note") == "muddy" and page.is_visible("#reveal")
    page.click("#picks button[data-pick='B']")  # a pick can be changed
    page.wait_for_function("current.pick === 'B'")
    assert judge.load_judgments(workdir)["song-one.separator.1"]["value"] == "roformer"
    assert page.inner_text("#progress") == "2 of 3 judged"

    page.click("#toggleResults")  # the picks so far
    page.wait_for_function("document.getElementById('results').innerText.includes('Isolate the bass')")
    rows = page.eval_on_selector_all("#results tr", "rows => rows.map(r => [...r.cells].map(c => c.innerText))")
    assert ["bass isolated by BS-RoFormer", "1"] in rows and ["bass isolated by Demucs", "0"] in rows
    assert rows.count(["Neither is right", "1"]) == 1 and rows.count(["Not judged yet", "1"]) == 1
    assert "Song One: too low" in page.inner_text("#results") and page.is_hidden("#judging")
    assert page.inner_text("#toggleResults") == "Back to judging"
    page.click("#toggleResults")
    page.wait_for_function("!document.getElementById('judging').classList.contains('hidden')")

    page.click("#steps button[data-step='tab']")  # a step is opened by its name; the tab is the last one
    page.wait_for_function("current.id === 'song-one.tab.1'")
    assert page.locator("#players button").all_inner_texts() == ["▶ Record", "▶ A"]  # one sound: the notes are the same
    page.click("#score-B")  # a click on a score plays from the note under it
    page.wait_for_function("playing === 'A' && !audio.A.paused")
    page.click("#stop")
    page.click("#picks button[data-pick='same']")
    page.wait_for_function("state.judged === 3")
    page.click("#next")
    page.wait_for_function("!document.getElementById('done').classList.contains('hidden')")  # nothing left to judge
    assert page.is_hidden("#judging") and page.inner_text("#progress") == "3 of 3 judged"
    assert judge.tally(workdir)["tab"]["same"] == 1
