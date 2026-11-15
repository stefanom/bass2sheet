"""Shared fixtures. Tests use synthesized audio only (tests/songsynth.py): no recordings are needed.

Tiers:
  default      unit tests plus the end-to-end run on pre-separated synthetic stems (no model download)
  -m full      also the tests that need the separation models and, optionally, an evaluation dataset;
               enable with BASS2SHEET_FULL_E2E=1 (and BASS2SHEET_EVAL_DATA=<BabySlakh dir> for the regression)
  -m browser   the pages in a real browser (tests/test_browser_*.py); skipped where Playwright or a browser to
               drive is missing

BASS2SHEET_TEST_BARE=1 runs the tests as a machine with nothing but `uv sync` on it would: the MuseScore and the
Rubber Band that are installed here are not found. That is what CI has, and what its coverage floor is set from.
"""

import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TQDM_DISABLE", "1")

if os.environ.get("BASS2SHEET_TEST_BARE") == "1":  # before any test module asks whether MuseScore is installed
    import imageio_ffmpeg

    from bass2sheet import copilot, score

    score.MSCORE_CANDIDATES = []
    copilot.stretcher = lambda: ("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())


def mscore_stand_in(directory):
    """Writes a program that takes MuseScore's place (tests/mscore_standin.py) into `directory` and returns its
    path. A test puts it where the tool looks for MuseScore: monkeypatch.setattr(score, "MSCORE_CANDIDATES", [path])."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mscore_standin.py")
    program = os.path.join(str(directory), "mscore")
    with open(program, "w") as f:
        f.write(f"#!{sys.executable}\nimport runpy, sys\nsys.argv[0] = {script!r}\n")
        f.write(f"runpy.run_path({script!r}, run_name='__main__')\n")
    os.chmod(program, os.stat(program).st_mode | stat.S_IEXEC)
    return program


@pytest.fixture(scope="session")
def song():
    from songsynth import make_song

    return make_song()


@pytest.fixture(scope="session")
def song_files(song, tmp_path_factory):
    """(mix path, stems dir) of the synthetic song on disk."""
    from songsynth import write_song

    return write_song(song, str(tmp_path_factory.mktemp("song")))


@pytest.fixture(scope="session")
def taste_song_files(tmp_path_factory):
    """(mix path, stems dir) of a synthetic song with passages that can be written either way (see make_song)."""
    from songsynth import make_song, write_song

    return write_song(make_song(taste=True), str(tmp_path_factory.mktemp("taste_song")))


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    """A private cache so tests never read or write ~/.cache/bass2sheet."""
    return str(tmp_path_factory.mktemp("cache"))


def note(start, end, pitch, **kw):
    """A note dict as the pipeline carries it after quantization."""
    d = dict(start=start, end=end, pitch=pitch, agree=1.0, flag=None)
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------ the pages in a browser


@pytest.fixture(scope="session")
def browser():
    """A browser without a window to open the pages in: the Chromium Playwright installs (`playwright install
    chromium`), else the Chrome or the Edge that is on the machine. The tests that need it are skipped where there
    is no Playwright or no such browser. BASS2SHEET_TEST_BROWSER=chromium (or chrome, or msedge) asks for one by
    name: then it has to be there, and a browser that is missing fails the tests instead of skipping them."""
    named = os.environ.get("BASS2SHEET_TEST_BROWSER")
    missing = pytest.fail if named else pytest.skip
    if os.environ.get("BASS2SHEET_TEST_BARE") == "1":
        pytest.skip("a machine with nothing but `uv sync` on it has no browser to drive")
    try:
        from playwright import sync_api
    except ImportError:
        missing("Playwright is not installed: uv sync --group browser")
    channels = [None if named == "chromium" else named] if named else [None, "chrome", "msedge"]
    # no GPU: what is drawn is the same on every machine. No sound device: what is played goes nowhere, at the pace
    # it would be heard at, so nothing sounds in the room and a machine without a sound card plays just the same.
    # And sound may start without a click having asked for it.
    args = ["--disable-gpu", "--disable-audio-output", "--autoplay-policy=no-user-gesture-required"]
    with sync_api.sync_playwright() as playwright:
        found, why = None, []
        for channel in channels:
            try:
                found = playwright.chromium.launch(
                    headless=True, args=args, **({"channel": channel} if channel else {})
                )
                break
            except Exception as e:  # not installed
                why.append(f"{channel or 'chromium'}: {str(e).splitlines()[0]}")
        if found is None:
            missing("no browser to drive (uv run playwright install chromium): " + "; ".join(why))
        yield found
        found.close()


# What a test waits for before it aims at something on a page: `__quiet(ms)` is true once the page has asked the
# server for nothing for that long. An answer can lay the page out again (a row of buttons, a list under the
# timeline), and what was aimed at before it came is somewhere else after.
QUIET = """(() => {
  const ask = window.fetch;
  let open = 0, since = performance.now();
  window.fetch = function (...args) {
    open++;
    const done = () => { if (--open === 0) since = performance.now(); };
    const answer = ask.apply(this, args);
    answer.then(done, done);
    return answer;
  };
  window.__quiet = ms => open === 0 && performance.now() - since >= ms;
})()"""


@pytest.fixture
def browse(browser):
    """Opens pages: browse(address, ready=a condition in the page's own words to wait for). Every test starts
    from a browser that remembers nothing. When the test is over, no page may have said anything was wrong: an
    error on its console (a request refused is one) or an error in its script fails the test."""
    context = browser.new_context(viewport=dict(width=1440, height=900))
    context.set_default_timeout(30000)
    context.add_init_script(QUIET)
    wrong = []

    def open_page(address, ready=None, size=None):
        page = context.new_page()
        if size:
            page.set_viewport_size(dict(width=size[0], height=size[1]))
        page.on(
            "console",
            lambda m: wrong.append(f"console: {m.text} ({m.location.get('url', '')})") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: wrong.append(f"script: {e}"))
        page.goto(address)
        if ready:
            page.wait_for_function(ready)
        return page

    open_page.wrong = wrong
    yield open_page
    context.close()
    assert not wrong, wrong
