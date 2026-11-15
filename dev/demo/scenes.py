"""The film's scenes: what the hand does on each page, and which line is said when.

Every scene drives a real page of the tool (or one of the film's own pages in stage/: the terminal, the cards)
and leaves a folder of frames, marks and sound events (film.py). What is shown is what happens: the command is
run while it is filmed, the song is prepared while it is filmed, every fix is made on the page by mouse events.

What is particular to the song is in its storyboard (songs/NAME.json): where the tool's mistakes are and what is
right there. A scene looks for each of them in what the tool wrote this time; if one is not there any more (the
tool got better, or worse), the film stops and says which, because the narration would no longer be true.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

SONG_READY = "typeof S !== 'undefined' && S && tiles.length > 0 && A.bass && A.record"


class StoryError(Exception):
    """What the storyboard says of the song is not what the tool did with it."""


async def into(f, seconds, began):
    """Wait until `seconds` after `began`: the moment in a line when the hand does what the line says."""
    await f.pause(began + seconds - time.time())


# ------------------------------------------------------------------------------------------------- the terminal
async def typed(f, text, each=0.045):
    for ch in text:
        await f.js("t => type(t)", ch)
        await f.pause(each)


async def again(kept, began):
    """A run of the command that was kept: what it printed, a line at a time at the moment it printed it, then its
    exit code. For filming the scene again without running the command again (make.py --replay)."""
    for at, line in kept["lines"]:
        await asyncio.sleep(max(0.0, began + at - time.time()))
        yield line
    await asyncio.sleep(max(0.0, began + kept["seconds"] - time.time()))
    yield kept["code"] if kept.get("code") is not None else 0


async def command(f, c):
    """One command: a song in, a sheet out, and what the quality gates say. The command runs while it is filmed;
    the wait is shown sped up. What it printed and when is kept in the work folder (command.json), and with
    --replay the scene is filmed from that instead: the same lines at the same moments, and the files that run
    wrote. The command is typed as the README has it, with `uv run`; it is run with this checkout's code."""
    name, out_dir = os.path.basename(c.song_path), "sheet"  # named, so that the tool speaks of it without a path
    argv, kept_at = [name, "-o", out_dir], os.path.join(c.work, "command.json")
    if c.args.replay:
        wrote = all(os.path.exists(os.path.join(c.work, out_dir, n)) for n in ("report.html", "bass.pdf"))
        if not os.path.exists(kept_at) or not wrote:
            raise StoryError(f"--replay films the command from a run that was kept, and {c.work} holds none")
        with open(kept_at) as kf:
            kept = json.load(kf)
        if kept["argv"] != argv:
            raise StoryError(f"the run that was kept is of another command: {kept['argv']}")
    else:
        shutil.rmtree(os.path.join(c.work, out_dir), ignore_errors=True)
    await f.goto(c.stage + "/terminal.html")
    await f.js("prompt()")
    await f.roll()
    await f.pause(0.5)
    await f.say("c1", wait=False)
    await f.pause(0.7)
    await typed(f, f'uv run bass2sheet "{name}" -o {out_dir}')
    await f.pause(0.4)
    await f.js("enter()")
    began = time.time()
    run = again(kept, began) if c.args.replay else c.run("cli", argv, cwd=c.work)
    said, at, code, hurried = [], [], None, False

    async def hurry_when_said():  # the first seconds of the run at their own pace, while the line is finished
        nonlocal hurried
        await f.said(0.2)
        f.hurry(True)
        await f.say("c2", wait=False)
        hurried = True

    waiting = asyncio.ensure_future(hurry_when_said())
    async for line in run:
        if isinstance(line, int):
            code = line
        else:
            said.append(line)
            at.append(round(time.time() - began, 2))
            await f.js("t => print(t)", line)
    took = round(time.time() - began, 2)
    await waiting
    f.hurry(False, seconds=6.5)
    verdict = next(
        (
            re.search(r"verdict (\w+); (\d+) bars to check by ear \((\d+) marked", line)
            for line in said
            if "verdict" in line
        ),
        None,
    )
    want = c.song["command"]
    if code not in (0, 2) or not verdict:
        raise StoryError(f"the command did not write a sheet (exit {code}): {said[-3:]}")
    if verdict.group(1) != want["verdict"] or int(verdict.group(3)) != want["marked"]:
        raise StoryError(
            f"the narration says {want['verdict']} and {want['marked']} bars marked; the tool now says: {verdict.group(0)})"
        )
    if not c.args.replay:  # kept, so that the scene can be filmed again without the two minutes of the run
        when = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(began))
        with open(kept_at, "w") as kf:
            json.dump(
                dict(argv=argv, when=when, seconds=took, code=code, lines=[list(x) for x in zip(at, said)]),
                kf,
                indent=1,
            )
    await f.pause(0.8)
    await f.js("prompt()")
    await typed(f, f"ls {out_dir}")
    await f.pause(0.25)
    await f.js("enter()")
    listing = subprocess.run(
        ["ls", "-C", out_dir], cwd=c.work, capture_output=True, text=True, env=dict(os.environ, COLUMNS="100")
    ).stdout
    for line in listing.splitlines():
        await f.js("t => print(t)", line)
    await f.js("prompt()")
    await f.pause(1.8)
    # the report the command wrote, and the sheet, opened as they are
    await f.go("file://" + os.path.join(c.work, out_dir, "report.html"))
    t = time.time()
    await f.say("c3", wait=False)
    await f.move(*(await f.spot(".verdict", where=(0.3, 0.6))), seconds=0.9)
    await into(f, 2.6, t)
    await f.move(*(await f.spot(".kv div:last-child")), seconds=0.9)
    await f.said(0.1)
    await f.move(820, 430, seconds=0.5)
    await f.scroll(380, seconds=1.4)
    await f.pause(1.4)
    await f.go("file://" + os.path.join(c.work, out_dir, "bass.pdf"), settle=2.0)
    f.note("sheet/bass.pdf")
    await f.pause(0.5)
    f.look((440, 46, 900), seconds=1.2)  # the first page, in Chrome's own PDF viewer
    await f.pause(3.4)
    await f.cut()


# --------------------------------------------------------------------------------------------- the first page
DROP = """async ([name, url]) => { const blob = await (await fetch(url)).blob(), data = new DataTransfer();
  data.items.add(new File([blob], name, {type: blob.type || 'audio/mpeg'})); window.__dropped = data; return true; }"""
DRAG = """kind => window.dispatchEvent(new DragEvent(kind, {dataTransfer: window.__dropped, bubbles: true, cancelable: true}))"""


async def landing(f, c):
    """The copilot's first page: the song is dropped in, the steps of its preparation run (shown sped up), and
    "Fix the notes" opens it."""
    name = os.path.basename(c.song_path)
    await f.page.route("**/__demo/song", lambda route: route.fulfill(path=c.song_path))
    f.x, f.y = 1250, 300
    await f.goto(c.url + "/", "document.querySelector('#finders input') && !document.querySelector('#empty.hidden')")
    await f.roll()
    await f.pause(0.5)
    await f.say("l1")
    # a file dragged in from outside: the page is given the file itself and the events a real drag sends
    await f.js(DROP, [name, c.url + "/__demo/song"])
    await f.say("l2a", wait=False)
    await f.carry(name)
    await f.move(1120, 470, seconds=0.7)
    await f.js(DRAG, "dragenter")
    x, y = await f.spot("#drop")
    await f.move(x, y + 14, seconds=1.0)
    await f.js(DRAG, "dragover")
    await f.pause(0.6)
    await f.js(DRAG, "drop")
    await f.carry(None)
    await f.ready("document.querySelector('#jobs .job')")
    began = time.time()
    await f.pause(1.0)
    await f.move(1200, 330, seconds=0.8)
    await f.said(0.2)
    f.hurry(True)
    await f.say("l2b", wait=False)
    await f.ready(
        "document.querySelector('#jobs .job.done') || document.querySelector('#jobs .job.failed')", timeout=1200000
    )
    f.hurry(False, seconds=7.5)
    if await f.js("!!document.querySelector('#jobs .job.failed')"):
        raise StoryError(
            "the song could not be prepared: " + await f.js("document.querySelector('#jobs .job .error').textContent")
        )
    c.facts["prepared_in"] = round(time.time() - began, 1)
    await f.pause(1.6)
    c.song_id = await f.js("way.jobs[0].id")
    await f.press("#jobs .job a.btn.main")
    await f.ready(SONG_READY)
    await f.pause(1.2)
    await f.cut()


# ---------------------------------------------------------------------------------------------- the song page
NOTE = """([t, name]) => { const p = +Object.keys(S.names).find(k => S.names[k] === name);
  const n = S.notes.filter(n => Math.abs(n.start - t) < 0.06 && (n.pitch === p || (n.also || []).includes(p))).sort((a, b) => Math.abs(a.start - t) - Math.abs(b.start - t))[0];
  return n ? {id: n.id, pitch: S.names[n.pitch], also: (n.also || []).map(q => S.names[q]), start: n.start, end: n.end, doubt: !!n.doubt, state: n.state} : null; }"""
BOX = """([id, name]) => { const n = S.notes.find(n => n.id === id), r = $('roll').getBoundingClientRect();
  const b = boxes(n).find(b => name === null || S.names[b[0]] === name);
  return {x: r.left + b[1], y: r.top + b[2], w: b[3], h: row, cx: r.left + b[1] + b[3] / 2, cy: r.top + b[2] + row / 2}; }"""
PLACE = """([t, name]) => { const p = +Object.keys(S.names).find(k => S.names[k] === name), r = $('roll').getBoundingClientRect();
  return [r.left + xOf(t), r.top + yOf(p) + row / 2]; }"""
QUIET = "!A.playing"


def low(f, x, width):
    """The part of the page `width` wide around x, down to the page's lower edge (where the song page has its
    buttons for the selected note): what film.look() takes."""
    w, h = f.size
    return (round(min(max(0, x - width / 2), w - width)), round(h - width * h / w), width)


def high(f, x, width):
    """The same, from the page's upper edge down."""
    return (round(min(max(0, x - width / 2), f.size[0] - width)), 0, width)


async def until(f, check, args, wrong, want=True, seconds=20):
    """A fix goes to the server and comes back: wait until the page has it (`check` with `args` is true, or false
    for want=False), and stop the filming with `wrong` if it never does."""
    end = time.time() + seconds
    while True:
        got = await f.js(check, args)
        if bool(got) == want:
            return got
        if time.time() > end:
            raise StoryError(wrong)
        await asyncio.sleep(0.1)


async def find(f, where, what):
    """The note the storyboard speaks of, as the tool wrote it this time."""
    n = await f.js(NOTE, [where["at"], where["pitch"]])
    if not n:
        raise StoryError(
            f"the storyboard has {what}: {where['pitch']} at {where['at']} s. The tool did not write that this time."
        )
    return n


async def play_to(f, t):
    """Press Play, and Stop when the song has got to `t`."""
    x, y = await f.spot("#play")
    await f.click(x, y)
    await f.ready(f"A.playing && playhead >= {t - 0.14}")
    await f.click()
    await f.pause(0.25)


async def song(f, c):
    """The song's page: play, stop at what is wrong, fix it. Five real mistakes of the tool, each fixed another
    way, and the offer for the bars that repeat."""
    b = c.song["song"]
    await f.goto(c.url + "/song#" + c.song_id, SONG_READY)
    await f.pause(0.8)
    if await f.js("S.fixes"):
        raise StoryError("the song has been worked on already: this scene starts from the tool's own notes")
    extra, drag, late, guess = [await find(f, b[k], k) for k in ("extra", "drag", "late", "guess")]
    if not drag["doubt"]:
        raise StoryError("the narration calls the note that is dragged one the tool doubts (amber); it is not")
    await f.roll()

    # what is on the page
    t = time.time()
    await f.say("s1", wait=False)
    await f.pause(1.1)
    x, y = await f.js(PLACE, [0.42, "B2"])  # a dark bar of the picture: an overtone of the first B
    await f.move(x, y, seconds=0.9)
    await into(f, 3.5, t)
    box = await f.js(BOX, [(await f.js(NOTE, [0.78, "E2"]))["id"], None])
    await f.move(box["cx"], box["cy"], seconds=0.7)
    await f.said()
    await f.say("s2", wait=False)
    await f.pause(0.5)
    await f.move(*(await f.spot("#play")), seconds=0.9)
    await f.said(0.2)
    await play_to(f, b["listen_to"])

    # a note that nobody plays: delete it
    box = await f.js(BOX, [extra["id"], None])
    f.look(low(f, box["x"] + 300, 880))
    t = time.time()
    await f.say("s3", wait=False)
    await f.pause(0.3)
    await f.click(box["cx"], box["cy"])
    await into(f, 2.3, t)
    await f.press("#panel button[data-act=delete]")
    await f.said(0.3)
    await until(
        f, NOTE, [b["extra"]["at"], b["extra"]["pitch"]], "the note that was to be deleted is still there", want=False
    )

    # the note the tool doubts: drag it to where it sounds, and hear it
    t = time.time()
    await f.say("s4", wait=False)
    box = await f.js(BOX, [drag["id"], None])
    rows = await f.js(
        "([a, b]) => { const p = n => +Object.keys(S.names).find(k => S.names[k] === n); return p(b) - p(a); }",
        [b["drag"]["pitch"], b["drag"]["to"]],
    )
    await f.move(box["cx"], box["cy"])
    await into(f, 3.3, t)
    await f.drag(
        box["cx"], box["cy"], box["cx"], box["cy"] - rows * box["h"], seconds=0.8
    )  # the page plays it in its place
    await f.said(0.1)
    await f.pause(0.3)
    await f.ready(QUIET)
    await until(
        f,
        NOTE,
        [b["drag"]["at"], b["drag"]["to"]],
        "the dragged note did not land where the storyboard says it belongs",
    )
    f.look(None)
    await f.pause(0.4)

    # on, to the next thing that is wrong
    await play_to(f, b["listen_on_to"])
    box = await f.js(BOX, [late["id"], None])
    line = await f.js("bar => xOf(S.bars[bar - 1].time) + $('roll').getBoundingClientRect().left", b["late"]["bar"])
    f.look(low(f, box["x"] + 160, 880))
    t = time.time()
    await f.say("s5", wait=False)
    await f.pause(0.4)
    await f.move(box["x"] + 1.5, box["cy"])
    await into(f, 2.3, t)
    await f.drag(box["x"] + 1.5, box["cy"], line - 1.5, box["cy"], seconds=0.8)
    await f.said(0.2)
    await f.ready(QUIET)
    await until(
        f,
        NOTE,
        [await f.js("bar => S.bars[bar - 1].time", b["late"]["bar"]), b["late"]["pitch"]],
        "the note did not get to the bar line",
    )

    # a wrong pitch: the tool's guesses
    box = await f.js(BOX, [guess["id"], None])
    t = time.time()
    await f.say("s6", wait=False)
    await f.pause(0.3)
    await f.click(box["cx"], box["cy"])
    await f.ready("document.querySelector('#panel button.guess')")
    first = await f.js("S.names[guesses[0]]")
    if first != b["guess"]["to"]:
        raise StoryError(f"the narration takes the tool's first guess; it is {first}, and {b['guess']['to']} is right")
    await f.move(*(await f.spot("#panel button.guess[data-guess='0']")))
    await into(f, 3.3, t)
    await f.click()
    await f.ready("dock && dock.loaded && dock.kind === 'offer'")
    await f.said(0.1)
    await f.ready(QUIET)

    # the bars that repeat: the same fix is offered, a bar at a time
    offered = await f.js("dock.strips.map(s => s.bar + 1)")
    take, leave = b["offer"]["take"], b["offer"]["leave"]
    if sorted(offered) != sorted([take, leave]):
        raise StoryError(
            f"the narration says two more bars have the same mistake (bars {take} and {leave}); the tool offers {offered}"
        )
    k_take, k_leave = offered.index(take), offered.index(leave)
    f.look(low(f, 0, 940))
    await f.pause(0.6)
    t = time.time()
    await f.say("s7", wait=False)
    await f.pause(0.9)
    await f.move(*(await f.spot("#dockHead b")), seconds=0.9)
    await into(f, 4.3, t)
    await f.move(*(await f.spot(f"#dockInner button[data-s=take][data-k='{k_take}']")), seconds=0.9)
    await f.said()
    await f.say("s8", wait=False)
    await f.move(*(await f.spot(f"#dockInner button[data-s=hearWith][data-k='{k_take}']")))
    await f.said(0.1)
    await f.click()
    await f.pause(0.4)
    await f.ready(QUIET)
    await f.pause(0.3)
    await f.press(f"#dockInner button[data-s=take][data-k='{k_take}']")
    await f.pause(0.7)
    t = time.time()
    await f.say("s9", wait=False)
    await f.move(*(await f.spot(f"#dockInner button[data-s=leave][data-k='{k_leave}']")), seconds=1.0)
    await into(f, 2.6, t)
    await f.click()
    await f.said(0.3)
    await f.press("#dockHead button[data-d=close]")
    f.look(None)
    await f.pause(0.3)

    # nothing more is wrong up to the end of the fourth bar: say so
    await play_to(f, b["checked_at"])
    f.look(high(f, 0, 940))
    t = time.time()
    await f.say("s10", wait=False)
    await f.move(*(await f.spot("#upto")), seconds=1.0)
    await into(f, 2.0, t)
    await f.click()
    await f.said(0.4)
    f.look(None)
    await f.pause(0.3)

    # further on: two notes written together. Slower, to hear what is played
    t = time.time()
    await f.say("s11", wait=False)
    spot = await f.js(
        "t => { const r = $('map').getBoundingClientRect(); return [r.left + mapX(t), r.top + r.height / 2]; }",
        b["pair"]["at"] + 0.4,
    )
    await f.click(*spot)
    await f.pause(0.5)
    pair = await find(f, b["pair"], "pair")
    if pair["also"] != [b["pair"]["with"]]:
        raise StoryError(
            f"the narration says the tool wrote two notes together at {b['pair']['at']} s; it wrote {pair['pitch']} {pair['also']}"
        )
    await into(f, 2.6, t)
    await f.choose("#speed", str(b["speed"]))
    f.hurry(True)
    await f.ready(f"A.speed === {b['speed'] / 100} && !$('speed').disabled", timeout=300000)
    f.hurry(False, seconds=1.8)
    await f.said(0.2)
    x, y = await f.js(PLACE, [b["pair"]["listen"][0], "E3"])
    await f.click(x, y)  # the playhead: just before the bar
    await f.pause(0.3)
    await play_to(f, b["pair"]["listen"][1])
    under, up = await f.js(BOX, [pair["id"], b["pair"]["pitch"]]), await f.js(BOX, [pair["id"], b["pair"]["with"]])
    f.look(low(f, under["cx"] + 60, 880))
    t = time.time()
    await f.say("s12", wait=False)
    await f.pause(0.7)
    await f.click(up["cx"], up["cy"])
    await into(f, 2.7, t)
    await f.press("#panel button.picked")  # "Delete" for the one that was clicked
    await f.pause(0.3)
    await f.said(0.1)
    await f.ready(QUIET)
    await until(
        f,
        "([t, name, gone]) => { const p = +Object.keys(S.names).find(k => S.names[k] === gone); return S.notes.some(n => Math.abs(n.start - t) < 0.06 && (n.also || []).includes(p)); }",
        [b["pair"]["at"], b["pair"]["pitch"], b["pair"]["with"]],
        "one note of the pair was to go; both are still there",
        want=False,
    )
    await find(f, b["pair"], "the note that is left of the pair")
    x, y = await f.js(PLACE, [b["pair"]["sounds_at"], b["pair"]["with"]])
    await f.click(x, y, twice=True)  # a double-click adds a note there; the note sounding there ends where it starts
    await f.pause(0.4)
    await f.ready(QUIET)
    await until(
        f,
        NOTE,
        [b["pair"]["sounds_at"], b["pair"]["with"]],
        "the note was not added where the storyboard says it sounds",
    )
    await f.pause(0.4)
    f.look(None)
    c.facts["fixes"] = await f.js("S.fixes")
    c.facts["worked"] = await f.js("S.worked")
    await f.pause(0.9)
    await f.cut()


# ------------------------------------------------------------------------------------------------- the sheet
SCORE_OPEN = "document.querySelector('#scorePanel') && document.querySelector('#scorePanel').style.display === 'block'"
PAGES_DRAWN = "document.querySelectorAll('#pages img').length > 0 && [...document.querySelectorAll('#pages img')].every(i => i.complete && i.naturalWidth)"


async def score_frame(f):
    """The score, shown in a frame over the song's page: wait until its pages are drawn."""
    await f.ready(SCORE_OPEN, timeout=180000)
    for _ in range(200):
        frame = next((fr for fr in f.page.frames if "/score" in fr.url), None)
        if frame:
            break
        await f.pause(0.05)
    await frame.wait_for_function(PAGES_DRAWN, timeout=60000)
    return frame


async def sheet(f, c):
    """ "Make the sheet": the score in the page; one more fix, the sheet made again, and both versions kept."""
    again = c.song["sheet"]["again"]
    await f.goto(c.url + "/song#" + c.song_id, SONG_READY)
    await f.pause(0.8)
    await f.roll()
    t = time.time()
    await f.say("h1", wait=False)
    await f.move(*(await f.spot("#sheet")), seconds=1.0)
    await into(f, 1.3, t)
    await f.click()
    frame = await score_frame(f)
    c.facts["first_sheet"] = (await frame.evaluate("document.querySelector('#made').textContent")).strip()
    f.look(high(f, 0, 1040), seconds=1.0)
    await f.move(*(await f.spot("#made", frame)), seconds=1.0)
    await f.said(0.5)
    await f.pause(1.0)
    f.look(None, seconds=0.7)
    await f.press("#close", frame)
    await f.pause(0.3)
    # one more fix: the bar that was left
    t = time.time()
    await f.say("h2", wait=False)
    spot = await f.js(
        "t => { const r = $('map').getBoundingClientRect(); return [r.left + mapX(t), r.top + r.height / 2]; }",
        again["at"] + 0.3,
    )
    await f.click(*spot)
    await f.pause(0.4)
    note = await find(f, again, "the note fixed before the second sheet")
    box = await f.js(BOX, [note["id"], None])
    await f.click(box["cx"], box["cy"])
    await f.ready("document.querySelector('#panel button.guess')")
    first = await f.js("S.names[guesses[0]]")
    if first != again["to"]:
        raise StoryError(f"the tool's first guess for the note at {again['at']} s is {first}; {again['to']} is right")
    await f.press("#panel button.guess[data-guess='0']")
    await until(f, NOTE, [again["at"], again["to"]], "the guess was taken and the note did not change")
    await f.said(0.1)
    await f.pause(0.3)
    await f.ready(QUIET)
    await f.press("#sheet")
    await f.pause(0.3)
    await f.ready("!$('sheet').disabled", timeout=180000)
    frame = await score_frame(f)
    await frame.wait_for_function(
        "!document.querySelector('#version').classList.contains('hidden') && document.querySelector('#version').options.length > 1"
    )
    c.facts["versions"] = await frame.evaluate(
        "[...document.querySelector('#version').options].map(o => o.textContent)"
    )
    f.look(high(f, 0, 1040), seconds=1.0)
    await f.move(*(await f.spot("#version", frame)), seconds=0.9)
    await f.pause(1.4)
    await f.choose("#version", "1", frame)  # the first one is still there
    await frame.wait_for_function("document.querySelector('#made').textContent.includes('earlier')")
    await f.pause(1.8)
    f.look(None, seconds=0.7)
    await f.pause(0.7)
    await f.cut()


# ----------------------------------------------------------------------------------------------- playing along
async def play(f, c):
    """The play-along page, the size of a phone held sideways: the record without the bass, the line being
    played and the next one."""
    await f.goto(c.stage + "/phone.html")
    await f.js("u => show(u)", f"{c.url}/play?song={c.song_id}")
    for _ in range(200):
        frame = next((fr for fr in f.page.frames if "/play" in fr.url), None)
        if frame:
            break
        await f.pause(0.05)
    await frame.wait_for_function(
        "typeof S !== 'undefined' && S && loaded && loaded.ready && document.querySelector('.line.cur img').complete",
        timeout=300000,
    )
    await f.pause(0.8)
    f.x, f.y = 1380, 350
    await f.roll()
    await f.say("p1")
    await f.say("p2", wait=False)
    await f.pause(0.4)
    await f.press("#play", frame)
    await f.move(1150, 150, seconds=1.0)
    await f.said(0.2)
    await frame.wait_for_function(f"now() >= {c.song['play']['until']}", timeout=120000)
    await f.press("#play", frame)
    await f.pause(0.8)
    await f.cut()


# ------------------------------------------------------------------------------------------- the test songs
async def tests(f, c):
    """How the tool is checked: the test songs, their scores, and whose songs count."""
    await f.goto(c.tests_url + "/tests", "document.querySelectorAll('.song').length > 0")
    rated = await f.js("T.rated")
    if rated:
        raise StoryError(
            "the test songs have answers already: this scene starts with none (delete ratings.json in that working directory)"
        )
    await f.pause(0.5)
    f.x, f.y = 1000, 400
    await f.roll()
    t = time.time()
    await f.say("t1a", wait=False)
    await f.pause(1.0)
    await f.move(*(await f.spot("#summary table", where=(0.8, 0.9))), seconds=1.0)
    await into(f, 3.4, t)
    await f.move(*(await f.spot(".song .scores b")), seconds=0.9)
    await f.said()
    t = time.time()
    await f.say("t1b", wait=False)
    songs = await f.js("[...document.querySelectorAll('.song')].slice(0, 2).map(s => s.dataset.song)")
    await f.press(f".song[data-song='{songs[0]}'] button.yes")
    await f.pause(0.5)
    await f.press(f".song[data-song='{songs[1]}'] button.no")
    await f.pause(0.3)
    x, y = await f.spot("#summary table", where=(0, 0))
    f.look((max(0, x - 50), max(0, y - 22), 900), seconds=0.9)  # the table, and room beside it for the command
    await f.move(*(await f.spot("#summary table", where=(0.85, 0.2))), seconds=0.9)
    await f.said(0.4)
    f.note("uv run bass2sheet-copilot reference SONG transcription.mid", right=True)
    await f.say("t2")
    f.note(None)
    await f.pause(0.3)
    await f.cut()
    for k, song in enumerate(songs):  # the answers are taken back: the test songs are left as they were found
        await f.press(f".song[data-song='{song}'] button.{'yes' if k == 0 else 'no'}")
        await f.pause(0.4)


# ------------------------------------------------------------------------------------- the first and last pages
async def opening(f, c):
    """The problem in a sentence, and the result first: the sheet the film ends up with."""
    score = await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(f"{c.url}/api/score?song={c.song_id}")))
    if not score.get("pages"):
        raise StoryError("the sheet has not been made yet: the opening shows the one made in the scene `sheet`")
    f.pointer = False  # the pointer has no part in this
    await f.goto(c.stage + "/card.html")
    await f.js(
        "c => show(c)", dict(kind="title", title="bass2sheet", lead="From a song to a bass part you can read and play.")
    )
    await f.roll()
    f.music(f"{c.url}/data/{c.song_id}/audio/record.flac", at=c.song["music"]["opening"])
    await f.pause(0.7)  # the film does not begin on a word
    await f.say("o1", after=0.5)
    await f.go(c.stage + "/paper.html")
    await f.js("u => show(u, '')", [c.url + p for p in score["pages"]])
    x, y = await f.spot("#pages img", where=(0, 0))
    f.look((x - 12, y + 110, 1104), seconds=0.01)
    await f.pause(0.3)
    await f.js("glide(250, 9)")
    await f.say("o2", after=1.4)
    f.music(None)
    await f.cut()


async def ending(f, c):
    """What the fixes are kept for, and how to get the tool: the report as the tool prints it, then the card."""
    f.pointer = False
    await f.goto(c.stage + "/terminal.html")
    await f.js("prompt()")
    await f.roll()
    await f.say("e1", wait=False)
    await f.pause(0.5)
    await typed(f, "uv run bass2sheet-copilot report --dir scores")
    await f.pause(0.3)
    await f.js("enter()")
    said = []
    async for line in c.run("copilot", ["report", "--dir", "scores"], cwd=c.work):
        if not isinstance(line, int):
            said.append(line)
            await f.js("t => print(t)", line)
    if not any("wrong pitch" in line for line in said):
        raise StoryError(f"the report does not list the fixes: {said[:6]}")
    await f.js("prompt()")
    await f.said(0.8)
    await f.go(c.stage + "/card.html")
    credit = c.song["credit"]
    tests = ["Test songs on screen: BabySlakh, from Slakh2100 (CC BY 4.0)."] if c.tests_url else []
    await f.js(
        "c => show(c)",
        dict(
            kind="get",
            heading="Try it",
            steps=[
                ["./bootstrap", "sets everything up, once"],
                ["uv run bass2sheet-copilot serve --dir scores", "then open the page it names"],
            ],
            boxes=[
                dict(heading="The music", lines=credit),
                dict(
                    heading="The tool",
                    lines=["bass2sheet is open source: MIT licence.", *tests, "The narration is a synthetic voice."],
                ),
            ],
        ),
    )
    f.music(f"{c.url}/data/{c.song_id}/audio/record.flac", at=c.song["music"]["ending"])
    await f.say("e2", after=0.5)
    await f.pause(3.2)
    f.music(None)
    await f.cut()
