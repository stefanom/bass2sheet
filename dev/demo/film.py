"""Filming a page: Chrome without a window, driven by Playwright, every frame kept with its time.

A scene is a folder: the frames Chrome's own screencast sends (JPEG, each with the moment it was drawn), `scene.json`
(the frames, and the marks the scene's script left: a line of narration begins here, look closer at this part, this
wait is shown sped up) and `events.jsonl` (what the page played and when: hooks.js). Nothing is drawn over the
pictures here; edit.py does that, and sound.py makes the soundtrack from the log.

The pointer is moved the way a hand moves it: along a path, in the time a hand takes. The page sees ordinary mouse
events; hooks.js draws the pointer the recording would otherwise lack.
"""

import asyncio
import base64
import hashlib
import json
import math
import os
import time
import urllib.request

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = (1440, 709)  # the page's size in its own pixels. The film is 1920 wide, so everything is a third larger; under
# the picture (945 of the film's 1080 lines) is a band for the captions, which therefore never cover anything
SCALE = 2  # frames are taken at twice the page's size, so a closer look is still sharp


def ease(u):
    """0..1 -> 0..1, starting and ending gently."""
    return u * u * u * (u * (6 * u - 15) + 10)


class Film:
    def __init__(self, folder, lines=None, size=SIZE, scale=SCALE, say=print):
        self.folder, self.size, self.scale, self.tell = folder, size, scale, say
        self.lines = lines or {}  # the narration: id -> {seconds: how long the spoken line lasts}
        self.frames, self.marks, self.events = [], [], []
        self.x, self.y = size[0] * 0.6, size[1] * 0.55  # where the pointer is
        self.pointer = True  # False: the scene has no use for the pointer (a card), and it is not shown
        self.rolling, self.started, self.ended = False, None, None
        self.spoken_until = 0.0

    # ---------------------------------------------------------------------------------------------- the browser
    async def __aenter__(self):
        os.makedirs(os.path.join(self.folder, "frames"), exist_ok=True)
        self._pw = await async_playwright().start()
        # the real scale factor has to be set too: the screencast is taken at the window's, not at the emulated one
        # no sound leaves the machine: the film's sound is made from the page's own account of what it played (sound.py)
        args = [
            "--autoplay-policy=no-user-gesture-required",
            f"--force-device-scale-factor={self.scale}",
            "--hide-scrollbars",
            "--disable-audio-output",
        ]
        self.browser = await self._pw.chromium.launch(channel="chrome", headless=True, args=args)
        w, h = self.size
        self.context = await self.browser.new_context(viewport=dict(width=w, height=h), device_scale_factor=self.scale)
        with open(os.path.join(HERE, "hooks.js")) as f:
            await self.context.add_init_script(f.read())
        await self.context.expose_binding("__demoSend", lambda source, batch: self._heard(batch))
        self.page = await self.context.new_page()
        self.page.on("pageerror", lambda e: self.tell(f"  page error: {e}"))
        self.cdp = await self.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", lambda ev: asyncio.ensure_future(self._frame(ev)))
        return self

    async def __aexit__(self, *exc):
        if self.rolling:
            await self.cut()
        await self.browser.close()
        await self._pw.stop()

    def _heard(self, batch):
        self.events += json.loads(batch)

    async def _frame(self, ev):
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:
            return
        if not self.rolling:
            return
        name = f"{len(self.frames):06d}.jpg"
        with open(os.path.join(self.folder, "frames", name), "wb") as f:
            f.write(base64.b64decode(ev["data"]))
        self.frames.append([ev["metadata"]["timestamp"], name])

    async def goto(self, url, ready=None, timeout=60000):
        await self.flush()
        await self.page.goto(url)
        if ready:
            await self.page.wait_for_function(ready, timeout=timeout)
        if self.pointer:
            await self.page.mouse.move(self.x, self.y)

    async def ready(self, js, timeout=60000):
        await self.page.wait_for_function(js, timeout=timeout)

    async def js(self, code, arg=None, frame=None):
        return await (frame or self.page).evaluate(code, arg)

    async def flush(self):
        """Take what the page has logged so far (it also sends it by itself, a few times a second)."""
        for frame in self.page.frames:
            try:
                await frame.evaluate("window.__demo && window.__demo.flush()")
            except Exception:
                pass

    # ------------------------------------------------------------------------------------------------ recording
    async def roll(self):
        """Start keeping frames."""
        self.rolling = True
        await self.cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": 92, "everyNthFrame": 1, "maxWidth": 8000, "maxHeight": 8000},
        )
        if self.pointer:
            await self.page.mouse.move(self.x, self.y)
        else:
            await self.page.evaluate("window.__demo && window.__demo.hide()")
        # the screencast's first frame is whatever was last drawn; the scene starts once the page as it is now has been
        await self.page.evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
        await self.pause(0.3)
        self.started = time.time()

    async def cut(self, sounds=True):
        """Stop, and write down what was filmed. The sounds the page played are fetched again from where the page
        got them and kept beside the frames: the soundtrack is made from those files."""
        await self.pause(0.15)
        self.ended = time.time()
        await self.flush()
        await asyncio.sleep(0.3)
        self.rolling = False
        try:
            await self.cdp.send("Page.stopScreencast")
        except Exception:
            pass
        files = {}
        if sounds:
            os.makedirs(os.path.join(self.folder, "sounds"), exist_ok=True)
            played = {e.get("url") for e in self.events if e.get("ev") in ("buffer", "media") and e.get("url")}
            for url in sorted(played | {m["url"] for m in self.marks if m["kind"] == "music" and m.get("url")}):
                if "/play/awake" in url:  # the silent film that keeps a phone's screen on
                    continue
                name = hashlib.sha1(url.encode()).hexdigest()[:12] + os.path.splitext(url.split("?")[0])[1]
                with urllib.request.urlopen(url) as r, open(os.path.join(self.folder, "sounds", name), "wb") as f:
                    f.write(r.read())
                files[url] = name
        with open(os.path.join(self.folder, "events.jsonl"), "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in self.events)
        scene = dict(
            start=self.started,
            end=self.ended,
            size=self.size,
            scale=self.scale,
            frames=self.frames,
            marks=self.marks,
            sounds=files,
        )
        with open(os.path.join(self.folder, "scene.json"), "w") as f:
            json.dump(scene, f)
        self.tell(
            f"  filmed {self.ended - self.started:.1f} s, {len(self.frames)} frames, {len(self.events)} sound events"
        )

    def mark(self, kind, **what):
        self.marks.append(dict(what, kind=kind, t=time.time()))

    async def pause(self, seconds):
        await asyncio.sleep(max(0.0, seconds))

    # ------------------------------------------------------------------------------------- narration and looks
    async def say(self, line, wait=True, after=0.3):
        """A line of the narration begins now. With `wait`, the scene goes on when it has been said; without, it
        is said over what happens next (and `said()` waits for it later). Lines never overlap."""
        await self.said(0.25)
        seconds = self.lines.get(line, {}).get("seconds", 0.0)
        self.mark("say", line=line, seconds=seconds)
        self.spoken_until = time.time() + seconds
        if wait:
            await self.pause(seconds + after)

    async def said(self, after=0.3):
        """Wait until the line being said is over."""
        left = self.spoken_until - time.time()
        if left > 0:
            await self.pause(left + after)

    def look(self, at=None, seconds=0.9):
        """Look closer: `at` is (x, y, width) in the page's pixels, the part of the page that fills the picture
        from now on (its height follows from the picture's shape); None is the whole page again."""
        self.mark("look", at=at, seconds=seconds)

    def hurry(self, on, seconds=None):
        """From now on the film runs faster than the clock (a wait), until hurry(False, seconds): the stretch is
        shown in about `seconds`, or as long as the lines said during it take. The picture says how much faster."""
        self.mark("hurry", on=bool(on), seconds=seconds)

    async def go(self, url, ready=None, settle=0.5):
        """Another page, without showing the blank moment between the two: the film cuts from one to the other."""
        self.mark("skip", on=True)
        try:
            await self.goto(url, ready)
        finally:
            await self.pause(settle)
            self.mark("skip", on=False)

    def music(self, url, at=0.0):
        """Music under the picture from now on, where the page itself plays none (the title, the last card): the
        file at `url`, from `at` seconds into it; None ends it. It fades in and out."""
        self.mark("music", url=url, at=at)

    def note(self, text, right=False):
        """A few words shown in a corner of the picture from now on (None: no more): top left, or top right."""
        self.mark("note", text=text, right=right)

    # ----------------------------------------------------------------------------------------------- the pointer
    async def move(self, x, y, seconds=None):
        """Move the pointer to (x, y) in the page's pixels, along a slightly curved path."""
        x0, y0 = self.x, self.y
        far = math.hypot(x - x0, y - y0)
        if far < 0.5:
            return
        seconds = seconds if seconds is not None else min(1.1, 0.28 + far / 1500)
        bend = min(40.0, far * 0.08)
        nx, ny = -(y - y0) / far, (x - x0) / far
        began, last = time.time(), None
        while True:
            u = min(1.0, (time.time() - began) / seconds)
            e, arc = ease(u), math.sin(math.pi * ease(u)) * bend
            px, py = x0 + (x - x0) * e + nx * arc, y0 + (y - y0) * e + ny * arc
            if (round(px, 1), round(py, 1)) != last:
                await self.page.mouse.move(px, py)
                last = (round(px, 1), round(py, 1))
            self.x, self.y = px, py
            if u >= 1.0:
                break
            await asyncio.sleep(1 / 75)
        self.x, self.y = x, y

    async def spot(self, selector, frame=None, where=(0.5, 0.5)):
        """Where an element is on the page: (x, y) of its middle, or of the fraction `where` of its box. In a
        frame, the frame's place and its size on the page are counted in (a frame may be drawn larger or smaller)."""
        target = frame or self.page
        box = await target.locator(selector).first.evaluate(
            "el => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }"
        )
        x, y = box[0] + box[2] * where[0], box[1] + box[3] * where[1]
        return await self._outside(frame, x, y)

    async def _frame_box(self, frame):
        """A frame's place on the page and how much larger it is drawn than its own pixels."""
        if frame is None or frame == self.page.main_frame:
            return 0.0, 0.0, 1.0
        el = await frame.frame_element()
        return tuple(
            await el.evaluate(
                "el => { const r = el.getBoundingClientRect(); return [r.left + el.clientLeft, r.top + el.clientTop, r.width / el.offsetWidth]; }"
            )
        )

    async def _outside(self, frame, x, y):
        """A place in a frame's own pixels, on the page."""
        left, top, k = await self._frame_box(frame)
        return left + x * k, top + y * k

    async def click(self, x=None, y=None, hold=0.09, seconds=None, twice=False):
        if x is not None:
            await self.move(x, y, seconds)
            await self.pause(0.12)
        for _ in range(2 if twice else 1):
            await self.page.mouse.down(click_count=2 if twice and _ else 1)
            await self.pause(hold)
            await self.page.mouse.up(click_count=2 if twice and _ else 1)
            if twice:
                await self.pause(0.07)

    async def press(self, selector, frame=None, where=(0.5, 0.5), seconds=None, hold=0.09):
        """Move to an element and click it."""
        x, y = await self.spot(selector, frame, where)
        await self.click(x, y, hold, seconds)

    async def drag(self, x, y, to_x, to_y, seconds=0.9, hold=0.25):
        """Press at (x, y), move to (to_x, to_y), let go."""
        await self.move(x, y)
        await self.pause(0.15)
        await self.page.mouse.down()
        await self.pause(hold)
        await self.move(to_x, to_y, seconds)
        await self.pause(hold)
        await self.page.mouse.up()

    async def choose(self, selector, value, frame=None):
        """Choose from a list. The list itself is drawn by the system and not by the page, so a recording without
        a window cannot show it or click in it: the pointer goes there, a click is shown, and the choice is made."""
        x, y = await self.spot(selector, frame)
        await self.move(x, y)
        await self.pause(0.12)
        await (frame or self.page).evaluate("([x, y]) => window.__demo.click(x, y)", await self._inside(frame, x, y))
        await self.pause(0.35)
        await (frame or self.page).locator(selector).select_option(value)

    async def _inside(self, frame, x, y):
        """A place on the page, in the pixels of a frame within it."""
        left, top, k = await self._frame_box(frame)
        return [(x - left) / k, (y - top) / k]

    async def scroll(self, dy, seconds=0.8):
        """Turn the wheel: the page under the pointer moves by `dy` of its pixels, gently."""
        began, done = time.time(), 0.0
        while True:
            u = min(1.0, (time.time() - began) / seconds)
            to = dy * ease(u)
            if abs(to - done) >= 1:
                await self.page.mouse.wheel(0, to - done)
                done = to
            if u >= 1.0:
                break
            await asyncio.sleep(1 / 60)

    async def carry(self, text):
        """A label that rides with the pointer (a file being dragged), or None."""
        await self.page.evaluate("t => window.__demo.carry(t)", text)
