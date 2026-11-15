"""Putting the film together from the filmed scenes: the picture, the words on it, the sound.

Each scene was filmed at the clock's own pace (film.py). Here its time becomes the film's:

- a wait that was marked `hurry` is shown in a few seconds, with a sign that says how much faster it runs;
- the blank moment between two pages (`skip`) is cut out;
- a line of narration starts where the scene said it does, and never before the line before it is over: if a
  recording is longer than the scene allowed for (another voice, say), the picture holds still for the difference.

The picture is the screencast's frames, cropped where the scene looks closer (`look`), with the caption of the
line being said at the bottom. The sound is what the pages played (sound.py), cut to the same times, lowered
while the narrator speaks; the narration; and, where a scene asks for it, music under a card.
"""

import bisect
import json
import math
import os
import subprocess

import numpy as np
import sound
import soundfile as sf
from PIL import Image

W, H, FPS = 1920, 1080, 30
PICTURE = 945  # the film's upper lines are the page; the band under them holds the captions
BAND = (31, 35, 40)  # the pages' ink
GAP = 0.28  # between two lines of narration, at least
RATE = sound.RATE
VOICE = 10 ** (-21 / 20)  # the narration's level (RMS while speaking)
MUSIC, UNDER = 0.36, 0.12  # what the pages play: its level on its own, and how far it goes down under the voice
BED = 0.13  # music under a card
LOUD = -16.0  # the whole film's loudness, in LUFS


def smooth(u):
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------------------------------- a scene's time
class Scene:
    """A filmed scene and its place in the film: which moment of the filming is shown when."""

    def __init__(self, folder, lines):
        self.folder, self.name = folder, os.path.basename(folder)
        with open(os.path.join(folder, "scene.json")) as f:
            s = json.load(f)
        self.start, self.end, self.size, self.scale = s["start"], s["end"], s["size"], s["scale"]
        self.frames, self.marks, self.sounds = s["frames"], sorted(s["marks"], key=lambda m: m["t"]), s["sounds"]
        self.frame_times = [f[0] for f in self.frames]
        self.warnings = []
        self._lay(lines)

    def _lay(self, lines):
        # the stretches that are not shown at the clock's pace: [from, to, seconds in the film], in order
        special, open_ = [], {}
        for m in self.marks:
            if m["kind"] in ("skip", "hurry"):
                if m["on"]:
                    open_[m["kind"]] = m["t"]
                elif m["kind"] in open_:
                    special.append(
                        dict(a=open_.pop(m["kind"]), b=m["t"], kind=m["kind"], seconds=m.get("seconds") or 0.0)
                    )
        special.sort(key=lambda x: x["a"])
        says = [
            dict(m, seconds=lines[m["line"]]["seconds"])
            for m in self.marks
            if m["kind"] == "say" and m["line"] in lines
        ]
        for x in special:  # a hurried stretch lasts as long as the lines said during it need
            x["lines"] = [s for s in says if x["a"] <= s["t"] <= x["b"]] if x["kind"] == "hurry" else []
            if x["kind"] == "hurry":
                x["seconds"] = max(x["seconds"], sum(s["seconds"] + GAP for s in x["lines"]) + 0.3)
        pieces, at = [], self.start  # [real from, real to, seconds in the film]
        for x in special:
            if x["a"] > at:
                pieces.append([at, x["a"], x["a"] - at])
            pieces.append([x["a"], x["b"], x["seconds"] if x["kind"] == "hurry" else 0.0])
            at = x["b"]
        if self.end > at:
            pieces.append([at, self.end, self.end - at])
        self.pieces = pieces
        # when each line is said; a hold wherever the line before it is not over yet
        said, last_end, holds = [], -1e9, []
        for s in says:
            x = next((x for x in special if x["kind"] == "hurry" and s in x["lines"]), None)
            if x:  # in a hurried stretch: one after the other from its start
                k = x["lines"].index(s)
                base = self.film(x["a"]) + 0.15 + sum(p["seconds"] + GAP for p in x["lines"][:k])
                where = x["a"]
            else:
                base, where = self.film(s["t"]), s["t"]
            base += sum(h[1] for h in holds if h[0] <= where)
            if base < last_end + GAP:
                hold = last_end + GAP - base
                holds.append((where, hold))
                base += hold
            said.append(dict(line=s["line"], at=base, seconds=s["seconds"]))
            last_end = base + s["seconds"]
        if said and last_end + 0.3 > self.film(self.end) + sum(h[1] for h in holds):
            holds.append((self.end, last_end + 0.3 - self.film(self.end) - sum(h[1] for h in holds)))
        for where, hold in holds:  # a hold: no time passes in the scene, `hold` seconds pass in the film
            k = next(
                (i for i, p in enumerate(self.pieces) if p[0] <= where <= p[1] and p[1] > p[0]), len(self.pieces) - 1
            )
            p = self.pieces[k]
            share = (where - p[0]) / (p[1] - p[0]) if p[1] > p[0] else 1.0
            self.pieces[k : k + 1] = [
                [p[0], where, p[2] * share],
                [where, where, hold],
                [where, p[1], p[2] * (1 - share)],
            ]
            if hold > 0.05:
                self.warnings.append(
                    f"{self.name}: the picture holds {hold:.2f} s at {where - self.start:.1f} s for the narration"
                )
        self.said = said
        self.length = sum(p[2] for p in self.pieces)
        self.hurried = [
            dict(a=self.film(x["a"]), b=self.film(x["b"]), times=(x["b"] - x["a"]) / max(x["seconds"], 1e-6))
            for x in special
            if x["kind"] == "hurry"
        ]
        self.cuts = [self.film(x["b"]) for x in special if x["kind"] == "skip"]

    def film(self, real):
        """A moment of the filming -> seconds into the scene as the film shows it."""
        t = 0.0
        for a, b, seconds in self.pieces:
            if real >= b:
                t += seconds
            elif real > a:
                return t + seconds * (real - a) / (b - a)
            else:
                break
        return t

    def real(self, t):
        """Seconds into the scene as the film shows it -> the moment of the filming that is shown."""
        for a, b, seconds in self.pieces:
            if t < seconds:
                return a + (b - a) * (t / seconds) if seconds > 0 else a
            t -= seconds
        return self.end

    def camera(self, real):
        """The part of the page that fills the picture at a moment: (x, y, width) in the page's pixels."""
        full = (0.0, 0.0, float(self.size[0]))
        cam = full
        for m in self.marks:
            if m["kind"] != "look" or m["t"] > real:
                continue
            to = tuple(float(v) for v in m["at"]) if m["at"] else full
            took = max(m["seconds"], 1e-3)
            u = smooth(min(1.0, (real - m["t"]) / took))
            cam = tuple(c + (v - c) * u for c, v in zip(cam, to))
        w = min(cam[2], full[2])  # never past the page's edges
        h = w * self.size[1] / self.size[0]
        return (min(max(0.0, cam[0]), self.size[0] - w), min(max(0.0, cam[1]), self.size[1] - h), w)

    def note(self, real):
        """The few words shown in a corner at a moment, and which corner: (text, "left" or "sign")."""
        text, side = None, "left"
        for m in self.marks:
            if m["kind"] == "note" and m["t"] <= real:
                text, side = m["text"], "sign" if m.get("right") else "left"
        return text, side

    def frame(self, real):
        return self.frames[max(0, bisect.bisect_right(self.frame_times, real) - 1)][1]


# ------------------------------------------------------------------------------------------- words as pictures
async def draw_words(items, folder):
    """Words as pictures with nothing behind them, drawn by the browser in the pages' own type (stage/words.html):
    `items` are (kind, text), kind "caption" (a line of the narration, at the bottom), "sign" (top right) or
    "left" (top left). Returns {(kind, text): (picture, left, top)}."""
    from playwright.async_api import async_playwright

    os.makedirs(folder, exist_ok=True)
    out = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=True, args=["--disable-audio-output"])
        page = await browser.new_page(viewport=dict(width=W, height=H))
        await page.goto("file://" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage", "words.html"))
        for k, (kind, text) in enumerate(items):
            await page.evaluate(
                "([kind, text]) => kind === 'caption' ? caption(text) : sign(text, kind === 'left')", [kind, text]
            )
            path = os.path.join(folder, f"{k:03d}.png")
            await page.screenshot(path=path, omit_background=True)
            im = Image.open(path).convert("RGBA")
            box = im.getbbox()
            out[(kind, text)] = (im.crop(box), box[0], box[1])
        await browser.close()
    return out


def faster(times):
    return f"<b>▶▶</b>{round(times)}× faster"


# ------------------------------------------------------------------------------------------------ the picture
def render(scenes, words, lines, out_path, threads=None, fade=8, log=print):
    """Every frame of the film, in order, to ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
    ]
    # with `threads` the encoder leaves one of them to this program, which draws the pictures on one
    cmd += (["-filter_threads", "1", "-threads", str(max(1, threads - 1))] if threads else []) + [out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total, last, last_key, blend_from, blend_left = 0, None, None, None, 0
    for scene in scenes:
        n = int(round(scene.length * FPS))
        opened, opened_name = None, None
        if last is not None:  # one scene dissolves into the next
            blend_from, blend_left = last, fade
        cuts = sorted(int(c * FPS) for c in scene.cuts)
        for k in range(n):
            t = k / FPS
            real = scene.real(t)
            name = scene.frame(real)
            cam = scene.camera(real)
            said = next((s for s in scene.said if s["at"] <= t < s["at"] + s["seconds"] + 0.2), None)
            fast = next((h for h in scene.hurried if h["a"] <= t < h["b"] and h["times"] >= 1.5), None)
            note, note_side = scene.note(real)
            key = (
                scene.name,
                name,
                tuple(round(v, 2) for v in cam),
                said["line"] if said else None,
                round(fast["times"]) if fast else None,
                note,
            )
            if k in cuts and last is not None:  # a cut inside a scene (another page): dissolve too
                blend_from, blend_left = last, fade
            if key != last_key or blend_left:
                if name != opened_name:
                    opened, opened_name = Image.open(os.path.join(scene.folder, "frames", name)).convert("RGB"), name
                s = opened.width / scene.size[0]
                x, y, w = cam
                h = w * scene.size[1] / scene.size[0]
                im = Image.new("RGB", (W, H), BAND)
                im.paste(
                    opened.resize(
                        (W, PICTURE),
                        Image.LANCZOS,
                        box=(x * s, y * s, min(opened.width, (x + w) * s), min(opened.height, (y + h) * s)),
                    ),
                    (0, 0),
                )
                for what in (
                    ([("caption", lines[said["line"]]["text"])] if said else [])
                    + ([("sign", faster(fast["times"]))] if fast else [])
                    + ([(note_side, note)] if note else [])
                ):
                    pic, left, top = words[what]
                    im.paste(pic, (left, top), pic)
                if blend_left:
                    im = Image.blend(blend_from, im, 1 - blend_left / (fade + 1))
                    blend_left -= 1
                    key = None  # a picture on its way to another: the next one is drawn afresh
                last, last_key = im, key
                data = im.tobytes()
            enc.stdin.write(data)
            total += 1
        log(f"  {scene.name}: {n / FPS:.1f} s")
    for k in range(12):  # the last picture fades into the band's colour
        enc.stdin.write(Image.blend(last, Image.new("RGB", (W, H), BAND), (k + 1) / 12).tobytes())
    total += 12
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg could not write the picture")
    return total / FPS


# -------------------------------------------------------------------------------------------------- the sound
def ramp(n):
    return np.linspace(0, 1, max(1, n), dtype="float32")[:, None]


def put(track, at, x):
    """Add `x` to `track` from sample `at` on, as far as there is room."""
    if at < 0:
        x, at = x[-at:], 0
    n = min(len(x), len(track) - at)
    if n > 0:
        track[at : at + n] += x[:n]


def mix(scenes, starts, lines, length, log=print):
    """The film's sound: (with the narrator, without) as (samples, 2) arrays at RATE."""
    n = int(length * RATE) + RATE
    music, voice = np.zeros((n, 2), dtype="float32"), np.zeros((n, 2), dtype="float32")
    speaking = []  # [from, to] in the film: where what the pages play is lowered
    for scene, t0 in zip(scenes, starts):
        track, said = sound.scene_sound(scene.folder)
        for line in said:
            log("  " + scene.name + ":" + line)
        edge = int(0.008 * RATE)
        for a, b, seconds in scene.pieces:
            i, j = int((a - scene.start) * RATE), int((b - scene.start) * RATE)
            part = track[max(0, i) : max(0, j)]
            loud = float(np.sqrt((part**2).mean())) if len(part) else 0.0
            if b > a and abs(seconds - (b - a)) < 1e-6:  # at the clock's pace: heard as it was played
                part = part.copy() * MUSIC
                if len(part) > 2 * edge:
                    part[:edge] *= ramp(edge)
                    part[-edge:] *= ramp(edge)[::-1]
                put(music, int((t0 + scene.film(a)) * RATE), part)
            elif loud > 0.003:
                scene.warnings.append(
                    f"{scene.name}: the page was playing at {a - scene.start:.1f} s, in a stretch the film does not show at its own pace: that sound is left out"
                )
        # music under a card
        on = None
        for m in scene.marks + [dict(kind="music", url=None, t=scene.end)]:
            if m["kind"] != "music":
                continue
            if on and scene.film(m["t"]) > scene.film(on["t"]):
                data, own = sf.read(
                    os.path.join(scene.folder, "sounds", scene.sounds[on["url"]]), dtype="float32", always_2d=True
                )
                if own != RATE:
                    raise ValueError("the music under a card has to be at 44.1 kHz")
                seconds = scene.film(m["t"]) - scene.film(on["t"])
                part = data[int(on["at"] * RATE) : int((on["at"] + seconds) * RATE)].copy() * BED
                if part.shape[1] == 1:
                    part = np.repeat(part, 2, axis=1)
                rise, fall = min(len(part) // 2, int(1.2 * RATE)), min(len(part) // 2, int(1.8 * RATE))
                part[:rise] *= ramp(rise) ** 2
                part[-fall:] *= (ramp(fall) ** 2)[::-1]
                put(music, int((t0 + scene.film(on["t"])) * RATE), part[:, :2])
            on = m if m.get("url") else None
        for s in scene.said:
            clip = lines[s["line"]].get("clip")
            if not clip:
                continue
            x, own = sf.read(clip, dtype="float32", always_2d=True)
            if own != RATE:
                raise ValueError(f"{clip} is not at 44.1 kHz")
            x = x.mean(axis=1, keepdims=True)
            frames = np.sqrt((x[: len(x) // 2205 * 2205].reshape(-1, 2205) ** 2).mean(axis=1))  # 50 ms each
            active = frames[frames > 10 ** (-45 / 20)]
            level = float(np.sqrt((active**2).mean())) if len(active) else 1.0
            put(
                voice, int((t0 + s["at"]) * RATE), np.repeat(x * (VOICE / level), 2, axis=1)
            )  # every line as loud as the others
            speaking.append((t0 + s["at"], t0 + s["at"] + s["seconds"]))
    # what the pages play goes down while the narrator speaks, and comes back after
    duck = np.ones(n, dtype="float32")
    for a, b in speaking:
        i, j = int((a - 0.10) * RATE), int((b + 0.05) * RATE)
        duck[max(0, i) : max(0, j)] = UNDER
    k = int(0.22 * RATE)
    duck = np.convolve(np.pad(duck, k, mode="edge"), np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), mode="valid")[
        :n
    ]
    music *= duck[:, None]

    end = int(length * RATE)
    return music[:end], voice[:end], speaking


def master(x):
    """Nothing over full scale: the loudest moments are rounded off."""
    return (np.tanh(x / 0.9) * 0.9).astype("float32")


def loudness(path):
    """A sound file's loudness as broadcasters measure it (LUFS, integrated), by ffmpeg."""
    said = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-threads",
            "1",
            "-i",
            path,
            "-filter_threads",
            "1",
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr
    return float(said.rsplit("I:", 1)[1].split("LUFS")[0])


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def timeline(folders, lines):
    """The scenes one after the other: (scenes, where each starts in the film, the film's length, the lines as they
    are said: id, text, start and length in seconds, and the recording's file name)."""
    scenes = [Scene(f, lines) for f in folders]
    starts, t = [], 0.0
    for s in scenes:
        starts.append(t)
        t += round(s.length * FPS) / FPS
    said = [
        dict(
            id=x["line"],
            text=lines[x["line"]]["text"],
            start=round(t0 + x["at"], 3),
            length=round(x["seconds"], 3),
            file=lines[x["line"]].get("source", ""),
        )
        for s, t0 in zip(scenes, starts)
        for x in s.said
    ]
    return scenes, starts, t + 12 / FPS, said


def write_words(said, base):
    """BASE.narration.json (when each line is said, for laying another voice in) and BASE.srt (the captions)."""
    with open(base + ".narration.json", "w") as f:
        json.dump(said, f, indent=1)
    with open(base + ".srt", "w") as f:
        f.writelines(
            f"{k + 1}\n{srt_time(x['start'])} --> {srt_time(x['start'] + x['length'])}\n{x['text']}\n\n"
            for k, x in enumerate(said)
        )


def assemble(folders, lines, out, work, threads=None, small=True, log=print):
    """The film from its filmed scenes: OUT.mp4; the same without the narrator; a small copy; the captions; and
    when each line is said (narration.json), so that another voice can be laid in."""
    import asyncio

    scenes, starts, length, said = timeline(folders, lines)
    signs = sorted({faster(h["times"]) for s in scenes for h in s.hurried if h["times"] >= 1.5})
    notes = sorted(
        {
            ("sign" if m.get("right") else "left", m["text"].replace("&", "&amp;").replace("<", "&lt;"))
            for s in scenes
            for m in s.marks
            if m["kind"] == "note" and m["text"]
        }
    )
    items = [("caption", x["text"]) for x in said] + [("sign", s) for s in signs] + notes
    words = asyncio.run(draw_words(items, os.path.join(work, "words")))
    base = os.path.splitext(out)[0]
    write_words(said, base)
    log("the picture")
    picture = os.path.join(work, "picture.mp4")
    render(scenes, words, lines, picture, threads, log=log)
    log("the sound")
    music, voice, speaking = mix(scenes, starts, lines, length, log)
    for s in scenes:
        for w in s.warnings:
            log("  note: " + w)
    # how far the narrator stands above what plays under each line: the film must be understood at once
    margins = []
    for a, b in speaking:
        i, j = int(a * RATE), int(b * RATE)
        under = float(np.sqrt((music[i:j] ** 2).mean()))
        margins.append(20 * math.log10(float(np.sqrt((voice[i:j] ** 2).mean())) / under) if under > 1e-5 else 99.0)
    busy = [m for m in margins if m < 99]
    if busy:
        log(
            f"  the voice is {min(busy):.0f} to {max(busy):.0f} dB above what the pages play under it ({len(busy)} of {len(margins)} lines have something under them)"
        )
    stems = {}
    for name, track in (("voice", voice), ("music", music)):
        stems[name] = os.path.join(work, name + ".wav")
        sf.write(stems[name], track, RATE, subtype="PCM_24")
    whole = os.path.join(work, "whole.wav")
    sf.write(whole, master(music + voice), RATE, subtype="PCM_24")
    gain = 10 ** ((LOUD - loudness(whole)) / 20)  # the film as loud as films on the web are
    few = ["-threads", str(threads)] if threads else []  # for an encoder
    one = ["-threads", "1"] if threads else []  # for a decoder
    for name, track in ((out, music + voice), (base + "-no-voice.mp4", music)):
        wav = os.path.join(work, os.path.basename(name) + ".wav")
        sf.write(wav, master(track * gain), RATE, subtype="PCM_24")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                *one,
                "-i",
                picture,
                *one,
                "-i",
                wav,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                name,
            ],
            check=True,
        )
    log(f"  loudness: {loudness(os.path.join(work, os.path.basename(out) + '.wav')):.1f} LUFS")
    if small:  # under 10 MB: what a README on GitHub takes when a film is dragged in
        rate = int(9.4 * 8 * 1024 / length - 72)
        common = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *one,
            "-i",
            out,
            *(["-filter_threads", "1"] if threads else []),
            "-vf",
            "scale=1280:720:flags=lanczos:out_range=tv,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-b:v",
            f"{rate}k",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            *few,
            "-passlogfile",
            os.path.join(work, "pass"),
        ]
        subprocess.run([*common, "-pass", "1", "-an", "-f", "mp4", os.devnull], check=True)
        subprocess.run(
            [*common, "-pass", "2", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", base + "-small.mp4"],
            check=True,
        )
    return dict(
        length=length,
        said=said,
        scenes=[dict(name=s.name, start=round(t0, 2), length=round(s.length, 2)) for s, t0 in zip(scenes, starts)],
        voice_above_music=[round(m, 1) for m in margins],
        hurried=[
            dict(scene=s.name, start=round(t0 + h["a"], 2), end=round(t0 + h["b"], 2), times=round(h["times"], 1))
            for s, t0 in zip(scenes, starts)
            for h in s.hurried
        ],
    )
