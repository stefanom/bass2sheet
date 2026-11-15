"""Is the film's sound what the browser really played? A check of sound.py against the browser itself.

    python dev/demo/check_sound.py http://localhost:8770/song#SONG_ID

The song page is opened without a window and made to play: the record with the notes, a note by itself, the song
slowed down. Two things are kept: the log that sound.py works from, and what actually left the page's audio
graph, taken by a tap on the audio thread (an AudioWorklet that copies every sample it is given, with its place
on the audio clock). The soundtrack sound.py makes from the log is then laid on the tap's, sample by sample.

They should differ by 40 dB or more under the sound (what is left is how the browser and numpy round a note
that is played faster or slower), and by much less at a shift of one sample: that is the proof that every sound
is placed where the page placed it. The song is not changed: nothing is fixed, only played.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile

import numpy as np
import sound
from film import Film

READY = "typeof S !== 'undefined' && S && tiles.length > 0 && A.bass && A.record"
TAP = """async () => {
  const c = A.ctx;
  const code = `class Tap extends AudioWorkletProcessor { process(inputs) { const i = inputs[0];
    if (i.length) this.port.postMessage({frame: currentFrame, l: i[0].slice(), r: (i[1] || i[0]).slice()}); return true; } }
    registerProcessor('tap', Tap);`;
  await c.audioWorklet.addModule(URL.createObjectURL(new Blob([code], {type: 'application/javascript'})));
  const tap = new AudioWorkletNode(c, 'tap', {numberOfInputs: 1, numberOfOutputs: 0, channelCount: 2, channelCountMode: 'explicit'});
  window.__tap = [];
  tap.port.onmessage = e => window.__tap.push(e.data);
  for (const k of ['record', 'bass', 'notes']) A.pan[k].connect(tap);
  window.__tapped = () => { const n = window.__tap.length, l = new Float32Array(n * 128), r = new Float32Array(n * 128), frames = [];
    window.__tap.forEach((b, k) => { l.set(b.l, k * 128); r.set(b.r, k * 128); frames.push(b.frame); });
    const b64 = a => { const u = new Uint8Array(a.buffer); let s = ''; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); };
    return {frames, l: b64(l), r: b64(r)}; };
}"""


async def listen(url, folder):
    async with Film(folder) as f:
        await f.goto(url, READY)
        await f.js(TAP)
        await f.roll()
        await f.press("#play")
        await f.pause(4.0)
        await f.press("#play")
        await f.pause(0.5)
        await f.page.keyboard.press("ArrowLeft")  # the note before the playhead, heard in its place
        await f.pause(0.3)
        await f.page.keyboard.press("Enter")
        await f.pause(2.2)
        await f.choose("#speed", "70")
        await f.ready("A.speed === 0.7 && !$('speed').disabled", timeout=300000)
        await f.press("#play")
        await f.pause(3.0)
        await f.press("#play")
        await f.pause(0.6)
        tap = await f.js("window.__tapped()")
        await f.cut()
    return tap


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = tempfile.mkdtemp(prefix="check-sound-")
    tap = asyncio.run(listen(sys.argv[1], folder))
    frames = np.array(tap["frames"])
    left, right = (np.frombuffer(base64.b64decode(tap[k]), dtype="float32") for k in ("l", "r"))
    real = np.zeros((int(frames[-1]) + 128, 2), dtype="float32")
    for k, at in enumerate(frames):
        real[at : at + 128, 0], real[at : at + 128, 1] = left[k * 128 : (k + 1) * 128], right[k * 128 : (k + 1) * 128]
    with open(os.path.join(folder, "scene.json")) as f:
        scene = json.load(f)
    events = sound.read_events(folder)
    mine = {e["node"] for e in events if e["ev"] == "node"} | {"out"}
    events = [
        e for e in events if not (e["ev"] == "connect" and e["to"] not in mine)
    ]  # the tap's own wires are not the page's
    _, b, wandered = sound.clock_fit([e for e in events if e["ev"] in ("clock", "start")])
    if wandered > 0.04:
        raise SystemExit(
            f"the browser's audio clock did not keep time (it wandered {wandered * 1000:.0f} ms): the machine was busy. Try again."
        )
    said = []
    made = sound.web_audio(
        events, sound.Files(folder, scene["sounds"]), b, len(real) / sound.RATE + 1, said
    )  # on the audio clock itself
    print("\n".join(said))
    n = min(len(made), len(real))
    level = np.sqrt((real[:n] ** 2).mean())

    def apart(shift):
        a, r = made[max(0, shift) : n + min(0, shift)], real[max(0, -shift) : n - max(0, shift)]
        k = min(len(a), len(r))
        return 20 * np.log10(np.sqrt(((a[:k] - r[:k]) ** 2).mean()) / level + 1e-12)

    print(
        f"{n / sound.RATE:.1f} s compared. The difference lies {-apart(0):.1f} dB under the sound; with one sample's shift, {-apart(1):.1f} dB; with a hundredth of a second's, {-apart(441):.1f} dB."
    )
    if apart(0) > -35:
        raise SystemExit("sound.py does not make what the browser played: has the page's way of playing changed?")
    print("sound.py makes what the browser played.")


if __name__ == "__main__":
    main()
