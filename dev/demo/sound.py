"""A scene's sound, made again from the files the page played and the log of how it played them (hooks.js).

Nothing is recorded from the browser and no timing is guessed. The page says, for every sound it starts: which
file, from where in it, at what speed, when on its audio clock, through which volume controls and how those move.
This module does the same sums the browser did and lays the result on the film's clock. Two kinds of player:

- Web Audio (the song page): buffers started at exact moments, each through a chain of gains. A note of "My notes"
  is a piece of the bass track cut from another note, played faster or slower to reach its pitch, with a short
  fade at each end; it is made here the same way.
- <audio> elements (the play-along page, the test songs): a file that plays, stops and jumps.

    python dev/demo/sound.py SCENE_FOLDER      writes SCENE_FOLDER/sound.wav and says what was played when
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

RATE = 44100


def read_events(folder):
    with open(os.path.join(folder, "events.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class Files:
    """The sounds of a scene, read once each, at the rate they are played at."""

    def __init__(self, folder, names):
        self.folder, self.names, self.kept = folder, names, {}

    def get(self, url, rate=RATE):
        if (url, rate) not in self.kept:
            data, own = sf.read(os.path.join(self.folder, "sounds", self.names[url]), dtype="float32", always_2d=True)
            if own != rate:  # a browser brings what it decodes to its own rate
                n = int(round(len(data) * rate / own))
                at = np.arange(n) * (own / rate)
                data = np.stack(
                    [np.interp(at, np.arange(len(data)), data[:, c]) for c in range(data.shape[1])], axis=1
                ).astype("float32")
            self.kept[(url, rate)] = data
        return self.kept[(url, rate)]


# ------------------------------------------------------------------------------------------- a volume that moves
def automation(calls):
    """What an AudioParam was told, as a list of (time, how, args) in the order the browser keeps them: by time,
    and for the same time in the order of the calls."""
    timed = []
    for i, c in enumerate(calls):
        how, a = c["how"], c["args"]
        when = (
            a[1]
            if how in ("setValueAtTime", "linearRampToValueAtTime", "exponentialRampToValueAtTime", "setTargetAtTime")
            else a[0]
        )
        timed.append((when, i, how, a))
    return [(w, how, a) for w, _, how, a in sorted(timed, key=lambda x: (x[0], x[1]))]


def values(timeline, default, t):
    """The value of an AudioParam at the moments `t` (seconds on the audio clock, rising), by the rules of Web
    Audio: a value holds until the next event; a linear ramp runs from the event before it to its own time; a
    target is approached from its start with the given time constant."""
    out = np.full(len(t), default, dtype="float64")
    if not len(t) or not timeline:
        return out
    t0, t1 = t[0], t[-1]
    held, since = default, -math.inf  # the value at the last event, and when that was
    approach = None  # (target, start, time constant, value at the start) while a target is being approached

    def now(x):  # the value at moment x as things stand
        if approach:
            target, start, tau, v0 = approach
            return target + (v0 - target) * math.exp(-(x - start) / tau) if x >= start else v0
        return held

    for k, (when, how, a) in enumerate(timeline):
        until = timeline[k + 1][0] if k + 1 < len(timeline) else math.inf
        if how == "linearRampToValueAtTime":
            v0, start = now(when) if approach else held, since  # a ramp starts from the value the event before left
            if when > t0 and start < t1 and when > start:
                seg = (t >= start) & (t < when)
                out[seg] = v0 + (a[0] - v0) * (t[seg] - start) / (when - start)
            held, since, approach = a[0], when, None
        elif how == "setValueAtTime":
            held, since, approach = a[0], when, None
        elif how == "setTargetAtTime":
            v0 = now(when)
            approach, since = (a[0], when, max(a[2], 1e-6), v0), when
        elif how == "cancelScheduledValues":
            continue
        else:
            raise ValueError(f"the page uses {how}, which sound.py does not know yet")
        if until > t0 and when <= t1:
            seg = (t >= when) & (t < until)
            if approach:
                target, start, tau, v0 = approach
                out[seg] = target + (v0 - target) * np.exp(-(t[seg] - start) / tau)
            else:
                out[seg] = held
        if when > t1:
            break
    return out


def pan(x, p):
    """A StereoPannerNode: `x` is (n, 1) or (n, 2), `p` the pan at each sample (-1 left .. 1 right)."""
    if x.shape[1] == 1:
        angle = (p + 1) * (math.pi / 4)
        return np.stack([x[:, 0] * np.cos(angle), x[:, 0] * np.sin(angle)], axis=1)
    left, right = x[:, 0], x[:, 1]
    angle = np.where(p <= 0, p + 1, p) * (math.pi / 2)
    to_left = np.where(p <= 0, left + right * np.cos(angle), left * np.cos(angle))
    to_right = np.where(p <= 0, right * np.sin(angle), right + left * np.sin(angle))
    return np.stack([to_left, to_right], axis=1)


# ------------------------------------------------------------------------------------------------- Web Audio
def stretches(ticks, slack=0.04):
    """The audio clock against the wall clock. The browser says, several times a second, which moment of the audio
    clock is leaving for the loudspeaker and when; the two clocks run alike, so their difference stays the same,
    and one number places everything. With one exception: when a page has been silent for half a minute, Chrome
    stops driving its audio by the sound output and the audio clock falls behind (it runs at a quarter of its
    pace) until the page plays again. So the difference is followed: [(audio time from, to, wall = audio + this,
    steady)], a new stretch wherever it moves by more than `slack` seconds; a stretch is steady when it lasts.
    Within a stretch every sound keeps its exact place against the others."""
    seen = sorted((e["outCtx"], e["outWall"] / 1000 - e["outCtx"]) for e in ticks if e.get("outCtx", 0) > 0)
    if not seen:
        return []
    groups, now = [], [seen[0]]
    for at, off in seen[1:]:
        if abs(off - float(np.median([o for _, o in now[-24:]]))) > slack:
            groups.append(now)
            now = [(at, off)]
        else:
            now.append((at, off))
    groups.append(now)
    out = [[g[0][0], g[-1][0], float(np.median([o for _, o in g])), g[-1][0] - g[0][0] >= 0.6] for g in groups]
    for k in range(len(out) - 1):  # each stretch lasts until the next begins
        out[k][1] = out[k + 1][0]
    out[0][0], out[-1][1] = -math.inf, math.inf
    return out


def clock_fit(ticks):
    """The one number that places a page's sound, where the clocks ran alike throughout: (1, wall minus audio
    time, how far the difference wandered)."""
    s = stretches(ticks, slack=math.inf)
    if not s:
        return 1.0, 0.0, 0.0
    offs = [e["outWall"] / 1000 - e["outCtx"] for e in ticks if e.get("outCtx", 0) > 0]
    return 1.0, s[0][2], float(max(offs) - min(offs))


def web_audio(events, files, start, length, said):
    """Everything one page's audio contexts played, on the film's clock: (samples, 2) from `start` (seconds since
    1970) for `length` seconds. `said` collects a line for each thing played."""
    track = np.zeros((int(length * RATE) + 1, 2), dtype="float32")
    docs = defaultdict(list)
    for e in events:
        docs[e.get("doc")].append(e)
    for evs in docs.values():
        contexts = {e["id"]: e for e in evs if e["ev"] == "context"}
        if not contexts:
            continue
        nodes = {e["node"]: e for e in evs if e["ev"] == "node"}
        wires = {e["node"]: e["to"] for e in evs if e["ev"] == "connect"}
        buffers = {e["buf"]: e for e in evs if e["ev"] == "buffer"}
        params = defaultdict(list)
        for e in evs:
            if e["ev"] == "param":
                params[e["param"]].append(e)
        lines = {p: automation(calls) for p, calls in params.items()}
        stops = {e["node"]: e for e in evs if e["ev"] == "stop"}
        for cid, made in contexts.items():
            rate = made["rate"]
            if rate != RATE:
                raise ValueError(f"the page's audio runs at {rate} Hz; sound.py makes the film's sound at {RATE} Hz")
            clock = stretches(
                [
                    e
                    for e in evs
                    if e["ev"] in ("clock", "start")
                    and e.get("id", e.get("ac")) == cid
                    and e.get("state", "running") == "running"
                ]
            )
            starts = [e for e in evs if e["ev"] == "start" and e["ac"] == cid and e["buf"] in buffers]
            if not starts or not clock:
                continue

            def where(t):  # the stretch of the clock a moment falls in
                return next(k for k, c in enumerate(clock) if c[0] <= t < c[1])

            mixes = {}  # per stretch of the clock: [audio time of the first sample, samples]
            end = max((e["ctx"] for e in evs if "ctx" in e), default=0) + 2
            for e in starts:
                buf = buffers[e["buf"]]
                if not buf["url"]:
                    raise ValueError("a sound was played that did not come from a file: sound.py cannot make it again")
                data = files.get(buf["url"], rate)
                speed = e["speed"]
                t0 = max(e["when"] or 0.0, e["ctx"])
                offset = e["offset"] or 0.0
                span = (
                    len(data) / rate - offset
                    if e["duration"] is None
                    else min(e["duration"], len(data) / rate - offset)
                )
                t1 = t0 + max(0.0, span) / speed
                stop = stops.get(e["node"])
                if stop:
                    t1 = min(t1, max(stop["when"] or 0.0, stop["ctx"]))
                n = int((t1 - t0) * rate)
                if n <= 0:
                    continue
                at = offset * rate + np.arange(n) * speed  # where in the file each sample comes from
                last = len(data) - 1
                lo = np.minimum(at.astype("int64"), last)
                frac = (at - lo).astype("float32")[:, None]
                x = data[lo] * (1 - frac) + data[np.minimum(lo + 1, last)] * frac
                t = t0 + np.arange(n) / rate
                node, steps = e["node"], 0
                while node in wires and wires[node] != "out" and steps < 20:
                    node, steps = wires[node], steps + 1
                    kind = nodes.get(node)
                    if not kind:
                        raise ValueError("the page's sound goes through something sound.py does not know yet")
                    v = values(lines.get(kind["param"], []), kind["value"], t)
                    x = x * v[:, None].astype("float32") if kind["kind"] == "gain" else pan(x, v).astype("float32")
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                k = where(t0)
                if not clock[k][3]:  # the first sound after a long silence wakes the clock: it keeps time from then on
                    k = next((j for j in range(k + 1, len(clock)) if clock[j][3]), k)
                if not clock[k][3] or clock[k][0] - t0 > 0.5 or where(t1 - 1e-6) > k:
                    off = clock[k][2] - start
                    said.append(
                        f"  WARNING: the browser's audio clock was not keeping time while a sound played ({t0 + off:.1f} s to {t1 + off:.1f} s of the scene): what was heard there cannot be made again. Film the scene again."
                    )
                if k not in mixes:
                    mixes[k] = np.zeros((int(end * rate) + 1, 2), dtype="float32")
                i = int(round(t0 * rate))
                room = min(n, len(mixes[k]) - i)
                mixes[k][i : i + room] += x[:room]
            for (
                k,
                part,
            ) in (
                mixes.items()
            ):  # onto the film's clock: the sample of audio time T is heard at T + the stretch's difference
                i = int(round((clock[k][2] - start) * RATE))
                src0, dst0 = max(0, -i), max(0, i)
                n = min(len(part) - src0, len(track) - dst0)
                if n > 0:
                    track[dst0 : dst0 + n] += part[src0 : src0 + n]
            # what was played, for the record: the long sounds by file, the notes counted
            by_bus = defaultdict(list)
            for e in starts:
                by_bus[os.path.basename(buffers[e["buf"]]["url"]) if e["duration"] is None else "notes"].append(e)

            def scene_time(t):
                k = where(t)
                return (
                    t
                    + clock[k if clock[k][3] else next((j for j in range(k + 1, len(clock)) if clock[j][3]), k)][2]
                    - start
                )

            for name, group in sorted(by_bus.items()):
                if name == "notes":
                    said.append(
                        f"  {len(group)} notes, each a piece of the bass track, between {scene_time(group[0]['when']):.2f} s and {scene_time(group[-1]['when']):.2f} s"
                    )
                else:
                    for e in group:
                        stop = stops.get(e["node"])
                        till = f"{scene_time(max(stop['when'] or 0, stop['ctx'])):.2f} s" if stop else "its end"
                        said.append(
                            f"  {name} from {e['offset'] or 0:.2f} s of the file, at {scene_time(max(e['when'] or 0, e['ctx'])):.2f} s of the scene until {till}"
                        )
    return track


# ------------------------------------------------------------------------------------------ <audio> elements
def media(events, files, start, length, said):
    """Everything the page's <audio> elements played, on the film's clock."""
    track = np.zeros((int(length * RATE) + 1, 2), dtype="float32")
    players = defaultdict(list)
    for e in events:
        if e["ev"] == "media" and e.get("url") in files.names:
            players[(e["doc"], e["el"])].append(e)
    for evs in players.values():
        spans, open_ = [], None  # a span: the element played without a break
        for e in sorted(evs, key=lambda e: e["wall"]):
            going = not e["paused"] and e["type"] not in ("pause", "waiting", "seeking", "ended", "emptied")
            if open_ and (not going or e["url"] != open_["url"] or e["speed"] != open_["speed"]):
                open_["end"] = e["wall"] / 1000
                spans.append(open_)
                open_ = None
            if going and not open_ and e["type"] in ("playing", "seeked", "tick", "timeupdate"):
                open_ = dict(url=e["url"], speed=e["speed"], volume=e["volume"], begin=e["wall"] / 1000, seen=[])
            if going and open_:
                open_["seen"].append((e["wall"] / 1000, e["at"]))
        if open_:
            open_["end"] = start + length
            spans.append(open_)
        for s in spans:
            if s["end"] - s["begin"] < 0.02 or not s["seen"]:
                continue
            data = files.get(s["url"])
            # where in the file the element was at each look: the file's time 0 on the wall clock, the middle one
            zero = float(np.median([w - at / s["speed"] for w, at in s["seen"]]))
            n = int((s["end"] - s["begin"]) * RATE)
            at = ((s["begin"] - zero) * RATE + np.arange(n)) * s["speed"]
            ok = (at >= 0) & (at < len(data) - 1)
            lo = at[ok].astype("int64")
            frac = (at[ok] - lo).astype("float32")[:, None]
            x = np.zeros((n, data.shape[1]), dtype="float32")
            x[ok] = (data[lo] * (1 - frac) + data[lo + 1] * frac) * s["volume"]
            fade = min(n // 2, int(0.008 * RATE))  # an element starts and stops without a click
            if fade:
                ramp = np.linspace(0, 1, fade, dtype="float32")[:, None]
                x[:fade] *= ramp
                x[-fade:] *= ramp[::-1]
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            i = int(round((s["begin"] - start) * RATE))
            src0, dst0 = max(0, -i), max(0, i)
            room = min(n - src0, len(track) - dst0)
            if room > 0:
                track[dst0 : dst0 + room] += x[src0 : src0 + room, :2]
            said.append(
                f"  {os.path.basename(s['url'])} from {(s['begin'] - zero) * s['speed']:.2f} s of the file, at {s['begin'] - start:.2f} s of the scene for {s['end'] - s['begin']:.2f} s"
            )
    return track


def scene_sound(folder):
    """The sound of a filmed scene: (samples, 2) at RATE from the scene's start, and what was played, in words."""
    with open(os.path.join(folder, "scene.json")) as f:
        scene = json.load(f)
    events, said = read_events(folder), []
    files = Files(folder, scene["sounds"])
    length = scene["end"] - scene["start"]
    track = web_audio(events, files, scene["start"], length, said) + media(events, files, scene["start"], length, said)
    return track[: int(length * RATE)], said


if __name__ == "__main__":
    track, said = scene_sound(sys.argv[1])
    sf.write(os.path.join(sys.argv[1], "sound.wav"), track, RATE, subtype="FLOAT")
    peak = float(np.abs(track).max()) if len(track) else 0.0
    print(f"{len(track) / RATE:.1f} s, peak {20 * math.log10(peak + 1e-9):.1f} dB")
    print("\n".join(said))
