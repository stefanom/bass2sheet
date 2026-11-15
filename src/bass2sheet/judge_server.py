"""The judging page: a local web server on the standard library, for one person on their own machine.

The page never learns which version is which until a pick is recorded: the server hands it audio, pictures and
score snippets named A and B, and reveals what they were only in the answer to the pick. As picks come in, the
server prepares the steps they open in the background (judge.advance).
"""

import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from . import judge
from .common import Bass2SheetError, PageServer

mmss = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"
LEGENDS = dict(
    stem="Amber marks what sounds in one track and hardly in the other. Click a picture to play from there.",
    heard="The boxes are the notes found; amber ones are in this version only. The red line is the pitch that was "
    "tracked. E1, A1, D2 and G2 are the open strings. Click a picture to play from there.",
    marked="Shaded notes are the ones the two versions write differently. Click a score to play from there.",
    plain="Click a score to play from there.",
)


class Worker:
    """Prepares the steps that picks open, in the background, one run at a time."""

    def __init__(self, workdir):
        self.workdir, self.lock = workdir, threading.Lock()
        self.running = self.again = False
        self.error = None

    def kick(self):
        with self.lock:
            if self.running:
                self.again = True  # picks came in during a run: look again when it ends
                return
            self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                judge.advance(self.workdir, say=lambda text: None)
                self.error = None
            except Exception as e:  # shown on the page: the person cannot see this process
                self.error = f"{type(e).__name__}: {e}"
            with self.lock:
                if not self.again:
                    self.running = False
                    return
                self.again = False


def public(comp, judgment=None, steps=()):
    """What the page may know about a comparison before the pick: nothing that tells the versions apart."""
    ch = judge.choice_of(comp["choice"])
    media = comp.get("media", {})
    sources = comp.get("sources") or [
        [n, n.capitalize() if n == "record" else n] for n in ("record", "A", "B") if n in media
    ]
    alt = comp.get("alt")
    if alt is None and "A_bass" in media:  # a directory made before the steps
        alt = dict(
            label="Bass only", media={n: media[f"{n}_bass"] for n in ("record", "A", "B") if f"{n}_bass" in media}
        )
    sheets = comp.get("sheets", {})
    legend = (
        comp["kind"] if comp["kind"] in LEGENDS else "marked" if any(s["marks"] for s in sheets.values()) else "plain"
    )
    number = f"Step {steps.index(comp['choice']) + 1} of {len(steps)} · " if comp["choice"] in steps else ""
    out = dict(
        id=comp["id"],
        choice=comp["choice"],
        step=number + ch["name"],
        title=comp["title"],
        artist=comp["artist"],
        where=f"bars {comp['bars_a'][0]}–{comp['bars_a'][1]} · {mmss(comp['t0'])}–{mmss(comp['t1'])}",
        about=ch["about"],
        ask=ch["ask"],
        lead=comp.get("lead", 0.0),
        length=comp.get("length", 0.0),
        sources=[[name, label, key] for (name, label), key in zip(sources, "rab")],
        media={name: "/" + media[name] for name, _ in sources},
        alt=dict(label=alt["label"], media={n: "/" + path for n, path in alt["media"].items()}) if alt else None,
        sheets={name: dict(sheet, src="/" + sheet["src"]) for name, sheet in sheets.items()},
        legend=LEGENDS[legend] if sheets else "",
    )
    if judgment:  # judged before: the page shows the earlier pick and what the versions were
        out.update(pick=judgment["pick"], note=judgment["note"], reveal=judge.shown_labels(comp))
    return out


def state(workdir, want=None, after=None, step=None, worker=None):
    """Progress and the comparison to show: the one asked for; else the first one of `step` that is not judged;
    else the first not judged that follows `after` in the order, wrapping round to the ones skipped earlier."""
    # Asked before the comparisons are read. A step that becomes ready while they are read is then still told as
    # being prepared, and the page looks again; asked after, it would be told as neither there nor on its way.
    preparing = bool(worker and worker.running)
    judged = judge.load_judgments(workdir)
    comps = judge.live_comparisons(workdir, judged=judged)
    ids = [c["id"] for c in comps]
    comp = next((c for c in comps if c["id"] == want), None) if want else None
    if comp is None and step:
        mine = [c for c in comps if c["choice"] == step]
        comp = next((c for c in mine if c["id"] not in judged), mine[0] if mine else None)
    if comp is None:
        start = ids.index(after) + 1 if after in ids else 0
        comp = next((c for c in comps[start:] + comps[:start] if c["id"] not in judged), None)
    k = ids.index(comp["id"]) if comp else len(ids)
    steps = [s["id"] for s in judge.steps_of(judge.read_settings(workdir))]
    return dict(
        total=len(comps),
        judged=sum(i in judged for i in ids),
        steps=judge.progress(workdir, comps, judged),
        preparing=preparing,
        error=worker.error if worker else None,
        comparison=public(comp, judged.get(comp["id"]), steps) if comp else None,
        previous=ids[k - 1] if k > 0 else None,
    )


class Handler(BaseHTTPRequestHandler):
    workdir = "."
    worker = None
    server_version = "bass2sheet-judge"

    def log_message(self, fmt, *args):  # one person, one machine: no access log
        pass

    def send_body(self, body, ctype, status=200, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send_body(json.dumps(obj).encode(), "application/json", status)

    def send_file(self, rel):
        """A media file, with byte ranges (Safari will not play audio without them)."""
        root = os.path.realpath(os.path.join(self.workdir, "media"))
        path = os.path.realpath(os.path.join(self.workdir, rel))
        if not path.startswith(root + os.sep) or not os.path.isfile(path):
            return self.send_json(dict(error="not found"), 404)
        size = os.path.getsize(path)
        ctype = "audio/flac" if path.endswith(".flac") else mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start, end = int(m.group(1)), min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                return self.send_body(b"", ctype, 416, {"Content-Range": f"bytes */{size}"})
            status = 206
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        extra = {"Accept-Ranges": "bytes"}
        if status == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_body(body, ctype, status, extra)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_body(PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/next":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            self.send_json(state(self.workdir, q.get("id"), q.get("after"), q.get("step"), self.worker))
        elif url.path == "/api/results":
            self.send_json(judge.tally(self.workdir))
        elif url.path.startswith("/media/"):
            self.send_file(url.path.lstrip("/"))
        else:
            self.send_json(dict(error="not found"), 404)

    do_HEAD = do_GET

    def do_POST(self):
        if urlparse(self.path).path != "/api/judge":
            return self.send_json(dict(error="not found"), 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            comp = next((c for c in judge.load_comparisons(self.workdir) if c["id"] == data.get("id")), None)
            if comp is None:
                return self.send_json(dict(error="no such comparison"), 404)
            judge.record_judgment(self.workdir, comp, data.get("pick"), data.get("note", ""))
        except (ValueError, Bass2SheetError) as e:
            return self.send_json(dict(error=str(e)), 400)
        self.worker.kick()  # this pick may have opened a step
        self.send_json(dict(reveal=judge.shown_labels(comp), **state(self.workdir, comp["id"], worker=self.worker)))


def make_server(workdir, port=8765):
    if not os.path.exists(os.path.join(workdir, "comparisons.json")):
        raise Bass2SheetError(
            f"no comparisons in {workdir}: run `bass2sheet-judge build SONG... --dir {workdir}` first"
        )
    workdir = os.path.abspath(workdir)
    handler = type("JudgeHandler", (Handler,), dict(workdir=workdir, worker=Worker(workdir)))
    try:
        server = PageServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise Bass2SheetError(f"cannot listen on port {port} ({e}); pick another with --port") from e
    handler.worker.kick()  # steps opened by picks made before this run
    return server


def serve(workdir, port=8765):
    server = make_server(workdir, port)
    port = server.server_address[1]  # the one it got: `--port 0` asks for any that is free
    s = state(workdir)
    print(
        f"{s['total'] - s['judged']} passages to judge ({s['judged']} done). Open http://localhost:{port}", flush=True
    )
    print("Stop with Ctrl-C; your picks are saved as you make them.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bass2sheet · judging</title>
<link rel="icon" href="data:,">
<style>
  :root { --ink:#1c1f24; --soft:#5d6570; --line:#d9dde3; --bg:#f6f7f9; --card:#fff; --go:#1f6feb; --on:#0b5d2a; --warn:#9a6700; }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }
  header { display:flex; gap:16px; align-items:baseline; padding:14px 24px; border-bottom:1px solid var(--line); background:var(--card); }
  header h1 { font-size:17px; margin:0; }
  header .grow { flex:1; }
  header button { font:inherit; background:none; border:0; color:var(--go); cursor:pointer; padding:0; }
  nav { display:flex; flex-wrap:wrap; gap:8px; padding:10px 24px; background:var(--card); border-bottom:1px solid var(--line); }
  nav button { font:inherit; font-size:14px; border:1px solid var(--line); border-radius:999px; background:#fff; padding:5px 12px; cursor:pointer; color:var(--ink); }
  nav button span { color:var(--soft); margin-left:4px; font-variant-numeric:tabular-nums; }
  nav button.current { border-color:var(--go); box-shadow:0 0 0 2px rgba(31,111,235,.2); }
  nav button.done { background:#e9f6ee; border-color:#b7dfc5; }
  nav button:disabled { color:var(--soft); background:var(--bg); cursor:default; }
  main { max-width:1180px; margin:20px auto 60px; padding:0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px 22px; margin-bottom:16px; }
  h2 { font-size:22px; margin:0 0 2px; }
  .stepname { color:var(--go); font-weight:600; font-size:14px; margin-bottom:4px; }
  .where { color:var(--soft); margin-bottom:12px; }
  .about { margin:0 0 6px; }
  .ask { font-weight:600; margin:0; }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  button.play, button.pick, button.next { font:inherit; font-weight:600; border-radius:8px; border:1px solid var(--line); background:#fff; padding:10px 18px; cursor:pointer; }
  button.play.on { background:var(--on); color:#fff; border-color:var(--on); }
  button.pick { min-width:150px; }
  button.pick.chosen { background:var(--go); color:#fff; border-color:var(--go); }
  button.chip { font:inherit; font-size:14px; border-radius:16px; border:1px solid var(--line); background:#fff; padding:5px 12px; cursor:pointer; }
  button.chip.chosen { background:var(--go); color:#fff; border-color:var(--go); }
  #why { margin-top:12px; align-items:center; }
  button.next { background:var(--go); color:#fff; border-color:var(--go); }
  button:disabled { opacity:.45; cursor:default; }
  label.opt { color:var(--soft); display:flex; gap:6px; align-items:center; }
  .bar { position:relative; height:8px; background:#e6e9ee; border-radius:4px; margin-top:14px; cursor:pointer; }
  .bar .fill { position:absolute; left:0; top:0; bottom:0; width:0; background:var(--on); border-radius:4px; }
  .bar .mark { position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--soft); }
  .keys { color:var(--soft); font-size:13px; margin-top:10px; }
  .sheet { margin-top:6px; }
  .sheet h3 { margin:14px 0 4px; font-size:15px; }
  .score { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; }
  .score.active { border-color:var(--go); box-shadow:0 0 0 2px rgba(31,111,235,.25); }
  .score img { position:absolute; max-width:none; display:block; user-select:none; -webkit-user-drag:none; }
  .score svg { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
  .score svg .guide { stroke:rgba(0,0,0,.14); stroke-width:1; }
  .score svg .name { font:11px -apple-system, Helvetica, Arial, sans-serif; fill:var(--soft); }
  .score svg .n { fill:rgba(11,93,42,.20); stroke:#0b5d2a; stroke-width:1.3; }
  .score svg .n.d { fill:rgba(255,193,7,.45); stroke:#9a6700; }
  .score svg .line { fill:none; stroke:#c62828; stroke-width:1.5; }
  .score .mark { position:absolute; background:rgba(255,193,7,.30); border-radius:3px; pointer-events:none; }
  .score .cursor { position:absolute; width:2px; margin-left:-1px; background:var(--go); box-shadow:0 0 0 5px rgba(31,111,235,.16); pointer-events:none; }
  .legend { color:var(--soft); font-size:13px; margin-top:8px; }
  textarea { width:100%; min-height:54px; font:inherit; padding:8px 10px; border:1px solid var(--line); border-radius:8px; margin-top:12px; }
  .reveal { margin-top:14px; padding:12px 14px; background:#eef6ff; border:1px solid #c8defc; border-radius:8px; }
  .reveal b { display:inline-block; min-width:1.6em; }
  .note { color:var(--soft); font-size:14px; }
  table { border-collapse:collapse; width:100%; margin:6px 0 14px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
  td.n { text-align:right; font-variant-numeric:tabular-nums; width:90px; }
  .hidden { display:none; }
  .error { color:#b42318; }
</style>
</head>
<body>
<header>
  <h1>bass2sheet · judging</h1>
  <span id="progress" class="note"></span>
  <span class="grow"></span>
  <button id="toggleResults">Picks so far</button>
</header>
<nav id="steps"></nav>
<main>
  <div id="judging">
    <div class="card">
      <div class="stepname" id="stepname"></div>
      <h2 id="title"></h2>
      <div class="where" id="where"></div>
      <p class="about" id="about"></p>
      <p class="ask" id="ask"></p>
    </div>
    <div class="card">
      <div class="row" id="players"></div>
      <div class="row" style="margin-top:12px">
        <label class="opt" id="soloOpt"><input type="checkbox" id="solo"> <span id="soloLabel">Bass only</span></label>
        <label class="opt"><input type="checkbox" id="loop" checked> Loop</label>
        <button class="play" id="stop">Stop</button>
      </div>
      <div class="bar" id="bar"><div class="fill" id="fill"></div><div class="mark" id="mark"></div></div>
      <div class="keys" id="keys"></div>
      <div class="sheet" id="sheets"></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <div class="row" id="picks"></div>
      <div class="row hidden" id="why"></div>
      <textarea id="note" placeholder="Optional: what is wrong, or what you would want instead"></textarea>
      <div class="reveal hidden" id="reveal"></div>
      <div class="row" style="margin-top:14px">
        <button class="next hidden" id="next">Next passage</button>
        <button class="play" id="back">Previous</button>
        <button class="play" id="skip">Skip for now</button>
        <span class="error" id="error"></span>
      </div>
    </div>
  </div>
  <div id="waiting" class="card hidden">
    <h2>Preparing the next step</h2>
    <p>It is being built on your picks so far. This takes about a minute, and the page carries on by itself.</p>
    <p class="error" id="workerError"></p>
  </div>
  <div id="done" class="card hidden">
    <h2>Everything is judged</h2>
    <p>Your picks are saved. They are listed under “Picks so far”.</p>
  </div>
  <div id="results" class="card hidden"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let current = null, state = null, audio = {}, playing = null, at = 0, from = 0, timer = null;
const PICKS = [["A", "A is better"], ["B", "B is better"], ["same", "No difference"], ["neither", "Neither is right"]];
const WHY = ["wrong notes", "too high", "too low", "extra notes", "missing notes", "wrong timing"];  // for "neither"
const STRINGS = [[28, "E1"], [33, "A1"], [38, "D2"], [43, "G2"], [50, "D3"], [55, "G3"]];

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
async function api(path, body) {
  const r = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function url(name) {
  const other = current.alt && $("solo").checked ? current.alt.media[name] : null;
  return other || current.media[name];
}
function stop() {
  if (playing && audio[playing]) { at = audio[playing].currentTime; audio[playing].pause(); }
  playing = null; paint();
}
// Where the cursor stands on a score at `t` seconds into the audio: a straight line between the notes' places.
function xAt(keys, t) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let k = 1; k < keys.length; k++) {
    if (t <= keys[k][0]) {
      const [t0, x0] = keys[k - 1], [t1, x1] = keys[k];
      return t1 > t0 ? x0 + (x1 - x0) * (t - t0) / (t1 - t0) : x1;
    }
  }
  return keys[keys.length - 1][1];
}
function tAt(keys, x) {
  if (x <= keys[0][1]) return keys[0][0];
  for (let k = 1; k < keys.length; k++) {
    if (x <= keys[k][1]) {
      const [t0, x0] = keys[k - 1], [t1, x1] = keys[k];
      return x1 > x0 ? t0 + (t1 - t0) * (x - x0) / (x1 - x0) : t1;
    }
  }
  return keys[keys.length - 1][0];
}
function place(t) {
  if (!current) return;
  for (const [name, s] of Object.entries(current.sheets || {})) {
    $("cursor-" + name).style.left = (100 * (xAt(s.keys, t) - s.crop[0]) / (s.crop[2] - s.crop[0])) + "%";
  }
}
// A pitch picture's drawing: the open strings, the notes found (amber: in this version only), the tracked pitch.
function rollSvg(s) {
  const [W, H] = s.page, r = s.roll, length = s.keys[1][0], band = H / (r.hi - r.lo + 1);
  const x = t => W * t / length, y = m => H * (r.hi + 0.5 - m) / (r.hi - r.lo + 1);
  const guides = STRINGS.filter(([m]) => m >= r.lo && m <= r.hi).map(([m, name]) =>
    `<line class="guide" x1="0" x2="${W}" y1="${y(m).toFixed(1)}" y2="${y(m).toFixed(1)}"/><text class="name" x="6" y="${(y(m) - 3).toFixed(1)}">${name}</text>`).join("");
  const notes = r.notes.map(([t0, t1, m, differs]) =>
    `<rect class="${differs ? "n d" : "n"}" x="${x(t0).toFixed(1)}" y="${(y(m) - band / 2).toFixed(1)}" width="${Math.max(2, x(t1) - x(t0) - 1).toFixed(1)}" height="${band.toFixed(1)}" rx="2"/>`).join("");
  let line = "", run = [];
  for (const p of r.line.concat([null])) {
    if (p) { run.push(x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)); continue; }
    if (run.length > 1) line += `<polyline class="line" points="${run.join(" ")}"/>`;
    run = [];
  }
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${guides}${notes}${line}</svg>`;
}
function sheetHtml(name, s) {
  const [x0, y0, x1, y1] = s.crop, w = x1 - x0, h = y1 - y0, pct = v => (100 * v) + "%";
  const top = pct((s.staff[0] - y0) / h), height = pct((s.staff[1] - s.staff[0]) / h);
  const marks = s.marks.map(x => `<div class="mark" style="left:${pct((x - 11 - x0) / w)};width:${pct(22 / w)};top:${top};height:${height}"></div>`).join("");
  return `<h3>${name}</h3><div class="score" id="score-${name}" style="aspect-ratio:${w}/${h}">` +
    `<img alt="${name}" src="${s.src}" draggable="false" style="left:${pct(-x0 / w)};top:${pct(-y0 / h)};width:${pct(s.page[0] / w)}">` +
    (s.roll ? rollSvg(s) : "") + marks + `<div class="cursor" id="cursor-${name}" style="top:${top};height:${height}"></div></div>`;
}
// A click on a score plays from the note under it: that score's version, or whatever is playing already.
function seek(name, e) {
  const s = current.sheets[name];
  const t = tAt(s.keys, s.crop[0] + (s.crop[2] - s.crop[0]) * e.offsetX / e.currentTarget.clientWidth);
  const source = playing || (audio[name] ? name : audio.A ? "A" : "record");
  if (playing && playing !== source) audio[playing].pause();
  playing = source;
  audio[source].currentTime = t;
  audio[source].play().catch(err => { $("error").textContent = "Cannot play audio: " + err.message; });
  paint();
}
function load() {
  stop();
  audio = {};
  for (const [name] of current.sources) {
    const a = new Audio(url(name));
    a.preload = "auto";
    a.addEventListener("ended", () => { if ($("loop").checked && playing === name) { a.currentTime = 0; a.play(); } else if (playing === name) { playing = null; paint(); } });
    audio[name] = a;
  }
}
function play(name) {
  const next = audio[name];
  if (!next) return;
  if (playing && playing !== name) { at = audio[playing].currentTime; audio[playing].pause(); }
  else if (!playing) { at = from; from = 0; }     // from the start, or from where the bar was clicked while stopped
  next.currentTime = playing === name ? 0 : at;   // the same button again starts over; another one keeps the place
  playing = name;
  next.play().catch(e => { $("error").textContent = "Cannot play audio: " + e.message; });
  paint();
}
function paint() {
  for (const b of document.querySelectorAll("#players button")) b.classList.toggle("on", b.dataset.name === playing);
  for (const name of Object.keys((current && current.sheets) || {})) $("score-" + name).classList.toggle("active", playing === name);
}
function tick() {
  if (current && playing && audio[playing] && current.length) {
    at = audio[playing].currentTime;
    $("fill").style.width = (100 * at / current.length) + "%";
    place(at);
  }
  requestAnimationFrame(tick);
}
function stepsBar(s) {
  const numbered = s.steps.filter(st => st.id !== "change");
  $("steps").innerHTML = s.steps.map(st => {
    const cls = "step" + (current && current.choice === st.id ? " current" : "") + (st.total && st.judged === st.total ? " done" : "");
    const label = (st.id === "change" ? "" : (numbered.indexOf(st) + 1) + " ") + esc(st.name);
    return `<button class="${cls}"${st.total ? "" : " disabled"} data-step="${st.id}">${label}<span>${st.total ? st.judged + "/" + st.total : "–"}</span></button>`;
  }).join("");
  for (const b of document.querySelectorAll("#steps button")) b.onclick = () => go("?step=" + encodeURIComponent(b.dataset.step));
}
function show(s, query) {
  stop();
  if (timer) { clearTimeout(timer); timer = null; }
  state = s; current = s.comparison;
  $("progress").textContent = (s.total ? `${s.judged} of ${s.total} judged` : "nothing to judge yet") + (s.preparing ? " · preparing the next step" : "");
  $("error").textContent = "";
  stepsBar(s);
  const wait = !current && (s.preparing || s.error);
  $("judging").classList.toggle("hidden", !current);
  $("waiting").classList.toggle("hidden", !wait);
  $("done").classList.toggle("hidden", !!current || !!wait);
  $("workerError").textContent = s.error ? "Preparing failed: " + s.error : "";
  if (!current) {
    if (s.preparing) timer = setTimeout(() => go(query), 2000);   // look again until the step is there
    return;
  }
  $("stepname").textContent = current.step;
  $("title").textContent = (current.artist ? current.artist + " – " : "") + current.title;
  $("where").textContent = current.where;
  $("about").textContent = current.about;
  $("ask").textContent = current.ask;
  $("solo").checked = false;
  $("soloOpt").classList.toggle("hidden", !current.alt);
  $("soloLabel").textContent = current.alt ? current.alt.label : "";
  $("players").innerHTML = current.sources.map(([n, label]) => `<button class="play" data-name="${n}">▶ ${esc(label)}</button>`).join("");
  for (const b of document.querySelectorAll("#players button")) b.onclick = () => play(b.dataset.name);
  $("keys").textContent = "Keys: " + current.sources.map(([n, label, k]) => `${k.toUpperCase()} ${label}`).join(" · ") + " · Space stop" + (current.alt ? " · S " + current.alt.label.toLowerCase() : "");
  $("mark").style.left = (current.length ? 100 * current.lead / current.length : 0) + "%";
  $("fill").style.width = "0%";
  const sheets = current.sheets || {};
  $("sheets").innerHTML = ["A", "B"].filter(n => sheets[n]).map(n => sheetHtml(n, sheets[n])).join("");
  for (const n of Object.keys(sheets)) $("score-" + n).onclick = e => seek(n, e);
  $("legend").textContent = current.legend || "";
  at = 0; from = 0; place(0);
  $("picks").innerHTML = PICKS.map(([p, label]) => `<button class="pick" data-pick="${p}">${label}</button>`).join("");
  for (const b of document.querySelectorAll("#picks button")) b.onclick = () => choose(b.dataset.pick);
  $("note").value = current.note || "";
  $("back").disabled = !s.previous;
  $("skip").classList.toggle("hidden", !!current.pick);
  reveal(current.pick ? current : null);
  load();
}
function reveal(done) {
  for (const b of document.querySelectorAll("#picks button")) b.classList.toggle("chosen", !!done && b.dataset.pick === done.pick);
  $("reveal").classList.toggle("hidden", !done);
  $("next").classList.toggle("hidden", !done);
  if (done) $("reveal").innerHTML = `<div><b>A</b> was ${esc(done.reveal.A)}</div><div><b>B</b> was ${esc(done.reveal.B)}</div><div class="note">Saved. Pick again to change it.</div>`;
  whyRow();
}
// After "neither is right": what is wrong, in one click. Each reason is a phrase in the note.
const said = () => $("note").value.split(",").map(s => s.trim()).filter(Boolean);
function whyRow() {
  const on = !!current && current.pick === "neither";
  $("why").classList.toggle("hidden", !on);
  if (!on) return;
  const now = said();
  $("why").innerHTML = `<span class="note">What is wrong?</span>` + WHY.map(w => `<button class="chip${now.includes(w) ? " chosen" : ""}" data-why="${w}">${w}</button>`).join("");
  for (const b of document.querySelectorAll("#why button")) b.onclick = () => toggleWhy(b.dataset.why);
}
function toggleWhy(w) {
  const now = said(), k = now.indexOf(w);
  if (k >= 0) now.splice(k, 1); else now.push(w);
  $("note").value = now.join(", ");
  whyRow();
  return saveNote();
}
// A note written or changed after the pick is saved too (the pick is sent again with it).
async function saveNote() {
  if (!current || !current.pick || $("note").value === (current.note || "")) return;
  const id = current.id, pick = current.pick, note = $("note").value;
  try { await api("/api/judge", {id, pick, note}); if (current && current.id === id) current.note = note; }
  catch (e) { $("error").textContent = e.message; }
}
async function choose(pick) {
  try {
    const s = await api("/api/judge", {id: current.id, pick, note: $("note").value});
    state = s; current.pick = pick; current.reveal = s.reveal; current.note = $("note").value;
    $("progress").textContent = `${s.judged} of ${s.total} judged`;
    stepsBar(s);
    $("skip").classList.add("hidden");
    reveal(current);
  } catch (e) { $("error").textContent = e.message; }
}
async function go(query) {
  await saveNote();
  try { show(await api("/api/next" + (query || "")), query); }
  catch (e) { $("error").textContent = e.message; }
}
const after = () => "?after=" + encodeURIComponent(current.id);
async function results() {
  const t = await api("/api/results");
  let html = "<h2>Picks so far</h2>";
  if (!Object.keys(t).length) html += "<p>No comparisons yet.</p>";
  const names = Object.fromEntries(state.steps.map(st => [st.id, st.name]));
  for (const [id, row] of Object.entries(t)) {
    html += `<h3>${esc(names[id] || id)}</h3><table>`;
    for (const [value, n] of Object.entries(row.picks)) html += `<tr><td>${esc(row.labels[value] || value)}</td><td class="n">${n}</td></tr>`;
    html += `<tr><td>No difference</td><td class="n">${row.same}</td></tr><tr><td>Neither is right</td><td class="n">${row.neither}</td></tr>`;
    html += `<tr><td class="note">Not judged yet</td><td class="n note">${row.open}</td></tr></table>`;
    for (const n of row.notes) html += `<p class="note">${esc(n.song)}: ${esc(n.note)}</p>`;
  }
  $("results").innerHTML = html;
}
$("stop").onclick = stop;
$("solo").onchange = () => { const was = playing, t = was ? audio[was].currentTime : 0; load(); if (was) { audio[was].currentTime = t; playing = was; audio[was].play(); paint(); } };
$("bar").onclick = e => {
  if (!current.length) return;
  const t = current.length * e.offsetX / $("bar").clientWidth;
  if (playing) audio[playing].currentTime = t; else { at = from = t; place(t); $("fill").style.width = (100 * t / current.length) + "%"; }
};
$("note").onchange = saveNote;
$("next").onclick = () => go(after());
$("back").onclick = () => state.previous && go("?id=" + encodeURIComponent(state.previous));
$("skip").onclick = () => go(after());
$("toggleResults").onclick = async () => {
  const open = $("results").classList.toggle("hidden");
  $("toggleResults").textContent = open ? "Picks so far" : "Back to judging";
  if (open) { show(state); return; }
  stop();
  for (const id of ["judging", "waiting", "done"]) $(id).classList.add("hidden");
  await results();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || !current) return;
  const k = e.key.toLowerCase(), hit = current.sources.find(s => s[2] === k);
  if (hit) { play(hit[0]); e.preventDefault(); }
  else if (k === " ") { playing ? stop() : play(current.sources[0][0]); e.preventDefault(); }
  else if (k === "s" && current.alt) { $("solo").checked = !$("solo").checked; $("solo").onchange(); }
});
requestAnimationFrame(tick);
go();
</script>
</body>
</html>
"""
