"""The judging page: a local web server on the standard library, for one person on their own machine.

The page never learns which version is which until a pick is recorded: the server hands it audio and score
snippets named A and B, and reveals what they were only in the answer to the pick.
"""

import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import judge
from .common import Bass2SheetError

mmss = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"


def public(comp, judgment=None):
    """What the page may know about a comparison before the pick: nothing that tells the versions apart."""
    ch = judge.choice_of(comp["choice"])
    out = dict(
        id=comp["id"],
        title=comp["title"],
        artist=comp["artist"],
        where=f"bars {comp['bars_a'][0]}–{comp['bars_a'][1]} · {mmss(comp['t0'])}–{mmss(comp['t1'])}",
        about=ch["about"],
        ask=ch["ask"],
        kind=comp["kind"],
        same_audio=comp.get("same_audio", False),
        lead=comp.get("lead", 0.0),
        length=comp.get("length", 0.0),
        media={k: "/" + v for k, v in comp.get("media", {}).items()},
        sheets={name: dict(sheet, src="/" + sheet["src"]) for name, sheet in comp.get("sheets", {}).items()},
    )
    if judgment:  # judged before: the page shows the earlier pick and what the versions were
        out.update(pick=judgment["pick"], note=judgment["note"], reveal=judge.shown_labels(comp))
    return out


def state(workdir, want=None, after=None):
    """Progress and the comparison to show: the one asked for; else the first not judged yet that follows
    `after` in the order, wrapping round to the ones skipped earlier."""
    comps = judge.order(judge.load_comparisons(workdir))
    judged = judge.load_judgments(workdir)
    ids = [c["id"] for c in comps]
    comp = next((c for c in comps if c["id"] == want), None) if want else None
    if comp is None:
        start = ids.index(after) + 1 if after in ids else 0
        comp = next((c for c in comps[start:] + comps[:start] if c["id"] not in judged), None)
    k = ids.index(comp["id"]) if comp else len(ids)
    return dict(
        total=len(comps),
        judged=sum(1 for i in ids if i in judged),
        comparison=public(comp, judged.get(comp["id"])) if comp else None,
        previous=ids[k - 1] if k > 0 else None,
    )


class Handler(BaseHTTPRequestHandler):
    workdir = "."
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
            q = parse_qs(url.query)
            self.send_json(state(self.workdir, q.get("id", [None])[0], q.get("after", [None])[0]))
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
        self.send_json(dict(reveal=judge.shown_labels(comp), **state(self.workdir, comp["id"])))


def make_server(workdir, port=8765):
    if not os.path.exists(os.path.join(workdir, "comparisons.json")):
        raise Bass2SheetError(
            f"no comparisons in {workdir}: run `bass2sheet-judge build SONG... --dir {workdir}` first"
        )
    handler = type("JudgeHandler", (Handler,), dict(workdir=os.path.abspath(workdir)))
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise Bass2SheetError(f"cannot listen on port {port} ({e}); pick another with --port") from e


def serve(workdir, port=8765):
    server = make_server(workdir, port)
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
<style>
  :root { --ink:#1c1f24; --soft:#5d6570; --line:#d9dde3; --bg:#f6f7f9; --card:#fff; --go:#1f6feb; --on:#0b5d2a; --warn:#9a6700; }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }
  header { display:flex; gap:16px; align-items:baseline; padding:14px 24px; border-bottom:1px solid var(--line); background:var(--card); }
  header h1 { font-size:17px; margin:0; }
  header .grow { flex:1; }
  header button { font:inherit; background:none; border:0; color:var(--go); cursor:pointer; padding:0; }
  main { max-width:1180px; margin:20px auto 60px; padding:0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px 22px; margin-bottom:16px; }
  h2 { font-size:22px; margin:0 0 2px; }
  .where { color:var(--soft); margin-bottom:12px; }
  .about { margin:0 0 6px; }
  .ask { font-weight:600; margin:0; }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  button.play, button.pick, button.next { font:inherit; font-weight:600; border-radius:8px; border:1px solid var(--line); background:#fff; padding:10px 18px; cursor:pointer; }
  button.play.on { background:var(--on); color:#fff; border-color:var(--on); }
  button.pick { min-width:150px; }
  button.pick.chosen { background:var(--go); color:#fff; border-color:var(--go); }
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
  .score .mark { position:absolute; background:rgba(255,193,7,.30); border-radius:3px; pointer-events:none; }
  .score .cursor { position:absolute; width:2px; margin-left:-1px; background:var(--go); box-shadow:0 0 0 5px rgba(31,111,235,.16); pointer-events:none; }
  .legend { color:var(--soft); font-size:13px; margin-top:8px; }
  .legend i { display:inline-block; width:14px; height:12px; background:rgba(255,193,7,.45); border-radius:3px; vertical-align:-1px; margin-right:4px; }
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
<main>
  <div id="judging">
    <div class="card" id="what">
      <h2 id="title"></h2>
      <div class="where" id="where"></div>
      <p class="about" id="about"></p>
      <p class="ask" id="ask"></p>
    </div>
    <div class="card" id="listen">
      <div class="row" id="players"></div>
      <div class="row" style="margin-top:12px">
        <label class="opt" id="soloOpt"><input type="checkbox" id="solo"> Bass only</label>
        <label class="opt"><input type="checkbox" id="loop" checked> Loop</label>
        <button class="play" id="stop">Stop</button>
      </div>
      <div class="bar" id="bar"><div class="fill" id="fill"></div><div class="mark" id="mark"></div></div>
      <div class="keys" id="keys"></div>
      <div class="sheet" id="sheets"></div>
      <div class="legend hidden" id="legend"><i></i>Shaded notes are the ones the two versions write differently. Click a score to play from there.</div>
    </div>
    <div class="card" id="decide">
      <div class="row" id="picks"></div>
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
  <div id="done" class="card hidden">
    <h2>Everything is judged</h2>
    <p>Your picks are saved. They are listed under “Picks so far”.</p>
  </div>
  <div id="results" class="card hidden"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let current = null, state = null, audio = {}, playing = null, at = 0, from = 0;
const PICKS = [["A", "A is better"], ["B", "B is better"], ["same", "No difference"], ["neither", "Neither is right"]];

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
async function api(path, body) {
  const r = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function sources() {
  if (!current) return [];
  const m = current.media;
  if (current.same_audio) return [["record", "Record", "r"], ["A", "Transcription", "a"]].filter(s => m[s[0]]);
  return [["record", "Record", "r"], ["A", "A", "a"], ["B", "B", "b"]].filter(s => m[s[0]]);
}
function url(name) {
  const m = current.media, solo = $("solo").checked && m[name + "_bass"];
  return solo ? m[name + "_bass"] : m[name];
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
function sheetHtml(name, s) {
  const [x0, y0, x1, y1] = s.crop, w = x1 - x0, h = y1 - y0, pct = v => (100 * v) + "%";
  const top = pct((s.staff[0] - y0) / h), height = pct((s.staff[1] - s.staff[0]) / h);
  const marks = s.marks.map(x => `<div class="mark" style="left:${pct((x - 11 - x0) / w)};width:${pct(22 / w)};top:${top};height:${height}"></div>`).join("");
  return `<h3>${name}</h3><div class="score" id="score-${name}" style="aspect-ratio:${w}/${h}">` +
    `<img alt="Score ${name}" src="${s.src}" draggable="false" style="left:${pct(-x0 / w)};top:${pct(-y0 / h)};width:${pct(s.page[0] / w)}">` +
    marks + `<div class="cursor" id="cursor-${name}" style="top:${top};height:${height}"></div></div>`;
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
  for (const [name] of sources()) {
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
function show(s) {
  stop();
  state = s; current = s.comparison;
  $("progress").textContent = s.total ? `${s.judged} of ${s.total} judged` : "nothing to judge yet";
  $("error").textContent = "";
  $("judging").classList.toggle("hidden", !current);
  $("done").classList.toggle("hidden", !!current);
  if (!current) { stop(); return; }
  $("title").textContent = (current.artist ? current.artist + " – " : "") + current.title;
  $("where").textContent = current.where;
  $("about").textContent = current.about;
  $("ask").textContent = current.ask;
  $("solo").checked = false;
  $("soloOpt").classList.toggle("hidden", !current.media.record_bass);
  $("players").innerHTML = sources().map(([n, label]) => `<button class="play" data-name="${n}">▶ ${label}</button>`).join("");
  for (const b of document.querySelectorAll("#players button")) b.onclick = () => play(b.dataset.name);
  $("keys").textContent = "Keys: " + sources().map(([n, label, k]) => `${k.toUpperCase()} ${label}`).join(" · ") + " · Space stop" + (current.media.record_bass ? " · S bass only" : "");
  $("mark").style.left = (current.length ? 100 * current.lead / current.length : 0) + "%";
  $("fill").style.width = "0%";
  const sheets = current.sheets || {};
  $("sheets").innerHTML = ["A", "B"].filter(n => sheets[n]).map(n => sheetHtml(n, sheets[n])).join("");
  for (const n of Object.keys(sheets)) $("score-" + n).onclick = e => seek(n, e);
  $("legend").classList.toggle("hidden", !Object.values(sheets).some(s => s.marks.length));
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
}
async function choose(pick) {
  try {
    const s = await api("/api/judge", {id: current.id, pick, note: $("note").value});
    state = s; current.pick = pick; current.reveal = s.reveal;
    $("progress").textContent = `${s.judged} of ${s.total} judged`;
    $("skip").classList.add("hidden");
    reveal(current);
  } catch (e) { $("error").textContent = e.message; }
}
async function go(query) {
  try { show(await api("/api/next" + (query || ""))); }
  catch (e) { $("error").textContent = e.message; }
}
const after = () => "?after=" + encodeURIComponent(current.id);
async function results() {
  const t = await api("/api/results");
  let html = "<h2>Picks so far</h2>";
  if (!Object.keys(t).length) html += "<p>No comparisons yet.</p>";
  for (const [id, row] of Object.entries(t)) {
    html += `<h3>${esc(id)}</h3><table>`;
    for (const [value, n] of Object.entries(row.picks)) html += `<tr><td>${esc(row.labels[value] || value)}</td><td class="n">${n}</td></tr>`;
    html += `<tr><td>No difference</td><td class="n">${row.same}</td></tr><tr><td>Neither is right</td><td class="n">${row.neither}</td></tr>`;
    html += `<tr><td class="note">Not judged yet</td><td class="n note">${row.open}</td></tr></table>`;
    for (const n of row.notes) html += `<p class="note">${esc(n.song)}: ${esc(n.note)}</p>`;
  }
  $("results").innerHTML = html;
}
$("stop").onclick = stop;
$("solo").onchange = () => { const was = playing, at = was ? audio[was].currentTime : 0; load(); if (was) { audio[was].currentTime = at; playing = was; audio[was].play(); paint(); } };
$("bar").onclick = e => {
  if (!current.length) return;
  const t = current.length * e.offsetX / $("bar").clientWidth;
  if (playing) audio[playing].currentTime = t; else { at = from = t; place(t); $("fill").style.width = (100 * t / current.length) + "%"; }
};
$("next").onclick = () => go(after());
$("back").onclick = () => state.previous && go("?id=" + encodeURIComponent(state.previous));
$("skip").onclick = () => go(after());
$("toggleResults").onclick = async () => {
  const open = $("results").classList.toggle("hidden");
  $("toggleResults").textContent = open ? "Picks so far" : "Back to judging";
  $("judging").classList.toggle("hidden", !open || !current);
  $("done").classList.toggle("hidden", !open || !!current);
  if (!open) { stop(); await results(); }
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || !current) return;
  const k = e.key.toLowerCase(), hit = sources().find(s => s[2] === k);
  if (hit) { play(hit[0]); e.preventDefault(); }
  else if (k === " ") { playing ? stop() : play(sources()[0][0]); e.preventDefault(); }
  else if (k === "s" && current.media.record_bass) { $("solo").checked = !$("solo").checked; $("solo").onchange(); }
});
requestAnimationFrame(tick);
go();
</script>
</body>
</html>
"""
