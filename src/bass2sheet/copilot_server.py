"""The copilot's page: one person, one machine, one song at a time. See copilot.py for what it serves."""

import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import copilot
from .common import Bass2SheetError

PAGE_PATH = os.path.join(os.path.dirname(__file__), "copilot_page.html")
SERVED = re.compile(
    r"^(audio/(record|bass)\.flac|picture\.bin|pitch\.bin|sheet/bass\.(pdf|musicxml)|sheet/bass_as_played\.mid)$"
)


SLOWED = re.compile(r"^audio/(record|bass)-(\d+)\.flac$")


class Desks:
    """The songs that are open, at most two at a time (each holds its bass track), and one lock for all of them:
    a single person works here, and their fixes are applied in the order they were made."""

    def __init__(self, workdir):
        self.workdir, self.open, self.lock = workdir, {}, threading.RLock()

    def get(self, song_id):
        if song_id not in copilot.load_songs(self.workdir):
            raise Bass2SheetError(f"no song {song_id}")
        if song_id not in self.open:
            if len(self.open) >= 2:
                self.open.pop(next(iter(self.open)))
            self.open[song_id] = copilot.Desk(self.workdir, song_id)
        return self.open[song_id]


def songs(workdir):
    out = []
    for song_id, song in copilot.load_songs(workdir).items():
        folder = copilot.song_dir(workdir, song_id)
        machine, mine = (copilot.read_json(os.path.join(folder, name)) for name in ("machine.json", "notes.json"))
        if machine is None or mine is None:
            continue
        out.append(
            dict(
                id=song_id,
                title=song.get("title", song_id),
                artist=song.get("artist", ""),
                length=machine["length"],
                notes=len(mine["notes"]),
                checked=round(sum(b - a for a, b in mine["checked"]), 1),
            )
        )
    return out


class Handler(BaseHTTPRequestHandler):
    workdir = "."
    desks = None
    server_version = "bass2sheet-copilot"

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

    def send_file(self, song_id, rel):
        """A file of a song's folder, with byte ranges (Safari will not play audio without them)."""
        slow = SLOWED.match(rel)
        if song_id not in copilot.load_songs(self.workdir) or not (SERVED.match(rel) or slow):
            return self.send_json(dict(error="not found"), 404)
        if slow:  # the song at a slower speed: made when it is first asked for
            if int(slow.group(2)) not in copilot.SPEEDS:
                return self.send_json(dict(error="not found"), 404)
            try:
                copilot.slowed(self.workdir, song_id, slow.group(1), int(slow.group(2)))
            except (Bass2SheetError, OSError) as e:
                return self.send_json(dict(error=str(e)), 500)
        path = os.path.join(copilot.song_dir(self.workdir, song_id), rel)
        if not os.path.isfile(path):
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
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                with open(PAGE_PATH, "rb") as f:
                    self.send_body(f.read(), "text/html; charset=utf-8")
            elif url.path == "/api/songs":
                self.send_json(songs(self.workdir))
            elif url.path == "/api/song":
                with self.desks.lock:
                    desk = self.desks.get(q.get("id", ""))
                    self.send_json(desk.public())
                threading.Thread(target=self.warm, args=(desk,), daemon=True).start()
            elif url.path == "/api/guesses":
                with self.desks.lock:
                    self.send_json(dict(guesses=self.desks.get(q.get("song", "")).guesses(q.get("id", ""))))
            elif url.path.startswith("/data/"):
                song_id, _, rel = unquote(url.path[len("/data/") :]).partition("/")
                self.send_file(song_id, rel)
            else:
                self.send_json(dict(error="not found"), 404)
        except (Bass2SheetError, StopIteration) as e:
            self.send_json(dict(error=str(e) or "no such note"), 404)

    do_HEAD = do_GET

    def warm(self, desk):
        """Load what the first guess needs while the page is still loading the sound."""
        with self.desks.lock:
            try:
                if desk.notes:
                    desk.guesses(desk.notes[len(desk.notes) // 2]["id"])
            except (Bass2SheetError, OSError, ValueError):
                pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            with self.desks.lock:
                desk = self.desks.get(data.get("song", ""))
                if path == "/api/act":
                    result = desk.act(data.get("action") or {})
                elif path == "/api/checked":
                    desk.mark_checked(float(data["start"]), float(data["end"]))
                    return self.send_json(dict(checked=desk.state["checked"]))
                elif path == "/api/sheet":
                    files = copilot.sheet(self.workdir, desk.song_id)
                    folder = copilot.song_dir(self.workdir, desk.song_id)
                    rel = {k: f"/data/{desk.song_id}/{os.path.relpath(v, folder)}" for k, v in files.items()}
                    return self.send_json(rel)
                else:
                    return self.send_json(dict(error="not found"), 404)
                state = desk.public()
            self.send_json(dict(result=result, **{k: state[k] for k in ("notes", "checked", "undo", "fixes")}))
        except (ValueError, KeyError, TypeError, StopIteration, Bass2SheetError) as e:
            self.send_json(dict(error=str(e) or type(e).__name__), 400)


def make_server(workdir, port=8770):
    workdir = os.path.abspath(workdir)
    if not songs(workdir):
        raise Bass2SheetError(f"no prepared songs in {workdir}: run `bass2sheet-copilot add SONG... --dir {workdir}`")
    handler = type("CopilotHandler", (Handler,), dict(workdir=workdir, desks=Desks(workdir)))
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise Bass2SheetError(f"cannot listen on port {port} ({e}); pick another with --port") from e


def serve(workdir, port=8770):
    server = make_server(workdir, port)
    print(f"{len(songs(workdir))} songs. Open http://localhost:{port}", flush=True)
    print("Stop with Ctrl-C; your fixes are saved as you make them.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
