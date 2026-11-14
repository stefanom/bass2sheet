"""The copilot's pages: one person, one machine. The songs and the way in for a new one are at `/`
(copilot_landing.html); one song at a time is worked on at `/song` (copilot_page.html). See copilot.py for what is
served, and copilot_adding.py for how a new song comes in."""

import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import copilot, copilot_adding
from .common import Bass2SheetError

PAGE_PATH = os.path.join(os.path.dirname(__file__), "copilot_page.html")
LANDING_PATH = os.path.join(os.path.dirname(__file__), "copilot_landing.html")
SCORE_PATH = os.path.join(os.path.dirname(__file__), "copilot_score.html")
SCORE_PANEL_PATH = os.path.join(os.path.dirname(__file__), "copilot_score_panel.js")
PAGES = {"/": LANDING_PATH, "/song": PAGE_PATH, "/score": SCORE_PATH}
LOCAL = ("127.0.0.1", "localhost", "::1")  # where a page that may change things here is served from
SERVED = re.compile(
    r"^(audio/(record|bass)\.flac|picture\.bin|pitch\.bin|sheet/(v\d+/)?bass(\.pdf|\.musicxml|-\d+\.svg)"
    r"|sheet/(v\d+/)?bass_as_played\.mid)$"
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
    """The prepared songs, for the list: each with how far the person is with it (copilot.progress) and whose
    notes it started from."""
    out = []
    for song_id, song in copilot.load_songs(workdir).items():
        folder = copilot.song_dir(workdir, song_id)
        machine, mine = (copilot.read_json(os.path.join(folder, name)) for name in ("machine.json", "notes.json"))
        if machine is None or mine is None:
            continue
        far, made = copilot.progress(machine, mine), copilot.score_made(workdir, song_id)
        out.append(
            dict(
                id=song_id,
                title=song.get("title", song_id),
                artist=song.get("artist", ""),
                length=machine["length"],
                notes=len(mine["notes"]),
                checked=round(sum(b - a for a, b in mine["checked"]), 1),
                worked=round(sum(b - a for a, b in far["worked"]), 1),
                spans=dict(heard=mine["checked"], worked=far["worked"]),
                fixes=far["fixes"],
                finder=machine["settings"].get("notes", "own"),
                at=mine.get("at"),
                touched=os.path.getmtime(os.path.join(folder, "notes.json")),
                score={k: made[k] for k in ("when", "fixes", "checked")} if made else None,  # as it was last made
            )
        )
    return out


def score_of(workdir, song_id, version=None):
    """A song's score for the page that shows it: as it was last made, or the `version` asked for, with the
    addresses of its pages and files, and every version there is to choose from."""
    song = copilot.load_songs(workdir).get(song_id)
    if song is None or not os.path.exists(os.path.join(copilot.song_dir(workdir, song_id), "machine.json")):
        raise Bass2SheetError(f"no song {song_id}")
    from .score import find_mscore

    made = copilot.score_made(workdir, song_id, int(version) if version else None)
    if made is None and version:
        raise Bass2SheetError(f"{song_id} has no version {version} of its score")
    facts = ("version", "when", "notes", "checked", "fixes", "worked")
    out = dict(id=song_id, title=song.get("title", song_id), artist=song.get("artist", ""), made=None)
    out["mscore"] = find_mscore() is not None  # whether a score made now would be drawn
    out["versions"] = [{k: v[k] for k in facts} for v in copilot.score_versions(workdir, song_id)]
    if made:
        where = f"/data/{song_id}/sheet/" + (f"v{made['version']}/" if "version" in made else "")
        out["made"] = {k: made.get(k) for k in facts}
        out["files"] = {k: where + name for k, name in made["files"].items()}
        out["pages"] = [where + name for name in made["pages"]]
    return out


class Counted:
    """What is being read of a request, counting the bytes."""

    def __init__(self, stream):
        self.stream, self.count = stream, 0

    def read(self, n):
        block = self.stream.read(n)
        self.count += len(block)
        return block


class Handler(BaseHTTPRequestHandler):
    workdir = "."
    desks = None
    adding = None
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
            if url.path in PAGES:  # the songs; one song (an old link to /#song is sent on by the page); its score
                with open(PAGES[url.path], "rb") as f:
                    self.send_body(f.read(), "text/html; charset=utf-8")
            elif url.path == "/score-panel.js":  # what a song's page needs to show the score over itself
                with open(SCORE_PANEL_PATH, "rb") as f:
                    self.send_body(f.read(), "text/javascript; charset=utf-8")
            elif url.path == "/api/score":
                self.send_json(score_of(self.workdir, q.get("song", ""), q.get("version")))
            elif url.path == "/api/songs":
                self.send_json(songs(self.workdir))
            elif url.path == "/api/adding":
                self.send_json(self.adding_now())
            elif url.path == "/api/strip":
                pic = copilot_adding.strip(self.workdir, q.get("song", ""), int(q.get("columns", "600")))
                if pic is None or q.get("song") not in copilot.load_songs(self.workdir):
                    return self.send_json(dict(error="not found"), 404)
                size = {"X-Rows": str(pic.shape[0]), "X-Columns": str(pic.shape[1])}
                self.send_body(pic.tobytes(), "application/octet-stream", extra=size)
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
        except (Bass2SheetError, StopIteration, ValueError) as e:
            self.send_json(dict(error=str(e) or "no such note"), 404)

    do_HEAD = do_GET

    def adding_now(self):
        """What the landing page shows of the way in: the songs being added, and what a new one may be."""
        return dict(
            jobs=self.adding.listing(),
            finders=copilot_adding.finders(),
            links=copilot_adding.downloader() is not None,
            kinds=copilot_adding.AUDIO,
            limit=copilot_adding.MAX_BYTES,
        )

    def from_here(self):
        """Whether a request that changes things comes from one of this server's own pages. A page of another
        site, open in the same browser, can send requests here too; the browser says where they come from."""
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).hostname in LOCAL

    def upload(self, q):
        """A new song: the file is the body of the request, its name and the note finder are in the address."""
        length = self.headers.get("Content-Length")
        length = int(length) if length and length.isdigit() else None
        body = Counted(self.rfile)
        try:
            job = self.adding.accept(q.get("name", ""), length, body, q.get("notes") or "own")
        except copilot_adding.Refused as e:
            # The browser hears the answer only once it has sent the whole file: the rest of a file of a
            # reasonable size is read and dropped; one that is too large is not read, and the line is cut.
            left = (length or 0) - body.count
            while 0 < left <= copilot_adding.MAX_BYTES and (block := self.rfile.read(min(left, 1 << 20))):
                left -= len(block)
            self.close_connection = True
            return self.send_json(dict(error=str(e)), e.status)
        self.send_json(dict(id=job["id"], title=job["title"]))

    def warm(self, desk):
        """Load what the first guess needs while the page is still loading the sound."""
        with self.desks.lock:
            try:
                if desk.notes:
                    desk.guesses(desk.notes[len(desk.notes) // 2]["id"])
            except (Bass2SheetError, OSError, ValueError):
                pass

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        if not self.from_here():
            self.close_connection = True
            return self.send_json(dict(error="this copilot only takes changes from its own pages"), 403)
        try:
            if path == "/api/upload":
                return self.upload({k: v[0] for k, v in parse_qs(url.query).items()})
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if path == "/api/link":  # a new song that comes as a link: fetched when its turn comes
                job = self.adding.link(data.get("url"), data.get("notes") or "own")
                return self.send_json(dict(key=job["key"]))
            if path == "/api/adding":  # a song being added: try it again, or take it out of the row
                getattr(self.adding, dict(again="again", remove="remove")[data.get("do")])(data.get("id", ""))
                return self.send_json(self.adding_now())
            with self.desks.lock:
                desk = self.desks.get(data.get("song", ""))
                if path == "/api/act":
                    result = desk.act(data.get("action") or {})
                elif path == "/api/checked":
                    desk.mark_checked(float(data["start"]), float(data["end"]))
                    played = dict(checked=desk.state["checked"], worked=desk.worked())
                    return self.send_json(dict(played, differences=desk.differences()))
                elif path == "/api/sheet":
                    files = copilot.sheet(self.workdir, desk.song_id)
                    folder = copilot.song_dir(self.workdir, desk.song_id)
                    rel = {k: f"/data/{desk.song_id}/{os.path.relpath(v, folder)}" for k, v in files.items()}
                    return self.send_json(rel)
                else:
                    return self.send_json(dict(error="not found"), 404)
                state = desk.public()
            fields = {k: state[k] for k in ("notes", "checked", "worked", "undo", "fixes")}
            self.send_json(dict(result=result, differences=state["repeats"]["differences"], **fields))
        except copilot_adding.Refused as e:
            self.send_json(dict(error=str(e)), e.status)
        except (ValueError, KeyError, TypeError, StopIteration, Bass2SheetError) as e:
            self.send_json(dict(error=str(e) or type(e).__name__), 400)


def make_server(workdir, port=8770, stems_dir=None):
    """The server for a working directory, which may hold no song yet: the first one can be added on the page.
    `stems_dir` is for tests (copilot_adding.Adding)."""
    workdir = os.path.abspath(workdir)
    if os.path.exists(workdir) and not os.path.isdir(workdir):
        raise Bass2SheetError(f"{workdir} is not a directory")
    adding = copilot_adding.Adding(workdir, stems_dir)
    handler = type("CopilotHandler", (Handler,), dict(workdir=workdir, desks=Desks(workdir), adding=adding))
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise Bass2SheetError(f"cannot listen on port {port} ({e}); pick another with --port") from e


def serve(workdir, port=8770):
    server = make_server(workdir, port)
    n = len(songs(server.RequestHandlerClass.workdir))
    print(f"{n} {'song' if n == 1 else 'songs'}. Open http://localhost:{port}", flush=True)
    if not n:
        print("Add your first song on the page, or with `bass2sheet-copilot add SONG`.", flush=True)
    print("Stop with Ctrl-C; your fixes are saved as you make them.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.RequestHandlerClass.adding.close()
        server.server_close()
