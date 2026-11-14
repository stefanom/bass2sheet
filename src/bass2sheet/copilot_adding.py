"""New songs for the copilot, from its page.

A file comes in from the browser, is kept under the working directory (`uploads/`) and is prepared with the code
`bass2sheet-copilot add` runs: one song at a time, each in a process of its own, so the memory the models take
goes back to the system when a song is done and a crash there does not take the page down. Where a downloader
(yt-dlp) is installed, a link to a video is taken as well: its audio is fetched to the same place first. What
the preparation reports as it goes (`common.step`) is kept per song for the page: which step is running, which
are done, how long each took and what it found.

Nothing here listens on the network; `copilot_server` does, on this machine only.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse

import numpy as np

from . import copilot, judge
from .audio import MAX_SECONDS, MIN_SECONDS
from .common import Bass2SheetError, atomic_write_text, clock, listening
from .pipeline import NOTE_FINDERS

# What is taken: audio files, and the video files a song often comes in, for the sound in them
AUDIO = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".mp4", ".webm", ".mkv")
MAX_BYTES = 500 * 1024 * 1024  # 20 minutes, the longest song the tool takes, is 212 MB as a CD-quality WAV
MARK = "@@copilot "  # what a line of the preparing process starts with when it is meant for the page
FINDERS = dict(own="bass2sheet's own note finder", muscriptor="MuScriptor")
LINK_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be")
FETCHING = ("fetching", "Fetching the audio")  # the step before all others, for a song that comes as a link
FETCH_TIMEOUT = 600  # s
# The steps of preparing a song, in the order they run: (what the code calls it, what the page calls it, which
# note finders it is part of).
STEPS = (
    ("reading", "Reading the file", NOTE_FINDERS),
    ("isolating", "Isolating the bass", NOTE_FINDERS),
    ("pitch", "Following the pitch", NOTE_FINDERS),
    ("notes", "Finding the notes", NOTE_FINDERS),
    ("muscriptor", "MuScriptor's notes", ("muscriptor",)),
    ("second", "Second opinion", ("muscriptor",)),
    ("beats", "Beats and bars", NOTE_FINDERS),
    ("checks", "Own-sound check", NOTE_FINDERS),
    ("picture", "The picture", NOTE_FINDERS),
    ("ready", "Ready", NOTE_FINDERS),
)


class Refused(Bass2SheetError):
    """A file the page offered and the copilot does not take; `status` is the HTTP answer."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def safe_name(name):
    """A file name as it is kept: the last part of what the browser sent, without anything that is not a name."""
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = "".join(c for c in name if c.isprintable() and c not in '<>:"|?*').strip(" .")
    stem, ext = os.path.splitext(name)
    return stem[:150].rstrip() + ext.lower() if stem else ""


def megabytes(n):
    return f"{n / 1024 / 1024:.0f} MB"


def probe(path, name):
    """How long the audio in a file lasts, in seconds (None when the file does not say). Refused when the file
    holds no audio that can be read, or audio the tool does not take: under ten seconds, over twenty minutes."""
    import av

    try:
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise Refused(415, f"{name} has no sound in it")
            next(container.decode(stream))  # it opens as audio and the first piece of it decodes
            if stream.duration and stream.time_base:
                seconds = float(stream.duration * stream.time_base)
            else:
                seconds = container.duration / 1e6 if container.duration else None
    except Refused:
        raise
    except Exception as e:  # PyAV raises several kinds of error for what it cannot read; StopIteration: no audio
        raise Refused(415, f"{name} is not an audio file this tool can read") from e
    if seconds is not None and seconds < MIN_SECONDS:
        raise Refused(400, f"{name} lasts {seconds:.0f} s; a song needs at least {MIN_SECONDS:.0f} s")
    if seconds is not None and seconds > MAX_SECONDS:
        raise Refused(
            413, f"{name} lasts {clock(seconds)[:-2]}; the tool takes songs up to {MAX_SECONDS / 60:.0f} minutes"
        )
    return seconds


def downloader():
    """The program that fetches the audio a link points to (yt-dlp), or None where there is none; the page then
    has no box for links. BASS2SHEET_DOWNLOADER names the program where it is not on the PATH; set to `none`,
    links are not taken even where it is installed."""
    named = os.environ.get("BASS2SHEET_DOWNLOADER")
    if named is not None:
        return named if os.path.isfile(named) and os.access(named, os.X_OK) else None
    for path in (shutil.which("yt-dlp"), "/opt/homebrew/bin/yt-dlp", "/usr/local/bin/yt-dlp"):
        if path and os.path.isfile(path):
            return path
    return None


def why_not(said):
    """Why the downloader fetched nothing, from what it printed."""
    lines = [line.strip() for line in said.splitlines() if line.strip()]
    if any("does not pass filter" in line for line in lines):
        return f"it is a live stream, or lasts more than {MAX_SECONDS / 60:.0f} minutes"
    if any("larger than max-filesize" in line for line in lines):
        return f"its audio is larger than {megabytes(MAX_BYTES)}"
    errors = [line[len("ERROR:") :].strip() for line in lines if line.startswith("ERROR:")]
    return errors[-1] if errors else lines[-1] if lines else "the downloader did not say why"


def finders():
    """The note finders a new song can start from, for the page: MuScriptor only where it is set up."""
    from . import muscriptor

    out = [dict(id="own", label=FINDERS["own"], ready=True, why="")]
    try:
        muscriptor.program()
        out.append(dict(id="muscriptor", label=FINDERS["muscriptor"], ready=True, why=""))
    except Bass2SheetError:
        why = "not set up on this machine: see docs/muscriptor.md"
        out.append(dict(id="muscriptor", label=FINDERS["muscriptor"], ready=False, why=why))
    return out


def strip(workdir, song_id, columns):
    """A song's pitch picture squeezed to `columns` across, the whole song at a glance: (rows, columns) bytes,
    each the darkest of the picture's columns it stands for. None when the song has no picture yet."""
    path = os.path.join(copilot.song_dir(workdir, song_id), "picture.bin")
    rows = copilot.PER_SEMI * (copilot.HIGH - copilot.LOW + 1)
    if not os.path.isfile(path) or os.path.getsize(path) < rows or os.path.getsize(path) % rows:
        return None
    pic = np.memmap(path, dtype=np.uint8, mode="r").reshape(rows, -1)
    columns = max(1, min(int(columns), pic.shape[1]))
    edges = (np.arange(columns) * pic.shape[1]) // columns
    return np.maximum.reduceat(pic, edges, axis=1)


def new_steps(finder, first=()):
    steps = [*first, *((sid, label) for sid, label, part_of in STEPS if finder in part_of)]
    return [dict(id=sid, label=label, state="waiting", seconds=None, started=None, found={}) for sid, label in steps]


class Adding:
    """The songs being added, in the order they came. One is prepared at a time, by one worker thread; the same
    thread does everything that writes `songs.json` from here, so nothing is written from two sides at once."""

    def __init__(self, workdir, stems_dir=None):
        self.workdir, self.uploads = workdir, os.path.join(workdir, "uploads")
        self.stems_dir = stems_dir  # for tests: every new song's stems, already separated
        self.jobs, self.arriving, self.links, self.lock = {}, set(), 0, threading.RLock()  # jobs: by their `key`
        self.tasks, self.child, self.worker, self.closed = queue.Queue(), None, None, False
        self._interrupted()

    # ---- what the page asks

    def listing(self):
        now = time.time()
        with self.lock:
            out = []
            for job in self.jobs.values():
                steps = [
                    dict(
                        {k: s[k] for k in ("id", "label", "state", "seconds", "found")},
                        elapsed=round(now - s["started"], 1) if s["state"] == "running" else None,
                    )
                    for s in job["steps"]
                ]
                shown = ("key", "id", "title", "artist", "name", "url", "finder", "state", "error", "said")
                shown = {k: job[k] for k in shown}
                out.append(dict(shown, steps=steps))
            return out

    def accept(self, name, length, stream, finder="own"):
        """Take a file the page sends: `length` bytes from `stream`, to be kept as `name`. Refused (with the HTTP
        status to answer with) unless it is an audio file of a reasonable size that is not already here. Returns
        the song's job, which is waiting its turn to be prepared."""
        kept = safe_name(name)
        if os.path.splitext(kept)[1] not in AUDIO:
            raise Refused(415, f"{kept or 'that'} is not an audio file: the copilot takes {' '.join(AUDIO)}")
        self._finder(finder)
        if length is None:
            raise Refused(411, "the file came without its size")
        if length <= 0:
            raise Refused(400, f"{kept} is empty")
        if length > MAX_BYTES:
            raise Refused(413, f"{kept} is {megabytes(length)}: the copilot takes files up to {megabytes(MAX_BYTES)}")
        path = os.path.join(self.uploads, kept)
        entry = judge.song_entry(path)
        with self.lock:
            self._free(entry)
            self.arriving.add(entry["id"])
        try:
            os.makedirs(self.uploads, exist_ok=True)
            with open(path + ".part", "wb") as f:
                left = length
                while left > 0:
                    block = stream.read(min(left, 1 << 20))
                    if not block:
                        raise Refused(400, f"{kept} did not arrive whole")
                    f.write(block)
                    left -= len(block)
            probe(path + ".part", kept)
            os.replace(path + ".part", path)
        except BaseException:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
            raise
        finally:
            with self.lock:
                self.arriving.discard(entry["id"])
        return self._queue(entry, kept, finder)

    def link(self, url, finder="own"):
        """Take a link to a video: its audio is fetched when the song's turn comes, then prepared like a file
        that was sent. Refused where no downloader is installed, and for what is not a link to YouTube."""
        url = (url or "").strip()
        if downloader() is None:
            raise Refused(400, "links are not taken here: no downloader (yt-dlp) is installed on this computer")
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in LINK_HOSTS:
            raise Refused(400, "that is not a link to YouTube")
        self._finder(finder)
        with self.lock:
            if any(j["url"] == url and j["state"] in ("waiting", "running") for j in self.jobs.values()):
                raise Refused(409, "that link is already being fetched")
            self.links += 1
            job = self._job(
                f"link-{self.links}", dict(id=None, title="A song from YouTube", artist=""), "", finder, url
            )
            self.jobs[job["key"]] = job
        self._task("prepare", job)
        return job

    def _finder(self, finder):
        ready = {f["id"]: f for f in finders()}
        if finder not in ready:
            raise Refused(400, f"unknown note finder {finder!r}")
        if not ready[finder]["ready"]:
            raise Refused(400, f"{ready[finder]['label']} is {ready[finder]['why']}")

    def _free(self, entry, but=None):
        """Refuse a song that is already here, or on its way (`but`: the job that is this song)."""
        if os.path.exists(os.path.join(copilot.song_dir(self.workdir, entry["id"]), "machine.json")):
            raise Refused(409, f"{entry['title']} is already among your songs")
        busy = (j for j in self.jobs.values() if j is not but and j["id"] == entry["id"])
        if entry["id"] in self.arriving or any(j["state"] in ("waiting", "running") for j in busy):
            raise Refused(409, f"{entry['title']} is already being prepared")

    def _job(self, key, entry, name, finder, url=None):
        return dict(
            key=key,
            id=entry["id"],
            title=entry["title"],
            artist=entry["artist"],
            name=name,
            path=entry.get("path"),
            url=url,
            finder=finder,
            state="waiting",
            error=None,
            said="",
            removed=False,
            steps=new_steps(finder, (FETCHING,) if url else ()),
        )

    def _queue(self, entry, name, finder):
        job = self._job(entry["id"], entry, name, finder)
        with self.lock:
            for key in [k for k, j in self.jobs.items() if j["id"] == job["id"]]:
                del self.jobs[key]  # one that failed before: the song goes to the end of the row
            self.jobs[job["key"]] = job
        self._task("prepare", job)
        return job

    def again(self, key):
        """Prepare once more a song whose preparation failed, from the file that is kept; one that came as a
        link and was not fetched is fetched again."""
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job["state"] != "failed":
                raise Bass2SheetError("there is nothing to try again")
            if not job["url"] and not os.path.isfile(job["path"]):
                raise Bass2SheetError(f"{job['name']} is not kept any more: add it again")
            first = (FETCHING,) if job["url"] else ()
            job.update(state="waiting", error=None, said="", steps=new_steps(job["finder"], first))
        self._task("prepare", job)

    def remove(self, key):
        """Take a song out of the row. One that is ready stays among the songs; one that is waiting, running or
        failed is given up: its file and whatever was made of it are deleted."""
        with self.lock:
            job = self.jobs.get(key)
            if job is None:
                raise Bass2SheetError("there is no such song being added")
            if job["state"] == "done":
                del self.jobs[key]
                return
            job["removed"] = True
            running = job["state"] == "running"
            if running and self.child is not None:
                self.child.terminate()  # the worker clears up when the process has gone
            elif not running:
                del self.jobs[key]
        if not running:
            self._task("clear", job)

    def close(self):
        """Stop: nothing more is prepared, and what is being prepared is given up as it stands."""
        self.closed = True
        with self.lock:
            if self.child is not None:
                self.child.terminate()
        self.tasks.put(("stop", None))

    # ---- the worker

    def _task(self, kind, job):
        with self.lock:
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._work, daemon=True)
                self.worker.start()
        self.tasks.put((kind, job))

    def _work(self):
        while not self.closed:
            kind, job = self.tasks.get()
            if kind == "stop":
                return
            try:
                if kind == "clear":
                    self._clear(job)
                elif not job["removed"] and job["state"] == "waiting":
                    self._prepare(job)
            except Exception as e:  # the worker stays up whatever one song does
                traceback.print_exc()
                with self.lock:
                    job.update(state="failed", error=f"{type(e).__name__}: {e}")

    def command(self, job):
        """What prepares a job's song: this module, run as a program in a process of its own."""
        cmd = [sys.executable, "-m", "bass2sheet.copilot_adding", self.workdir, job["path"], job["finder"]]
        return cmd + ([self.stems_dir] if self.stems_dir else [])

    def _fetch(self, job):
        """Fetch the audio a job's link points to into `uploads/`, as the step before all others. Returns whether
        there is now a file to prepare; if not, the job says why."""
        step, began = job["steps"][0], time.time()
        if job["path"] and os.path.isfile(job["path"]):  # fetched at an earlier try
            step.update(state="done", seconds=0)
            return True
        tmp = os.path.join(self.uploads, "." + job["key"])
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        cmd = [downloader() or "yt-dlp", "--no-config", "--no-playlist", "--no-progress", "--no-warnings"]
        cmd += ["--socket-timeout", "30", "-f", "bestaudio/best", "--max-filesize", f"{MAX_BYTES // (1 << 20)}M"]
        cmd += ["--match-filters", f"!is_live & duration<=?{MAX_SECONDS:.0f}"]
        cmd += ["-P", tmp, "-o", "%(title).150B [%(id)s].%(ext)s", "--", job["url"]]
        try:
            with self.lock:
                job["state"] = "running"
                step.update(state="running", started=began)
                self.child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                said, _ = self.child.communicate(timeout=FETCH_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.child.kill()
                said = f"ERROR: it took more than {FETCH_TIMEOUT // 60} minutes\n" + self.child.communicate()[0]
            got = sorted(f for f in os.listdir(tmp) if not f.endswith((".part", ".ytdl")))
            if job["removed"] or self.closed:
                return False
            if self.child.returncode != 0 or not got:
                raise Refused(400, f"The audio could not be fetched: {why_not(said)}")
            name = safe_name(got[0])
            if os.path.splitext(name)[1] not in AUDIO:
                raise Refused(415, f"The audio came as {name}, a kind of file the copilot does not take")
            probe(os.path.join(tmp, got[0]), name)
            entry = judge.song_entry(os.path.join(self.uploads, name))
            with self.lock:
                self._free(entry, but=job)
                size = os.path.getsize(os.path.join(tmp, got[0]))
                os.replace(os.path.join(tmp, got[0]), entry["path"])
                job.update(id=entry["id"], title=entry["title"], artist=entry["artist"], name=name, path=entry["path"])
                found = dict(megabytes=round(size / (1 << 20), 1))
                step.update(state="done", seconds=round(time.time() - began, 1), started=None, found=found)
            return True
        except (Refused, OSError) as e:
            with self.lock:
                step.update(state="failed", seconds=round(time.time() - began, 1), started=None)
                job.update(state="failed", error=str(e))
            return False
        finally:
            with self.lock:
                self.child = None
            shutil.rmtree(tmp, ignore_errors=True)

    def _prepare(self, job):
        os.makedirs(self.uploads, exist_ok=True)
        if job["url"] and not self._fetch(job):
            with self.lock:
                if job["removed"] or self.closed:
                    self.jobs.pop(job["key"], None)
            return
        env = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
        package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the code the page itself runs
        env["PYTHONPATH"] = os.pathsep.join(p for p in (package, env.get("PYTHONPATH")) if p)
        log = os.path.join(self.uploads, job["key"] + ".log")
        with self.lock:
            job["state"] = "running"
        with open(log, "w") as errors:
            with self.lock:
                self.child = subprocess.Popen(
                    self.command(job), stdout=subprocess.PIPE, stderr=errors, text=True, env=env
                )
            for line in self.child.stdout:
                if line.startswith(MARK):
                    try:
                        self._heard(job, json.loads(line[len(MARK) :]))
                    except ValueError:
                        pass
            code = self.child.wait()
        with self.lock:
            self.child = None
            if job["removed"] or self.closed:
                self.jobs.pop(job["key"], None)
            elif code == 0:
                for s in job["steps"]:  # a step that had nothing to do (kept from an earlier time) is done too
                    s.update(state="done", started=None)
                job.update(state="done", said="")
            else:
                with open(log) as f:
                    tail = f.read().strip().splitlines()[-1:]
                job["error"] = job["error"] or (tail[0] if tail else f"the preparation stopped (exit {code})")
                job["state"] = "failed"
                for s in job["steps"]:
                    if s["state"] == "running":
                        s.update(state="failed", seconds=round(time.time() - s["started"], 1), started=None)
        if job["removed"] and not self.closed:
            self._clear(job)
        elif code == 0 and os.path.exists(log):
            os.remove(log)

    def _heard(self, job, event):
        """One thing the preparing process said: a step began or ended, a word for the person, or what went wrong."""
        with self.lock:
            if "error" in event:
                job["error"] = str(event["error"])
            if "say" in event:
                job["said"] = str(event["say"])
            step = next((s for s in job["steps"] if s["id"] == event.get("step")), None)
            if step is None:
                return
            if event.get("state") == "running":
                step.update(state="running", started=time.time(), seconds=None, found={})
            elif event.get("state") == "done":
                found = {k: v for k, v in event.items() if k not in ("step", "state", "seconds")}
                took = event.get("seconds")
                if took is None and step["started"]:
                    took = round(time.time() - step["started"], 2)
                step.update(state="done", seconds=took, started=None, found=found)

    def _clear(self, job):
        """Delete what a song that was given up left behind: its file here, its folder and its line in the list
        of songs, unless it got as far as being prepared."""
        if os.path.exists(os.path.join(self.uploads, job["key"] + ".log")):
            os.remove(os.path.join(self.uploads, job["key"] + ".log"))
        if not job["path"]:  # a link that was never fetched: nothing was made
            return
        if os.path.dirname(os.path.abspath(job["path"])) == os.path.abspath(self.uploads):
            for path in (job["path"], job["path"] + ".part"):
                if os.path.exists(path):
                    os.remove(path)
        folder = copilot.song_dir(self.workdir, job["id"])
        if os.path.exists(os.path.join(folder, "machine.json")):
            return
        songs = copilot.load_songs(self.workdir)
        if songs.pop(job["id"], None) is not None:
            atomic_write_text(os.path.join(self.workdir, "songs.json"), json.dumps(songs, indent=1))
        shutil.rmtree(folder, ignore_errors=True)

    def _interrupted(self):
        """Songs that came from the page and were never prepared, because the server stopped first: they are
        shown as failed, to be tried again or removed."""
        for song_id, song in copilot.load_songs(self.workdir).items():
            mine = os.path.dirname(os.path.abspath(song.get("path", ""))) == os.path.abspath(self.uploads)
            if not mine or os.path.exists(os.path.join(copilot.song_dir(self.workdir, song_id), "machine.json")):
                continue
            finder = (song.get("settings") or {}).get("notes", "own")
            entry = dict(id=song_id, title=song.get("title", song_id), artist=song.get("artist", ""), path=song["path"])
            job = self._job(song_id, entry, os.path.basename(song["path"]), finder if finder in NOTE_FINDERS else "own")
            self.jobs[song_id] = dict(job, state="failed", error="It was being prepared when the copilot was stopped.")


def main(argv=None):
    """Prepare one song, as `bass2sheet-copilot add` does, and say on standard output how it goes: one line per
    event, MARK and then JSON. Usage: python -m bass2sheet.copilot_adding WORKDIR FILE FINDER [STEMS_DIR]"""
    workdir, path, finder, *stems = sys.argv[1:] if argv is None else argv

    def tell(event):
        print(MARK + json.dumps(event, default=float), flush=True)
        if event.get("step") == "picture" and event.get("state") == "done":  # what is left: writing it all down
            tell(dict(step="ready", state="running"))

    try:
        with listening(tell):
            copilot.add(
                [path],
                workdir,
                settings=dict(notes=finder) if finder != "own" else None,
                stems_dirs={path: stems[0]} if stems else None,
                say=lambda text: tell(dict(say=text)),
            )
        tell(dict(step="ready", state="done"))
    except Bass2SheetError as e:
        tell(dict(error=str(e)))
        return 1
    except Exception as e:  # anything else: the page shows it, the log keeps where it happened
        traceback.print_exc()
        tell(dict(error=f"{type(e).__name__}: {e}"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
