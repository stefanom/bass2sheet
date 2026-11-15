"""Make the demo film: python dev/demo/make.py --song FILE --out OUT.mp4

Needs: Python with playwright, Pillow, numpy and soundfile; Google Chrome; ffmpeg and ffprobe; and bass2sheet
itself, with MuseScore 4 (the film shows the PDF and the score in the page). See docs/demo.md.
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import scenes  # noqa: E402
import story  # noqa: E402
from film import Film  # noqa: E402

ORDER = ["command", "landing", "song", "sheet", "play", "tests", "opening", "ending"]  # as they are filmed
SHOWN = ["opening", "command", "landing", "song", "sheet", "play", "tests", "ending"]  # as the film shows them
STATEFUL = ["landing", "song", "sheet", "play"]  # these change the working directory: it is kept after each


class Context:
    """What the scenes share: where things are, how the tool is run, what is known of the song."""

    def __init__(self, args, song):
        self.args, self.song, self.facts = args, song, {}
        self.work = os.path.abspath(args.work)
        self.scores = os.path.join(self.work, "scores")  # the copilot's working directory
        self.stage = "file://" + os.path.join(HERE, "stage")
        self.url = f"http://localhost:{args.port}"
        self.tests_url = f"http://localhost:{args.port + 1}" if args.tests else None
        self.song_path = os.path.join(self.work, song["file"])
        self.song_id = None
        self.servers = []

    def tool(self, module, argv):
        """The command line that runs one of bass2sheet's programs with this checkout's code."""
        nice = ["nice", "-n", "15"] if self.args.gentle else []
        return [
            *nice,
            self.args.python,
            "-c",
            f"import sys; from bass2sheet import {module}; sys.exit({module}.main())",
            *argv,
        ]

    def env(self, **more):
        path = os.pathsep.join(p for p in (os.path.join(REPO, "src"), os.environ.get("PYTHONPATH")) if p)
        few = (
            {k: "2" for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
            if self.args.gentle
            else {}
        )
        return dict(os.environ, PYTHONPATH=path, PYTHONUNBUFFERED="1", **few, **more)

    async def run(self, module, argv, cwd):
        """Run a program of the tool; yields what it prints, a line at a time as it prints it, then its exit code."""
        p = await asyncio.create_subprocess_exec(
            *self.tool(module, argv),
            cwd=cwd,
            env=self.env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        while line := await p.stdout.readline():
            yield line.decode(errors="replace").rstrip("\n")
        yield await p.wait()

    def serve(self, workdir, port):
        """Start the copilot on a working directory; links to videos are left off (the page is for files here)."""
        try:
            urllib.request.urlopen(f"http://localhost:{port}/api/songs", timeout=1).read()
            raise SystemExit(f"something already answers on port {port}: stop it, or film on another with --port")
        except OSError:
            pass
        p = subprocess.Popen(
            self.tool("copilot", ["serve", "--dir", workdir, "--port", str(port)]),
            cwd=self.work,
            env=self.env(BASS2SHEET_DOWNLOADER="none"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/api/songs", timeout=1).read()
                break
            except OSError:
                if p.poll() is not None:
                    raise SystemExit(f"the copilot did not start on port {port}") from None
                time.sleep(0.2)
        self.servers.append(p)
        return p

    def stop(self):
        for p in self.servers:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.servers = []


CACHE = os.path.expanduser("~/.cache/bass2sheet")  # where the tool keeps what it worked out of a song


def cache_record(c):
    import hashlib

    with open(c.song_path, "rb") as f:
        return os.path.join(CACHE, f"demo-{hashlib.sha1(f.read()).hexdigest()[:16]}.json")


def forget(c):
    """So that the command is filmed as on a first time: what an earlier filming left of this song in the tool's
    cache is deleted (the separated tracks, the pitch tracks: all of it worked out again in a minute or two).
    Which files those are was written down by that filming, beside them."""
    record = cache_record(c)
    if os.path.exists(record):
        for name in json.load(open(record)):
            if os.path.isfile(os.path.join(CACHE, name)):
                os.remove(os.path.join(CACHE, name))
    return set(os.listdir(CACHE)) if os.path.isdir(CACHE) else set()


def remember(c, before):
    """Write down what the command left in the tool's cache: the files that were not there before it ran."""
    new = sorted(
        n
        for n in (set(os.listdir(CACHE)) if os.path.isdir(CACHE) else set()) - before
        if "-" in n and not n.startswith(("demo-", "sep-"))
    )
    record = cache_record(c)
    old = json.load(open(record)) if os.path.exists(record) else []
    with open(record, "w") as f:
        json.dump(sorted(set(old) | set(new)), f, indent=1)


def narration(folder, lines, work, fallback=True):
    """The recordings of the lines: `folder` holds one sound file per line, named by the line's id (o1.wav,
    c2.mp3 ...: any kind ffmpeg reads). Each line's length is its recording's, and each recording is brought to
    one form (44.1 kHz, mono) in the work folder. A line with no recording is spoken by the Mac's own voice
    where there is one (`say`: a stand-in, flat but intelligible); elsewhere it is only shown, for a guessed time."""
    out = os.path.join(work, "narration")
    os.makedirs(out, exist_ok=True)
    kinds = (".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg", ".opus")
    stand_in = shutil.which("say") if fallback else None
    spoken = 0
    for i, line in lines.items():
        found = next(
            (
                os.path.join(folder, i + ext)
                for ext in kinds
                if folder and os.path.exists(os.path.join(folder, i + ext))
            ),
            None,
        )
        if not found and stand_in:
            found = os.path.join(out, i + ".say.aiff")
            if not os.path.exists(found) or open(found + ".txt").read() != line["voice"]:
                subprocess.run([stand_in, "-r", "165", "-o", found, line["voice"]], check=True)
                with open(found + ".txt", "w") as f:
                    f.write(line["voice"])
            spoken += 1
        if not found:
            continue
        wav = os.path.join(out, i + ".wav")
        if not os.path.exists(wav) or os.path.getmtime(wav) < os.path.getmtime(found):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", found, "-ac", "1", "-ar", "44100", wav], check=True)
        seconds = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        line.update(seconds=round(seconds, 3), clip=wav, source=os.path.basename(found))
    if spoken:
        print(
            f"{spoken} of {len(lines)} lines have no recording in {folder or '--narration'}: the Mac's `say` speaks them, as a stand-in"
        )
    silent = [i for i, line in lines.items() if "clip" not in line]
    if silent:
        print(f"no recording and no voice for {', '.join(silent)}: these lines are shown, not spoken")
    return lines


def fresh_scores(c):
    """An empty working directory for the copilot, with a cache of its own so that the song is prepared as on a
    first time (the models are borrowed from the tool's usual cache, where they are, to spare the download)."""
    shutil.rmtree(c.scores, ignore_errors=True)
    cache = os.path.join(c.work, "cache")
    if not c.args.warm:
        shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(c.scores)
    os.makedirs(cache, exist_ok=True)
    models = os.path.join(CACHE, "models")
    if os.path.isdir(models) and not os.path.exists(os.path.join(cache, "models")):
        os.symlink(models, os.path.join(cache, "models"))
    with open(os.path.join(c.scores, "settings.json"), "w") as f:
        json.dump(dict(cache_dir=cache), f)


def keep_state(c, name):
    to = os.path.join(c.work, "state", name)
    shutil.rmtree(to, ignore_errors=True)
    shutil.copytree(c.scores, to, symlinks=True)


def restore_state(c, name, scene):
    kept = os.path.join(c.work, "state", name)
    if not os.path.isdir(kept):
        raise SystemExit(
            f"the scene `{scene}` starts where `{name}` left the song, and `{name}` has not been filmed in {c.work}: film it first"
        )
    shutil.rmtree(c.scores, ignore_errors=True)
    shutil.copytree(kept, c.scores, symlinks=True)


async def film(c, names, lines):
    for name in names:
        folder = os.path.join(c.work, "scenes", name)
        shutil.rmtree(folder, ignore_errors=True)
        print(f"filming {name}", flush=True)
        before = forget(c) if name == "command" and not c.args.warm else None
        if name == "landing":
            fresh_scores(c)
        elif name not in ("command", "tests"):  # from where the scene before it left the working directory
            restore_state(c, STATEFUL[STATEFUL.index(name) - 1] if name in STATEFUL else STATEFUL[-1], name)
        if name == "tests":
            c.serve(os.path.abspath(c.args.tests), c.args.port + 1)
        elif name != "command":
            c.serve(c.scores, c.args.port)
        try:
            if c.song_id is None and name not in ("command", "landing", "tests"):
                c.song_id = json.load(urllib.request.urlopen(c.url + "/api/songs"))[0]["id"]
            async with Film(folder, lines) as f:
                await getattr(scenes, name)(f, c)
        finally:
            c.stop()
            if before is not None:
                remember(c, before)
        if name in STATEFUL:
            keep_state(c, name)
        with open(os.path.join(c.work, "facts.json"), "w") as f:
            json.dump(c.facts, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="the song's audio file; its storyboard is dev/demo/songs/NAME.json")
    ap.add_argument(
        "--storyboard", help="the song's storyboard (default: the one in songs/ whose `file` is the song's name)"
    )
    ap.add_argument("--out", default="bass2sheet-demo.mp4")
    ap.add_argument(
        "--work",
        default=os.path.join(HERE, "work"),
        help="where the film is put together: scenes, sound, the copilot's working directory (default: dev/demo/work, which git ignores)",
    )
    ap.add_argument("--only", help="film only these scenes (comma-separated) and keep the others as they were filmed")
    ap.add_argument(
        "--port",
        type=int,
        default=8782,
        help="the copilot is served here while it is filmed (and the test songs on the next)",
    )
    ap.add_argument(
        "--python", default=None, help="the Python that runs bass2sheet (default: the checkout's .venv, else this one)"
    )
    ap.add_argument(
        "--tests",
        help="a copilot working directory of test songs (bass2sheet-copilot testset ...), for the scene about them",
    )
    ap.add_argument(
        "--edit",
        action="store_true",
        help="film nothing: put the film together again from the scenes as they were filmed",
    )
    ap.add_argument(
        "--film-only", action="store_true", help="film, and leave putting the film together for later (--edit)"
    )
    ap.add_argument("--no-small", action="store_true", help="do not make the copy under 10 MB")
    ap.add_argument(
        "--warm",
        action="store_true",
        help="for trying scenes out: let the tool use what it has kept of the song, so the command and the preparation take seconds (the film then shows a second run, not a first)",
    )
    ap.add_argument(
        "--gentle",
        action="store_true",
        help="leave the machine to others: low priority and two threads for everything that is run",
    )
    ap.add_argument(
        "--narration",
        help="a folder of recordings of the lines, one file each, named by the line's id (o1.wav, c1.mp3 ...)",
    )
    args = ap.parse_args()
    if args.python is None:
        venv = os.path.join(REPO, ".venv", "bin", "python")
        args.python = venv if os.path.exists(venv) else sys.executable
    boards = [
        json.load(open(os.path.join(HERE, "songs", n)))
        for n in sorted(os.listdir(os.path.join(HERE, "songs")))
        if n.endswith(".json")
    ]
    song = (
        json.load(open(args.storyboard))
        if args.storyboard
        else next((b for b in boards if b["file"] == os.path.basename(args.song)), None)
    )
    if song is None:
        raise SystemExit(
            f"no storyboard for {os.path.basename(args.song)} in dev/demo/songs: the film shows real mistakes of the tool on a song, and where they are has to be written down (see docs/demo.md)"
        )
    c = Context(args, song)
    os.makedirs(c.work, exist_ok=True)
    if not os.path.exists(c.song_path) or os.path.getsize(c.song_path) != os.path.getsize(args.song):
        shutil.copyfile(args.song, c.song_path)
    lines = narration(args.narration, story.lines(), c.work)
    names = [n for n in ORDER if (not args.only or n in args.only.split(",")) and (n != "tests" or args.tests)]
    if not args.edit:
        try:
            asyncio.run(film(c, names, lines))
        finally:
            c.stop()
    if args.film_only:
        return
    shown = [n for n in SHOWN if os.path.exists(os.path.join(c.work, "scenes", n, "scene.json"))]
    missing = [n for n in SHOWN if n not in shown and (n != "tests" or args.tests)]
    if missing:
        raise SystemExit(f"not filmed yet: {', '.join(missing)}")
    import edit

    if args.gentle:
        os.nice(15)  # the browser was filmed at its own pace; the editing can wait for others
    made = edit.assemble(
        [os.path.join(c.work, "scenes", n) for n in shown],
        lines,
        os.path.abspath(args.out),
        c.work,
        threads=2 if args.gentle else None,
        small=not args.no_small,
    )
    with open(os.path.join(c.work, "film.json"), "w") as f:
        json.dump(made, f, indent=1)
    print(f"{args.out}: {made['length']:.1f} s")
    for s in made["scenes"]:
        print(f"  {s['start']:6.1f} s  {s['name']} ({s['length']:.1f} s)")


if __name__ == "__main__":
    main()
