"""bass2sheet-doctor: what the tool needs on this machine, each thing found or missing.

    bass2sheet-doctor                  one line for each thing
    bass2sheet-doctor --fetch-models   first download the model files the default pipeline uses

Exit code 0 when what is required is there, 1 when it is not, and 2 when it is there and a model file that was
asked for could not be downloaded.

A line gives the version and the place of what was found. For what is missing it says what stops working
without it and the one command that fixes it. Three kinds of line:

    ok        it is there
    missing   it is not; MISSING, in capitals, when the tool cannot work without it
    later     a model file that is not downloaded yet: the first song that needs it fetches it

Nothing here changes the machine, except --fetch-models: it downloads into the places the pipeline reads, the
separation model into bass2sheet's cache and the others where their libraries keep them.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass

from . import __version__

OK, MISSING, LATER = "ok", "missing", "later"
CACHE = "~/.cache/bass2sheet"  # where the pipeline keeps models, stems and pitch tracks unless told otherwise
MB = 1000 * 1000
LOW_DISK = 5 * 1000 * MB  # less free space than this is worth a line
FFMPEG_VERSION = r"version (\S+)"  # ffmpeg's versions are not always numbers
BEAT_CHECKPOINT = "final0"  # the one rhythm.model_beats asks Beat This! for
DEMUCS_MODEL = "htdemucs_ft"  # the one separate._demucs asks Demucs for

# The packages pyproject.toml asks for: the module that is imported, and the name its version goes by
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "av": "av",
    "pretty_midi": "pretty_midi",
    "mir_eval": "mir_eval",
    "torch": "torch",
    "torchcrepe": "torchcrepe",
    "demucs": "demucs",
    "audio_separator": "audio-separator",
    "audioread": "audioread",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "yaml": "pyyaml",
    "beat_this": "beat-this",
}
SHOWN = ("torch", "audio-separator", "demucs", "torchcrepe", "beat-this")  # the versions worth a place on the line

# The one command that installs a program: Homebrew's on a Mac, elsewhere what Debian and Ubuntu take
INSTALL = {
    "Darwin": {
        "ffmpeg": "brew install ffmpeg",
        "musescore": "brew install --cask musescore",
        "rubberband": "brew install rubberband",
        "yt-dlp": "brew install yt-dlp",
        "node": "brew install node",
    },
    "Linux": {
        "ffmpeg": "sudo apt install ffmpeg",
        "musescore": "install MuseScore 4 from musescore.org so that `mscore` is on the PATH",
        "rubberband": "sudo apt install rubberband-cli",
        "yt-dlp": "uv tool install yt-dlp",
        "node": "sudo apt install nodejs",
    },
}


@dataclass
class Check:
    """One thing the tool needs, and how it stands on this machine."""

    name: str
    status: str  # OK, MISSING or LATER
    found: str = ""  # the version and the place of what is there, or how it is missing
    without: str = ""  # what stops working without it
    fix: str = ""  # one command
    required: bool = False  # the tool cannot work without it

    def line(self, width=0):
        label = self.status.upper() if self.required and self.status == MISSING else self.status
        parts = [self.found] if self.found else []
        if self.status != OK:
            parts += [f"Without it: {self.without}"] if self.without else []
            parts += [f"Fix: {self.fix}"] if self.fix else []
        return f"  {label:<8} {self.name:<{width}}  " + ". ".join(p.rstrip(".") for p in parts)


@dataclass
class Model:
    """A model file the pipeline downloads on first use."""

    name: str  # what its line is called
    what: str  # the model's own name
    megabytes: int  # about how much is downloaded
    later: str  # what happens while it is not there
    files: object  # (cache_dir) -> its files when all of them are there, else []
    fetch: object  # (cache_dir) -> downloads them
    folder: object  # (cache_dir) -> where they go


# ------------------------------------------------------------------------------------------------- small helpers


def cache_home(cache_dir=None):
    return os.path.abspath(os.path.expanduser(cache_dir or CACHE))


def short(path):
    """A path as a person writes it: ~ for the home directory."""
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path == home or path.startswith(home + os.sep) else path


def megabytes(paths):
    return f"{sum(os.path.getsize(p) for p in paths) / MB:.0f} MB"


def said(command, timeout=20):
    """(exit code, what was printed) of a program run, or (None, "") when it cannot be run."""
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def version_of(command, pattern=r"(\d+(?:\.\d+)+)"):
    """The version a program gives for itself, or "version not told"."""
    m = re.search(pattern, said(command)[1])
    return m.group(1) if m else "version not told"


def install(program):
    return INSTALL.get(platform.system(), INSTALL["Linux"])[program]


def reason(error):
    """An error in a few words, with the error under it when a library wrapped one in another."""
    under = error.__cause__ or error.__context__
    text = f"{type(error).__name__}: {error}"
    return f"{text} ({type(under).__name__}: {under})" if under else text


# ------------------------------------------------------------------------------------------------- what is needed


def check_python(version=None):
    version = tuple(version or sys.version_info[:3])
    told, where = ".".join(map(str, version)), short(os.path.normpath(sys.executable))
    if version[:2] == (3, 12):
        return Check("Python 3.12", OK, f"{told} {where}", required=True)
    return Check(
        "Python 3.12",
        MISSING,
        f"this is Python {told} ({where})",
        "the audio models are built for Python 3.12 and may not load",
        "./bootstrap",
        required=True,
    )


def check_packages():
    absent = [name for module, name in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if absent:
        return Check(
            "Python packages",
            MISSING,
            f"not installed: {', '.join(absent)}",
            "the tool does not start",
            "./bootstrap",
            required=True,
        )
    versions = []
    for name in SHOWN:
        try:
            versions.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{name} (version not told)")
    return Check("Python packages", OK, ", ".join(versions), required=True)


def check_gpu():
    """Whether PyTorch loads, and what it will run the models on."""
    try:
        import torch
    except Exception as e:  # a build for another machine fails in its own ways
        return Check("PyTorch", MISSING, f"does not load ({reason(e)})", "no model runs", "./bootstrap", required=True)
    if torch.backends.mps.is_available():
        return Check("GPU", OK, "mps, the Mac's own")
    if torch.cuda.is_available():
        return Check("GPU", OK, f"cuda, {torch.cuda.get_device_name(0)}")
    return Check("GPU", MISSING, "none that PyTorch can use", "the models run on the processor, which is much slower")


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return Check("ffmpeg", OK, f"{version_of([path, '-version'], FFMPEG_VERSION)} {path}", required=True)
    try:
        import imageio_ffmpeg

        own = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or no program for this machine in it
        own = None
    if own:
        version = version_of([own, "-version"], FFMPEG_VERSION)
        return Check(
            "ffmpeg", OK, f"{version}, the copy inside the Python environment (none on the PATH)", required=True
        )
    return Check(
        "ffmpeg",
        MISSING,
        "none on the PATH and none inside the Python environment",
        "the bass cannot be separated from the song",
        install("ffmpeg"),
        required=True,
    )


def check_crepe():
    spec = importlib.util.find_spec("torchcrepe")
    path = os.path.join(os.path.dirname(spec.origin), "assets", "full.pth") if spec and spec.origin else ""
    if os.path.isfile(path):
        return Check("Pitch model", OK, f"CREPE, {megabytes([path])} inside the torchcrepe package", required=True)
    return Check(
        "Pitch model",
        MISSING,
        "CREPE's weights are not inside the torchcrepe package",
        "no pitch is tracked, so no notes are found",
        "uv sync --locked --reinstall-package torchcrepe",
        required=True,
    )


# ------------------------------------------------------------------------------------------------- the model files


def models_dir(cache_dir=None):
    """Where the separation library reads its models: `models` in bass2sheet's cache, unless
    AUDIO_SEPARATOR_MODEL_DIR is set, which the library puts above what it is asked for."""
    return os.environ.get("AUDIO_SEPARATOR_MODEL_DIR") or os.path.join(cache_home(cache_dir), "models")


def roformer_files(cache_dir=None):
    from .separate import ROFORMER_MODEL

    names = (ROFORMER_MODEL, os.path.splitext(ROFORMER_MODEL)[0] + ".yaml")
    paths = [os.path.join(models_dir(cache_dir), name) for name in names]
    return paths if all(os.path.isfile(p) for p in paths) else []


def fetch_roformer(cache_dir=None):
    import logging

    from audio_separator.separator import Separator

    from .separate import ROFORMER_MODEL

    os.makedirs(models_dir(cache_dir), exist_ok=True)
    # info_only: no GPU is set up and nothing is loaded; the files are checked against their published checksums
    separator = Separator(model_file_dir=models_dir(cache_dir), info_only=True, log_level=logging.WARNING)
    separator.download_model_files(ROFORMER_MODEL)


def torch_checkpoints(cache_dir=None):
    import torch

    return os.path.join(torch.hub.get_dir(), "checkpoints")


def beat_files(cache_dir=None):
    path = os.path.join(torch_checkpoints(), f"beat_this-{BEAT_CHECKPOINT}.ckpt")
    return [path] if os.path.isfile(path) else []


def fetch_beats(cache_dir=None):
    from beat_this.inference import load_checkpoint

    load_checkpoint(BEAT_CHECKPOINT)


def demucs_files(cache_dir=None):
    """The files of Demucs's model where Demucs looks for them. A recent Demucs keeps them in the Hugging Face
    cache. An older one (the lock file has one for Intel Macs) keeps them with PyTorch's checkpoints."""
    import yaml

    try:
        from demucs import hf
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        hf = None
    if hf:
        repo = f"{hf.DEFAULT_NAMESPACE}/{hf.hf_repo_name(DEMUCS_MODEL)}"
        bag = try_to_load_from_cache(repo, f"{DEMUCS_MODEL}.yaml")
        if isinstance(bag, str):
            with open(bag) as f:
                parts = [try_to_load_from_cache(repo, f"{sig}.safetensors") for sig in yaml.safe_load(f)["models"]]
            if all(isinstance(p, str) for p in parts):
                return [bag] + parts
    from demucs.pretrained import REMOTE_ROOT, _parse_remote_files

    with open(REMOTE_ROOT / f"{DEMUCS_MODEL}.yaml") as f:
        sigs = yaml.safe_load(f)["models"]
    urls = _parse_remote_files(REMOTE_ROOT / "files.txt")
    paths = [os.path.join(torch_checkpoints(), os.path.basename(urls[sig])) for sig in sigs]
    return paths if all(os.path.isfile(p) for p in paths) else []


def fetch_demucs(cache_dir=None):
    from demucs.pretrained import get_model

    get_model(DEMUCS_MODEL)


def demucs_folder(cache_dir=None):
    try:
        from demucs import hf  # noqa: F401
        from huggingface_hub import constants
    except ImportError:
        return torch_checkpoints()
    return constants.HF_HUB_CACHE


MODELS = [
    Model(
        "Separation model",
        "BS-RoFormer SW",
        699,
        "the first song downloads it before it starts",
        roformer_files,
        fetch_roformer,
        models_dir,
    ),
    Model(
        "Beat model",
        "Beat This!",
        81,
        "the first song downloads it. If that fails, `bass2sheet` goes on with its built-in beat tracker; "
        "the copilot and the judging page have no other and cannot take a song",
        beat_files,
        fetch_beats,
        torch_checkpoints,
    ),
    Model(
        "Second separation model",
        "Demucs htdemucs_ft",
        336,
        "it is downloaded the first time BS-RoFormer fails on a song or its result fails the quality gates",
        demucs_files,
        fetch_demucs,
        demucs_folder,
    ),
]


def check_model(model, cache_dir=None):
    files = model.files(cache_dir)
    if files:
        return Check(model.name, OK, f"{model.what}, {megabytes(files)} in {short(model.folder(cache_dir))}")
    found = f"{model.what} is not downloaded yet ({model.megabytes} MB)"
    return Check(model.name, LATER, found, model.later, "bass2sheet-doctor --fetch-models")


def fetch_models(cache_dir=None, say=print):
    """Download the model files that are not there yet. Returns the names of those that could not be had."""
    if not sys.stderr.isatty():
        os.environ.setdefault("TQDM_DISABLE", "1")  # the libraries' progress bars fill a log with lines
    failed = []
    for model in MODELS:
        label = f"{model.name} ({model.what})"
        try:
            files = model.files(cache_dir)
            if files:
                say(f"{label}: already there, {megabytes(files)}")
                continue
            say(f"{label}: downloading about {model.megabytes} MB into {short(model.folder(cache_dir))}")
            began = time.monotonic()
            model.fetch(cache_dir)
            files = model.files(cache_dir)
            if not files:
                raise RuntimeError("the download ended and the files are not where the pipeline looks for them")
            say(f"{label}: {megabytes(files)} in {time.monotonic() - began:.0f} s")
        except Exception as e:  # the libraries raise many kinds: no network, a full disk, a bad checksum
            failed.append(label)
            say(f"{label}: NOT downloaded. {reason(e)}")
    return failed


# ------------------------------------------------------------------------------------------------- what is optional


def check_musescore():
    from .score import find_mscore

    path = find_mscore()
    if not path:
        return Check(
            "MuseScore 4",
            MISSING,
            "not found",
            "no PDF, only MusicXML, and no pictures of the score in the copilot, in play-along and in judging",
            install("musescore"),
        )
    version = version_of([path, "--version"])
    other = version[:1].isdigit() and not version.startswith("4.")
    return Check("MuseScore 4", OK, f"{version} {path}" + (" (bass2sheet is tested with MuseScore 4)" if other else ""))


def check_rubberband():
    from .copilot import stretcher

    kind, path = stretcher()
    if kind == "rubberband":
        return Check("Rubber Band", OK, f"{version_of([path, '--version'])} {path}")
    return Check(
        "Rubber Band",
        MISSING,
        "not found",
        "a song is slowed down by ffmpeg instead, which is rougher on a bass line",
        install("rubberband"),
    )


def check_downloader():
    from .copilot_adding import downloader

    path = downloader()
    if path:
        return Check("yt-dlp", OK, f"{version_of([path, '--version'])} {path}")
    without = "the copilot takes audio files only, no links"
    named = os.environ.get("BASS2SHEET_DOWNLOADER")
    if named is not None:
        found = f"BASS2SHEET_DOWNLOADER is set to {named!r}, which is not a program"
        return Check("yt-dlp", MISSING, found, without, "unset BASS2SHEET_DOWNLOADER")
    return Check("yt-dlp", MISSING, "not found", without, install("yt-dlp"))


def check_basic_pitch():
    from .second_opinion import find_python

    without = "the two quality gates that compare the notes with a second model are skipped"
    fix = "./bootstrap --with-basic-pitch"
    python = find_python()
    if not python:
        return Check("Basic Pitch", MISSING, "no Python environment of its own found (.venv-bp)", without, fix)
    probe = (
        "import importlib.metadata as m, importlib.util as u, sys; "
        "there = u.find_spec('basic_pitch') and u.find_spec('onnxruntime'); "
        "print(m.version('basic-pitch')) if there else sys.exit(1)"
    )
    code, text = said([python, "-c", probe])
    if code != 0:
        return Check("Basic Pitch", MISSING, f"{short(python)} has no basic-pitch or no onnxruntime", without, fix)
    return Check("Basic Pitch", OK, f"{text.strip() or 'version not told'} {short(python)}")


def check_muscriptor():
    from . import muscriptor
    from .common import Bass2SheetError

    try:
        binary, _, size = muscriptor.program()
    except Bass2SheetError:
        return Check(
            "MuScriptor",
            MISSING,
            "not set up",
            "`--notes muscriptor` is refused; bass2sheet's own note finder is not affected",
            "see docs/muscriptor.md",
        )
    return Check("MuScriptor", OK, f"the {size} model, {short(binary)}")


def check_node():
    path = shutil.which("node")
    if path:
        return Check("Node.js", OK, f"{version_of([path, '--version'])} {path}")
    return Check(
        "Node.js",
        MISSING,
        "not found",
        "the tests of the pages' scripts are skipped; the tool itself does not use it",
        install("node"),
    )


def check_disk(cache_dir=None):
    folder = cache_home(cache_dir)
    there = folder
    while not os.path.exists(there):
        there = os.path.dirname(there)
    free = shutil.disk_usage(there).free
    found = f"{free / (1000 * MB):.0f} GB free where the cache is ({short(folder)})"
    if free >= LOW_DISK:
        return Check("Disk space", OK, found)
    without = "the models take 1.1 GB there and each song about half a GB of separated tracks"
    return Check("Disk space", MISSING, found, without, "free some space, or give the commands --cache-dir DIR")


def checks(cache_dir=None):
    """Every check, in the order of the lines. A check that fails to run is a line too, not a crash."""
    todo = [
        ("Python 3.12", True, check_python),
        ("Python packages", True, check_packages),
        ("GPU", False, check_gpu),
        ("ffmpeg", True, check_ffmpeg),
        ("Pitch model", True, check_crepe),
        *[(m.name, False, lambda m=m: check_model(m, cache_dir)) for m in MODELS],
        ("MuseScore 4", False, check_musescore),
        ("Rubber Band", False, check_rubberband),
        ("yt-dlp", False, check_downloader),
        ("Basic Pitch", False, check_basic_pitch),
        ("MuScriptor", False, check_muscriptor),
        ("Node.js", False, check_node),
        ("Disk space", False, lambda: check_disk(cache_dir)),
    ]
    out = []
    for name, required, check in todo:
        try:
            out.append(check())
        except Exception as e:  # most often a Python package that is missing or does not load
            found = f"could not be checked ({reason(e)})"
            out.append(Check(name, MISSING, found, "", "./bootstrap", required=required))
    return out


def system():
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def report(found, say=print):
    """Print the lines and what they add up to. Returns the exit code: 0 when what is required is there."""
    width = max(len(c.name) for c in found)
    say(f"bass2sheet {__version__} on {system()}")
    for c in found:
        say(c.line(width))
    missing = [c for c in found if c.status == MISSING]
    required = [c.name for c in missing if c.required]
    later = [c for c in found if c.status == LATER]
    if later:
        names = ", ".join(c.name.lower() for c in later)
        say(f"Not downloaded yet: {names}. To download now: bass2sheet-doctor --fetch-models")
    if required:
        say(f"Not ready. Required and missing: {', '.join(required)}.")
        return 1
    optional = len(missing)
    say("Ready." + (f" {optional} optional thing{'s are' if optional > 1 else ' is'} missing." if optional else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bass2sheet-doctor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    total = sum(m.megabytes for m in MODELS)
    ap.add_argument(
        "--fetch-models",
        action="store_true",
        help=f"first download the model files the default pipeline uses, about {total / 1000:.1f} GB",
    )
    ap.add_argument(
        "--cache-dir", default=CACHE, help="bass2sheet's cache, as the other commands take it (default: %(default)s)"
    )
    ap.add_argument("--version", action="version", version=f"bass2sheet {__version__}")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")  # what the model libraries say about each other as they load
    failed = []
    if args.fetch_models:
        failed = fetch_models(args.cache_dir)
        print()
    code = report(checks(args.cache_dir))
    if failed:
        print(
            f"Could not download: {', '.join(failed)}. The line marked NOT downloaded above says why. If it is "
            "the network: behind a proxy, set HTTPS_PROXY. Then run `bass2sheet-doctor --fetch-models` again."
        )
        return code or 2
    return code


if __name__ == "__main__":
    sys.exit(main())
