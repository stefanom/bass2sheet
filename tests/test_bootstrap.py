"""./bootstrap, run in a copy of the checkout where brew, uv and uname are stand-ins that write down how they
were called. Nothing is installed and nothing is downloaded."""

import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REAL_HOMEBREW = any(os.path.exists(p) for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"))

# Each stand-in appends its arguments to $FAKES/calls and takes its exit code from a file there, 0 by default.
BREW = """#!/bin/sh
echo "brew $* [muscriptor=${HOMEBREW_BASS2SHEET_MUSCRIPTOR:-}]" >> "$FAKES/calls"
case "$1 ${2:-}" in
    "--version ") echo "Homebrew 7.0.13" ;;
    "--prefix ") echo "$FAKES/prefix" ;;
    "shellenv ") ;;
    "bundle check")
        code=$(cat "$FAKES/brew-check" 2>/dev/null || echo 0)
        [ "$code" = 0 ] || echo "brew bundle can't satisfy your Brewfile's dependencies."
        [ "$code" = 0 ] || echo "→ Formula uv needs to be installed."
        exit "$code" ;;
    "bundle install")
        code=$(cat "$FAKES/brew-install" 2>/dev/null || echo 0)
        [ "$code" = 0 ] && echo 0 > "$FAKES/brew-check"
        exit "$code" ;;
esac
"""
UV = """#!/bin/sh
echo "uv $*" >> "$FAKES/calls"
python() {
    mkdir -p "$1/bin" && cp "$FAKES/python" "$1/bin/python"
}
case "$1" in
    --version) echo "uv 0.13.7" ;;
    sync)
        case "$*" in *--check*) exit "$(cat "$FAKES/uv-check" 2>/dev/null || echo 0)" ;; esac
        code=$(cat "$FAKES/uv-sync" 2>/dev/null || echo 0)
        echo "${SDKROOT:-}" > "$FAKES/sdkroot"
        [ "$code" = 0 ] && python .venv
        exit "$code" ;;
    venv) python .venv-bp ;;
    pip) touch .venv-bp/has-basic-pitch ;;
esac
"""
PYTHON = """#!/bin/sh
case "$1" in
    --version) echo "Python 3.12.16" ;;
    -c) case "$2" in *basic_pitch*) [ -f "$(dirname "$0")/../has-basic-pitch" ] || exit 1 ;; esac ;;
    -m)
        echo "doctor $*" >> "$FAKES/calls"
        echo "Ready."
        exit "$(cat "$FAKES/doctor" 2>/dev/null || echo 0)" ;;
esac
"""
UNAME = """#!/bin/sh
cat "$FAKES/os"
"""
XCRUN = """#!/bin/sh
echo "$FAKES/sdk"
exit "$(cat "$FAKES/xcrun" 2>/dev/null || echo 0)"
"""


class Clone:
    """A copy of what ./bootstrap reads, and a machine of stand-ins around it."""

    def __init__(self, tmp_path, system="Darwin"):
        self.dir, self.fakes, self.bin = tmp_path / "clone", tmp_path / "fakes", tmp_path / "bin"
        for folder in (self.dir, self.fakes, self.bin):
            folder.mkdir()
        for name in ("bootstrap", "Brewfile", "uv.lock", "pyproject.toml"):
            shutil.copy(os.path.join(ROOT, name), self.dir / name)
        (self.fakes / "sdk").mkdir()
        for name, text in (("brew", BREW), ("uv", UV), ("uname", UNAME), ("xcrun", XCRUN)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        (self.fakes / "python").write_text(PYTHON)
        (self.fakes / "python").chmod(0o755)
        (self.fakes / "os").write_text(system + "\n")
        homebrew_python = self.fakes / "prefix" / "opt" / "python@3.12" / "bin"
        homebrew_python.mkdir(parents=True)
        shutil.copy(self.fakes / "python", homebrew_python / "python3.12")
        self.python = str(homebrew_python / "python3.12")

    def set(self, name, code):
        (self.fakes / name).write_text(f"{code}\n")

    def environment(self):
        """The Python environment as an earlier run left it."""
        (self.dir / ".venv" / "bin").mkdir(parents=True)
        shutil.copy(self.fakes / "python", self.dir / ".venv" / "bin" / "python")

    def run(self, *options):
        env = dict(PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.fakes), FAKES=str(self.fakes))
        r = subprocess.run([str(self.dir / "bootstrap"), *options], capture_output=True, text=True, env=env, timeout=60)
        return r.returncode, r.stdout, r.stderr

    @property
    def calls(self):
        path = self.fakes / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def called(self, start):
        return [c for c in self.calls if c.startswith(start)]


@pytest.fixture
def clone(tmp_path):
    return Clone(tmp_path)


def test_a_first_run_on_a_mac_installs_creates_fetches_checks_and_says_what_to_try(clone):
    clone.set("brew-check", 1)  # something in the Brewfile is not installed
    code, out, err = clone.run()
    assert code == 0, err
    assert "→ Formula uv needs to be installed." in out
    assert clone.called("brew bundle install") == ["brew bundle install --file=Brewfile --no-upgrade [muscriptor=]"]
    assert clone.called("uv sync") == [f"uv sync --locked --python {clone.python}"]  # Homebrew's Python 3.12
    assert clone.called("doctor") == ["doctor -m bass2sheet.doctor --fetch-models"]
    assert (clone.fakes / "sdkroot").read_text().strip() == str(clone.fakes / "sdk")  # the compiler is told its SDK
    for command in ('uv run bass2sheet "Artist - Title.mp3"', "uv run bass2sheet-copilot serve", "uv run pytest"):
        assert command in out
    steps = re.findall(r"^==> (.*)$", out, re.M)
    assert [s.split(" (")[0].split(",")[0] for s in steps] == [
        "Homebrew",
        "Homebrew's packages",
        "Python environment",
        "Model files",
    ]


def test_a_second_run_installs_nothing_and_keeps_the_environment_it_finds(clone):
    clone.environment()
    code, out, err = clone.run()
    assert code == 0, err
    assert "everything in the Brewfile is installed" in out
    assert clone.called("brew bundle install") == []
    assert clone.called("uv sync") == ["uv sync --locked"]  # no --python: the one it has stays


def test_check_changes_nothing_and_says_what_is_not_there(clone):
    clone.set("brew-check", 1)
    code, out, err = clone.run("--check")
    assert code == 1
    assert "NOT THERE: the packages above" in out and "NOT THERE: the Python environment" in out
    assert "NOT THERE: the self-check needs the Python environment" in out
    assert "not everything is there" in err
    assert clone.called("brew bundle install") == [] and clone.called("uv sync") == [] and clone.called("doctor") == []
    assert not (clone.dir / ".venv").exists()


def test_check_passes_where_everything_is_there(clone):
    clone.environment()
    code, out, err = clone.run("--check")
    assert code == 0, err
    assert out.strip().splitlines()[-1] == "Everything is there. Nothing was changed."
    assert clone.called("uv sync") == ["uv sync --locked --check"]
    assert clone.called("doctor") == ["doctor -m bass2sheet.doctor"]  # it checks, it does not download
    clone.set("uv-check", 1)  # the environment has other packages than uv.lock lists
    code, out, err = clone.run("--check")
    assert code == 1 and "NOT THERE: some of the packages uv.lock lists" in out
    clone.set("uv-check", 0)
    clone.set("doctor", 1)  # the self-check finds something required missing
    assert clone.run("--check")[0] == 1


def test_no_models_runs_the_self_check_without_downloading(clone):
    code, out, err = clone.run("--no-models")
    assert code == 0, err
    assert clone.called("doctor") == ["doctor -m bass2sheet.doctor"]


def test_the_optional_parts_are_asked_for_by_name(clone):
    code, out, err = clone.run("--with-muscriptor", "--with-basic-pitch")
    assert code == 0, err
    assert clone.called("brew bundle check")[0].endswith("[muscriptor=1]")  # the Brewfile reads the variable
    assert "docs/muscriptor.md" in out
    assert clone.called("uv venv") == [f"uv venv --clear --python {clone.python} .venv-bp"]
    assert clone.called("uv pip") == [
        "uv pip install --python .venv-bp/bin/python basic-pitch onnxruntime setuptools<81"
    ]
    clone.run("--with-basic-pitch")  # it is there now: not installed again
    assert len(clone.called("uv pip")) == 1


@pytest.mark.parametrize(
    "what,sentence,never",
    [
        ("brew-install", "Homebrew could not install everything the Brewfile lists", "uv sync"),
        ("uv-sync", "uv could not create the Python environment", "doctor"),
        ("doctor", "the self-check did not pass", None),
    ],
)
def test_a_step_that_fails_stops_the_run_with_a_sentence(clone, what, sentence, never):
    clone.set("brew-check", 1)
    clone.set(what, 1)
    code, out, err = clone.run()
    assert code == 1
    assert f"bootstrap: {sentence}" in err and "run ./bootstrap again" in err.lower()
    assert "Try it" not in out
    if never:
        assert clone.called(never) == []


def test_a_model_that_cannot_be_downloaded_does_not_hide_that_the_rest_is_ready(clone):
    clone.set("doctor", 2)  # what is required is there; a download failed
    code, out, err = clone.run()
    assert code == 1
    assert "Not complete" in out and "uv run bass2sheet-copilot serve" in out and "Done in" not in out
    assert "bootstrap: not complete: a model file could not be downloaded" in err


def test_without_a_c_compiler_it_says_how_to_get_one_before_anything_is_built(clone):
    clone.set("xcrun", 1)  # as on a Mac without the Xcode command line tools
    code, out, err = clone.run()
    assert code == 1
    assert "there is no C compiler" in err and "xcode-select --install" in err
    assert clone.called("uv sync") == []
    code, out, err = clone.run("--check")
    assert code == 1 and "NOT THERE: a C compiler" in out
    clone.environment()  # with the environment there, nothing is left to compile
    code, out, err = clone.run()
    assert code == 0, err


def test_a_stop_nobody_foresaw_is_told_too(clone):
    (clone.bin / "uname").write_text("#!/bin/sh\nexit 3\n")
    code, out, err = clone.run()
    assert code == 3
    assert 'bootstrap: stopped in the step "finding out what this machine is" (exit code 3)' in err


@pytest.mark.skipif(REAL_HOMEBREW, reason="this machine has Homebrew where the script looks for it")
def test_without_homebrew_it_says_how_to_install_it_and_installs_nothing(clone):
    os.remove(clone.bin / "brew")
    code, out, err = clone.run()
    assert code == 1
    assert "Homebrew is not installed, and this script does not install it" in err
    assert "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh" in err
    assert clone.calls == []


def test_on_linux_it_lists_the_packages_and_does_the_python_part(tmp_path):
    clone = Clone(tmp_path, "Linux")
    code, out, err = clone.run()
    assert code == 0, err
    for line in ("MuseScore 4", "sudo apt install rubberband-cli", "uv tool install yt-dlp", "sudo apt install nodejs"):
        assert line in out
    assert clone.called("brew") == []
    assert clone.called("uv sync") == ["uv sync --locked --python 3.12"]
    assert clone.called("doctor") == ["doctor -m bass2sheet.doctor --fetch-models"]
    os.remove(clone.bin / "uv")
    code, out, err = clone.run()
    assert code == 1 and "uv is not installed" in err and "https://astral.sh/uv/install.sh" in err


def test_another_system_and_an_unknown_option_are_refused(tmp_path):
    clone = Clone(tmp_path, "FreeBSD")
    code, out, err = clone.run()
    assert code == 1 and "this script knows macOS and Linux, and this is FreeBSD" in err
    code, out, err = clone.run("--everything")
    assert code == 1 and "unknown option: --everything" in err and "--no-models" in err
    code, out, err = clone.run("--help")
    assert code == 0 and "--check" in out and clone.calls == []


def test_the_brewfile_lists_what_the_tool_takes_from_homebrew():
    with open(os.path.join(ROOT, "Brewfile")) as f:
        text = f.read()
    always = re.findall(r'^(?:brew|cask) "([^"]+)"', text, re.M)
    assert set(always) == {"python@3.12", "uv", "musescore", "ffmpeg", "rubberband", "yt-dlp", "node"}
    asked_for = re.findall(r'^  brew "([^"]+)"', text, re.M)  # inside the `if` for MuScriptor
    assert asked_for == ["cmake", "ninja"]
    assert os.access(os.path.join(ROOT, "bootstrap"), os.X_OK)
