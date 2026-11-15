"""bass2sheet-doctor. The outside programs are stand-ins (shell scripts that print a version), the model files
are small files in temporary caches, and nothing is downloaded."""

import collections
import os
import tomllib
import types

import pytest

from bass2sheet import copilot, copilot_adding, doctor, muscriptor, score, second_opinion
from bass2sheet.common import Bass2SheetError

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1000 * doctor.MB
real_gpu, real_downloader = doctor.check_gpu, copilot_adding.downloader  # the fixture puts stand-ins in their place


def program(folder, name, says="", code=0):
    """A stand-in for an outside program: it prints `says` and exits with `code`."""
    path = folder / name
    path.write_text(f"#!/bin/sh\necho '{says}'\nexit {code}\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine where everything the tool needs is there: every program a stand-in, every model file a small
    file. `m.bin` holds the programs on the PATH, `m.there` says which model files exist, `m.fetched` lists the
    models a test had downloaded, and `m.cache` is bass2sheet's cache."""
    bin_dir, cache = tmp_path / "bin", tmp_path / "cache"
    bin_dir.mkdir()
    program(bin_dir, "ffmpeg", "ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers")
    program(bin_dir, "mscore", "MuseScore4 4.7.6")
    program(bin_dir, "node", "v26.13.0")
    rubberband = program(tmp_path, "rubberband", "4.0.0")
    downloader = program(tmp_path, "yt-dlp", "2026.10.26")
    bp_python = program(tmp_path, "bp-python", "0.3.0")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("BASS2SHEET_DOWNLOADER", raising=False)
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", ["mscore"])
    monkeypatch.setattr(copilot, "stretcher", lambda: ("rubberband", rubberband))
    monkeypatch.setattr(copilot_adding, "downloader", lambda: downloader)
    monkeypatch.setattr(second_opinion, "find_python", lambda explicit=None: bp_python)
    monkeypatch.setattr(muscriptor, "program", lambda: ("/opt/muscriptor_notes", "/opt/m.gguf", "medium"))
    monkeypatch.setattr(doctor, "check_gpu", lambda: doctor.Check("GPU", doctor.OK, "mps, the Mac's own"))
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda path: usage(500 * GB, 400 * GB, 100 * GB))

    m = types.SimpleNamespace(bin=bin_dir, cache=str(cache), there={}, fetched=[], broken={})
    models = []
    for key, megabytes in (("separation", 699), ("beats", 81), ("second", 336)):
        path = tmp_path / f"{key}.ckpt"
        path.write_bytes(b"x" * 2 * doctor.MB)
        m.there[key] = True

        def fetch(cache_dir, key=key):
            if key in m.broken:
                raise m.broken[key]
            m.fetched.append(key)
            m.there[key] = True

        files = lambda cache_dir, key=key, path=path: [str(path)] if m.there[key] else []
        models.append(
            doctor.Model(
                f"{key} model", key.upper(), megabytes, "the first song waits", files, fetch, lambda c: "/models"
            )
        )
    monkeypatch.setattr(doctor, "MODELS", models)
    return m


def by_name(cache_dir=None):
    return {c.name: c for c in doctor.checks(cache_dir)}


def test_everything_there_is_one_line_each_and_exit_0(machine, capsys):
    assert doctor.main(["--cache-dir", machine.cache]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert len(lines) == len(doctor.checks(machine.cache)) == 15
    assert all(line.split()[0] == "ok" for line in lines), out
    for name, version in (
        ("ffmpeg", "9.0.2"),
        ("MuseScore 4", "4.7.6"),
        ("Rubber Band", "4.0.0"),
        ("yt-dlp", "2026.10.26"),
        ("Basic Pitch", "0.3.0"),
        ("Node.js", "26.13.0"),
        ("Python packages", "torch "),
        ("separation model", "SEPARATION, 2 MB in /models"),
        ("MuScriptor", "medium"),
        ("Disk space", "100 GB free"),
    ):
        line = next(x for x in lines if f" {name} " in x)
        assert version in line, line
    assert "Without it" not in out and "Fix:" not in out
    assert out.strip().splitlines()[-1] == "Ready."
    assert out.splitlines()[0].startswith("bass2sheet ")


def gone(m, monkeypatch, what):
    """Take one optional thing off the machine."""
    if what == "MuseScore 4":
        os.remove(m.bin / "mscore")
    elif what == "Node.js":
        os.remove(m.bin / "node")
    elif what == "Rubber Band":
        monkeypatch.setattr(copilot, "stretcher", lambda: ("ffmpeg", "/venv/ffmpeg"))
    elif what == "yt-dlp":
        monkeypatch.setattr(copilot_adding, "downloader", lambda: None)
    elif what == "Basic Pitch":
        monkeypatch.setattr(second_opinion, "find_python", lambda explicit=None: None)
    elif what == "MuScriptor":

        def not_set_up():
            raise Bass2SheetError("MuScriptor's notes were asked for, and it is not set up")

        monkeypatch.setattr(muscriptor, "program", not_set_up)


@pytest.mark.parametrize(
    "what,costs,on_a_mac,on_linux",
    [
        ("MuseScore 4", "no PDF, only MusicXML", "brew install --cask musescore", "musescore.org"),
        ("Rubber Band", "slowed down by ffmpeg", "brew install rubberband", "sudo apt install rubberband-cli"),
        ("yt-dlp", "no links", "brew install yt-dlp", "uv tool install yt-dlp"),
        ("Node.js", "tests of the pages' scripts are skipped", "brew install node", "sudo apt install nodejs"),
        ("Basic Pitch", "gates", "./bootstrap --with-basic-pitch", "./bootstrap --with-basic-pitch"),
        ("MuScriptor", "--notes muscriptor", "docs/muscriptor.md", "docs/muscriptor.md"),
    ],
)
def test_an_optional_thing_missing_says_what_it_costs_and_how_to_fix_it(
    machine, monkeypatch, capsys, what, costs, on_a_mac, on_linux
):
    gone(machine, monkeypatch, what)
    check = by_name(machine.cache)[what]
    assert (check.status, check.required) == (doctor.MISSING, False)
    assert costs in check.without and on_a_mac in check.fix
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    assert on_linux in by_name(machine.cache)[what].fix
    assert doctor.main(["--cache-dir", machine.cache]) == 0  # optional: the tool works without it
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if f" {what} " in x)
    assert line.split()[0] == "missing" and f"Without it: {check.without}" in line and "Fix: " in line
    assert out.strip().splitlines()[-1] == "Ready. 1 optional thing is missing."


def test_a_basic_pitch_environment_without_basic_pitch_in_it(machine, monkeypatch, tmp_path):
    empty = program(tmp_path, "python-without", code=1)
    monkeypatch.setattr(second_opinion, "find_python", lambda explicit=None: empty)
    check = by_name(machine.cache)["Basic Pitch"]
    assert check.status == doctor.MISSING and "has no basic-pitch" in check.found


def test_links_turned_off_by_the_environment(machine, monkeypatch):
    monkeypatch.setattr(copilot_adding, "downloader", real_downloader)  # it reads the variable
    monkeypatch.setenv("BASS2SHEET_DOWNLOADER", "none")
    check = doctor.check_downloader()
    assert check.status == doctor.MISSING and "BASS2SHEET_DOWNLOADER" in check.found
    assert check.fix == "unset BASS2SHEET_DOWNLOADER"


def test_ffmpeg_from_the_python_environment_when_none_is_on_the_path(machine, monkeypatch, tmp_path):
    import imageio_ffmpeg

    own = program(tmp_path, "ffmpeg-own", "ffmpeg version 9.0-static Copyright")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: own)
    assert by_name(machine.cache)["ffmpeg"].found == f"9.0.2 {machine.bin / 'ffmpeg'}"
    os.remove(machine.bin / "ffmpeg")
    check = by_name(machine.cache)["ffmpeg"]
    assert check.status == doctor.OK and check.found.startswith("9.0-static, the copy inside the Python environment")


def test_no_ffmpeg_at_all_is_required_and_missing(machine, monkeypatch, capsys):
    import imageio_ffmpeg

    def none():
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", none)
    os.remove(machine.bin / "ffmpeg")
    check = by_name(machine.cache)["ffmpeg"]
    assert (check.status, check.required, check.fix) == (doctor.MISSING, True, "brew install ffmpeg")
    assert doctor.main(["--cache-dir", machine.cache]) == 1
    out = capsys.readouterr().out
    assert next(x for x in out.splitlines() if " ffmpeg " in x).split()[0] == "MISSING"
    assert out.strip().splitlines()[-1] == "Not ready. Required and missing: ffmpeg."


def test_a_python_package_missing_is_required_and_missing(machine, monkeypatch):
    real = doctor.importlib.util.find_spec
    monkeypatch.setattr(
        doctor.importlib.util, "find_spec", lambda name, package=None: None if name == "torch" else real(name, package)
    )
    check = by_name(machine.cache)["Python packages"]
    assert (check.status, check.required) == (doctor.MISSING, True)
    assert check.found == "not installed: torch" and check.fix == "./bootstrap"
    assert doctor.report(doctor.checks(machine.cache), say=lambda line: None) == 1


def test_another_python_is_required_and_missing():
    assert doctor.check_python((3, 12, 16)).status == doctor.OK
    check = doctor.check_python((3, 13, 1))
    assert (check.status, check.required) == (doctor.MISSING, True) and "this is Python 3.13.1" in check.found


def test_a_check_that_cannot_run_is_a_line_not_a_crash(machine, monkeypatch, capsys):
    def boom(explicit=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(score, "find_mscore", boom)
    check = by_name(machine.cache)["MuseScore 4"]
    assert check.status == doctor.MISSING and "could not be checked (RuntimeError: boom)" in check.found
    assert doctor.main(["--cache-dir", machine.cache]) == 0


def test_no_gpu_and_a_full_disk_are_told_and_do_not_fail(machine, monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu = real_gpu()
    assert (gpu.status, gpu.required) == (doctor.MISSING, False) and "much slower" in gpu.without
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert real_gpu().status == doctor.OK
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda path: usage(500 * GB, 498 * GB, 2 * GB))
    disk = doctor.check_disk(machine.cache)
    assert (disk.status, disk.required) == (doctor.MISSING, False) and disk.found.startswith("2 GB free")


def test_model_files_not_downloaded_yet_are_later_and_do_not_fail(machine, capsys):
    machine.there.update(separation=False, second=False)
    assert doctor.main(["--cache-dir", machine.cache]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if " separation model " in x)
    assert line.split()[0] == "later" and "SEPARATION is not downloaded yet (699 MB)" in line
    assert "Without it: the first song waits. Fix: bass2sheet-doctor --fetch-models" in line
    assert "Not downloaded yet: separation model, second model." in out
    assert out.strip().splitlines()[-1] == "Ready."
    assert machine.fetched == []  # checking downloads nothing


def test_fetch_models_downloads_what_is_not_there_and_only_that(machine, capsys):
    machine.there.update(separation=False, second=False)
    assert doctor.main(["--fetch-models", "--cache-dir", machine.cache]) == 0
    out = capsys.readouterr().out
    assert machine.fetched == ["separation", "second"]
    assert "separation model (SEPARATION): downloading about 699 MB into /models" in out
    assert "beats model (BEATS): already there, 2 MB" in out
    assert "Not downloaded yet" not in out and out.strip().splitlines()[-1] == "Ready."


def test_a_download_that_fails_is_told_and_the_exit_code_is_2(machine, capsys, monkeypatch):
    machine.there.update(separation=False, second=False)
    machine.broken["separation"] = OSError("no route to host")
    assert doctor.main(["--fetch-models", "--cache-dir", machine.cache]) == 2
    out = capsys.readouterr().out
    assert "separation model (SEPARATION): NOT downloaded. OSError: no route to host" in out
    assert machine.fetched == ["second"]  # the others are still fetched
    last = out.strip().splitlines()[-1]
    assert last.startswith("Could not download: separation model (SEPARATION).") and "HTTPS_PROXY" in last
    # with something required missing as well, the exit code says that
    monkeypatch.setattr(doctor, "check_crepe", lambda: doctor.Check("Pitch model", doctor.MISSING, required=True))
    assert doctor.main(["--fetch-models", "--cache-dir", machine.cache]) == 1


def test_a_download_that_leaves_no_files_counts_as_failed(machine, monkeypatch):
    machine.there["beats"] = False
    model = doctor.MODELS[1]
    monkeypatch.setattr(model, "fetch", lambda cache_dir: None)  # says nothing, writes nothing
    said = []
    assert doctor.fetch_models(machine.cache, say=said.append) == ["beats model (BEATS)"]
    assert any("beats model (BEATS): NOT downloaded" in x and "not where the pipeline looks" in x for x in said)


def test_reason_shows_the_error_a_library_wrapped():
    try:
        try:
            raise OSError("no network")
        except OSError:
            raise ValueError("could not load the checkpoint")  # noqa: B904 (as Beat This! raises it)
    except ValueError as e:
        assert doctor.reason(e) == "ValueError: could not load the checkpoint (OSError: no network)"


# ---- where the real model files are looked for


def test_the_separation_model_is_looked_for_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    cache = tmp_path / "cache"
    assert doctor.roformer_files(str(cache)) == []
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "BS-Roformer-SW.ckpt").write_bytes(b"x")
    assert doctor.roformer_files(str(cache)) == []  # its settings file is part of it
    (cache / "models" / "BS-Roformer-SW.yaml").write_text("x")
    assert [os.path.basename(p) for p in doctor.roformer_files(str(cache))] == [
        "BS-Roformer-SW.ckpt",
        "BS-Roformer-SW.yaml",
    ]
    check = doctor.check_model(doctor.MODELS[0], str(cache))
    assert check.status == doctor.OK and str(cache / "models") in check.found
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "elsewhere"))  # the library reads this first
    assert doctor.roformer_files(str(cache)) == []
    assert doctor.models_dir(str(cache)) == str(tmp_path / "elsewhere")


def test_the_beat_model_is_looked_for_with_pytorchs_checkpoints(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.hub, "_hub_dir", None)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert doctor.beat_files() == []
    folder = tmp_path / "hub" / "checkpoints"
    folder.mkdir(parents=True)
    (folder / "beat_this-final0.ckpt").write_bytes(b"x")
    assert doctor.beat_files() == [str(folder / "beat_this-final0.ckpt")]


def test_demucs_is_looked_for_in_the_hugging_face_cache_and_with_pytorchs_checkpoints(tmp_path, monkeypatch):
    import torch
    from huggingface_hub import constants

    pytest.importorskip("demucs.hf")  # the Demucs an Intel Mac gets knows nothing of the Hugging Face cache
    monkeypatch.setattr(torch.hub, "_hub_dir", None)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf"))
    assert doctor.demucs_files() == []
    # as a recent Demucs keeps it: in the Hugging Face cache, a list of four models and the four
    repo = tmp_path / "hf" / "models--adefossez--HTDemucs-ft"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    (snapshot / "htdemucs_ft.yaml").write_text("models: ['aa', 'bb']\n")
    (snapshot / "aa.safetensors").write_bytes(b"x")
    assert doctor.demucs_files() == []  # one of the two is not there yet
    (snapshot / "bb.safetensors").write_bytes(b"x")
    assert [os.path.basename(p) for p in doctor.demucs_files()] == [
        "htdemucs_ft.yaml",
        "aa.safetensors",
        "bb.safetensors",
    ]
    # as an older Demucs keeps it: with PyTorch's checkpoints, under the names its own list gives
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    assert doctor.demucs_files() == []
    folder = tmp_path / "torch" / "hub" / "checkpoints"
    folder.mkdir(parents=True)
    for name in ("f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th", "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th"):
        (folder / name).write_bytes(b"x")
    assert len(doctor.demucs_files()) == 4


# ---- the lines, and the command


def test_a_line_says_what_was_found_or_what_is_lost_and_the_fix():
    ok = doctor.Check("MuseScore 4", doctor.OK, "4.7.6 /opt/homebrew/bin/mscore", "no PDF", "brew install it")
    assert ok.line(12) == "  ok       MuseScore 4   4.7.6 /opt/homebrew/bin/mscore"
    missing = doctor.Check("MuseScore 4", doctor.MISSING, "not found", "no PDF, only MusicXML", "brew install it")
    assert (
        missing.line(12)
        == "  missing  MuseScore 4   not found. Without it: no PDF, only MusicXML. Fix: brew install it"
    )
    required = doctor.Check("ffmpeg", doctor.MISSING, "none", "no separation", "brew install ffmpeg", required=True)
    assert required.line().startswith("  MISSING  ffmpeg")


def test_the_command_is_installed_with_the_package():
    with open(os.path.join(HERE, "..", "pyproject.toml"), "rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    assert scripts["bass2sheet-doctor"] == "bass2sheet.doctor:main"
    assert set(doctor.INSTALL["Darwin"]) == set(doctor.INSTALL["Linux"])


def test_the_cache_is_where_the_command_says_then_where_the_environment_says(monkeypatch, tmp_path):
    from bass2sheet import common, pipeline

    monkeypatch.delenv(common.CACHE_ENV, raising=False)
    assert common.cache_home() == os.path.expanduser("~/.cache/bass2sheet")
    monkeypatch.setenv(common.CACHE_ENV, str(tmp_path / "elsewhere"))
    assert common.cache_home() == str(tmp_path / "elsewhere")  # $BASS2SHEET_CACHE moves it for every command
    assert common.cache_home("~/given") == os.path.expanduser("~/given")  # a command's --cache-dir comes first
    assert doctor.models_dir() == str(tmp_path / "elsewhere" / "models")
    assert pipeline.cache_home is common.cache_home
