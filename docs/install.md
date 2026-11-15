# Installing bass2sheet

On a Mac with [Homebrew](https://brew.sh), `./bootstrap` does all of it. This page says what it does, what each piece is for, how to check an install, and how to do it by hand on a Mac or on Linux.

The numbers on this page were measured on 14 November 2026 on an Apple M4 Max MacBook, in a fresh clone with an empty home directory, so that the Python packages and the models were downloaded again. The line was fast (the 699 MB model came in 27 seconds); on yours the times follow the download sizes.

## What bass2sheet needs

| Thing | What it is for | Without it |
|---|---|---|
| Python 3.12 | The tool runs on it. 3.12 and no newer: several of the audio models have no builds for later ones | Nothing runs |
| [uv](https://docs.astral.sh/uv/) | Creates the Python environment from `uv.lock` | No environment |
| A C compiler | One Python package (diffq, which the separation library asks for) has no built package for Python 3.12 and is compiled when the environment is created. On a Mac the compiler comes with the Xcode command line tools, which Homebrew needs as well | The environment is not created |
| MuseScore 4 | Engraves the PDF, and the pictures of the score in the copilot, in play-along and in judging | MusicXML and no PDF; no score pictures |
| ffmpeg | The separation library and yt-dlp call it | bass2sheet uses the copy inside its Python environment |
| Rubber Band | Slows a song down without changing its pitch, in the copilot and in play-along | ffmpeg does it, and a bass line comes out rougher |
| yt-dlp | Adding a song to the copilot by a link | The copilot takes files only |
| Node.js | The tests run the pages' scripts under it. The tool does not use it | Those tests are skipped |
| The model files | Separation, beats (below) | The first song downloads them |
| Basic Pitch, optional | A second opinion on the notes, behind two of the quality gates | Those two gates are skipped |
| MuScriptor, optional | Another note finder ([muscriptor.md](muscriptor.md)) | `--notes muscriptor` is refused |

`uv run bass2sheet-doctor` prints this list for your machine: see [The self-check](#the-self-check).

## ./bootstrap

```
./bootstrap                      # everything below
./bootstrap --check              # change nothing: say what is there and what is not
./bootstrap --no-models          # skip the model files; the first song downloads them
./bootstrap --with-basic-pitch   # also Basic Pitch, in an environment of its own
./bootstrap --with-muscriptor    # also the tools MuScriptor's program is built with
```

It is safe to run again: each step looks first and does only what is missing. A second run took 6 seconds. When a step fails the script stops there, says which step and what to do, and keeps what was done.

What it does on a Mac:

1. **Homebrew.** It looks for `brew` on the `PATH` and in the two places Homebrew installs itself. If there is none it prints the command from brew.sh and stops: it does not install Homebrew.
2. **Homebrew's packages.** `brew bundle install --no-upgrade` on the [Brewfile](../Brewfile): Python 3.12, uv, MuseScore 4, ffmpeg, Rubber Band, yt-dlp and Node.js. What is installed already is left as it is, at the version it has. ffmpeg is skipped where Homebrew's `bin` already holds an `ffmpeg`, because `ffmpeg-full` provides one too and the two cannot both be linked. On the test machine, which lacked Python 3.12 and uv, this step downloaded 33 MB and took 12 seconds. MuseScore is the large one, a 207 MB download.
3. **The Python environment.** `uv sync --locked` creates `.venv` from `uv.lock` with Homebrew's Python 3.12, whatever other Python is on the `PATH`. An environment that is already there keeps the Python it has. 86 packages, 435 MB downloaded, 1.3 GB on disk, 16 seconds. One of them is compiled; the script names the SDK for the compiler (see [Trouble](#trouble)).
4. **The model files**, 1.1 GB, so that the first song does not wait for them:

   | Model | Size | Where it goes | Used |
   |---|---|---|---|
   | BS-RoFormer SW, separation | 699 MB | `~/.cache/bass2sheet/models` | For every song |
   | Beat This!, beats and bar lines | 81 MB | `~/.cache/torch/hub/checkpoints` | For every song |
   | Demucs `htdemucs_ft`, separation | 336 MB | `~/.cache/huggingface/hub` | When BS-RoFormer fails on a song, or its result fails the quality gates |

   The pitch model (CREPE, 89 MB) is inside a Python package and needs no download. A model that cannot be downloaded leaves the run "not complete": the script says which, finishes the rest and exits with an error. Run it again later.
5. **The self-check**, and three commands to try.

The whole first run took 65 seconds on the test machine, with Homebrew's packages in place. The beat model was not among what it downloaded: see [What is not proven](#what-is-not-proven).

On Linux the script installs no system packages. It lists them, then does steps 3 to 5. It needs uv and a C compiler.

Behind a proxy, set `HTTPS_PROXY` before you run it. Homebrew, uv and the model downloads all read it. The script names no proxy of its own.

## The self-check

```
uv run bass2sheet-doctor
```

One line for each thing the tool needs. This is the test machine after `./bootstrap --with-basic-pitch`, on the day the beat model could not be had:

```
bass2sheet 0.2.0 on macOS 27.2.1 (arm64)
  ok       Python 3.12              3.12.16 /private/tmp/rs/boot/clone/.venv/bin/python
  ok       Python packages          torch 2.15.0, audio-separator 0.48.0, demucs 4.1.0, torchcrepe 0.0.24, beat-this 1.1.0
  ok       GPU                      mps, the Mac's own
  ok       ffmpeg                   9.0.2 /opt/homebrew/bin/ffmpeg
  ok       Pitch model              CREPE, 89 MB inside the torchcrepe package
  ok       Separation model         BS-RoFormer SW, 699 MB in ~/.cache/bass2sheet/models
  later    Beat model               Beat This! is not downloaded yet (81 MB). Without it: the first song downloads it. If that fails, `bass2sheet` and the copilot go on with the built-in beat tracker and say so; the judging page, which compares the two, cannot take a song. Fix: bass2sheet-doctor --fetch-models
  ok       Second separation model  Demucs htdemucs_ft, 336 MB in ~/.cache/huggingface/hub
  ok       MuseScore 4              4.7.6 /opt/homebrew/bin/mscore
  ok       Rubber Band              4.0.0 /opt/homebrew/bin/rubberband
  ok       yt-dlp                   2026.10.26 /opt/homebrew/bin/yt-dlp
  ok       Basic Pitch              0.3.0 /private/tmp/rs/boot/clone/.venv-bp/bin/python
  missing  MuScriptor               not set up. Without it: `--notes muscriptor` is refused; bass2sheet's own note finder is not affected. Fix: see docs/muscriptor.md
  ok       Node.js                  26.13.0 /opt/homebrew/bin/node
  ok       Disk space               132 GB free where the cache is (~/.cache/bass2sheet)
Not downloaded yet: beat model. To download now: bass2sheet-doctor --fetch-models
Ready. 1 optional thing is missing.
```

- `ok`: it is there, with its version and its place.
- `missing`: it is not. The line says what stops working without it and the one command that fixes it. `MISSING`, in capitals, is something the tool cannot work without.
- `later`: a model file that is not downloaded yet. The first song that needs it fetches it; `uv run bass2sheet-doctor --fetch-models` fetches it now.

The exit code is 0 when what is required is there, 1 when it is not, and 2 when it is there and a model that `--fetch-models` was asked for could not be downloaded. `--cache-dir DIR` checks another cache, as on the other commands. It takes about 3 seconds and changes nothing unless you pass `--fetch-models`.

## By hand

### On a Mac

```
brew bundle install --no-upgrade
uv sync --locked --python "$(brew --prefix)/opt/python@3.12/bin/python3.12"
uv run bass2sheet-doctor --fetch-models
```

Without Homebrew: install uv with the command from its site, run `uv sync --locked --python 3.12` (uv downloads a Python 3.12 if it finds none), install the Xcode command line tools (`xcode-select --install`) and MuseScore 4 from musescore.org. bass2sheet finds MuseScore in `/Applications`.

Homebrew has built packages of everything in the Brewfile for Apple Silicon. For Intel Macs it has none today, so there it compiles them from source. That, and bass2sheet on an Intel Mac, are not tested.

### On Linux

```
sudo apt install build-essential        # the C compiler
curl -LsSf https://astral.sh/uv/install.sh | sh
./bootstrap                             # or: uv sync --locked --python 3.12 && uv run bass2sheet-doctor --fetch-models
```

Then, as you need them: MuseScore 4 from musescore.org, with `mscore` on the `PATH` (or `--mscore PATH` on the commands); `sudo apt install rubberband-cli ffmpeg nodejs`; `uv tool install yt-dlp`. The self-check names the ones it does not find. Linux is covered by CI for the tests that need no models; a whole song on Linux is not tested here.

## Basic Pitch

Basic Pitch is the second opinion behind two of the quality gates. Its TensorFlow and ONNX packages do not share an environment with PyTorch well, so it gets one of its own:

```
./bootstrap --with-basic-pitch
```

or by hand:

```
uv venv --python 3.12 .venv-bp
uv pip install --python .venv-bp/bin/python basic-pitch onnxruntime 'setuptools<81'
```

About 350 MB are downloaded, `.venv-bp` takes 1.5 GB, and it took 29 seconds. bass2sheet finds `.venv-bp` in the checkout, beside it, or in `~/.cache/bass2sheet`; `BASS2SHEET_BASIC_PITCH_PYTHON` names another interpreter. Its first run on the test machine took 68 seconds for a 34-second song.

## MuScriptor

`./bootstrap --with-muscriptor` installs cmake and ninja with Homebrew. The build also needs Xcode, for its C++ compiler and its Metal toolchain. Building the program and getting the weights stays with you: the weights have conditions of their own, and nothing here downloads them. See [muscriptor.md](muscriptor.md).

## Where things go, and how to move them

| What | Size | Where | To move it |
|---|---|---|---|
| The Python environment | 1.3 GB | `.venv` in the checkout | |
| uv's downloads | 1.3 GB | `~/.cache/uv` | `UV_CACHE_DIR`; `uv cache clean` empties it |
| The separation model | 699 MB | `~/.cache/bass2sheet/models` | `BASS2SHEET_CACHE` for every command, `--cache-dir DIR` for one, or `AUDIO_SEPARATOR_MODEL_DIR` for the model alone |
| Each song's separated tracks and pitch | about 0.45 GB for a 4-minute song | `~/.cache/bass2sheet` | `BASS2SHEET_CACHE` for every command, `--cache-dir DIR` for one |
| The beat model | 81 MB | `~/.cache/torch/hub/checkpoints` | `TORCH_HOME` |
| The second separation model | 336 MB | `~/.cache/huggingface/hub` | `HF_HOME` |
| Basic Pitch | 1.5 GB | `.venv-bp` in the checkout | `BASS2SHEET_BASIC_PITCH_PYTHON` |

The cache of songs grows with every song and is never emptied by the tool. Its files can be deleted at any time: a song whose files are gone is separated again. `bass2sheet-copilot serve` has no `--cache-dir`: songs added on its page use `~/.cache/bass2sheet`.

To remove bass2sheet: delete the checkout and `~/.cache/bass2sheet`, then the two other model files if nothing else uses them.

## Trouble

- **`unknown architecture` when a package is compiled** (`ld: tapi error: malformed file ... libSystem.B.tbd`). The Mac's Xcode is older than its command line tools, and the compiler takes an SDK its linker cannot read. `./bootstrap` avoids it by naming the SDK. By hand: `export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)`, then `uv sync` again.
- **A model could not be downloaded.** Run `./bootstrap` or `uv run bass2sheet-doctor --fetch-models` again: what is there is kept. Without the beat model `bass2sheet` and the copilot go on with the built-in beat tracker and say so (the copilot on the song's "Beats and bars" step and on its card). That tracker often counts at the wrong level, half the tempo for the most part, so the bar lines may be wrong; once the model is there, add the song again and its bars are found anew, with your fixes kept. The judging page, which compares the two trackers, cannot take a song. Without the separation model nothing can be separated.
- **`uv: command not found` in a new terminal.** Homebrew is not on your `PATH`. The script's last lines give the line to add to `~/.zprofile`.

## What is not proven

- Homebrew installing everything in the Brewfile on a Mac that has none of it. The test machine had all but Python 3.12 and uv, so only those two were installed from nothing. For the others `brew bundle` could only find them there.
- The beat model's download. On the day of the test its server refused the test machine's network address (the machine shares a proxy address with many others, and the server's brute-force protection had it blocked). `./bootstrap` reported the failure as designed. The songs in the test were written with the built-in beat tracker.
- A first launch of MuseScore on a Mac that has never run it, an Intel Mac, Linux beyond the tests in CI, and the MuScriptor build.
