# Contributing

## Setup

```
uv sync --python 3.12        # creates .venv with the runtime and dev dependencies
uv run pytest                # the default tests (about 5 minutes)
uv run ruff check . && uv run ruff format --check .
```

Python 3.12 is required: several audio ML dependencies have no wheels for newer versions yet.

## Tests

Tests never use recorded music. `tests/songsynth.py` synthesizes a short song (drums, a chord pad and a plucked bass with a known line), and every test that needs audio uses it.

| Tier | What it covers | How to run | Takes |
|---|---|---|---|
| Default | Each stage's pure logic (note segmentation on synthetic features, onset refinement, quantization, MusicXML, tab fingering, quality gates, the evaluation metrics, the dataset adapters, the report). The whole pipeline from audio files to MusicXML, MIDI, renders and report on the synthesized song with its stems given, scored against the song's known notes, and its fallbacks: which separator is tried, which result is kept. Judging, the copilot (its fixes, its servers and what they refuse, adding a song, the score and its versions), playing along, the test songs, answer keys. The command lines and what they say when something is wrong. The pages' own logic, under Node.js | `uv run pytest` | about 5 minutes |
| Coverage | The default tier, with what it reaches of the code counted, lines and branches | `uv run pytest --cov` | about 6 minutes |
| Browser | Every page opened in a real browser from the real server, and its main things done with clicks and keys: a note selected, changed, split, joined and deleted on the song's page, "Checked up to here", "Make the sheet"; the first page and adding a song; the score and its versions; playing along; rating a test song; judging a passage. No page may log an error | `uv run pytest -m browser` | under a minute |
| Full | The mix alone goes in: real separation (BS-RoFormer and Demucs), the production pitch model, the Beat This! beats, Basic Pitch when installed | `BASS2SHEET_FULL_E2E=1 uv run pytest -m full` | about a minute a separator, after the models are there |
| Dataset regression | A four-track BabySlakh subset must not score below `eval/baselines/babyslakh.json` | `BASS2SHEET_FULL_E2E=1 BASS2SHEET_EVAL_DATA=/path/to/babyslakh_16k uv run pytest -m full` | several minutes |

One file, or one test: `uv run pytest tests/test_copilot.py`, `uv run pytest -k "split_in_two"`.

### What each tier needs

- **Default: nothing beyond `uv sync`.** No model download, no network, no MuseScore, no browser; it runs in CI on Linux and macOS. Where a model or another program would run, a stand-in takes its place: the separation models, the Beat This! model, Basic Pitch and its interpreter, MuScriptor, yt-dlp, and MuseScore (`tests/mscore_standin.py` reads the MusicXML it is given and writes the pictures and positions MuseScore writes, laid out simply). What is tested is everything of bass2sheet's around them.
- **Node.js**, for the tests of the pages' own logic (`tests/*_logic.js`, `judge_page_smoke.js`). Without it they are skipped.
- **MuseScore 4 and Rubber Band are used where they are installed**: the tests that draw a score then run with the stand-in and with MuseScore itself. `BASS2SHEET_TEST_BARE=1 uv run pytest` runs the tests as a machine without them would, which is what CI has.
- **Browser: Playwright and a browser.** `uv sync --group browser`, then `uv run playwright install chromium` once (about 600 MB on disk; a Chrome or an Edge that is installed does as well). Without Playwright or a browser the tests are skipped. Once both are there, a plain `uv run pytest` runs them too: `-m "not browser"` leaves them out. `BASS2SHEET_TEST_BROWSER=chromium` (or `chrome`) asks for one browser by name and fails if it is not there. `uv sync` without the group takes Playwright out again.
- **Full: the models.** The first run downloads about 1 GB of separation models into `~/.cache/bass2sheet`, and the Beat This! model (77 MB). Basic Pitch is used when its environment is there (see the README).

`BASS2SHEET_TEST_CREPE=full` makes the end-to-end tests use the production pitch model in place of the fast `tiny` one.

### Coverage

`uv run pytest --cov` prints, for each module, the lines and branches the tests did not reach (the settings are in `pyproject.toml`); `--cov-report=html` writes pages to browse into `build/coverage/html/`. The programs the code starts as processes of their own are counted too: a song prepared by `python -m bass2sheet.copilot_adding`, and `bp_runner.py`.

The default tier reaches about 96% of the lines and branches, and no module less than 90%. CI fails below 95%: raise that number when the tests reach more, and do not lower it to let a change through. What the default tier cannot reach is what needs the real thing, and the full tier runs it: the separation models, the production pitch model, the Beat This! model and Basic Pitch.

A number is not the point. A test is worth having when it fails as soon as the behaviour it names breaks: before adding one for a line the report shows, say what a person would lose if that line were wrong.

### The pages in a browser

`tests/test_browser_copilot.py` and `tests/test_browser_judge.py` start the real server on a free port, on a working directory with the synthesized song prepared, and drive a browser without a window (Playwright). They do what a person does, with clicks and keys, and then look at the page and at what the server wrote to disk. The `browse` fixture (in `tests/conftest.py`) opens the pages; when a test is over it fails the test if a page logged an error to its console or its script threw one.

A test waits for a condition, never for a time: `page.wait_for_function(...)` with the page's own state, or a locator. Before it aims at something on the timeline it lets the page come to rest (`__quiet(200)`, and nothing playing): an answer from the server can lay the page out again.

### Writing a test

- A test's name says what holds, as a sentence: `test_a_file_at_another_tempo_in_another_key_is_found_back`.
- Use the synthesized song (`song`, `song_files`, `taste_song_files` in `tests/conftest.py`), a private cache (`cache_dir`) and `tmp_path`. Nothing is read from or written to the home directory, and nothing is fetched.
- A program the tool would start (MuseScore, yt-dlp, an interpreter with Basic Pitch) is stood in for by a small script the test writes, so the command line the tool builds and the files it expects are tested too.

## Changing transcription behaviour

Accuracy claims need ground truth. Before and after a change to separation, pitch tracking, note segmentation or the rhythm grid:

1. Run the evaluation (see [docs/evaluation.md](docs/evaluation.md)): `uv run bass2sheet-eval <dataset> --out <run dir> --baseline eval/baselines/<dataset>.json`. It exits 3 when a mean score falls by more than 0.02.
2. For quick iteration on note segmentation, `python dev/seg_eval.py <BabySlakh dir> 'param=value' ...` re-scores cached stems and pitch tracks in seconds.
3. Tune on BabySlakh or Slakh's validation split first, and keep the other datasets for checking. When a failure shows only on real recordings, a few songs of another dataset may guide the fix: name them, and report their scores apart from the rest (docs/evaluation.md does this for the nine RWC songs behind the re-pluck splitting).
4. When a change improves the scores on purpose, update the baseline with `--write-baseline` and say so in the pull request.

The standing choices in `common.DEFAULTS` (which separator goes first, how re-plucks, rhythm and tab are written, where beats come from) are matters of fit as well as accuracy. Change one on the strength of picks from `bass2sheet-judge` ([docs/judging.md](docs/judging.md)) as well as the scores, and say which in the pull request.

Quality-gate thresholds follow the same rule: a gate should be shown to track real accuracy (the evaluation report has a gate-vs-accuracy table) before it can fail a run.

## Style

- `ruff format` and `ruff check` must pass (line length 120).
- Errors a user can act on are `Bass2SheetError` with a message that says what went wrong and what to do; the CLI prints them and exits 1.
- No audio, model weights or dataset files in the repository.
