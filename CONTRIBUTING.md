# Contributing

## Setup

```
./bootstrap                  # on a Mac: Homebrew's packages, .venv, the models, a self-check (docs/install.md)
uv sync --python 3.12        # or only this: .venv with the runtime and dev dependencies
uv run pytest                # unit tests + end-to-end on a synthesized song (3 to 5 minutes)
uv run ruff check . && uv run ruff format --check .
```

Python 3.12 is required: several audio ML dependencies have no wheels for newer versions yet. So is a C compiler: one dependency is compiled when the environment is created.

## Tests

Tests never use recorded music. `tests/songsynth.py` synthesizes a short song (drums, a chord pad and a plucked bass with a known line), and every test that needs audio uses it.

| Tier | What it covers | How to run |
|---|---|---|
| Unit | Each stage's pure logic: note segmentation on synthetic features, onset refinement, quantization, MusicXML (every measure adds up, ties pair, text is escaped), tab fingering, quality gates, the evaluation metrics, the dataset adapters, the report, the CLI's error paths | `uv run pytest` |
| End to end, stems given | The whole pipeline from audio files to MusicXML, MIDI, renders, report and (with MuseScore installed) PDF, scored against the song's known notes; garbage and silent bass stems must fail | `uv run pytest tests/test_e2e.py` |
| End to end, full | The mix alone goes in: real separation (BS-RoFormer and Demucs), the production pitch model, Basic Pitch when installed | `BASS2SHEET_FULL_E2E=1 uv run pytest -m full` |
| Judging | Passage selection, pitch pictures, picks, the steps of a song in order, the page's server and its background preparation, the page's own script (run under Node.js when it is installed), and update on the synthesized song | `uv run pytest tests/test_judge.py` |
| Dataset regression | A four-track BabySlakh subset must not score below `eval/baselines/babyslakh.json` | `BASS2SHEET_FULL_E2E=1 BASS2SHEET_EVAL_DATA=/path/to/babyslakh_16k uv run pytest -m full` |

The default tiers need no model download and no network, and they run in CI. The full tier downloads about 1 GB of separation models into `~/.cache/bass2sheet` on first use.

`BASS2SHEET_TEST_CREPE=full` makes the end-to-end tests use the production pitch model in place of the fast `tiny` one.

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
