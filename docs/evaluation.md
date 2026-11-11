# Evaluation

`bass2sheet-eval` runs the tool on every track of a dataset that has a bass annotation and scores what it wrote. Use it to measure a change before and after, and to check that the quality gates track real accuracy.

```
uv run bass2sheet-eval DATASET_DIR --out RUN_DIR [--conditions mix,stem] [--sample N] [--baseline FILE]
```

Each track gets its own directory under `RUN_DIR` with the tool's normal outputs. `results.jsonl` holds one row of scores per track and condition, and `summary.md` / `summary.json` hold the tables: per-track scores, where the errors are, and how each quality gate relates to the measured score. Runs are resumable: a track that already has its outputs is re-scored without being re-run, so a re-run after a code change that only affects scoring takes seconds. Stems, pitch tracks and Basic Pitch output are cached in `~/.cache/bass2sheet`, so a re-run after a change to note segmentation or the grid takes about half a minute per track.

Run at most two or three evaluation processes at once on one machine. Separation and pitch tracking are GPU-bound and each process holds several GB of GPU memory, so more processes only make every one of them slower.

## Conditions

- `mix`: the full mix goes in, as a user would run it.
- `stem`: the same, with the dataset's clean bass stem in place of the separated bass. The difference from `mix` is what separation costs. Only for datasets that ship a clean stem.

## Datasets

| Dataset | What it is | Ground truth | License | Get it |
|---|---|---|---|---|
| BabySlakh | 20 songs rendered from MIDI with sampled instruments (the first 20 tracks of Slakh2100), 16 kHz mono mixes | Exact notes; beats from the MIDI | CC BY 4.0 | [zenodo.org/records/4603870](https://zenodo.org/records/4603870), 0.9 GB |
| Slakh2100-redux | 1,710 such songs at 44.1 kHz | The same | CC BY 4.0 | [zenodo.org/records/4599666](https://zenodo.org/records/4599666), 104 GB in one archive |
| MDB-bass-synth | 71 real multitrack songs (MedleyDB) whose bass was replaced by a resynthesis | The exact f0 curve of the bass; no note list | CC BY-NC 4.0 | [zenodo.org/records/1481165](https://zenodo.org/records/1481165), 3 GB |
| RWC Popular | 100 studio pop songs | MIDI transcribed by ear, aligned to the audio at the beat level; beats | CC BY-NC 4.0 | Audio: [zenodo.org/records/17177919](https://zenodo.org/records/17177919), 13 GB. Annotations: [github.com/rwc-music/rwc-annotations](https://github.com/rwc-music/rwc-annotations) |

```
uv run bass2sheet-eval ~/data/babyslakh_16k   --out runs/babyslakh
uv run bass2sheet-eval ~/data/MDB-bass-synth  --out runs/mdb --sample 24
uv run bass2sheet-eval ~/data/RWC/RWC-P       --out runs/rwc --annotations ~/data/rwc-annotations --sample 25
```

None of these datasets is redistributed here, and the non-commercial ones are for research and evaluation only.

### What each one can tell you

- **BabySlakh / Slakh** has exact ground truth and clean stems, so it separates the stages: separation (`mix` vs `stem`), notes, and the rhythm grid.
  - The audio is synthetic and the mixes are mono, with the bass louder and cleaner than in a commercial mix. Good scores here are necessary, not sufficient.
  - Slakh's bass stems sound one octave below their MIDI notes. The harness measures that from each clean stem and corrects the reference.
  - The sampled basses start to sound 5–40 ms after their MIDI note-on, depending on the patch, so a small positive onset bias is the dataset's, not the tool's.
  - BabySlakh's bass stays between B1 and about G3: the lowest notes of a bass (E1–A#1) are not tested.
  - Tracks with two bass stems are skipped unless `--multi-bass` is given.
- **MDB-bass-synth** has real mixes and real playing with an exact pitch curve, so it tests separation plus pitch on realistic material. It has no note list, so its scores are frame-level and say nothing about note starts or the rhythm grid. The note list is deliberately not derived from the f0 curve: the rules for doing so would be the tool's own.
- **RWC Popular** is real studio pop, and the least likely of the three to have been seen by a separation model in training.
  - The MIDI was transcribed by ear and aligned by dynamic time warping, so notes can be missing, extra or tens of milliseconds off. Read the 100 ms score beside the 50 ms one.
  - There is no clean stem, so the reference's octave convention cannot be measured. `F 8va-fit` is the score after shifting by the track's commonest whole-octave difference; a large gap between it and `F` means a convention difference or a whole-song octave error, which this dataset cannot tell apart.
  - Only songs whose MIDI has exactly one bass track are used.

## Scores

Note datasets (mir_eval.transcription):

- **F / P / R**: a note counts when its onset is within 50 ms of a reference note of the same pitch. Offsets are ignored.
- **F name**: the same, ignoring the octave. **F 8va-fit**: after whole-octave alignment (see RWC above).
- **F 100ms**: onsets within 100 ms. **F+off**: the note end must also be within 20% of the note's length (or 50 ms).
- **8va↑ / 8va↓**: share of onset-matched notes an octave above or below the reference.
- **bias ms**: median of our onset minus the reference's.
- **beat pos / bar pos**: of the correct notes, the share written on the right sixteenth of the beat / of the bar. This is what the printed score depends on. **tempo×**: our tempo over the reference's.
- **Where the errors are**: every reference note is one of ok, octave, wrong pitch, merged (a repeated note we did not split), covered by another of our notes, mistimed, or missed; every note of ours is one of ok, octave, wrong pitch, split (a re-attack that is not there), mistimed, ghost, or other.

f0 datasets (mir_eval.melody, 50 cents): **OA** overall accuracy, **RPA** raw pitch accuracy, **RCA** raw chroma accuracy, voicing recall and false alarm. Because our notes are whole semitones, **note acc** also gives the share of voiced frames where we have the nearest semitone, and **8va err** the share where we are whole octaves off.

## Regression checks

```
uv run bass2sheet-eval ~/data/babyslakh_16k --out runs/new --baseline eval/baselines/babyslakh.json
```

exits 3 when a mean score over the tracks shared with the baseline falls by more than `--max-drop` (0.02). `--write-baseline FILE` records a run as the new baseline. The baselines in `eval/baselines/` hold scores only.

## Results

<!-- RESULTS -->
