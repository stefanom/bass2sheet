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

Version 0.2.0 on an Apple M4 Max, full mixes in, default settings. `eval/baselines/` holds the per-track scores.

### BabySlakh (16 songs with a single bass stem)

| Condition | F | P | R | F name | F 100ms | F+off | 8va↓ | bias | beat pos | bar pos |
|---|---|---|---|---|---|---|---|---|---|---|
| mix, mean | 0.88 | 0.91 | 0.87 | 0.91 | 0.88 | 0.73 | 0.03 | +15 ms | 0.95 | 0.66 |
| mix, median | 0.95 | 0.94 | 0.94 | 0.95 | 0.95 | 0.84 | 0.00 | +11 ms | 0.98 | 0.77 |
| stem, mean | 0.92 | 0.92 | 0.93 | 0.92 | 0.92 | 0.74 | 0.00 | +13 ms | 0.95 | 0.66 |
| stem, median | 0.95 | 0.94 | 0.95 | 0.95 | 0.96 | 0.79 | 0.00 | +12 ms | 0.98 | 0.77 |

- 12 of 16 songs score F ≥ 0.90 from the mix. Of the reference's notes, 87% are right, 7% missed, 3% an octave off, 1% merged into a neighbour; of ours, 91% are right, 2% are re-attacks that are not there, 1% ghosts.
- Separation costs 0.04 of mean F, most of it on three songs: Track00019 (0.49 from the mix, 0.87 from the clean stem; the difference is notes an octave low, not yet explained), Track00009 and Track00008.
- Track00009 scores 0.38 from the mix and 0.55 from the clean stem. Its notes last about 100 ms, and the pitch tracker reports no pitch on most of them.
- The grid is measured on the 11 songs whose MIDI has a 4/4 time signature throughout. Tempo is within 2% on 10 of them. Bar position is 0.95 or more on 5. On the others the bar lines are off for part or all of the song; the meter gate (the share of the beat tracker's downbeats that fit one 4/4 phase) is low on those songs, and its rank correlation with bar position is +0.84.

### MDB-bass-synth (24 of 71 songs, evenly spaced through the list)

| | OA | RPA | RCA | note acc | 8va err | voicing recall | voicing false alarm |
|---|---|---|---|---|---|---|---|
| mean | 0.81 | 0.77 | 0.78 | 0.77 | 0.01 | 0.92 | 0.05 |
| median | 0.85 | 0.81 | 0.82 | 0.81 | 0.00 | 0.96 | 0.02 |

Note acc ranges from 0.46 to 0.96. The five lowest are TablaBreakbeatScience_WhoIsIt (0.46), MusicDelta_Disco (0.47), TheSoSoGlos_Emergency (0.51), Creepoid_OldTree (0.59) and EthanHein_BluesForNofi (0.61); why has not been looked into.

### RWC Popular (24 of the 75 songs with one bass track, evenly spaced)

| | F | P | R | F name | F 100ms | F+off | bias | bar pos |
|---|---|---|---|---|---|---|---|---|
| mean | 0.68 | 0.70 | 0.66 | 0.71 | 0.74 | 0.42 | +14 ms | 1.00 |
| median | 0.63 | 0.70 | 0.67 | 0.66 | 0.73 | 0.36 | +13 ms | 1.00 |

- 6 songs score F 100ms ≥ 0.90 and 7 score below 0.60.
- Of the reference's notes, 66% are right, 14% merged (a repeated note we wrote as one), 7% covered by a note of another pitch, 6% the wrong pitch, 4% missed, 3% an octave off. Of ours, 70% are right, 9% are re-attacks the reference does not have, 8% start inside a reference note of another pitch, 6% are the wrong pitch.
- Bar position is 0.96 or more on all 21 songs with a 4/4 reference. Beat This! was trained on RWC Popular, so this is not a test of the grid.
- Nine of these songs were used to choose how held notes are split at re-plucks. They average F 100ms 0.78; the other fifteen average 0.72.

### Verdicts against measured scores

The score is F for BabySlakh, note acc for MDB-bass-synth and F 100ms for RWC.

| Verdict | Runs | Mean score | Lowest | Runs at 0.85 or more |
|---|---|---|---|---|
| PASS | 15 | 0.96 | 0.90 | 100% |
| WARN | 47 | 0.85 | 0.44 | 51% |
| FAIL | 18 | 0.60 | 0.38 | 6% |

The gate thresholds and the "five warnings fail" rule were set on these runs, so this table is in-sample. The gate that tracks the score best is the share of note starts confirmed by Basic Pitch (rank correlation +0.80 on BabySlakh, +0.77 on MDB, +0.59 on RWC). The meter gate does not track note accuracy; it tracks bar position, which is what it is for. Notes in bars marked "check" are right 68% of the time on BabySlakh, against 88% in other bars; on RWC, 41% against 70%.

### What changed the scores in 0.2.0

| | 0.1.0 | 0.2.0 |
|---|---|---|
| BabySlakh, mean F from the mix | 0.75 | 0.88 |
| BabySlakh, mean F 100ms from the mix | 0.89 | 0.88 |
| BabySlakh, mean bar position | 0.49 | 0.66 |

- **Note starts** are placed where the pluck begins to sound. 0.1.0 found the same notes 15–25 ms late, which is why its F at 100 ms was already 0.89. This accounts for nearly all of the gain in F.
- **Beat This!** replaced the built-in beat tracker, which chose the wrong pulse level on 7 BabySlakh songs.
- **Held notes are split at re-plucks.** Measured on the same stems, this moved mean F on five RWC songs from 0.60 to 0.69 and on BabySlakh from 0.888 to 0.882.
