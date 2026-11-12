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
| IDMT-SMT-Bass-Single-Track | 17 lines played on a real bass guitar, alone, in five plucking styles | Exact notes (948), with string, fret, plucking and expression style | CC BY-NC-ND 4.0 | [zenodo.org/records/7544099](https://zenodo.org/records/7544099), 20 MB |

```
uv run bass2sheet-eval ~/data/babyslakh_16k   --out runs/babyslakh
uv run bass2sheet-eval ~/data/MDB-bass-synth  --out runs/mdb --sample 24
uv run bass2sheet-eval ~/data/RWC/RWC-P       --out runs/rwc --annotations ~/data/rwc-annotations --sample 25
```

The last one has no mix, so it does not go through `bass2sheet-eval`: `python dev/idmt_eval.py ~/data/idmt-bass` runs the note finder on the bass tracks as they are.

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
- **IDMT-SMT-Bass-Single-Track** is the only one with a real bass guitar and exact notes. It tests the note finder alone: what it does with slaps, pops, muted notes and lines that jump. It says nothing about separation, and its lines are short (20 s).

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
| mix, mean | 0.91 | 0.90 | 0.92 | 0.94 | 0.92 | 0.74 | 0.03 | +16 ms | 0.95 | 0.66 |
| mix, median | 0.95 | 0.94 | 0.96 | 0.95 | 0.96 | 0.84 | 0.00 | +11 ms | 0.98 | 0.77 |
| stem, mean | 0.93 | 0.91 | 0.96 | 0.94 | 0.94 | 0.74 | 0.00 | +13 ms | 0.95 | 0.66 |
| stem, median | 0.95 | 0.94 | 0.97 | 0.95 | 0.95 | 0.78 | 0.00 | +11 ms | 0.98 | 0.78 |

- 12 of 16 songs score F ≥ 0.90 from the mix, and all but one score above 0.84. Of the reference's notes, 92% are right, 3% missed, 3% an octave off, 1% merged into a neighbour; of ours, 90% are right, 2% are re-attacks that are not there, 2% ghosts.
- Separation costs 0.02 of mean F, all of it on one song: Track00019 scores 0.50 from the mix and 0.87 from the clean stem. The difference is notes an octave low. A sawtooth synth plays below the bass guitar there, 9 dB louder at the octave under the bass's notes, and the separator puts both in the bass track. The pitch tracker then follows the synth: the lowest line of the song, but not the reference's.
- Track00009, whose notes last about 100 ms, scores 0.85 from the mix and 0.79 from the clean stem. It scored 0.45 and 0.55 while the pitch path could not jump: most of its notes then had no pitch.
- The grid is measured on the 11 songs whose MIDI has a 4/4 time signature throughout. Tempo is within 2% on 10 of them. Bar position is 0.95 or more on 5. On the others the bar lines are off for part or all of the song; the meter gate (the share of the beat tracker's downbeats that fit one 4/4 phase) is low on those songs, and its rank correlation with bar position is +0.84.

### MDB-bass-synth (24 of 71 songs, evenly spaced through the list)

| | OA | RPA | RCA | note acc | 8va err | voicing recall | voicing false alarm |
|---|---|---|---|---|---|---|---|
| mean | 0.82 | 0.78 | 0.79 | 0.78 | 0.01 | 0.94 | 0.06 |
| median | 0.85 | 0.81 | 0.82 | 0.81 | 0.00 | 0.97 | 0.03 |

Note acc ranges from 0.47 to 0.96. The five lowest are TablaBreakbeatScience_WhoIsIt (0.47), MusicDelta_Disco (0.50), TheSoSoGlos_Emergency (0.52), EthanHein_BluesForNofi (0.61) and Creepoid_OldTree (0.70); why has not been looked into. From the dataset's own bass track the mean is 0.84, so separation costs 0.06 here.

### RWC Popular (24 of the 75 songs with one bass track, evenly spaced)

| | F | P | R | F name | F 100ms | F+off | bias | bar pos |
|---|---|---|---|---|---|---|---|---|
| mean | 0.70 | 0.71 | 0.70 | 0.73 | 0.76 | 0.44 | +15 ms | 1.00 |
| median | 0.64 | 0.67 | 0.68 | 0.69 | 0.71 | 0.37 | +13 ms | 1.00 |

- 6 songs score F 100ms ≥ 0.90 and 4 score below 0.60.
- Of the reference's notes, 70% are right, 12% merged (a repeated note we wrote as one), 6% covered by a note of another pitch, 6% the wrong pitch, 2% missed, 3% an octave off. Of ours, 71% are right, 9% are re-attacks the reference does not have, 8% start inside a reference note of another pitch, 6% are the wrong pitch.
- Of the 777 reference notes that follow a jump up of a fifth or more, 62% are written right; 53% were while the pitch path could not jump.
- Bar position is 0.98 or more on all 21 songs with a 4/4 reference. Beat This! was trained on RWC Popular, so this is not a test of the grid.
- Nine of these songs were used to choose how held notes are split at re-plucks. They average F 100ms 0.79; the other fifteen average 0.73.

### Real bass guitar played alone (IDMT-SMT-Bass-Single-Track, 17 lines, 948 notes)

`python dev/idmt_eval.py DIR --basic-pitch --yourmt3`. "Jumps" is the share of notes that follow a jump up of a fifth or more (91 of them) and are written with the right pitch at the right time.

| Note finder | F | P | R | F 100ms | An octave off | Other wrong pitch | Jumps |
|---|---|---|---|---|---|---|---|
| bass2sheet | 0.78 | 0.70 | 0.89 | 0.82 | 1% | 3% | 79% |
| bass2sheet before the pitch path could jump | 0.76 | 0.70 | 0.84 | 0.80 | 2% | 2% | 46% |
| bass2sheet without splitting held notes at re-plucks | 0.85 | 0.82 | 0.88 | 0.89 | 0% | 1% | 77% |
| Basic Pitch | 0.70 | 0.62 | 0.81 | 0.76 | 4% | 5% | 78% |
| YourMT3+ (YPTF.MoE+Multi, no pitch shift) | 0.77 | 0.80 | 0.75 | 0.78 | 5% | 10% | 79% |

- **Lines that jump were written too low.** torchcrepe's Viterbi decoder lets the pitch move by a few bins per frame and no further. After a jump of a fifth or more the path lagged or stayed on the note before, where CREPE's probability is low, so the new note counted as unpitched and the old, lower note was held over it. A second cause sat in the octave rule: when CREPE and pYIN were an octave apart it took the lower note whenever that octave had a third of the energy, and it has whenever the note before still rings. pYIN smooths too, so on all 21 such notes in this set CREPE was right and pYIN was on the old note. The path may now leap (`pitch.decode_pitch`), and CREPE's octave stands when it is confident (`SegParams.trust_per`).
- **By plucking style**, the share of notes found: finger 94%, muted 95%, pick 93%, slap with the thumb 80%, slap pops 72% (36 of 50; 9 of 50 before).
- **Splitting held notes at re-plucks costs 0.07 here**: 14% of our notes are re-attacks that are not in the reference. On RWC Popular the same rule gains 0.02, because there the separated bass smears repeated notes into one. The default has not changed; `--replucks clear` is the stricter policy, and the judging page asks which you prefer on your own songs.
- **YourMT3+** places notes cleanly (almost no extra notes) but names the wrong pitch three times as often, and its spectrogram starts at 50 Hz, below which the lowest five notes of a bass have their fundamental. On bass tracks separated from six commercial recordings it was not usable: 14 notes for a whole song with a dark, sustained bass, a bass line labelled as guitar throughout, some 2,700 notes for a distorted one, and no note below B1 in any of them. It was trained on Slakh and on RWC Popular, so it cannot be compared on those.

### Notes played together

`python dev/together_eval.py IDMT_DIR --dataset BABYSLAKH_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS`

None of the datasets has them to speak of: 4 pairs in BabySlakh's 6,099 reference notes and 37 in RWC's 49,229, half of them octaves. So the test lines are made here, from 218 real notes of the IDMT set played alone, two together (a third to a tenth apart) or an octave apart, which must come out as one note.

| Line | Pairs | Both notes, as one event | One of the two | Nothing | A wrong second note | Single notes and octaves written as two |
|---|---|---|---|---|---|---|
| Each note ends where the next begins | 77 | 60 (59 + 1 in pieces) | 13 | 2 | 2 | 0 of 73 |
| Each note rings on under the next | 67 | 52 (49 + 3 in pieces) | 14 | 1 | 0 | 1 of 83 |
| The first line, one note at a time (`--together single`) | 77 | 0 | 65 | 12 | | 0 of 73 |

- **About three pairs in four are found, and what is found is right.** The pitch trackers alone write one of the two notes (more often the upper one, which is not the bass note) or nothing at all.
- **Where there are none, few are written.** On the bass tracks separated from BabySlakh and RWC Popular, 6 of 6,363 and 14 of 14,738 notes come out as two (0.09%). The note finder's scores there do not move (F 0.911 and 0.690 with and without).
- **Slap playing draws false pairs**: 8 of 1,207 notes on the IDMT lines (0.7%), nearly all on the slap tracks.
- A second note must pass four tests (`together.py`): it is a note of its own in the spectrum, with two partials the first note cannot explain; it starts with the first (a note still ringing does not); it lasts with it (the pluck's noise does not); and it is louder in the bass track than in the rest of the record (a kick drum or a piano note that leaked into the bass track is not). On one of the six songs judged by ear the last test turned down 14 of 38 candidates, most of them with the kick drum's pitch as the lower note.
- Mixed single notes are not a double stop played by a hand: the two strings of a real one start within a few milliseconds of each other and share the instrument's resonances. Recordings of real double stops with exact notes would be the better test.

### Verdicts against measured scores

The score is F for BabySlakh, note acc for MDB-bass-synth and F 100ms for RWC.

| Verdict | Runs | Mean score | Lowest | Runs at 0.85 or more |
|---|---|---|---|---|
| PASS | 15 | 0.96 | 0.89 | 100% |
| WARN | 49 | 0.85 | 0.57 | 51% |
| FAIL | 16 | 0.65 | 0.47 | 12% |

The gate thresholds and the "five warnings fail" rule were set on the runs of the version before the pitch path could jump, which gave nearly the same table, so this is close to in-sample. The gate that tracks the score best is the share of note starts confirmed by Basic Pitch (rank correlation +0.79 on BabySlakh, +0.76 on MDB, +0.57 on RWC). The meter gate does not track note accuracy; it tracks bar position, which is what it is for. Notes in bars marked "check" are right 69% of the time on BabySlakh, against 88% in other bars; on RWC, 42% against 70%.

### Choosing between the two separators

When the BS-RoFormer result fails its gates, Demucs is tried and one of the two results is kept. Ten songs with ground truth were transcribed from both separators: nine where the RoFormer result failed its gates and one where it did not. Demucs was the more accurate on 5 (by 0.02 to 0.09 F), RoFormer on 3 (by 0.07 to 0.20), and 2 were level. Keeping the result with fewer failed gates, then the one with more note starts confirmed by Basic Pitch, picks the better or an equal result on 8 of the 10. The earlier rule, which counted warning gates second, did so on 7. That is no real difference on ten songs. The rule was changed because warning counts are measured against two different stems, and because the share of confirmed note starts is the gate that tracks accuracy best.

### What changed the scores in 0.2.0

| | 0.1.0 | 0.2.0 |
|---|---|---|
| BabySlakh, mean F from the mix | 0.75 | 0.91 |
| BabySlakh, mean F 100ms from the mix | 0.89 | 0.92 |
| BabySlakh, mean bar position | 0.49 | 0.66 |

- **Note starts** are placed where the pluck begins to sound. 0.1.0 found the same notes 15–25 ms late, which is why its F at 100 ms was already 0.89. This accounts for most of the gain in F (0.75 to 0.88).
- **A pitch path that can jump** accounts for the rest (0.88 to 0.91 on BabySlakh, 0.68 to 0.70 on RWC, 0.77 to 0.78 on MDB): notes after a large jump, and very short notes, had no pitch before. See the real-bass section above for the cause.
- **Beat This!** replaced the built-in beat tracker, which chose the wrong pulse level on 7 BabySlakh songs.
- **Held notes are split at re-plucks.** Measured on the same stems, this moved mean F on the 24 RWC songs from 0.66 to 0.68 (0.72 to 0.75 at 100 ms) and on BabySlakh from 0.888 to 0.884. The gain is concentrated in songs with driving repeated notes.
