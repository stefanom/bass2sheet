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
| mix, mean | 0.91 | 0.91 | 0.92 | 0.94 | 0.92 | 0.73 | 0.03 | +15 ms | 0.95 | 0.66 |
| mix, median | 0.96 | 0.95 | 0.96 | 0.96 | 0.97 | 0.83 | 0.00 | +11 ms | 0.98 | 0.77 |
| stem, mean | 0.94 | 0.92 | 0.95 | 0.94 | 0.94 | 0.74 | 0.00 | +13 ms | 0.95 | 0.66 |
| stem, median | 0.95 | 0.94 | 0.97 | 0.95 | 0.95 | 0.78 | 0.00 | +11 ms | 0.98 | 0.78 |

- 12 of 16 songs score F ≥ 0.90 from the mix, and all but one score above 0.84. Of the reference's notes, 92% are right, 2% missed, 3% an octave off, 1% merged into a neighbour; of ours, 91% are right, 2% are re-attacks that are not there, 1% ghosts.
- Separation costs 0.02 of mean F, all of it on one song: Track00019 scores 0.49 from the mix and 0.89 from the clean stem. The difference is notes an octave low. A sawtooth synth plays below the bass guitar there, 9 dB louder at the octave under the bass's notes, and the separator puts both in the bass track. The pitch tracker then follows the synth: the lowest line of the song, but not the reference's.
- Track00009, whose notes last about 100 ms, scores 0.85 from the mix and 0.80 from the clean stem. It scored 0.45 and 0.55 while the pitch path could not jump: most of its notes then had no pitch.
- The grid is measured on the 11 songs whose MIDI has a 4/4 time signature throughout. Tempo is within 2% on 10 of them. Bar position is 0.95 or more on 5. On the others the bar lines are off for part or all of the song; the meter gate (the share of the beat tracker's downbeats that fit one 4/4 phase) is low on those songs, and its rank correlation with bar position is +0.84.

### MDB-bass-synth (24 of 71 songs, evenly spaced through the list)

| | OA | RPA | RCA | note acc | 8va err | voicing recall | voicing false alarm |
|---|---|---|---|---|---|---|---|
| mean | 0.81 | 0.77 | 0.78 | 0.77 | 0.01 | 0.93 | 0.06 |
| median | 0.84 | 0.80 | 0.80 | 0.80 | 0.00 | 0.97 | 0.03 |

Note acc ranges from 0.44 to 0.96. The five lowest are MusicDelta_Disco (0.44), TablaBreakbeatScience_WhoIsIt (0.47), TheSoSoGlos_Emergency (0.52), EthanHein_BluesForNofi (0.59) and Creepoid_OldTree (0.68); why has not been looked into. From the dataset's own bass track the mean is 0.82, so separation costs 0.05 here.

These scores are 0.01 lower than before slides stopped being written as notes (see the last section): this set is scored frame by frame against the pitch curve, so a slide written as the pitches it passes through counts as right, and a slide written as the note before it does not.

### RWC Popular (24 of the 75 songs with one bass track, evenly spaced)

| | F | P | R | F name | F 100ms | F+off | bias | bar pos |
|---|---|---|---|---|---|---|---|---|
| mean | 0.71 | 0.73 | 0.70 | 0.74 | 0.77 | 0.45 | +15 ms | 1.00 |
| median | 0.67 | 0.71 | 0.68 | 0.72 | 0.73 | 0.37 | +14 ms | 1.00 |

- 7 songs score F 100ms ≥ 0.90 and 3 score below 0.60.
- Of the reference's notes, 70% are right, 12% merged (a repeated note we wrote as one), 7% covered by a note of another pitch, 6% the wrong pitch, 2% missed, 3% an octave off. Of ours, 73% are right, 9% are re-attacks the reference does not have, 6% start inside a reference note of another pitch, 6% are the wrong pitch.
- Of the 777 reference notes that follow a jump up of a fifth or more, 62% are written right; 53% were while the pitch path could not jump.
- Bar position is 0.98 or more on all 21 songs with a 4/4 reference. Beat This! was trained on RWC Popular, so this is not a test of the grid.
- Nine of these songs were used to choose how held notes are split at re-plucks. They average F 100ms 0.80; the other fifteen average 0.75.
- The scores are of the notes as heard (`--lowest none`). With the tool's default, which writes nothing below the E string of a four-string bass, the mean F is 0.02 lower: 4% of these songs' notes are below E1, played on five-string or synth basses.

### Real bass guitar played alone (IDMT-SMT-Bass-Single-Track, 17 lines, 948 notes)

`python dev/idmt_eval.py DIR --basic-pitch --yourmt3`. "Jumps" is the share of notes that follow a jump up of a fifth or more (91 of them) and are written with the right pitch at the right time.

| Note finder | F | P | R | F 100ms | An octave off | Other wrong pitch | Jumps |
|---|---|---|---|---|---|---|---|
| bass2sheet | 0.79 | 0.71 | 0.89 | 0.83 | 1% | 3% | 78% |
| bass2sheet two days earlier, before the pitch path could jump | 0.76 | 0.70 | 0.84 | 0.80 | 2% | 2% | 46% |
| bass2sheet without splitting held notes at re-plucks | 0.86 | 0.86 | 0.87 | 0.91 | 0% | 2% | 76% |
| Basic Pitch | 0.70 | 0.62 | 0.81 | 0.76 | 4% | 5% | 78% |
| YourMT3+ (YPTF.MoE+Multi, no pitch shift) | 0.77 | 0.80 | 0.75 | 0.78 | 5% | 10% | 79% |

- **Lines that jump were written too low.** torchcrepe's Viterbi decoder lets the pitch move by a few bins per frame and no further. After a jump of a fifth or more the path lagged or stayed on the note before, where CREPE's probability is low, so the new note counted as unpitched and the old, lower note was held over it. A second cause sat in the octave rule: when CREPE and pYIN were an octave apart it took the lower note whenever that octave had a third of the energy, and it has whenever the note before still rings. pYIN smooths too, so on all 21 such notes in this set CREPE was right and pYIN was on the old note. The path may now leap (`pitch.decode_pitch`), and CREPE's octave stands when it is confident (`SegParams.trust_per`).
- **By plucking style**, the share of notes found: finger 94%, muted 95%, pick 93%, slap with the thumb 81%, slap pops 66% (33 of 50; 9 of 50 before the pitch path could jump, and 36 of 50 before slides stopped being written as notes: three pops are short enough to pass for one).
- **Splitting held notes at re-plucks costs 0.07 here**: 13% of our notes are re-attacks that are not in the reference. On RWC Popular the same rule gains 0.02, because there the separated bass smears repeated notes into one. The default has not changed; `--replucks clear` is the stricter policy, and the judging page asks which you prefer on your own songs.
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
- **Where there are none, few are written.** On the bass tracks separated from BabySlakh and RWC Popular, 7 of 6,289 and 16 of 13,988 notes come out as two (0.11%). The note finder's scores there are the same with it and without, to within 0.001.
- **Slap playing draws false pairs**: 9 of 1,207 notes on the IDMT lines (0.7%), most of them on the slap tracks.
- A second note must pass four tests (`together.py`): it is a note of its own in the spectrum, with two partials the first note cannot explain; it starts with the first (a note still ringing does not); it lasts with it (the pluck's noise does not); and it is louder in the bass track than in the rest of the record (a kick drum or a piano note that leaked into the bass track is not). On one of the six songs judged by ear the last test turned down 14 of 38 candidates.
- Mixed single notes are not a double stop played by a hand: the two strings of a real one start within a few milliseconds of each other and share the instrument's resonances. Recordings of real double stops with exact notes would be the better test.

### The notes checked by the bass track's own sound

`python dev/ownsound_eval.py --versions --idmt IDMT_DIR --dataset BABYSLAKH_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS`

Every note the tool finds is also a sample of what this bass sounds like at that pitch on this recording. `ownsound.py` compares each note's spectrum with those of the song's other notes written with its pitch, never with itself, and with the notes of other pitches moved to it. A note is **doubtful** when its namesakes explain less than half of it, or when another pitch's notes explain it clearly better. No reference is used; below, the references only say afterwards which notes were wrong.

| | Real bass alone (IDMT) | BabySlakh, separated | RWC Popular, separated |
|---|---|---|---|
| Notes judged | 1,099 | 6,230 | 13,682 |
| Of all notes, wrong by the reference | 12% | 12% | 17% |
| Of the doubtful notes, wrong | 44% (of 169) | 58% (of 374) | 61% (of 922) |
| Of the notes another pitch explains better, wrong | 67% (of 21) | 74% (of 141) | 77% (of 309) |
| Wrong notes that are doubted | 55% | 28% | 24% |
| Right notes that are doubted | 10% | 3% | 3% |
| Per song, share of doubtful notes against the true score (rank correlation) | −0.57 | −0.87 | −0.88 |
| Two versions of a song 0.03 or more apart in true score: the one with fewer doubtful notes is the better | 126 of 162 | 13 of 22 | 70 of 95 |
| The same, 0.06 or more apart | 87 of 109 | none | 41 of 43 |

- **It ranks songs.** The share of doubtful notes follows a song's true score more closely than any of the gates above.
- **It points at notes to listen to, and misses most of the wrong ones.** A doubted note is wrong about three times in five, where one note in six to eight is. But three wrong notes in four pass in a mix, half on a bass recorded alone. Wrong notes pass when the same mistake is made every time, because then they explain each other: the 360 notes an octave low in one BabySlakh song score 0.78, right notes 0.88.
- **It says whether a change helped a song only when the change is large.** Six settings of the note finder were run on every song. Of two versions whose true scores differ by 0.06 or more, the one with fewer doubtful notes is the better in 80% (real bass alone) to 95% (RWC) of pairs; at 0.03, in about three of four; below that it cannot tell.
- **It does not correct.** Replacing a doubted note with the pitch that explains it best would fix 20 notes and break 27 on BabySlakh, and fix 106 and break 151 on RWC. Letting the spectrum's own peaks propose the pitch and the other notes confirm it does better there (RWC: 106 fixed, 22 broken, F +0.005; BabySlakh: 15 and 11, F −0.001). On a song whose transcription was approved by ear it moved 9 notes, 7 of them an octave down, and agreement fell from 0.89 to 0.88. Not adopted.
- **Comparing the whole rebuilt track with the bass track adds nothing.** `audio/bass_rebuilt.wav` is the transcription played with these samples, each note by a sample cut from another note; it is for listening. As a measure, the overlap of its spectrogram with the bass track's picked the better of two versions no more often (71% to 75% of pairs 0.03 apart), and it rated the exact reference below the tool's own transcription on 11 of 17 real bass lines: it rewards borrowed samples that happen to fit in length.
- **A tone that changes during the song lowers every score.** Spectra are compressed before they are compared, because how loud each partial is depends on how the note was played and not on its pitch. On a song with a heavily distorted bass that took the doubtful share from 43% to 35%; on the three sets it changed nothing.
- **On six songs judged by ear**, the four where passages were picked have 4% to 14% doubtful notes, and the two where every passage was rejected have 35% and 43%, more than any song in the three sets (28% at most).

The share is reported as a gate (`own_sound`, warning above 15%) that is not part of the verdict yet: the verdict's thresholds were set before it existed.

### Verdicts against measured scores

The score is F for BabySlakh, note acc for MDB-bass-synth and F 100ms for RWC.

| Verdict | Runs | Mean score | Lowest | Runs at 0.85 or more |
|---|---|---|---|---|
| PASS | 15 | 0.96 | 0.90 | 100% |
| WARN | 50 | 0.84 | 0.52 | 54% |
| FAIL | 15 | 0.66 | 0.44 | 7% |

The gate thresholds and the "five warnings fail" rule were set on runs of an earlier version, which gave nearly the same table, so this is close to in-sample. The gate that tracks the score best is the share of note starts confirmed by Basic Pitch (rank correlation +0.79 on BabySlakh, +0.76 on MDB, +0.57 on RWC). The meter gate does not track note accuracy; it tracks bar position, which is what it is for. Notes in bars marked "check" are right 69% of the time on BabySlakh, against 88% in other bars; on RWC, 42% against 70%.

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
- **Slides, dying notes and pluck thumps are not notes.** Between two notes the pitch tracker reports the pitches a slide passes through, the pitch sinking as a note is let go, and a few frames of something else at a pluck. Each came out as a note of 70 to 130 ms, a step or two from its neighbour. A short stretch without a pluck of its own now joins the note before it when it is a step away or much quieter, and a short one at a pluck joins the note after it. RWC goes from 0.70 to 0.71 (precision 0.71 to 0.73, recall unchanged) and the real-bass set from 0.77 to 0.79. MDB, scored frame by frame, goes from 0.78 to 0.77. Its exact pitch curve says what the 1,646 stretches no longer written there were: 63% the pitch passing through or bending, 15% nothing the reference has, and 22% a steady pitch for a median 80 ms, which is a real short note lost (a hammer-on, a grace note). Requiring the pitch to move within the stretch saves some of those and keeps as many slides; it was not adopted.
- **Beat This!** replaced the built-in beat tracker, which chose the wrong pulse level on 7 BabySlakh songs.
- **Held notes are split at re-plucks.** Measured on the same stems, this moved mean F on the 24 RWC songs from 0.66 to 0.68 (0.72 to 0.75 at 100 ms) and on BabySlakh from 0.888 to 0.884. The gain is concentrated in songs with driving repeated notes.
