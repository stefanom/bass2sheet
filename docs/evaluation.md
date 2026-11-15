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

None of these is a commercial release, and whether they are like the music you want scores for is for you to judge: `bass2sheet-copilot testset` puts a dataset's songs in the copilot to be heard and rated, and counts the scores on the ones that are ([copilot.md](copilot.md), "Judging the test songs").

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
| MuScriptor medium, told to write only the bass (see below) | 0.91 | 0.90 | 0.92 | 0.96 | 0% | 2% | 90% |

- **Lines that jump were written too low.** torchcrepe's Viterbi decoder lets the pitch move by a few bins per frame and no further. After a jump of a fifth or more the path lagged or stayed on the note before, where CREPE's probability is low, so the new note counted as unpitched and the old, lower note was held over it. A second cause sat in the octave rule: when CREPE and pYIN were an octave apart it took the lower note whenever that octave had a third of the energy, and it has whenever the note before still rings. pYIN smooths too, so on all 21 such notes in this set CREPE was right and pYIN was on the old note. The path may now leap (`pitch.decode_pitch`), and CREPE's octave stands when it is confident (`SegParams.trust_per`).
- **By plucking style**, the share of notes found: finger 94%, muted 95%, pick 93%, slap with the thumb 81%, slap pops 66% (33 of 50; 9 of 50 before the pitch path could jump, and 36 of 50 before slides stopped being written as notes: three pops are short enough to pass for one).
- **Splitting held notes at re-plucks costs 0.07 here**: 13% of our notes are re-attacks that are not in the reference. On RWC Popular the same rule gains 0.02, because there the separated bass smears repeated notes into one. The default has not changed; `--replucks clear` is the stricter policy, and the judging page asks which you prefer on your own songs.
- **YourMT3+** places notes cleanly (almost no extra notes) but names the wrong pitch three times as often, and its spectrogram starts at 50 Hz, below which the lowest five notes of a bass have their fundamental. On bass tracks separated from six commercial recordings it was not usable: 14 notes for a whole song with a dark, sustained bass, a bass line labelled as guitar throughout, some 2,700 notes for a distorted one, and no note below B1 in any of them. It was trained on Slakh and on RWC Popular, so it cannot be compared on those.

### Against MuScriptor

`MUSCRIPTOR_NOTES=... MUSCRIPTOR_GGUF=... python dev/muscriptor_eval.py --size medium --idmt IDMT_DIR --dataset BABYSLAKH_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS`

[MuScriptor](https://github.com/muscriptor/muscriptor) (Rouard et al., 2026, by Kyutai and Mirelo) is a transformer that writes every instrument's notes from the mix and says which instrument each belongs to. Its code is MIT; its weights are CC BY-NC 4.0 with conditions of use of their authors. It was run here through its C++ port, [muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp), on an M4 Max (`dev/muscriptor_cpp`). bass2sheet does not use it. The score is F for the bass alone and BabySlakh, F 100 ms for RWC.

| | bass2sheet | MuScriptor medium (307M) on the mix, the notes it calls bass | MuScriptor medium on the bass bass2sheet isolated, told to write only the bass | MuScriptor large (1.4B), the same |
|---|---|---|---|---|
| Real bass played alone (17 lines) | 0.79 | | 0.91 | 0.93 |
| BabySlakh (16 songs) | 0.92 | 0.50 | 0.94 | 0.92 |
| RWC Popular (24 songs) | 0.76 | 0.72 | 0.83 | 0.84 |

- **On an isolated bass it is the better note finder.** On RWC Popular it is better by 0.02 or more on 16 of the 24 songs and worse on 3; its worst song scores 0.56, ours 0.47. On the bass played alone, 10% of the notes it writes are not in the reference against 29% of ours; it has the right note at the right time after a jump up of a fifth or more 90% of the time (ours 78%), and 96% of slap pops (ours 66%). BabySlakh, where both are near the ceiling, is level: better on 4 songs, worse on 5.
- **From the full mix it is worse than separating first.** It hears the notes but mislabels the instruments: on one RWC song every note is written as piano and none as bass, on a BabySlakh song 1,418 notes are called bass where the bass plays 241. On RWC it is better than bass2sheet on 11 songs and worse on 11; on BabySlakh worse on 15 of 16. Given the isolated bass and no hint, it scores 0.83 on RWC and 0.85 on BabySlakh: telling it that the track is a bass matters on the synthesized set.
- **Its instrument conditioning says what is there; it does not pick an instrument out.** Told to write only the bass, a full mix comes out as every sound in it written as bass: 8,406 notes for a song with 355, F 0.00.
- **The large model is not better than the medium one here**, and three to five times slower. With the electric and the acoustic bass both allowed it sometimes writes each note twice, once for each; the scoring keeps one.
- **Two note finders together.** A note both write is right 88% of the time on RWC and 94% on BabySlakh. A note only MuScriptor writes is right 60% and 58%; a note only bass2sheet writes, 21% and 14%. Adding ours to MuScriptor's does not raise the score (0.825 against 0.828): what the second finder is good for is saying which notes to doubt.
- **Timing.** Its note starts are on a 10 ms grid and 10 to 20 ms late in the median. Taking each track's median lag out changes the score by less than 0.01 on the songs and by 0.03 on the bass alone.
- **Speed**, medium model on the M4 Max's GPU: an isolated bass track of four minutes in 10 to 25 s; a full mix in one to four minutes.
- **What the comparison cannot say.** BabySlakh is rendered from the Lakh MIDI collection, which is part of MuScriptor's synthetic training data; its authors hold RWC out.

**Through the whole pipeline.** `bass2sheet-eval DIR --notes muscriptor` writes MuScriptor's notes in place of bass2sheet's own (`muscriptor.py`, [muscriptor.md](muscriptor.md)): made into one line, starts moved to the plucks, ends taken from bass2sheet's note where both hear it, and bass2sheet's notes kept as the second opinion. From the mix, on the same songs as the tables above:

| | F | F 100 ms | F with note ends | F on note names | Better / worse by 0.02 | PASS / WARN / FAIL |
|---|---|---|---|---|---|---|
| BabySlakh, bass2sheet's notes | 0.92 | 0.92 | 0.73 | 0.94 | | 5 / 9 / 2 |
| BabySlakh, MuScriptor's notes | 0.94 | 0.95 | 0.71 | 0.95 | 5 / 1 | 3 / 11 / 2 |
| RWC Popular, bass2sheet's notes | 0.71 | 0.77 | 0.45 | 0.74 | | 3 / 15 / 6 |
| RWC Popular, MuScriptor's notes | 0.79 | 0.84 | 0.54 | 0.83 | 18 / 3 | 4 / 9 / 11 |

- **Note ends.** MuScriptor holds the synthesized notes of BabySlakh too long: with its own ends the score that also asks for the end is 0.45, with bass2sheet's it is 0.71. On RWC the borrowed ends beat both finders' own (0.54 against 0.48 and 0.44), and on the bass played alone too (0.71 against 0.63 and 0.57).
- **Which of its notes to doubt.** A note is flagged when bass2sheet's finder hears another pitch there, or nothing, or holds that pitch with no pluck to be heard. On RWC that is 11% of the notes, 56% of them wrong (17% of all its notes are), holding 36% of the wrong notes; on BabySlakh 8%, 21% wrong. Where bass2sheet holds a pitch and MuScriptor writes a new pluck that can be heard (11% of the notes on RWC), MuScriptor is right 89% of the time: those are not flagged.
- **The verdicts are stricter with MuScriptor's notes, and should not be.** The gates' thresholds were set on bass2sheet's own notes. The flag above trips the "notes flagged uncertain" gate on 15 of the 24 RWC songs (6 with bass2sheet's notes), and five warnings make a FAIL: 11 songs fail with a mean F of 0.72, where bass2sheet's own notes fail 6 with a mean of 0.56. The verdict still ranks the songs (PASS 0.96, WARN 0.81, FAIL 0.72), but a FAIL here is as accurate as a WARN there. Not recalibrated yet.

### Re-plucks: the isolated bass against the original recording

`python dev/repluck_eval.py --muscriptor KEEP_DIR --dataset BABYSLAKH_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS`

Separation blurs the attack of a string plucked again on the same pitch, and the original recording still has it. Does listening to the original, at the note's own partials, tell a re-pluck from a held note better? On RWC Popular there are 4,639 notes that repeat the pitch before with no gap, and 9,223 moments inside notes of half a second or more.

| How a re-pluck is told from a held note | Re-plucks found | False ones in held notes |
|---|---|---|
| Rise in level of the isolated bass | 14% | 3% |
| Rise in level of the original, at the note's partials | 19% | 3% |
| Both must rise | 36% | 3% |
| bass2sheet's note finder as a whole | 59% | 2.5% |
| MuScriptor on the isolated bass | 86% | 2.9% |
| MuScriptor on the original | 83% | 13.6% |

- **The attack is clearer in the original**: a re-pluck raises the level by 8.7 dB there and by 4.8 dB in the isolated bass. But other instruments rise at the same frequencies: a held note shows 3.3 dB in the original and 0.6 dB in the isolated bass. Alone the original is hardly better; asking for a rise in both finds two and a half times as many as the isolated bass alone.
- **What the blur defeats is a rule on the level, not a trained model.** MuScriptor, reading the isolated bass, writes 86% of the re-plucks. The information is still in the track.
- On the synthesized BabySlakh there is no blur: the isolated bass's level alone finds 99% at 3% false alarms.
- Nothing in the tool uses the original this way. As a third witness for MuScriptor's notes it was tried and not kept: see below.

### Three witnesses: the original recording and the doubt flag

`python dev/witness_eval.py --muscriptor KEEP_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS --dataset BABYSLAKH_DIR`

With `--notes muscriptor` a note is flagged on the word of two witnesses, both reading the isolated bass: MuScriptor wrote it and bass2sheet's finder does not hear it. The original recording would be the third: the level must rise at the note's own partials around its start, in the original and in the isolated bass (the "both must rise" of the table above; `muscriptor.rises`). Would it say better which notes to doubt? Every rule was tried on every note MuScriptor writes, 15,858 on RWC Popular and 6,224 on BabySlakh. The notes are the same under all of them, and so are the note scores.

| The flag | RWC Popular: notes flagged | Of them wrong | Wrong notes it holds | BabySlakh: notes flagged | Of them wrong | Wrong notes it holds |
|---|---|---|---|---|---|---|
| As it is: the two finders | 10.6% | 55.8% | 35.6% | 7.8% | 20.8% | 22.1% |
| Taken off where both rise by 6 dB or more | 5.2% | 56.0% | 17.7% | 1.8% | 57.9% | 14.4% |
| Taken off where both rise by 12 dB or more | 7.8% | 57.8% | 27.3% | 5.8% | 25.9% | 20.4% |
| Taken off where both rise by 16 dB or more | 9.1% | 57.8% | 31.7% | 7.0% | 22.1% | 21.2% |
| Also put on where they rise by less than 1 dB | 13.0% | 49.9% | 39.3% | 8.3% | 20.9% | 23.6% |
| Also put on where they rise by less than 2 dB | 20.0% | 37.6% | 45.5% | 9.3% | 20.0% | 25.4% |
| Where bass2sheet holds the pitch: a rise under 2 dB in place of "no pluck heard" | 13.0% | 48.0% | 37.8% | 7.8% | 21.1% | 22.5% |
| Every note bass2sheet lacks, unless both rise by 12 dB | 17.8% | 31.6% | 34.1% | 7.1% | 37.2% | 36.1% |

- **No rule is kept: on the real recordings each one makes the flag worse.** The flagged notes the original supports are wrong as often as the rest: 847 of them rise by 6 dB or more in both, and 55% of those are wrong; of the 234 that rise by 16 dB or more, 43% are. Taking the flag off them gives up wrong notes and gains nothing. And a note that rises in neither is still right three times in four: 393 unflagged notes rise by less than 1 dB, 25% of them wrong, against 56% of the flagged notes.
- **Why: the rise tells one kind of mistake, and it is not the common one.** A note written where the reference holds a note has no attack: the rise tells it from a right note 78% of the time (median 2.6 dB against 7.5 dB). Those are a quarter of the wrong notes on RWC (a re-pluck that was not played 16%, a note written while another pitch is held 10%). The rest start where something is plucked, and what is wrong is which note: another pitch starts there (39%), the note is an octave away (20%), its start is off by more than a tenth of a second (8%). For those the rise says nothing (51 to 56%). And 1,416 of the 13,234 right notes rise by less than 2 dB, played legato or softly.
- **MuScriptor has already heard the attack.** The rise finds re-plucks among held notes; MuScriptor writes a note at 3% of the held moments and at 86% of the re-plucks. Once it has written a note, the level adds little: within each group of notes (bass2sheet has it, holds the pitch, hears another, hears nothing) the rise tells right from wrong 56 to 63% of the time on RWC.
- **On the synthesized BabySlakh the original would sharpen the flag**: taken off where both rise by 6 dB, it falls on 1.8% of the notes and 58% of them are wrong; the 371 notes that lose it are right 91% of the time. There the flag's trouble is another one (bass2sheet's finder and MuScriptor differ by an octave on notes MuScriptor has right), and the isolated bass has clean attacks. A rule had to hold on both sets, and RWC is the one made of recordings.

### Lining a transcription up with a recording

`python dev/reference_eval.py --dataset RWC_DIR --annotations RWC_ANNOTATIONS --dataset BABYSLAKH_DIR`

`bass2sheet-copilot reference` takes somebody's transcription of a song (a MIDI file, a tab) as an answer key for your recording of it ([copilot.md](copilot.md)). For that the file has to be lined up with the recording first (`reference.py`). How close that comes is measured where the truth is known: from each test song's answer key a file is made as someone would have sequenced it, and the step has to find the way back. For RWC Popular every note of the arrangement is put on a steady beat, 7% faster than the recording, after three seconds of nothing; the recording's own tempo wanders, the file's does not. For BabySlakh the arrangement is slowed by 7%, with a tempo that wanders by 3% either way.

| | RWC Popular (24 songs) | BabySlakh (16 songs) |
|---|---|---|
| Bass notes of the key that are placed | 99.3% | 100% |
| A song's own offset from the dataset's times | +24 ms (+14 to +43) | +19 ms (+12 to +25) |
| Spread of the notes around that offset | 6 ms | 2 ms |
| Notes within 50 ms / 100 ms of their true time | 98.0% / 100% | 99.6% / 100% |
| bass2sheet's score against the dataset's key | 0.761 | 0.915 |
| ... against the key lined up by this step | 0.765 | 0.913 |
| Largest difference between the two for one song | 0.021 | 0.017 |
| With eight seconds left out of the file: the gap found | 23 of 24 songs | 15 of 16 songs |
| ... and notes placed more than 100 ms off | 25 of 15,109 | 11 of 5,846 |

- **A key lined up this way scores a transcription as the dataset's own key does**: the two scores differ by 0.005 a song on average.
- **The offset is the datasets', not the step's.** The step puts a note where its sound starts. Slakh's samplers start to sound 5 to 40 ms after their MIDI note, and RWC's aligned MIDI runs ahead of the sound by about as much: bass2sheet's and MuScriptor's own notes are 11 to 20 ms later than both keys too.
- **What made it work.** Chroma alone runs two readings of a song side by side to within a few tenths of a second, and is stiff where the tempo wanders. Onsets alone are exact and cannot tell a stretch from the same stretch half a beat later: left to themselves they moved two whole songs of the 24 by an eighth note. Together: chroma says where, to a tenth of a second; the onsets say exactly where, along one smooth course that may not stray far from what the chroma said.
- **What it does not measure**: files that are wrong about the notes, which real tabs are. The files here are the answer keys themselves, re-timed. On a real file the step still finds where each bar belongs; how much the tab's notes are worth is another matter, and two tabs of one song are the way to see it.

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
- **MuScriptor's second notes are not to be taken at its word.** Where it starts two notes together, the second is in the answer key 2 times in 79 on RWC Popular and 3 in 120 on BabySlakh (bass2sheet's own finder, with the four tests: 4 in 12 and 1 in 6). Many are partials of the note, a twelfth or two octaves and a third above it; in one song of the six judged by ear it wrote the same G on top of 24 notes in a row, where the track holds almost nothing at that pitch (on most of them 9 to 50 dB under the note). So with `--notes muscriptor` a second note is written only where bass2sheet's own finder wrote the same two notes together: on the test songs that keeps 1 of the 199, and in the six songs 9 of 75.
- **Two notes a step apart are never written together.** In the bass parts of 15,191 MIDI files (the Lakh clean subset, 7.8 million notes) notes that start together are 2.5% of all notes, and of those pairs 53% are octaves, 17% fifths, 9% fourths; 0.7% are a semitone apart and 1.6% a tone: one note in 5,900 and one in 2,600. Where MuScriptor writes two that close, the one that sounds is kept: the one bass2sheet's finder hears there, or else the one whose second to fourth partials are stronger in the bass track.
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

- **It ranks songs.** As a gate in the full pipeline, the share of doubtful notes has a rank correlation with the score of −0.83 on BabySlakh and −0.85 on RWC, where the best of the other gates reaches 0.80 and 0.58. On MDB-bass-synth it is −0.61, behind the agreement with Basic Pitch (0.77).
- **It points at notes to listen to, and misses most of the wrong ones.** A doubted note is wrong about three times in five, where one note in six to eight is. But three wrong notes in four pass in a mix, half on a bass recorded alone. Wrong notes pass when the same mistake is made every time, because then they explain each other: the 360 notes an octave low in one BabySlakh song score 0.78, right notes 0.88.
- **It says whether a change helped a song only when the change is large.** Six settings of the note finder were run on every song. Of two versions whose true scores differ by 0.06 or more, the one with fewer doubtful notes is the better in 80% (real bass alone) to 95% (RWC) of pairs; at 0.03, in about three of four; below that it cannot tell.
- **It does not correct.** Replacing a doubted note with the pitch that explains it best would fix 20 notes and break 27 on BabySlakh, and fix 106 and break 151 on RWC. Letting the spectrum's own peaks propose the pitch and the other notes confirm it does better there (RWC: 106 fixed, 22 broken, F +0.005; BabySlakh: 15 and 11, F −0.001). On a song whose transcription was approved by ear it moved 9 notes, 7 of them an octave down, and agreement fell from 0.89 to 0.88. Not adopted.
- **Comparing the whole rebuilt track with the bass track adds nothing.** `audio/bass_rebuilt.wav` is the transcription played with these samples, each note by a sample cut from another note; it is for listening. As a measure, the overlap of its spectrogram with the bass track's picked the better of two versions no more often (71% to 75% of pairs 0.03 apart), and it rated the exact reference below the tool's own transcription on 11 of 17 real bass lines: it rewards borrowed samples that happen to fit in length.
- **A tone that changes during the song lowers every score.** Spectra are compressed before they are compared, because how loud each partial is depends on how the note was played and not on its pitch. On a song with a heavily distorted bass that took the doubtful share from 43% to 35%; on the three sets it changed nothing.
- **On six songs judged by ear**, the four where passages were picked have 4% to 14% doubtful notes, and the two where every passage was rejected have 35% and 43%. No song of BabySlakh or RWC has more than 25%. One song of MDB-bass-synth has 42% with a note accuracy of 0.78, so a high share is a strong warning and not a proof.

The share is reported as a gate (`own_sound`, warning above 15%) that is not part of the verdict yet: the verdict's thresholds were set before it existed.

### Repeats

`python dev/repeats_eval.py --dataset BABYSLAKH_DIR --dataset RWC_DIR --annotations RWC_ANNOTATIONS`

The copilot finds the bars of a song that sound alike (`repeats.py`): two bars are compared sixteenth by sixteenth by the spectrum of the separated bass, on the tool's own beat grid. 79% of the bars of BabySlakh and 86% of RWC's have a repeat. Where two such bars sound the same in a sixteenth, the reference has the same pitch in both 99% of the time on BabySlakh and 94% on RWC.

To measure what that is worth, the reference plays the person: it goes through each song bar by bar and note by note and makes what is wrong right. A sixteenth is right when the pitch at its middle is the reference's and a note starts there if and only if one does in the reference. As in the copilot, a fix that only makes a note shorter or longer is not carried.

| | BabySlakh (16 songs) | RWC Popular (24 songs) |
|---|---|---|
| Sixteenths to fix by hand, without help | 3,645 | 10,401 |
| With every fix carried to the bars with the same mistake | 3,140 (14% fewer) | 8,464 (19% fewer) |
| The saving per song: 10th percentile / median / 90th | 0% / 3% / 13% | −1% / 13% / 59% |
| The best song and the worst | 29% fewer, 3% more | 65% fewer, 61% more |
| Carried: the octave, right | 100% (of 232) | 83% (of 425) |
| Carried: another pitch, right | 5 of 11 | 88% (of 784) |
| Carried: where a note starts (a re-pluck), right | 6 of 6 | 90% (of 962) |
| Carried: a note taken out, right | 95% (of 288) | 59% (of 469) |
| Carried: a note put in, right | 11 of 14 | 97% (of 225) |
| Marked, a repeat the person fixed: the marked sixteenth is wrong / the repeat's version is right | 98% / 75% (of 199) | 67% / 57% (of 1,035) |
| Marked, a repeat the person played past | 69% / 63% (of 67) | 68% / 61% (of 537) |
| Marked, most of the tool's own repeats | 92% / 29% (of 27) | 62% / 51% (of 98) |

- **A fix is carried** to the other bars that sound the same at that spot and are written there as the fixed bar was: the same mistake. "Right" counts a carried change that made a wrong sixteenth right against one that broke a right one. It saves about a sixth of the fixing, very unevenly: more than half on the four songs where the tool makes one mistake in every repeat, a few percent on most. On two RWC songs, taking every offer unheard would have made more work than it saved, which is why the page lets the person hear the bars an offer would change before taking it.
- **RWC's figures are the pessimistic ones.** Its references are aligned by hand at the beat, so two repeats can differ in the reference by a sixteenth, or by an octave, where the record does not. BabySlakh's references are exact, but its songs have few wrong pitches to carry, apart from one song an octave low throughout.
- **A bar the person has been through is never changed or marked.** Carrying into such bars broke more than it fixed (113 sixteenths in one song), and marking them was wrong every time in this simulation, where the person misses nothing.
- **Marks are hints.** A sixteenth marked as written unlike a repeat is wrong about two times in three, more often when the repeat is one the person fixed; what the repeat has there is the right answer about six times in ten. The tool against itself (most of its repeats agree against one bar) points at wrong notes as often, and its answer is right less often. A note against silence is marked only as a whole note: how long a note is held is not. Nor is the line between two notes falling a sixteenth earlier in one bar than in another.
- Tried and not adopted: counting a sixteenth where neither bar sounds as sounding the same (it carried changes between bars the reference has different notes in), and carrying every bar the person has passed, fixed or not, to its repeats (as much saved on average, far more broken on the songs where it goes wrong).

### Verdicts against measured scores

The score is F for BabySlakh, note acc for MDB-bass-synth and F 100ms for RWC.

| Verdict | Runs | Mean score | Lowest | Runs at 0.85 or more |
|---|---|---|---|---|
| PASS | 15 | 0.96 | 0.90 | 100% |
| WARN | 48 | 0.85 | 0.52 | 56% |
| FAIL | 17 | 0.66 | 0.44 | 6% |

The gate thresholds and the "five warnings fail" rule were set on runs of an earlier version, which gave nearly the same table, so this is close to in-sample. Of the gates that decide the verdict, the one that tracks the score best is the share of note starts confirmed by Basic Pitch (rank correlation +0.79 on BabySlakh, +0.72 on MDB, +0.56 on RWC). The share of notes unlike their namesakes, which is reported but does not decide, tracks it better on two of the three sets (−0.83, −0.61, −0.85). The meter gate does not track note accuracy; it tracks bar position, which is what it is for.

Notes in bars marked "check" are right 73% of the time on BabySlakh, against 90% in other bars; on RWC, 50% against 75%. A note that does not sound like its namesakes now counts towards a bar's mark. That marked more bars, and moved two runs from WARN to FAIL through the gate on the share of marked bars; before it, the figures were 63% against 87% and 46% against 74%.

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
