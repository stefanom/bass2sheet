# bass2sheet

Turn a song into a bass part you can read and play. One command takes an audio file and writes a printable PDF with bass clef and tab, plus MusicXML, MIDI, listening files and a quality report that says how far to trust the result and which bars to check by ear.

[![The demo film. This frame: the notes the tool heard, on a picture of the isolated bass, and one of them being dragged to where it belongs. Click to watch it.](docs/demo-poster.jpg)](https://stefanom.github.io/bass2sheet/)

**See it work:** [a four-minute film](https://stefanom.github.io/bass2sheet/) takes one song from the recording to a sheet you fixed yourself, then to playing along. [What it shows and how it is made](docs/demo.md).

```
bass2sheet "Artist - Title.mp3"
```

![The first lines of a score bass2sheet wrote for the synthesized test song](docs/example-score.png)

The transcription is done by machine and is not perfect. That is why every run measures itself: twelve quality gates decide a PASS, WARN or FAIL verdict, and the bars where independent checks disagree are marked in the PDF. See [Accuracy](#accuracy) for measured results on public datasets.

Feed it audio you have the rights to use. The commands read files and download no music. The copilot's first page can also fetch the audio behind a link you paste, with yt-dlp: use that only where you have the right to download it.

## Quick start on a Mac

With [Homebrew](https://brew.sh) installed:

```
git clone https://github.com/stefanom/bass2sheet.git && cd bass2sheet
./bootstrap
uv run bass2sheet "Artist - Title.mp3"
uv run bass2sheet-copilot serve --dir scores
```

What you see after each:

1. `./bootstrap` installs what the [Brewfile](Brewfile) lists and you do not have (Python 3.12, uv, MuseScore 4, ffmpeg, Rubber Band, yt-dlp, Node.js), creates the Python environment in `.venv` and downloads the models. That is 435 MB of Python packages and 1.1 GB of models, and Homebrew's packages on top (MuseScore is 207 MB). It ends with a self-check, one line for each thing the tool needs, then `Ready.` and the commands to try. On an M4 MacBook with a fast line the first run took 65 seconds and a second run 6. If a step fails it stops, says which step and what to do, and can be run again.
2. `uv run bass2sheet SONG` logs its steps and ends with a verdict: PASS, WARN or FAIL. The sheet is `bass.pdf`, in a folder named after the song, beside it. A 34-second song took 39 seconds.
3. `uv run bass2sheet-copilot serve --dir scores` prints `0 songs. Open http://localhost:8770`. The page there takes a song, shows each step of hearing it, and opens it for you to fix.

`uv run bass2sheet-doctor` repeats the self-check at any time. `./bootstrap --check` says what is missing and changes nothing.

## Install: the details

macOS on Apple Silicon is the tested platform; Linux should work and is covered by CI for the tests that need no models. Python 3.12, [uv](https://docs.astral.sh/uv/) and a C compiler are required: one package is compiled when the environment is created. On Linux `./bootstrap` lists the system packages to install by hand and does the rest. Without the script:

```
uv sync --locked --python 3.12
uv run bass2sheet-doctor --fetch-models     # the models now; otherwise the first song downloads them
```

Optional pieces:

- **MuseScore 4**, for the PDF: `brew install --cask musescore` on macOS, or any install that puts `mscore` on the `PATH`. Without it you still get MusicXML, which opens in any notation program.
- **Basic Pitch**, for the second-opinion gates: `./bootstrap --with-basic-pitch`. It gets an environment of its own (`.venv-bp`), because its ONNX/TensorFlow stack does not share one with PyTorch well.
- **Rubber Band** and **yt-dlp**, for the copilot: a song slowed down with its bass line kept clean, and a song added by a link.

The models take 1.1 GB. The separation model goes into `~/.cache/bass2sheet`, the two others where their libraries keep them. `~/.cache/bass2sheet` also holds each song's stems and pitch tracks, so re-running a song takes seconds.

[docs/install.md](docs/install.md) has every step by hand, where each file goes and how to move it, what to do when a step fails, and what is not tested.

## Use

```
bass2sheet SONG [-o OUT_DIR] [--title T] [--artist A] [--strict]
```

`SONG` is any audio or video file ffmpeg can read. Title and artist default to the file name (`Artist - Title.ext`). The output directory defaults to `<song name> bass` beside the song:

| File | What it is |
|---|---|
| `bass.pdf` | The printable score: bass clef (written an octave up, as bass parts are) over 4-string tab, chord symbols, the recording time at the start of each line, uncertain notes in parentheses, a red "check" over bars to verify |
| `bass.musicxml` | The same score, editable in MuseScore or any notation program |
| `bass.mid`, `bass_as_played.mid` | MIDI on a steady grid, and MIDI aligned with the recording |
| `notes.json` | Every note with its time, pitch, bar position and flags; the tempo; the song's harmony: its chords with where each starts and ends and how sure it is, and its key or keys |
| `report.html`, `report.json` | The verdict, each gate's value and threshold, the bars to check and why |
| `audio/` | The separated bass, the transcription synthesized, the transcription played with the bass's own sound (`bass_rebuilt.wav`: each note by a sample cut from another note), the song with the transcription in place of the real bass, and real bass left / transcription right for headphones |

Exit codes: `0` the sheet was written (PASS or WARN); `2` the result failed its gates (the sheet is marked DRAFT, or not written with `--strict`); `1` error (unreadable input, no bass found).

Other options: `--separator auto|roformer|demucs`, `--stems DIR` (use stems you already have and skip separation), `--beat-tracker auto|beat_this|librosa`, `--replucks every|clear`, `--together write|single`, `--lowest NOTE`, `--rhythm literal|simple`, `--tab low|position`, `--no-audio`, `--no-pdf`, `--second-opinion off`, `--mscore PATH`, `--cache-dir DIR`, `--device mps|cuda|cpu`, `--crepe-model full|tiny`.

A 4-minute song takes about 3 minutes on an M-series MacBook the first time.

## Judge it on your own songs

Accuracy numbers cannot tell whether the sheet suits you. `bass2sheet-judge` can. It takes your songs through the pipeline one step at a time and, at each step, shows you short passages where the tool's choice and its alternative differ. You hear and see that step's own output and pick, without being told which is which.

```
bass2sheet-judge build SONG... --dir judging
bass2sheet-judge serve --dir judging        # then open http://localhost:8765
```

`bass2sheet-judge check --dir judging` needs no listening: it compares every note with the sound of the song's other notes of its pitch and reports the share that do not match, per song, and how that share moved since the tool last changed.

The steps are: which model isolates the bass (you hear the two isolated tracks), which notes are found in it, whether two notes played together are written as a chord, where the bar lines fall, whether the rhythm is written as played or simplified, and whether the tab stays low on the neck or in one position. Each step of a song is built on what you picked before it. Your picks decide the defaults, and every passage you judged comes back if a later change alters it. See [docs/judging.md](docs/judging.md).

## Make the score yours

The machine's sheet is a first draft. `bass2sheet-copilot` puts you in charge of the final one and makes the machine your copilot.

```
bass2sheet-copilot add SONG... --dir scores
bass2sheet-copilot serve --dir scores        # then open http://localhost:8770
```

The song plays through on a timeline: the isolated bass as a picture of pitch against time, the notes the tool heard drawn over it and played in the bass's own sound against the record. You stop where a note is wrong and fix it. For another pitch the tool offers its guesses, best first, and one key takes one; you can also nudge a note, delete it, add one, split one in two or join two. The tool finds the bars that sound alike and names them over the timeline: after a fix it offers the same fix for the repeats that have the same mistake, and it marks where a bar is written unlike a repeat that sounds the same. You decide. What you leave as it is, as far as you have worked, counts as right. The sheet is written from your notes.

Every fix is the tool's record of a mistake, with the right answer attached: `bass2sheet-copilot report` lists them by kind. The same page has a chord for every bar, starting from the tool's best guess: click one to hear the bar with it, try others on against the record, and keep the one that fits. A bass neck under the timeline shows the shape of each bar, in the key and the fingering you pick. A second page for each song shows how it moves and changes nothing: what each chord does, the few loops the bass line goes round, and what the bass plays in them. See [docs/copilot.md](docs/copilot.md).

## Another note finder: MuScriptor

`--notes muscriptor` writes the notes of [MuScriptor](https://github.com/muscriptor/muscriptor), Kyutai and Mirelo's transcription model, in place of bass2sheet's own, which become the second opinion that says which notes to doubt. On the bass bass2sheet isolates, MuScriptor is the better note finder (0.83 against 0.76 on RWC Popular, 0.91 against 0.79 on a bass played alone). It is optional, runs through a program you build separately, and its weights are for non-commercial use under conditions of their authors. See [docs/muscriptor.md](docs/muscriptor.md).

## How it works

1. **Decode and validate.** Any format to 44.1 kHz stereo. Unreadable, silent, very short or very long input is refused with a message.
2. **Separate the bass.** BS-RoFormer SW (via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)) is the default. Demucs `htdemucs_ft` is tried when it fails or its result fails the gates; of the two results, the one with fewer failed gates is kept, then the one whose note starts Basic Pitch confirms more often.
3. **Track pitch.** CREPE on the GPU, with pYIN as a cross-check. CREPE's output is decoded into a pitch path that moves smoothly within a note and may leap between notes, as a bass line does.
4. **Find the notes.** Hysteresis on pitch confidence, so a note survives the brief loss of pitch at each pluck; a held note is split where the string is plucked again, and only there; notes hold until the next one; each start moves to where its pluck begins to sound, at 1 ms resolution; notes whose spectrum contradicts their octave are corrected. What the pitch tracker reports between notes is not written: the slide from one note to the next, the pitch sinking as a note is let go, the thump of a pluck. The pitch trackers follow one note at a time, so every stretch is then looked at again for two notes played together: the spectrum's peaks must hold a second note of their own that starts with the first, lasts with it and is louder in the bass track than in the rest of the record. An octave is never written as two notes.
5. **Find the grid.** Beats and downbeats come from the [Beat This!](https://github.com/CPJKU/beat_this) model on the full mix. A built-in tracker (beats from drums and bass, the downbeat where the harmony changes) is the fallback. Notes are quantized to sixteenths in 4/4.
6. **Chords and key.** The record without its bass and drums is heard eighth note by eighth note, and each eighth is matched against major and minor triads and seventh chords on every root: by the notes that sound in it, and by the bass notes just found, since the bass holds the root most of the time. The chords are one line through the song that changes where the music does, on a bar line or in the middle of a bar far more readily than elsewhere. The key is read from the chords, and may change during the song; with it comes the next most likely key, which for most songs is the relative major or minor. The sheet's chord symbols and the spelling of its notes come from these.
7. **Engrave.** MusicXML with a tab staff whose fingering comes from a search that keeps the fretting hand in place; MuseScore turns it into the PDF.
8. **Check the notes by the bass's own sound.** Every note found is also a sample of what this bass sounds like at that pitch on this recording. Each note is compared with the other notes written with its pitch. One they do not explain, or that another pitch's notes explain better, is probably wrong: it goes in parentheses and on the list of bars to check.
9. **Judge the result.** The quality gates below.

## Quality gates

There is no reference transcription for your song, so each gate is an independent proxy. None is trusted alone.

| Gate | Passes at | Fails beyond | What it catches |
|---|---|---|---|
| Bass stem level vs the mix | ≥ −25 dB | −35 dB | The separator found no bass |
| Sounding bass covered by notes | ≥ 97% | 85% | Missing notes |
| Notes cut short while the bass rings | ≤ 4% | 12% | Choppy note ends |
| Same note name as the recording | ≥ 90% | 78% | Wrong notes, judged on the rendered transcription |
| Notes with a doubtful octave | ≤ 1% | 5% | Octave errors |
| Notes unlike the song's other notes of their pitch | ≤ 15% | reported only | A wrong note among right ones. Not part of the verdict yet |
| Same pitch as Basic Pitch | ≥ 90% | 70% | Disagreement with an independent model |
| Note starts shared with Basic Pitch | F ≥ 0.75 | 0.50 | Missed, extra or mistimed notes |
| Note starts' distance from the 16th grid | ≤ 32 ms | 50 ms | A wrong tempo or grid, or swing |
| Downbeats that fit 4/4 | ≥ 80% | warns only | 3/4, 6/8 or a changing meter, which the 4/4 score cannot show; bar lines may be shifted |
| Notes on 8th-note positions | ≥ 60% | warns only | A shifted grid or swing (or simply a busy sixteenth-note line) |
| Notes flagged uncertain | ≤ 6% | 20% | The two pitch trackers disagree |
| Bars that need an ear check | ≤ 6% | 25% | Too many bars where the checks disagree |

FAIL if any gate fails, or if five or more of the accuracy gates warn at once; WARN if any gate warns; otherwise PASS. Each bar also gets a severity that adds up independent signals (missing notes, a pitch mismatch with the recording, Basic Pitch disagreeing, a doubtful octave, a note that does not sound like its namesakes); the report lists the bars worst first, and severity 3 or more puts the red "check" in the PDF.

The thresholds come from evaluation runs with ground truth: a gate warns below what 90% of the accurate runs reach, and fails where only inaccurate runs were seen. `bass2sheet-eval` reports, for each gate, how well it tracks the measured score.

## Accuracy

Version 0.2.0, measured with `bass2sheet-eval`. The full mix goes in, with the same settings for every song.

| Dataset | Songs | Measure | Mean | Median |
|---|---|---|---|---|
| BabySlakh: synthetic songs, exact notes | 16 | Notes right: start within 50 ms, exact pitch (F-measure) | 0.91 | 0.96 |
| | 16 | The same, given the dataset's clean bass track | 0.94 | 0.95 |
| | 11 | Correct notes written on the right sixteenth of the bar | 0.66 | 0.77 |
| MDB-bass-synth: real multitrack songs, exact pitch curve | 24 of 71 | Share of the time the bass plays that has the right note | 0.77 | 0.80 |
| RWC Popular: studio pop, bass transcribed by ear | 24 of 100 | Notes right, start within 100 ms (F-measure) | 0.77 | 0.73 |
| | | Notes right, start within 50 ms | 0.71 | 0.67 |
| | | Chords: share of the time the root is right | 0.86 | 0.87 |
| | | Chords: root and major or minor right | 0.86 | 0.87 |
| IDMT-SMT-Bass-Single-Track: a real bass guitar played alone, exact notes | 17 lines | Notes right: start within 50 ms, exact pitch (F-measure); no separation involved | 0.79 | 0.80 |

What that means in practice:

- **Clean material mostly comes out right.** 12 of the 16 BabySlakh songs score 0.90 or more and all but one score above 0.84. The one is a song where a synth plays an octave below the bass guitar: the separated bass track holds both, and over a third of the notes are written an octave low (0.49, against 0.89 from the dataset's clean bass track).
- **A real bass guitar is harder than a sampled one.** Played alone, with nothing to separate, 89% of the notes are found, but 13% of what the tool writes are re-attacks that were not played, and slap pops are found 66% of the time. Lines that jump a fifth or more upward are written right 78% of the time.
- **Real recordings are harder.** On RWC about a third of the notes differ from the reference: repeated notes written as one long note (12% of the reference's notes), re-attacks the reference does not have (9% of ours) and wrong pitches (6%). The reference was transcribed by ear, so part of the difference is its own.
- **Pitch is more reliable than rhythmic detail.** Octave errors are 1–3% of the notes.
- **Chords: the root is right 86% of the time.** So is major or minor. Whether a chord is a plain triad or a seventh chord is right about two times in three, and where the bass is not on the root, the root is right 0.59 of the time. Each chord carries a score, and it means what it says: chords scored 0.9 and up are right 92% of the time, chords scored 0.5 to 0.7 about 60%. The main key was right on all 24 songs. Measured on RWC Popular, to which nothing in the chord finder was fitted.
- **Tempo is right on nearly every song; bar lines are not.** On BabySlakh, 5 of 11 songs have their notes on the right sixteenth of the bar throughout. On the others the bar lines are off for part or all of the song: finding the downbeat is the hard part, and the fixed 4/4 grid cannot follow a song whose bars change length. RWC scores 1.00 here, but the beat tracker was trained on RWC, so that number says little.
- **The verdict tracks accuracy.** Over the 80 runs, PASS results averaged 0.96 with none below 0.90, WARN 0.85 and FAIL 0.66. The thresholds were set on these same runs, so expect a looser relation on new songs.

How to reproduce these numbers, and what each dataset can and cannot tell you, is in [docs/evaluation.md](docs/evaluation.md).

## Limits

- The score is always written in 4/4 with straight sixteenths. Other meters and swing are not notated; the meter and grid gates flag them. Tempo changes are followed beat by beat, but the score shows one tempo.
- One bass. Two notes played together are written as a chord when the tool is sure of both; it misses about a quarter of them, and an octave is always written as its lower note. Three notes or more, and two basses at once, are out of scope.
- The tab assumes a 4-string bass in standard tuning, and so do the notes: what is heard below E1 (an octave effect, a synth under the bass) is written an octave up. `--lowest B0` is a five-string, `--lowest D1` drop-D, `--lowest none` writes the notes as heard.
- Very short staccato notes (under about 100 ms) are often missed, and fast repeated notes on one pitch can come out as one long note.
- A bar of another length (a 2/4 bar in a 4/4 song) shifts the bar lines after it.
- Without the Beat This! model the built-in beat tracker finds the beats, and it often counts at the wrong level, half the tempo for the most part (it did on 7 of the BabySlakh test songs), which puts the bar lines in the wrong places and makes the rhythm coarser. The notes and their pitches are not affected. The command's log says when it was used, and so does the song's card on the copilot's first page; `uv run bass2sheet-doctor --fetch-models` gets the model.
- Chords are major and minor triads and three seventh chords (7, maj7, m7); a suspended, diminished or augmented chord is written as the nearest of those. A chord over another bass note is often named after the bass. Slides, ghost notes and dead notes are not notated.

## Development

```
uv run pytest                 # the default tests: about 5 minutes, nothing needed beyond `uv sync`
uv run pytest --cov           # the same, with what they reach of the code: about 97% of lines and branches
uv run pytest -m browser      # every page in a real browser (uv sync --group browser; uv run playwright install chromium)
BASS2SHEET_FULL_E2E=1 uv run pytest -m full    # the real models: about 1 GB to download the first time
```

The default tests run the unit tests and the whole pipeline, the copilot and its pages' servers on a synthesized song, with no model download, no network and no browser: where a model or MuseScore would run, a stand-in does. CI runs them on every push to `main` and on every pull request, on Linux and macOS, and fails if they reach less than 95% of the code. [CONTRIBUTING.md](CONTRIBUTING.md) covers the tiers, what each needs and how long it takes, the evaluation workflow and the rule for changing transcription behaviour: measure it against ground truth first.

## Credits and licenses

bass2sheet is MIT licensed (see [LICENSE](LICENSE)). It stands on other people's work, each under its own license:

- Separation: BS-RoFormer (the SW checkpoint by jarredou, run through audio-separator) and [Demucs](https://github.com/adefossez/demucs)
- Pitch: [CREPE](https://github.com/marl/crepe) through [torchcrepe](https://github.com/maxrmorrison/torchcrepe), and pYIN in [librosa](https://librosa.org)
- Beats: [Beat This!](https://github.com/CPJKU/beat_this)
- Second opinion: [Basic Pitch](https://github.com/spotify/basic-pitch)
- Optional note finder: [MuScriptor](https://github.com/muscriptor/muscriptor) through [muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp) (weights CC BY-NC 4.0 with conditions of use; not downloaded by bass2sheet)
- Chords: the weights that match a chord to what is heard are fitted on [BabySlakh](https://zenodo.org/records/4603870), the first 20 songs of Slakh2100 (Manilow, Wichern, Seetharaman and Le Roux, 2019; CC BY 4.0)
- Evaluation: [mir_eval](https://github.com/mir-evaluation/mir_eval), [pretty_midi](https://github.com/craffel/pretty-midi)
- Engraving: [MuseScore](https://musescore.org)

Model weights are downloaded at run time and are not part of this repository; check each model's terms before commercial use. The evaluation datasets have their own licenses, listed in [docs/evaluation.md](docs/evaluation.md).
