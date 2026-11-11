# bass2sheet

Turn a song into a bass part you can read and play. One command takes an audio file and writes a printable PDF with bass clef and tab, plus MusicXML, MIDI, listening files and a quality report that says how far to trust the result and which bars to check by ear.

```
bass2sheet "Artist - Title.mp3"
```

![The first lines of a score bass2sheet wrote for the synthesized test song](docs/example-score.png)

The transcription is done by machine and is not perfect. That is why every run measures itself: thirteen quality gates decide a PASS, WARN or FAIL verdict, and the bars where independent checks disagree are marked in the PDF. See [Accuracy](#accuracy) for measured results on public datasets.

Feed it audio you have the rights to use. bass2sheet downloads nothing from video or streaming sites.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. macOS on Apple Silicon is the tested platform; Linux should work and is covered by CI for the tests that need no models.

```
git clone <this repository> && cd bass2sheet
uv sync --python 3.12
uv run bass2sheet --help
```

Two optional pieces:

- **MuseScore 4**, for the PDF: `brew install --cask musescore` on macOS, or any install that puts `mscore` on the `PATH`. Without it you still get MusicXML, which opens in any notation program.
- **Basic Pitch**, for the second-opinion gates. It needs its own environment, because its ONNX/TensorFlow stack does not share one with PyTorch well:

  ```
  uv venv --python 3.12 .venv-bp
  uv pip install --python .venv-bp/bin/python basic-pitch onnxruntime 'setuptools<81'
  ```

  bass2sheet finds `.venv-bp` in the checkout, or set `BASS2SHEET_BASIC_PITCH_PYTHON` to another interpreter.

The first run downloads the separation models (about 1 GB) into `~/.cache/bass2sheet`. The same cache holds stems and pitch tracks, so re-running a song takes seconds.

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
| `notes.json` | Every note with its time, pitch, bar position and flags; key, tempo, chords |
| `report.html`, `report.json` | The verdict, each gate's value and threshold, the bars to check and why |
| `audio/` | The separated bass, the transcription synthesized, the song with the transcription in place of the real bass, and real bass left / transcription right for headphones |

Exit codes: `0` the sheet was written (PASS or WARN); `2` the result failed its gates (the sheet is marked DRAFT, or not written with `--strict`); `1` error (unreadable input, no bass found).

Other options: `--separator auto|roformer|demucs`, `--stems DIR` (use stems you already have and skip separation), `--no-audio`, `--no-pdf`, `--second-opinion off`, `--mscore PATH`, `--cache-dir DIR`, `--device mps|cuda|cpu`, `--crepe-model full|tiny`.

A 4-minute song takes about 3 minutes on an M-series MacBook the first time.

## How it works

1. **Decode and validate.** Any format to 44.1 kHz stereo. Unreadable, silent, very short or very long input is refused with a message.
2. **Separate the bass.** BS-RoFormer SW (via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)) is the default; Demucs `htdemucs_ft` is the fallback when it fails or its result fails the gates.
3. **Track pitch.** CREPE (Viterbi decoding) on the GPU, with pYIN as a cross-check.
4. **Find the notes.** Hysteresis on pitch confidence, so a note survives the brief loss of pitch at each pluck; a repeated pitch is split only where the low-band energy jumps; notes hold until the next one; each start moves to where its pluck begins to sound, at 1 ms resolution; notes whose spectrum contradicts their octave are corrected.
5. **Find the grid.** Beats from drums and bass; kick/snare alternation decides between quarter- and eighth-note pulses; the downbeat is where the harmony changes. Notes are quantized to sixteenths.
6. **Key and chords.** Rough estimates for the chord symbols and note spelling.
7. **Engrave.** MusicXML with a tab staff whose fingering comes from a search that keeps the fretting hand in place; MuseScore turns it into the PDF.
8. **Judge the result.** The quality gates below.

## Quality gates

There is no reference transcription for your song, so each gate is an independent proxy. None is trusted alone.

| Gate | Passes at | Fails beyond | What it catches |
|---|---|---|---|
| Bass stem level vs the mix | ≥ −25 dB | −35 dB | The separator found no bass |
| Sounding bass covered by notes | ≥ 95% | 85% | Missing notes |
| Notes cut short while the bass rings | ≤ 5% | 15% | Choppy note ends |
| Synth silent while the bass sounds | ≤ 5% | 15% | Gaps, judged on the transcription rendered back to audio |
| Same note name as the recording | ≥ 85% | 70% | Wrong notes, judged on the rendered transcription |
| Notes with a doubtful octave | 0 | 5% | Octave errors |
| Same pitch as Basic Pitch | ≥ 80% | 60% | Disagreement with an independent model |
| Note starts shared with Basic Pitch | F ≥ 0.6 | 0.4 | Rhythm disagreement |
| Note starts' distance from the 16th grid | ≤ 25 ms | 40 ms | A wrong tempo or grid |
| Notes on 8th-note positions | ≥ 75% | 55% | A wrong pulse level, a shifted downbeat, swing |
| Notes flagged uncertain | ≤ 10% | 25% | The two pitch trackers disagree |
| Bars that need an ear check | ≤ 15% | 30% | Too many bars where the checks disagree |

FAIL if any gate fails, WARN if any warns, otherwise PASS. Each bar also gets a severity that adds up independent signals (missing notes, a pitch mismatch with the recording, Basic Pitch disagreeing, a doubtful octave); the report lists the bars worst first, and severity 3 or more puts the red "check" in the PDF.

The thresholds are being calibrated against real accuracy; `bass2sheet-eval` reports, for each gate, how well it tracks the measured score.

## Accuracy

<!-- ACCURACY -->

How to reproduce these numbers, and what each dataset can and cannot tell you, is in [docs/evaluation.md](docs/evaluation.md).

## Limits

- Built for 4/4 rock and pop with a steady tempo. Swing, tempo changes and odd meters are not modelled; the grid gates usually catch them.
- One bass line. Two basses at once, and chords on the bass, are out of scope.
- The tab assumes a 4-string bass in standard tuning.
- Very short staccato notes (under about 100 ms) are often missed.
- Chord symbols are rough. Slides, ghost notes and dead notes are not notated.

## Development

`uv run pytest` runs the unit tests and an end-to-end run on a synthesized song in under a minute, with no model download. [CONTRIBUTING.md](CONTRIBUTING.md) covers the test tiers, the evaluation workflow and the rule for changing transcription behaviour: measure it against ground truth first.

## Credits and licenses

bass2sheet is MIT licensed (see [LICENSE](LICENSE)). It stands on other people's work, each under its own license:

- Separation: BS-RoFormer (the SW checkpoint by jarredou, run through audio-separator) and [Demucs](https://github.com/adefossez/demucs)
- Pitch: [CREPE](https://github.com/marl/crepe) through [torchcrepe](https://github.com/maxrmorrison/torchcrepe), and pYIN in [librosa](https://librosa.org)
- Second opinion: [Basic Pitch](https://github.com/spotify/basic-pitch)
- Evaluation: [mir_eval](https://github.com/mir-evaluation/mir_eval), [pretty_midi](https://github.com/craffel/pretty-midi)
- Engraving: [MuseScore](https://musescore.org)

Model weights are downloaded at run time and are not part of this repository; check each model's terms before commercial use. The evaluation datasets have their own licenses, listed in [docs/evaluation.md](docs/evaluation.md).
