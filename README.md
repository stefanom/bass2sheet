# bass2sheet

Song audio file in, printable bass sheet out: a PDF with bass clef and tab, plus MusicXML, MIDI, listening files, and a quality report that says whether the result can be trusted and which bars to check by ear.

```
bass2sheet "Wilco - How To Fight Loneliness.mp3"
# -> "Wilco - How To Fight Loneliness bass/": bass.pdf, bass.musicxml, bass.mid, bass_as_played.mid,
#    notes.json, report.html, report.json, audio/{bass_stem,synth_bass,song_resynth,bass_ab_lr}.wav
```

Feed it audio you have the rights to use. bass2sheet does not download anything from video sites.

## Pipeline

1. **Decode and validate.** PyAV produces 44.1 kHz stereo. Input that is unreadable, silent, under 20 s or over 20 min is refused. Clipped or mono input gets a warning.
2. **Separate.** BS-RoFormer SW (6 stems, via `audio-separator`) is the default. Demucs `htdemucs_ft` is the fallback when RoFormer fails or its result fails the gates.
3. **Track pitch.** CREPE (full model, Viterbi decoding) on the GPU (MPS or CUDA; about 10× faster than CPU, within 0.4 semitone of it), with pYIN running alongside as a cross-check. Tracks are cached by a hash of the stem.
4. **Segment into notes.**
   - Hysteresis voicing: a note starts above 0.45 periodicity, and ends only after 200 ms below 0.15 or a 15 dB decay.
   - Notes hold until the next note.
   - A repeated pitch is split only on a measured pluck (a jump of 3 dB or more in the 30–400 Hz band).
   - Octave sanity: a note whose octave is 25 dB louder than its fundamental moves up an octave.
5. **Rhythm.** Beats come from drums + bass. Kick/snare alternation decides between quarter- and eighth-note pulses. The downbeat comes from harmonic change. Notes are quantized to 16ths.
6. **Key and chords.** Krumhansl key profiles; triad templates on the guitar + piano + other stems. Chords are rough.
7. **Engrave.** MusicXML with bass written an octave up, a stemless 4-string tab staff (Viterbi fingering), chord symbols, the recording time at each line start, flagged notes in parentheses, and a red "check" on bars to verify. MuseScore 4's CLI turns it into a US Letter PDF.
8. **Quality gates.** See below.

## Quality gates

There is no reference transcription, so every run measures itself with independent proxies. None is trusted alone.

| Gate | Pass | Fail | What it catches |
|---|---|---|---|
| Bass stem level vs the mix | ≥ −25 dB | < −35 dB | No bass found by the separator |
| Drum bleed in the bass stem | ≤ 0.15 | > 0.30 | Kick and snare leaking into the bass |
| Sounding bass covered by notes | ≥ 95% | < 85% | Missing notes |
| Notes cut short while the bass rings | ≤ 5% | > 15% | Choppy playback |
| Synth silent while the bass sounds (audio level) | ≤ 5% | > 15% | Gaps, judged on the rendered result |
| Same note name as the recording (audio level) | ≥ 85% | < 70% | Wrong notes; harmonic summation makes this timbre-independent |
| Notes with a doubtful octave | 0 | > 5% | Octave errors, from sub-octave or fifth energy, or a missing fundamental |
| Same pitch as Basic Pitch | ≥ 80% | < 60% | Disagreement with an independent model on the same stem |
| Note starts shared with Basic Pitch | F ≥ 0.6 | < 0.4 | Rhythm disagreement |
| Note starts' distance from the 16th grid | ≤ 25 ms | > 40 ms | Wrong tempo or grid |
| Notes on 8th-note positions | ≥ 75% | < 55% | Wrong pulse level, shifted downbeat, swing |
| Notes flagged uncertain | ≤ 10% | > 25% | Tracker disagreement |
| Bars that need an ear check (severity ≥ 3) | ≤ 15% | > 30% | Too many bars where independent checks disagree |

- **Verdict.** FAIL if any gate fails, WARN if any warns, otherwise PASS.
- **Separator fallback.** With `--separator auto`, a failing RoFormer stem or result triggers Demucs. The candidate with the fewest failing, then warning, gates wins.
- **Exit codes.** 0 means the sheet was written (PASS or WARN). 2 means FAIL: the sheet is marked DRAFT, or not written at all with `--strict`. 1 means error.
- **Ear-check bars.** Each bar gets a severity that adds up independent signals:
  - 2 points: coverage under 80%, or the rendered notes match the recording's pitch under 70% of the time;
  - 1 point, or 2 below 50%: Basic Pitch agrees under 70% of the time;
  - 2 points: the spectrum contradicts a note's octave;
  - 1 point: an ambiguous octave or an uncertain note.

  The report lists every bar, most severe first. Bars with severity 3 or more get a red "check" in the PDF, and the share of such bars is itself a gate (warn above 15%, fail above 30%).

## Install (macOS, Apple Silicon)

```
cd bass2sheet
uv sync -p 3.12                     # Python 3.12: most audio ML wheels don't support 3.13+ yet
brew install --cask musescore       # for the PDF; without it you still get MusicXML
```

**Second opinion (optional).** Basic Pitch needs its own environment, because its ONNX/TensorFlow stack doesn't mix well with torch:

```
uv venv -p 3.12 ../.venv-bp
uv pip install --python ../.venv-bp/bin/python basic-pitch onnxruntime 'setuptools<81'
export BASS2SHEET_BASIC_PITCH_PYTHON=$PWD/../.venv-bp/bin/python   # auto-found at ../.venv-bp otherwise
```

On first use, models download into `~/.cache/bass2sheet/models` (BS-RoFormer SW, about 300 MB) and the Hugging Face cache (Demucs). That cache also holds stems and pitch tracks, so a re-run on the same song takes about 30 s.

## Runtime (M-series MacBook, 3:55 song)

About 2:40 for a new song once the models are downloaded: separation about 50 s, pitch tracking about 60 s, Basic Pitch about 45 s, score and renders about 5 s. A re-run on the same song takes about 25 s, because stems, tracks and Basic Pitch output are cached.

## Options

```
bass2sheet SONG [-o OUT] [--title T] [--artist A] [--separator auto|roformer|demucs] [--strict]
                [--no-audio] [--second-opinion auto|off] [--basic-pitch-python PY] [--mscore PATH]
                [--cache-dir DIR] [--device mps|cuda|cpu] [-v]
```

## Known limits

- 4/4 rock and pop with a steady tempo. Swing, tempo changes and odd meters aren't modeled; the grid gates usually catch them.
- 4-string standard tuning for the tab.
- Chord symbols are rough. Slides, ghost notes and dead notes aren't transcribed.
- The thresholds were tuned on one song. The next step is an evaluation set with real bass MIDI (for example Slakh2100) to calibrate them against note-level F-measure.
