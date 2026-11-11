# Judging on your own songs

Scores against reference transcriptions ([evaluation.md](evaluation.md)) say whether the tool is exact. They cannot say whether its choices suit the person who will play from the sheet: whether a simpler rhythm reads better, which fingering is easier to find, which of two imperfect transcriptions sounds closer. `bass2sheet-judge` puts those choices in front of you, on your own songs, and remembers what you picked.

```
uv run bass2sheet-judge build SONG... --dir judging     # transcribe each song under every alternative
uv run bass2sheet-judge serve --dir judging             # judge in the browser: http://localhost:8765
uv run bass2sheet-judge report --dir judging            # the picks so far
uv run bass2sheet-judge update --dir judging            # after the tool changed: what came out differently
```

Everything stays on your machine: the page is served to `localhost` only, and the audio never leaves the working directory.

## What you judge

Each comparison is a passage of 4 bars (2 at slow tempos) where two versions of the transcription differ. You can play the record, version A and version B, switch between them without losing your place, hear the bass alone, and read both scores for those bars. You pick A, B, "no difference" or "neither is right", and can leave a note. The page does not know which version is which; it is told only after your pick is saved, and then shows you.

What you hear for A and B is the record with the real bass taken out and each version's bass put in, played exactly as its sheet is written. That is deliberate: it is what you would play from that sheet.

Each comparison sets one of the tool's standing choices against its alternative:

| Choice | The two alternatives | What differs |
|---|---|---|
| `separator` | BS-RoFormer or Demucs isolates the bass | The notes heard |
| `replucks` | A repeated pitch is a new note at every pluck heard, or only at a clear, bright one | The notes heard |
| `rhythm` | The rhythm as played, or stray sixteenths moved onto eighths and short rests closed | The writing of the same line |
| `barlines` | Beats and bar lines from the Beat This! model, or from the built-in tracker | The count; each version plays the record with its own click, high on beat 1 |
| `tab` | Tab low on the neck with open strings, or with the hand kept in one position | The fingering; the sound is the same |

`build` picks the passages: for each song and choice, the bars where the two versions differ most, taking a figure that repeats only once. Passages where one version has next to nothing (an intro, a fade) come last. `--per-song N` sets how many passages each song gives each choice (default 3), and `--choices a,b` limits the choices. Running `build` again adds songs or choices and keeps every pick.

The page shows one passage of every choice for every song before a second of any, so a short sitting still touches everything. You can stop at any time; picks are saved as you make them.

## What the picks do

- **They decide the defaults.** The standing choices are `DEFAULTS` in `src/bass2sheet/common.py`. `report` shows how often each alternative was picked; changing a default is then a one-line change, and the command-line flags (`--separator`, `--replucks`, `--rhythm`, `--beat-tracker`, `--tab`) choose the other alternative for one run.
- **They become tests.** Every judged passage is kept with the version you picked. After a change to the tool, `update` transcribes your songs again and sorts the judged passages three ways: still written as you picked; written the way you turned down (the tool does not follow that pick); or written differently from both. Only the last kind comes back to the page, as a comparison between your pick and what the tool writes now, ahead of everything else.

## The working directory

```
judging/
  songs.json          the songs, by path
  versions/<song>/    one snapshot per version, named by a hash of what it writes; never overwritten
  comparisons.json    the passages, which two versions each compares, and the order they were shown in
  media/<id>/         record.flac, A.flac, B.flac, the bass alone, A.png and B.png
  judgments.jsonl     your picks, appended; the latest pick for a passage counts
```

The directory holds excerpts of your recordings. Keep it out of any repository.

## Limits

- The score snippets need MuseScore 4. Without it the comparisons have audio only.
- The written bass is a plain synthesized pluck. It tells you the notes and rhythm, not the tone.
- A passage is judged in isolation. A choice that helps one song and hurts another shows up only in the tally.
- `barlines` appears only for songs where the two beat trackers disagree.
