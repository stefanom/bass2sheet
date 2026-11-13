# Judging on your own songs

Scores against reference transcriptions ([evaluation.md](evaluation.md)) say whether the tool is exact. They cannot say whether its choices suit the person who will play from the sheet: which isolated bass sounds right, whether a simpler rhythm reads better, which fingering is easier to find. `bass2sheet-judge` puts those choices in front of you, on your own songs, one step of the pipeline at a time, and remembers what you picked.

```
uv run bass2sheet-judge build SONG... --dir judging     # add songs and prepare the first step
uv run bass2sheet-judge serve --dir judging             # judge in the browser: http://localhost:8765
uv run bass2sheet-judge report --dir judging            # the picks so far
uv run bass2sheet-judge update --dir judging            # after the tool changed: what came out differently
uv run bass2sheet-judge sheets --dir judging            # engrave the score snippets again (after an upgrade)
```

Everything stays on your machine: the page is served to `localhost` only, and the audio never leaves the working directory.

## The steps

You judge the pipeline in the order the tool runs it, and each step on its own output, not on the finished sheet. Each step sets one of the tool's standing choices against its alternative.

| Step | The two alternatives | What you hear | What you see |
|---|---|---|---|
| 1. Isolate the bass (`separator`) | BS-RoFormer or Demucs | The two isolated bass tracks; with "What was left behind", the record without each | A pitch picture of each track. Amber marks what sounds in one and hardly in the other |
| 2. Find the notes (`replucks`) | A repeated pitch is a new note at every pluck heard, or only at a clear, bright one | The isolated bass, and each version's notes as they were heard; with "Real bass left, notes right", both at once | The pitch picture of the isolated bass with the tracked pitch as a line and the notes as boxes. Amber boxes are in that version only |
| 3. Notes played together (`together`) | Two notes where the tool hears two played together, or one note at a time as the pitch tracker heard it | As in step 2 | As in step 2; the passages are the ones where two notes were heard |
| 4. Count the bars (`barlines`) | Beats from the Beat This! model, or from the built-in tracker | The record with each version's count, high on beat 1 | The sheet |
| 5. Write the rhythm (`rhythm`) | The rhythm as played, or stray sixteenths moved onto eighths and short rests closed | The record with the bass played as each sheet is written | The sheet |
| 6. Place it on the neck (`tab`) | Tab low on the neck with open strings, or with the hand kept in one position | The same sound for both | The tab |

A pitch picture shows what sounds in a bass track: time from left to right, low notes at the bottom, darker where it is louder. The lines marked E1, A1, D2 and G2 are the open strings. The lowest notes are blurred in time; that is the picture, not the track.

Pitch tracking has no second option to compare against yet. You judge it inside step 2, where the line shows what the tracker followed.

**The steps of a song open in order, and each is built on what you picked before it.** If you prefer the Demucs bass for a song, that song's notes are found in the Demucs bass, its bars counted on those notes, and so on: you never judge a step on an earlier result you turned down. A song takes a step's alternative when you picked it more often than the default in that song's passages. The page prepares the next step in the background as you finish one; that takes about a minute for a song.

If you change an earlier pick, the later comparisons built on the old one stop counting, and new ones are prepared. Change it back and the old ones return with their picks.

## A passage

Each comparison is a passage of 4 bars (2 at slow tempos) where the two versions differ. You can play its sources, switch between them without losing your place, and loop. You pick A, B, "no difference" or "neither is right", and can leave a note. After "neither is right" the page asks what is wrong, with one-click reasons (wrong notes, too high, too low, extra notes, missing notes, wrong timing) that go into the note. A note written or changed after the pick is saved as well. The page does not know which version is which; it is told only after your pick is saved, and then shows you.

While the audio plays, a line sweeps over both pictures or scores in step with it. On a score it passes through each note as it sounds, and the notes the two versions write differently are shaded, unless most of the passage differs. Click a picture or a score to play from there.

`build` picks the passages: for each song and step, the bars where the two versions differ most, taking a figure that repeats only once. Passages where one version has next to nothing (an intro, a fade) come last. `--per-song N` sets how many passages each song gives each step (default 3), and `--choices a,b` limits the steps. Running `build` again with more songs adds them and keeps every pick.

You can stop at any time; picks are saved as you make them. The bar under the title shows each step's progress and jumps to it.

## What the picks do

- **They decide what each song's later steps are built on**, as above.
- **They decide the defaults.** The standing choices are `DEFAULTS` in `src/bass2sheet/common.py`. `report` shows how often each alternative was picked and which songs took one; changing a default is then a one-line change, and the command-line flags (`--separator`, `--replucks`, `--rhythm`, `--beat-tracker`, `--tab`) choose the other alternative for one run.
- **They become tests.** Every judged passage is kept with the version you picked. After a change to the tool, `update` transcribes your songs again under the settings each pick was made with. A passage that still reads as you picked needs nothing; one that now comes out differently returns to the page, ahead of everything else, as a comparison between your pick and what the tool writes now. What is not settled is built again by the tool as it is now: steps nobody has judged, and passages that are still open or where you found neither version right, if they come out differently.

## Without listening: `check`

```
bass2sheet-judge check --dir judging
```

Every note the tool finds is also a sample of what the bass sounds like at that pitch in that song. `check` compares each note with the song's other notes of the same pitch and prints, per song, the share that do not sound like them, how many another pitch's notes explain better, and the ten seconds with the most such notes. It uses the settings your picks have settled on. Each run is appended to `checks.jsonl`, and the next run after a change to the tool shows what the share was before.

What it is good for, measured against ground truth in [evaluation.md](evaluation.md): ranking songs by how wrong they are, pointing at notes to listen to, and saying whether a change to the tool made a song better or worse when the difference is sizeable. What it cannot do: tell whether the isolated track is the bass, see a mistake that is made every time, or decide a question of taste. Your picks still decide those.

## The working directory

```
judging/
  settings.json       what `build` was run with
  songs.json          the songs, by path, and which steps are built for each
  heard/<song>/       the notes heard under each combination of settings, by version of the tool's code
  versions/<song>/    one snapshot per version, named by a hash of what it writes; never overwritten
  comparisons.json    the passages: which two versions each compares and what it was built on
  media/<id>/         the audio (FLAC), the pitch pictures, and the score snippets with MuseScore's positions
                      of their notes, which the cursor follows
  judgments.jsonl     your picks, appended; the latest pick for a passage counts
  checks.jsonl        what `check` measured, appended, with the version of the tool's code
```

The directory holds excerpts of your recordings. Keep it out of any repository.

## Limits

- The score snippets need MuseScore 4. Without it steps 3 to 5 have audio only.
- The written bass is a plain synthesized pluck. It tells you the notes and rhythm, not the tone.
- A passage is judged in isolation. A choice that helps one song and hurts another shows up only in the report.
- Step 3 appears only for songs where the two beat trackers disagree.
- With stems you supply yourself there is no step 1.
