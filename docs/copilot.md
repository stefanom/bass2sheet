# The copilot

`bass2sheet` writes a sheet by machine, and the machine is wrong often enough that the sheet needs a person. `bass2sheet-copilot` turns that around: you are in charge of the score and the tool is your copilot. You get a sheet you have checked by ear, and the tool gets a record of every mistake it made, with the right answer.

```
bass2sheet-copilot add SONG... --dir scores [--judging DIR]
bass2sheet-copilot serve --dir scores        # then open http://localhost:8770
bass2sheet-copilot report --dir scores       # what the tool got wrong so far
bass2sheet-copilot sheet SONG_ID --dir scores
```

`add` hears each song as the tool would for a sheet (a few minutes the first time, under a minute with the stems and pitch tracks in the cache). With `--judging DIR`, a song that is also in a [judging](judging.md) directory is heard with the settings your picks settled on there.

## The page

One song at a time, on a timeline that scrolls with the record.

- **The picture** is the isolated bass: pitch upwards, time to the right, dark where it sounds. The thin red line is the pitch the tracker followed. Faint vertical lines are the bars as the tool counts them; they are not settled at this stage and are only there to find your way.
- **The notes** are boxes on the picture, on the row of their pitch. Green is as the tool heard it, amber is a note the tool itself doubts (it does not sound like the song's other notes of that pitch), blue is one you have changed or added. A dashed outline shows where the tool had a note you moved to another pitch.
- **Moving about.** Drag the timeline with the mouse (the pointer is a hand wherever it is not on a note) or scroll; a click without dragging puts the playhead there. While the song plays the timeline follows the playhead until you move it yourself; then it stays where you put it, and follows again the next time you press play.
- **Slower.** "Speed" plays the song at 90, 80, 70, 60 or 50% at the same pitch (`[` and `]` step through them). The record, the isolated bass and the notes slow down together and stay in step. The first time a speed is used for a song it takes up to a minute to prepare; after that it is kept. With [Rubber Band](https://breakfastquay.com/rubberband/) installed (`brew install rubberband`) the slowed sound keeps the attacks and the low notes clean; without it the ffmpeg that comes with the tool is used, which is rougher.
- **What you hear** is your choice of three: the record, the isolated bass, and the notes. The notes are played in the bass's own sound: each by a piece of the bass track cut from another note of the same pitch. A right note doubles the record; a wrong one stands out. With headphones, "bass left, notes right" puts the isolated bass in one ear and the notes in the other.

## Working through a song

Press Space and listen. When something is wrong, press Space again, then `←` to select the note you just heard (again for the one before).

| To | Do |
|---|---|
| change its pitch | `1` to `6` takes one of the tool's guesses; `↑` `↓` moves a semitone, with Shift an octave; or drag the note |
| hear it again in its place | Enter |
| remove a note that is not there | Delete |
| add a note that is missing | `A` at the playhead, or double-click where it belongs; a note sounding there ends where the new one starts |
| make one note two | `S` splits it at the playhead, or in the middle |
| make two notes one | `M` joins it with the next; Shift+`M` with every repeat of it that follows (a held note the tool chopped up) |
| work on several notes at once | Shift-click another note, or Shift with `←` `→`, selects all the notes in between; then Delete removes them, `M` joins them into one, `↑` `↓` moves them together (Shift for an octave) |
| move where it starts or ends | drag an end, or Alt with `←` `→` (Shift for the end), 10 ms a press |
| take what a repeat has, or keep what is written | click a dashed note or a mark under the letters, then "Use it" (`U`) or "Keep what is written" (`K`) |
| take it back | `Z` |

After every change the passage plays again, so you hear the result.

**The guesses** come from everything that had an opinion about the note: the spectrum's own peaks, the second pitch tracker, the first tracker's runner-up pitches, the octave either way, and how well the song's notes of each pitch explain this one's sound. Measured on recordings with known notes, the right pitch is the first guess for 59% of wrong notes on real bass alone, and among the first four for 74%; for a note an octave off it is among the first three nearly always. When it is not among them, the arrow keys are as quick.

**After a fix** in a bar that repeats, the tool offers the same fix for the repeats (see below). In a bar that does not, it looks for the other notes that carry the label you just changed and sound like the note you fixed, and offers to change those. It never changes a note you have touched or played past. Of what it offers that way, less than half needs the fix unless the mistake is one the tool made all through the song, so listen to a few before taking them.

## Repeats

Music repeats itself, and the tool does not hear every repeat the same way. So the tool looks for the bars that sound alike, in the bass track itself and not in its own notes, which are the part that may be wrong. Two bars are compared sixteenth by sixteenth: where they sound the same, they should be written the same.

- **The letters over the timeline** name the bars that sound alike: every A is a repeat of the other As. Click a letter to see how many bars sound like that one and to step through them.
- **After a fix**, the tool looks at the other bars that sound the same at that spot and are written there as this bar was before you fixed it: the same mistake. A yellow bar offers to fix them too. You can step through them first, take them all, or decline. This is where one fix can stand for forty. It is never offered for a bar you have already been through: that bar is yours.
- **Dashed notes and the marks under the letters** show where a bar you have not reached is written unlike a repeat that sounds the same there. Dark orange means the repeat is one you fixed, or one you played past and left as it is; pale means only that most of the tool's own repeats agree against this bar. Click one to see what the repeat has, hear both, and take it or keep what is written. These are hints: worth a listen, not certainties.

How far to trust this, measured on recordings with known notes ([evaluation.md](evaluation.md)): carrying fixes to the repeats saves about a sixth of the fixing overall, a few percent on most songs and more than half on the ones where the tool makes the same mistake in every repeat. A fix carried to a bar with the same mistake is right about nine times in ten for a pitch or a re-pluck, less often for a note taken out; on two test songs in forty, taking every offer unheard would have cost more than it saved. So use "Show me the next" on the first few of an offer. A dashed note points at something wrong about two times in three, and what the repeat has is the right answer about six times in ten.

The bars are the tool's bars. Where its beat grid slips (an odd bar, a missed beat), the bars after it are still found as repeats of each other, but not of the bars before.

**What you play past without touching counts as right.** The green band over the timeline shows it. Only playing through counts, not the short replays after a change. The page picks up where you stopped.

## The sheet

"Make the sheet" writes the PDF, the MusicXML and a MIDI file from your notes, with the bars, the rhythm and the tab as the tool writes them (those pieces get their own steps later). The line under the title says how much of the song you have played through.

## What the tool learns

`report` compares the tool's version with yours: wrong pitch, wrong octave, extra note, missed note, two notes written as one, one note written as two, a start or an end moved. Notes you played past and left alone count as right. That is ground truth on your own songs, which no dataset has.

## The working directory

```
scores/
  songs.json            the songs and the settings each was heard with
  <song>/machine.json   what the tool heard: never changed
  <song>/notes.json     your version, what you have played through, and the undo history
  <song>/log.jsonl      every fix, offer and play-through, appended
  <song>/audio/         the record and the isolated bass, for the page, and their slowed versions once used
  <song>/picture.bin, pitch.bin, desk.npz   the pitch picture, the tracked pitch, what the guesses need
  <song>/repeats.npz    which bars sound alike, measured once from the bass track
  <song>/sheet/         the score made from your notes
```

The directory holds your recordings. Keep it out of any repository. Preparing a song again after the tool has changed replaces `machine.json` and leaves your notes alone.

## Limits

- This is the notes piece. Bar lines, written rhythm and tab are still the tool's; steps for them come next.
- Two notes played together are shown and played, but cannot be added or removed on the page yet.
- The tool's doubts (amber) find about a quarter of its wrong notes in a full mix, so they are no substitute for playing through.
