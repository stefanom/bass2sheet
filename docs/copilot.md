# The copilot

`bass2sheet` writes a sheet by machine, and the machine is wrong often enough that the sheet needs a person. `bass2sheet-copilot` turns that around: you are in charge of the score and the tool is your copilot. You get a sheet you have checked by ear, and the tool gets a record of every mistake it made, with the right answer.

```
bass2sheet-copilot add SONG... --dir scores [--judging DIR]
bass2sheet-copilot serve --dir scores        # then open http://localhost:8770: your songs, and the way in for a new one
bass2sheet-copilot report --dir scores       # what the tool got wrong so far
bass2sheet-copilot sheet SONG_ID --dir scores
```

`add` hears each song as the tool would for a sheet (a few minutes the first time, under a minute with the stems and pitch tracks in the cache). With `--judging DIR`, a song that is also in a [judging](judging.md) directory is heard with the settings your picks settled on there.

`serve` works on a directory with no song in it yet: the first one can be added on the page.

## From a recording to its score

`http://localhost:8770` shows the whole way in four stages, at the top of the page and in that order below it. The stage you are at is lit.

1. **Add a song**: drop a recording, or paste a link.
2. **Watch it being prepared**: the bass is isolated and its notes are found, then double-checked. The two double checks are marked in the row of steps: the own-sound check holds each note against the song's other notes of its pitch, and, when MuScriptor writes the notes, the second opinion holds them against bass2sheet's own. What they doubt is amber on the song's page. Under the row, a sentence says what the running step does.
3. **Fix the notes**: "Fix the notes" on a song's card opens its page, where you play the record and fix what is wrong.
4. **Make the score**: "Make the sheet" on the song's page, or "Make the score" on its card, writes the score and shows it. Once it is made the card says when, and "See the score" opens it.

## Your songs

The page lists the songs of the working directory. Each card shows the whole song as a strip of its pitch picture, and under it how far you are: dark green is what you have worked through, pale green what you have only heard. The lines below say it in numbers (how much is worked through, how many fixes) and whose notes the song started from: bass2sheet's own or MuScriptor's. "Fix the notes" (or a click on the card) opens the song where you stopped; the button beside it makes or shows its score.

A song's own page is at `/song#SONG_ID`. Links from before the songs had a page of their own (`/#SONG_ID`) still open the song.

## Adding a song on the page

Drop an audio file on the page, or choose one (or paste a link: see below). The file is copied to the copilot on this computer, kept in `uploads/` under the working directory, and prepared in the background with the code `bass2sheet-copilot add` runs. You can keep working on another song meanwhile. Several files wait their turn: one song is prepared at a time.

- **Whose notes.** Under the drop box you choose the note finder the song starts from: bass2sheet's own, or MuScriptor's where it is set up (`MUSCRIPTOR_NOTES` and `MUSCRIPTOR_GGUF` in the environment `serve` is started from: see [muscriptor.md](muscriptor.md)). Where it is not, the choice is greyed out and says so.
- **The steps.** While a song is prepared, the page shows the steps in the order they run, each waiting, running or done, with the time it took and what it found: reading the file (how long the music is), isolating the bass (by which model), following the pitch (in how much of the song a pitch was found), finding the notes (how many, from which to which), MuScriptor's notes and the second opinion when MuScriptor is chosen (how many notes only MuScriptor hears), beats and bars (tempo, bars, key), the own-sound check (how many notes the tool doubts), the picture (shown under the steps once it exists), ready. The steps are reported by the preparation itself as it goes, not guessed from the clock. Isolating the bass is the long one the first time: a minute or two on a recent Mac.
- **If it fails**, the step that failed is marked and the reason stays on the page until you try again or remove the song. Removing a song that never got ready deletes its file and everything made of it. "Cancel" does the same for one that is waiting or running.
- **What is taken.** Audio files (`.wav .flac .mp3 .m4a .aac .ogg .oga .opus .aif .aiff`) and the video files a song often comes in (`.mp4 .webm .mkv`), for the sound in them; up to 500 MB, from ten seconds to twenty minutes. A file with another ending, one with no sound in it that can be read, or a song that is already there is refused, and nothing of it is kept.
- **From a link.** Where [yt-dlp](https://github.com/yt-dlp/yt-dlp) is installed (`brew install yt-dlp`, or `pip install yt-dlp`), a box beside the drop box takes a YouTube link: the audio is fetched into `uploads/` and then prepared exactly like a file, with "Fetching the audio" as the first step. This is optional. Nothing is installed for you, and where yt-dlp is not found the box is not shown. `BASS2SHEET_DOWNLOADER=/path/to/yt-dlp` names the program where it is not on the PATH; `BASS2SHEET_DOWNLOADER=none` turns links off. It is for audio you have the right to download: your own recordings, music under a licence that allows it, or what the law where you live lets you copy for your own practice. A live stream, a video over twenty minutes or over 500 MB is not fetched, and a playlist link fetches only the one video.
- **Your recordings.** Add only recordings you have the rights to use. The copilot listens on this computer only (127.0.0.1) and takes changes only from its own pages; the file goes nowhere else.

Each song is prepared in a process of its own, so the memory the models take is given back when it is done.

## The page of a song

One song at a time, on a timeline that scrolls with the record. "‹ Songs" at the top left goes back to the songs.

- **The picture** is the isolated bass: pitch upwards, time to the right. Each dark bar is a steady tone. A note is a stack of bars: the note itself at the bottom and its overtones above it, which on a bass are often stronger than the note. A bar that bends is a slide. The gap between two bars is where one note gives way to the next, and a note played several times usually shows as several bars. (A plain spectrogram cannot show this for a bass: telling the low E from the F beside it takes most of a second of sound, which smears every note over its neighbours. Here each partial is located exactly from a tenth of a second.) The thin red line is the pitch the tracker followed. Faint vertical lines are the bars as the tool counts them; they are not settled at this stage and are only there to find your way.
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

**After a fix** in a bar that repeats, the tool offers the same fix for the repeats (see below). In a bar that does not, it looks for the other notes that carry the label you just changed and sound like the note you fixed, and offers to change those. It never changes a note you have touched or worked through. Of what it offers that way, less than half needs the fix unless the mistake is one the tool made all through the song, so listen to a few before taking them.

## Repeats

Music repeats itself, and the tool does not hear every repeat the same way. So the tool looks for the bars that sound alike, in the bass track itself and not in its own notes, which are the part that may be wrong. Two bars are compared sixteenth by sixteenth: where they sound the same, they should be written the same.

- **The letters over the timeline** name the bars that sound alike: every A is a repeat of the other As. Click a letter to see how many bars sound like that one and to step through them.
- **After a fix**, the tool looks at the other bars that sound the same at that spot and are written there as this bar was before you fixed it: the same mistake. A yellow bar offers to fix them too. You can step through them first, take them all, or decline. This is where one fix can stand for forty. It is never offered for a bar you have already worked through: that bar is yours.
- **Dashed notes and the marks under the letters** show where a bar you have not reached is written unlike a repeat that sounds the same there. Dark orange means the repeat is one you fixed, or one you worked through and left as it is; pale means only that most of the tool's own repeats agree against this bar. Click one to see what the repeat has, hear both, and take it or keep what is written. These are hints: worth a listen, not certainties.

How far to trust this, measured on recordings with known notes ([evaluation.md](evaluation.md)): carrying fixes to the repeats saves about a sixth of the fixing overall, a few percent on most songs and more than half on the ones where the tool makes the same mistake in every repeat. A fix carried to a bar with the same mistake is right about nine times in ten for a pitch or a re-pluck, less often for a note taken out; on two test songs in forty, taking every offer unheard would have cost more than it saved. So use "Show me the next" on the first few of an offer. A dashed note points at something wrong about two times in three, and what the repeat has is the right answer about six times in ten.

The bars are the tool's bars. Where its beat grid slips (an odd bar, a missed beat), the bars after it are still found as repeats of each other, but not of the bars before.

**What you leave as it is counts as right, as far as you have worked.** Playing past a note does not make it yours: people listen ahead, before they start fixing and after. What counts is everything you played up to your last fix. The band over the timeline shows it in dark green; what you have only heard is pale. When you have listened further and found nothing to fix, "Checked up to here" moves the mark to the playhead. The page picks up where you stopped.

## Starting from MuScriptor's notes

`bass2sheet-copilot add SONG --notes muscriptor --dir scores-muscriptor` prepares a song with MuScriptor's notes in place of the tool's own ([muscriptor.md](muscriptor.md)). The page says "notes by MuScriptor" beside the progress. A note is amber where the tool's own finder does not hear it, and the note's line says so. A song keeps the notes it was added with, so MuScriptor's version of a song you have worked on goes in another working directory; `report` then counts MuScriptor's mistakes there and the tool's in the first.

## The sheet

"Make the sheet" writes the PDF, the MusicXML and a MIDI file from your notes, with the bars, the rhythm and the tab as the tool writes them (those pieces get their own steps later). The line under the title says how much of the song you have worked through.

The score then opens in a panel over the song's page: its pages one under the other, as MuseScore draws them, with buttons to download the PDF, the MusicXML and the MIDI file, and "Make it again" after more fixes. "Close", Esc or a click beside the panel goes back to the notes, which are as you left them. The same view has an address of its own, `/score#SONG_ID`.

**Versions.** Every time the score is made it is kept, not written over: the panel has a list of the versions, each with when it was made, how many fixes it has and how much of the song was checked by ear. Choose one to see it as it was and to download its files; "Make it again" adds a new one on top. A score made again with no note changed since is the version it was. The files of the latest are also where they always were (`sheet/bass.pdf` and so on).

Drawing the pages needs [MuseScore 4](https://musescore.org) on the computer. Without it the MusicXML and the MIDI file are still written, and the panel says so and offers them.

## What the tool learns

`report` compares the tool's version with yours: wrong pitch, wrong octave, extra note, missed note, two notes written as one, one note written as two, a start or an end moved. Notes you left alone in what you have worked through count as right. That is ground truth on your own songs, which no dataset has.

## The working directory

```
scores/
  songs.json            the songs and the settings each was heard with
  uploads/              the files added on the page, as they came, and the audio fetched from links
  <song>/machine.json   what the tool heard: never changed
  <song>/notes.json     your version, what you have played through, and the undo history
  <song>/log.jsonl      every fix, offer and play-through, appended
  <song>/audio/         the record and the isolated bass, for the page, and their slowed versions once used
  <song>/picture.bin, pitch.bin, desk.npz   the pitch picture, the tracked pitch, what the guesses need
  <song>/repeats.npz    which bars sound alike, measured once from the bass track
  <song>/sheet/         the score made from your notes: PDF, MusicXML, MIDI and a picture of each page
  <song>/sheet/v1, v2…  every version of the score as it was made, each with its own files
```

The directory holds your recordings. Keep it out of any repository. Preparing a song again after the tool has changed replaces `machine.json` and leaves your notes alone.

## Limits

- This is the notes piece. Bar lines, written rhythm and tab are still the tool's; steps for them come next.
- Two notes played together are shown and played, but cannot be added or removed on the page yet.
- The tool's doubts (amber) find about a quarter of its wrong notes in a full mix, so they are no substitute for playing through.
