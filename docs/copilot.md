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
- **Your recordings.** Add only recordings you have the rights to use. The copilot listens on this computer only (127.0.0.1), unless it is started with `--lan` for the play-along page, and even then it serves another device only when that device shows the key of its link, and takes changes only from its own pages on this computer; the file goes nowhere else.

Each song is prepared in a process of its own, so the memory the models take is given back when it is done.

## The page of a song

One song at a time, on a timeline that scrolls with the record and fills the browser window. "‹ Songs" at the top left goes back to the songs. Everything can be done with the mouse; the keys are shortcuts. The Help button (or `?`) shows what the colours mean, what the mouse does and the keys.

- **The picture** is the isolated bass: pitch upwards, time to the right. Each dark bar is a steady tone. A note is a stack of bars: the note itself at the bottom and its overtones above it, which on a bass are often stronger than the note. A bar that bends is a slide. The gap between two bars is where one note gives way to the next, and a note played several times usually shows as several bars. (A plain spectrogram cannot show this for a bass: telling the low E from the F beside it takes most of a second of sound, which smears every note over its neighbours. Here each partial is located exactly from a tenth of a second.) The thin red line is the pitch the tracker followed. Faint vertical lines are the bars as the tool counts them; they are not settled at this stage and are only there to find your way.
- **The notes** are boxes on the picture, on the row of their pitch. Green is as the tool heard it, amber is a note the tool itself doubts (it does not sound like the song's other notes of that pitch), blue is one you have changed or added. A dashed outline shows where the tool had a note you moved to another pitch.
- **Moving about.** The mouse wheel, or a pinch on the trackpad, zooms in time around the pointer (`+` and `-` do it around the playhead). A sideways swipe, or Shift with the wheel, moves through the song, and so does dragging the timeline (the pointer is a hand wherever it is not on a note); a click without dragging puts the playhead there. Drag in the ruler at the top and the playhead follows the pointer, playing the spot under it as it goes: the way to find the moment a note begins. While the song plays the timeline follows the playhead until you move it yourself; then it stays where you put it, and follows again the next time you press play.
- **Rows.** "Rows" makes the pitch rows taller or shorter; "Fit" puts all the pitches in the window. With rows taller than the window, drag the timeline up and down, or hold Alt and turn the wheel.
- **The picture's look.** "Picture" chooses its colours (blue on white, dark, black on white, white on black), how strongly the faint tones show (the slider bends the curve from loudness to colour: to the right the faint tones come up, to the left only the strongest are left), and how solid the note boxes are (to the left they are outlines and the picture shows through). The page remembers these, the zoom and the rows.
- **Slower.** "Speed" plays the song at 90, 80, 70, 60 or 50% at the same pitch (`[` and `]` step through them). The record, the isolated bass and the notes slow down together and stay in step. The first time a speed is used for a song it takes up to a minute to prepare; after that it is kept. With [Rubber Band](https://breakfastquay.com/rubberband/) installed (`brew install rubberband`) the slowed sound keeps the attacks and the low notes clean; without it the ffmpeg that comes with the tool is used, which is rougher.
- **What you hear** is your choice of three: the record, the isolated bass, and the notes. The notes are played in the bass's own sound: each by a piece of the bass track cut from another note of the same pitch. A right note doubles the record; a wrong one stands out. With headphones, "bass left, notes right" puts the isolated bass in one ear and the notes in the other.

## Working through a song

Press Space and listen. When something is wrong, press Space again, then `←` to select the note you just heard (again for the one before).

| To | Do |
|---|---|
| change its pitch | `1` to `6` takes one of the tool's guesses (or click one); `↑` `↓` moves a semitone, with Shift an octave ("▲ Octave", "▼ Octave"); or drag the note |
| put it where it sounds | drag the note: up and down changes its pitch, left and right moves it in time, in one go. While you hold it, it is see-through, a line marks its start and its end right up the picture, and dotted boxes show where its overtones lie. Moved earlier, the note before gives way; moved later against the next note, its end stays where that note starts |
| hear it again in its place | Enter, or "Hear it" |
| remove a note that is not there | Delete, or the "Delete" button |
| add a note that is missing | `A` at the playhead, or double-click where it belongs; a note sounding there ends where the new one starts |
| make one note two | `S`, or "Split", splits it at the playhead, or in the middle |
| make two notes one | `M`, or "Join with next"; Shift+`M` joins it with every repeat of it that follows (a held note the tool chopped up) |
| work on several notes at once | Shift-click another note, or Shift with `←` `→`, selects all the notes in between; then Delete removes them, `M` joins them into one, `↑` `↓` moves them together (Shift for an octave); the same as buttons under the timeline |
| move where it starts or ends | drag an end: it can be grabbed from a little outside the note too, and the pointer turns into a double arrow; or Alt with `←` `→` (Shift for the end), 10 ms a press |
| take what a repeat has, or keep what is written | click a dashed note or a mark under the letters, then "Use it" (`U`) or "Keep what is written" (`K`) |
| take it back | `Z` |

After every change the passage plays again, so you hear the result.

**The guesses** come from everything that had an opinion about the note: the spectrum's own peaks, the second pitch tracker, the first tracker's runner-up pitches, the octave either way, and how well the song's notes of each pitch explain this one's sound. Measured on recordings with known notes, the right pitch is the first guess for 59% of wrong notes on real bass alone, and among the first four for 74%; for a note an octave off it is among the first three nearly always. When it is not among them, the arrow keys are as quick.

**After a fix** in a bar that repeats, the tool offers the same fix for the repeats (see below). In a bar that does not, it looks for the other notes that carry the label you just changed and sound like the note you fixed, and offers to change those. It never changes a note you have touched or worked through. Of what it offers that way, less than half needs the fix unless the mistake is one the tool made all through the song, so listen to a few before taking them.

## Repeats

Music repeats itself, and the tool does not hear every repeat the same way. So the tool looks for the bars that sound alike, in the bass track itself and not in its own notes, which are the part that may be wrong. Two bars are compared sixteenth by sixteenth: where they sound the same, they should be written the same.

- **The song map**, the thin line over the timeline, is the whole song at once: the patterns with their letters, a frame around the part the timeline shows, the playhead, what you have worked through (dark green) and only heard (pale), the orange marks, and a dark line under every bar that sounds like the current one (the bar you chose, or the one the selected note or the playhead is in). Click or drag in it and the timeline goes there.
- **The letters over the timeline** name the bars that sound alike: every A is a repeat of the other As. Click a letter and the other bars that sound like that one line up under the timeline, one strip a bar (see below).
- **After a fix**, the tool looks at the other bars that sound the same at that spot and are written there as this bar was before you fixed it: the same mistake. They line up under the timeline too, under a yellow line that says how many there are. This is where one fix can stand for forty. It is never offered for a bar you have already worked through: that bar is yours.
- **The strips.** Each strip is one bar that sounds like the one above it, at the timeline's own scale and right under that bar, so the eye goes straight down: its own piece of the pitch picture, its notes, and in dashed orange what the bar above has where the two are written differently (the line of the strip says it in words: "written F2, bar 50 has F3"). The timeline itself stays where it is. Songs repeat with small differences that are meant, so **nothing changes in a bar until you say so for that bar**:
  - "▶ As written" plays that bar against the record as it is written now; "▶ With the change" plays it as it would be. The playhead and the timeline do not move.
  - "✓ Change it here" changes that bar and only that bar. "Leave it" keeps it as written.
  - A changed strip says "✓ Changed" and has "↶ Undo this one", which puts that bar back whatever you did after it (unless you have changed the same notes again since: then take that back first). `Z` and the Undo button take back the last thing as always.
  - "Change all N without hearing them" is there for a mistake you know is everywhere. It is the last button of the line, asks twice, and counts as one fix: Undo takes all of them back together.
  - With an offer open, `O` goes to the next bar still to decide and plays it with the change, `X` leaves all that are left, `Y` changes all that are left.
- **Dashed notes and the marks under the letters** show where a bar you have not reached is written unlike a repeat that sounds the same there. Dark orange means the repeat is one you fixed, or one you worked through and left as it is; pale means only that most of the tool's own repeats agree against this bar. Click one to see what the repeat has, hear both, and take it or keep what is written. These are hints: worth a listen, not certainties.

How far to trust this, measured on recordings with known notes ([evaluation.md](evaluation.md)): carrying fixes to the repeats saves about a sixth of the fixing overall, a few percent on most songs and more than half on the ones where the tool makes the same mistake in every repeat. A fix carried to a bar with the same mistake is right about nine times in ten for a pitch or a re-pluck, less often for a note taken out; on two test songs in forty, taking every offer unheard would have cost more than it saved. So hear a bar both ways before changing it. A dashed note points at something wrong about two times in three, and what the repeat has is the right answer about six times in ten.

The bars are the tool's bars. Where its beat grid slips (an odd bar, a missed beat), the bars after it are still found as repeats of each other, but not of the bars before.

**What you leave as it is counts as right, as far as you have worked.** Playing past a note does not make it yours: people listen ahead, before they start fixing and after. What counts is everything you played up to your last fix. The band over the timeline shows it in dark green; what you have only heard is pale. When you have listened further and found nothing to fix, "Checked up to here" moves the mark to the playhead. The page picks up where you stopped.

## Starting from MuScriptor's notes

`bass2sheet-copilot add SONG --notes muscriptor --dir scores-muscriptor` prepares a song with MuScriptor's notes in place of the tool's own ([muscriptor.md](muscriptor.md)). The page says "notes by MuScriptor" beside the progress. A note is amber where the tool's own finder does not hear it, and the note's line says so. A song keeps the notes it was added with, so MuScriptor's version of a song you have worked on goes in another working directory; `report` then counts MuScriptor's mistakes there and the tool's in the first.

## The sheet

"Make the sheet" writes the PDF, the MusicXML and a MIDI file from your notes, with the bars and the rhythm as the tool writes them (those pieces get their own steps later) and the tab in the song's fingering, which you choose on its harmony page. The line under the title says how much of the song you have worked through.

The score then opens in a panel over the song's page: its pages one under the other, as MuseScore draws them, with buttons to download the PDF, the MusicXML and the MIDI file, and "Make it again" after more fixes. "Close", Esc or a click beside the panel goes back to the notes, which are as you left them. The same view has an address of its own, `/score#SONG_ID`.

**Versions.** Every time the score is made it is kept, not written over: the panel has a list of the versions, each with when it was made, how many fixes it has and how much of the song was checked by ear. Choose one to see it as it was and to download its files; "Make it again" adds a new one on top. A score made again with no note changed since is the version it was. The files of the latest are also where they always were (`sheet/bass.pdf` and so on).

Drawing the pages needs [MuseScore 4](https://musescore.org) on the computer. Without it the MusicXML and the MIDI file are still written, and the panel says so and offers them.

## What the tool learns

`report` compares the tool's version with yours: wrong pitch, wrong octave, extra note, missed note, two notes written as one, one note written as two, a start or an end moved. Notes you left alone in what you have worked through count as right. That is ground truth on your own songs, which no dataset has.

## Judging the test songs

The tool's scores ([evaluation.md](evaluation.md)) are measured on research datasets. A score measured on songs unlike yours says little about yours, and whether they are like yours is for you to say.

```bash
bass2sheet-copilot testset ~/data/RWC/RWC-P --annotations ~/data/rwc-annotations \
    --scores bass2sheet=runs/rwc --scores MuScriptor=runs/rwc-muscriptor --dir tests
bass2sheet-copilot testset ~/data/babyslakh_16k --name BabySlakh --scores bass2sheet=runs/babyslakh --dir tests
bass2sheet-copilot serve --dir tests        # then open http://localhost:8770/tests
```

`testset` adds a dataset's songs to a working directory as ordinary songs: the ones a `bass2sheet-eval` run measured, each with the score every note finder named in `--scores` got on it. The page at `/tests` lists them by set, says what kind of recordings the set holds and what is known of each song (for RWC Popular, for one, whether the bass was played live).

- **Listen** plays the record from the stretch where the bass is busiest; **Bass alone** plays the isolated bass from the same moment. The bar at the bottom moves through the song and switches between the two. **Open** shows the song's notes in the copilot.
- For each song you answer **Like my music**, **Partly** or **Not like it**, and can write what is different. Pressing the answer again takes it back.
- The table at the top counts the scores by your answers as you give them: what the note finders score on the songs that are like yours, and how many such songs there are. `bass2sheet-copilot report --dir tests` prints the same table and your notes.

Your answers are kept in `ratings.json` in the working directory, and every change in `ratings.jsonl`.

## Two notes played together

A note with another played together with it is drawn as two boxes, one above the other, joined at the left; under the timeline it reads `F#2 + G2`. Each of the two can be handled by itself:

- **Click the one you mean.** It gets the bright outline, the other keeps its own.
- **Delete** has a button for each (`Delete F#2`, `Delete G2`) and one for both. ⌫ takes out the one you clicked. When the lower one goes, the other becomes the note.
- **Drag the upper one** up or down to change it alone, or use `↑` `↓` after clicking it. Laid on the note itself, it is gone. Dragging the lower one moves the note in pitch and time as always; its time is the time of both.

Two notes are only written together where they are heard. bass2sheet's own finder has strict tests for it ([evaluation.md](evaluation.md#notes-played-together)), and with `--notes muscriptor` a second note is written only where bass2sheet's own finder wrote the same two: left to itself MuScriptor writes a second note every hundred notes or so, and two or three in a hundred of those are in the answer key ([evaluation.md](evaluation.md)). Two notes a semitone or a tone apart are never written together: that low they grate, and nobody plays them.

## An answer key from somebody's transcription

For most well-known songs somebody has written the bass line down: a MIDI file of the whole arrangement, a Guitar Pro tab, a score. `reference` takes such a file as an answer key for one of your songs:

```bash
bass2sheet-copilot reference SONG_ID billie-jean.mid --dir scores
bass2sheet-copilot report --dir scores
```

The file counts bars and beats, not the seconds of your recording, and it may be in another key, at another tempo, with an intro of another length or a verse more or less. So it is first lined up with the recording (`reference.py`): its key and tempo are found, the two are run side by side, what one has and the other lacks is jumped over, and the timing is settled on the recording's onsets. How close that comes is measured in [evaluation.md](evaluation.md). Then the file's bass part, in your recording's time, is kept beside the song (`reference/NAME.json`).

`report` scores the song's notes against each key, on the part of the recording the key covers: the tool's notes, and where you have worked, the tool's notes and your version side by side. A note counts as right when the key has that pitch starting within a tenth of a second. Beside the score it says what becomes of the key's notes: the version has the note, has it an octave away, has another pitch there, or has nothing there.

- **It is a reading, not the truth.** Two transcriptions of one song differ, most often in the octave of a note and in what they leave out. Give it several and see where they agree.
- **Formats.** MIDI is read as it is. Guitar Pro, MusicXML and MuseScore files are turned into MIDI by MuseScore first, which has to be installed.
- **The bass part** is the one called a bass, by its name or its General MIDI program; the command says which it took and which others it could have been (`--part N` takes another). If the file writes the bass an octave from where it sounds, the whole part is moved to where the isolated bass sounds.
- **A file that is only part of the song** is placed where it fits best. In a song that repeats, that can be another occurrence of the same music.
- **Where the file comes from matters.** Some sites do not allow their tabs to be downloaded by a program, or to be used to evaluate software at all. Read their terms; `reference` only reads a file you already have.
- The key's notes are not shown on the song's page yet.

## Playing along

The sheet is for playing the song yourself, with the record. The play-along page is the sheet made for that: on a phone held sideways, on the music stand or the amp.

```
bass2sheet-copilot serve --dir scores --lan
```

prints the link to open on the phone, for example `http://192.168.1.20:8770/play?key=40718253`; the copilot's first page shows the same link at the top, with a Copy button. The phone has to be on the same Wi-Fi as the computer. On the computer itself the page is at `http://localhost:8770/play`, with or without `--lan`. In Safari, Share > Add to Home Screen makes it an app of its own that opens full screen, without the browser's bars; Chrome on Android offers "Install app" or "Add to Home screen".

- **What you hear** is the record without the bass: you are the bass player. "Record" plays all of it, "Bass" the isolated bass alone, to check a passage against what the record's player does. The speeds are the copilot's: 100 to 50%, at the same pitch. The first time a speed is used for a song it takes up to a minute to prepare, and the page says so while it waits.
- **What you see** is the score written from your notes, staff and tab, one line at a time. The line being played is large at the top, the next one is under it, a little smaller. When the line ends the next one moves up and the one after it appears: there is never a page to turn, and you can always read ahead.
- **The note to play is highlighted**, and a thin red line moves through the bar. Both follow the record itself: every bar and every beat is where the song's own beat grid has it, so a band that pushes a chorus or slows into an ending is followed, bar by bar. A rest is highlighted in grey.
- **The controls** are big enough for a thumb. Play and pause. The arrows go back to the start of the line (again: the line before) and on to the next line. Touch a bar, in either line, to go there. The strip above the controls is the whole song: touch or drag it to go anywhere. "Loop" repeats four bars from the bar being played; `−` and `+` make it one to sixteen bars, and with the loop on, touching a bar starts the loop there. The looped bars are tinted blue. With a keyboard: Space, `←` `→`, `L`, `[` `]`.
- **Songs and settings** are behind the button at the left: the songs of the working directory; how many bars a line holds at most (2, 3 or 4: fewer bars are wider apart, and a busy song gets fewer by itself where that many do not fit across the line); "the sound arrives late by", for a wireless speaker or headphones, which delay the sound by a tenth of a second or more: raise it until the highlight is on the beat; and whether the screen stays on while the song plays. The page remembers these, and where you stopped in each song, on the phone.

The score is the one "Make the sheet" writes, from your notes as they stand when the page opens the song: fix a note in the copilot, open the song again on the phone, and the line is engraved again. Until every step has its own page, the bars, the rhythm and the tab are the tool's.

**How it is made.** The record without the bass is the sum of the other separated stems, at the level they have in the record, written as `audio/nobass.flac` beside `record.flac` and `bass.flac` when a song is prepared. For a song prepared before this existed it is made the first time it is asked for, from the stems in the cache (a second or two); if they are no longer there, the isolated bass is taken out of the record instead, which leaves a little more of the bass in. The score is engraved by MuseScore with each line on a page of its own, as pictures, together with MuseScore's own account of where it put every measure and every note; the pictures are cut to the lines and kept in `<song>/play/` until your notes change. Each bar and note is then given the time of the sixteenth of the beat grid it is written on. A note the score writes a sixteenth early or late is highlighted where it is written, not where it was played.

**On the home network.** Without `--lan` the server answers only the computer it runs on. With it, another device is served only when it shows the key: the eight digits at the end of the link. A home network's addresses are every network's addresses, the office's and the cafe's too, and a laptop moves; so being on the same network is not enough, and a device without the key gets a page that says where the link is and nothing of your songs. The key is made once and kept in the working directory (`phone_key`), so a link saved on the phone goes on working; delete the file to make a new one. A device opens the link once and keeps a cookie from then on; an app added to the home screen starts from the link itself. After twenty wrong keys in a row no key is taken for ten minutes. A device that is let in can open the pages, listen and read along; it cannot change anything: every fix, every "checked up to here" and "Make the sheet" is refused unless it comes from the computer itself. The server answers the addresses a home router gives out and nothing further away. On a phone's hotspot, or any network that gives the computer no such address, `serve --lan` says so and the phone cannot open the page.

**What is not there yet.**

- It has been looked at in a desktop browser at a phone's size, not yet on a phone. Two things can only be found out there: whether the screen stays on (a page opened over plain http cannot ask the phone to stay awake, so the page plays a small silent film beside the song, which keeps most phones lit; if yours dims, set Auto-Lock to Never while you play), and how late the sound is on your speaker.
- A loop jumps back with a short gap, a tenth of a second or two, while the phone finds its place in the sound.
- No count-in before a loop or before the song, and no click.
- The phone needs the computer: nothing is kept on the phone, and a company laptop's firewall may not let the phone in at all.

## The harmony of a song

"Harmony" on a song's card, or at the top of its page, opens `http://localhost:8770/harmony#SONG`: the song as its chords. The page is for understanding a song before you play it, and for remembering it.

- **The song in short**, under its title: the few loops that carry it, written as steps above home. `i – ♭VII – ♭VI – V, 2 bars each, 39% of the song` is most of what there is to remember of a verse.
- **Home** is the note the song comes back to. The tool finds a key; a major key and a minor key share the same notes, and which of the two is home is heard, not counted. Pick the other one, the next best key the tool found, or any key, and every name on the page follows. Where the tool hears the song change key, it says where. The choice is kept in `harmony.json` beside the song; nothing of your notes changes.
- **The map** shows every chord by the job it does in the key: it rests (Home), drifts (Away) or pulls (Tension). Inside a region the darker, flatter chords are on the left and the brighter ones on the right; the row facing the middle is in the key, the next is one step outside it, the rim is far outside. Filled circles are the chords of this song, and the arrows are the steps it takes, bolder where it takes them often. Two more views of the same chords: **Wheel**, the circle of fifths with home at the top, and **Semitones**, the twelve notes in order, which a bass player reads as frets. Click a chord to go to where the song is on it next.
- **The bar that stays at the top** has the chord that sounds with its name, its step and its job; the loop of the part that is playing, one box a bar; Play, "Loop this part", and what you hear (the record, the record without its bass, the bass alone). Under it is **the bass neck**, which shows only what the bass plays: the places of this bar's notes, ringed and joined in the order they are played, each marked with what it is to the chord (R, 5, ♭7); the note that sounds, filled, with its fret and string in words beside it; and faintly the other places this part of the song uses. It is the shape your hand makes, and a grey box marks where the hand is: four frets from the first finger.
- **Fingering**, beside the neck, is how the notes are put on it. A pitch can be played in up to four places, and the choice decides how hard the line is to play. "Fewest hand moves" is what every song gets: the hand stays where it is for as long as the notes allow, a move is made where there is time for it, strings are skipped as little as possible, an open string is used where it saves a move, and a bar that comes again is fingered the same way. "No open strings" frets every note that can be fretted: shapes that can be moved along the neck, at the price of more moves. "Lowest frets" puts every note as near the nut as it goes, where notes are easiest to find. The page says what each asks of the hand over the whole song: the times it has to move, the string skips, the share of notes on open strings. The choice is kept for the song, and the tab of its score and of its play-along page follows it.
- **Hear the chords** plays the chords the page names in a plain keyboard sound, over whatever you are hearing. A wrong chord is heard at once; with "Without bass" it is also a backing to play over.
- **Change this chord** is for the bar that sounds. It stops the song and opens a row of chords to try: what the tool heard, the chords this song uses, the chords of the key, or any root and kind you pick. A click on one plays the bar from the record with that chord in the keyboard sound and changes nothing. When it is the right one, "Change this bar" puts it there, and "Change all N bars like it" puts it at the same place in every round of the part, in the bars that hold the same chord as this one. "Back to what the tool heard" undoes a bar. Your chords are kept in `harmony.json`; the summary, the parts, the map and the profile follow them, and each change is written to the song's log, as a note fix is.
- **The song in parts** cuts the song into the loops it goes round and the passages between them. A part is boxed like a rehearsal mark, so that part A is not read as the chord A. Each loop is shown once, with how often it goes round and where; under each chord is what the bass plays in that bar as steps of the chord (`R R 5 8 5 4`: root, root, fifth, octave, fifth, fourth). "Play it" goes to the part.
- **The profile** draws how far from rest the song is from start to end: flat stretches at home, the climbs before a return. **What the bass plays** says how long the bass spends on roots, fifths, thirds, sevenths and passing notes, how many chords it starts on their root, and how the roots move.

The map's rules are rules of thumb from functional harmony, written down in `copilot_harmony.py` (`job`, `ring`, `bright`); they are not measured from anything. The look and the map come from the author's "Where next?", a map of harmony for improvising.

The chords themselves are the tool's, and a song has two readings of them. The switch under the map chooses which one the page shows; your own chords stand in either.

- **Eighth by eighth** ([evaluation.md](evaluation.md#chords-and-key)): heard eighth note by eighth note, with sevenths, and the bass notes as evidence for the root; the key is read from the chords, and the page says where the tool hears it change. On the real pop test songs the root and major or minor are right about 86% of the time. Each chord comes with how sure the tool is, and a `?` marks a bar with a chord it is unsure of: such a chord is right about half the time. Where no chord is heard at all, only bass and drums, the bar gets the chord the key has on the note the bass plays there, in slanted letters.
- **One a bar**: a major or minor chord for each bar. Coarser, and right less often on those test songs (73%), but it is steadier from one round of a loop to the next. Over a bass riff with little else playing, which none of the test songs is, the finer reading follows the moving bass line and can write another chord in every round; there the coarser one may show the song's form better. Your ears decide: "Hear the chords" plays either over the record.

The song is cut into parts by its triads, so a seventh heard in one round and not in the next is still the same loop, and a chord the tool is unsure of does not break one.

A song prepared before the tool heard chords eighth by eighth has that reading heard the first time its harmony page is opened: a few seconds, from its separated tracks in the cache, its notes and its beat grid. It is added to what the song has. Nothing is heard again, none of your work changes, and the page goes on showing the reading the song had until you switch. Where the separated tracks are gone the page says so; adding the song again hears it anew. The chord symbols of the score follow the reading the page shows.

## The working directory

```
scores/
  songs.json            the songs and the settings each was heard with
  uploads/              the files added on the page, as they came, and the audio fetched from links
  <song>/machine.json   what the tool heard: never changed
  <song>/notes.json     your version, what you have played through, and the undo history
  <song>/log.jsonl      every fix, offer and play-through, appended
  <song>/reference/     answer keys: somebody's transcription of the song, lined up with the recording
  <song>/harmony.json   what you set on the song's harmony page: its home, the chords you changed by bar, which reading of its chords, its fingering
  tests.json, ratings.json, ratings.jsonl   in a directory of test songs: what they are, and your answers
  <song>/audio/         the record, the isolated bass and the record without the bass, and their slowed versions once used
  <song>/picture.bin, pitch.bin, desk.npz   the pitch picture, the tracked pitch, what the guesses need
  <song>/repeats.npz    which bars sound alike, measured once from the bass track
  <song>/sheet/         the score made from your notes: PDF, MusicXML, MIDI and a picture of each page
  <song>/sheet/v1, v2…  every version of the score as it was made, each with its own files
  <song>/play/          the same score as pictures, a line each, for the play-along page
```

The directory holds your recordings. Keep it out of any repository. Preparing a song again after the tool has changed replaces `machine.json` and leaves your notes alone.

## Limits

- This is the notes piece. Bar lines, written rhythm and tab are still the tool's; steps for them come next.
- Two notes played together are shown and played, but cannot be added or removed on the page yet.
- The tool's doubts (amber) find about a quarter of its wrong notes in a full mix, so they are no substitute for playing through.
