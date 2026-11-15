# The demo film

A film of just under four minutes that shows what bass2sheet does, end to end, on one song. It is made by
a script, from the real pages: `dev/demo/make.py` runs the command, starts the copilot, drives the pages in a
browser as a hand would, and puts the film together. When a page changes, run it again and the film shows the
new page.

## What it shows

1. **The problem, and the result first.** The sheet the film ends up with: staff and tab.
2. **One command.** `bass2sheet SONG -o sheet` runs while it is filmed (the wait is shown sped up, and the
   picture says by how much); what it wrote; the report with the verdict and the quality gates; the PDF.
3. **The copilot's first page.** The song is dropped on it and prepared, step by step (sped up, as the picture
   says). The song is prepared with a cache of its own, so every step runs as on a first time.
4. **The song's page.** Play, stop at what is wrong, fix it. Five mistakes the tool really made on this song, each
   fixed another way: a note that is not played is deleted; a wrong pitch is dragged to its row; a late start is
   dragged to the bar line; a wrong pitch is replaced by the tool's first guess, and the same fix, offered for
   the two bars that repeat the mistake, is taken in one and left in the other; of two notes written together,
   one is deleted and put where it sounds, after the song has been slowed down. "Checked up to here".
5. **The sheet.** "Make the sheet", the score in the page; the bar that was left is fixed, the sheet is made
   again, and both versions are there.
6. **Playing along.** The play-along page at the size of a phone held sideways.
7. **How the tool is checked.** The test songs and their scores; one line on answer keys.
8. **What the fixes are kept for, and how to get the tool.** `bass2sheet-copilot report` as it prints, then a
   card with the two commands, the licence and the song's credit.

## Making it

```
python dev/demo/make.py --song "Kevin MacLeod - Hustle.mp3" --narration NARRATION_DIR \
    --tests TESTS_DIR --out bass2sheet-demo.mp4
```

It needs, beside bass2sheet itself with MuseScore 4:

- a Python with `playwright`, `Pillow`, `numpy` and `soundfile` (`pip install playwright pillow numpy soundfile`;
  no browser has to be downloaded for Playwright),
- Google Chrome (the film is taken from Chrome without a window),
- `ffmpeg` and `ffprobe` on the `PATH`.

`--python` names the Python that runs bass2sheet when it is not the checkout's `.venv`. The script writes, beside
`OUT.mp4`: `OUT-small.mp4` (the same film under 10 MB, for a README), `OUT-no-voice.mp4` (the picture, the
captions and the music, without the narrator), `OUT.srt` (the captions) and `OUT.narration.json` (each line's
id, text, start and length in the film).

It takes about fifteen minutes. Two of its steps use the machine as the tool itself does (the command, and the
song prepared on the page: a minute or two each on an Apple Silicon Mac, on its GPU); the rest is a browser
being filmed at the clock's pace, and the encoding. No sound comes out of the machine while it films: the
browser is started without a sound output, and the film's sound is made afterwards from what the pages say they
played.

- `--only song,sheet` films those scenes again and keeps the others as they were filmed; `--edit` films nothing
  and puts the film together again (after a change to the words, the sound levels or the editing);
  `--film-only` films and leaves the putting together for later.
- The scenes that change the song (`landing`, `song`, `sheet`, `play`) each start from the working directory as
  the scene before left it; a copy is kept after each in the work folder (`--work`, by default `dev/demo/work`,
  which git ignores: a filming's frames and the tool's own files for the song take about three gigabytes).
- `--warm` is for trying a scene out: the command and the preparation use what the tool has kept of the song and
  take seconds. The film then shows a second run, not a first, so do not publish that.
- `--gentle` runs the tool and the encoding at low priority on two threads.
- Ports 8782 and 8783 are used while filming (`--port`).

The film is 1920 by 1080. The pages are filmed 1440 by 709 of their own pixels and shown a third larger; the
band under the picture holds the captions, so a caption never covers anything on a page.

So that the command is filmed as on a first time, the script deletes what an earlier filming left of **this
song** in the tool's cache (`~/.cache/bass2sheet`): the separated tracks and pitch tracks of the demo song,
which the command works out again. It keeps the list of those files beside them (`demo-….json`).

### The narration

The words are in `dev/demo/story.py`: one entry per line, with an id. `--narration DIR` is a folder with one
sound file per line, named by its id (`o1.wav`, `c1.mp3`, …; anything ffmpeg reads). Record them yourself, or
have any voice read them: a line's `voice` text is the line spelt for a synthetic voice (which reads "bass" as
the fish). Each scene is paced by the length of its lines, so film again after changing the recordings; or use
`--edit`, which keeps the pictures and holds one still wherever a line has become longer than the scene allowed.

A line with no recording is spoken by the Mac's `say` where there is one. That is a stand-in to work with, not
the film's voice.

Every line is also shown as a caption. A line says only what the picture shows; where it names a number, the
scene checks the number against what the tool did and stops if it is no longer true.

### The test songs

`--tests DIR` is a copilot working directory of test songs, for part 7. The film's was made with

```
bass2sheet-copilot testset ~/data/babyslakh_16k --name BabySlakh --scores bass2sheet=runs/babyslakh --dir TESTS_DIR
```

(BabySlakh: [zenodo.org/records/4603870](https://zenodo.org/records/4603870), CC BY 4.0; the run directory is a
`bass2sheet-eval` run, see [evaluation.md](evaluation.md)). The film shows the page and two answers being given,
and takes the answers back afterwards; it plays none of those songs. Without `--tests` the part is left out.

## The song

"Hustle" by Kevin MacLeod: a twelve-bar blues with a rolling bass riff, two minutes long. The tool gets nearly
all of it right (verdict WARN: usable, with warnings), and leaves a few mistakes of several kinds, which is what
the film needs.

The file is `https://incompetech.com/music/royalty-free/mp3-royaltyfree/Hustle.mp3`
(SHA-256 `8acbe21e78162eb14e75e914554d774585e6e7e9f29cc641047bccbd7b42dfd1`), named
`Kevin MacLeod - Hustle.mp3` for the film. It is not in this repository. Its licence is Creative Commons
Attribution 4.0; the credit, as the licence page of incompetech.com
([incompetech.com/music/royalty-free/licenses](https://incompetech.com/music/royalty-free/licenses/)) words it,
is on the film's last card and has to stay with the film:

```
"Hustle"
Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0
http://creativecommons.org/licenses/by/4.0/
```

### The storyboard, and why each fix is a real one

What is particular to a song is in `dev/demo/songs/NAME.json`: where the tool's mistakes are, and what is right
there. Nothing in the film is staged: each fix is a mistake the tool made on this recording, and what replaces
it is what the isolated bass holds at that place. For "Hustle" (times in seconds of the song):

| Where | The tool wrote | What sounds there | In the film |
|---|---|---|---|
| 0.20 | a second G1, 70 ms long | nothing: the first G has died away (the level falls from −11 dB to −30 dB) and the B has not begun | deleted |
| 1.53 | C#2 (and doubted it) | D2: the partials of the tone are D2's, as at the same place in every other bar of the riff | dragged up a row |
| 4.12 | G1 starting a sixteenth late, the B1 before it held over the bar line | G1 from the bar line on | its start dragged to the bar line |
| 5.52 | E2 (and doubted it) | D2 | the tool's first guess taken |
| 15.52, 63.52 | E2, in the two bars the tool then offers (bars 8 and 32) | D2 in both | changed in bar 8; left in bar 32, and fixed there before the second sheet |
| 50.29 | B1 and D2 together, for half a second | B1 until 50.5, then D2: one after the other, as in the bar before | D2 deleted from the pair, then added at 50.53 |

"What sounds there" was established without ears, from the isolated bass track: the energy at the partials of
each candidate pitch (a harmonic comb on a long, zero-padded spectrum of the note's stretch), the level from
moment to moment, and the pitch the tool's own tracker followed. With these fixes the first four bars, which the
film marks as checked, are the riff as it is played: G B D E F E D B.

On this song every offer the tool made for a repeating bar was right. The film therefore does not show an offer
being refused because it is wrong; it shows one being left for later, and says so.

If the tool changes and no longer makes one of these mistakes, the scene stops and names it. Then the storyboard
has to be looked at again: find what the tool gets wrong now, check it against the bass track, and write it down.
A film of another song needs a storyboard of its own, and its own words where they name a number.

How the script works, file by file, is in [dev/demo/README.md](../dev/demo/README.md).
