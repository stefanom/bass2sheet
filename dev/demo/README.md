# dev/demo: the script that makes the demo film

What the film shows, how to run this and what it needs: [docs/demo.md](../../docs/demo.md).

## How the film is made

- **`film.py`** opens a page in Chrome without a window and keeps every frame of Chrome's own screencast with
  the moment it was drawn. The pointer is moved along a path, as a hand moves it; the pages get ordinary mouse
  events. `hooks.js`, put into each page before its own scripts, draws the pointer (a recording without a window
  has none) and writes down what the page plays.
- **`scenes.py`** is what the hand does and which line is said when. **`story.py`** is the words.
- **`sound.py`** makes the sound again from the files the pages played and the log of how they played them: for
  the song page, every buffer the page started, with its offset, speed and the gains it went through, on the
  page's own audio clock; for the play-along page, the audio element's file and place. Nothing is recorded from
  the browser (it is started without a sound output, so the machine stays silent) and no timing is guessed: the
  browser says which moment of its audio clock is heard when. `check_sound.py` proves the rest: it taps what
  Chrome's audio graph really puts out and lays the remade sound on it, sample by sample (they differ by 40 dB
  under the sound). One thing to know: a page that has been silent for half a minute lets its audio clock fall
  behind until it plays again; the first sound after such a silence is placed by the clock as it runs from
  then on, which can be a tenth of a second off at that sound's very start.
- **`edit.py`** turns each scene's time into the film's (a wait sped up, with a sign that says so; the blank
  moment between two pages cut), crops to where a scene looks closer, lays the captions on, lowers what the pages
  play while the narrator speaks, and encodes. Under the title and the last card, where no page plays, the
  record itself is the music, quietly.
- **`stage/`** holds the film's own pages: the terminal (which shows what the command really printed, as it
  printed it), the cards, the sheet on its desk, the phone around the play-along page, the captions.

A list of choices (the speed, the score's versions) is drawn by the system and not by the page, so a recording
without a window cannot show it open: the pointer goes there, a click is shown, and the choice is made.
