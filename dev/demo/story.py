"""The film's words. Each line is said by the narrator and shown at the bottom of the picture while it is said.

`text` is the caption. `voice`, where given, is the same line spelt for a synthetic voice, which reads "bass" as
the fish and stumbles over "bass2sheet"; a person recording the lines reads `text`.

The lines say only what the picture shows at that moment. Where a line names a number (seven bars, two more
bars), the scene checks it against what the tool really did with the song, and stops the film if it is no longer
true: see scenes.py and the song's storyboard in songs/.
"""

LINES = [
    # 1. the problem, and the result first
    (
        "o1",
        "You want to play a song's bass line, and nobody has written it down.",
        "You want to play a song's base line, and nobody has written it down.",
    ),
    (
        "o2",
        "bass2sheet listens to the recording and writes it out: notes and tab.",
        "Base to sheet listens to the recording, and writes it out: notes and tab.",
    ),
    # 2. one command
    (
        "c1",
        "One command takes a song and writes the sheet: a PDF, MusicXML and MIDI.",
        "One command takes a song and writes the sheet: a P D F, Music X M L, and middy.",
    ),
    ("c2", "The machine makes mistakes. So each run checks itself, and says how far to trust the result.", None),
    ("c3", "For this song: usable, with warnings. Seven bars are marked, to check by ear first.", None),
    # 3. the copilot's first page
    ("l1", "That sheet is a first draft. The copilot is where you make it right.", None),
    ("l2a", "Drop the song on its page. It shows every step as it runs.", None),
    (
        "l2b",
        "The bass is isolated, its pitch is followed, the notes are found and double-checked.",
        "The base is isolated, its pitch is followed, the notes are found and double-checked.",
    ),
    # 4. the song page
    (
        "s1",
        "This is the song: the isolated bass as a picture, and on it the notes the tool heard.",
        "This is the song: the isolated base as a picture, and on it the notes the tool heard.",
    ),
    (
        "s2",
        "Play it. You hear the record, and the notes in the bass's own sound.",
        "Play it. You hear the record, and the notes in the base's own sound.",
    ),
    ("s3", "This short note is not played. Delete it.", None),
    ("s4", "The amber note is one the tool doubts. It belongs a row higher: drag it there.", None),
    (
        "s5",
        "Here the bass comes in on the beat. Drag the note's start to the bar line.",
        "Here the base comes in on the beat. Drag the note's start to the bar line.",
    ),
    ("s6", "For a wrong pitch the tool offers its guesses. Take the first.", None),
    ("s7", "This bar repeats. Two more bars have the same mistake. Nothing changes until you say so.", None),
    ("s8", "Hear a bar with the change, then change it there.", None),
    ("s9", "The other bar comes much later in the song. Leave it for now.", None),
    ("s10", "What you played and left alone now counts as checked.", None),
    ("s11", "Further on, the tool wrote two notes together. Slow the song down to hear them.", None),
    ("s12", "They are played one after the other. Delete the upper one, then double-click where it sounds.", None),
    # 5. the sheet
    ("h1", "Make the sheet. It is written from your notes, and it says how much of it you have checked.", None),
    ("h2", "Fix one more note and make it again. Every version is kept.", None),
    # 6. playing along
    ("p1", "To practise, open the play-along page on a phone held sideways.", None),
    (
        "p2",
        "It plays the record without the bass. You see the line you are playing, and the next one.",
        "It plays the record without the base. You see the line you are playing, and the next one.",
    ),
    # 7. how it is checked
    ("t1a", "How good is it? The tool is measured on test songs that come with answer keys.", None),
    ("t1b", "You say which are like your music, and the scores are counted on those.", None),
    (
        "t2",
        "A MIDI file or a tab of your own song can serve as an answer key too.",
        "A middy file, or a tab of your own song, can serve as an answer key too.",
    ),
    # 8. what it is, and how to get it
    ("e1", "The tool makes mistakes. You stay in charge, and every fix is kept as data.", None),
    (
        "e2",
        "It is open source, under the MIT licence. Run the bootstrap script, then start the copilot.",
        "It is open source, under the M I T licence. Run the bootstrap script, then start the copilot.",
    ),
]

PACE = 15.0  # characters a second, for a line that has no recording yet


def lines():
    """{id: {text, voice, seconds}}: seconds is a guess from the length until a recording says otherwise."""
    return {i: dict(text=text, voice=voice or text, seconds=round(len(text) / PACE, 2)) for i, text, voice in LINES}
