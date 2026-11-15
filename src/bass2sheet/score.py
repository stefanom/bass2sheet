"""MusicXML (bass clef, written an octave up, plus 4-string tab) and PDF engraving through MuseScore's CLI."""

import collections
import os
import shutil
import subprocess
from xml.sax.saxutils import escape

import numpy as np

from .common import FLATS, SHARPS, log

TRIADS = {"": (0, 4, 7), "m": (0, 3, 7)}
STRINGS = [28, 33, 38, 43]  # E1 A1 D2 G2, sounding MIDI
MAX_FRET = 20
TYPES = {
    16: ("whole", 0),
    12: ("half", 1),
    8: ("half", 0),
    6: ("quarter", 1),
    4: ("quarter", 0),
    3: ("eighth", 1),
    2: ("eighth", 0),
    1: ("16th", 0),
}
MSCORE_CANDIDATES = ["mscore", "mscore4", "musescore", "/Applications/MuseScore 4.app/Contents/MacOS/mscore"]
STYLE = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.70"><Style>
<pageWidth>8.5</pageWidth><pageHeight>11</pageHeight><pagePrintableWidth>7.5</pagePrintableWidth>
<pageEvenLeftMargin>0.5</pageEvenLeftMargin><pageOddLeftMargin>0.5</pageOddLeftMargin>
<pageEvenTopMargin>0.5</pageEvenTopMargin><pageEvenBottomMargin>0.6</pageEvenBottomMargin>
<pageOddTopMargin>0.5</pageOddTopMargin><pageOddBottomMargin>0.6</pageOddBottomMargin><pageTwosided>0</pageTwosided>
<createMultiMeasureRests>1</createMultiMeasureRests><minEmptyMeasures>2</minEmptyMeasures>
<showMeasureNumberOne>1</showMeasureNumberOne><measureNumberSystem>1</measureNumberSystem>
</Style></museScore>
"""


def find_mscore(explicit=None):
    for c in ([explicit] if explicit else []) + MSCORE_CANDIDATES:
        p = shutil.which(c) or (c if os.path.isabs(c) and os.path.exists(c) else None)
        if p:
            return p
    return None


def note_pieces(p, e):
    """Split a note over sixteenths [p, e) of a 4/4 bar into readable tied values: nothing hides beat 3, and only
    common syncopations (an eighth or quarter on an off-beat, a dotted quarter on the 'and')."""
    out = []
    while p < e:
        for d in (16, 12, 8, 6, 4, 3, 2, 1):
            if p + d > e:
                continue
            ok = {
                16: p == 0,
                12: p == 0,
                8: p in (0, 8),
                6: p in (0, 2, 8, 10),
                4: p % 4 == 0 or p in (2, 10),
                3: p % 4 in (0, 1),
                2: p % 2 == 0 or p % 4 == 1,
                1: True,
            }[d]
            if ok:
                out.append((p, d))
                p += d
                break
    return out


def rest_pieces(p, e):
    out = []
    while p < e:
        for d in (16, 8, 4, 3, 2, 1):
            if p + d > e:
                continue
            if {16: p == 0, 8: p in (0, 8), 4: p % 4 == 0, 3: p % 4 == 0, 2: p % 2 == 0, 1: True}[d]:
                out.append((p, d))
                p += d
                break
    return out


TAB_STYLES = ("easy", "closed", "low", "position")
# What the fretting hand pays, for the `easy` style: the numbers are rules of thumb, in no unit but each other.
EASY = dict(
    shift=2.0,  # moving the hand at all
    per_fret=0.35,  # and for every fret it moves
    stretch=1.2,  # a fifth fret from the first finger
    wide=0.3,  # the fourth fret near the nut, where frets are far apart
    open=0.5,  # an open string in the middle of fretted notes
    open_low=0.15,  # an open string with the hand at the nut
    cross=(0.0, 0.15, 0.9, 1.8),  # going to the same string, the next one, one further, the furthest
    barre=0.35,  # the same fret on the next string: one finger for two notes
    moved=1.0,  # the same pitch again somewhere else
    fret=0.03,  # every fret up the neck: of two places as easy, the lower
    high=0.15,  # and every fret above the twelfth
    again=0.8,  # what a bar gets for being fingered the way it is elsewhere in the song
)


def chord_frets(string, fret, also):
    """Where the notes played together with one at (string index, fret) go: [(string index, fret)] on the strings
    above it, one string each, as close to its fret as they can be (an open string is always in reach). None
    when they do not fit above it within a four-fret stretch."""
    out, free = [], [s for s in range(len(STRINGS)) if s > string]
    for m in sorted(also):
        best = None
        for s in free:
            f = m - STRINGS[s]
            reach = 0 if f == 0 or fret == 0 else abs(f - fret)
            if 0 <= f <= MAX_FRET and reach <= 4 and (best is None or reach < best[0]):
                best = (reach, s, f)
        if best is None:
            return None
        out.append(best[1:])
        free = [s for s in free if s > best[1]]  # a higher note sits on a higher string
    return out


def _easy(pitches, opts, starts, bars, closed=False):
    """The places that are easiest on the fretting hand. The hand has a position, the fret its first finger is
    at, and reaches four frets from there (a fifth with a stretch); an open string needs no hand. What costs:
    moving the hand, the more the further and the less time there is; crossing strings, the more the further;
    a stretch; an open string among fretted notes; the same pitch somewhere else than a moment ago. Where the
    song plays the same bar again it is fingered the same way, and a pitch that comes twice in a bar is played
    in one place: the notes are placed once, each bar that comes more than once takes the fingering most of its
    rounds got, and the notes are placed again with that in their favour. `closed` keeps off the open strings
    wherever a fretted place will do: every shape can then be moved along the neck, at the price of more moves."""
    free, free_low = (3.0, 3.0) if closed else (EASY["open"], EASY["open_low"])
    positions = np.arange(1, MAX_FRET - 2)
    states = []  # per note: string, fret and hand position of every way to play it
    for places in opts:
        rows = [(s, f, p) for s, f in places for p in (positions if f == 0 else range(max(1, f - 4), f + 1))]
        rows = rows or [(s, f, max(1, min(f, MAX_FRET - 3))) for s, f in places]
        states.append(np.array(rows))

    def own(k, favoured=None):  # what a way to play note k costs wherever the hand comes from
        s, f, p = states[k].T
        reach = f - p
        cost = np.where(
            f == 0,
            np.where(p <= 2, free_low, free),
            EASY["fret"] * f
            + EASY["high"] * np.maximum(0, f - 12)
            + EASY["stretch"] * (reach >= 4)
            + EASY["wide"] * ((reach == 3) & (p <= 3)),
        )
        if favoured is not None:
            cost = cost - EASY["again"] * ((s == favoured[0]) & (f == favoured[1]))
        return cost

    def run(favoured):
        total = own(0, favoured.get(0)) + 0.02 * states[0][:, 2]
        back = []
        for k in range(1, len(states)):
            s0, f0, p0 = (states[k - 1][:, j][:, None] for j in range(3))
            s, f, p = (states[k][:, j][None, :] for j in range(3))
            # how little time there is to move the hand, from the sixteenths since the last note; after an open
            # string, since the note before it: the hand is free while an open string sounds
            gaps = (2.0, 4.0) if starts is None else (starts[k] - starts[k - 1], starts[k] - starts[max(0, k - 2)])
            short, long = (min(3.0, max(0.5, 2.0 / max(float(gap), 0.5))) for gap in gaps)
            hurry = np.where(f0 == 0, long, short)
            moved = np.abs(p - p0)
            step = np.where(moved > 0, (EASY["shift"] + EASY["per_fret"] * moved) * hurry, 0.0)
            step = np.where((f == 0) & (moved > 0), np.inf, step)  # an open string leaves the hand where it is
            step = step + np.take(EASY["cross"], np.abs(s - s0)) * (0.6 + 0.4 * hurry)
            step = step + EASY["barre"] * ((f == f0) & (f > 0) & (np.abs(s - s0) == 1))
            if pitches[k] == pitches[k - 1]:
                step = step + EASY["moved"] * ((s != s0) | (f != f0))
            through = total[:, None] + step
            back.append(np.argmin(through, axis=0))
            total = through[back[-1], np.arange(through.shape[1])] + own(k, favoured.get(k))
        j = int(np.argmin(total))
        path = [j]
        for bk in reversed(back):
            j = int(bk[j])
            path.append(j)
        path.reverse()
        return [(int(states[k][j, 0]), int(states[k][j, 1])) for k, j in enumerate(path)]

    first = run({})
    if bars is None:
        return first
    rounds = collections.defaultdict(list)  # the notes of each bar, by what the bar holds
    for k, bar in enumerate(bars):
        rounds[bar].append(k)
    same = collections.defaultdict(list)
    for notes in rounds.values():
        same[tuple(pitches[k] for k in notes)].append(notes)
    favoured = {}
    for held in same.values():
        if len(held) > 1:
            most = collections.Counter(tuple(first[k] for k in notes) for notes in held).most_common(1)[0][0]
            for notes in held:
                favoured.update(zip(notes, most))
    for notes in rounds.values():  # and inside a bar a pitch that comes twice is played in one place
        places = collections.defaultdict(collections.Counter)
        for k in notes:
            places[pitches[k]][favoured.get(k, first[k])] += 1
        for k in notes:
            if len(places[pitches[k]]) > 1:
                favoured[k] = places[pitches[k]].most_common(1)[0][0]
    return run(favoured) if favoured else first


def effort(places):
    """What a fingering asks of the fretting hand, counted, for comparing fingerings of the same notes: `moves`,
    the times the hand has to move (it reaches four frets from its first finger, and an open string needs no
    hand: the fewest moves that play the notes where they are put); `frets`, how far it moves in all; `skips`,
    the string crossings that pass over a string; `open`, the notes on open strings; `notes`."""
    moves = far = skips = 0
    lo = hi = None  # where the first finger can be for the fretted notes since the last move
    for k, (s, f) in enumerate(places):
        if k and abs(s - places[k - 1][0]) >= 2:
            skips += 1
        if f == 0:
            continue
        if lo is None:
            lo, hi = f - 3, f
        elif max(lo, f - 3) <= min(hi, f):
            lo, hi = max(lo, f - 3), min(hi, f)
        else:
            moves, far = moves + 1, far + (f - 3 - hi if f - 3 > hi else lo - f)
            lo, hi = f - 3, f
    return dict(notes=len(places), moves=moves, frets=far, skips=skips, open=sum(f == 0 for _, f in places))


def hand(places):
    """Where the fretting hand is for each note of a fingering: the fret of its first finger, which reaches four
    frets from there. The hand stays where it is for as long as the notes allow (an open string needs no hand),
    with the first finger on the lowest fret of each stretch. None for the notes before the first fretted one."""
    out, run, lo, hi = [None] * len(places), [], None, None

    def settle():
        for k in run:
            out[k] = max(1, lo)

    for k, (_, f) in enumerate(places):
        if f and lo is not None and (max(lo, f) - min(lo, f) > 3 or max(hi, f) - min(lo, f) > 3):
            settle()
            run, lo, hi = [], None, None
        if f:
            lo, hi = (f, f) if lo is None else (min(lo, f), max(hi, f))
        run.append(k)
    if lo is not None:
        settle()
    return out


def frets(pitches, style="low", also=None, starts=None, bars=None):
    """(string index, fret) per note.

    `easy` is for playing: the hand moves as little as it can (see `_easy`). It is helped by `starts`, when each
    note starts in sixteenths (a move is easier with time for it), and by `bars`, the bar each note is in (a bar
    that comes again is fingered the same way). `closed` is the same without open strings, wherever a fretted
    place will do.

    The two others are a Viterbi over playable positions, each state remembering the last fret the hand was at
    (open strings leave it in place); stretches beyond 3 frets cost more. `low` keeps to the first frets and
    open strings, where the notes are easiest to find. `position` keeps the hand in one place: it does not
    favour low frets, and it frets a note rather than play it on an open string unless that would mean a large
    shift.

    `also[k]` lists the pitches played together with note k: that note then goes where they fit on the strings
    above it."""
    if style not in TAB_STYLES:
        raise ValueError(f"unknown tab style {style!r}; expected one of {', '.join(TAB_STYLES)}")
    if not pitches:
        return []
    low = style == "low"
    opts = [
        [(s, m - o) for s, o in enumerate(STRINGS) if 0 <= m - o <= MAX_FRET] or [(0, max(0, m - 28))] for m in pitches
    ]
    for k, extra in enumerate(also or []):
        if extra:
            opts[k] = [o for o in opts[k] if chord_frets(*o, extra) is not None] or opts[k]
    if style in ("easy", "closed"):
        return _easy(list(pitches), opts, starts, bars, closed=style == "closed")

    def place(f):  # what a fret costs wherever the hand is
        return 0.02 * f + 0.3 * max(0, f - 4) if low else 0.005 * f + (1.2 if f == 0 else 0.0)

    def step(hand, s0, s, f):
        reach = 0.0 if f == 0 else 0.1 * abs(f - hand) + 1.0 * max(0, abs(f - hand) - 3)
        return reach + 0.1 * abs(s - s0) + place(f), (hand if f == 0 else f)

    states = [(place(f), (f or 2), s) for s, f in opts[0]]
    back = []
    for k in range(1, len(opts)):
        new, bk = [], []
        for s, f in opts[k]:
            best = None
            for j, (c0, h0, s0) in enumerate(states):
                c, h = step(h0, s0, s, f)
                if best is None or c0 + c < best[0]:
                    best = (c0 + c, h, s, j)
            new.append(best[:3])
            bk.append(best[3])
        states, back = new, back + [bk]
    k = min(range(len(states)), key=lambda i: states[i][0])
    path = [k]
    for bk in reversed(back):
        k = bk[k]
        path.append(k)
    path.reverse()
    return [opts[i][j] for i, j in enumerate(path)]


def pitch_xml(midi, name):
    """Written pitch (an octave above sounding); `name` is the spelled name, e.g. 'F#' or 'Bb'."""
    step, acc = name[0], name[1:]
    alter = {"": 0, "#": 1, "b": -1}[acc]
    octave = (midi + 12 - alter) // 12 - 1
    return (
        f"<pitch><step>{step}</step>"
        + (f"<alter>{alter}</alter>" if alter else "")
        + f"<octave>{octave}</octave></pitch>"
    )


def written_events(notes, tab_style="low"):
    """What the score shows for quantized notes, in order: each note with `grace` (it shares its sixteenth with the
    next note), its written span `g0`..`g1` in sixteenths (held to the next note across a gap of one sixteenth, cut
    where the next note starts) and its place on the neck (`string` 1 = G ... 4 = E, `fret`). A note with others
    played together (`also`) has their places in `also_tab`."""
    notes = sorted(notes, key=lambda n: (n["grid_start"], n["start"]))
    events = []
    for k, nt in enumerate(notes):  # notes sharing a 16th: all but the last become grace notes
        nxt = notes[k + 1]["grid_start"] if k + 1 < len(notes) else None
        if nxt is not None and nxt == nt["grid_start"]:
            events.append(dict(nt, grace=True, g0=nt["grid_start"], g1=nt["grid_start"]))
            continue
        end = nt["grid_start"] + nt["grid_len"]
        if nxt is not None and (nxt < end or nxt - end <= 1):
            end = nxt
        events.append(dict(nt, grace=False, g0=nt["grid_start"], g1=max(end, nt["grid_start"] + 1)))
    together = [e.get("also") or [] for e in events]
    when, bars = [e["g0"] for e in events], [e["g0"] // 16 for e in events]
    for e, (s, f) in zip(events, frets([e["pitch"] for e in events], tab_style, together, starts=when, bars=bars)):
        e["string"], e["fret"] = 4 - s, f  # MusicXML strings count from the highest (G = 1)
        if e.get("also"):  # the notes played with it: [pitch, string, fret], or without a place when none fits
            where = chord_frets(s, f, e["also"]) or [None] * len(e["also"])
            e["also_tab"] = [[m, *((4 - w[0], w[1]) if w else (None, None))] for m, w in zip(sorted(e["also"]), where)]
    return events


def musicxml(
    notes,
    chords,
    key,
    tempo,
    first_downbeat,
    title,
    artist,
    check_bars,
    draft,
    phrase_start=None,
    bars_per_line=4,
    tab_style="low",
    bars=None,
    bar_times=None,
    subtitle=None,
):
    """MusicXML text. `key` is (name, root, is_major); `check_bars` maps bar -> reasons (marked "check").
    `subtitle` replaces the line under the title that says the bass is a machine transcription.
    Lines hold `bars_per_line` bars counted from `phrase_start` (default: the first bar with notes).

    `bars=(first, last)` writes only those bars as one line with no title: a snippet of the same score, with the
    same fingering. `bar_times` maps a bar to where it starts in the recording, for the time printed at each line
    start; without it the times assume a steady tempo from `first_downbeat`."""
    key_name, key_root, key_major = key
    chord_of = {c["bar"]: c for c in chords}
    rel_major = key_root if key_major else (key_root + 3) % 12
    fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}[rel_major]
    key_names = SHARPS if fifths >= 0 else FLATS

    def spell(nt, pitch=None):
        pitch = nt["pitch"] if pitch is None else pitch
        c = chord_of.get(nt["bar"])
        if c and pitch % 12 in {(c["root"] + i) % 12 for i in TRIADS[c["quality"]]}:
            return c["names"][pitch % 12]
        return key_names[pitch % 12]

    events = written_events(notes, tab_style)
    first_bar = min(1, min(e["grid_start"] // 16 + 1 for e in events))
    if phrase_start is None:  # lines of `bars_per_line` bars start where the bass comes in
        phrase_start = max(1, min(e["grid_start"] // 16 + 1 for e in events))
    last_bar = max((e["g1"] - 1) // 16 + 1 for e in events if not e["grace"])
    snippet = bars is not None
    if snippet:
        first_bar, last_bar = bars
    by_bar = {}
    for e in events:
        if e["grace"]:
            by_bar.setdefault(e["grid_start"] // 16 + 1, []).append((e["grid_start"] % 16, 0, e, False, False))
            continue
        g = e["g0"]
        while g < e["g1"]:
            bar, p = g // 16 + 1, g % 16
            seg_end = min(e["g1"], bar * 16)
            by_bar.setdefault(bar, []).append((p, seg_end - (bar - 1) * 16, e, g > e["g0"], seg_end < e["g1"]))
            g = seg_end

    bar_s = 4 * 60 / tempo
    mmss = lambda s: f"{int(max(s, 0) // 60)}:{int(max(s, 0) % 60):02d}"
    bar_time = lambda bar: (
        bar_times[bar] if bar_times is not None and bar in bar_times else first_downbeat + (bar - 1) * bar_s
    )
    doubtful = lambda e: bool(e["flag"] or e.get("sounds_like"))  # shown in parentheses
    flagged = any(doubtful(e) for e in events)

    def one_note(staff, e, d, tie_start, tie_stop, grace, pitch, string, fret, chord):
        x = "<note>" + ('<grace slash="yes"/>' if grace else "") + ("<chord/>" if chord else "")
        x += pitch_xml(pitch, spell(e, pitch))
        if not grace:
            x += f"<duration>{d}</duration>"
        x += ('<tie type="stop"/>' if tie_stop else "") + ('<tie type="start"/>' if tie_start else "")
        x += "<voice>1</voice>" if staff == 1 else "<voice>5</voice>"
        typ, dots = TYPES[d] if not grace else ("16th", 0)
        x += f"<type>{typ}</type>" + "<dot/>" * dots + ("<stem>none</stem>" if staff == 2 else "")
        if doubtful(e) and staff == 1 and not tie_stop:
            x += '<notehead parentheses="yes">normal</notehead>'
        x += f"<staff>{staff}</staff>"
        nots = (['<tied type="stop"/>'] if tie_stop else []) + (['<tied type="start"/>'] if tie_start else [])
        if staff == 2:
            nots.append(f"<technical><string>{string}</string><fret>{fret}</fret></technical>")
        return x + ("<notations>" + "".join(nots) + "</notations>" if nots else "") + "</note>"

    def note_xml(staff, e, d, tie_start, tie_stop, grace=False):
        """A note of the line and, as chord notes, those played together with it. In the tab a chord note
        without a place on the neck (none in reach) is left out; the staff above still shows it."""
        x = one_note(staff, e, d, tie_start, tie_stop, grace, e["pitch"], e["string"], e["fret"], False)
        for pitch, string, fret in [] if grace else e.get("also_tab", []):
            if staff == 1 or string is not None:
                x += one_note(staff, e, d, tie_start, tie_stop, False, pitch, string, fret, True)
        return x

    def rest_xml(staff, d, whole_bar=False):
        typ, dots = TYPES[d]
        r = '<rest measure="yes"/>' if whole_bar else "<rest/>"
        t = "" if whole_bar else f"<type>{typ}</type>" + "<dot/>" * dots
        hide = "" if staff == 1 else ' print-object="no"'  # the staff above carries the rhythm
        voice = 1 if staff == 1 else 5
        return f"<note{hide}>{r}<duration>{d}</duration><voice>{voice}</voice>{t}<staff>{staff}</staff></note>"

    words = lambda text, size=7, style="italic", color=None: (
        '<direction placement="above"><direction-type>'
        f'<words font-size="{size}" font-style="{style}"'
        + (f' color="{color}"' if color else "")
        + f">{escape(text)}</words>"
        "</direction-type><staff>1</staff></direction>"
    )
    M, prev_chord = [], None
    for bar in range(first_bar, last_bar + 1):
        x = f'<measure number="{bar}">'
        line_start = bar == first_bar or (
            not snippet and bar >= phrase_start and (bar - phrase_start) % bars_per_line == 0
        )
        if line_start and bar != first_bar:
            x += '<print new-system="yes"/>'
        if bar == first_bar:
            x += (
                "<attributes><divisions>4</divisions>"
                f"<key><fifths>{fifths}</fifths><mode>{'major' if key_major else 'minor'}</mode></key>"
                "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
                '<clef number="1"><sign>F</sign><line>4</line></clef>'
                '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                '<staff-details number="2"><staff-lines>4</staff-lines>'
                + "".join(
                    f'<staff-tuning line="{i + 1}"><tuning-step>{SHARPS[m % 12][0]}</tuning-step>'
                    f"<tuning-octave>{m // 12}</tuning-octave></staff-tuning>"
                    for i, m in enumerate(STRINGS)
                )
                + "</staff-details>"
                "<transpose><diatonic>0</diatonic><chromatic>0</chromatic><octave-change>-1</octave-change></transpose>"
                "</attributes>"
            )
            x += (
                '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                f"<per-minute>{tempo:.0f}</per-minute></metronome></direction-type><staff>1</staff>"
                f'<sound tempo="{tempo:.0f}"/></direction>'
            )
        if line_start:
            x += words(mmss(bar_time(bar)))
        if bar in check_bars:
            x += words("check", size=8, style="normal", color="#C62828")
        c = chord_of.get(bar)
        if c and bar in by_bar and c["chord"] != prev_chord:
            root = c["names"][c["root"]]
            alter = {"": 0, "#": 1, "b": -1}[root[1:]]
            x += (
                f"<harmony><root><root-step>{root[0]}</root-step>"
                + (f"<root-alter>{alter}</root-alter>" if alter else "")
                + f"</root><kind>{'major' if c['quality'] == '' else 'minor'}</kind><staff>1</staff></harmony>"
            )
            prev_chord = c["chord"]
        elif bar not in by_bar:
            prev_chord = None
        for staff in (1, 2):
            if staff == 2:
                x += "<backup><duration>16</duration></backup>"
            items = sorted(by_bar.get(bar, []), key=lambda it: (it[0], it[1] > 0))
            if not any(it[1] for it in items):
                x += rest_xml(staff, 16, whole_bar=True)
                continue
            cur = 0
            for p, end, e, tie_stop, tie_start in items:
                if snippet:  # a snippet shows no ties to notes outside it
                    tie_stop, tie_start = tie_stop and bar > first_bar, tie_start and bar < last_bar
                if end == 0:  # grace note
                    if p > cur:
                        x += "".join(rest_xml(staff, d) for _, d in rest_pieces(cur, p))
                        cur = p
                    x += note_xml(staff, e, 1, False, False, grace=True)
                    continue
                if p < cur:
                    continue
                x += "".join(rest_xml(staff, d) for _, d in rest_pieces(cur, p))
                pieces = note_pieces(p, end)
                for k, (_, d) in enumerate(pieces):
                    x += note_xml(staff, e, d, tie_start or k < len(pieces) - 1, tie_stop or k > 0)
                cur = end
            x += "".join(rest_xml(staff, d) for _, d in rest_pieces(cur, 16))
        if bar == last_bar:
            x += '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
        M.append(x + "</measure>")

    sub = subtitle or (
        ("DRAFT: failed quality gates, see report. " if draft else "") + "Bass: machine transcription, check it by ear"
    )
    foot = " · ".join(
        filter(
            None,
            [
                "Chord symbols approximate" if chords else "",
                "(notes) uncertain" if flagged else "",
                "red 'check': verify by ear (see report)" if check_bars else "",
                "times: where each line starts in the recording",
            ],
        )
    )
    head = (
        ""
        if snippet
        else f"<movement-title>{escape(title)}</movement-title>"
        f'<identification><creator type="composer">{escape(artist)}</creator>'
        f'<creator type="lyricist">{escape(sub)}</creator><rights>{escape(foot)}</rights></identification>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n<score-partwise version="4.0">'
        + head
        + '<part-list><score-part id="P1"><part-name>Bass</part-name>'
        '<score-instrument id="P1-I1"><instrument-name>Electric Bass</instrument-name></score-instrument>'
        '<midi-instrument id="P1-I1"><midi-program>34</midi-program></midi-instrument></score-part></part-list>'
        '<part id="P1">' + "".join(M) + "</part></score-partwise>\n"
    )


def engrave_pages(xml_path, pdf_path, mscore, timeout=300):
    """MusicXML -> PDF and a picture of each page, in one run of MuseScore: the pictures are SVG files beside
    the PDF, named after it (bass.pdf: bass-1.svg, bass-2.svg, ...). Pictures left from an earlier, longer score
    are removed. Returns (an error string or None, the pictures in page order)."""
    import glob
    import json

    folder, stem = os.path.dirname(pdf_path), os.path.splitext(pdf_path)[0]
    numbered = lambda: sorted(
        (p for p in glob.glob(glob.escape(stem) + "-*.svg") if p[len(stem) + 1 : -4].isdigit()),
        key=lambda p: int(p[len(stem) + 1 : -4]),
    )
    for old in numbered():
        os.remove(old)
    style, job = os.path.join(folder, ".bass2sheet.mss"), os.path.join(folder, ".bass2sheet.json")
    with open(style, "w") as f:
        f.write(STYLE)
    with open(job, "w") as f:
        json.dump([{"in": xml_path, "out": [pdf_path, stem + ".svg"]}], f)
    try:
        r = subprocess.run([mscore, "-S", style, "-j", job], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"MuseScore timed out after {timeout} s", []
    finally:
        os.remove(style)
        os.remove(job)
    if r.returncode != 0 or not os.path.exists(pdf_path):
        tail = "\n".join(line for line in (r.stderr or "").splitlines() if "sandbox_extension" not in line)[-800:]
        return f"MuseScore exited {r.returncode}: {tail}", []
    log.info("engraved %s", pdf_path)
    return None, numbered()


def engrave(xml_path, pdf_path, mscore, timeout=300):
    """MusicXML -> PDF with MuseScore; returns an error string or None."""
    style = os.path.join(os.path.dirname(xml_path), ".bass2sheet.mss")
    with open(style, "w") as f:
        f.write(STYLE)
    try:
        r = subprocess.run(
            [mscore, "-S", style, "-o", pdf_path, xml_path], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"MuseScore timed out after {timeout} s"
    finally:
        os.remove(style)
    if r.returncode != 0 or not os.path.exists(pdf_path):
        tail = "\n".join(line for line in (r.stderr or "").splitlines() if "sandbox_extension" not in line)[-800:]
        return f"MuseScore exited {r.returncode}: {tail}"
    log.info("engraved %s", pdf_path)
    return None
