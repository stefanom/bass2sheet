import xml.dom.minidom

import pytest

from bass2sheet import score
from bass2sheet.rhythm import TRIADS


@pytest.mark.parametrize("p,e", [(p, e) for p in range(16) for e in range(p + 1, 17)])
def test_note_and_rest_pieces_tile_the_span(p, e):
    for pieces in (score.note_pieces(p, e), score.rest_pieces(p, e)):
        assert pieces[0][0] == p
        assert sum(d for _, d in pieces) == e - p
        pos = p
        for start, d in pieces:
            assert start == pos and d in score.TYPES
            pos += d


def test_note_pieces_never_hide_beat_three():
    for p in range(8):
        for e in range(9, 17):
            for start, d in score.note_pieces(p, e):
                assert not (start < 8 < start + d) or start == 0, (p, e, start, d)


def test_rest_pieces_stay_on_the_beat_structure():
    assert score.rest_pieces(0, 16) == [(0, 16)]
    assert score.rest_pieces(2, 8) == [(2, 2), (4, 4)]
    assert score.rest_pieces(1, 4) == [(1, 1), (2, 2)]


def test_frets_are_playable_and_sound_the_pitch():
    pitches = [28, 33, 40, 45, 47, 52, 35, 38, 43, 55, 60]
    for (string, fret), pitch in zip(score.frets(pitches), pitches):
        assert 0 <= fret <= score.MAX_FRET
        assert score.STRINGS[string] + fret == pitch
    assert score.frets([]) == []


def test_frets_prefer_low_positions_and_open_strings():
    assert score.frets([28, 33, 38, 43]) == [(0, 0), (1, 0), (2, 0), (3, 0)]  # the open strings
    assert all(f <= 5 for _, f in score.frets([40, 42, 43, 45, 47, 45, 43, 42]))


@pytest.mark.parametrize(
    "midi,name,expected",
    [
        (28, "E", ("E", None, 2)),
        (30, "F#", ("F", 1, 2)),
        (34, "Bb", ("B", -1, 2)),
        (36, "C", ("C", None, 3)),
        (47, "B", ("B", None, 3)),
        (48, "B#", ("B", 1, 3)),
        (35, "Cb", ("C", -1, 3)),
    ],
)
def test_pitch_xml_is_written_an_octave_up(midi, name, expected):
    step, alter, octave = expected
    x = score.pitch_xml(midi, name)
    assert f"<step>{step}</step>" in x and f"<octave>{octave}</octave>" in x
    assert (f"<alter>{alter}</alter>" in x) == (alter is not None)


def _notes():
    """Two bars: a held note across the barline, an eighth-note run, a flagged note and two notes in one 16th."""
    raw = [
        (0, 4, 28),
        (4, 2, 28),
        (6, 2, 35),
        (8, 12, 40),
        (20, 2, 38),
        (22, 2, 36),
        (24, 4, 31),
        (28, 1, 33),
        (28, 3, 35),
    ]
    out = []
    for k, (g, length, pitch) in enumerate(raw):
        out.append(
            dict(
                start=1.0 + g * 0.125 + (0.01 if k == 8 else 0),
                end=1.0 + (g + length) * 0.125,
                pitch=pitch,
                grid_start=g,
                grid_len=length,
                bar=g // 16 + 1,
                beat=(g % 16) // 4 + 1,
                sixteenth=g % 4 + 1,
                agree=1.0,
                flag="octave" if pitch == 36 else None,
            )
        )
    return out


CHORDS = [
    dict(bar=1, root=4, quality="m", names=score.SHARPS, chord="Em"),
    dict(bar=2, root=0, quality="", names=score.FLATS, chord="C"),
]


def _xml(**kw):
    args = dict(
        notes=_notes(),
        chords=CHORDS,
        key=("E minor", 4, False),
        tempo=120.0,
        first_downbeat=1.0,
        title="T & <t>",
        artist="A",
        check_bars={2: ["why"]},
        draft=False,
    )
    args.update(kw)
    return score.musicxml(**args)


def test_musicxml_is_well_formed_and_escapes_text():
    doc = xml.dom.minidom.parseString(_xml())
    assert doc.getElementsByTagName("movement-title")[0].firstChild.data == "T & <t>"
    assert len(doc.getElementsByTagName("measure")) == 2


def test_every_measure_has_sixteen_sixteenths_on_both_staves():
    doc = xml.dom.minidom.parseString(_xml())
    for measure in doc.getElementsByTagName("measure"):
        total = {"1": 0, "2": 0}
        for n in measure.getElementsByTagName("note"):
            if n.getElementsByTagName("grace"):
                continue
            staff = n.getElementsByTagName("staff")[0].firstChild.data
            total[staff] += int(n.getElementsByTagName("duration")[0].firstChild.data)
        assert total == {"1": 16, "2": 16}, measure.getAttribute("number")


def test_ties_open_and_close_in_pairs():
    x = _xml()
    assert x.count('<tie type="start"/>') == x.count('<tie type="stop"/>') > 0


def test_score_marks_uncertain_notes_checks_and_chords():
    x = _xml()
    assert 'parentheses="yes"' in x  # the flagged note
    assert x.count(">check<") == 1  # bar 2 only
    assert "<root-step>E</root-step>" in x and "<kind>minor</kind>" in x
    assert '<grace slash="yes"/>' in x  # the first of the two notes sharing a 16th
    assert "<octave-change>-1</octave-change>" in x  # plays back at sounding pitch
    assert "DRAFT" not in x and "DRAFT: failed quality gates" in _xml(draft=True)


def test_tab_staff_carries_string_and_fret():
    doc = xml.dom.minidom.parseString(_xml())
    tabs = [n for n in doc.getElementsByTagName("note") if n.getElementsByTagName("fret")]
    assert tabs
    for n in tabs:
        string = int(n.getElementsByTagName("string")[0].firstChild.data)
        fret = int(n.getElementsByTagName("fret")[0].firstChild.data)
        assert 1 <= string <= 4 and 0 <= fret <= score.MAX_FRET


def test_triads_match_rhythm_module():
    assert score.TRIADS == TRIADS


def test_find_mscore_rejects_a_missing_explicit_path(monkeypatch):
    monkeypatch.setattr(score, "MSCORE_CANDIDATES", [])
    assert score.find_mscore("/nonexistent/mscore") is None


def test_lines_start_where_the_bass_comes_in():
    notes = [dict(n, grid_start=n["grid_start"] + 48, bar=n["bar"] + 3) for n in _notes()]  # the bass enters in bar 4
    notes += [
        dict(n, grid_start=n["grid_start"] + 64, bar=n["bar"] + 4, start=n["start"] + 8, end=n["end"] + 8)
        for n in notes
    ]
    doc = xml.dom.minidom.parseString(_xml(notes=notes, chords=[], check_bars={}))
    breaks = [m.getAttribute("number") for m in doc.getElementsByTagName("measure") if m.getElementsByTagName("print")]
    assert breaks == ["4", "8"]  # four-bar lines counted from bar 4, after the first line's lead-in
