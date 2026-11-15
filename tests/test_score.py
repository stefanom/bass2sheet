import xml.dom.minidom

import pytest

from bass2sheet import score


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


def test_the_sheet_has_a_symbol_for_every_chord_the_finder_can_give():
    from bass2sheet import harmony

    assert set(score.KINDS) == set(harmony.QUALITIES)


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


def _n(g, length, pitch):
    return dict(
        start=g * 0.125,
        end=(g + length) * 0.125,
        pitch=pitch,
        grid_start=g,
        grid_len=length,
        bar=g // 16 + 1,
        flag=None,
    )


def _measures(text):
    return xml.dom.minidom.parseString(text).getElementsByTagName("measure")


def test_tab_styles_low_uses_open_strings_and_position_keeps_the_hand_in_place():
    line = [33, 35, 36, 38, 33, 31, 33]  # A B C D A G A, from the open A string up
    low, pos = score.frets(line, "low"), score.frets(line, "position")
    assert low[0] == (1, 0) and (2, 0) in low  # the open A and D strings
    assert all(f > 0 for _, f in pos)
    assert max(f for _, f in pos) - min(f for _, f in pos) <= 4
    for (s, f), pitch in zip(low + pos, line + line):
        assert score.STRINGS[s] + f == pitch
    with pytest.raises(ValueError, match="tab style"):
        score.frets(line, "sideways")


def test_the_easy_fingering_keeps_the_hand_where_it_is():
    place = lambda places: " ".join("EADG"[s] + str(f) for s, f in places)  # noqa: E731
    # a low F, its octave and back: the octave shape, one finger on the E string and one two frets up on the D
    assert place(score.frets([29, 41, 29], "easy")) == "E1 D3 E1"
    # G Bb C Bb G F G lies under one hand at the third fret; nearest the nut it takes three open strings
    box = [43, 46, 48, 46, 43, 41, 43]
    assert (
        place(score.frets(box, "easy")) == "D5 G3 G5 G3 D5 D3 D5"
        and place(score.frets(box, "low")) == "G0 G3 G5 G3 G0 D3 G0"
    )
    assert score.effort(score.frets(box, "easy")) == dict(notes=7, moves=0, frets=0, skips=0, open=0)
    # a walk up and down an octave: with open strings where they save a stretch, or without any
    walk = [40, 43, 45, 47, 48, 47, 45, 43, 40, 38, 36, 35, 33]
    easy, closed, position = (score.frets(walk, style) for style in ("easy", "closed", "position"))
    assert place(easy) == "D2 G0 G2 G4 G5 G4 G2 G0 D2 D0 A3 A2 A0"
    assert place(closed) == "D2 D5 G2 G4 G5 G4 G2 D5 D2 A5 A3 A2 E5" and all(f > 0 for _, f in closed)
    assert score.effort(easy)["moves"] == score.effort(closed)["moves"] == 0 < score.effort(position)["moves"]
    for style in ("easy", "closed"):  # every note is where its pitch is, and the lowest E has only its open string
        for (s, f), pitch in zip(score.frets(walk + [28], style), walk + [28]):
            assert score.STRINGS[s] + f == pitch
    assert score.frets([28, 28], "closed") == [(0, 0), (0, 0)] and score.frets([], "easy") == []
    with pytest.raises(ValueError, match="expected one of easy, closed, low, position"):
        score.frets(walk, "high")


def test_a_bar_that_comes_again_is_fingered_the_same_way():
    bar = [43, 46, 48, 46]  # G Bb C Bb, after a low F the first time and after a high run the second
    song = [29, 29, 41, 41] + bar + [52, 55, 57, 55] + bar + [33, 33, 33, 33]
    when, bars = [2 * k for k in range(len(song))], [k // 4 for k in range(len(song))]
    placed = score.frets(song, "easy", starts=when, bars=bars)
    assert placed[4:8] == placed[12:16]
    for (s, f), pitch in zip(placed, song):
        assert score.STRINGS[s] + f == pitch
    # two notes played together are still placed where both fit
    assert score.frets([36, 40], "easy", also=[[43], []])[0] == (1, 3)


def test_what_a_fingering_asks_of_the_hand_is_counted():
    # fret 1, fret 3 (the same hand), an open string (no hand), fret 8 (a move of four frets), fret 1 (four back)
    places = [(0, 1), (2, 3), (3, 0), (3, 8), (0, 1)]
    assert score.effort(places) == dict(notes=5, moves=2, frets=8, skips=2, open=1)
    assert score.effort([]) == dict(notes=0, moves=0, frets=0, skips=0, open=0)
    # where the hand is for each note: its first finger on the lowest fret of each stretch between two moves
    assert score.hand([(0, 1), (0, 1), (2, 3), (3, 0), (2, 5), (3, 3), (3, 5), (2, 5), (0, 1)]) == [
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        1,
    ]
    assert score.hand([(3, 0), (2, 0)]) == [None, None] and score.hand([]) == []
    assert score.hand([(3, 0), (2, 7), (3, 9)]) == [7, 7, 7]  # an open string before the first fretted note waits there


def test_a_snippet_is_the_same_bars_with_the_same_fingering_and_no_title():
    notes = [_n(k * 4, 4, 40 + k % 5) for k in range(24)]  # six bars of quarter notes
    full = _xml(notes=notes, chords=[], check_bars={})
    snip = _xml(notes=notes, chords=[], check_bars={}, bars=(3, 4))
    assert "<movement-title>" in full and "<movement-title>" not in snip
    assert [m.getAttribute("number") for m in _measures(snip)] == ["3", "4"]
    assert "new-system" not in snip
    frets_of = lambda text, bar: [
        f.firstChild.data
        for m in _measures(text)
        if m.getAttribute("number") == bar
        for f in m.getElementsByTagName("fret")
    ]
    assert frets_of(snip, "3") == frets_of(full, "3") != []
    held = _xml(notes=[_n(0, 40, 40)], chords=[], check_bars={}, bars=(2, 2))  # tied in from bar 1, out to bar 3
    assert "<tie" not in held


def test_line_times_follow_the_given_bar_times():
    notes = [_n(k * 16, 16, 40) for k in range(8)]
    text = _xml(notes=notes, chords=[], check_bars={}, bar_times={k: 10.0 * k for k in range(1, 9)})
    assert ">0:10<" in text and ">0:50<" in text  # bars 1 and 5 start the two lines


def test_chord_frets_put_the_notes_played_together_on_the_strings_above():
    assert score.chord_frets(1, 0, [40]) == [(2, 2)]  # open A with the E above it: D string, 2nd fret
    assert score.chord_frets(1, 3, [43]) == [(3, 0)]  # C with G: the open G string
    assert score.chord_frets(1, 3, [44]) == [(3, 1)]  # C with G#: the fret nearest the hand
    assert score.chord_frets(0, 8, [52]) == [(2, 14)] or score.chord_frets(0, 8, [52]) == [(3, 9)]  # a tenth
    assert score.chord_frets(3, 2, [50]) is None  # nothing above the G string
    assert score.chord_frets(0, 1, [52]) is None  # a tenth from the 1st fret: out of reach on any string
    assert score.chord_frets(0, 3, [38, 43]) == [(2, 0), (3, 0)]  # open strings are always in reach


def test_a_note_with_others_goes_where_they_fit_and_the_score_shows_a_chord():
    line = [_n(0, 4, 33), dict(_n(4, 4, 36), also=[43]), _n(8, 4, 43), dict(_n(12, 4, 43), also=[50])]
    events = score.written_events(line)
    assert [e.get("also_tab") for e in events][:3] == [None, [[43, 1, 0]], None]  # the open G string
    assert (events[1]["string"], events[1]["fret"]) == (3, 3)  # the C on the A string, not where it has no room
    last = events[3]  # G with the D above: alone the G is the open G string; with the D it moves down a string
    assert (last["string"], last["fret"]) == (2, 5) and last["also_tab"] == [[50, 1, 7]]
    for e in events:
        for pitch, string, fret in e.get("also_tab", []):
            assert score.STRINGS[4 - string] + fret == pitch and string < e["string"]
    text = score.musicxml(line, [], ("C major", 0, True), 120.0, 0.0, "t", "a", {}, False)
    doc = xml.dom.minidom.parseString(text)
    chord = [n for n in doc.getElementsByTagName("note") if n.getElementsByTagName("chord")]
    assert len(chord) == 4  # two chord notes, on the staff and in the tab
    assert [n.getElementsByTagName("step")[0].firstChild.data for n in chord] == ["G", "D", "G", "D"]
    assert all(n.firstChild.tagName == "chord" for n in chord)  # MusicXML wants it first
    total = {"1": 0, "2": 0}
    for n in doc.getElementsByTagName("note"):
        if not n.getElementsByTagName("chord"):  # a chord note takes no time of its own
            total[n.getElementsByTagName("staff")[0].firstChild.data] += int(
                n.getElementsByTagName("duration")[0].firstChild.data
            )
    assert total == {"1": 16, "2": 16}


def test_a_chord_note_out_of_reach_stays_on_the_staff_and_out_of_the_tab():
    line = [dict(_n(0, 8, 29), also=[52])]  # low F with the E an octave and a seventh above
    e = score.written_events(line)[0]
    assert e["also_tab"] == [[52, None, None]]
    doc = xml.dom.minidom.parseString(score.musicxml(line, [], ("C major", 0, True), 120.0, 0.0, "t", "a", {}, False))
    chord = [n for n in doc.getElementsByTagName("note") if n.getElementsByTagName("chord")]
    assert [n.getElementsByTagName("staff")[0].firstChild.data for n in chord] == ["1"]


def test_a_held_chord_ties_every_note_of_it():
    line = [dict(_n(12, 8, 36), also=[43])]  # across the bar line
    text = score.musicxml(line, [], ("C major", 0, True), 120.0, 0.0, "t", "a", {}, False)
    assert text.count('<tie type="start"/>') == text.count('<tie type="stop"/>') == 4  # two notes, two staves


def test_a_note_that_sounds_like_another_pitch_is_shown_in_parentheses():
    line = [_n(0, 4, 33), dict(_n(4, 4, 36), sounds_like=48, own_sound=0.3), dict(_n(8, 4, 38), own_sound=0.9)]
    text = score.musicxml(line, [], ("C major", 0, True), 120.0, 0.0, "t", "a", {}, False)
    assert text.count('parentheses="yes"') == 1 and "(notes) uncertain" in text
    plain = score.musicxml(
        [_n(0, 4, 33), dict(_n(4, 4, 36), own_sound=0.9)], [], ("C major", 0, True), 120.0, 0.0, "t", "a", {}, False
    )
    assert 'parentheses="yes"' not in plain and "(notes) uncertain" not in plain


def _mscore(tmp_path, body):
    """A program in MuseScore's place that does `body`."""
    import os
    import stat
    import sys

    path = str(tmp_path / "mscore")
    with open(path, "w") as f:
        f.write(f"#!{sys.executable}\nimport sys, time\n{body}")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def test_a_musescore_that_fails_or_hangs_is_an_error_in_words_and_its_style_is_not_left_behind(tmp_path):
    import os

    xml_path, pdf = str(tmp_path / "bass.musicxml"), str(tmp_path / "bass.pdf")
    with open(xml_path, "w") as f:
        f.write(_xml())
    for stale in ("bass-1.svg", "bass-2.svg", "bass-old.svg"):  # pictures of the score as it was before
        (tmp_path / stale).write_text("<svg/>")
    broken = _mscore(tmp_path, "sys.stderr.write('sandbox_extension: noise\\nno such font\\n')\nsys.exit(3)\n")
    assert score.engrave(xml_path, pdf, broken) == "MuseScore exited 3: no such font"
    assert score.engrave_pages(xml_path, pdf, broken) == ("MuseScore exited 3: no such font", [])
    silent = _mscore(tmp_path, "sys.exit(0)\n")  # it says it is done and wrote nothing
    assert score.engrave(xml_path, pdf, silent) == "MuseScore exited 0: "
    stuck = _mscore(tmp_path, "time.sleep(30)\n")
    assert score.engrave(xml_path, pdf, stuck, timeout=0.5) == "MuseScore timed out after 0.5 s"
    assert score.engrave_pages(xml_path, pdf, stuck, timeout=0.5) == ("MuseScore timed out after 0.5 s", [])
    # nothing of the attempts is left, and the pictures of the earlier score are gone (the one not numbered stays)
    assert sorted(os.listdir(tmp_path)) == ["bass-old.svg", "bass.musicxml", "mscore"]


def test_the_pages_come_back_in_their_order_and_those_of_a_longer_score_are_gone(tmp_path):
    import os

    from conftest import mscore_stand_in

    xml_path, pdf = str(tmp_path / "bass.musicxml"), str(tmp_path / "bass.pdf")
    with open(xml_path, "w") as f:
        f.write(_xml())
    pages = (  # a stand-in that draws as many pages as its name says
        "import json, os\n"
        "for job in json.load(open(sys.argv[4])):\n"
        "    for out in job['out']:\n"
        "        if out.endswith('.pdf'):\n"
        "            open(out, 'wb').write(b'%PDF-1.4')\n"
        "        else:\n"
        "            for k in range(int(os.path.basename(sys.argv[0]).split('-')[1])):\n"
        "                open(f'{out[:-4]}-{k + 1}.svg', 'w').write('<svg/>')\n"
    )
    eleven = _mscore(tmp_path, pages)
    os.rename(eleven, str(tmp_path / "mscore-11"))
    error, drawn = score.engrave_pages(xml_path, pdf, str(tmp_path / "mscore-11"))
    assert error is None and open(pdf, "rb").read(5) == b"%PDF-"
    assert [os.path.basename(p) for p in drawn] == [f"bass-{k}.svg" for k in range(1, 12)]  # 10 after 9, not after 1
    os.rename(str(tmp_path / "mscore-11"), str(tmp_path / "mscore-2"))
    error, drawn = score.engrave_pages(xml_path, pdf, str(tmp_path / "mscore-2"))
    assert error is None and [os.path.basename(p) for p in drawn] == ["bass-1.svg", "bass-2.svg"]
    assert sorted(os.listdir(tmp_path)) == ["bass-1.svg", "bass-2.svg", "bass.musicxml", "bass.pdf", "mscore-2"]
    program = mscore_stand_in(tmp_path)  # and the tests' own stand-in for MuseScore draws what is asked of it
    assert score.find_mscore(program) == program  # the one that is named comes before the ones looked for
    error, drawn = score.engrave_pages(xml_path, pdf, program)
    assert error is None and [os.path.basename(p) for p in drawn] == ["bass-1.svg"]
    assert "<svg" in open(drawn[0]).read() and score.engrave(xml_path, pdf, program) is None
