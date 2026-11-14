"""Repeats: bars that sound alike, what they say about each other, and a fix carried from one to the others."""

import numpy as np
import pytest

from bass2sheet import repeats as R

STEP = 0.125  # a sixteenth at 120 bpm


def _rep(bars=6, groups=((0, 1, 2, 3), (4, 5)), rise=None, sounding=None, alike=None, grid=None):
    """Repeats made by hand: the bars of a group sound alike in every sixteenth, the others not at all."""
    a = np.zeros((bars, bars, R.SLOTS))
    for g in groups if alike is None else ():
        for i in (i for i in g if i < bars):
            for j in (j for j in g if j < bars):
                a[i, j] = 1.0
    measured = dict(
        alike=a if alike is None else alike,
        sounding=np.ones((bars, R.SLOTS), bool) if sounding is None else sounding,
        rise=np.zeros((bars, R.SLOTS)) if rise is None else rise,
    )
    return R.Repeats(measured, np.arange(bars * R.SLOTS + 1) * STEP if grid is None else grid, 0)


RIFF = [(0, 4, 40), (4, 2, 40), (6, 2, 47), (8, 4, 52), (12, 2, 47), (14, 2, 45)]  # (sixteenth, length, pitch)


def _notes(bars, riff=RIFF, change=None):
    """The riff written in every bar; `change` maps (bar, index in the riff) to another pitch, or None for no note."""
    out = []
    for b in bars:
        for k, (slot, length, pitch) in enumerate(riff):
            pitch = (change or {}).get((b, k), pitch)
            if pitch is not None:
                t = (b * R.SLOTS + slot) * STEP
                out.append(dict(id=f"m{len(out)}", start=t, end=t + length * STEP - 0.01, pitch=pitch, state="machine"))
    return out


def _levels(rep, notes, fixed=(), passed=()):
    level = np.zeros((rep.bars, R.SLOTS), int)
    for b in passed:
        level[b] = 1
    for b, k in fixed:
        level[b, k] = 2
    return level


# ------------------------------------------------------------------------------------------------ the patterns


def test_bars_that_sound_alike_are_one_pattern_lettered_as_they_come():
    rep = _rep(7, groups=((1, 3, 4), (2, 6)))
    assert rep.family.tolist() == [-1, 0, 1, 0, 0, -1, 1]  # a bar with no repeat has no pattern
    assert rep.related[1].tolist() == [False, False, False, True, True, False, False]
    assert [R.letter(k) for k in (0, 1, 25, 26, 27, 52)] == ["A", "B", "Z", "AA", "AB", "BA"]


def test_two_bars_are_repeats_only_if_enough_of_them_sounds_and_sounds_the_same():
    alike = np.ones((3, 3, R.SLOTS))
    alike[0, 1, 9:] = alike[1, 0, 9:] = 0.5  # the same in nine sixteenths of sixteen: still a repeat
    alike[0, 2, 7:] = alike[2, 0, 7:] = 0.5  # in seven: not
    alike[1, 2, 7:] = alike[2, 1, 7:] = 0.5
    rep = _rep(3, alike=alike)
    assert rep.related.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    assert rep.same[0, 1, :9].all() and not rep.same[0, 1, 9:].any()
    sounding = np.zeros((2, R.SLOTS), bool)
    sounding[:, :3] = True  # silent together most of the bar, and only three sixteenths to go by
    assert not _rep(2, groups=((0, 1),), sounding=sounding).related.any()


def test_patterns_whose_bars_are_mostly_repeats_of_each_other_are_one():
    alike = np.zeros((4, 4, R.SLOTS))
    for i, j in ((0, 1), (2, 3), (0, 3), (1, 2), (1, 3)):  # 2 is no repeat of 0, where its pattern would start
        alike[i, j] = alike[j, i] = 1.0
    assert _rep(4, alike=alike).family.tolist() == [0, 0, 0, 0]
    for i, j in ((0, 3), (1, 2)):
        alike[i, j] = alike[j, i] = 0.0
    assert _rep(4, alike=alike).family.tolist() == [0, 0, 1, 1]  # one pair in four across: two patterns


# --------------------------------------------------------------------------------------------- what is written


def test_what_is_written_sixteenth_by_sixteenth():
    rep = _rep(2, groups=((0, 1),))
    notes = _notes([0]) + [dict(id="x", start=2.26, end=2.30, pitch=50, state="added")]  # the last: between middles
    pitch, start, owner = rep.tokens(notes)
    assert pitch[0].tolist() == [40] * 6 + [47] * 2 + [52] * 4 + [47] * 2 + [45] * 2
    assert np.flatnonzero(start[0]).tolist() == [0, 4, 6, 8, 12, 14] and owner[0, 5] == 1 and owner[1, 0] == -1
    assert not pitch[1].any() and not start[1].any()  # a note too short to cover a sixteenth's middle is not seen
    early = [dict(start=0.5 - 0.03, end=1.0, pitch=40), dict(start=1.0 + 0.05, end=1.5, pitch=41)]
    pitch, start, _ = rep.tokens(early)  # a note a little early or late still starts on its sixteenth
    assert np.flatnonzero(start[0]).tolist() == [4, 8] and pitch[0, 3:9].tolist() == [0, 40, 40, 40, 40, 41]
    assert rep.cover(dict(start=0.0, end=99.0)) == (0, 32) and rep.bar_of(2.1) == 1 and rep.bar_of(99) is None


def test_a_moment_of_one_bar_is_found_in_another_as_far_into_its_sixteenth():
    grid = np.concatenate([np.arange(17) * 0.125, 2.0 + np.arange(1, 17) * 0.15])  # the second bar is slower
    rep = _rep(2, groups=((0, 1),), grid=grid)
    assert rep.to_bar(0.25, 0, 1) == pytest.approx(2.3) and rep.to_bar(0.3125, 0, 1) == pytest.approx(2.375)
    assert rep.to_bar(2.375, 1, 0) == pytest.approx(0.3125) and rep.slot_time(16) == 2.0


def test_whose_word_a_sixteenth_is():
    rep = _rep(4)
    notes = _notes([0, 1, 2, 3])
    machine = rep.tokens(notes)[0]
    assert not rep.trust(notes, machine, []).any()
    notes[8]["pitch"], notes[8]["state"] = 48, "fixed"  # bar 1, sixteenths 6 and 7
    del notes[20]  # bar 3, sixteenths 6 and 7: a note of the tool's taken out
    level = rep.trust(notes, machine, [[0, 3.2], [6.5, 6.9]])
    assert level[1, 6:8].tolist() == [2, 2] and level[3, 6:8].tolist() == [2, 2]  # what they changed
    assert (level[0] == 1).all() and level[1, :6].tolist() == [1] * 6  # what they worked through and left
    assert level[2].tolist() == [0] * 16  # the rest is the tool's
    assert level[3].tolist() == [0, 0, 0, 0, 1, 1, 2, 2] + [0] * 8
    assert rep.within([[2.0, 2.2], [7.9, 9]])[[1, 3]].sum() == 2 + 1


# ------------------------------------------------------------------------------------ what the repeats say


def test_the_tool_against_itself_most_repeats_have_another_note_there():
    rep = _rep()
    notes = _notes(range(6), change={(2, 2): 59})  # bar 2 has another pitch on sixteenths 6 and 7
    pitch, start, _ = rep.tokens(notes)
    found = rep.differences_of(pitch, start, _levels(rep, notes))
    assert len(found) == 1 and found[0] == dict(bar=2, k0=6, k1=7, source=found[0]["source"], why="most", count=3)
    assert found[0]["source"] in (0, 1, 3)
    # one repeat against one is no majority; and a note against silence is not marked on the tool's word alone
    pairs = _rep(groups=((0, 1), (2, 3), (4, 5)))
    assert pairs.differences_of(pitch, start, _levels(rep, notes)) == []
    gone = _notes(range(6), change={(2, 2): None})
    assert rep.differences_of(*rep.tokens(gone)[:2], _levels(rep, gone)) == []


def test_what_the_person_fixed_or_left_outweighs_what_the_tool_wrote():
    rep = _rep()
    notes = _notes(range(6), change={(1, 3): 64})  # bar 1 was fixed on sixteenths 8 to 11: the others differ now
    pitch, start, _ = rep.tokens(notes)
    fixed = [(1, k) for k in range(8, 12)]
    found = rep.differences_of(pitch, start, _levels(rep, notes, fixed=fixed))
    assert [(d["bar"], d["k0"], d["k1"], d["source"], d["why"]) for d in found] == [
        (b, 8, 11, 1, "fixed") for b in (0, 2, 3)
    ]
    # a bar they played past and left is theirs: it is not marked, whatever they fixed elsewhere ...
    found = rep.differences_of(pitch, start, _levels(rep, notes, fixed=fixed, passed=[0]))
    assert [(d["bar"], d["source"], d["why"]) for d in found] == [(2, 1, "fixed"), (3, 1, "fixed")]
    # ... and it counts against the bars they have not reached
    other = _notes(range(6), change={(0, 3): 64})  # bar 0 was played past and left with a pitch of its own
    found = rep.differences_of(*rep.tokens(other)[:2], _levels(rep, other, passed=[0]))
    assert [(d["bar"], d["k0"], d["k1"], d["source"], d["why"]) for d in found] == [
        (b, 8, 11, 0, "passed") for b in (1, 2, 3)
    ]
    # silence is the person's word too: a note they took out is marked in the repeats
    gone = _notes(range(6), change={(1, 3): None})
    found = rep.differences_of(*rep.tokens(gone)[:2], _levels(rep, gone, fixed=fixed))
    assert [(d["bar"], d["why"]) for d in found] == [(0, "fixed"), (2, "fixed"), (3, "fixed")]
    # ... but how long a note is held is not: a note they cut short is not marked in the repeats
    short = _notes(range(6))
    short[3]["end"] -= 2 * STEP  # bar 0: the fourth note ends two sixteenths early
    found = rep.differences_of(*rep.tokens(short)[:2], _levels(rep, short, fixed=[(0, 10), (0, 11)]))
    assert found == []
    # and a note they added is marked where the repeats have none, from its start
    more = _notes(range(6), riff=[*RIFF[:3], (8, 2, 52), (12, 2, 47), (14, 2, 45)])  # nothing on 10 and 11 ...
    more.append(dict(id="x", start=(R.SLOTS + 10) * STEP, end=(R.SLOTS + 12) * STEP - 0.01, pitch=55, state="added"))
    found = rep.differences_of(*rep.tokens(more)[:2], _levels(rep, more, fixed=[(1, 10), (1, 11)]))  # ... but in bar 1
    assert [(d["bar"], d["k0"], d["k1"], d["source"]) for d in found] == [(b, 10, 11, 1) for b in (0, 2, 3)]
    # nor is the line between two notes a sixteenth earlier or later: bar 0 was fixed to change note on 7, not 6
    early = [(0, 4, 40), (4, 3, 40), (7, 1, 47), (8, 4, 52), (12, 2, 47), (14, 2, 45)]
    moved = _notes([0], riff=early) + [n for n in _notes(range(6)) if n["start"] >= 2]
    found = rep.differences_of(*rep.tokens(moved)[:2], _levels(rep, moved, fixed=[(0, 6), (0, 7)]))
    assert found == []
    # where they said to keep what is written, nothing is marked
    settled = np.zeros((6, R.SLOTS), bool)
    settled[2, 8:12] = True
    found = rep.differences_of(pitch, start, _levels(rep, notes, fixed=fixed), settled)
    assert [d["bar"] for d in found] == [0, 3]


def test_a_note_start_is_taken_from_a_repeat_only_where_this_bars_own_sound_agrees():
    held = [(0, 6, 40), (6, 2, 47), (8, 4, 52), (12, 2, 47), (14, 2, 45)]  # the first note held, not plucked again
    notes = _notes([0, 1]) + [
        dict(n, start=n["start"] + 4, end=n["end"] + 4) for n in _notes([0], riff=held)
    ]  # bar 2 holds where bars 0 and 1 pluck again on sixteenth 4
    for n in notes[:12]:
        n["state"] = "fixed"
    rise = np.zeros((3, R.SLOTS))
    rep = _rep(3, groups=((0, 1, 2),), rise=rise)
    pitch, start, _ = rep.tokens(notes)
    level = np.array([[2] * 16, [2] * 16, [0] * 16])
    assert rep.differences_of(pitch, start, level) == []  # no rise in level there in bar 2: it is held
    rise[2, 4] = R.PLUCK_DB + 1
    found = _rep(3, groups=((0, 1, 2),), rise=rise).differences_of(pitch, start, level)
    assert [(d["bar"], d["k0"], d["k1"], d["why"]) for d in found] == [(2, 4, 4, "fixed")]
    # and the other way: a start is dropped only where the level does not rise
    level = np.array([[0] * 16, [0] * 16, [2] * 16])
    rise = np.full((3, R.SLOTS), R.HELD_DB + 1)
    assert _rep(3, groups=((0, 1, 2),), rise=rise).differences_of(pitch, start, level) == []
    found = _rep(3, groups=((0, 1, 2),)).differences_of(pitch, start, level)
    assert [(d["bar"], d["k0"], d["source"]) for d in found] == [(0, 4, 2), (1, 4, 2)]


def test_after_a_fix_the_bars_written_as_this_one_was_and_sounding_the_same():
    rep = _rep()
    wrong = {(b, 2): 59 for b in (0, 1, 3)}  # the same wrong pitch in three of the four repeats
    notes = _notes(range(6), change={**wrong, (5, 2): 59})  # and in a bar of the other pattern
    before = rep.tokens(notes)
    fixed = _notes(range(6), change={(1, 2): 59, (3, 2): 59, (5, 2): 59})  # bar 0 put right
    after = rep.tokens(fixed)
    assert rep.changed(before, after) == [(0, 6, 7)]
    level = _levels(rep, fixed, fixed=[(0, 6), (0, 7)])
    assert rep.same_mistake(before, after, level) == [
        dict(source=0, bar=1, k0=6, k1=7),
        dict(source=0, bar=3, k0=6, k1=7),
    ]  # not bar 2, which was right; not bar 5, which is no repeat
    level[3, 0] = 1  # played past elsewhere in the bar: no matter
    level[1, 7] = 1  # the person has been through this spot of bar 1: it is left to them
    assert [i["bar"] for i in rep.same_mistake(before, after, level)] == [3]
    settled = np.zeros((6, R.SLOTS), bool)
    settled[3, 6] = True  # ... and they said to keep what bar 3 has
    assert rep.same_mistake(before, after, level, settled) == []
    alike = np.zeros((6, 6, R.SLOTS))
    alike[:4, :4] = 1.0
    alike[0, 1, 6:8] = alike[1, 0, 6:8] = 0.3  # bar 1 sounds different just there
    assert [i["bar"] for i in _rep(alike=alike).same_mistake(before, after, _levels(rep, fixed))] == [3]


def test_a_split_is_carried_only_to_bars_where_the_level_rises_too():
    rep = _rep(3, groups=((0, 1, 2),))
    held = [(0, 6, 40), (6, 2, 47), (8, 4, 52), (12, 2, 47), (14, 2, 45)]
    before = rep.tokens(_notes(range(3), riff=held))
    after = rep.tokens(_notes([0]) + [n for n in _notes(range(3), riff=held) if n["start"] >= 2])
    assert rep.changed(before, after) == [(0, 4, 4)]
    assert rep.same_mistake(before, after, _levels(rep, [])) == []
    rise = np.zeros((3, R.SLOTS))
    rise[2, 4] = R.PLUCK_DB
    assert _rep(3, groups=((0, 1, 2),), rise=rise).same_mistake(before, after, _levels(rep, [])) == [
        dict(source=0, bar=2, k0=4, k1=4)
    ]


def test_what_a_bar_has_in_a_stretch_as_it_would_lie_in_another():
    rep = _rep()
    notes = _notes([0])
    got = rep.pieces(notes, 0, 2, 5, 9)  # from inside the second note to inside the fourth
    assert [(p["pitch"], p["starts_here"], p["ends_here"], p["slot"]) for p in got] == [
        (40, False, True, 37),
        (47, True, True, 38),
        (52, True, False, 40),
    ]
    assert got[0]["start"] == 4 + 5 * STEP and got[0]["end"] == pytest.approx(4 + 6 * STEP - 0.01)
    assert got[1]["start"] == pytest.approx(4 + 6 * STEP) and got[2]["end"] == 4 + 10 * STEP
    assert rep.pieces(notes, 1, 2, 0, 15) == []


# --------------------------------------------------------------------------------------------- from the sound


def test_repeats_are_found_in_the_sound_of_the_bass_track():
    import librosa

    from bass2sheet.common import SR
    from bass2sheet.render import pluck

    riffs = dict(a=RIFF, b=[(0, 4, 45), (4, 4, 45), (8, 2, 50), (10, 2, 48), (12, 4, 43)], rest=[])
    order = ["rest", "a", "b", "a", "a", "b", "a"]
    y = np.zeros(int((len(order) * 2 + 0.5) * SR), dtype=np.float32)
    for bar, name in enumerate(order):
        for slot, length, pitch in riffs[name]:
            pitch += 2 if (bar, slot) == (4, 6) else 0  # one note of one bar is another note
            w = 0.4 * pluck(librosa.midi_to_hz(pitch), length * STEP * 0.95, 100)
            i = int((bar * 2 + slot * STEP) * SR)
            y[i : i + len(w)] += w
    grid = np.arange(len(order) * R.SLOTS + 1) * STEP
    measured = R.measure(y, grid, 0)
    assert measured["alike"].shape == (7, 7, 16) and measured["sounding"].shape == measured["rise"].shape == (7, 16)
    assert not measured["sounding"][0].any() and measured["sounding"][1].all()
    assert measured["rise"][1, [0, 4, 6, 8, 12, 14]].min() > 3 and measured["rise"][1, [1, 2, 9, 10]].max() < 2
    rep = R.Repeats(measured, grid, 0)
    assert rep.family.tolist() == [-1, 0, 1, 0, 0, 1, 0]
    assert rep.same[1, 3].all() and rep.same[1, 4, :6].all() and rep.same[1, 4, 8:].all()
    assert not rep.same[1, 4, 6:8].any()  # where bar 4 has another note, it does not sound the same
    assert not rep.same[1, 2].any() and not rep.related[0].any()
    # written the same everywhere: where it sounds the same nothing is marked, and the odd note is not either
    notes = _notes([1, 3, 4, 6]) + _notes([2, 5], riff=riffs["b"])
    assert rep.differences(notes, rep.tokens(notes)[0], []) == []
    notes[2]["pitch"] = 59  # bar 1, sixteenths 6 and 7: unlike bars 3 and 6, which sound the same there
    found = rep.differences(notes, rep.tokens(notes)[0], [])
    assert [(d["bar"], d["k0"], d["k1"], d["why"], d["count"]) for d in found] == [(1, 6, 7, "most", 2)]
