import librosa
import numpy as np

from bass2sheet import notes as N
from bass2sheet.common import HOP, SR
from bass2sheet.render import pluck


def _stem(spec, seconds):
    """Mono stem from (start s, duration s, MIDI pitch) plucks."""
    y = np.zeros(int(seconds * SR))
    for start, dur, pitch in spec:
        w = 0.5 * pluck(librosa.midi_to_hz(pitch), dur, 100)
        i = int(start * SR)
        y[i : i + len(w)] += w[: len(y) - i]
    return y.astype(np.float32)


def _features(y, spec, lag_frames=2):
    """Features as the pitch stage would produce them, built from the known notes instead of CREPE/pYIN: the
    pitch flips `lag_frames` after each true start, as real trackers do."""
    n = len(y) // HOP + 1
    f0 = np.full(n, 50.0)
    per = np.full(n, 0.02)
    for start, dur, pitch in spec:
        a, b = int(round(start * 100)) + lag_frames, int(round((start + dur) * 100))
        f0[a:b], per[a:b] = librosa.midi_to_hz(pitch), 0.9
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0] + 1e-9)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=1000)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True)
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72, bins_per_octave=12))
    return dict(f0_c=f0, per_c=per, f0_p=f0.copy(), rms_db=rms_db, onsets=onsets, cqt=cqt)


SPEC = [(0.5, 0.45, 40), (1.0, 0.45, 40), (1.5, 0.2, 43), (1.75, 0.2, 45), (2.0, 0.9, 47), (3.5, 0.4, 35)]


def test_segment_notes_recovers_plucked_notes():
    y = _stem(SPEC, 4.5)
    out, _, _ = N.segment_notes(_features(y, SPEC), y)
    assert [n["pitch"] for n in out] == [p for _, _, p in SPEC]
    for n, (start, _, _) in zip(out, SPEC):
        assert abs(n["start"] - start) < 0.012, (n["start"], start)
        assert n["end"] > n["start"] and n["flag"] is None and n["plucked"]


def test_repeated_pitch_is_split_only_on_a_pluck():
    held = [(0.5, 1.4, 40)]  # one long note; the tracker blips mid-note
    y = _stem(held, 2.5)
    feat = _features(y, held)
    feat["per_c"][100:103] = 0.02  # a 30 ms dropout in periodicity must not create a second note
    feat["onsets"] = np.append(feat["onsets"], 110)  # nor may a spurious onset
    out, _, _ = N.segment_notes(feat, y)
    assert len(out) == 1 and out[0]["end"] - out[0]["start"] > 1.0
    replucked = [(0.5, 0.65, 40), (1.2, 0.7, 40)]
    y2 = _stem(replucked, 2.5)
    out2, _, _ = N.segment_notes(_features(y2, replucked), y2)
    assert [n["pitch"] for n in out2] == [40, 40]


def test_notes_hold_until_the_next_note():
    spec = [(0.5, 0.5, 40), (1.0, 0.5, 43)]
    y = _stem(spec, 2.0)
    feat = _features(y, spec)
    feat["per_c"][85:100] = 0.02  # the tracker loses the first note's tail 150 ms early
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [40, 43]
    assert abs(out[0]["end"] - out[1]["start"]) < 0.03  # the note still holds to the next one: no gap


def test_octave_low_tracking_is_corrected():
    spec = [(0.5, 0.8, 40)]
    y = _stem(spec, 2.0)
    feat = _features(y, spec)
    feat["f0_c"][:] = np.where(feat["per_c"] > 0.5, librosa.midi_to_hz(28), feat["f0_c"])  # tracker an octave low
    feat["f0_p"] = feat["f0_c"].copy()
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [40] and out[0]["flag"] == "octave"


def test_refine_onsets_moves_starts_to_the_pluck():
    y = _stem(SPEC, 4.5)
    late = [dict(start=s + 0.03, end=s + d, pitch=p) for s, d, p in SPEC]
    N.refine_onsets(late, y, N.SegParams())
    for n, (start, _, _) in zip(late, SPEC):
        assert n["plucked"] and abs(n["start"] - start) < 0.006
    for a, b in zip(late, late[1:]):
        assert a["end"] <= b["start"] + 1e-6


def test_refine_onsets_leaves_legato_changes_alone():
    t = np.arange(int(2.0 * SR)) / SR  # a steady tone that changes pitch without a new attack
    y = (0.3 * np.where(t < 1.0, np.sin(2 * np.pi * 82.4 * t), np.sin(2 * np.pi * 98.0 * t))).astype(np.float32)
    notes = [dict(start=0.5, end=1.0, pitch=40), dict(start=1.0, end=1.9, pitch=43)]
    N.refine_onsets(notes, y, N.SegParams())
    assert notes[1]["start"] == 1.0 and not notes[1]["plucked"]


def test_attack_rise_finds_the_jump():
    low = np.r_[np.full(50, -40.0), np.full(50, -20.0)]
    rise = N.attack_rise(low, 2, 3)
    assert rise[49] == 20 and rise[10] == 0 and rise[80] == 0


def test_repeated_notes_the_tracker_sees_as_one_are_split_at_each_pluck():
    plucks = [(0.5, 0.24, 40), (0.75, 0.24, 40), (1.0, 0.24, 40), (1.25, 0.5, 40)]  # eighth notes on one pitch
    y = _stem(plucks, 2.5)
    feat = _features(y, [(0.5, 1.25, 40)])  # ...which the pitch tracker reports as one long E
    feat["onsets"] = feat["onsets"][:1]  # ...and whose re-plucks the onset detector missed
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [40, 40, 40, 40]
    for n, (start, _, _) in zip(out, plucks):
        assert abs(n["start"] - start) < 0.02
    unsplit, _, _ = N.segment_notes(feat, y, N.SegParams(split_held=False))
    assert len(unsplit) == 1


def test_a_cut_right_after_a_notes_own_attack_is_not_a_split():
    spec = [(0.5, 1.0, 40)]
    y = _stem(spec, 2.0)
    out, _, _ = N.segment_notes(_features(y, spec, lag_frames=5), y)  # the tracker locks on 50 ms after the pluck
    assert len(out) == 1


def test_a_level_jump_alone_keeps_two_notes_the_tracker_saw_apart():
    # a soft re-pluck: the level jumps but the attack is dull, so there is little spectral flux. The tracker saw
    # two notes (its pitch confidence dipped at the pluck); they must stay two.
    spec = [(0.5, 0.6, 40), (1.1, 0.8, 40)]
    y = _stem(spec, 2.5)
    feat = _features(y, spec)
    out, _, _ = N.segment_notes(feat, y, N.SegParams(flux_support=1e9, flux_thr=1e9))  # flux can prove nothing
    assert [n["pitch"] for n in out] == [40, 40]


def test_an_octave_jump_onto_a_ringing_low_note_keeps_its_octave():
    """E1 rings on while E2 is played over it. The smoothing second tracker still reports the low note; the
    confident first tracker's octave stands, and the note is not folded back into the E1 before it."""
    spec = [(0.5, 1.5, 28), (1.0, 0.5, 40)]  # the E1 sounds through the E2
    y = _stem(spec, 2.5)
    feat = _features(y, [(0.5, 0.5, 28), (1.0, 0.5, 40), (1.5, 0.5, 28)])
    feat["f0_p"][:] = np.where(feat["per_c"] > 0.5, librosa.midi_to_hz(28), feat["f0_p"])  # pYIN never leaves E1
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out][:2] == [28, 40] and out[1]["flag"] is None
    before, _, _ = N.segment_notes(feat, y, N.SegParams(trust_per=9.0))  # the old rule: the lower note wins
    assert 40 not in [n["pitch"] for n in before]


def test_an_unsure_tracker_still_defers_to_the_spectrum_on_the_octave():
    spec = [(0.5, 0.8, 28)]
    y = _stem(spec, 2.0)
    feat = _features(y, spec)
    feat["per_c"][:] = np.where(feat["per_c"] > 0.5, 0.47, feat["per_c"])  # pitched, but not confidently
    feat["f0_c"][:] = np.where(feat["per_c"] > 0.4, librosa.midi_to_hz(40), feat["f0_c"])  # an octave high
    out, _, _ = N.segment_notes(feat, y)
    assert [(n["pitch"], n["flag"]) for n in out] == [(28, "octave")]


def test_the_slide_between_two_notes_is_not_a_note():
    spec = [(0.5, 0.5, 40), (1.0, 0.5, 37)]  # E2 then C#2
    y = _stem(spec, 2.0)
    heard = [(0.5, 0.4, 40), (0.9, 0.1, 39), (1.0, 0.5, 37)]  # the tracker reads Eb2 for 100 ms on the way down
    feat = _features(y, heard, lag_frames=0)
    feat["onsets"] = np.array([50, 100])
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [40, 37] and abs(out[0]["end"] - out[1]["start"]) < 0.03
    kept, _, _ = N.segment_notes(feat, y, N.SegParams(glide_frames=0))
    assert [n["pitch"] for n in kept] == [40, 39, 37]


def test_a_short_plucked_note_a_step_away_is_kept():
    spec = [(0.5, 0.5, 40), (1.0, 0.1, 39), (1.1, 0.5, 37)]  # here the Eb2 is played: it has its own pluck
    y = _stem(spec, 2.0)
    out, _, _ = N.segment_notes(_features(y, spec), y)
    assert [n["pitch"] for n in out] == [40, 39, 37]


def test_a_slide_longer_than_a_slide_is_a_note():
    spec = [(0.5, 0.5, 40), (1.2, 0.5, 37)]
    y = _stem(spec, 2.0)
    heard = [(0.5, 0.5, 40), (1.0, 0.2, 39), (1.2, 0.5, 37)]  # 200 ms on Eb2 with no pluck: a hammered note
    feat = _features(y, heard, lag_frames=0)
    feat["onsets"] = np.array([50, 120])
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [40, 39, 37]


def test_the_thump_of_a_pluck_is_the_start_of_its_note():
    spec = [(0.5, 0.45, 37), (1.0, 0.5, 40)]
    y = _stem(spec, 2.0)
    heard = [(0.5, 0.45, 37), (1.0, 0.07, 34), (1.07, 0.43, 40)]  # 70 ms of Bb1 at the pluck of the E2
    feat = _features(y, heard, lag_frames=0)
    feat["onsets"] = np.array([50, 100])
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [37, 40] and abs(out[1]["start"] - 1.0) < 0.02
    kept, _, _ = N.segment_notes(feat, y, N.SegParams(thump_frames=0))
    assert [n["pitch"] for n in kept] == [37, 34, 40]


def test_the_pitch_sinking_as_a_note_dies_away_is_not_a_note():
    spec = [(0.5, 0.4, 37), (1.0, 0.5, 40)]  # C#2, let go at 0.9 s, then E2
    y = _stem(spec, 2.0)
    heard = [(0.5, 0.4, 37), (0.9, 0.08, 34), (1.0, 0.5, 40)]  # 80 ms of Bb1 in the silence after the C#2
    feat = _features(y, heard, lag_frames=0)
    feat["onsets"] = np.array([50, 100])
    feat["rms_db"][90:98] = feat["rms_db"][55:85].mean() - 12  # the sound there is the C#2 fading
    out, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in out] == [37, 40]
    kept, _, _ = N.segment_notes(feat, y, N.SegParams(tail_db=0.0))
    assert [n["pitch"] for n in kept] == [37, 34, 40]
    feat["rms_db"][90:98] = feat["rms_db"][55:85].mean()  # as loud as the note before: a note of its own
    loud, _, _ = N.segment_notes(feat, y)
    assert [n["pitch"] for n in loud] == [37, 34, 40]


def test_what_is_heard_below_the_instrument_is_written_an_octave_up():
    spec = [(0.5, 0.45, 38), (1.0, 0.45, 40), (1.5, 0.45, 28)]  # D2, E2, E1
    y = _stem(spec, 2.5)
    heard = [(0.5, 0.45, 38), (1.0, 0.45, 26), (1.5, 0.45, 28)]  # the tracker follows a D1 under the second note
    feat = _features(y, heard)
    feat["f0_p"] = feat["f0_c"].copy()
    out, _, _ = N.segment_notes(feat, y, N.SegParams(together=False))
    assert [n["pitch"] for n in out] == [38, 38, 28]  # D1 is under a four-string bass: written as D2
    five_string, _, _ = N.segment_notes(feat, y, N.SegParams(together=False, lowest=23))
    assert [n["pitch"] for n in five_string] == [38, 26, 28]
    as_heard, _, _ = N.segment_notes(feat, y, N.SegParams(together=False, lowest=0))
    assert [n["pitch"] for n in as_heard] == [38, 26, 28]


def _partials(amps, f0, start, dur, seconds):
    """A note built from its partials: `amps[k]` is the amplitude of partial k + 1."""
    t = np.arange(int(seconds * SR)) / SR
    env = (
        ((t >= start) & (t < start + dur))
        * np.exp(-np.maximum(t - start, 0) * 2.0)
        * np.minimum(1, (t - start) / 0.005)
    )
    return sum(a * np.sin(2 * np.pi * (k + 1) * f0 * t) for k, a in enumerate(amps)) * np.clip(env, 0, 1)


def _octave_case(under):
    """An E2 with something an octave under it; CREPE is sure of the lower octave, pYIN of the upper."""
    e1, e2 = librosa.midi_to_hz(28), librosa.midi_to_hz(40)
    y = (_partials([0.4, 0.2, 0.13, 0.1], e2, 0.5, 0.8, 2.0) + _partials(under, e1, 0.5, 0.8, 2.0)).astype(np.float32)
    feat = _features(y, [(0.5, 0.8, 28)])
    feat["f0_p"] = np.where(feat["per_c"] > 0.5, e2, feat["f0_p"])
    return feat, y


def test_a_faint_octave_under_a_note_is_not_the_note():
    feat, y = _octave_case([0.08])  # 14 dB under the E2 and nothing else: an effect, or the note before ringing
    out, _, _ = N.segment_notes(feat, y)
    assert [(n["pitch"], n["flag"]) for n in out] == [(40, "octave")]
    as_it_was, _, _ = N.segment_notes(feat, y, N.SegParams(trust_low=True))
    assert [n["pitch"] for n in as_it_was] == [28]


def test_two_octaves_sounding_together_are_written_as_the_lower_one():
    feat, y = _octave_case([0.1, 0.0, 0.08])  # the lower note is soft too, but it has its own third partial
    out, _, _ = N.segment_notes(feat, y)
    assert [(n["pitch"], n["flag"]) for n in out] == [(28, None)]
    feat, y = _octave_case([0.3])  # ...or it is loud
    out, _, _ = N.segment_notes(feat, y)
    assert [(n["pitch"], n["flag"]) for n in out] == [(28, None)]
