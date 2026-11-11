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
        y[i:i + len(w)] += w[:len(y) - i]
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
    for n, (start, dur, _) in zip(out, SPEC):
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
