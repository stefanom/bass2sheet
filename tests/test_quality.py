import numpy as np
import pytest

from bass2sheet import quality as Q


@pytest.mark.parametrize("value,higher,expected", [
    (0.96, True, "pass"), (0.90, True, "warn"), (0.80, True, "fail"), (None, True, "skipped"),
    (0.04, False, "pass"), (0.10, False, "warn"), (0.20, False, "fail"),
])
def test_gate_judge(value, higher, expected):
    warn, fail = (0.95, 0.85) if higher else (0.05, 0.15)
    assert Q.Gate("g", "G", value, warn, fail, higher, "").judge().status == expected


def test_gate_thresholds_are_inclusive_on_the_passing_side():
    assert Q.Gate("g", "G", 0.95, 0.95, 0.85, True, "").judge().status == "pass"
    assert Q.Gate("g", "G", 0.05, 0.05, 0.15, False, "").judge().status == "pass"


def test_verdict_is_the_worst_gate():
    g = lambda status: Q.Gate("g", "G", 1.0, 0, 0, True, "", status=status)
    assert Q.verdict([g("pass"), g("skipped")]) == "PASS"
    assert Q.verdict([g("pass"), g("warn")]) == "WARN"
    assert Q.verdict([g("warn"), g("fail"), g("pass")]) == "FAIL"


def test_stem_gate_tells_bass_from_no_bass(song):
    mix = song["mix"].mean(0)
    assert Q.verdict(Q.stem_gates(song["stems"], mix)) == "PASS"
    silent = dict(song["stems"], bass=np.zeros_like(song["stems"]["bass"]))
    gates = Q.stem_gates(silent, mix)
    assert Q.verdict(gates) == "FAIL" and gates[0].name == "bass_level"


def test_check_list_orders_by_severity_and_dedupes_reasons():
    grid = np.arange(0, 60, 0.125)
    checks = {3: dict(severity=1, reasons=["a"]), 7: dict(severity=4, reasons=["b", "b", "c"]), 5: dict(severity=4, reasons=["d"])}
    rows = Q.check_list(checks, grid, origin=8)
    assert [r["bar"] for r in rows] == [5, 7, 3]
    assert rows[1]["reasons"] == ["b", "c"]
    assert rows[0]["time"] == "0:09.0"  # bar 5 starts 4 bars of 2 s after the 1 s origin


def test_octave_doubts_reads_the_spectrum():
    C = np.full((72, 400), 1e-4)
    note = lambda pitch: dict(start=1.0, end=2.0, pitch=pitch, beat=1, sixteenth=1, bar=1)
    C[40 - 21, 100:200] = 1.0  # a clear E2 with its octave
    C[52 - 21, 100:200] = 0.5
    assert Q.octave_doubts(C, [note(40)]) == []
    assert [why for _, why in Q.octave_doubts(C, [note(52)])] == ["may be an octave too high"]  # energy an octave below
    assert [why for _, why in Q.octave_doubts(C, [note(28)])] == ["may be an octave too low"]  # no energy at E1 itself


def test_transcription_gates_on_a_perfect_and_a_broken_transcription(song):
    import librosa
    from bass2sheet import render, rhythm
    from bass2sheet.common import HOP, SR
    from conftest import note

    stems = song["stems"]
    mono = stems["bass"].mean(0)
    feat = dict(rms_db=librosa.amplitude_to_db(librosa.feature.rms(y=mono, frame_length=2048, hop_length=HOP)[0] + 1e-9),
                cqt=np.abs(librosa.cqt(mono, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72, bins_per_octave=12)))
    grid = np.arange(-1.0, 40, 0.125)
    origin = 16  # the song's first downbeat is at 1.0 s

    def gates_for(truth):
        notes = rhythm.quantize([note(s, e, p) for s, e, p in truth], grid, origin)
        gates, checks = Q.transcription_gates(stems, feat, notes, grid, origin, render.synth(notes, len(mono)), None, "off")
        return {g.name: g for g in gates}, checks

    good, checks = gates_for(song["notes"])
    assert Q.verdict([g for g in good.values() if g.name != "check_bars"]) == "PASS"
    assert good["second_model_pitch"].status == "skipped" and good["second_model_pitch"].note == "off"
    assert not [b for b, c in checks.items() if c["severity"] >= Q.CHECK_SEVERITY]

    wrong_pitch, _ = gates_for([(s, e, p + 3) for s, e, p in song["notes"]])
    assert wrong_pitch["audio_pitch"].status == "fail"
    sparse, checks = gates_for(song["notes"][::4])  # three quarters of the notes missing
    assert sparse["coverage"].status == "fail" and sparse["audio_gaps"].status == "fail"
    assert any("notes missing" in r for c in checks.values() for r in c["reasons"])
    shifted, _ = gates_for([(s + 0.06, e + 0.06, p) for s, e, p in song["notes"]])  # half a 16th late
    assert shifted["grid_offset_ms"].status == "fail"
