import json
import os

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from bass2sheet import evaluate as ev

REF = [(1.0 + 0.5 * k, 1.4 + 0.5 * k, 40 + (k % 5)) for k in range(16)]


def test_perfect_transcription_scores_one():
    m = ev.score(REF, list(REF))
    assert m["f"] == m["precision"] == m["recall"] == m["f_with_offsets"] == m["f_note_name"] == 1.0
    assert m["ref_errors"]["ok"] == m["est_errors"]["ok"] == 1.0
    assert abs(m["onset_bias_ms"]) < 1e-6 and m["pitch_same"] == 1.0


def test_onset_tolerance_and_bias():
    late = [(s + 0.03, e + 0.03, p) for s, e, p in REF]
    m = ev.score(REF, late)
    assert m["f"] == 1.0 and abs(m["onset_bias_ms"] - 30) < 1
    very_late = [(s + 0.08, e + 0.08, p) for s, e, p in REF]
    m = ev.score(REF, very_late)
    assert m["f"] == 0.0 and m["f_100ms"] == 1.0 and abs(m["onset_bias_ms"] - 80) < 1


def test_octave_errors_are_counted_apart_from_wrong_notes():
    est = [(s, e, p + 12 if k % 2 else p) for k, (s, e, p) in enumerate(REF)]
    m = ev.score(REF, est)
    assert m["f"] == 0.5 and m["f_note_name"] == 1.0 and m["pitch_octave_up"] == 0.5
    assert m["ref_errors"]["octave"] == 0.5 and m["est_errors"]["octave"] == 0.5
    est = [(s, e, p + 3 if k == 0 else p) for k, (s, e, p) in enumerate(REF)]
    m = ev.score(REF, est)
    assert m["ref_errors"]["wrong_pitch"] == pytest.approx(1 / 16) and m["pitch_other"] == pytest.approx(1 / 16)


def test_breakdown_names_merged_split_missed_and_ghost_notes():
    ref = [(1.0, 1.5, 40), (1.5, 2.0, 40), (3.0, 3.5, 43), (5.0, 5.5, 45)]
    est = [
        (1.0, 2.0, 40),  # the repeated E merged into one note
        (3.0, 3.5, 43),
        (3.25, 3.5, 43),  # a re-attack that is not there
        (8.0, 8.4, 38),
    ]  # nothing sounds here
    m = ev.score(ref, est)
    assert m["ref_errors"]["merged"] == 0.25 and m["ref_errors"]["missed"] == 0.25
    assert m["est_errors"]["split"] == 0.25 and m["est_errors"]["ghost"] == 0.25


def test_empty_transcription_scores_zero():
    m = ev.score(REF, [])
    assert m["f"] == 0.0 and m["ref_notes"] == 16 and m["est_notes"] == 0


def test_overlap_share():
    assert ev.overlap_share([(0, 2, 40), (1, 3, 47)]) == pytest.approx(1 / 3)
    assert ev.overlap_share([(0, 1, 40), (1, 2, 47)]) == 0.0


def test_rhythm_positions_against_the_reference_grid():
    beats = 1.0 + 0.5 * np.arange(40)
    grid = dict(beats=beats, first_downbeat_beat=0, four_four=True)
    ref = [(1.0 + 0.25 * k, 1.2 + 0.25 * k, 40) for k in range(32)]  # eighth notes from the first downbeat
    right = [(2 * k, k // 8 + 1) for k in range(32)]
    m = ev.score(ref, ref, right, grid, set())
    assert m["beat_position_acc"] == m["bar_position_acc"] == 1.0 and m["grid_shift_mode"] == 0
    assert m["ref_tempo"] == pytest.approx(120.0)
    half_bar_off = [(g + 8, bar) for g, bar in right]
    m = ev.score(ref, ref, half_bar_off, grid, set())
    assert m["beat_position_acc"] == 1.0 and m["bar_position_acc"] == 0.0 and m["grid_shift_mode"] == 8
    one_sixteenth_off = [(g + 1, bar) for g, bar in right]
    assert ev.score(ref, ref, one_sixteenth_off, grid, set())["beat_position_acc"] == 0.0


def test_precision_inside_and_outside_check_bars():
    ref = [(1.0, 1.4, 40), (2.0, 2.4, 40), (3.0, 3.4, 40), (4.0, 4.4, 40)]
    est = [(1.0, 1.4, 40), (2.0, 2.4, 40), (3.0, 3.4, 47), (4.0, 4.4, 47)]  # bar 2 is wrong
    m = ev.score(ref, est, [(0, 1), (8, 1), (16, 2), (24, 2)], None, {2})
    assert m["precision_in_check_bars"] == 0.0 and m["precision_in_other_bars"] == 1.0 and m["check_bar_share"] == 0.5


@pytest.fixture()
def slakh_dir(tmp_path):
    """A one-track dataset in Slakh's layout whose bass MIDI is written an octave above the audio, as Slakh's is."""
    sr = 16000
    track = tmp_path / "Track00001"
    (track / "MIDI").mkdir(parents=True)
    (track / "stems").mkdir()
    sounding = [(0.5 + 0.5 * k, 0.9 + 0.5 * k, 36 + (k % 4) * 2) for k in range(12)]
    y = np.zeros(int(8 * sr))
    for s, e, p in sounding:
        t = np.arange(int((e - s) * sr)) / sr
        y[int(s * sr) : int(s * sr) + len(t)] += (
            0.4 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * t) * np.exp(-2 * t)
        )
    sf.write(str(track / "stems" / "S00.wav"), y, sr)
    sf.write(str(track / "stems" / "S01.wav"), np.zeros_like(y), sr)
    sf.write(str(track / "mix.wav"), y, sr)
    for name, shift, drum in (("S00", 12, False), ("S01", 0, True)):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=33, is_drum=drum)
        inst.notes = [pretty_midi.Note(100, p + shift, s, e) for s, e, p in sounding]
        pm.instruments.append(inst)
        pm.write(str(track / "MIDI" / f"{name}.mid"))
    pm.write(str(track / "all_src.mid"))
    (track / "metadata.yaml").write_text(
        "stems:\n  S00:\n    inst_class: Bass\n    is_drum: false\n    plugin_name: test_bass.nkm\n"
        "    program_num: 33\n"
        "    audio_rendered: false\n  S01:\n    inst_class: Drums\n    is_drum: true\n    plugin_name: kit.nkm\n"
        "    program_num: 128\n"
    )
    (tmp_path / "Track00002").mkdir()  # no metadata: ignored
    return str(tmp_path), sounding


def test_slakh_adapter_finds_bass_and_corrects_the_octave(slakh_dir, tmp_path):
    root, sounding = slakh_dir
    tracks = ev.slakh_tracks(root)
    assert [t["id"] for t in tracks] == ["Track00001"] and [b["id"] for b in tracks[0]["bass"]] == ["S00"]
    cache = str(tmp_path / "octave.json")
    ref = ev.reference(tracks[0], cache)
    notes, grid, shifts = ref["notes"], ref["grid"], ref["shifts"]
    assert ref["f0"] is None and shifts["S00"][0] == -12 and shifts["S00"][1] > 0.9
    assert [p for _, _, p in notes] == [p for _, _, p in sounding]
    assert grid["four_four"] and np.allclose(np.diff(grid["beats"]), 0.5)
    assert json.load(open(cache))["S00"][0] == -12  # measured once, then cached
    assert ev.reference(tracks[0], cache)["shifts"] == shifts
    assert ev.load_dataset(root)[0] == "slakh"


def _row(track, cond, f, verdict="PASS", gate=0.97):
    return dict(
        track=track,
        condition=cond,
        status="ok",
        verdict=verdict,
        patch="p",
        ref_notes=10,
        est_notes=10,
        f=f,
        score=f,
        precision=f,
        recall=f,
        f_note_name=f,
        f_with_offsets=f - 0.1,
        f_onset_only=f,
        f_100ms=f,
        pitch_octave_up=0.0,
        pitch_octave_down=0.0,
        pitch_same=1.0,
        pitch_other=0.0,
        onset_bias_ms=5.0,
        beat_position_acc=1.0,
        bar_position_acc=1.0,
        tempo_ratio=1.0,
        duration_ratio=1.0,
        check_bar_share=0.1,
        precision_in_check_bars=0.5,
        precision_in_other_bars=f,
        gates={"coverage": gate},
        gate_status={"coverage": "pass" if gate >= 0.95 else "warn"},
        ref_errors=dict(ok=f, missed=1 - f),
        est_errors=dict(ok=f, ghost=1 - f),
    )


def test_summarize_reports_conditions_verdicts_and_gates():
    rows = [_row(f"T{k}", "mix", 0.5 + 0.08 * k, "PASS" if k > 2 else "WARN", 0.90 + 0.02 * k) for k in range(6)]
    summary, md = ev.summarize(rows)
    assert summary["conditions"]["mix"]["f"]["mean"] == pytest.approx(0.7)
    assert (
        summary["headline"] == "F"
        and summary["verdicts"]["PASS"]["runs"] == 3
        and summary["verdicts"]["WARN"]["mean_f"] < summary["verdicts"]["PASS"]["mean_f"]
    )
    assert summary["gates"]["coverage"]["rho"] == pytest.approx(1.0)
    assert "## Condition: mix" in md and "## Where the errors are" in md and "Spearman" in md


def test_baseline_check_flags_a_regression_only_beyond_the_allowed_drop():
    base = ev.baseline_of([_row("T1", "mix", 0.90), _row("T2", "mix", 0.80)])
    assert ev.check_baseline([_row("T1", "mix", 0.89), _row("T2", "mix", 0.80)], base, 0.02) == []
    assert (
        ev.check_baseline([_row("T1", "mix", 0.95), _row("T2", "mix", 0.90), _row("T3", "mix", 0.1)], base, 0.02) == []
    )
    problems = ev.check_baseline([_row("T1", "mix", 0.80), _row("T2", "mix", 0.80)], base, 0.02)
    assert problems and "mean f fell from 0.850 to 0.800" in problems[0]
    assert ev.check_baseline([_row("T9", "mix", 0.9)], base, 0.02) == ["mix: no tracks in common with the baseline"]
    assert ev.check_baseline([_row("T1", "stem", 0.1)], base, 0.02) == []  # the baseline has no stem condition
    both = ev.baseline_of([_row("T1", "mix", 0.9), _row("T1", "stem", 0.9)])
    assert ev.check_baseline([_row("T1", "mix", 0.9)], both, 0.02) == []  # ...and a run may leave a condition out


def test_f0_scores_reward_right_notes_and_expose_octave_errors():
    t = np.arange(0, 4, 0.0029)
    hz = np.where(
        (t >= 1) & (t < 2), 82.41, np.where((t >= 2) & (t < 3), 110.0 * 1.01, 0.0)
    )  # E2, then a slightly sharp A2
    right = ev.score_f0(t, hz, [(1.0, 2.0, 40), (2.0, 3.0, 45)])
    assert (
        right["oa"] > 0.98 and right["rpa"] > 0.98 and right["note_acc"] > 0.98 and right["voicing_false_alarm"] < 0.02
    )
    octave = ev.score_f0(t, hz, [(1.0, 2.0, 52), (2.0, 3.0, 45)])
    assert (
        octave["rpa"] == pytest.approx(0.5, abs=0.02)
        and octave["rca"] > 0.98
        and octave["octave_err"] == pytest.approx(0.5, abs=0.02)
    )
    silent = ev.score_f0(t, hz, [])
    assert silent["voicing_recall"] == 0.0 and silent["rpa"] == 0.0 and silent["oa"] == pytest.approx(0.5, abs=0.02)
    always_on = ev.score_f0(t, hz, [(0.0, 4.0, 40)])
    assert always_on["voicing_false_alarm"] > 0.98


def test_mdb_adapter_and_f0_summary(tmp_path):
    for d in ("audio_mix", "audio_bass", "annotation_bass"):
        (tmp_path / d).mkdir()
    sf.write(str(tmp_path / "audio_mix" / "Song_MIX_basssynth.wav"), np.zeros(8000), 8000)
    sf.write(str(tmp_path / "audio_bass" / "Song_STEM_02.RESYN.wav"), np.zeros(8000), 8000)
    (tmp_path / "annotation_bass" / "Song_STEM_02.RESYN.csv").write_text("0.0,0.0\n0.0029,82.4\n0.0058,82.4\n")
    kind, tracks = ev.load_dataset(str(tmp_path))
    assert (
        kind == "mdb" and [t["id"] for t in tracks] == ["Song"] and tracks[0]["bass"][0]["audio"].endswith("RESYN.wav")
    )
    ref = ev.reference(tracks[0])
    assert ref["notes"] is None and ref["f0"][1].tolist() == [0.0, 82.4, 82.4]
    rows = [
        dict(
            track=f"S{k}",
            condition="mix",
            status="ok",
            verdict="PASS",
            patch="resynthesized",
            est_notes=10,
            oa=0.6 + 0.05 * k,
            rpa=0.7,
            rca=0.8,
            note_acc=0.75,
            octave_err=0.05,
            voicing_recall=0.9,
            voicing_false_alarm=0.1,
            score=0.6 + 0.05 * k,
            gates={"coverage": 0.9 + 0.01 * k},
            gate_status={"coverage": "pass"},
        )
        for k in range(6)
    ]
    summary, md = ev.summarize(rows)
    assert summary["headline"] == "OA" and summary["conditions"]["mix"]["oa"]["mean"] == pytest.approx(0.725)
    assert "| OA |" in md and "Where the errors are" not in md
    assert ev.check_baseline(rows, ev.baseline_of(rows), 0.02) == []


def test_unknown_dataset_is_refused(tmp_path):
    from bass2sheet.common import Bass2SheetError

    with pytest.raises(Bass2SheetError, match="not a dataset"):
        ev.load_dataset(str(tmp_path))
    assert ev.main([str(tmp_path), "--out", str(tmp_path / "run")]) == 1


def test_no_run_scores_what_exists_and_merges_rows(slakh_dir, tmp_path, capsys):
    root, sounding = slakh_dir
    run = tmp_path / "run"
    assert ev.main([root, "--out", str(run), "--no-run", "--conditions", "mix"]) == 1  # nothing there yet
    out = run / "Track00001" / "mix"
    out.mkdir(parents=True)
    notes = [dict(start=s, end=e, pitch=p, grid_start=k * 8, bar=k // 2 + 1) for k, (s, e, p) in enumerate(sounding)]
    (out / "notes.json").write_text(json.dumps(dict(notes=notes, tempo_bpm=120.0)))
    gate = dict(name="coverage", value=1.0, status="pass")
    (out / "report.json").write_text(
        json.dumps(dict(verdict="PASS", ear_checks=[], check_severity=3, gates=[gate], summary=dict(separator="given")))
    )
    (run / "results.jsonl").write_text(json.dumps(_row("Other", "mix", 0.5)) + "\n")  # a row from another process
    assert ev.main([root, "--out", str(run), "--no-run", "--conditions", "mix"]) == 0
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert [(r["track"], r["f"]) for r in rows] == [("Other", 0.5), ("Track00001", 1.0)]
    assert "mean F 0.750" in capsys.readouterr().out


def _the_tool(monkeypatch, sounding, wrong=(), fails=()):
    """A stand-in for the pipeline: it writes the notes that sound (those of the tracks in `wrong` a fourth too
    high) and a report, as a run does, or fails for the tracks in `fails`. Returns the list of what it was asked."""
    from bass2sheet import pipeline
    from bass2sheet.common import Bass2SheetError

    asked = []

    def run(audio_path, out_dir, **kw):
        asked.append(dict(kw, audio=audio_path, out=out_dir))
        if kw["title"] in fails:
            raise Bass2SheetError("only 3 bass notes found; the stem has no clearly pitched bass")
        up = 5 if kw["title"] in wrong else 0
        notes = [
            dict(start=s, end=e, pitch=p + up, grid_start=8 * k, bar=k // 2 + 1) for k, (s, e, p) in enumerate(sounding)
        ]
        with open(os.path.join(out_dir, "notes.json"), "w") as f:
            json.dump(dict(notes=notes, tempo_bpm=120.0), f)
        gate = dict(name="coverage", value=1.0, status="pass")
        summary = dict(separator="roformer" + (" + given bass stem" if kw["bass_stem"] else ""))
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(dict(verdict="PASS", ear_checks=[], check_severity=3, gates=[gate], summary=summary), f)
        return 0

    monkeypatch.setattr(pipeline, "run", run)
    return asked


def _more_tracks(root, *names):
    import shutil

    for name in names:
        shutil.copytree(os.path.join(root, "Track00001"), os.path.join(root, name))


def test_a_run_transcribes_every_track_scores_it_and_is_held_against_a_baseline(
    slakh_dir, tmp_path, monkeypatch, capsys
):
    root, sounding = slakh_dir
    _more_tracks(root, "Track00003", "Track00004")
    asked = _the_tool(monkeypatch, sounding)
    run, baseline, cache = str(tmp_path / "run"), str(tmp_path / "baseline.json"), str(tmp_path / "cache")
    assert ev.main([root, "--out", run, "--cache-dir", cache, "--write-baseline", baseline]) == 0
    assert [(os.path.basename(os.path.dirname(a["out"])), os.path.basename(a["out"])) for a in asked] == [
        (track, cond) for track in ("Track00001", "Track00003", "Track00004") for cond in ("mix", "stem")
    ]
    mix, stem = asked[0], asked[1]
    assert mix["audio"].endswith("Track00001/mix.wav") and mix["bass_stem"] is None and mix["cache_dir"] == cache
    assert stem["bass_stem"].endswith("Track00001/stems/S00.wav")  # the dataset's clean bass, in place of the separated
    assert (mix["lowest"], mix["notes"], mix["renders"], mix["pdf"]) == (0, None, False, False)
    rows = [json.loads(line) for line in open(os.path.join(run, "results.jsonl"))]
    assert len(rows) == 6 and all(r["f"] == 1.0 and r["status"] == "ok" and r["seconds"] >= 0 for r in rows)
    assert all(r["ref_octave_shift"] == {"S00": -12} for r in rows)  # the MIDI is written an octave up: allowed for
    said = capsys.readouterr().out
    assert (
        "[3/3] Track00004 stem PASS  F 1.00" in said
        and "mix: mean F 1.000, median 1.000, min 1.000 (3 tracks, 0 errors)" in said
    )
    assert sorted(json.load(open(baseline))["conditions"]) == ["mix", "stem"]
    assert "| mix |" in open(os.path.join(run, "summary.md")).read()

    # run again: what is there is scored and nothing is transcribed twice, unless it is forced
    del asked[:]
    assert ev.main([root, "--out", run, "--conditions", "mix", "--baseline", baseline]) == 0
    assert asked == [] and "no regression against the baseline" in capsys.readouterr().out
    assert ev.main([root, "--out", run, "--conditions", "mix", "--tracks", "Track00003", "--force"]) == 0
    assert [a["title"] for a in asked] == ["Track00003"]

    # the tool gets worse on one track of three: beyond the allowed drop that is a regression, exit 3
    _the_tool(monkeypatch, sounding, wrong={"Track00004"})
    code = ev.main(
        [root, "--out", run, "--conditions", "mix", "--force", "--baseline", baseline, "--notes", "muscriptor"]
    )
    out, err = capsys.readouterr()
    assert code == 3 and "REGRESSION mix: mean f fell from 1.000 to 0.667 over 3 tracks (allowed drop 0.02)" in err
    assert "[3/3] Track00004 mix  PASS  F 0.00" in out
    assert ev.main([root, "--out", run, "--conditions", "mix", "--baseline", baseline, "--max-drop", "0.4"]) == 0


def test_a_track_the_tool_cannot_transcribe_counts_as_a_score_of_zero(slakh_dir, tmp_path, monkeypatch, capsys):
    root, sounding = slakh_dir
    _more_tracks(root, "Track00003")
    _the_tool(monkeypatch, sounding, fails={"Track00003"})
    run = str(tmp_path / "run")
    assert ev.main([root, "--out", run, "--conditions", "mix", "--cache-dir", str(tmp_path)]) == 0
    good, bad = (json.loads(line) for line in open(os.path.join(run, "results.jsonl")))
    assert (bad["status"], bad["verdict"], bad["f"], bad["score"]) == ("error", "ERROR", 0.0, 0.0)
    assert bad["error"].startswith("only 3 bass notes found") and good["f"] == 1.0
    said = capsys.readouterr().out
    assert "Track00003 mix  ERROR F 0.00" in said and "only 3 bass notes found" in said
    assert "mix: mean F 0.500, median 0.500, min 0.000 (2 tracks, 1 errors)" in said
    asked = _the_tool(monkeypatch, sounding)
    assert ev.main([root, "--out", run, "--conditions", "mix"]) == 0  # a failure is a result too: not tried again
    assert asked == []


def test_tracks_are_chosen_by_name_by_sample_or_by_count(slakh_dir, tmp_path, monkeypatch, capsys):
    root, sounding = slakh_dir
    _more_tracks(root, *(f"Track{k:05d}" for k in range(3, 8)))  # six tracks: 1 and 3 to 7
    asked = _the_tool(monkeypatch, sounding)
    go = lambda *more: ev.main([root, "--out", str(tmp_path / "run"), "--conditions", "mix", "--force", *more])
    chosen = lambda: [a["title"] for a in asked]
    assert go("--sample", "3") == 0 and chosen() == ["Track00001", "Track00004", "Track00007"]  # evenly through it
    del asked[:]
    assert go("--limit", "2") == 0 and chosen() == ["Track00001", "Track00003"]
    del asked[:]
    assert go("--tracks", "Track00005,Track00009") == 0 and chosen() == ["Track00005"]
    capsys.readouterr()
    assert go("--tracks", "Track00009") == 1
    assert capsys.readouterr().err == f"bass2sheet-eval: no usable tracks under {root}\n"


def test_two_bass_parts_are_left_out_unless_asked_for_and_then_their_sum_is_the_clean_bass(
    slakh_dir, tmp_path, monkeypatch, capsys
):
    import shutil

    root, sounding = slakh_dir
    track = os.path.join(root, "Track00001")
    shutil.copy(os.path.join(track, "stems", "S00.wav"), os.path.join(track, "stems", "S02.wav"))
    shutil.copy(os.path.join(track, "MIDI", "S00.mid"), os.path.join(track, "MIDI", "S02.mid"))
    with open(os.path.join(track, "metadata.yaml"), "a") as f:
        f.write(
            "  S02:\n    inst_class: Bass\n    is_drum: false\n    plugin_name: second_bass.nkm\n    program_num: 34\n"
        )
    asked = _the_tool(monkeypatch, sounding)
    run = str(tmp_path / "run")
    assert ev.main([root, "--out", run, "--conditions", "stem"]) == 1  # two basses at once: outside the tool's scope
    assert "no usable tracks" in capsys.readouterr().err and asked == []
    assert ev.main([root, "--out", run, "--conditions", "stem", "--multi-bass"]) == 0
    given = asked[0]["bass_stem"]
    assert given == os.path.join(run, "Track00001", "stem", "bass_stems_sum.wav")
    both, sr = sf.read(given)
    one, _ = sf.read(os.path.join(track, "stems", "S00.wav"))
    assert sr == 16000 and np.allclose(both, 2 * one, atol=1e-6)
    row = json.loads(open(os.path.join(run, "results.jsonl")).read())
    assert row["bass_stems"] == 2 and row["patch"] == "test_bass.nkm, second_bass.nkm"


def test_rwc_adapter_takes_the_songs_with_one_bass_part_inside_the_annotated_span(tmp_path):
    audio, ann = tmp_path / "RWC-P" / "disc1", tmp_path / "annotations"
    pre = ann / "01_annotations_preprocessed"
    for folder in (audio, pre / "MIDI_aligned" / "RWC-P", pre / "beats" / "RWC-P"):
        folder.mkdir(parents=True)
    rows = ["RWCID;Title;Artist;GenreSub;Tempo;LiveInstruments;DrumInformation;audio_start;audio_end"]
    for k, basses in ((1, ["Bass"]), (2, ["E.Bass", "Synth Bass"]), (3, ["Bass"]), (4, [])):
        rid = f"RWC_P{k:03d}"
        sf.write(str(audio / f"{rid}.wav"), np.zeros(1000), 16000)
        rows.append(f"{rid};Song {k};Singer {k};Pop;120;Bass, Drums;Live drums;2.0;9.5")
        if k == 3:
            continue  # a song whose MIDI is not in the annotations
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        melody = pretty_midi.Instrument(program=0, name="Piano")
        melody.notes = [pretty_midi.Note(90, 72, 1.0, 2.0)]
        pm.instruments.append(melody)
        for b, name in enumerate(basses):
            part = pretty_midi.Instrument(program=33 if name == "Bass" else 0, name="" if name == "Bass" else name)
            # one note before the annotated span, eight in it, one after, and one too short to be a note
            starts = [0.5, *(2.0 + 0.5 * j for j in range(8)), 9.8]
            part.notes = [pretty_midi.Note(100, 40 + b + j % 3, s, s + 0.4) for j, s in enumerate(starts)]
            part.notes.append(pretty_midi.Note(100, 52, 5.1, 5.105))
            pm.instruments.append(part)
        pm.write(str(pre / "MIDI_aligned" / "RWC-P" / f"{rid}.mid"))
    (ann / "metadata.csv").write_text("\n".join(rows) + "\n")
    beats = ["beat;number"] + [f"{2.0 + 0.5 * j:.3f};{(j + 2) % 4 + 1}" for j in range(16)]  # the song enters on beat 3
    (pre / "beats" / "RWC-P" / "RWC_P001.csv").write_text("\n".join(beats) + "\n")

    kind, tracks = ev.load_dataset(str(tmp_path / "RWC-P"), str(ann))
    assert kind == "rwc" and [t["id"] for t in tracks] == ["RWC_P001"]  # two bass parts, no MIDI, no bass: left out
    (track,) = tracks
    assert track["midi_track"] == 1 and track["span"] == (2.0, 9.5) and track["bass"][0]["audio"] is None
    assert track["about"] == dict(
        title="Song 1", artist="Singer 1", genre="Pop", tempo="120", live="Bass, Drums", drums="Live drums"
    )
    ref = ev.reference(track)
    assert [round(s, 2) for s, _, _ in ref["notes"]] == [2.0 + 0.5 * j for j in range(8)]  # the annotated part only
    assert ref["f0"] is None and ref["shifts"] == {}
    assert ref["grid"]["first_downbeat_beat"] == 2 and ref["grid"]["four_four"] and len(ref["grid"]["beats"]) == 16
    os.remove(track["beats_csv"])
    (track,) = ev.load_dataset(str(tmp_path / "RWC-P"), str(ann))[1]
    assert track["beats_csv"] is None and ev.reference(track)["grid"] is None  # no beats: the notes are still scored
