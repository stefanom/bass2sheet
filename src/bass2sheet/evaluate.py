"""Evaluate bass2sheet against ground truth: run the tool on each track of a dataset and score what it wrote
against the dataset's bass annotation.

    bass2sheet-eval DATASET_DIR --out RUN_DIR [--conditions mix,stem] [--tracks ID,ID] [--sample N] [--force]
                    [--annotations DIR] [--baseline FILE] [--write-baseline FILE]

Datasets (detected from the directory):
  Slakh2100 layout   TrackXXXXX/{mix.*, metadata.yaml, all_src.mid, MIDI/Sxx.mid, stems/Sxx.*}; BabySlakh and
                     Slakh2100-redux. Exact notes, and a beat grid from the MIDI.
  MDB-bass-synth     audio_mix/, audio_bass/, annotation_bass/. Real mixes with a resynthesized bass and its exact
                     f0 curve: no note list, so the scores are frame-level (mir_eval.melody).
  RWC Popular        a directory of RWC_Pxxx.wav plus --annotations <rwc-annotations checkout>. Real recordings
                     with MIDI aligned at the beat level, and beat annotations. No clean stem.

Conditions:
  mix   the full mix in, as a user would run it
  stem  the same, but with the dataset's clean bass stem in place of the separated bass: the difference from
        `mix` is what separation costs (datasets with a clean stem only)

Runs are resumable: a track/condition that already has notes.json is scored without being re-run.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
import warnings

import numpy as np

from .common import Bass2SheetError, atomic_write_text

ONSET_TOL = 0.05


# ---------------------------------------------------------------- datasets


def slakh_tracks(root):
    """Tracks of a Slakh-layout dataset that have a rendered bass stem."""
    import yaml

    out = []
    for d in sorted(glob.glob(os.path.join(root, "Track*"))):
        meta_path = os.path.join(d, "metadata.yaml")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        find = lambda base: next((p for p in (base + ext for ext in (".wav", ".flac")) if os.path.exists(p)), None)
        bass = []
        for sid, st in (meta.get("stems") or {}).items():
            # the files decide: BabySlakh's audio_rendered/midi_saved flags are false even for stems it ships
            if st.get("inst_class") == "Bass" and not st.get("is_drum"):
                audio, midi = find(os.path.join(d, "stems", sid)), os.path.join(d, "MIDI", sid + ".mid")
                if audio and os.path.exists(midi):
                    bass.append(
                        dict(
                            id=sid,
                            audio=audio,
                            midi=midi,
                            patch=st.get("plugin_name") or st.get("midi_program_name"),
                            program=st.get("program_num"),
                        )
                    )
        mix = find(os.path.join(d, "mix"))
        if bass and mix:
            out.append(dict(id=os.path.basename(d), dir=d, mix=mix, bass=bass, all_midi=os.path.join(d, "all_src.mid")))
    return out


def mdb_tracks(root):
    """MDB-bass-synth: one mix, one resynthesized bass stem and one f0 annotation per song."""
    out = []
    for mix in sorted(glob.glob(os.path.join(root, "audio_mix", "*_MIX_basssynth.wav"))):
        name = os.path.basename(mix)[: -len("_MIX_basssynth.wav")]
        ann = sorted(glob.glob(os.path.join(root, "annotation_bass", name + "_STEM_*.RESYN.csv")))
        if len(ann) != 1:
            continue
        stem = os.path.join(root, "audio_bass", os.path.basename(ann[0])[:-4] + ".wav")
        if os.path.exists(stem):
            out.append(
                dict(
                    id=name,
                    kind="mdb",
                    mix=mix,
                    f0_csv=ann[0],
                    bass=[dict(id="bass", audio=stem, patch="resynthesized")],
                )
            )
    return out


def rwc_tracks(audio_root, annotations):
    """RWC Popular: RWC_Pxxx.wav with the aligned MIDI and beats of the rwc-annotations repository. Only songs
    whose MIDI has exactly one bass track are used (found by track name or a General MIDI bass program)."""
    import csv

    import pretty_midi

    pre = os.path.join(annotations, "01_annotations_preprocessed")
    meta = {}
    with open(os.path.join(annotations, "metadata.csv"), newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            meta[r["RWCID"]] = r
    wavs = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(audio_root, "**", "RWC_P*.wav"), recursive=True)
    }
    out = []
    for rid, wav in sorted(wavs.items()):
        midi = os.path.join(pre, "MIDI_aligned", "RWC-P", rid + ".mid")
        beats = os.path.join(pre, "beats", "RWC-P", rid + ".csv")
        if not (os.path.exists(midi) and rid in meta):
            continue
        pm = pretty_midi.PrettyMIDI(midi)
        bass = [
            k
            for k, i in enumerate(pm.instruments)
            if not i.is_drum and i.notes and ("bass" in (i.name or "").lower() or 32 <= i.program <= 39)
        ]
        if len(bass) != 1:
            continue
        m = meta[rid]
        out.append(
            dict(
                id=rid,
                kind="rwc",
                mix=wav,
                midi=midi,
                midi_track=bass[0],
                beats_csv=beats if os.path.exists(beats) else None,
                span=(float(m["audio_start"]), float(m["audio_end"])),
                bass=[dict(id="bass", audio=None, patch=(pm.instruments[bass[0]].name or "").strip())],
            )
        )
    return out


def load_dataset(root, annotations=None):
    """(kind, tracks) for a dataset directory."""
    if os.path.isdir(os.path.join(root, "audio_mix")) and os.path.isdir(os.path.join(root, "annotation_bass")):
        return "mdb", mdb_tracks(root)
    if glob.glob(os.path.join(root, "Track*", "metadata.yaml")):
        return "slakh", slakh_tracks(root)
    if annotations:
        return "rwc", rwc_tracks(root, annotations)
    raise Bass2SheetError(
        f"{root!r} is not a dataset bass2sheet-eval knows (Slakh layout, MDB-bass-synth, or RWC "
        "Popular audio with --annotations)"
    )


def octave_shift(audio_path, notes):
    """How far the stem sounds from its MIDI, as a whole number of octaves in semitones, measured with pYIN on
    the clean stem. Slakh's bass patches sound an octave below their MIDI notes, so the MIDI alone is not the
    sounding pitch. Returns (shift, share of measured notes that agree)."""
    import librosa
    import soundfile as sf

    y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    y = y.mean(1)
    hop = sr // 100
    f0, _, _ = librosa.pyin(y, fmin=28, fmax=600, sr=sr, frame_length=2048 * max(1, sr // 16000), hop_length=hop)
    t = np.arange(len(f0)) * hop / sr
    diffs = []
    for s, e, p in notes:
        if e - s < 0.08:
            continue
        seg = f0[(t >= s + 0.03) & (t < min(e, s + 0.4))]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            diffs.append(int(round(librosa.hz_to_midi(np.median(seg)) - p)))
    if not diffs:
        return 0, 0.0
    counts = {k: diffs.count(k) for k in (-24, -12, 0, 12)}
    best = max(counts, key=counts.get)
    return (best, counts[best] / len(diffs)) if counts[best] / len(diffs) >= 0.6 else (0, counts[0] / len(diffs))


def reference(track, cache=None):
    """The ground truth of a track as dict(notes, grid, f0, shifts):
    notes  [(start, end, sounding pitch)] or None when the dataset has no note list
    grid   dict(beats, first_downbeat_beat, four_four) or None
    f0     (times, Hz with 0 = unvoiced) or None
    shifts octave corrections applied to the MIDI per bass stem, {stem: [semitones, agreement]}; `cache` is a
           JSON path that stores them."""
    kind = track.get("kind", "slakh")
    if kind == "mdb":
        d = np.loadtxt(track["f0_csv"], delimiter=",")
        return dict(notes=None, grid=None, f0=(d[:, 0], d[:, 1]), shifts={})
    if kind == "rwc":
        return _rwc_reference(track)
    return _slakh_reference(track, cache)


def _rwc_reference(track):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(track["midi"])
    t0, t1 = track["span"]
    notes = sorted(
        (n.start, n.end, n.pitch)
        for n in pm.instruments[track["midi_track"]].notes
        if n.end - n.start >= 0.01 and t0 <= n.start <= t1
    )
    grid = None
    if track["beats_csv"]:
        rows = [line.strip().split(";") for line in open(track["beats_csv"]) if line.strip() and line[0].isdigit()]
        beats = np.array([float(r[0]) for r in rows])
        nums = [int(float(r[1])) for r in rows]
        if len(beats) > 8 and 1 in nums:
            grid = dict(beats=beats, first_downbeat_beat=nums.index(1), four_four=max(nums) == 4)
    return dict(notes=notes, grid=grid, f0=None, shifts={})


def _slakh_reference(track, cache=None):
    import pretty_midi

    shifts = {}
    if cache and os.path.exists(cache):
        shifts = json.load(open(cache))
    notes = []
    for b in track["bass"]:
        pm = pretty_midi.PrettyMIDI(b["midi"])
        raw = sorted(
            (n.start, n.end, n.pitch)
            for inst in pm.instruments
            if not inst.is_drum
            for n in inst.notes
            if n.end - n.start >= 0.01
        )
        if b["id"] not in shifts:
            shifts[b["id"]] = list(octave_shift(b["audio"], raw))
        notes += [(s, e, p + shifts[b["id"]][0]) for s, e, p in raw]
    if cache:
        atomic_write_text(cache, json.dumps(shifts))
    notes.sort()
    grid = None
    if os.path.exists(track["all_midi"]):
        pm = pretty_midi.PrettyMIDI(track["all_midi"])
        ts = pm.time_signature_changes
        # no time signature at all means the file was recorded without a click: its beats say nothing about the music
        four_four = bool(ts) and all(t.numerator == 4 and t.denominator == 4 for t in ts)
        beats, downbeats = pm.get_beats(), pm.get_downbeats()
        if len(beats) > 8 and len(downbeats):
            grid = dict(
                beats=beats, first_downbeat_beat=int(np.argmin(np.abs(beats - downbeats[0]))), four_four=four_four
            )
    return dict(notes=notes, grid=grid, f0=None, shifts=shifts)


# ---------------------------------------------------------------- metrics


def _arrays(notes, fold=False):
    import pretty_midi

    iv = np.array([[s, e] for s, e, _ in notes], dtype=float).reshape(-1, 2)
    hz = np.array([pretty_midi.note_number_to_hz((p % 12) + 60 if fold else p) for _, _, p in notes], dtype=float)
    return iv, hz


def overlap_share(notes):
    """Share of the reference's note time during which another reference note also sounds."""
    if not notes:
        return 0.0
    ev = sorted([(s, 1) for s, _, _ in notes] + [(e, -1) for _, e, _ in notes])
    active, last, over, total = 0, ev[0][0], 0.0, 0.0
    for t, d in ev:
        if active >= 1:
            total += t - last
        if active >= 2:
            over += t - last
        active, last = active + d, t
    return over / total if total else 0.0


def breakdown(ref, est, good):
    """What happened to every reference note and every estimated note, as shares.

    Reference notes: ok (onset within 50 ms, right pitch) | octave | wrong_pitch (onset found, pitch not) |
    merged (it starts inside one of our notes of the same pitch: a repeated note we did not split) |
    covered_other (it starts inside one of our notes of another pitch) | timing (we have it 50-150 ms off) | missed.
    Our notes: ok | octave | wrong_pitch | split (it starts inside a held reference note of the same pitch: a
    re-attack that is not there) | timing | ghost (nothing sounds in the reference) | extra_other."""
    import mir_eval

    ri, ei = _arrays(ref)[0], _arrays(est)[0]
    r_ok, e_ok = {i for i, _ in good}, {j for _, j in good}
    r_left, e_left = [i for i in range(len(ref)) if i not in r_ok], [j for j in range(len(est)) if j not in e_ok]
    rc = dict(ok=len(r_ok), octave=0, wrong_pitch=0, merged=0, covered_other=0, timing=0, missed=0)
    ec = dict(ok=len(e_ok), octave=0, wrong_pitch=0, split=0, timing=0, ghost=0, extra_other=0)
    r_done, e_done = set(), set()
    if r_left and e_left:
        for a, b in mir_eval.transcription.match_note_onsets(ri[r_left], ei[e_left], onset_tolerance=ONSET_TOL):
            i, j = r_left[a], e_left[b]
            kind = "octave" if abs(est[j][2] - ref[i][2]) in (12, 24) else "wrong_pitch"
            rc[kind] += 1
            ec[kind] += 1
            r_done.add(i)
            e_done.add(j)
    for i in r_left:
        if i in r_done:
            continue
        t, p = ref[i][0], ref[i][2]
        inside = [e for e in est if e[0] < t - ONSET_TOL and e[1] > t + 0.01]
        if any(e[2] == p for e in inside):
            rc["merged"] += 1
        elif inside:
            rc["covered_other"] += 1
        elif any(abs(e[0] - t) <= 0.15 and e[2] == p for e in est):
            rc["timing"] += 1
        else:
            rc["missed"] += 1
    for j in e_left:
        if j in e_done:
            continue
        t, p = est[j][0], est[j][2]
        inside = [r for r in ref if r[0] < t - ONSET_TOL and r[1] > t + 0.01]
        if any(r[2] == p for r in inside):
            ec["split"] += 1
        elif any(abs(r[0] - t) <= 0.15 and r[2] == p for r in ref):
            ec["timing"] += 1
        elif not inside and not any(abs(r[0] - t) <= 0.15 for r in ref):
            ec["ghost"] += 1
        else:
            ec["extra_other"] += 1
    return ({k: v / len(ref) for k, v in rc.items()}, {k: v / len(est) for k, v in ec.items()})


def score(ref, est, est_grid=None, ref_grid=None, check_bars=()):
    """Note and rhythm metrics. `ref`/`est` are [(start, end, pitch)]; `est_grid` is [(grid_start, bar)] per
    estimated note; `ref_grid` comes from reference(); `check_bars` are bars the tool marked for an ear check."""
    import mir_eval

    m = dict(ref_notes=len(ref), est_notes=len(est), ref_overlap=overlap_share(ref))
    if not ref or not est:
        m.update(f=0.0, precision=0.0, recall=0.0)
        return m
    T = mir_eval.transcription
    (ri, rp), (ei, ep) = _arrays(ref), _arrays(est)
    (_, rpf), (_, epf) = _arrays(ref, True), _arrays(est, True)
    P, R, F, _ = T.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=None)
    m.update(precision=P, recall=R, f=F)
    m["f_with_offsets"] = T.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=0.2)[2]
    m["f_note_name"] = T.precision_recall_f1_overlap(ri, rpf, ei, epf, onset_tolerance=ONSET_TOL, offset_ratio=None)[2]
    m["f_100ms"] = T.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=0.1, offset_ratio=None)[2]
    m["f_onset_only"] = T.onset_precision_recall_f1(ri, ei, onset_tolerance=ONSET_TOL)[2]
    # among notes whose onsets match: how do the pitches differ?
    pairs = T.match_note_onsets(ri, ei, onset_tolerance=ONSET_TOL)
    if pairs:
        d = np.array([est[j][2] - ref[i][2] for i, j in pairs])
        m.update(
            pitch_same=float(np.mean(d == 0)),
            pitch_octave_up=float(np.mean(d == 12)),
            pitch_octave_down=float(np.mean(d == -12)),
            pitch_other=float(np.mean((d != 0) & (np.abs(d) != 12))),
        )
        dur = np.array([(est[j][1] - est[j][0]) / max(1e-3, ref[i][1] - ref[i][0]) for i, j in pairs])
        m["duration_ratio"] = float(np.median(dur))
        # the commonest whole-octave difference: a reference transcribed by ear may use another octave convention
        octs = {k: int(np.sum(d == k)) for k in (-24, -12, 0, 12, 24)}
        mode = max(octs, key=octs.get)
        m.update(octave_mode=mode, octave_mode_share=octs[mode] / len(d))
        if mode:
            shifted = [(a, b, p - mode) for a, b, p in est]
            m["f_octave_aligned"] = T.precision_recall_f1_overlap(
                ri, rp, ei, _arrays(shifted)[1], onset_tolerance=ONSET_TOL, offset_ratio=None
            )[2]
        else:
            m["f_octave_aligned"] = F
    wide = T.match_note_onsets(ri, ei, onset_tolerance=0.1)
    if wide:
        b = np.array([est[j][0] - ref[i][0] for i, j in wide]) * 1000
        m.update(onset_bias_ms=float(np.median(b)), onset_spread_ms=float(np.percentile(b, 75) - np.percentile(b, 25)))
    # matches on onset and pitch, for the rhythm and ear-check analyses
    good = T.match_notes(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=None)
    matched_est = {j for _, j in good}
    m["ref_errors"], m["est_errors"] = breakdown(ref, est, good)
    if est_grid and check_bars is not None:
        in_check = np.array([bar in check_bars for _, bar in est_grid])
        hit = np.array([j in matched_est for j in range(len(est))])
        m["check_bar_share"] = float(in_check.mean())
        m["precision_in_check_bars"] = float(hit[in_check].mean()) if in_check.any() else None
        m["precision_in_other_bars"] = float(hit[~in_check].mean()) if (~in_check).any() else None
    if est_grid and ref_grid and ref_grid["four_four"] and good:
        beats, k0 = ref_grid["beats"], ref_grid["first_downbeat_beat"]

        def ref_sixteenth(t):  # global 16th index on the reference grid
            i = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
            return int(round((i + (t - beats[i]) / (beats[i + 1] - beats[i])) * 4))

        g_ref = np.array([ref_sixteenth(ref[i][0]) - 4 * k0 for i, _ in good])
        g_est = np.array([est_grid[j][0] for _, j in good])
        shift = (g_est - g_ref) % 16
        m["beat_position_acc"] = float(np.mean(shift % 4 == 0))  # same 16th within the beat
        m["bar_position_acc"] = float(np.mean(shift == 0))  # same 16th within the bar
        counts = np.bincount(shift, minlength=16)
        m["grid_shift_mode"] = int(counts.argmax())  # the commonest shift, in 16ths (0 = grid agrees)
        m["grid_shift_mode_share"] = float(counts.max() / counts.sum())
        span = [beats[i] for i in range(len(beats)) if ref[0][0] <= beats[i] <= ref[-1][1]] or list(beats)
        m["ref_tempo"] = float(60 / np.median(np.diff(span))) if len(span) > 2 else None
    return m


def score_f0(ref_t, ref_hz, est):
    """Frame-level scores against an f0 curve (mir_eval.melody), with our notes held at their pitch.

    Standard measures at 50 cents: voicing recall and false alarm, raw pitch accuracy (RPA), raw chroma accuracy
    (RCA), overall accuracy (OA). Our notes are whole semitones while a played bass drifts and slides, so `note_acc`
    also gives the share of voiced frames where we have the nearest semitone, and `octave_err` the share where we
    are exactly an octave or two off."""
    import mir_eval
    import pretty_midi

    est_hz = np.zeros(len(ref_t))
    est_midi = np.full(len(ref_t), -1)
    for s, e, p in est:
        sel = (ref_t >= s) & (ref_t < e)
        est_hz[sel], est_midi[sel] = pretty_midi.note_number_to_hz(p), p
    M = mir_eval.melody
    rv, rc, ev_, ec = M.to_cent_voicing(ref_t, ref_hz, ref_t, est_hz)
    vr, vfa = M.voicing_measures(rv, ev_)
    voiced = ref_hz > 0
    ref_midi = np.round(69 + 12 * np.log2(np.maximum(ref_hz, 1e-6) / 440)).astype(int)
    both = voiced & (est_midi >= 0)
    d = (est_midi - ref_midi)[both]
    return dict(
        oa=float(M.overall_accuracy(rv, rc, ev_, ec)),
        rpa=float(M.raw_pitch_accuracy(rv, rc, ev_, ec)),
        rca=float(M.raw_chroma_accuracy(rv, rc, ev_, ec)),
        voicing_recall=float(vr),
        voicing_false_alarm=float(vfa),
        note_acc=float(np.sum(d == 0) / max(1, voiced.sum())),
        octave_err=float(np.sum((d != 0) & (d % 12 == 0)) / max(1, voiced.sum())),
        ref_voiced_share=float(voiced.mean()),
        est_notes=len(est),
    )


# ---------------------------------------------------------------- runner


def run_one(track, cond, out_dir, force, cache_dir):
    """Run the tool for one track and condition (unless already done) and score it."""
    from . import pipeline

    notes_path, report_path = os.path.join(out_dir, "notes.json"), os.path.join(out_dir, "report.json")
    row = dict(
        track=track["id"],
        condition=cond,
        kind=track.get("kind", "slakh"),
        patch=", ".join(str(b["patch"]) for b in track["bass"]),
        bass_stems=len(track["bass"]),
    )
    err_path = os.path.join(out_dir, "error.txt")
    os.makedirs(out_dir, exist_ok=True)
    if force or not (os.path.exists(notes_path) and os.path.exists(report_path) or os.path.exists(err_path)):
        for p in (notes_path, report_path, err_path):
            if os.path.exists(p):
                os.remove(p)
        stem = None
        if cond == "stem":
            stem = track["bass"][0]["audio"]
            if len(track["bass"]) > 1:  # several bass stems: their sum is the clean bass
                import soundfile as sf

                parts = [sf.read(b["audio"], always_2d=True, dtype="float32") for b in track["bass"]]
                n = min(len(y) for y, _ in parts)
                stem = os.path.join(out_dir, "bass_stems_sum.wav")
                sf.write(stem, sum(y[:n] for y, _ in parts), parts[0][1], subtype="FLOAT")
        t0 = time.time()
        try:
            pipeline.run(
                track["mix"],
                out_dir,
                title=track["id"],
                artist="",
                renders=False,
                pdf=False,
                bass_stem=stem,
                cache_dir=cache_dir,
            )
        except Bass2SheetError as e:
            atomic_write_text(err_path, str(e))
        row["seconds"] = time.time() - t0
    ref = reference(track, os.path.join(os.path.dirname(out_dir), "reference_octave.json"))
    if ref["shifts"]:
        row["ref_octave_shift"] = {k: v[0] for k, v in ref["shifts"].items()}
        row["ref_octave_agreement"] = min(v[1] for v in ref["shifts"].values())
    if os.path.exists(err_path):
        row.update(status="error", error=open(err_path).read().strip(), verdict="ERROR")
        row.update(score(ref["notes"], []) if ref["notes"] is not None else score_f0(*ref["f0"], []))
    else:
        data, rep = json.load(open(notes_path)), json.load(open(report_path))
        est = [(n["start"], n["end"], n["pitch"]) for n in data["notes"]]
        est_grid = [(n["grid_start"], n["bar"]) for n in data["notes"]]
        check_bars = {c["bar"] for c in rep["ear_checks"] if c["severity"] >= rep["check_severity"]}
        row.update(
            status="ok",
            verdict=rep["verdict"],
            separator=rep["summary"]["separator"],
            est_tempo=data["tempo_bpm"],
            gates={g["name"]: g["value"] for g in rep["gates"]},
            gate_status={g["name"]: g["status"] for g in rep["gates"]},
        )
        if ref["notes"] is not None:
            row.update(score(ref["notes"], est, est_grid, ref["grid"], check_bars))
            if row.get("ref_tempo"):
                row["tempo_ratio"] = data["tempo_bpm"] / row["ref_tempo"]
        else:
            row.update(score_f0(*ref["f0"], est))
    row["score"] = row["f"] if ref["notes"] is not None else row["oa"]  # the headline number of this dataset
    return row


# ---------------------------------------------------------------- summary

NOTE_COLS = [
    ("f", "F"),
    ("precision", "P"),
    ("recall", "R"),
    ("f_note_name", "F name"),
    ("f_octave_aligned", "F 8va-fit"),
    ("f_100ms", "F 100ms"),
    ("f_with_offsets", "F+off"),
    ("pitch_octave_up", "8va↑"),
    ("pitch_octave_down", "8va↓"),
    ("onset_bias_ms", "bias ms"),
    ("beat_position_acc", "beat pos"),
    ("bar_position_acc", "bar pos"),
    ("tempo_ratio", "tempo×"),
]
F0_COLS = [
    ("oa", "OA"),
    ("rpa", "RPA"),
    ("rca", "RCA"),
    ("note_acc", "note acc"),
    ("octave_err", "8va err"),
    ("voicing_recall", "voicing recall"),
    ("voicing_false_alarm", "voicing FA"),
]
EXTRA = [
    "precision_in_check_bars",
    "precision_in_other_bars",
    "check_bar_share",
    "duration_ratio",
    "pitch_same",
    "pitch_other",
    "f_onset_only",
    "score",
]
LEGEND = {
    "note": "F: note F-measure, onset within 50 ms and the exact pitch. F name: the same ignoring the octave. "
    "F 8va-fit: F "
    "after moving our notes by the track's commonest whole-octave difference from the reference (for references "
    "that use another octave convention). F 100ms: onsets within 100 ms. F+off: also requiring the note end within "
    "20% of the note's length. 8va↑/↓: share of onset-matched notes an octave above/below the reference. bias ms: "
    "median of our onset minus the reference's. beat pos / bar pos: share of correct notes on the right 16th of "
    "the beat / of the bar. tempo×: our tempo over the reference's.",
    "f0": "Frame-level scores against the bass's f0 curve. OA: overall accuracy (voicing and pitch within 50 cents). "
    "RPA: "
    "raw pitch accuracy on voiced frames. RCA: the same ignoring the octave. note acc: share of voiced frames where "
    "we have the nearest semitone. 8va err: share where we are a whole number of octaves off. voicing recall / FA: "
    "share of voiced frames we cover with a note / of unvoiced frames we cover.",
}


def _fmt(v, key):
    if v is None:
        return "–"
    return f"{v:+.0f}" if key == "onset_bias_ms" else f"{v:.2f}"


def summarize(rows):
    """Aggregate numbers plus a Markdown report."""
    from scipy.stats import spearmanr

    f0 = any("oa" in r for r in rows)
    cols, head = (F0_COLS, "OA") if f0 else (NOTE_COLS, "F")
    score_of = lambda r: r.get("score", r.get("f"))
    out = dict(conditions={}, verdicts={}, gates={}, headline=head)
    L = ["# bass2sheet evaluation", ""]
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        agg = {}
        for key in [k for k, _ in cols] + EXTRA:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                agg[key] = dict(mean=float(np.mean(vals)), median=float(np.median(vals)), min=float(np.min(vals)))
        agg["tracks"] = len(rs)
        agg["errors"] = sum(r["status"] != "ok" for r in rs)
        out["conditions"][cond] = agg
        counts = "" if f0 else " ref notes |"
        L += [
            f"## Condition: {cond} ({len(rs)} tracks, {agg['errors']} errors)",
            "",
            "| track | verdict | " + " | ".join(h for _, h in cols) + f" |{counts} est notes | bass |",
            "|---|---|" + "---|" * (len(cols) + (2 if f0 else 3)),
        ]
        for r in sorted(rs, key=lambda r: r["track"]):
            n_ref = "" if f0 else f" {r['ref_notes']} |"
            L.append(
                f"| {r['track']} | {r['verdict']} | "
                + " | ".join(_fmt(r.get(k), k) for k, _ in cols)
                + f" |{n_ref} {r['est_notes']} | {r['patch']} |"
            )
        pad = " | |" + ("" if f0 else " |") + " |"
        L.append("| **mean** | | " + " | ".join(_fmt(agg.get(k, {}).get("mean"), k) for k, _ in cols) + pad)
        L.append("| **median** | | " + " | ".join(_fmt(agg.get(k, {}).get("median"), k) for k, _ in cols) + pad)
        L.append("")
    if any(r.get("ref_errors") for r in rows):
        L += [
            "## Where the errors are",
            "",
            "Mean share of the reference's notes (what we did with each) and of our notes (what each is).",
            "",
        ]
        for side, title in (("ref_errors", "reference notes"), ("est_errors", "our notes")):
            keys = list(next((r[side] for r in rows if r.get(side)), {}))
            L += [f"| {title} | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
            for cond in sorted({r["condition"] for r in rows}):
                rs = [r[side] for r in rows if r["condition"] == cond and r.get(side)]
                mean = {k: float(np.mean([x[k] for x in rs])) for k in keys}
                out["conditions"][cond][side] = mean
                L.append(f"| {cond} | " + " | ".join(f"{mean[k]:.1%}" for k in keys) + " |")
            L.append("")
    ok = [r for r in rows if r["status"] == "ok"]
    L += [
        "## Do the quality gates predict real accuracy?",
        "",
        f"| verdict | runs | mean {head} | min {head} |",
        "|---|---|---|---|",
    ]
    for v in ("PASS", "WARN", "FAIL", "ERROR"):
        fs = [score_of(r) for r in rows if r["verdict"] == v]
        if fs:
            out["verdicts"][v] = dict(runs=len(fs), mean_f=float(np.mean(fs)), min_f=float(np.min(fs)))
            L.append(f"| {v} | {len(fs)} | {np.mean(fs):.2f} | {np.min(fs):.2f} |")
    L += [
        "",
        f"| gate | Spearman ρ with {head} | ρ with bar pos | runs warn/fail | mean {head} when pass | "
        f"mean {head} when warn/fail |",
        "|---|---|---|---|---|---|",
    ]

    def rank_corr(pts):
        if len(pts) < 5 or len({a for a, _ in pts}) < 2 or len({b for _, b in pts}) < 2:
            return None
        rho = spearmanr([a for a, _ in pts], [b for _, b in pts])[0]
        return None if np.isnan(rho) else float(rho)

    show = lambda v, spec: "–" if v is None else format(v, spec)
    for name in ok[0]["gates"] if ok else {}:
        pts = [(r["gates"][name], score_of(r)) for r in ok if r["gates"].get(name) is not None]
        bars = [
            (r["gates"][name], r["bar_position_acc"])
            for r in ok
            if r["gates"].get(name) is not None and r.get("bar_position_acc") is not None
        ]
        bad = [score_of(r) for r in ok if r["gate_status"].get(name) in ("warn", "fail")]
        good = [score_of(r) for r in ok if r["gate_status"].get(name) == "pass"]
        out["gates"][name] = dict(
            rho=rank_corr(pts),
            rho_bar_position=rank_corr(bars),
            flagged=len(bad),
            f_pass=float(np.mean(good)) if good else None,
            f_flagged=float(np.mean(bad)) if bad else None,
        )
        g = out["gates"][name]
        L.append(
            f"| {name} | {show(g['rho'], '+.2f')} | {show(g['rho_bar_position'], '+.2f')} | {len(bad)} | "
            f"{show(g['f_pass'], '.2f')} | {show(g['f_flagged'], '.2f')} |"
        )
    pc = [
        (r["precision_in_check_bars"], r["precision_in_other_bars"])
        for r in ok
        if r.get("precision_in_check_bars") is not None and r.get("precision_in_other_bars") is not None
    ]
    if pc:
        out["check_bars"] = dict(
            precision_in_check_bars=float(np.mean([a for a, _ in pc])),
            precision_in_other_bars=float(np.mean([b for _, b in pc])),
            runs=len(pc),
        )
        L += [
            "",
            f'Notes in bars marked "check" are right {out["check_bars"]["precision_in_check_bars"]:.0%} of the time; '
            f"notes in other bars {out['check_bars']['precision_in_other_bars']:.0%} "
            f"({len(pc)} runs with both kinds of bar).",
        ]
    L += ["", LEGEND["f0" if f0 else "note"]]
    return out, "\n".join(L) + "\n"


BASELINE_KEYS = ("f", "f_note_name", "f_with_offsets", "bar_position_acc", "oa", "note_acc")


def baseline_of(rows):
    """What a later run is compared with: per-track scores by condition."""
    out = {}
    for r in rows:
        out.setdefault(r["condition"], {})[r["track"]] = {k: r[k] for k in BASELINE_KEYS if r.get(k) is not None}
    return dict(conditions=out)


def check_baseline(rows, baseline, max_drop):
    """Regressions against a baseline: for each condition and score, the mean over the tracks both runs share may
    not fall by more than `max_drop`. Returns a list of messages (empty = no regression)."""
    problems = []
    now = baseline_of(rows)["conditions"]
    for cond, base_tracks in baseline.get("conditions", {}).items():
        shared = sorted(set(base_tracks) & set(now.get(cond, {})))
        if not shared:
            problems.append(f"{cond}: no tracks in common with the baseline")
            continue
        for key in BASELINE_KEYS:
            pairs = [(base_tracks[t].get(key), now[cond][t].get(key)) for t in shared]
            pairs = [(b, n) for b, n in pairs if b is not None and n is not None]
            if not pairs:
                continue
            b, n = float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))
            if n < b - max_drop:
                problems.append(
                    f"{cond}: mean {key} fell from {b:.3f} to {n:.3f} over {len(pairs)} tracks "
                    f"(allowed drop {max_drop})"
                )
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bass2sheet-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", help="dataset directory (Slakh layout, MDB-bass-synth, or RWC Popular audio)")
    ap.add_argument("--out", required=True, help="run directory (per-track outputs, results.jsonl, summary.md/json)")
    ap.add_argument("--annotations", help="for RWC Popular: a checkout of github.com/rwc-music/rwc-annotations")
    ap.add_argument(
        "--conditions", default="mix,stem", help="mix, stem or both (stem only where the dataset has clean stems)"
    )
    ap.add_argument("--tracks", help="comma-separated track ids (default: all usable tracks)")
    ap.add_argument("--sample", type=int, help="use this many tracks, evenly spaced through the dataset")
    ap.add_argument("--limit", type=int, help="use the first N tracks")
    ap.add_argument("--force", action="store_true", help="re-run tracks that already have results")
    ap.add_argument(
        "--multi-bass",
        action="store_true",
        help="include tracks with more than one bass stem (two bass parts at once are outside the tool's scope)",
    )
    ap.add_argument("--cache-dir", default="~/.cache/bass2sheet")
    ap.add_argument("--baseline", help="JSON written by --write-baseline: exit 3 if the scores regress against it")
    ap.add_argument("--max-drop", type=float, default=0.02, help="allowed fall of a mean score against the baseline")
    ap.add_argument("--write-baseline", help="write this run's per-track scores as a baseline JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for noisy in ("audio_separator", "numba", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        kind, tracks = load_dataset(args.dataset, args.annotations)
    except Bass2SheetError as e:
        print(f"bass2sheet-eval: {e}", file=sys.stderr)
        return 1
    tracks = [t for t in tracks if args.multi_bass or len(t["bass"]) == 1]
    if args.tracks:
        want = set(args.tracks.split(","))
        tracks = [t for t in tracks if t["id"] in want]
    if args.sample and args.sample < len(tracks):
        tracks = [tracks[int(round(k))] for k in np.linspace(0, len(tracks) - 1, args.sample)]
    tracks = tracks[: args.limit] if args.limit else tracks
    if not tracks:
        print(f"bass2sheet-eval: no usable tracks under {args.dataset}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    conds = [c for c in args.conditions.split(",") if c == "mix" or tracks[0]["bass"][0]["audio"]]
    rows = []
    for k, t in enumerate(tracks):
        for cond in conds:
            row = run_one(t, cond, os.path.join(args.out, t["id"], cond), args.force, args.cache_dir)
            rows.append(row)
            detail = (
                f"F {row['f']:.2f}  P {row['precision']:.2f}  R {row['recall']:.2f}  "
                f"name-F {row.get('f_note_name', 0):.2f}"
                if "f" in row
                else f"OA {row['oa']:.2f}  RPA {row['rpa']:.2f}  RCA {row['rca']:.2f}  note acc {row['note_acc']:.2f}"
            )
            print(
                f"[{k + 1}/{len(tracks)}] {t['id']} {cond:4s} {row['verdict']:5s} {detail}  "
                f"{row.get('seconds', 0):.0f} s  "
                f"{row.get('error', '')}",
                flush=True,
            )
            atomic_write_text(
                os.path.join(args.out, "results.jsonl"), "".join(json.dumps(r, default=float) + "\n" for r in rows)
            )
    summary, md = summarize(rows)
    summary["dataset"] = kind
    atomic_write_text(os.path.join(args.out, "summary.json"), json.dumps(summary, indent=1, default=float))
    atomic_write_text(os.path.join(args.out, "summary.md"), md)
    for cond, agg in summary["conditions"].items():
        a = agg["score"]
        print(
            f"{cond}: mean {summary['headline']} {a['mean']:.3f}, median {a['median']:.3f}, min {a['min']:.3f} "
            f"({agg['tracks']} tracks, {agg['errors']} errors)"
        )
    print(f"summary: {os.path.join(args.out, 'summary.md')}")
    if args.write_baseline:
        atomic_write_text(args.write_baseline, json.dumps(baseline_of(rows), indent=1, default=float))
    if args.baseline:
        problems = check_baseline(rows, json.load(open(args.baseline)), args.max_drop)
        for p in problems:
            print(f"REGRESSION {p}", file=sys.stderr)
        if problems:
            return 3
        print("no regression against the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
