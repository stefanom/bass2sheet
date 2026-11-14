"""End to end: audio file -> stems -> notes -> grid -> quality gates -> PDF/MusicXML/MIDI + report."""

import json
import os
import time

import numpy as np
import soundfile as sf

from . import __version__, ownsound, render, report
from .audio import decode, validate
from .common import (
    ALTERNATIVES,
    DEFAULTS,
    LOWEST_NOTE,
    SR,
    Bass2SheetError,
    atomic_write_text,
    clean_name,
    clock,
    log,
    note_name,
    sha1_file,
    step,
)
from .notes import SegParams, segment_notes
from .pitch import features
from .quality import CHECK_SEVERITY, check_list, gates_json, stem_gates, transcription_gates, verdict
from .rhythm import (
    beat_grid,
    beats_and_downbeats,
    chords_per_bar,
    estimate_key,
    grid_from_model_beats,
    model_beats,
    quantize,
    simplify,
)
from .score import engrave, find_mscore, musicxml
from .second_opinion import basic_pitch_notes
from .separate import load_stems, separate

EXIT_OK, EXIT_ERROR, EXIT_FAILED_GATES = 0, 1, 2


NOTE_FINDERS = ("own", "muscriptor")  # whose notes are written (seg_params)


def guess_title(path):
    """'Artist - Title (Official Video) [videoid].ext' -> (title, artist)."""
    base = clean_name(path)
    if " - " in base:
        artist, title = base.split(" - ", 1)
        return title.strip(), artist.strip()
    return base, ""


BEAT_TRACKERS = ("auto", "beat_this", "librosa")


def seg_params(replucks, together=None, lowest=LOWEST_NOTE, notes=None):
    """Note segmentation for a re-pluck policy, for whether notes played together are written (see DEFAULTS), for
    the instrument's lowest note (a MIDI number; 0 for no limit) and for whose notes are written: bass2sheet's
    own ("own", the default) or MuScriptor's with bass2sheet's as the second opinion ("muscriptor")."""
    together = together or DEFAULTS["together"]
    if (notes or "own") not in NOTE_FINDERS:
        raise Bass2SheetError(f"unknown note finder {notes!r}; expected one of {', '.join(NOTE_FINDERS)}")
    if replucks not in ALTERNATIVES["replucks"]:
        raise Bass2SheetError(f"unknown re-pluck policy {replucks!r}; expected 'every' or 'clear'")
    if together not in ALTERNATIVES["together"]:
        raise Bass2SheetError(f"unknown choice for notes played together {together!r}; expected 'write' or 'single'")
    return SegParams(
        strict_replucks=replucks == "clear",
        split_held=replucks == "every",
        together=together == "write",
        lowest=int(lowest or 0),
        finder=notes or "own",
    )


def track_grid(stems, mix, audio_id, cache_dir, device, beat_tracker):
    """(tempo, pulses, pulses per quarter, phase of beat 1, 4/4 fit or None, tracker used)."""
    if beat_tracker in ("auto", "beat_this"):
        try:
            beats, downbeats = model_beats(mix, device, os.path.join(cache_dir, f"beats-{audio_id[:16]}.npz"))
            return (*grid_from_model_beats(beats, downbeats), "beat_this")
        except Exception as e:  # not installed, no network for the first download, too few beats
            if beat_tracker == "beat_this":
                raise Bass2SheetError(f"the Beat This! beat tracker failed: {e}") from e
            log.warning("Beat This! beat tracker unavailable (%s); using the built-in one", e)
    try:
        return (*beats_and_downbeats(stems["drums"], stems["bass"], stems["harmony"]), None, "librosa")
    except ValueError as e:
        raise Bass2SheetError(f"could not find a steady beat ({e})") from e


def find_notes(stems, cache_dir, params, device, crepe_model="full", mix=None, audio_id="", beat_tracker="auto"):
    """Everything the sheet is written from: the notes heard in the bass stem, placed on the beat grid, with the
    key and the chords. Returns them with the pitch features (`feat`) and the 4/4 fit of the beats (`meter_fit`)."""
    bass_mono = stems["bass"].mean(0)
    with step("pitch") as found:
        feat = features(bass_mono, cache_dir, device, crepe_model)
        found["pitched"] = round(float(np.mean((feat["per_c"] >= 0.3) & (feat["f0_c"] > 0))), 3)
    others = [v.mean(0) for k, v in stems.items() if k != "bass"]
    rest = sum(v[: len(bass_mono)] for v in others) if others else None  # the record without the bass track
    with step("notes") as found:
        notes, _, _ = segment_notes(feat, bass_mono, params, rest)
        found["notes"] = len(notes)
        if notes:
            found["low"], found["high"] = (note_name(f(n["pitch"] for n in notes)) for f in (min, max))
    if params.finder == "muscriptor":  # its notes are written; the ones just found say which of them to doubt
        from . import muscriptor

        with step("muscriptor") as found:
            raw = muscriptor.heard(bass_mono, feat["stem_id"], cache_dir)
            found["notes"] = len(raw)
        with step("second") as found:
            notes = muscriptor.as_notes(raw, notes, bass_mono, params)
            found["notes"], found["alone"] = len(notes), sum(n.get("flag") == muscriptor.ALONE for n in notes)
    if len(notes) < 8:
        raise Bass2SheetError(f"only {len(notes)} bass notes found; the stem has no clearly pitched bass")
    if mix is None:
        mix = sum(stems.values()).mean(0)
    with step("beats") as found:
        tempo, pulses, per_quarter, phase, meter_fit, tracker = track_grid(
            stems, mix, audio_id, cache_dir, device, beat_tracker
        )
        grid, origin = beat_grid(pulses, per_quarter, phase, len(bass_mono) / SR)
        notes = quantize(notes, grid, origin)
        chords, chroma = chords_per_bar(stems["harmony"], notes, grid, origin)
        key = estimate_key(chroma)
        found.update(tempo=round(float(tempo), 1), bars=max(0, (len(grid) - int(origin) + 15) // 16), key=key[0])
    return dict(
        notes=notes,
        tempo=tempo,
        per_quarter=per_quarter,
        beat_tracker=tracker,
        grid=grid,
        origin=origin,
        chords=chords,
        key=key,
        feat=feat,
        meter_fit=meter_fit,
    )


def transcribe(
    stems,
    cache_dir,
    params,
    second_mode,
    bp_python,
    device,
    crepe_model="full",
    mix=None,
    audio_id="",
    beat_tracker="auto",
):
    """find_notes, then the quality gates on what it found."""
    c = find_notes(stems, cache_dir, params, device, crepe_model, mix, audio_id, beat_tracker)
    feat, meter_fit = c.pop("feat"), c.pop("meter_fit")
    synth_bass = render.synth(c["notes"], stems["bass"].shape[1])
    if second_mode == "off":
        second, reason = None, "turned off (--second-opinion off)"
    else:
        second, reason = basic_pitch_notes(stems["bass"], feat["stem_id"], cache_dir, bp_python)
    gates, checks = transcription_gates(
        stems, feat, c["notes"], c["grid"], c["origin"], synth_bass, second, reason, meter_fit, c["beat_tracker"]
    )
    return dict(c, synth=synth_bass, gates=gates, checks=checks, second_reason=reason)


def bar_times(grid, origin):
    """{bar: where it starts in the recording, in seconds}, following the tracked beats rather than one tempo."""
    return {k + 1: float(grid[i]) for k, i in enumerate(range(origin, len(grid), 16))}


def rank_candidate(gates):
    """Sort key for results from different separators: lowest is best.

    Failed gates count first. After that the share of note starts confirmed by the second model decides, and
    coverage when there is no second model. The number of warning gates is not used: each gate is measured
    against the candidate's own stem, and on ten songs with ground truth transcribed from both separators the
    candidate with fewer warnings was the less accurate one on 5 and the more accurate on 2, while the one with
    more confirmed note starts was the more accurate one on 7 and the less accurate on 1 (docs/evaluation.md)."""
    value = lambda name: next((g.value for g in gates if g.name == name and g.value is not None), 0.0)
    return (sum(g.status == "fail" for g in gates), -value("second_model_onsets"), -value("coverage"))


def run(
    audio_path,
    out_dir,
    title=None,
    artist=None,
    separator="auto",
    strict=False,
    cache_dir=None,
    mscore=None,
    renders=True,
    second_mode="auto",
    bp_python=None,
    device=None,
    bass_stem=None,
    pdf=True,
    stems_dir=None,
    crepe_model="full",
    beat_tracker="auto",
    replucks=None,
    rhythm=None,
    tab=None,
    together=None,
    lowest=LOWEST_NOTE,
    notes=None,
):
    """Runs the whole pipeline and writes the outputs; returns the exit code.

    `replucks`, `together`, `rhythm` and `tab` default to the standing choices in DEFAULTS, as do the first
    separator and the beat tracker when those are left on "auto". `lowest` is the instrument's lowest note as a
    MIDI number (E1 by default, for the four-string bass the tab is written for): what is heard below it is
    written an octave up; 0 leaves the notes as heard. `notes` is whose notes are written: bass2sheet's own
    (the default) or "muscriptor" (see muscriptor.py; it has to be set up).

    `bass_stem` is an audio file used as the bass in place of the separated one (the rest of the separation is kept
    for the beat grid and chords). It exists for evaluation: with a dataset's clean stem it shows how much of the
    error comes from separation. `stems_dir` skips separation altogether and reads the stems from a directory
    (see separate.load_stems)."""
    t_start = time.time()
    timings, warnings = {}, []
    tick = lambda name, t0: timings.__setitem__(name, timings.get(name, 0) + time.time() - t0)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir or "~/.cache/bass2sheet"))
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    g_title, g_artist = guess_title(audio_path)
    title, artist = title or g_title, artist if artist is not None else g_artist

    t0 = time.time()
    wav = decode(audio_path)
    seconds, input_warnings = validate(wav)
    warnings += input_warnings
    audio_id = sha1_file(audio_path)
    mix_path = os.path.join(cache_dir, f"mix-{audio_id[:16]}.wav")
    if not os.path.exists(mix_path):
        sf.write(mix_path, wav.T, SR, subtype="FLOAT")
    mix = wav.mean(0)
    tick("decode", t0)
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    replucks, rhythm, tab = (replucks or DEFAULTS["replucks"], rhythm or DEFAULTS["rhythm"], tab or DEFAULTS["tab"])
    together = together or DEFAULTS["together"]
    for name, value in (("rhythm", rhythm), ("tab", tab)):
        if value not in ALTERNATIVES[name]:
            raise Bass2SheetError(f"unknown {name} {value!r}; expected one of {', '.join(ALTERNATIVES[name])}")
    if beat_tracker == "auto" and DEFAULTS["beat_tracker"] == "librosa":
        beat_tracker = "librosa"
    order = [DEFAULTS["separator"]] + [m for m in ALTERNATIVES["separator"] if m != DEFAULTS["separator"]]
    methods = ["given stems"] if stems_dir else order if separator == "auto" else [separator]
    attempts, candidates = [], []
    params = seg_params(replucks, together, lowest, notes)
    for i, method in enumerate(methods):
        last = i == len(methods) - 1
        t0 = time.time()
        try:
            stems = (
                load_stems(stems_dir, wav.shape[1])
                if stems_dir
                else separate(method, wav, mix_path, audio_id, cache_dir, device)
            )
        except Exception as e:  # a missing model, a download failure, an unsupported op
            if stems_dir:  # the given stems are the only source: nothing to fall back to
                raise
            attempts.append(dict(method=method, outcome=f"separation failed: {e}"))
            log.warning("separation with %s failed: %s", method, e)
            continue
        finally:
            tick("separation", t0)
        if bass_stem:
            b = decode(bass_stem)
            n = wav.shape[1]
            stems = dict(stems, bass=np.pad(b[:, :n], ((0, 0), (0, max(0, n - b.shape[1])))))
            method = f"{method} + given bass stem"
        sg = stem_gates(stems, mix)
        if verdict(sg) == "FAIL":  # no bass in the stem: nothing to transcribe
            attempts.append(
                dict(
                    method=method,
                    outcome="no bass found ("
                    + ", ".join(f"{g.label}: {g.value:.1f} dB" for g in sg if g.status == "fail")
                    + ")"
                    + ("" if last else "; trying the next separator"),
                )
            )
            continue
        t0 = time.time()
        try:
            c = transcribe(
                stems, cache_dir, params, second_mode, bp_python, device, crepe_model, mix, audio_id, beat_tracker
            )
        except Bass2SheetError as e:
            attempts.append(dict(method=method, outcome=str(e)))
            continue
        finally:
            tick("transcription", t0)
        c.update(method=method, stems=stems, gates=sg + c["gates"])
        candidates.append(c)
        v = verdict(c["gates"])
        flagged = [f"{g.label} ({g.status})" for g in c["gates"] if g.status in ("fail", "warn")]
        attempts.append(
            dict(method=method, outcome=f"transcribed: {v}" + (f": {'; '.join(flagged)}" if flagged else ""))
        )
        if v != "FAIL":
            break
        if not last:
            log.warning("result with %s failed its quality gates; trying the next separator", method)
    if not candidates:
        raise Bass2SheetError(
            "no usable transcription: " + "; ".join(f"{a['method']}: {a['outcome']}" for a in attempts)
        )
    best = min(candidates, key=lambda c: rank_candidate(c["gates"]))
    if len(candidates) > 1:
        attempts.append(
            dict(
                method=best["method"],
                outcome="chosen: fewest failed gates, then most note starts confirmed by the second model",
            )
        )
    v = verdict(best["gates"])
    notes, grid, origin, tempo = best["notes"], best["grid"], best["origin"], best["tempo"]
    checks = check_list(best["checks"], grid, origin)
    if rhythm == "simple":  # the gates above judged the notes as heard; the sheet shows them written for reading
        notes = simplify(notes)

    # ---- outputs
    t0 = time.time()
    files = {}
    path = lambda name: os.path.join(out_dir, name)
    refuse = strict and v == "FAIL"
    if refuse:
        warnings.append("strict mode: the result failed its quality gates, so no sheet was written")
    else:
        render.write_midi(notes, tempo, path("bass.mid"), steady=True)
        render.write_midi(notes, tempo, path("bass_as_played.mid"), steady=False)
        files["bass.mid"] = "MIDI on a steady 16th grid, bar 1 at time 0"
        files["bass_as_played.mid"] = "MIDI aligned with the recording"
        xml = musicxml(
            notes,
            best["chords"],
            best["key"],
            tempo,
            float(grid[origin]),
            title,
            artist or "",
            {c["bar"]: c["reasons"] for c in checks if c["severity"] >= CHECK_SEVERITY},
            draft=v == "FAIL",
            tab_style=tab,
            bar_times=bar_times(grid, origin),
        )
        atomic_write_text(path("bass.musicxml"), xml)
        files["bass.musicxml"] = "the score, editable in MuseScore"
        ms = find_mscore(mscore) if pdf else None
        if not pdf:
            pass
        elif ms:
            err = engrave(path("bass.musicxml"), path("bass.pdf"), ms)
            if err:
                warnings.append(f"PDF not written: {err}")
            else:
                files["bass.pdf"] = "printable score: bass clef + tab" + (" (DRAFT)" if v == "FAIL" else "")
        else:
            warnings.append(
                "PDF not written: MuseScore 4 not found (install it, or pass --mscore PATH); the MusicXML "
                "opens in any notation program"
            )
    if renders:
        os.makedirs(path("audio"), exist_ok=True)
        sf.write(path("audio/bass_stem.wav"), best["stems"]["bass"].T, SR, subtype="PCM_16")
        render.write_renders(
            best["stems"],
            best["synth"],
            {
                k: path(f"audio/{f}")
                for k, f in (("resynth", "song_resynth.wav"), ("ab", "bass_ab_lr.wav"), ("synth", "synth_bass.wav"))
            },
        )
        rebuilt = ownsound.rebuild(best["stems"]["bass"].mean(0), best["notes"])
        peak = float(np.abs(rebuilt).max())
        sf.write(path("audio/bass_rebuilt.wav"), 0.95 * rebuilt / peak if peak > 0 else rebuilt, SR, subtype="PCM_16")
        files["audio/bass_rebuilt.wav"] = (
            "the transcription played with the bass's own sound: each note by a sample cut from another note"
        )
        files.update(
            {
                "audio/bass_stem.wav": "the separated bass",
                "audio/synth_bass.wav": "the transcription, synthesized",
                "audio/song_resynth.wav": "the song with the transcription in place of the real bass",
                "audio/bass_ab_lr.wav": "real bass left, transcription right (headphones)",
            }
        )
    data_notes = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in nt.items()} for nt in notes]
    atomic_write_text(
        path("notes.json"),
        json.dumps(
            dict(
                notes=data_notes,
                chords=best["chords"],
                key=best["key"][0],
                tempo_bpm=tempo,
                first_downbeat_s=float(grid[origin]),
                tracked_pulse="eighth" if best["per_quarter"] == 2 else "quarter",
                beat_tracker=best["beat_tracker"],
                written=dict(replucks=replucks, together=together, rhythm=rhythm, tab=tab),
                params=params.as_dict(),
            ),
            indent=1,
            default=float,
        ),
    )
    files["notes.json"] = "notes, chords, key, tempo, grid"
    tick("outputs", t0)
    timings["total"] = time.time() - t_start
    files["report.html"] = "this report"
    report.write(
        path("report.html"),
        path("report.json"),
        dict(
            version=__version__,
            title=title,
            artist=artist or "",
            input=os.path.basename(audio_path),
            verdict=v,
            gates=gates_json(best["gates"]),
            ear_checks=checks,
            attempts=attempts,
            warnings=warnings,
            files=files,
            check_severity=CHECK_SEVERITY,
            timings=timings,
            summary=dict(
                duration=clock(seconds),
                notes=len(notes),
                bars=max(nt["bar"] for nt in notes),
                tempo=tempo,
                key=best["key"][0],
                separator=best["method"],
            ),
        ),
    )
    log.info(
        "verdict %s; %d bars to check by ear (%d marked in the PDF); report: %s",
        v,
        len(checks),
        sum(c["severity"] >= CHECK_SEVERITY for c in checks),
        path("report.html"),
    )
    for g in best["gates"]:
        if g.status in ("warn", "fail"):
            log.warning("gate %s: %s (%s)", g.status.upper(), g.label, report.fmt(gates_json([g])[0]))
    return EXIT_FAILED_GATES if v == "FAIL" else EXIT_OK
