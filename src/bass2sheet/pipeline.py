"""End to end: audio file -> stems -> notes -> grid -> quality gates -> PDF/MusicXML/MIDI + report."""

import json
import os
import re
import time

import numpy as np
import soundfile as sf

from . import __version__, render, report
from .audio import decode, validate
from .common import SR, Bass2SheetError, atomic_write_text, clock, log, sha1_file
from .notes import SegParams, segment_notes
from .pitch import features
from .quality import CHECK_SEVERITY, check_list, gates_json, stem_gates, transcription_gates, verdict
from .rhythm import beat_grid, beats_and_downbeats, chords_per_bar, estimate_key, quantize
from .score import engrave, find_mscore, musicxml
from .second_opinion import basic_pitch_notes
from .separate import separate

EXIT_OK, EXIT_ERROR, EXIT_FAILED_GATES = 0, 1, 2


def guess_title(path):
    """'Artist - Title [videoid].ext' -> (title, artist)."""
    base = re.sub(r"\s*\[[^\]]+\]$", "", os.path.splitext(os.path.basename(path))[0]).strip()
    if " - " in base:
        artist, title = base.split(" - ", 1)
        return title.strip(), artist.strip()
    return base, ""


def transcribe(stems, cache_dir, params, second_mode, bp_python, device):
    bass_mono = stems["bass"].mean(0)
    feat = features(bass_mono, cache_dir, device)
    notes, _, _ = segment_notes(feat, bass_mono, params)
    if len(notes) < 8:
        raise Bass2SheetError(f"only {len(notes)} bass notes found; the stem has no clearly pitched bass")
    tempo, pulses, per_quarter, phase = beats_and_downbeats(stems["drums"], stems["bass"], stems["harmony"])
    grid, origin = beat_grid(pulses, per_quarter, phase, len(bass_mono) / SR)
    notes = quantize(notes, grid, origin)
    chords, chroma = chords_per_bar(stems["harmony"], notes, grid, origin)
    key = estimate_key(chroma)
    synth_bass = render.synth(notes, len(bass_mono))
    if second_mode == "off":
        second, reason = None, "turned off (--second-opinion off)"
    else:
        second, reason = basic_pitch_notes(stems["bass"], feat["stem_id"], cache_dir, bp_python)
    gates, checks = transcription_gates(stems, feat, notes, grid, origin, synth_bass, second, reason)
    return dict(notes=notes, tempo=tempo, per_quarter=per_quarter, grid=grid, origin=origin, chords=chords, key=key,
                synth=synth_bass, gates=gates, checks=checks, second_reason=reason)


def run(audio_path, out_dir, title=None, artist=None, separator="auto", strict=False, cache_dir=None, mscore=None,
        renders=True, second_mode="auto", bp_python=None, device=None):
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

    methods = ["roformer", "demucs"] if separator == "auto" else [separator]
    attempts, candidates = [], []
    params = SegParams()
    for i, method in enumerate(methods):
        last = i == len(methods) - 1
        t0 = time.time()
        try:
            stems = separate(method, wav, mix_path, audio_id, cache_dir, device)
        except Exception as e:  # a missing model, a download failure, an unsupported op
            attempts.append(dict(method=method, outcome=f"separation failed: {e}"))
            log.warning("separation with %s failed: %s", method, e)
            continue
        finally:
            tick("separation", t0)
        sg = stem_gates(stems, mix)
        no_bass = any(g.name == "bass_level" and g.status == "fail" for g in sg)
        if verdict(sg) == "FAIL" and (no_bass or not last):
            attempts.append(dict(method=method, outcome="stem failed its gates (" + ", ".join(
                f"{g.label}: {g.value:.2f}" for g in sg if g.status == "fail") + ")" + ("" if last else
                                                                                       "; trying the next separator")))
            continue
        t0 = time.time()
        try:
            c = transcribe(stems, cache_dir, params, second_mode, bp_python, device)
        except Bass2SheetError as e:
            attempts.append(dict(method=method, outcome=str(e)))
            continue
        finally:
            tick("transcription", t0)
        c.update(method=method, stems=stems, gates=sg + c["gates"])
        candidates.append(c)
        v = verdict(c["gates"])
        attempts.append(dict(method=method, outcome=f"transcribed: {v}"))
        if v != "FAIL":
            break
        if not last:
            log.warning("result with %s failed its quality gates; trying the next separator", method)
    if not candidates:
        raise Bass2SheetError("no usable transcription: " + "; ".join(f"{a['method']}: {a['outcome']}" for a in attempts))
    rank = lambda c: (sum(g.status == "fail" for g in c["gates"]), sum(g.status == "warn" for g in c["gates"]),
                      -next(g.value for g in c["gates"] if g.name == "coverage"))
    best = min(candidates, key=rank)
    if len(candidates) > 1:
        attempts.append(dict(method=best["method"], outcome="chosen: fewest failing, then warning gates"))
    v = verdict(best["gates"])
    notes, grid, origin, tempo = best["notes"], best["grid"], best["origin"], best["tempo"]
    checks = check_list(best["checks"], grid, origin)

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
        xml = musicxml(notes, best["chords"], best["key"], tempo, float(grid[origin]), title, artist or "",
                       {c["bar"]: c["reasons"] for c in checks if c["severity"] >= CHECK_SEVERITY}, draft=v == "FAIL")
        atomic_write_text(path("bass.musicxml"), xml)
        files["bass.musicxml"] = "the score, editable in MuseScore"
        ms = find_mscore(mscore)
        if ms:
            err = engrave(path("bass.musicxml"), path("bass.pdf"), ms)
            if err:
                warnings.append(f"PDF not written: {err}")
            else:
                files["bass.pdf"] = "printable score: bass clef + tab" + (" (DRAFT)" if v == "FAIL" else "")
        else:
            warnings.append("PDF not written: MuseScore 4 not found (install it, or pass --mscore PATH); the MusicXML "
                            "opens in any notation program")
    if renders:
        os.makedirs(path("audio"), exist_ok=True)
        sf.write(path("audio/bass_stem.wav"), best["stems"]["bass"].T, SR, subtype="PCM_16")
        render.write_renders(best["stems"], best["synth"], {k: path(f"audio/{f}") for k, f in (
            ("resynth", "song_resynth.wav"), ("ab", "bass_ab_lr.wav"), ("synth", "synth_bass.wav"))})
        files.update({"audio/bass_stem.wav": "the separated bass", "audio/synth_bass.wav": "the transcription, synthesized",
                      "audio/song_resynth.wav": "the song with the transcription in place of the real bass",
                      "audio/bass_ab_lr.wav": "real bass left, transcription right (headphones)"})
    data_notes = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in nt.items()} for nt in notes]
    atomic_write_text(path("notes.json"), json.dumps(dict(
        notes=data_notes, chords=best["chords"], key=best["key"][0], tempo_bpm=tempo, first_downbeat_s=float(grid[origin]),
        tracked_pulse="eighth" if best["per_quarter"] == 2 else "quarter", params=params.as_dict()), indent=1, default=float))
    files["notes.json"] = "notes, chords, key, tempo, grid"
    tick("outputs", t0)
    timings["total"] = time.time() - t_start
    files["report.html"] = "this report"
    report.write(path("report.html"), path("report.json"), dict(
        version=__version__, title=title, artist=artist or "", input=os.path.basename(audio_path), verdict=v,
        gates=gates_json(best["gates"]), ear_checks=checks, attempts=attempts, warnings=warnings, files=files,
        check_severity=CHECK_SEVERITY, timings=timings, summary=dict(duration=clock(seconds), notes=len(notes), bars=max(nt["bar"] for nt in notes),
                                      tempo=tempo, key=best["key"][0], separator=best["method"])))
    log.info("verdict %s; %d bars to check by ear (%d marked in the PDF); report: %s", v, len(checks),
             sum(c["severity"] >= CHECK_SEVERITY for c in checks), path("report.html"))
    for g in best["gates"]:
        if g.status in ("warn", "fail"):
            log.warning("gate %s: %s (%s)", g.status.upper(), g.label, report.fmt(gates_json([g])[0]))
    return EXIT_FAILED_GATES if v == "FAIL" else EXIT_OK
