# Changelog

## 0.2.0 (unreleased)

Measured against ground truth for the first time; the numbers are in `docs/evaluation.md`.

- **Evaluation.** `bass2sheet-eval` scores the tool on BabySlakh/Slakh2100 (exact notes and beats), MDB-bass-synth (exact f0 on real mixes) and RWC Popular (aligned MIDI on real recordings), with an error breakdown, a gate-vs-accuracy table and a baseline regression check.
- **Note starts** move to where the pluck begins to sound, at 1 ms resolution. Frame-level starts were 15–25 ms late.
- **Beats and downbeats** come from the Beat This! model. The built-in tracker chose the wrong pulse level on about half of the BabySlakh songs.
- **Repeated notes.** A held note is split where the string is plucked again. Driving eighth-note lines came out as single long notes on real recordings.
- **Separator choice.** When both separators were transcribed, the result with fewer failed gates is kept, then the one with more note starts confirmed by Basic Pitch. The report says which gates each attempt tripped.
- **Quality gates.** Thresholds are set from the evaluation runs, and five warnings at once count as a failure. The drum-bleed and synth-gap gates are gone: they fired on good results. A meter gate (downbeats that fit 4/4) is new; it and the eighth-note gate warn but never fail. Note coverage no longer penalizes a perfect transcription.
- **Memory.** CREPE's batch is bounded (it could need over 30 GB), and cached GPU memory is released between stages.
- **Inputs.** `--stems DIR` uses stems you already have. `--bass-stem`, `--beat-tracker`, `--crepe-model`, `--no-pdf`. The shortest accepted input is 10 s.
- **Score.** Lines start where the bass comes in.
- **Tests.** Unit tests for every stage and end-to-end tests on a synthesized song, with no recorded audio and no model download; an opt-in tier for real separation and the dataset regression.

## 0.1.0

First version: audio file to PDF bass sheet (bass clef and tab), MusicXML, MIDI, listening files and a quality report with a PASS/WARN/FAIL verdict.
