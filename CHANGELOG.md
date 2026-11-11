# Changelog

## 0.2.0 (unreleased)

Measured against ground truth for the first time; the numbers are in `docs/evaluation.md`.

- **Evaluation.** `bass2sheet-eval` scores the tool on BabySlakh/Slakh2100 (exact notes and beats), MDB-bass-synth (exact f0 on real mixes) and RWC Popular (aligned MIDI on real recordings), with an error breakdown, a gate-vs-accuracy table and a baseline regression check.
- **Note starts** move to where the pluck begins to sound, at 1 ms resolution. Frame-level starts were 15–25 ms late.
- **Beats and downbeats** come from the Beat This! model. The built-in tracker chose the wrong pulse level on about half of the BabySlakh songs.
- **Quality gates.** The drum-bleed gate is gone: it fired on clean stems. A meter gate (downbeats that fit 4/4) is new. Note coverage no longer penalizes a perfect transcription.
- **Memory.** CREPE's batch is bounded (it could need over 30 GB), and cached GPU memory is released between stages.
- **Inputs.** `--stems DIR` uses stems you already have. `--bass-stem`, `--beat-tracker`, `--crepe-model`, `--no-pdf`.
- **Score.** Lines start where the bass comes in.
- **Tests.** Unit tests for every stage and end-to-end tests on a synthesized song, with no recorded audio and no model download; an opt-in tier for real separation and the dataset regression.

## 0.1.0

First version: audio file to PDF bass sheet (bass clef and tab), MusicXML, MIDI, listening files and a quality report with a PASS/WARN/FAIL verdict.
