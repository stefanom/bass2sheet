# Changelog

## 0.2.0 (unreleased)

Measured against ground truth for the first time; the numbers are in `docs/evaluation.md`.

- **Judging on your own songs.** `bass2sheet-judge` takes your songs through the pipeline one step at a time: the isolated bass tracks, the notes heard in them, the count, the written rhythm, the tab. Each step sets the tool's choice against its alternative on short passages, by ear and by eye, without saying which version is which, and is built on what you picked at the steps before it. A cursor follows the audio over the pictures and scores, and what differs is marked. Picks decide the defaults, and a judged passage comes back when a later change alters it. See `docs/judging.md`.
- **Lines that jump are no longer written too low.** The pitch path can leap between notes; it could only move smoothly, so after a jump of a fifth or more the note before was held over the new one. On real bass guitar the share of such notes written right went from 46% to 79%, and slap pops from 18% to 72%. When the two pitch trackers are an octave apart and CREPE is confident, its octave stands. Found by judging: the timing was right and the notes were too low.
- **After a change, `bass2sheet-judge update` builds again what is not settled**: passages still open, or where neither version was right, if the tool writes them differently now.
- **Pitch cache.** The cache holds CREPE's bin probabilities (`crepe-*.npz`) instead of the decoded pitch, so the decoding can change without running the model again. Older `tracks-*.npz` files are no longer read and can be deleted.
- **Measured on real bass guitar.** `dev/idmt_eval.py` runs the note finder on IDMT-SMT-Bass-Single-Track, by plucking style, beside Basic Pitch and YourMT3+ (`dev/mt3_runner.py`). Neither does better; see `docs/evaluation.md`.
- **Choices you can make per run.** `--replucks every|clear`, `--rhythm literal|simple` (stray sixteenths onto eighths, short rests closed) and `--tab low|position` (the hand kept in one position).
- **Line times** in the score follow the tracked beats; they assumed a steady tempo before.
- **File names.** "(Official Video)" and the like are dropped from the title and the output folder.
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
