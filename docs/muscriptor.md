# MuScriptor's notes instead of bass2sheet's own

[MuScriptor](https://github.com/muscriptor/muscriptor) (Rouard et al., 2026, by Kyutai and Mirelo) is a transformer that writes notes straight from audio. Given the bass track bass2sheet isolates, and told that it is a bass, it finds the notes better than bass2sheet's own note finder does:

| | bass2sheet's own notes | MuScriptor's, on the same isolated bass |
|---|---|---|
| Real bass played alone (17 lines) | 0.79 | 0.91 |
| RWC Popular (24 real pop songs) | 0.76 | 0.83 |
| BabySlakh (16 synthesized songs) | 0.92 | 0.94 |

Given the whole mix it does worse than separating first (0.72 on RWC, 0.50 on BabySlakh), so the separation stays. The measurements are in [evaluation.md](evaluation.md).

It is optional and not part of bass2sheet. Nothing of it runs unless you ask for it.

## Before you use it

**The weights are not free of conditions.** They are under CC BY-NC 4.0, which rules out commercial use, and their authors add conditions of use: among them, that you have the necessary rights in the music you transcribe with the model. Read them where you get the weights. bass2sheet's MIT licence covers bass2sheet's code and nothing of MuScriptor.

## Setting it up

MuScriptor runs here through [muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp), its C++ port, with the converted weights from [DamRsn/muscriptor-gguf](https://huggingface.co/DamRsn/muscriptor-gguf). On a Mac it uses the GPU.

```bash
git clone https://github.com/DamRsn/muscriptor.cpp
cmake -S dev/muscriptor_cpp -B build/muscriptor -G Ninja -DCMAKE_BUILD_TYPE=Release -DMUSCRIPTOR_CPP=$PWD/muscriptor.cpp
cmake --build build/muscriptor                      # needs CMake 3.24+, Ninja, a C++23 compiler, the network

export MUSCRIPTOR_NOTES=$PWD/build/muscriptor/muscriptor_notes
export MUSCRIPTOR_GGUF=<the directory holding muscriptor-medium-f16.gguf>
export MUSCRIPTOR_SIZE=medium                       # the default; large is no better on a bass track and slower
```

On a Mac where the build fails at the first link with "unknown architecture", point it at Xcode's SDK: `export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)`.

## Using it

```bash
bass2sheet song.mp3 --notes muscriptor
bass2sheet-copilot add song.mp3 --notes muscriptor --dir scores-muscriptor
bass2sheet-eval ~/data/babyslakh_16k --out runs/babyslakh-muscriptor --notes muscriptor
```

The copilot keeps a song's notes from when it was added: to see MuScriptor's version of a song you have already worked on, add it under another working directory, as above. Your fixes in the first are not touched.

## What happens

1. The bass is isolated from the mix as always.
2. bass2sheet's own finder writes its notes. They are no longer the result: they become the second opinion.
3. MuScriptor writes the isolated bass's notes, told to write a bass and nothing else. A four-minute track takes 10 to 25 seconds on an M4 Max; the notes are kept in the cache.
4. Its notes are made into one line (`muscriptor.as_notes`): a pitch written twice at once is one note, of the notes it starts together one is the note (the one bass2sheet's finder hears too, or else the lowest; of two a semitone or a tone apart, the one that sounds) and another is written with it only where bass2sheet's own finder wrote the same two, a note ends where the next begins, and what lies under the instrument's lowest note goes up an octave. The starts move from MuScriptor's 10 ms steps to where the pluck begins to sound. A note bass2sheet's finder hears too ends where bass2sheet ends it, at the fall of the track's level: MuScriptor holds the synthesized notes of BabySlakh too long (the score that also asks for the note's end falls from 0.73 to 0.45 with its ends, and is 0.71 with bass2sheet's), and on RWC Popular and the bass played alone the borrowed ends are better than either finder's own (0.54 against 0.48 and 0.44; 0.71 against 0.63 and 0.57).
5. Everything after that is unchanged: the beat grid, the rhythm, the tab, the checks, the sheet.

**Which notes are doubted.** A note of MuScriptor's is flagged when bass2sheet's own finder hears another pitch there, or nothing, or holds that pitch from the note before with no pluck to be heard. A note both finders write is not, and neither is a pitch bass2sheet holds where MuScriptor writes a new pluck and a pluck is heard: on real recordings MuScriptor is right about those nearly nine times in ten. On RWC Popular the flag falls on 11% of the notes; 56% of them are wrong, against 17% of all notes, and they hold 36% of the wrong notes. On the synthesized BabySlakh it is weaker: 8% of the notes, 21% of them wrong. Flagged notes are in parentheses on the sheet and amber in the copilot.

The original recording has no say in the flag. It was tried as a third witness: the level must rise at the note's own partials around its start, in the original and in the isolated bass. Taking the flag off the notes it supports, or putting it on the notes it does not, made the flag worse on RWC Popular at every threshold: the flagged notes it supports are wrong as often as the others (55%), and a note that rises in neither is still right three times in four. The rise tells a re-pluck that was not played, and most wrong notes are another kind: something is plucked there, and the note written is not the one. The table is in [evaluation.md](evaluation.md).

## Limits

- The numbers above are for the medium model on the isolated bass. From the full mix, or told to write only the bass of a full mix, MuScriptor is not usable for this (see [evaluation.md](evaluation.md)).
- The PASS, WARN and FAIL verdicts are stricter with MuScriptor's notes than with bass2sheet's own: their thresholds were set on the latter. On RWC Popular 11 of 24 songs fail with MuScriptor's notes (mean score 0.72) against 6 with bass2sheet's (0.56), though the notes are better. Read the verdict as a ranking until it is recalibrated.
- BabySlakh is rendered from a MIDI collection that is part of MuScriptor's training data; RWC Popular was held out by its authors.
- The reference code (`pip install muscriptor`, with the original weights behind a sign-in on Hugging Face) is supported for measuring only: `dev/muscriptor_runner.py`.
