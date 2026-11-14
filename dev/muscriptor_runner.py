"""Runs MuScriptor (Rouard et al., 2026: a transformer by Kyutai and Mirelo that writes every instrument's notes
straight from audio; code MIT, https://github.com/muscriptor/muscriptor) on audio files and writes the notes it
finds in each as CSV: start,end,pitch,instrument.

It is here to measure bass2sheet's own note finder against (dev/muscriptor_eval.py, docs/evaluation.md); the tool
does not use it. It runs under the interpreter of an environment that has the `muscriptor` package:

    uv venv --python 3.12 .venv-musc && uv pip install --python .venv-musc/bin/python muscriptor

The weights are not free to download: they are under CC BY-NC 4.0 with conditions of use that each user has to
accept on the model's Hugging Face page, signed in, before `hf auth login` on the machine lets them through.

    python muscriptor_runner.py small|medium|large any|NAME[,NAME...] AUDIO OUT.csv [AUDIO OUT.csv ...]

The second argument is `any` for whatever instruments the model hears, or the instrument groups it may write and
no others (`electric_bass,acoustic_bass`). A job whose OUT.csv exists is skipped.
"""

import csv
import os
import sys
import time

size, instruments = sys.argv[1], sys.argv[2]
jobs = [(a, b) for a, b in zip(sys.argv[3::2], sys.argv[4::2]) if not os.path.exists(b)]
if not jobs:
    sys.exit(0)

from muscriptor import NoteEndEvent, NoteStartEvent, TranscriptionModel  # noqa: E402

model = TranscriptionModel.load_model(size)
only = None if instruments == "any" else instruments.split(",")
for audio, out in jobs:
    t0 = time.time()
    notes = {}
    for event in model.transcribe(audio, instruments=only):
        if isinstance(event, NoteStartEvent):
            notes[event.index] = [event.start_time, None, event.pitch, event.instrument]
        elif isinstance(event, NoteEndEvent):
            notes[event.start_event.index][1] = event.end_time
    rows = sorted(n for n in notes.values() if n[1] is not None and n[1] > n[0])  # drum hits have no end
    with open(out + ".part", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "pitch", "instrument"])
        w.writerows([f"{s:.4f}", f"{e:.4f}", p, i] for s, e, p, i in rows)
    os.replace(out + ".part", out)
    print(f"{os.path.basename(out)}: {len(rows)} notes in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
