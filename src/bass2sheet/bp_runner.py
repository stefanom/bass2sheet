"""Basic Pitch run in its own Python environment (its TensorFlow/ONNX stack doesn't share an env with torch well).
This file is executed by that interpreter and must not import bass2sheet.

    BASIC_PITCH_PYTHON bp_runner.py STEM.wav OUT.csv
"""
import os
import sys

import scipy.signal
import scipy.signal.windows

scipy.signal.gaussian = scipy.signal.windows.gaussian  # removed from scipy.signal in recent SciPy
import basic_pitch  # noqa: E402
from basic_pitch.inference import Model, predict  # noqa: E402

stem, out_csv = sys.argv[1], sys.argv[2]
model = Model(os.path.join(os.path.dirname(basic_pitch.__file__), "saved_models", "icassp_2022", "nmp.onnx"))
_, _, events = predict(stem, model, minimum_frequency=35, maximum_frequency=400, minimum_note_length=60,
                       onset_threshold=0.5, frame_threshold=0.3, multiple_pitch_bends=False, melodia_trick=True)
tmp = out_csv + ".tmp"
with open(tmp, "w") as f:
    f.write("start,end,pitch,amplitude\n")
    for s, e, p, a, _ in sorted(events):
        f.write(f"{s:.3f},{e:.3f},{p},{a:.3f}\n")
os.replace(tmp, out_csv)
print(len(events))
