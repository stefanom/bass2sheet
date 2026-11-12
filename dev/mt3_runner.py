"""Runs YourMT3+ (Chang et al., 2024: a transformer that writes every instrument's notes straight from audio) on
audio files and writes the notes it finds in each as CSV: start,end,pitch,program,is_drum.

It is here to measure bass2sheet's own note finder against (dev/idmt_eval.py --yourmt3, docs/evaluation.md); the
tool does not use it. It runs under the interpreter of YourMT3's own environment, with YOURMT3_HOME pointing at a
checkout of its code and weights:

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/mimbres/YourMT3     # code, Apache-2.0
    # then fetch amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/
    # last.ckpt (560 MB) from the same place, or clone with Git LFS
    uv venv --python 3.11 .venv-mt3
    uv pip install --python .venv-mt3/bin/python torch torchaudio 'lightning>=2.2.1' transformers==4.45.1 \\
        numpy==1.26.4 einops mido mir_eval deprecated librosa wandb smart_open soundfile scikit-learn matplotlib

    python mt3_runner.py cpu|mps|cuda AUDIO.wav OUT.csv [AUDIO.wav OUT.csv ...]
"""

import csv
import os
import sys
from collections import Counter

HOME = os.environ.get("YOURMT3_HOME", "")
if not os.path.isdir(os.path.join(HOME, "amt", "src")):
    sys.exit(f"YOURMT3_HOME={HOME!r} is not a YourMT3 checkout (no amt/src)")
device = sys.argv[1]
jobs = [(os.path.abspath(a), os.path.abspath(b)) for a, b in zip(sys.argv[2::2], sys.argv[3::2])]
os.chdir(HOME)  # the checkpoint directory is looked up relative to the checkout
sys.path[:0] = [HOME, os.path.join(HOME, "amt", "src")]

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from model_helper import load_model_checkpoint  # noqa: E402
from utils.audio import slice_padded_array  # noqa: E402
from utils.event2note import merge_zipped_note_events_and_ties_to_notes  # noqa: E402
from utils.note2event import mix_notes  # noqa: E402

# "YPTF.MoE+Multi (noPS)": the model the authors' demo uses
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf"]
ARGS += ["-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1"]
ARGS += ["-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32"]


def transcribe(model, path, device):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(y.mean(1))[None]
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    n = model.audio_cfg["input_frames"]
    segments = torch.from_numpy(slice_padded_array(audio.numpy(), n, n).astype("float32")).unsqueeze(1)
    with torch.no_grad():
        tokens, _ = model.inference_file(bsz=8, audio_segments=segments.to(device))
    starts = [n * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
    per_channel, errors = [], Counter()
    for ch in range(model.task_manager.num_decoding_channels):
        arr = [a[:, ch, :] for a in tokens]
        zipped, _, _ = model.task_manager.detokenize_list_batches(arr, starts, return_events=True)
        notes, err = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_channel.append(notes)
        errors += err
    return mix_notes(per_channel)


model = load_model_checkpoint(args=ARGS, device="cpu")
for wav_path, out_path in jobs:
    try:
        notes = transcribe(model.to(device), wav_path, device)
    except Exception as e:  # an operation the device does not support
        if device == "cpu":
            raise
        print(f"YourMT3+ on {device} failed ({type(e).__name__}: {e}); using the CPU", file=sys.stderr)
        device = "cpu"
        notes = transcribe(model.to(device), wav_path, device)
    with open(out_path + ".part", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "pitch", "program", "is_drum"])
        for nt in sorted(notes, key=lambda nt: (nt.onset, nt.pitch)):
            w.writerow([f"{nt.onset:.4f}", f"{nt.offset:.4f}", int(nt.pitch), int(nt.program), int(bool(nt.is_drum))])
    os.replace(out_path + ".part", out_path)
    print("DONE", out_path, len(notes), dict(Counter(int(nt.program) for nt in notes).most_common(8)), flush=True)
