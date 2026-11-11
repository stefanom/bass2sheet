"""MIDI files and listening renders (synth bass, the song rebuilt with it, real-vs-synth left/right)."""

import numpy as np
import pretty_midi
import soundfile as sf

from .common import SR


def write_midi(notes, tempo, path, steady):
    """As played (real times) or on a steady 16th grid with bar 1 at time 0 (plus a pickup bar if needed)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=33, name="Bass")  # fingered electric bass
    sixteenth = 60.0 / tempo / 4
    pad = 16 if min(nt["grid_start"] for nt in notes) < 0 else 0
    for nt in notes:
        if steady:
            s = (nt["grid_start"] + pad) * sixteenth
            e = s + nt["grid_len"] * sixteenth
        else:
            s, e = nt["start"], nt["end"]
        inst.notes.append(pretty_midi.Note(velocity=64 if nt["flag"] else 96, pitch=nt["pitch"], start=s, end=e))
    pm.instruments.append(inst)
    pm.write(path)


def pluck(freq, dur, velocity):
    """Harmonics 1-8 at 1/k, a 5 ms attack, faster decay for higher harmonics, a 30 ms release."""
    n = int((dur + 0.03) * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 9):
        if k * freq < SR / 2:
            out += np.sin(2 * np.pi * k * freq * t) * np.exp(-t * (1.2 + 0.8 * k)) / k
    env = np.minimum(1, t / 0.005) * np.clip((dur + 0.03 - t) / 0.03, 0, 1)
    return out * env * velocity / 127


def synth(notes, n):
    out = np.zeros(n)
    for nt in notes:
        w = pluck(pretty_midi.note_number_to_hz(nt["pitch"]), nt["end"] - nt["start"], 64 if nt["flag"] else 96)
        s = int(nt["start"] * SR)
        if s < n:
            out[s : s + len(w)] += w[: n - s]
    return out


def write_renders(stems, synth_bass, paths):
    """Writes the song rebuilt with the synth bass, real-vs-synth L/R and the synth alone (level-matched)."""
    real = stems["bass"].mean(0).astype(np.float64)
    act = np.abs(real) > 1e-4
    gain = np.sqrt(np.mean(real[act] ** 2) / (np.mean(synth_bass[act] ** 2) + 1e-12)) if act.any() else 1.0
    s = synth_bass * gain
    rest = sum(v for k, v in stems.items() if k != "bass")
    mix = rest + s[None, :]
    norm = lambda a: (0.95 * a / (np.abs(a).max() + 1e-9)).T
    sf.write(paths["resynth"], norm(mix), SR, subtype="PCM_16")
    sf.write(paths["ab"], norm(np.stack([real, s])), SR, subtype="PCM_16")
    sf.write(paths["synth"], norm(np.stack([s, s])), SR, subtype="PCM_16")
