"""A synthesized song with a known bass line, for tests: no recorded or copyrighted audio is needed.

The song is 4/4 with kick on 1 and 3, snare on 2 and 4, eighth-note hi-hats, a chord pad that changes every bar and
a plucked bass that plays a fixed pattern over a four-chord loop. `make_song` returns the mix, the separate stems
and the ground truth (notes, beats, downbeats).
"""

import numpy as np

from bass2sheet.render import pluck

SR = 44100
# (root MIDI for the bass, chord tones for the pad), one per bar, looped: Em, C, G, D
CHORDS = [(28, (52, 55, 59)), (36, (48, 52, 55)), (31, (55, 59, 62)), (38, (50, 54, 57))]
# bass pattern within a bar: (start sixteenth, length in sixteenths, semitones above the root)
PATTERN = [(0, 4, 0), (4, 2, 0), (6, 2, 7), (8, 4, 12), (12, 2, 7), (14, 2, 5)]


def _place(buf, start_s, sound):
    i = int(round(start_s * SR))
    n = min(len(sound), len(buf) - i)
    if n > 0:
        buf[i : i + n] += sound[:n]


def _kick():
    t = np.arange(int(0.25 * SR)) / SR
    f = 50 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def _snare(rng):
    t = np.arange(int(0.18 * SR)) / SR
    noise = rng.standard_normal(len(t))
    noise = np.convolve(noise, np.ones(4) / 4, mode="same") - np.convolve(noise, np.ones(64) / 64, mode="same")
    return (0.8 * noise + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 28)


def _hat(rng):
    t = np.arange(int(0.04 * SR)) / SR
    noise = rng.standard_normal(len(t))
    return (noise - np.convolve(noise, np.ones(8) / 8, mode="same")) * np.exp(-t * 120)


def _swell(hz, dur, level):
    """A dull note: no pluck, the tone fades in over 40 ms. Its level rises but its attack has no brightness."""
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.04) ** 2 * np.exp(-t * 2.5) * np.minimum(1, (dur - t) / 0.02)
    return level * env * (np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t))


def make_song(
    tempo=120.0,
    bars=12,
    lead_in_s=1.0,
    tail_s=1.5,
    seed=0,
    swing_ms=0.0,
    taste=False,
    double_stops=False,
    chords=CHORDS,
):
    """Returns dict(mix, stems={bass, drums, harmony}, notes=[(start, end, pitch)], beats, downbeats, tempo, sr).
    Audio arrays are stereo float32 (2, n). `swing_ms` delays every off-beat eighth of the bass (0 = straight).
    `chords` is the loop the song goes round, one entry a bar: (root MIDI for the bass, chord tones for the pad),
    or a list of two of them for a bar whose chord changes on beat 3. The bass's root need not be the chord's.
    With `taste` the song has two things that can be written either way: in two bars of three the second note,
    which repeats the first, has no pluck (the re-pluck policies differ on it), and in every other bar the last
    note comes a sixteenth early (the rhythm policies differ on that). With `double_stops` the note on beat 3
    of every fourth bar is played together with the fifth above it; `together` lists those as (start, [pitches])."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / tempo
    n = int((lead_in_s + bars * 4 * beat + tail_s) * SR)
    bass, drums, harmony = np.zeros(n), np.zeros(n), np.zeros(n)
    notes, beats, downbeats, together = [], [], [], []
    for bar in range(bars):
        t_bar = lead_in_s + bar * 4 * beat
        downbeats.append(t_bar)
        halves = chords[bar % len(chords)]
        halves = halves if isinstance(halves, list) else [halves]
        for b in range(4):
            t = t_bar + b * beat
            beats.append(t)
            _place(drums, t, (0.9 * _kick()) if b % 2 == 0 else (0.5 * _snare(rng)))
            for h in range(2):
                _place(drums, t + h * beat / 2, 0.12 * _hat(rng))
        held = 4 * beat / len(halves)  # how long each of the bar's chords lasts
        tt = np.arange(int(held * SR)) / SR
        for k, (_, tones) in enumerate(halves):
            pad = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * tt) for m in tones)
            _place(harmony, t_bar + k * held, 0.05 * pad * np.minimum(1, tt / 0.02) * np.minimum(1, (held - tt) / 0.05))
        pattern = PATTERN[:-2] + [(12, 1, 7), (13, 3, 5)] if taste and bar % 2 else PATTERN
        for s16, len16, up in pattern:
            start = t_bar + s16 * beat / 4 + (swing_ms / 1000 if s16 % 4 == 2 else 0)
            dur = len16 * beat / 4 * 0.95
            pitch = halves[s16 * len(halves) // 16][0] + up
            if taste and s16 == 4 and bar % 3:
                _place(bass, start, _swell(440 * 2 ** ((pitch - 69) / 12), dur, 0.35))
            else:
                _place(bass, start, 0.5 * pluck(440 * 2 ** ((pitch - 69) / 12), dur, 100))
            notes.append((start, start + dur, pitch))
            if double_stops and s16 == 8 and bar % 4 == 1:
                _place(bass, start, 0.5 * pluck(440 * 2 ** ((pitch + 7 - 69) / 12), dur, 100))
                together.append((start, [pitch, pitch + 7]))
    stereo = lambda y: np.stack([y, y]).astype(np.float32)
    stems = dict(bass=stereo(bass), drums=stereo(drums), harmony=stereo(harmony))
    mix = sum(stems.values())
    peak = float(np.abs(mix).max())
    scale = 0.9 / peak
    return dict(
        mix=mix * scale,
        stems={k: v * scale for k, v in stems.items()},
        notes=notes,
        together=together,
        beats=np.array(beats),
        downbeats=np.array(downbeats),
        tempo=tempo,
        sr=SR,
    )


def write_song(song, directory):
    """Writes mix.wav and stems/{bass,drums,harmony}.wav under `directory`; returns (mix path, stems dir)."""
    import os

    import soundfile as sf

    stems_dir = os.path.join(directory, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    mix_path = os.path.join(directory, "mix.wav")
    sf.write(mix_path, song["mix"].T, song["sr"], subtype="PCM_16")
    for name, y in song["stems"].items():
        sf.write(os.path.join(stems_dir, name + ".wav"), y.T, song["sr"], subtype="PCM_16")
    return mix_path, stems_dir
