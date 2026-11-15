"""A song's chords, as the tool reads them, against a transcription of the song: a MIDI file of the whole
arrangement (or anything MuseScore opens), lined up with the recording as for an answer key (reference.py).

    python dev/chords_against_midi.py WORKDIR SONG FILE [FILE...]

WORKDIR is a copilot working directory and SONG one of its songs; nothing is written there. For each file this
prints how much of the recording the file speaks of; how often each reading of the song's chords (one a bar,
eighth by eighth) has the root, and the root with major or minor, that the file's notes have; and where the
file's bar lines fall on the tool's bars. With several files it says how far they agree with each other on the
root, which is how far any of them can be trusted.

The chord of each eighth is read from the file's notes as chords_eval.py reads Slakh's (midi_classes): the
instruments other than bass and drums, matched to a chord with weights set by hand. A transcription of another
recording of the song will do, as far as the two line up. The song's separated tracks must be in the cache:
nothing is separated here.
"""

import collections
import os
import sys
import warnings

import chords_eval as CE
import numpy as np

from bass2sheet import copilot, harmony, judge, reference
from bass2sheet.common import SR, cache_home, sha1_file

MINOR = [harmony.QUALITIES.index(q) for q in ("m", "m7")]


def the_song(workdir, song_id):
    """What the tool has of a song: its machine data, its mix (mono), and what harmony.hear makes of it."""
    song = copilot.load_songs(workdir)[song_id]
    machine = copilot.read_json(os.path.join(copilot.song_dir(workdir, song_id), "machine.json"))
    separator, cache = machine["settings"]["separator"], cache_home()
    kept = os.path.join(cache, f"stems-{sha1_file(song['path'])[:16]}-{separator}.npz")
    if not song.get("stems_dir") and not os.path.exists(kept):
        sys.exit(f"{song_id}: its separated tracks are not in the cache (add the song again first)")
    wav, stems, *_ = judge.load_song(song["path"], cache, "cpu", song.get("stems_dir"), separator)
    heard = harmony.hear(stems, machine["notes"], machine["grid"], machine["origin"])
    judge._song.cache_clear()
    if heard is None:
        sys.exit(f"{song_id}: no instruments beside bass and drums")
    return machine, wav.mean(0), heard


def readings(machine, heard, length):
    """The tool's two readings, eighth by eighth: {name: (frames, 2) of root and whether the chord is minor,
    -1 where it has no chord}, and how sure the tool is of the finer one in each frame."""
    mids = (heard["edges"][:-1] + heard["edges"][1:]) / 2
    fine, sure = np.full((len(mids), 2), -1), np.zeros(len(mids))
    for c in harmony.read(heard, length)["chords"]:
        inside = (mids >= c["start"]) & (mids < c["end"])
        fine[inside], sure[inside] = (c["root"], int(c["quality"] in ("m", "m7"))), c["score"]
    coarse = np.full((len(mids), 2), -1)
    per_bar = {c["bar"]: c for c in machine["chords"] if float(c.get("score", 1.0)) > 0}
    for k, g in enumerate(heard["index"]):
        c = per_bar.get((int(g) - machine["origin"]) // 16 + 1) if g >= machine["origin"] else None
        if c is not None:
            coarse[k] = (c["root"], int(c["quality"] == "m"))
    return {"one a bar": coarse, "eighth by eighth": fine}, sure


def transcribed(path, mono, heard):
    """A file lined up with the recording: (its chord in each frame as (root, minor), -1 where it has none or
    does not speak of the recording; the alignment; the file)."""
    pm = reference.read(path)
    bass = reference.bass_track(pm)[0]
    al = reference.align(pm, mono, SR)
    notes = lambda k: [dict(start=a, end=b, pitch=p) for a, b, p in al.notes(pm, k)]  # noqa: E731
    played = [n for k, inst in enumerate(pm.instruments) if not inst.is_drum and k != bass for n in notes(k)]
    found, _ = CE.midi_classes(heard, played, notes(bass))
    mids = (heard["edges"][:-1] + heard["edges"][1:]) / 2
    covered = np.zeros(len(mids), bool)
    for a, b in al.covered():
        covered |= (mids >= a) & (mids <= b)
    chord = np.where(covered[:, None] & (found[:, :1] >= 0), np.c_[found[:, 0], np.isin(found[:, 1], MINOR)], -1)
    return chord, al, pm


def bar_lines(pm, al, machine):
    """Where the file's bar lines fall on the tool's bars: {beat of the tool's bar (1 to 4): share of them}."""
    grid, origin = np.asarray(machine["grid"]), machine["origin"]
    down = al.at(pm.get_downbeats())
    down = down[np.isfinite(down)]
    beats = collections.Counter(round(((int(np.argmin(np.abs(grid - t))) - origin) % 16) / 4) % 4 + 1 for t in down)
    return {beat: beats[beat] / max(1, len(down)) for beat in (1, 2, 3, 4)}


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    warnings.filterwarnings("ignore")
    workdir, song_id, files = argv[0], argv[1], argv[2:]
    machine, mono, heard = the_song(workdir, song_id)
    seconds = np.diff(heard["edges"])
    mine, sure = readings(machine, heard, len(mono) / SR)
    share = lambda mask, of: float((seconds * mask).sum() / max(1e-9, (seconds * of).sum()))  # noqa: E731
    kept, rows = [], collections.defaultdict(list)
    for path in files:
        try:
            chord, al, pm = transcribed(path, mono, heard)
        except Exception as e:  # a file that cannot be read or lined up is said and left out
            print(f"{os.path.basename(path)}: left out ({e})")
            continue
        there = chord[:, 0] >= 0
        if (seconds * there).sum() < 20:
            print(f"{os.path.basename(path)}: less than 20 s in common with the recording: left out")
            continue
        kept.append(chord)
        said = [f"{os.path.basename(path)}: {(seconds * there).sum():.0f} s of {len(mono) / SR:.0f} in common"]
        for name, est in mine.items():
            root = there & (est[:, 0] == chord[:, 0])
            rows[name].append((share(root, there), share(root & (est[:, 1] == chord[:, 1]), there)))
            said.append(f"{name}: root {rows[name][-1][0]:.2f}, with major or minor {rows[name][-1][1]:.2f}")
        firm = there & (sure >= 0.6)
        said.append(
            f"where the tool is sure ({share(firm, there):.0%}): root {share(firm & (mine['eighth by eighth'][:, 0] == chord[:, 0]), firm):.2f}"
        )
        on = bar_lines(pm, al, machine)
        said.append("its bar lines on the tool's beat " + ", ".join(f"{b}: {on[b]:.0%}" for b in on if on[b] >= 0.01))
        print(" | ".join(said), flush=True)
    if not kept:
        sys.exit("no file could be lined up with the recording")
    print(f"\n{song_id}, {len(kept)} transcription{'s' if len(kept) > 1 else ''}:")
    for name, got in rows.items():
        got = np.array(got)
        print(f"  {name}: root {got[:, 0].mean():.2f}, with major or minor {got[:, 1].mean():.2f}")
    pairs = [
        share((a[:, 0] >= 0) & (a[:, 0] == b[:, 0]), (a[:, 0] >= 0) & (b[:, 0] >= 0))
        for k, a in enumerate(kept)
        for b in kept[k + 1 :]
    ]
    if pairs:
        print(f"  the transcriptions agree with each other on the root {np.mean(pairs):.2f} of the time")


if __name__ == "__main__":
    main(sys.argv[1:])
