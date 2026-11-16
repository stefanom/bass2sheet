"""How often the chord a person would want is among the first few the copilot offers for a beat
(copilot_harmony.suggest), on songs whose chords are known.

    python dev/chord_offers_eval.py --rwc AUDIO_DIR ANNOTATIONS_DIR
    python dev/chord_offers_eval.py --song WORKDIR SONG FILE [FILE...]     (may be given several times)

--rwc takes the RWC Popular songs that chords_eval.py keeps for checking, with their annotated chords.
--song takes a song of a copilot working directory with one or more transcriptions of it (MIDI files of the
whole arrangement, lined up with the recording as chords_against_midi.py does); nothing is written there.

A stretch is what the person marks on the harmony page: from where a chord starts to where the next one starts,
a bar at most. The chord wanted is the annotated one as a major or minor triad, which is all the copilot offers
(a seventh is made from the triad on the page); other chords are left out. For each of four ways of ranking,
the share of stretches whose chord is in the first 1, 2, 3, 4, 5, 6 and 8 offered:

    bass notes only             what the bass line plays in the stretch
    bass notes + key            and how common each chord is in the key
    what the tool hears + key   the chords harmony.find reads from the record, and the key: no bass notes
    all of it                   what the page uses
"""

import argparse
import os
import sys
import warnings

import chords_against_midi as CM
import chords_eval as CE
import mir_eval
import numpy as np

from bass2sheet import copilot, harmony
from bass2sheet import copilot_harmony as H
from bass2sheet.common import SR, cache_home

WAYS = {
    "bass notes only": dict(key=0, heard=0, kind=0),
    "bass notes + key": dict(heard=0, kind=0),
    "what the tool hears + key": dict(first=0, held=0, fifth=0, third=0),
    "all of it": dict(),
}
TOPS = (1, 2, 3, 4, 5, 6, 8)
SHORTEST = 0.4  # seconds: a chord shorter than this is nothing a person marks


def annotated(ref):
    """[(start, end, root, "" or "m")] of the major and minor chords in mir_eval's (intervals, labels)."""
    intervals, labels = ref
    roots, bitmaps, _ = mir_eval.chord.encode_many(labels, False)
    out = []
    for (start, end), root, notes in zip(intervals, roots, bitmaps):
        if root >= 0 and end - start >= SHORTEST:
            if notes[4] and not notes[3]:
                out.append((float(start), float(end), int(root), ""))
            elif notes[3] and not notes[4] and notes[7]:
                out.append((float(start), float(end), int(root), "m"))
    return out


def transcribed(chord, edges):
    """[(start, end, root, "" or "m")] from the chord of each eighth that a transcription gives
    (chords_against_midi.transcribed): eighths with the same chord are one stretch."""
    out, k = [], 0
    while k < len(chord):
        j = k
        while j + 1 < len(chord) and tuple(chord[j + 1]) == tuple(chord[k]):
            j += 1
        if chord[k][0] >= 0 and edges[j + 1] - edges[k] >= SHORTEST:
            out.append((float(edges[k]), float(edges[j + 1]), int(chord[k][0]), "m" if chord[k][1] else ""))
        k = j + 1
    return out


def places(stretches, notes, home, heard, grid):
    """For each way of ranking: [the place of the wanted chord among those offered, counted from 1]."""
    grid = np.asarray(grid)
    out = {way: [] for way in WAYS}
    for start, end, root, quality in stretches:
        g = int(np.argmin(np.abs(grid - start)))
        bar = float(grid[min(len(grid) - 1, g + 16)] - grid[g]) or 2.0
        for way, weights in WAYS.items():
            offered = H.suggest(notes, start, min(end, start + bar), home, heard=heard, most=24, weights=weights)
            out[way].append(offered.index((root, quality)) + 1)
    return out


def table(title, found):
    print(f"\n{title}")
    for way in WAYS:
        got = np.array(found[way])
        if len(got):
            line = "  ".join(f"top {k}: {np.mean(got <= k):.0%}" for k in TOPS)
            print(f"  {way:27s} {line}   ({len(got)} chords; median place {np.median(got):.0f})")


def what_was_heard(found):
    return [
        dict(start=c["start"], end=c["end"], root=c["root"], quality=c["quality"], score=c["score"])
        for c in found["chords"]
    ]


def rwc(audio, annotations):
    cache = cache_home()
    total, by_song = {way: [] for way in WAYS}, []
    for rid in CE.CHECK:
        keep = os.path.join(cache, "chords-eval")
        song = CE.prepare(rid, os.path.join(audio, rid + ".wav"), cache, keep, False, "cpu")
        song["ref"] = CE.reference(annotations, rid)
        song = CE.listen(song, cache)
        found = harmony.read(song["heard"], song["length"])
        key = harmony.main_key(found, song["notes"])
        got = places(annotated(song["ref"]), song["notes"], (key[1], key[2]), what_was_heard(found), song["grid"])
        for way in WAYS:
            total[way] += got[way]
        by_song.append((rid, float(np.mean(np.array(got["all of it"]) <= 3))))
    table(f"RWC Popular, {len(CE.CHECK)} songs (annotated chords):", total)
    worst = sorted(by_song, key=lambda row: row[1])[:4]
    print("  the songs it is worst on (in the first three):", ", ".join(f"{rid} {share:.0%}" for rid, share in worst))


def one_song(workdir, song_id, files):
    machine, mono, frames = CM.the_song(workdir, song_id)
    notes = sorted(copilot.Desk(workdir, song_id).notes, key=lambda n: n["start"])
    found = harmony.read(frames, len(mono) / SR)
    key = harmony.main_key(found, machine["notes"])
    total = {way: [] for way in WAYS}
    for path in files:
        chord, *_ = CM.transcribed(path, mono, frames)
        got = places(
            transcribed(chord, frames["edges"]), notes, (key[1], key[2]), what_was_heard(found), machine["grid"]
        )
        for way in WAYS:
            total[way] += got[way]
    plural = "s" if len(files) > 1 else ""
    table(f"{song_id} ({len(files)} transcription{plural}; key taken as {key[0]}):", total)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rwc", nargs=2, metavar=("AUDIO_DIR", "ANNOTATIONS_DIR"))
    ap.add_argument("--song", nargs="+", action="append", default=[], metavar="WORKDIR SONG FILE")
    args = ap.parse_args()
    if not args.rwc and not args.song:
        ap.error("give --rwc or --song")
    if any(len(song) < 3 for song in args.song):
        ap.error("--song takes a working directory, a song and at least one file")
    warnings.filterwarnings("ignore")
    if args.rwc:
        rwc(*args.rwc)
    for workdir, song_id, *files in args.song:
        one_song(workdir, song_id, files)


if __name__ == "__main__":
    sys.exit(main())
