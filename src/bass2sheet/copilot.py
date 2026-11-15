"""The copilot: the person is in charge of the score, the tool does the legwork and keeps a record of its mistakes.

A song is prepared once: the tool hears it as it would for a sheet, and what it heard is kept as it is
(`machine.json`). The person then goes through the notes with the record playing, and fixes what is wrong:
another pitch, a note that is not there, one that is missing, two that are one, one that is two, a start or an
end. Their version lives beside the tool's (`notes.json`), every fix is logged (`log.jsonl`), and what they
played past without touching, as far as they have worked, counts as checked. The sheet is written from their
version.

The tool helps in two ways. For a wrong note it offers guesses, best first (`Desk.guesses`): what the other
witnesses heard there and what the song's own notes make it sound like. After a fix it offers the other notes
that carry the same label and sound the same (`Desk.alike`); the person decides.

`mistakes` compares the two versions: that is the tool's record of what it got wrong on this song, with the
right answer attached. Nothing here needs a browser; `copilot_server` puts it on a page.
"""

import collections
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import soundfile as sf

from . import harmony, judge, ownsound, together
from . import repeats as repeats_
from .common import DEFAULTS, LOWEST_NOTE, SR, Bass2SheetError, atomic_write_text, clock, note_name, step

LOW, HIGH = 26, 72  # the pitches the picture shows: D1 to C5
PER_SEMI = 3  # picture rows per semitone
PICTURE_SR, PICTURE_HOP = 3000, 30  # 10 ms a column
PICTURE_WINDOW, PICTURE_FFT = 300, 1024  # 100 ms: it keeps a low E's partials apart and still shows a note begin
PICTURE_KIND = "steady tones 1"  # a song prepared with another kind of picture gets this one when it is opened
SEMI_LO, SEMI_HI = 23, 76  # the pitches the pitch tracker's bins are summed into
SHORTEST = 0.03  # no note is shorter than this
UNDO_DEPTH = 200
SPEEDS = (90, 80, 70, 60, 50)  # the slower speeds a song can be played at, in percent
CARRIED = ("pitch", "shift", "delete", "add", "split", "merge")  # the fixes that are offered to the repeats
FIXES = ("pitch", "pitches", "also", "shift", "delete", "add", "move", "split", "merge", "repeats")
PITCHES = range(SEMI_LO, 85)


def song_dir(workdir, song_id):
    return os.path.join(workdir, song_id)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_songs(workdir):
    return read_json(os.path.join(workdir, "songs.json"), {})


def sieve_pitch(mono, note):
    """The pitch the spectrum's own peaks point to in a note, or None."""
    peaks = together.peaks_of(mono, note["start"], note["end"])
    scored = {m: together.strength(peaks, float(together.hz(m)))[0] for m in together.candidates(peaks)}
    scored = {m: s for m, s in scored.items() if s > 0}
    return max(scored, key=scored.get) if scored else None


# ------------------------------------------------------------------------------------------------- preparing


def picture(mono):
    """The steady tones of the bass track, by pitch and time, as bytes: rows from LOW up, PER_SEMI to a
    semitone, one column every PICTURE_HOP / PICTURE_SR seconds; 0 is silence, 255 the loudest (42 dB above).

    A bass note cannot be placed in pitch and in time at once by looking at how much sound each pitch holds:
    telling the low E from the F beside it takes most of a second of sound. But its partials lie a whole
    fundamental apart, and a tenth of a second is enough to keep them apart. So each partial is located
    exactly, in pitch and in time, from the phase of a short-window spectrum (a reassigned spectrogram), and
    only what behaves like a steady tone is kept: bins that agree with their neighbours on the frequency they
    are hearing. The moment one note gives way to the next, where they do not, is left blank."""
    import librosa

    y = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=SR, target_sr=PICTURE_SR)
    rows = PER_SEMI * (HIGH - LOW + 1)
    if len(y) < PICTURE_FFT:
        return np.zeros((rows, 1 + len(y) // PICTURE_HOP), np.uint8)
    freq, when, mag = librosa.reassigned_spectrogram(
        y, sr=PICTURE_SR, n_fft=PICTURE_FFT, hop_length=PICTURE_HOP, win_length=PICTURE_WINDOW, fill_nan=True
    )
    columns = mag.shape[1]
    row = np.rint(PER_SEMI * (69 + 12 * np.log2(np.maximum(freq, 1e-3) / 440) - LOW) + 1).astype(np.int64)
    column = np.rint(when * PICTURE_SR / PICTURE_HOP).astype(np.int64)
    apart = np.abs(np.diff(freq, axis=0)) / (PICTURE_SR / PICTURE_FFT)  # 0 between two bins on one steady tone
    steady = np.zeros(freq.shape, bool)
    steady[:-1] |= apart < 0.4
    steady[1:] |= apart < 0.4
    ok = steady & (row >= 0) & (row < rows) & (column >= 0) & (column < columns) & (mag > mag.max() * 1e-4)
    energy = np.bincount(
        row[ok] * columns + column[ok], weights=mag[ok].astype(np.float64) ** 2, minlength=rows * columns
    )
    a = np.pad(np.sqrt(energy.reshape(rows, columns)), 1, mode="edge")
    a = (
        (  # a breath of blur: a line one cell thick flickers when the page stretches it
            4 * a[1:-1, 1:-1]
            + 2 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
            + (a[:-2, :-2] + a[:-2, 2:] + a[2:, :-2] + a[2:, 2:])
        )
        / 16
    )
    ref = np.percentile(a[a > 0], 99.7) if (a > 0).any() else 1.0
    v = np.clip(1 + 20 * np.log10(np.maximum(a, 1e-12) / ref) / 42, 0, 1)
    return (v * 255).astype(np.uint8)


def picture_file(folder, mono):
    """Write a song's picture and return what the page needs to know about it."""
    pic = picture(mono)
    pic.tofile(os.path.join(folder, "picture.bin.part"))
    os.replace(os.path.join(folder, "picture.bin.part"), os.path.join(folder, "picture.bin"))
    return dict(
        kind=PICTURE_KIND,
        rows=int(pic.shape[0]),
        columns=int(pic.shape[1]),
        low=LOW,
        per_semitone=PER_SEMI,
        step=PICTURE_HOP / PICTURE_SR,
    )


def semitone_mass(probs):
    """The pitch tracker's bin probabilities summed into semitones: (frames, SEMI_LO..SEMI_HI), float16."""
    from .pitch import bin_of_hz

    centers = bin_of_hz(440.0 * 2 ** ((np.arange(SEMI_LO, SEMI_HI + 1) - 69) / 12))
    cum = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs.astype(np.float32), axis=1)], axis=1)
    lo = np.clip(np.round(centers - 2).astype(int), 0, probs.shape[1])
    hi = np.clip(np.round(centers + 2).astype(int) + 1, 0, probs.shape[1])
    return (cum[:, hi] - cum[:, lo]).astype(np.float16)


def prepare(workdir, song, opts=None, say=None):
    """Hear one song and lay out everything the page and the copilot need in its folder. The person's own
    version (notes.json) is made from what the tool heard only if there is none yet: preparing a song again,
    after the tool has changed, never touches their work."""
    from .common import sha1_array
    from .pitch import features

    opts, say = opts or {}, say or (lambda text: print(text, flush=True))
    folder = song_dir(workdir, song["id"])
    os.makedirs(os.path.join(folder, "audio"), exist_ok=True)
    mine = {**(opts.get("defaults") or {}), **(song.get("settings") or {})}
    settings = dict(DEFAULTS, **mine)
    kw = dict(
        cache_dir=opts.get("cache_dir"),
        device=opts.get("device"),
        stems_dir=song.get("stems_dir"),
        crepe_model=opts.get("crepe_model", "full"),
    )
    say(f"{song['title']}: hearing it")
    # The beat model is the default, not a must. Where the person has not asked for a tracker and the model
    # cannot be had (not downloaded and no network, its server down), the built-in tracker finds the beats.
    asked = settings if "beat_tracker" in mine else dict(settings, beat_tracker="auto")
    heard = judge.hear(song["path"], asked, keep=os.path.join(folder, "heard"), **kw)
    if heard.get("beat_tracker", settings["beat_tracker"]) != settings["beat_tracker"]:
        settings["beat_tracker"] = heard["beat_tracker"]
        say(f"{song['title']}: the beat model could not be used; the built-in beat tracker found the beats")
    audio = judge.SongAudio(song["path"], kw["cache_dir"], kw["device"], kw["stems_dir"], settings["separator"])
    mono = np.asarray(audio.bass_mono, dtype=np.float32)
    notes = [
        dict(
            id=f"m{k}",
            start=round(n["start"], 4),
            end=round(n["end"], 4),
            pitch=int(n["pitch"]),
            also=n.get("also", []),
        )
        for k, n in enumerate(heard["notes"])
    ]
    with step("checks") as found:  # each note against the song's other notes of its pitch
        own_sound = ownsound.check(mono, notes)
        found["doubted"] = sum(bool(h and ownsound.doubtful(h)) for h in own_sound)
    for n, h in zip(notes, own_sound):
        if h and ownsound.doubtful(h):
            n["doubt"] = dict(explained=round(h["explained"], 2), better=h["better"])
    for n, h in zip(notes, heard["notes"]):
        if h.get("flag") == "alone":  # MuScriptor's note, and bass2sheet's own finder does not hear it (muscriptor.py)
            n["doubt"] = dict(n.get("doubt", {}), alone=True)
    with step("picture") as found:  # what the page shows of the bass track, and what the guesses are made from
        feat = features(audio.bass_mono, audio.cache_dir, audio.device, kw["crepe_model"])
        shown = picture_file(folder, mono)
        found["columns"] = shown["columns"]
    suffix = "" if kw["crepe_model"] == "full" else f"-{kw['crepe_model']}"
    probs = np.load(os.path.join(audio.cache_dir, f"crepe-{sha1_array(audio.bass_mono)[:16]}{suffix}.npz"))["probs"]
    f0, per = feat["f0_c"], feat["per_c"]
    line = np.where((per >= 0.3) & (f0 > 0), 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440), np.nan).astype(np.float32)
    pyin = feat["f0_p"]
    pyin = np.where(
        np.isfinite(pyin) & (pyin > 0), 69 + 12 * np.log2(np.maximum(np.nan_to_num(pyin), 1e-6) / 440), np.nan
    )
    line.tofile(os.path.join(folder, "pitch.bin"))
    np.savez_compressed(os.path.join(folder, "desk.npz"), semis=semitone_mass(probs), pyin=pyin.astype(np.float32))
    peak = max(float(np.abs(audio.mix).max()), float(np.abs(audio.bass).max()), 1e-9)
    sf.write(os.path.join(folder, "audio", "record.flac"), (0.97 * audio.mix / peak).T, SR, subtype="PCM_16")
    sf.write(os.path.join(folder, "audio", "bass.flac"), 0.97 * mono / peak, SR, subtype="PCM_16")
    from .playalong import write_backing  # the song without its bass, to play along with

    write_backing(os.path.join(folder, "audio"), audio.rest, 0.97 / peak, mix=audio.mix, bass=audio.bass)
    grid = [round(float(t), 4) for t in heard["grid"]]
    machine = dict(
        code=judge.code_id(),
        settings=settings,
        notes=notes,
        grid=grid,
        origin=int(heard["origin"]),
        tempo=float(heard["tempo"]),
        key=[heard["key"][0], int(heard["key"][1]), bool(heard["key"][2])],
        chords=heard["chords"],
        harmony=heard.get("harmony"),  # the keys and chords over time (harmony.find)
        length=round(audio.n / SR, 3),
        picture=shown,
    )
    atomic_write_text(os.path.join(folder, "machine.json"), json.dumps(machine))
    if not os.path.exists(os.path.join(folder, "notes.json")):
        mine = dict(notes=[dict(n, state="machine") for n in notes], checked=[], seq=0, undo=[])
        atomic_write_text(os.path.join(folder, "notes.json"), json.dumps(mine))
    say(f"{song['title']}: {len(notes)} notes, {sum('doubt' in n for n in notes)} of them doubted")


def add(audio_paths, workdir, opts=None, settings=None, stems_dirs=None, judging=None, say=None):
    """Add songs and prepare each. `settings` overrides the tool's defaults for these songs; with `judging`, a
    judging directory, a song found there takes the settings its picks have settled on."""
    os.makedirs(workdir, exist_ok=True)
    opts = dict(read_json(os.path.join(workdir, "settings.json"), {}), **(opts or {}))
    atomic_write_text(os.path.join(workdir, "settings.json"), json.dumps(opts, indent=1))
    songs = load_songs(workdir)
    judged = {}
    if judging:
        jsongs, jopts = judge.read_json(judging, "songs.json", {}), judge.read_settings(judging)
        comps, picks = judge.load_comparisons(judging), judge.load_judgments(judging)
        for sid, s in jsongs.items():
            judged[sid] = dict(settings=judge.chain(s, comps, picks, jopts)[1], stems_dir=s.get("stems_dir"))
    for path in audio_paths:
        entry = judge.song_entry(path)
        known = judged.get(entry["id"], {})
        entry["settings"] = dict(known.get("settings") or {}, **(settings or {}))
        entry["stems_dir"] = (stems_dirs or {}).get(path) or known.get("stems_dir")
        songs[entry["id"]] = dict(songs.get(entry["id"], {}), **entry)
        atomic_write_text(os.path.join(workdir, "songs.json"), json.dumps(songs, indent=1))
        prepare(workdir, songs[entry["id"]], opts, say)
    return songs


# ------------------------------------------------------------------------------------------- slower, same pitch

_slowing = collections.defaultdict(threading.Lock)


def stretcher():
    """(name, path) of the program that slows audio down without changing its pitch: Rubber Band where it is
    installed (`brew install rubberband`), which keeps a bass line's attacks and low notes best; otherwise the
    ffmpeg that comes with this package, whose `atempo` is rougher and a hundredth of a second less exact."""
    for path in (shutil.which("rubberband"), "/opt/homebrew/bin/rubberband", "/usr/local/bin/rubberband"):
        if path and os.path.exists(path):
            return "rubberband", path
    import imageio_ffmpeg

    return "ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()


def slowed(workdir, song_id, name, percent):
    """The path of a song's record or isolated bass (`name`) at `percent` of its speed and the same pitch. Made
    on first use and kept; it lasts exactly 100 / percent times as long, so a moment of the song is found in it
    by dividing."""
    if name not in ("record", "bass", "nobass") or percent not in SPEEDS:  # nobass: playalong.py makes it
        raise Bass2SheetError(f"no {name} at {percent}%: the speeds are {', '.join(map(str, SPEEDS))}")
    folder = os.path.join(song_dir(workdir, song_id), "audio")
    src, out = os.path.join(folder, f"{name}.flac"), os.path.join(folder, f"{name}-{percent}.flac")
    if not os.path.exists(src):
        raise Bass2SheetError(f"{song_id} is not prepared")
    with _slowing[out]:
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            return out
        kind, program = stretcher()
        tmp = f"{out}.{os.getpid()}.wav"
        if kind == "rubberband":
            cmd = [program, "-q", "--fine", "--tempo", str(percent / 100), src, tmp]
        else:
            cmd = [
                program,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src,
                "-filter:a",
                f"atempo={percent / 100}",
                tmp,
            ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode != 0 or not os.path.exists(tmp):
                raise Bass2SheetError(f"could not slow the song down ({kind}): {(r.stderr or r.stdout)[-300:]}")
            y, sr = sf.read(tmp, dtype="float32", always_2d=True)
            want = round(sf.info(src).frames * 100 / percent)
            y = y[:want] if len(y) >= want else np.concatenate([y, np.zeros((want - len(y), y.shape[1]), np.float32)])
            peak = float(np.abs(y).max())
            sf.write(f"{out}.part", y / max(1.0, peak / 0.99), sr, subtype="PCM_16", format="FLAC")
            os.replace(f"{out}.part", out)
        except subprocess.TimeoutExpired as e:
            raise Bass2SheetError("slowing the song down took too long") from e
        finally:
            for leftover in (tmp, f"{out}.part"):
                if os.path.exists(leftover):
                    os.remove(leftover)
    return out


# ------------------------------------------------------------------------------------------- one song, open


def progress(machine, state):
    """How far the person is with a song, from the tool's version (`machine.json`) and theirs (`notes.json`)
    alone, with nothing else of the song loaded: how many `fixes` they made (notes changed, added or taken
    out), how far they have `reached` (see Desk.reached) and the spans they have `worked` through."""
    notes = state["notes"]
    have = {n["id"] for n in notes} | {i for n in notes for i in n.get("merged", [])}
    gone = [m["start"] for m in machine["notes"] if m["id"] not in have]
    marks = [n["start"] for n in notes if n["state"] != "machine"] + gone
    marks += [state["upto"]] if state.get("upto") is not None else []
    far = round(max(marks), 4) if marks else None
    worked = [] if far is None else [[a, min(b, far)] for a, b in state["checked"] if a < far]
    return dict(fixes=sum(n["state"] != "machine" for n in notes) + len(gone), reached=far, worked=worked)


def merged(intervals, gap=0.05):
    """Intervals sorted and joined where they touch or nearly do."""
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def inside(t, intervals):
    return any(a <= t <= b for a, b in intervals)


class Desk:
    """One prepared song, open for work: the tool's version, the person's, and what the copilot needs to help."""

    def __init__(self, workdir, song_id):
        self.workdir, self.song_id, self.folder = workdir, song_id, song_dir(workdir, song_id)
        self.machine = read_json(os.path.join(self.folder, "machine.json"))
        if self.machine is None:
            raise Bass2SheetError(f"{song_id} is not prepared: run `bass2sheet-copilot add SONG --dir {workdir}`")
        self.song = load_songs(workdir).get(song_id, dict(id=song_id, title=song_id, artist=""))
        self.state = read_json(os.path.join(self.folder, "notes.json"))
        self._mono = self._extra = self._bank = self._repeats = None
        self._spectra = {}
        if self.machine["picture"].get("kind") != PICTURE_KIND:  # prepared when the picture was made another way
            self.machine["picture"] = picture_file(self.folder, self.mono)
            atomic_write_text(os.path.join(self.folder, "machine.json"), json.dumps(self.machine))

    # ---- what it holds

    @property
    def notes(self):
        return self.state["notes"]

    @property
    def mono(self):
        if self._mono is None:
            self._mono = sf.read(os.path.join(self.folder, "audio", "bass.flac"), dtype="float32")[0]
        return self._mono

    @property
    def extra(self):
        if self._extra is None:
            self._extra = np.load(os.path.join(self.folder, "desk.npz"))
        return self._extra

    def note(self, note_id):
        for n in self.notes:
            if n["id"] == note_id:
                return n
        raise Bass2SheetError(f"no note {note_id}")

    @property
    def repeats(self):
        """The bars of the song that sound alike (repeats.Repeats), or None for a song of less than two bars.
        Measured from the bass track the first time it is asked for, and kept in the song's folder."""
        if self._repeats is None:
            path, m = os.path.join(self.folder, "repeats.npz"), self.machine
            if os.path.exists(path):
                with np.load(path) as z:
                    measured = {k: z[k] for k in ("alike", "sounding", "rise")}
            else:
                measured = repeats_.measure(self.mono, m["grid"], m["origin"])
                np.savez_compressed(path + ".part.npz", **measured)
                os.replace(path + ".part.npz", path)
            rep = repeats_.Repeats(measured, m["grid"], m["origin"])
            self._repeats = (rep, rep.tokens(m["notes"])[0])
        rep, _ = self._repeats
        return rep if rep.bars >= 2 else None

    def fixes(self):
        """How many notes the person has changed, added or taken out."""
        return progress(self.machine, self.state)["fixes"]

    def reached(self):
        """How far the person has worked through the song, in seconds; None before they have. People listen
        ahead, before they start fixing and after: playing past something does not make it theirs. What does:
        everything up to their last fix, or up to where they said they have checked (`upto`)."""
        return progress(self.machine, self.state)["reached"]

    def worked(self):
        """The spans of the song the person has worked through: what they played, as far as `reached`. What they
        left untouched there counts as right. The rest of what they played is only heard."""
        return progress(self.machine, self.state)["worked"]

    def trust(self):
        """Whose word each sixteenth is (repeats.Repeats.trust), as things stand."""
        return self.repeats.trust(self.notes, self._repeats[1], self.worked())

    def differences(self):
        """Where a bar is written unlike its repeats that sound the same there, for the page: each with when it
        is (`start`, `end`), what the repeat has there (`notes`, as they would lie in this bar) and an `id`."""
        rep = self.repeats
        if rep is None:
            return []
        found = rep.differences(self.notes, self._repeats[1], self.worked(), self.state.get("kept", []))
        return [self._item(d) for d in found]

    def taken(self):
        """The fixes taken from a repeat that can still be taken back: their names."""
        return [e["fix"] for e in self.state["undo"] if e.get("fix")]

    def _item(self, d):
        """A stretch of a bar that a repeat has differently, for the page: when it is, and what the repeat has
        there as it would lie in this bar."""
        rep = self.repeats
        g = d["bar"] * repeats_.SLOTS
        pieces = rep.pieces(self.notes, d["source"], d["bar"], d["k0"], d["k1"])
        return dict(
            d,
            id=f"{d['bar']}:{d['k0']}:{d['k1']}:{d['source']}",
            start=round(rep.slot_time(g + d["k0"]), 3),
            end=round(rep.slot_time(g + d["k1"] + 1), 3),
            notes=[dict(start=p["start"], end=p["end"], pitch=p["pitch"]) for p in pieces],
        )

    def preview(self, items):
        """The notes as they would be with `items` taken (as the "repeats" fix takes them). Nothing changes."""
        notes, seq = json.dumps(self.state["notes"]), self.state["seq"]
        try:
            for i in items:
                self._copy(int(i["source"]), int(i["bar"]), int(i["k0"]), int(i["k1"]), lambda n: None, [])
            return sorted(json.loads(json.dumps(self.notes)), key=lambda n: (n["start"], n["pitch"]))
        finally:
            self.state["notes"], self.state["seq"] = json.loads(notes), seq

    def strips(self, items=None, bar=None):
        """For the page, to look at a bar's repeats side by side and decide one at a time. With `items` (an
        offer's): the bars they would change. With `bar`: every bar that sounds like it, with the stretches where
        it is written unlike that bar while sounding the same. One entry a bar: `items` (each as `_item` gives
        it), `after` (the notes around the bar as they would be with its items taken: to hear the change before
        taking it) and `yours` (the person has been through this bar). A stretch the two bars already write the same
        is left out: a bar with no `items` has nothing to take."""
        rep = self.repeats
        if rep is None:
            return []
        by_bar = {}
        if bar is not None:
            if not 0 <= int(bar) < rep.bars:
                raise Bass2SheetError("no such bar")
            pitch, start, _ = rep.tokens(self.notes)
            for j in np.flatnonzero(rep.related[int(bar)]):
                by_bar[int(j)] = [
                    dict(source=int(bar), bar=int(j), k0=a, k1=b) for a, b in rep.unlike(pitch, start, int(bar), j)
                ]
        for i in items or []:
            d = dict(source=int(i["source"]), bar=int(i["bar"]), k0=int(i["k0"]), k1=int(i["k1"]))
            by_bar.setdefault(d["bar"], []).append(d)
        level, out = self.trust(), []
        pitch, start, _ = rep.tokens(self.notes)

        def differs(d):  # the two bars are written differently somewhere in the stretch: there is something to take
            ks = slice(d["k0"], d["k1"] + 1)
            there, here = (pitch[d["source"], ks], start[d["source"], ks]), (pitch[d["bar"], ks], start[d["bar"], ks])
            return bool((there[0] != here[0]).any() or (there[1] != here[1]).any())

        for j, its in sorted(by_bar.items()):
            a, b = rep.slot_time(j * repeats_.SLOTS), rep.slot_time(min((j + 1) * repeats_.SLOTS, rep.n))
            its = [d for d in its if differs(d)]
            then = self.preview(its) if its else []
            out.append(
                dict(
                    bar=j,
                    items=[self._item(d) for d in its],
                    after=[n for n in then if n["end"] > a - 1 and n["start"] < b + 1],
                    yours=bool((level[j] >= 1).any()),
                )
            )
        return out

    def _copy(self, source, bar, k0, k1, keep, made):
        """Make `bar` read as bar `source` does from sixteenth k0 to k1, and leave the rest of it alone. What
        `bar` has there is cut out: a note reaching in from before, or going on after, keeps the part outside.
        The repeat's notes are put in, as far into their sixteenths as they are there. A note of `bar` that
        starts on the same sixteenth as one of the repeat's stays the same note, with the repeat's pitch and
        end. Where the repeat's note comes from before the stretch, it carries on from this bar's note."""
        rep, notes = self.repeats, self.notes
        if not (0 <= source < rep.bars and 0 <= bar < rep.bars and 0 <= k0 <= k1 < repeats_.SLOTS and source != bar):
            raise Bass2SheetError("no such stretch of a bar")
        pieces = rep.pieces(notes, source, bar, k0, k1)
        g0, g1 = bar * repeats_.SLOTS + k0, bar * repeats_.SLOTS + k1 + 1
        d0, d1 = round(rep.slot_time(g0), 4), round(rep.slot_time(g1), 4)
        starting = {p["slot"]: p for p in pieces if p["starts_here"]}

        def fresh(**fields):
            self.state["seq"] += 1
            n = dict(id=f"u{self.state['seq']}", state="added", via="repeat", **fields)
            notes.append(n)
            made.append(n["id"])
            return n

        tails, stays = [], {}  # what the cut leaves after the stretch; this bar's notes that stay, by sixteenth
        for n in sorted(notes, key=lambda n: n["start"]):
            a, b = rep.cover(n)
            blip = a >= b and d0 <= n["start"] < d1
            if not blip and not (a < b and b > g0 and a < g1):
                continue
            keep(n)
            reaches_in, goes_on = a < b and a < g0, a < b and b > g1
            if blip or (not reaches_in and a not in starting and not goes_on):
                notes.remove(n)
                continue
            self._touch(n)
            if goes_on and (reaches_in or a in starting):  # the part after the stretch becomes a note of its own
                tail = fresh(start=d1, end=n["end"], pitch=n["pitch"], also=list(n.get("also", [])), split_from=n["id"])
                tails.append(tail)
                n["end"] = d1
            if reaches_in:
                n["end"] = d0
            elif a in starting:
                stays[a] = n
            else:  # it starts in the stretch, where the repeat has no note starting, and goes on after it
                n["start"] = d1
                tails.append(n)
            self._settle(n)
        for p in pieces:
            end = p["end"] if p["ends_here"] else d1
            n = stays.get(p["slot"]) if p["starts_here"] else None
            if n is not None:  # the same note: the repeat's pitch and end
                self._touch(n)
                n["pitch"], n["also"], n["end"] = p["pitch"], list(p["also"]), max(end, n["start"] + SHORTEST)
                n["via"] = "repeat"
            elif not p["starts_here"] and (
                left := next(
                    (x for x in notes if abs(x["end"] - d0) < 0.03 and x["start"] < d0 and x["pitch"] == p["pitch"]),
                    None,
                )
            ):
                keep(left)  # the repeat's note comes from before the stretch: this bar's note carries on
                self._touch(left)
                left["end"] = end
                n = left
            elif end - p["start"] >= SHORTEST:
                n = fresh(start=p["start"] if p["starts_here"] else d0, end=end, pitch=p["pitch"], also=list(p["also"]))
            else:
                continue
            if not p["ends_here"]:  # ... and goes on after it: it takes back what the cut left there
                tail = next((x for x in tails if x is not n and x["pitch"] == n["pitch"] and x in notes), None)
                if tail is not None:
                    n["end"] = tail["end"]
                    notes.remove(tail)
                    if tail["id"] in made:
                        made.remove(tail["id"])
            for x in notes:  # no two notes at once: its neighbours give way by the few milliseconds it needs
                if x is n or x["end"] <= n["start"] or x["start"] >= n["end"]:
                    continue
                if x["start"] < n["start"]:
                    keep(x)
                    self._touch(x)
                    x["end"] = n["start"]
                    self._settle(x)
                else:
                    n["end"] = max(x["start"], n["start"] + SHORTEST)
            self._settle(n)

    def bank(self):
        """The song's notes as samples (ownsound.Bank), as they stand now."""
        if self._bank is None:
            self._bank = ownsound.Bank(self.mono, self.notes, spectra=self._spectra)
        return self._bank

    def save(self):
        self.state["notes"].sort(key=lambda n: (n["start"], n["pitch"]))
        self.state["undo"] = self.state["undo"][-UNDO_DEPTH:]
        atomic_write_text(os.path.join(self.folder, "notes.json"), json.dumps(self.state))
        self._bank = None

    def log(self, entry):
        with open(os.path.join(self.folder, "log.jsonl"), "a") as f:
            f.write(json.dumps(dict(entry, when=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")

    # ---- the copilot's help

    def guesses(self, note_id, most=6):
        """Other pitches for a note, best guess first. Each witness votes: the spectrum's own peaks, the second
        pitch tracker, the first tracker's runner-up pitches, the octave either way; to that is added how well the
        song's notes of each pitch explain this one's sound."""
        notes = self.notes
        i = next(k for k, n in enumerate(notes) if n["id"] == note_id)
        n, p = notes[i], notes[i]["pitch"]
        a = int(n["start"] * 100)
        b = max(int(n["end"] * 100), a + 1)
        votes = collections.Counter()
        vote = lambda c, w: votes.update({int(c): w}) if c is not None and int(c) != p else None
        vote(sieve_pitch(self.mono, n), 1.0)
        heard = self.extra["pyin"][a:b]
        heard = heard[np.isfinite(heard)]
        vote(round(float(np.median(heard))) if len(heard) else None, 1.0)
        mass = (
            self.extra["semis"][a + 2 : b].astype(np.float32).sum(0) if b > a + 2 else np.zeros(SEMI_HI - SEMI_LO + 1)
        )
        runners = [SEMI_LO + int(k) for k in np.argsort(mass)[::-1] if mass[k] > 0 and SEMI_LO + int(k) != p][:2]
        for c, w in zip(runners, (1.0, 0.5)):
            vote(c, w)
        vote(p + 12, 0.5)
        vote(p - 12, 0.5)
        bank = self.bank()
        by_sound = {}
        if not n.get("also") and ownsound._length(bank.lasts[i]) is not None:
            own = bank.explained(i)
            for c in set(bank.pitches_near(p, 14)) | set(votes):
                e = bank.explained(i, c, length=own[1] if own else None)
                if e:
                    by_sound[c] = e[0]
        for c in set(votes) | set(sorted(by_sound, key=by_sound.get, reverse=True)[:6]):
            votes[c] += by_sound.get(c, 0.0)
        order = [c for c, _ in votes.most_common()] + sorted(
            (c for c in by_sound if c not in votes), key=lambda c: -by_sound[c]
        )
        return [c for c in order if LOWEST_NOTE <= c <= HIGH + 12 and c != p][:most]

    def alike(self, note_id, floor=0.7):
        """After a fix: the other notes that still carry the label this note had, that the person has neither
        touched nor played past, and that sound like it. Where the person has already left notes of that label
        alone (played past them), a note is offered only if it is more like the fixed one than like those.
        Returns (the label they carry, [note ids, most alike first])."""
        notes = self.notes
        i = next(k for k, n in enumerate(notes) if n["id"] == note_id)
        fixed = notes[i]
        if "was" not in fixed or fixed["was"]["pitch"] == fixed["pitch"]:
            return None, []
        label, bank, checked = fixed["was"]["pitch"], self.bank(), self.worked()
        mine = ownsound._length(bank.lasts[i])
        if mine is None:
            return label, []

        def likeness(j, k):
            length = min(ownsound._length(bank.lasts[j]), ownsound._length(bank.lasts[k]))
            a, b = bank.spec(j, length), bank.spec(k, length)
            return ownsound.overlap(a, b) if a is not None and b is not None else 0.0

        same = [
            j
            for j, n in enumerate(notes)
            if j != i and n["pitch"] == label and not n.get("also") and ownsound._length(bank.lasts[j]) is not None
        ]
        right = [j for j in same if notes[j]["state"] == "machine" and inside(notes[j]["start"], checked)][:20]
        offered = []
        for j in same:
            if notes[j]["state"] != "machine" or inside(notes[j]["start"], checked):
                continue
            s = likeness(j, i)
            if s < floor:
                continue
            if len(right) >= 3 and s < float(np.median([likeness(j, k) for k in right])) + 0.05:
                continue
            offered.append((s, notes[j]["id"]))
        return label, [nid for _, nid in sorted(offered, reverse=True)]

    # ---- the person's fixes

    def _touch(self, n):
        """Called before a machine note changes: remember what the tool had."""
        if n["state"] == "machine":
            n["was"] = dict(start=n["start"], end=n["end"], pitch=n["pitch"], also=list(n.get("also", [])))
            n["state"] = "fixed"

    def _settle(self, n):
        """Called after a change: a note back to what the tool had is the tool's again."""
        w = n.get("was")
        if w and (n["start"], n["end"], n["pitch"], list(n.get("also", []))) == (
            w["start"],
            w["end"],
            w["pitch"],
            w["also"],
        ):
            if not n.get("merged"):
                n["state"] = "machine"
                del n["was"]
                n.pop("via", None)

    def _check_pitch(self, pitch):
        if not isinstance(pitch, int) or pitch not in PITCHES:
            raise Bass2SheetError(f"{pitch} is not a pitch a bass plays")
        return pitch

    def act(self, action):
        """Apply one fix (see the module's docstring for the kinds), log it and save. Returns what the page needs
        next: for a change of pitch, the other notes that sound the same (`offer`)."""
        kind = action.get("kind")
        if kind == "undo":
            return self.undo(action.get("fix"))
        before, made, result, seen = [], [], {}, set()

        def keep(n):  # a note as it was before this fix, once
            if n["id"] not in seen and n["id"] not in made:
                seen.add(n["id"])
                before.append(json.loads(json.dumps(n)))

        notes, rep = self.notes, self.repeats
        repitched = kind == "pitch" or (kind == "move" and "pitch" in action)  # a note dragged to another pitch
        carries = rep is not None and (kind in CARRIED or repitched) and action.get("offer", True)
        written = rep.tokens(notes) if carries else None
        kept, upto = len(self.state.get("kept", [])), self.state.get("upto")
        try:
            self._apply(kind, action, notes, keep, made, result)
        except Exception:  # nothing half done: the notes are as they were
            ids = {b["id"] for b in before} | set(made)
            self.state["notes"] = sorted(
                [n for n in notes if n["id"] not in ids] + before, key=lambda n: (n["start"], n["pitch"])
            )
            self.state["kept"] = self.state.get("kept", [])[:kept]
            self.state["upto"] = upto
            self._bank = None
            raise
        if kind == "refuse":
            return {}
        entry = dict(kind=kind, before=before, made=made, kept=kept, upto=upto)
        if kind == "repeats":  # a fix taken from a repeat has a name: it can be taken back on its own (`undo`)
            self.state["fixseq"] = entry["fix"] = result["fix"] = self.state.get("fixseq", 0) + 1
        self.state["undo"].append(entry)
        self.save()
        self.log(dict(action, song=self.song_id, before=before, made=made))
        if carries:  # the other bars that sound the same there and are written as this one was
            settled = rep.within(self.state.get("kept", []))
            items = rep.same_mistake(written, rep.tokens(self.notes), self.trust(), settled)
            if items:
                result["offer"] = dict(type="repeat", items=items, bars=len({i["bar"] for i in items}))
        if repitched and action.get("offer", True) and "offer" not in result:
            label, ids = self.alike(action["id"])
            if ids:
                pitch = self.note(action["id"])["pitch"]
                result["offer"] = dict(type="alike", seed=action["id"], label=label, pitch=pitch, ids=ids)
        return result

    def _apply(self, kind, action, notes, keep, made, result):
        if kind == "pitch":
            n = self.note(action["id"])
            keep(n)
            self._touch(n)
            n["pitch"] = self._check_pitch(action["pitch"])
            n["also"] = [q for q in n.get("also", []) if q != n["pitch"]]
            self._settle(n)
        elif kind == "pitches":  # an offer taken: the same fix for several notes
            pitch = self._check_pitch(action["pitch"])
            for nid in action["ids"]:
                n = self.note(nid)
                keep(n)
                self._touch(n)
                n["pitch"] = pitch
                self._settle(n)
        elif kind == "refuse":  # an offer turned down: nothing changes, but it is worth knowing
            self.log(dict(action, song=self.song_id))
        elif kind == "repeats":  # stretches of bars made to read as their repeats do
            if self.repeats is None or not action.get("items"):
                raise Bass2SheetError("there is nothing to take from a repeat")
            for item in action["items"]:
                self._copy(int(item["source"]), int(item["bar"]), int(item["k0"]), int(item["k1"]), keep, made)
        elif kind == "upto":  # "I have checked the song up to here": what was left as it is before that is right
            time_ = float(action["time"])
            if not 0 <= time_ <= self.machine["length"] + 1:
                raise Bass2SheetError("that is not a moment of the song")
            self.state["upto"] = round(time_, 3)
        elif kind == "keep":  # what is written here stands, whatever the repeats have
            start, end = float(action["start"]), float(action["end"])
            if not end > start:
                raise Bass2SheetError("nothing to keep")
            self.state.setdefault("kept", []).append([round(start, 3), round(end, 3)])
        elif kind == "also":
            n = self.note(action["id"])
            keep(n)
            self._touch(n)
            n["also"] = sorted({self._check_pitch(q) for q in action["also"]} - {n["pitch"]})
            self._settle(n)
        elif kind == "shift":  # several notes moved together, by so many semitones
            chosen = [self.note(nid) for nid in action["ids"]]
            for n in chosen:
                self._check_pitch(n["pitch"] + int(action["by"]))
            for n in chosen:
                keep(n)
                self._touch(n)
                n["pitch"] += int(action["by"])
                n["also"] = [self._check_pitch(q + int(action["by"])) for q in n.get("also", [])]
                self._settle(n)
        elif kind == "delete":
            for n in [self.note(nid) for nid in action.get("ids") or [action["id"]]]:
                keep(n)
                notes.remove(n)
        elif kind == "add":
            start, end = float(action["start"]), float(action["end"])
            later = [x["start"] for x in notes if x["start"] > start + 1e-6]
            end = min([end, *later])  # it ends where the next note starts, at the latest
            if not 0 <= start < end - SHORTEST + 1e-9 or end > self.machine["length"] + 1:
                raise Bass2SheetError("there is no room for a note here")
            for x in [x for x in notes if x["start"] <= start + 1e-6 < x["end"]]:  # the note sounding here gives way
                if start - x["start"] < SHORTEST:
                    raise Bass2SheetError("a note starts right here: change that one")
                keep(x)
                self._touch(x)
                x["end"] = round(start, 4)
                self._settle(x)
            self.state["seq"] += 1
            n = dict(
                id=f"u{self.state['seq']}",
                start=round(start, 4),
                end=round(end, 4),
                pitch=self._check_pitch(action["pitch"]),
                also=[],
                state="added",
            )
            notes.append(n)
            made.append(n["id"])
            result["id"] = n["id"]
        elif kind == "move":
            n = self.note(action["id"])
            order = sorted(notes, key=lambda x: x["start"])
            k = order.index(n)
            prev, nxt = (order[k - 1] if k else None), (order[k + 1] if k + 1 < len(order) else None)
            start = max(0.0, float(action.get("start", n["start"])))
            end = float(action.get("end", n["end"]))
            # dragged on the page, a note moves in time and in pitch in one go: one fix
            pitch = self._check_pitch(action["pitch"]) if "pitch" in action else n["pitch"]
            if prev is not None:
                start = max(start, prev["start"] + SHORTEST)
            if nxt is not None:
                end = min(end, nxt["start"])
            end = min(end, self.machine["length"])
            if end - start < SHORTEST:
                raise Bass2SheetError("that would leave no note")
            keep(n)
            self._touch(n)
            if prev is not None and prev["end"] > start:  # the note before gives way
                keep(prev)
                self._touch(prev)
                prev["end"] = round(start, 4)
                self._settle(prev)
            n["start"], n["end"] = round(start, 4), round(end, 4)
            if pitch != n["pitch"]:
                n["pitch"] = pitch
                n["also"] = [q for q in n.get("also", []) if q != pitch]
            self._settle(n)
        elif kind == "split":
            n = self.note(action["id"])
            at = float(action["at"])
            if not n["start"] + SHORTEST <= at <= n["end"] - SHORTEST:
                raise Bass2SheetError("a note is split somewhere inside it")
            keep(n)
            self._touch(n)
            self.state["seq"] += 1
            second = dict(
                id=f"u{self.state['seq']}",
                start=round(at, 4),
                end=n["end"],
                pitch=n["pitch"],
                also=list(n.get("also", [])),
                state="added",
                split_from=n["id"],
            )
            n["end"] = round(at, 4)
            notes.append(second)
            made.append(second["id"])
            result["id"] = second["id"]
        elif kind == "merge":
            # one note out of several that follow each other: the next one; with `run`, every repeat of the
            # pitch that follows without a gap; with `ids`, those notes
            order = sorted(notes, key=lambda x: x["start"])
            if action.get("ids"):
                chosen = sorted((self.note(nid) for nid in action["ids"]), key=lambda x: x["start"])
                k = order.index(chosen[0])
                if len(chosen) < 2 or order[k : k + len(chosen)] != chosen:
                    raise Bass2SheetError("only notes that follow each other can be joined")
            else:
                k = order.index(self.note(action["id"]))
                j = k + 1
                if action.get("run"):
                    while (
                        j < len(order)
                        and order[j]["pitch"] == order[k]["pitch"]
                        and order[j]["start"] - order[j - 1]["end"] < 0.06
                    ):
                        j += 1
                    if j == k + 1:
                        raise Bass2SheetError("the next note is not a repeat of this one")
                else:
                    j = k + 2
                    if j > len(order):
                        raise Bass2SheetError("there is no note after this one to join it with")
                chosen = order[k:j]
            n = chosen[0]
            for x in chosen:
                keep(x)
            self._touch(n)
            n["end"] = chosen[-1]["end"]
            n["merged"] = [*n.get("merged", []), *(i for x in chosen[1:] for i in (x["id"], *x.get("merged", [])))]
            for x in chosen[1:]:
                notes.remove(x)
            result["joined"] = len(chosen)
        else:
            raise Bass2SheetError(f"unknown kind of fix: {kind}")

    def undo(self, fix=None):
        """Take back the last fix. With `fix`, the name `act` gave a fix taken from a repeat: take back that one,
        whatever came after it, as long as nothing since has changed the same notes."""
        stack = self.state["undo"]
        if not stack:
            return dict(undone=None)

        def touched(entry):
            return {b["id"] for b in entry["before"]} | set(entry["made"])

        at = len(stack) - 1 if fix is None else next((k for k, e in enumerate(stack) if e.get("fix") == fix), None)
        if at is None:
            raise Bass2SheetError("that fix can no longer be taken back on its own")
        if any(touched(stack[at]) & touched(later) for later in stack[at + 1 :]):
            raise Bass2SheetError("these notes were changed again since: take back the later changes first")
        last = stack.pop(at)
        ids = touched(last)
        self.state["notes"] = [n for n in self.notes if n["id"] not in ids] + last["before"]
        if "kept" in last and fix is None:
            self.state["kept"] = self.state.get("kept", [])[: last["kept"]]
        if "upto" in last and fix is None:
            self.state["upto"] = last["upto"]
        self.save()
        self.log(dict(kind="undo", song=self.song_id, undone=last["kind"]))
        return dict(undone=last["kind"], ids=[b["id"] for b in last["before"]])

    def mark_checked(self, t0, t1):
        """The person played from t0 to t1: what they did not touch there, they found right."""
        self.state["checked"] = merged([*self.state["checked"], [max(0.0, t0), min(t1, self.machine["length"])]])
        self.state["at"] = round(min(t1, self.machine["length"]), 2)  # where they stopped: where they pick up again
        atomic_write_text(os.path.join(self.folder, "notes.json"), json.dumps(self.state))
        self.log(dict(kind="checked", song=self.song_id, start=round(t0, 3), end=round(t1, 3)))

    # ---- for the page

    def public(self):
        m = self.machine
        bars = [dict(bar=k + 1, time=m["grid"][i]) for k, i in enumerate(range(m["origin"], len(m["grid"]), 16))]
        pitches = [q for n in self.notes for q in (n["pitch"], *n.get("also", []))] or [28, 52]
        low = max(LOW, min(pitches) - 3)
        high = min(HIGH, max(max(pitches) + 3, low + 24))
        return dict(
            id=self.song_id,
            finder=m["settings"].get("notes", "own"),
            title=self.song.get("title", self.song_id),
            artist=self.song.get("artist", ""),
            length=m["length"],
            tempo=m["tempo"],
            bars=bars,
            picture=m["picture"],
            low=low,
            high=high,
            notes=self.notes,
            checked=self.state["checked"],
            worked=self.worked(),
            at=self.state.get("at"),
            undo=len(self.state["undo"]),
            fixes=self.fixes(),
            taken=self.taken(),
            names={str(p): note_name(p) for p in PITCHES},
            repeats=self.patterns(),
        )

    def patterns(self):
        """For the page: each bar's pattern (a letter; None for a bar that does not repeat) and the bars that
        sound like it (`of`, counted from 0), and the differences as things stand."""
        rep = self.repeats
        if rep is None:
            return dict(bars=[], differences=[])
        bars = [
            dict(pattern=repeats_.letter(int(f)) if f >= 0 else None, of=np.flatnonzero(rep.related[i]).tolist())
            for i, f in enumerate(rep.family)
        ]
        return dict(bars=bars, differences=self.differences())


# --------------------------------------------------------------------------------- what the tool got wrong


def mistakes(workdir, song_id):
    """The tool's version of a song against the person's: one row per difference, and what was found right.

    Kinds: `wrong octave`, `wrong pitch`, `extra note` (the person deleted it), `missed note` (they added it),
    `two notes written as one` (they split it), `one note written as two` (they joined them), `start` and `end`
    (moved by more than 50 ms), `second note` (a note played together added or removed). `right` counts the
    notes left as the tool had them in what the person worked through (`Desk.worked`); `unchecked` the rest."""
    desk = Desk(workdir, song_id)
    mine = {n["id"]: n for n in desk.notes}
    absorbed = {i for n in desk.notes for i in n.get("merged", [])}
    checked, rows, right, unchecked = desk.worked(), [], 0, 0
    row = lambda kind, m, n=None: rows.append(
        dict(kind=kind, time=round(m["start"], 2), had=note_name(m["pitch"]), now=note_name(n["pitch"]) if n else None)
    )
    for m in desk.machine["notes"]:
        n = mine.get(m["id"])
        if n is None:
            row("one note written as two" if m["id"] in absorbed else "extra note", m)
            continue
        if n["state"] == "machine":
            right += inside(m["start"], checked)
            unchecked += not inside(m["start"], checked)
            continue
        if n["pitch"] != m["pitch"]:
            row("wrong octave" if (n["pitch"] - m["pitch"]) % 12 == 0 else "wrong pitch", m, n)
        if sorted(n.get("also", [])) != sorted(m.get("also", [])):
            row("second note", m, n)
        if n.get("merged"):
            continue  # its end moved because the next note was joined to it: counted there
        if abs(n["start"] - m["start"]) > 0.05:
            row("start", m, n)
        split = any(x.get("split_from") == m["id"] for x in desk.notes)
        if split:
            row("two notes written as one", m, n)
        elif abs(n["end"] - m["end"]) > 0.05:
            row("end", m, n)
    for n in desk.notes:
        if n["state"] == "added" and not n.get("split_from"):
            rows.append(dict(kind="missed note", time=round(n["start"], 2), had=None, now=note_name(n["pitch"])))
    rows.sort(key=lambda r: r["time"])
    return dict(
        song=song_id,
        finder=desk.machine["settings"].get("notes", "own"),
        notes=len(desk.machine["notes"]),
        right=right,
        unchecked=unchecked,
        checked_seconds=round(sum(b - a for a, b in checked), 1),
        length=desk.machine["length"],
        kinds=dict(collections.Counter(r["kind"] for r in rows).most_common()),
        rows=rows,
    )


def report(workdir):
    """The mistakes so far, as Markdown: per song, how much was checked and what was wrong, by kind."""
    lines = ["# What the tool got wrong", ""]
    for song_id, song in load_songs(workdir).items():
        if not os.path.exists(os.path.join(song_dir(workdir, song_id), "machine.json")):
            continue
        m = mistakes(workdir, song_id)
        if song.get("test") and not m["kinds"] and not m["checked_seconds"]:
            continue  # a test song nobody has worked on: testsongs.report speaks of those
        share = m["checked_seconds"] / max(m["length"], 1e-9)
        lines += [f"## {song.get('title', song_id)}", ""]
        lines += [
            f"Worked through {clock(m['checked_seconds'])} of {clock(m['length'])} ({share:.0%}). Of "
            f"{'MuScriptor' if m['finder'] == 'muscriptor' else 'the tool'}'s "
            f"{m['notes']} notes, {m['right']} were left as they were there; {m['unchecked']} are not checked yet.",
            "",
        ]
        if m["kinds"]:
            lines += ["| Mistake | Times |", "|---|---|", *(f"| {k} | {v} |" for k, v in m["kinds"].items()), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


# ------------------------------------------------------------------------------------------------- the sheet


def sheet(workdir, song_id, mscore=None, tab=None, rhythm=None):
    """Write the score from the person's notes: MusicXML, MIDI and, with MuseScore, the PDF and a picture of each
    page for the copilot's own page to show (`score_made`). The line under the title says how much of it they
    have played through. Returns {file name: path}."""
    from . import render, score
    from .pipeline import bar_times
    from .rhythm import quantize, simplify

    desk = Desk(workdir, song_id)
    m, out = desk.machine, os.path.join(desk.folder, "sheet")
    os.makedirs(out, exist_ok=True)
    grid = np.array(m["grid"])
    notes = [
        dict(
            start=n["start"],
            end=n["end"],
            pitch=n["pitch"],
            flag=None,
            **({"also": n["also"]} if n.get("also") else {}),
        )
        for n in sorted(desk.notes, key=lambda n: n["start"])
    ]
    notes = quantize(notes, grid, m["origin"])
    if (rhythm or m["settings"]["rhythm"]) == "simple":
        notes = simplify(notes)
    checked = desk.worked()
    share = sum(inside(n["start"], checked) for n in notes) / max(1, len(notes))
    done = share >= 0.995
    subtitle = (
        "Bass: checked by ear against the record"
        if done
        else f"Bass: {share:.0%} checked by ear against the record; the rest is the machine's transcription"
    )
    xml = score.musicxml(
        notes,
        harmony.for_sheet(m),
        tuple(m["key"]),
        m["tempo"],
        float(grid[m["origin"]]),
        desk.song.get("title", song_id),
        desk.song.get("artist", ""),
        {},
        draft=False,
        tab_style=tab or m["settings"]["tab"],
        bar_times=bar_times(grid, m["origin"]),
        subtitle=subtitle,
    )
    files = dict(musicxml=os.path.join(out, "bass.musicxml"), midi=os.path.join(out, "bass_as_played.mid"))
    atomic_write_text(files["musicxml"], xml)
    render.write_midi(notes, m["tempo"], files["midi"], steady=False)
    ms, pages = score.find_mscore(mscore), []
    for name in ("bass.pdf", "sheet.json"):  # what is left of the sheet as it was made before
        if os.path.exists(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    if ms:
        err, pages = score.engrave_pages(files["musicxml"], os.path.join(out, "bass.pdf"), ms)
        if err:
            raise Bass2SheetError(f"the PDF was not written: {err}")
        files["pdf"] = os.path.join(out, "bass.pdf")
    made = dict(
        when=time.strftime("%Y-%m-%dT%H:%M:%S"),
        notes=len(notes),
        checked=round(share, 3),
        fixes=desk.fixes(),
        worked=round(sum(b - a for a, b in checked), 1),
        files={k: os.path.basename(v) for k, v in files.items()},
        pages=[os.path.basename(p) for p in pages],
    )
    # Every score that is made is kept as a version, in a folder of its own beside the latest: sheet/v1, v2, ...
    # A score made again with nothing changed is the version it was.
    kept = score_versions(workdir, song_id)
    last = kept[0] if kept else None
    if last is not None and bool(last["pages"]) == bool(pages):
        with open(os.path.join(out, f"v{last['version']}", last["files"]["musicxml"])) as f:
            if f.read() == xml:
                made = last
    if made is not last:
        made["version"] = last["version"] + 1 if last else 1
        folder = os.path.join(out, f"v{made['version']}")
        os.makedirs(folder, exist_ok=True)
        for name in (*made["files"].values(), *made["pages"]):
            shutil.copyfile(os.path.join(out, name), os.path.join(folder, name))
        atomic_write_text(os.path.join(folder, "sheet.json"), json.dumps(made))
    atomic_write_text(os.path.join(out, "sheet.json"), json.dumps(made))
    desk.log(dict(kind="sheet", song=song_id, notes=len(notes), checked=round(share, 3), version=made["version"]))
    return files


def score_made(workdir, song_id, version=None):
    """The score of a song as it was last made, or one of its earlier versions, for the page that shows it: when
    it was made, from how many fixes and how much worked through, the `files` to download and the pictures of its
    `pages` (file names; no pages where MuseScore was not there to draw them). A score with a `version` has its
    files in `sheet/v<version>`; the latest has them in `sheet` as well. None before a score is made, and for a
    version there is not."""
    folder = os.path.join(song_dir(workdir, song_id), "sheet")
    if version is not None:
        return read_json(os.path.join(folder, f"v{int(version)}", "sheet.json"))
    made = read_json(os.path.join(folder, "sheet.json"))
    if made is None:  # the last try broke off (MuseScore failed): the score last made is the latest version kept
        return next(iter(score_versions(workdir, song_id)), None)
    return made


def score_versions(workdir, song_id):
    """Every version kept of a song's score (see `score_made`), the latest first."""
    folder = os.path.join(song_dir(workdir, song_id), "sheet")
    names = os.listdir(folder) if os.path.isdir(folder) else []
    numbers = sorted((int(d[1:]) for d in names if d[:1] == "v" and d[1:].isdigit()), reverse=True)
    return [made for made in (score_made(workdir, song_id, n) for n in numbers) if made]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="bass2sheet-copilot",
        description="You are in charge of the score: fix the tool's notes against the record; it learns from that.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="add songs and prepare them")
    a.add_argument("songs", nargs="+", help="audio files you have the rights to use")
    a.add_argument("--judging", help="a bass2sheet-judge directory: its songs take the settings your picks settled on")
    a.add_argument(
        "--notes",
        choices=["own", "muscriptor"],
        help="whose notes you start from: bass2sheet's own (default), or MuScriptor's with bass2sheet's as the "
        "second opinion (a separate program with non-commercial weights: see docs/muscriptor.md)",
    )
    a.add_argument("--cache-dir", help="where models, stems and pitch tracks are cached")
    a.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    s = sub.add_parser("serve", help="open the page on this machine")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument(
        "--lan",
        action="store_true",
        help="listen on the home network too, so a phone can open the play-along page; it prints the address. "
        "Other machines can listen and read along, never change a score",
    )
    h = sub.add_parser("sheet", help="write the score of one song from your notes")
    h.add_argument("song", help="the song's id, as `report` lists it")
    h.add_argument("--mscore", help="path to the MuseScore 4 binary")
    r = sub.add_parser("report", help="print what the tool got wrong so far")
    t = sub.add_parser(
        "testset",
        help="add the songs of an evaluation dataset as test songs: hear what the tool's scores are measured on "
        "and say, at /tests, which are like your music",
    )
    t.add_argument("dataset", help="a dataset directory bass2sheet-eval knows")
    t.add_argument("--annotations", help="for RWC Popular: a checkout of github.com/rwc-music/rwc-annotations")
    t.add_argument(
        "--scores",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="a bass2sheet-eval run directory and the name its scores go by (repeat for each note finder); the "
        "songs of the first one are the ones added",
    )
    t.add_argument("--name", help="what to call the set (default: the kind of dataset)")
    t.add_argument("--tracks", help="comma-separated track ids, in place of the first run's")
    t.add_argument("--cache-dir", help="where models, stems and pitch tracks are cached")
    t.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    k = sub.add_parser(
        "reference",
        help="take somebody's transcription of a song (a MIDI file, a Guitar Pro tab, a MusicXML score) as an "
        "answer key: it is lined up with the recording, and the notes are scored against its bass part",
    )
    k.add_argument("song", help="the song's id, as `report` lists it")
    k.add_argument("file", help="the transcription: .mid, or anything MuseScore opens (.gp, .gpx, .gp5, .musicxml)")
    k.add_argument("--part", type=int, help="the number of the part to take as the bass, when the wrong one is found")
    k.add_argument("--mscore", help="path to the MuseScore 4 binary")
    for p in (a, s, h, r, t, k):
        p.add_argument("--dir", default="scores", help="working directory (default ./scores)")
    args = ap.parse_args(argv)
    try:
        if args.command == "add":
            missing = [p for p in args.songs if not os.path.isfile(p)]
            if missing:
                raise Bass2SheetError(f"no such file: {missing[0]}")
            opts = {k: v for k, v in dict(cache_dir=args.cache_dir, device=args.device).items() if v}
            chosen = dict(notes=args.notes) if args.notes else None
            add(args.songs, args.dir, opts, settings=chosen, judging=args.judging)
            print(f"Ready. Open the page with: bass2sheet-copilot serve --dir {args.dir}")
        elif args.command == "serve":
            from .copilot_server import serve

            serve(args.dir, args.port, args.lan)
        elif args.command == "sheet":
            for name, path in sheet(args.dir, args.song, args.mscore).items():
                print(f"{name}: {path}")
        elif args.command == "reference":
            from . import reference

            name, key = reference.add(args.dir, args.song, args.file, args.part, args.mscore)
            others = ", ".join(f'{o["index"]} "{o["name"]}" ({o["notes"]} notes)' for o in key["others"])
            print(
                f'Took part {key["part"]["index"]} "{key["part"]["name"]}" ({key["part"]["notes"]} notes) as the '
                f"bass" + (f"; it could also have been {others} (--part N)" if others else "") + "."
            )
            print(reference.report(args.dir), end="")
        elif args.command == "testset":
            from . import testsongs

            if any("=" not in x for x in args.scores):
                raise Bass2SheetError("--scores takes NAME=RUN_DIR")
            runs = dict(x.split("=", 1) for x in args.scores)
            opts = {k: v for k, v in dict(cache_dir=args.cache_dir, device=args.device).items() if v}
            only = args.tracks.split(",") if args.tracks else None
            n = len(testsongs.add_set(args.dataset, args.dir, args.annotations, runs, args.name, only, opts)["songs"])
            print(f"{n} test songs. Hear and rate them: bass2sheet-copilot serve --dir {args.dir}, then /tests")
        else:
            from . import reference, testsongs

            print(testsongs.report(args.dir) + report(args.dir) + reference.report(args.dir), end="")
    except Bass2SheetError as e:
        print(f"bass2sheet-copilot: {e}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
