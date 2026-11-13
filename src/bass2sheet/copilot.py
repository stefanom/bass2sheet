"""The copilot: the person is in charge of the score, the tool does the legwork and keeps a record of its mistakes.

A song is prepared once: the tool hears it as it would for a sheet, and what it heard is kept as it is
(`machine.json`). The person then goes through the notes with the record playing, and fixes what is wrong:
another pitch, a note that is not there, one that is missing, two that are one, one that is two, a start or an
end. Their version lives beside the tool's (`notes.json`), every fix is logged (`log.jsonl`), and what they
played past without touching counts as checked. The sheet is written from their version.

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

from . import judge, ownsound, together
from .common import DEFAULTS, LOWEST_NOTE, SR, Bass2SheetError, atomic_write_text, clock, note_name

LOW, HIGH = 26, 72  # the pitches the picture shows: D1 to C5
PER_SEMI = 3  # picture rows per semitone
PICTURE_SR, PICTURE_HOP = 11025, 128  # 11.6 ms a column
SEMI_LO, SEMI_HI = 23, 76  # the pitches the pitch tracker's bins are summed into
SHORTEST = 0.03  # no note is shorter than this
UNDO_DEPTH = 200
SPEEDS = (90, 80, 70, 60, 50)  # the slower speeds a song can be played at, in percent
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
    """What sounds in the bass track, by pitch and time, as bytes: rows from LOW up, PER_SEMI to a semitone,
    one column every PICTURE_HOP / PICTURE_SR seconds; 0 is silence and 255 the loudest (42 dB above)."""
    import librosa

    y = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=SR, target_sr=PICTURE_SR)
    c = np.abs(
        librosa.cqt(
            y,
            sr=PICTURE_SR,
            hop_length=PICTURE_HOP,
            fmin=librosa.midi_to_hz(LOW - 1 / 3),
            n_bins=PER_SEMI * (HIGH - LOW + 1),
            bins_per_octave=12 * PER_SEMI,
        )
    )
    c = np.pad(c, ((1, 1), (0, 0)), mode="edge")  # a light blur across pitch hides the filter bank's ripple
    c = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
    ref = np.percentile(c, 99.7) or 1.0
    v = np.clip(1 + 20 * np.log10(np.maximum(c, 1e-12) / ref) / 42, 0, 1)
    return (v * 255).astype(np.uint8)


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
    settings = dict(DEFAULTS, **(opts.get("defaults") or {}), **(song.get("settings") or {}))
    kw = dict(
        cache_dir=opts.get("cache_dir"),
        device=opts.get("device"),
        stems_dir=song.get("stems_dir"),
        crepe_model=opts.get("crepe_model", "full"),
    )
    say(f"{song['title']}: hearing it")
    heard = judge.hear(song["path"], settings, keep=os.path.join(folder, "heard"), **kw)
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
    for n, h in zip(notes, ownsound.check(mono, notes)):
        if h and ownsound.doubtful(h):
            n["doubt"] = dict(explained=round(h["explained"], 2), better=h["better"])
    feat = features(audio.bass_mono, audio.cache_dir, audio.device, kw["crepe_model"])
    suffix = "" if kw["crepe_model"] == "full" else f"-{kw['crepe_model']}"
    probs = np.load(os.path.join(audio.cache_dir, f"crepe-{sha1_array(audio.bass_mono)[:16]}{suffix}.npz"))["probs"]
    f0, per = feat["f0_c"], feat["per_c"]
    line = np.where((per >= 0.3) & (f0 > 0), 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440), np.nan).astype(np.float32)
    pyin = feat["f0_p"]
    pyin = np.where(
        np.isfinite(pyin) & (pyin > 0), 69 + 12 * np.log2(np.maximum(np.nan_to_num(pyin), 1e-6) / 440), np.nan
    )
    pic = picture(mono)
    pic.tofile(os.path.join(folder, "picture.bin"))
    line.tofile(os.path.join(folder, "pitch.bin"))
    np.savez_compressed(os.path.join(folder, "desk.npz"), semis=semitone_mass(probs), pyin=pyin.astype(np.float32))
    peak = max(float(np.abs(audio.mix).max()), float(np.abs(audio.bass).max()), 1e-9)
    sf.write(os.path.join(folder, "audio", "record.flac"), (0.97 * audio.mix / peak).T, SR, subtype="PCM_16")
    sf.write(os.path.join(folder, "audio", "bass.flac"), 0.97 * mono / peak, SR, subtype="PCM_16")
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
        length=round(audio.n / SR, 3),
        picture=dict(
            rows=int(pic.shape[0]),
            columns=int(pic.shape[1]),
            low=LOW,
            per_semitone=PER_SEMI,
            step=PICTURE_HOP / PICTURE_SR,
        ),
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
    if name not in ("record", "bass") or percent not in SPEEDS:
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
        self._mono = self._extra = self._bank = None
        self._spectra = {}

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
        label, bank, checked = fixed["was"]["pitch"], self.bank(), self.state["checked"]
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

    def _check_pitch(self, pitch):
        if not isinstance(pitch, int) or pitch not in PITCHES:
            raise Bass2SheetError(f"{pitch} is not a pitch a bass plays")
        return pitch

    def act(self, action):
        """Apply one fix (see the module's docstring for the kinds), log it and save. Returns what the page needs
        next: for a change of pitch, the other notes that sound the same (`offer`)."""
        kind = action.get("kind")
        if kind == "undo":
            return self.undo()
        before, made, result = [], [], {}
        keep = lambda n: before.append(json.loads(json.dumps(n)))
        notes = self.notes
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
            return {}
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
        self.state["undo"].append(dict(kind=kind, before=before, made=made))
        self.save()
        self.log(dict(action, song=self.song_id, before=before, made=made))
        if kind == "pitch" and action.get("offer", True):
            label, ids = self.alike(action["id"])
            if ids:
                result["offer"] = dict(seed=action["id"], label=label, pitch=self.note(action["id"])["pitch"], ids=ids)
        return result

    def undo(self):
        if not self.state["undo"]:
            return dict(undone=None)
        last = self.state["undo"].pop()
        ids = {b["id"] for b in last["before"]} | set(last["made"])
        self.state["notes"] = [n for n in self.notes if n["id"] not in ids] + last["before"]
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
            at=self.state.get("at"),
            undo=len(self.state["undo"]),
            fixes=sum(n["state"] != "machine" for n in self.notes) + self.deleted(),
            names={str(p): note_name(p) for p in PITCHES},
        )

    def deleted(self):
        have = {n["id"] for n in self.notes} | {i for n in self.notes for i in n.get("merged", [])}
        return sum(n["id"] not in have for n in self.machine["notes"])


# --------------------------------------------------------------------------------- what the tool got wrong


def mistakes(workdir, song_id):
    """The tool's version of a song against the person's: one row per difference, and what was found right.

    Kinds: `wrong octave`, `wrong pitch`, `extra note` (the person deleted it), `missed note` (they added it),
    `two notes written as one` (they split it), `one note written as two` (they joined them), `start` and `end`
    (moved by more than 50 ms), `second note` (a note played together added or removed). `right` counts the
    notes left as the tool had them in what the person played through; `unchecked` the rest."""
    desk = Desk(workdir, song_id)
    mine = {n["id"]: n for n in desk.notes}
    absorbed = {i for n in desk.notes for i in n.get("merged", [])}
    checked, rows, right, unchecked = desk.state["checked"], [], 0, 0
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
        share = m["checked_seconds"] / max(m["length"], 1e-9)
        lines += [f"## {song.get('title', song_id)}", ""]
        lines += [
            f"Checked {clock(m['checked_seconds'])} of {clock(m['length'])} ({share:.0%}). Of the tool's "
            f"{m['notes']} notes, {m['right']} were left as they were there; {m['unchecked']} are not checked yet.",
            "",
        ]
        if m["kinds"]:
            lines += ["| Mistake | Times |", "|---|---|", *(f"| {k} | {v} |" for k, v in m["kinds"].items()), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


# ------------------------------------------------------------------------------------------------- the sheet


def sheet(workdir, song_id, mscore=None, tab=None, rhythm=None):
    """Write the score from the person's notes: MusicXML, MIDI and, with MuseScore, the PDF. The line under the
    title says how much of it they have played through. Returns {file name: path}."""
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
    checked = desk.state["checked"]
    share = sum(inside(n["start"], checked) for n in notes) / max(1, len(notes))
    done = share >= 0.995
    subtitle = (
        "Bass: checked by ear against the record"
        if done
        else f"Bass: {share:.0%} checked by ear against the record; the rest is the machine's transcription"
    )
    xml = score.musicxml(
        notes,
        m["chords"],
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
    ms = score.find_mscore(mscore)
    if ms:
        err = score.engrave(files["musicxml"], os.path.join(out, "bass.pdf"), ms)
        if err:
            raise Bass2SheetError(f"the PDF was not written: {err}")
        files["pdf"] = os.path.join(out, "bass.pdf")
    desk.log(dict(kind="sheet", song=song_id, notes=len(notes), checked=round(share, 3)))
    return files


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
    a.add_argument("--cache-dir", help="where models, stems and pitch tracks are cached")
    a.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    s = sub.add_parser("serve", help="open the page on this machine")
    s.add_argument("--port", type=int, default=8770)
    h = sub.add_parser("sheet", help="write the score of one song from your notes")
    h.add_argument("song", help="the song's id, as `report` lists it")
    h.add_argument("--mscore", help="path to the MuseScore 4 binary")
    r = sub.add_parser("report", help="print what the tool got wrong so far")
    for p in (a, s, h, r):
        p.add_argument("--dir", default="scores", help="working directory (default ./scores)")
    args = ap.parse_args(argv)
    try:
        if args.command == "add":
            missing = [p for p in args.songs if not os.path.isfile(p)]
            if missing:
                raise Bass2SheetError(f"no such file: {missing[0]}")
            opts = {k: v for k, v in dict(cache_dir=args.cache_dir, device=args.device).items() if v}
            add(args.songs, args.dir, opts, judging=args.judging)
            print(f"Ready. Open the page with: bass2sheet-copilot serve --dir {args.dir}")
        elif args.command == "serve":
            from .copilot_server import serve

            serve(args.dir, args.port)
        elif args.command == "sheet":
            for name, path in sheet(args.dir, args.song, args.mscore).items():
                print(f"{name}: {path}")
        else:
            print(report(args.dir), end="")
    except Bass2SheetError as e:
        print(f"bass2sheet-copilot: {e}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
