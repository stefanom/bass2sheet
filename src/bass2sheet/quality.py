"""Quality gates: every run measures its own result, and a verdict decides whether the sheet can be trusted.

There is no reference transcription, so each gate is a proxy; none is trusted alone. They cover the stem
(is there any bass), the transcription against the stem (coverage, notes cut short), the
transcription rendered back to audio against the stem (pitch), octave sanity, an independent model
(Basic Pitch) on the same stem, and the rhythm grid. Bars where the checks disagree go on an ear-check list.
"""

from dataclasses import asdict, dataclass

import librosa
import mir_eval
import numpy as np
import pretty_midi

from . import ownsound
from .common import HOP, SR, clock, note_name

CHECK_SEVERITY = 3  # bars at or above this severity get a red "check" in the PDF


@dataclass
class Gate:
    name: str
    label: str
    value: float | None
    warn: float
    fail: float | None  # None: the gate can warn but never fail a run
    higher_is_better: bool
    meaning: str
    status: str = "pass"  # pass | warn | fail | skipped
    note: str = ""

    def judge(self):
        if self.value is None:
            self.status = "skipped"
        else:
            bad = (lambda v, t: v < t) if self.higher_is_better else (lambda v, t: v > t)
            failed = self.fail is not None and bad(self.value, self.fail)
            self.status = "fail" if failed else "warn" if bad(self.value, self.warn) else "pass"
        return self


def _harmonic_pitch(C, lo=24, hi=64):
    """Per frame, the MIDI note whose harmonics 1-4 carry the most energy (robust to weak fundamentals)."""
    A = np.sqrt(C)
    return np.stack([A[q - 21] + A[q - 9] + 0.8 * A[q - 2] + 0.6 * A[q + 3] for q in range(lo, hi + 1)]).argmax(0) + lo


def _cqt(y):
    return np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("A0"), n_bins=72, bins_per_octave=12))


def stem_gates(stems, mix):
    """A cheap check on the separated bass before spending minutes on pitch tracking: is there any bass at all?

    There is deliberately no drum-bleed gate. The obvious proxy (how closely the stem's high-band onsets follow the
    drums) fires on perfectly clean bass stems, because bass and drums play on the same beats; on BabySlakh it
    scored clean stems as high as separated ones. A bad stem shows up in the transcription gates instead (coverage,
    the audio-level match, agreement with the second model)."""
    bass = stems["bass"].mean(0)
    level = 20 * np.log10((np.sqrt(np.mean(bass**2)) + 1e-12) / (np.sqrt(np.mean(mix**2)) + 1e-12))
    return [
        Gate(
            "bass_level",
            "Bass stem level vs the mix",
            float(level),
            -25.0,
            -35.0,
            True,
            "dB; very low means the separator found little or no bass",
        ).judge(),
    ]


def octave_doubts(C, notes):
    """Notes whose written octave the spectrum contradicts: energy an octave below or at a fifth above (the 3rd
    harmonic of the octave below) means too high; a fundamental 25 dB under its octave means too low."""
    db = lambda x, y: 20 * np.log10(x / y)

    def energy(q, a, b):  # mean CQT magnitude at MIDI pitch q over frames a..b
        return C[q - 21, a:b].mean() + 1e-9 if 0 <= q - 21 < C.shape[0] else 1e-9

    out = []
    for nt in notes:
        a, b = int(nt["start"] * 100) + 2, int(min(nt["end"], nt["start"] + 0.3) * 100)
        if b - a < 6:
            continue
        p = nt["pitch"]
        strongest = max(energy(p, a, b), energy(p + 12, a, b))
        if p - 12 >= 28 and (db(energy(p - 12, a, b), strongest) > -10 or db(energy(p + 7, a, b), strongest) > -6):
            out.append((nt, "may be an octave too high"))
        elif p + 12 < 21 + C.shape[0] and db(energy(p, a, b), energy(p + 12, a, b)) < -25:
            out.append((nt, "may be an octave too low"))
    return out


def transcription_gates(
    stems, feat, notes, grid, origin, synth_bass, second, second_reason, meter_fit=None, beat_tracker=None
):
    """Gates on the transcription plus the per-bar ear-check list {bar: [reasons]}.

    Thresholds were set from evaluation runs with ground truth (BabySlakh, MDB-bass-synth, RWC Popular): a gate
    warns below what 90% of the accurate runs reach, and fails where only inaccurate runs were seen. Gates about
    the meter and the share of notes on eighth-note positions track the bar layout, not the notes; they warn but
    never fail a run."""
    rms_db = feat["rms_db"]
    n = len(rms_db)
    playing = rms_db > np.percentile(rms_db, 99) - 20
    cover = np.zeros(n, bool)
    for nt in notes:  # widened by the RMS window's half-length (23 ms): a note's energy reaches that far in rms_db
        cover[max(0, int(nt["start"] * 100) - 2) : int(nt["end"] * 100) + 3] = True
    coverage = float((cover & playing).sum() / max(1, playing.sum()))
    cut = []
    for a, b in zip(notes, notes[1:]):
        ioi = b["start"] - a["start"]
        if 0.1 < ioi < 1.5:
            mid = slice(int(a["end"] * 100), int(b["start"] * 100))
            still = playing[mid].mean() if mid.stop > mid.start else 0
            cut.append((a["end"] - a["start"]) / ioi < 0.8 and still > 0.5)
    cut_short = float(np.mean(cut)) if cut else 0.0

    # the transcription rendered back to audio, against the stem: where a note is written and the stem sounds, do
    # the two show the same note name? (There is no "synth silent" gate: the pluck synth decays, so it measured how
    # long the notes were held, not whether they were transcribed. Coverage already says what is missing.)
    C_real = feat["cqt"]
    r_semi = _harmonic_pitch(C_real)[:n]
    s_semi = _harmonic_pitch(_cqt(synth_bass))[:n]
    m = min(len(r_semi), len(s_semi), n)
    both = (playing & cover)[:m]
    same_name = both & ((r_semi[:m] - s_semi[:m]) % 12 == 0)
    pitch_match = float(same_name.sum() / max(1, both.sum()))
    doubts = octave_doubts(C_real, notes)

    # second model on the same stem
    sec_pitch = sec_onset = None
    fb = None
    if second:
        fa, fb = np.full(n, -1), np.full(n, -1)
        for arr, src in (
            (fa, [(x["start"], x["end"], x["pitch"]) for x in notes]),
            (fb, [(s, e, p) for s, e, p, _ in second]),
        ):
            for s, e, p in src:
                arr[int(s * 100) : int(e * 100)] = p
        bb = (fa >= 0) & (fb >= 0)
        sec_pitch = float(np.mean(fa[bb] == fb[bb])) if bb.any() else 0.0
        ia = np.array([[x["start"], x["end"]] for x in notes])
        pa = np.array([pretty_midi.note_number_to_hz(x["pitch"] % 12 + 48) for x in notes])
        ib = np.array([[s, e] for s, e, _, _ in second])
        pb = np.array([pretty_midi.note_number_to_hz(p % 12 + 48) for _, _, p, _ in second])
        _, _, sec_onset, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ia, pa, ib, pb, onset_tolerance=0.05, offset_ratio=None
        )
        sec_onset = float(sec_onset)

    # each note against the song's other notes of its pitch (see ownsound.py): `own_sound` is how well they
    # explain it, `sounds_like` the pitch whose notes explain it clearly better
    heard = ownsound.check(stems["bass"].mean(0), notes)
    for nt, h in zip(notes, heard):
        nt.pop("own_sound", None), nt.pop("sounds_like", None)
        if h:
            nt["own_sound"] = round(h["explained"], 2)
            if h["better"] is not None:
                nt["sounds_like"] = int(h["better"])
    judged = [h for h in heard if h]
    unlike = float(np.mean([ownsound.doubtful(h) for h in judged])) if judged else None

    offsets = [abs(nt["start"] - grid[nt["grid_start"] + origin]) for nt in notes]
    eighths = float(np.mean([nt["sixteenth"] in (1, 3) for nt in notes]))
    uncertain = float(np.mean([bool(nt["flag"]) for nt in notes]))
    gates = [
        Gate(
            "coverage",
            "Sounding bass covered by notes",
            coverage,
            0.97,
            0.85,
            True,
            "share of the time the stem clearly sounds that has a transcribed note",
        ).judge(),
        Gate(
            "cut_short",
            "Notes cut short while the bass rings",
            cut_short,
            0.04,
            0.12,
            False,
            "share of notes followed by a MIDI gap while the stem still sounds (heard as choppy)",
        ).judge(),
        Gate(
            "audio_pitch",
            "Same note name as the recording",
            pitch_match,
            0.90,
            0.78,
            True,
            "where both sound; pitch from harmonic summation, so the synth's timbre doesn't matter",
        ).judge(),
        Gate(
            "own_sound",
            "Notes unlike the song's other notes of their pitch",
            unlike,
            0.15,
            None,
            False,
            "each note is compared with the sound of the other notes written with its pitch; where those do not "
            "explain it, or another pitch's notes explain it better, the note is probably wrong",
            note="" if judged else "too few notes to compare",
        ).judge(),
        Gate(
            "octave_doubts",
            "Notes with a doubtful octave",
            len(doubts) / max(1, len(notes)),
            0.01,
            0.05,
            False,
            "share of notes whose octave the spectrum contradicts",
        ).judge(),
        Gate(
            "second_model_pitch",
            "Same pitch as Basic Pitch",
            sec_pitch,
            0.90,
            0.70,
            True,
            "independent model on the same stem, where both sound",
        ).judge(),
        Gate(
            "second_model_onsets",
            "Note starts shared with Basic Pitch",
            sec_onset,
            0.75,
            0.50,
            True,
            "onset F-measure within 50 ms, same note name",
        ).judge(),
        Gate(
            "grid_offset_ms",
            "Median distance of note starts from the 16th grid",
            float(np.median(offsets) * 1000),
            32.0,
            50.0,
            False,
            "ms; large means the beat grid or tempo is off",
        ).judge(),
        Gate(
            "meter_fit",
            "Downbeats that fit 4/4",
            meter_fit,
            0.80,
            None,
            True,
            "share of the beat tracker's downbeats that fall on one beat in four; low means 3/4, 6/8 or a changing "
            "meter, which the 4/4 score cannot show",
            note="" if meter_fit is not None else f"not measured by the {beat_tracker or 'built-in'} beat tracker",
        ).judge(),
        Gate(
            "grid_eighths",
            "Notes on 8th-note positions",
            eighths,
            0.60,
            None,
            True,
            "rock bass sits mostly on beats and 'ands'; low suggests a wrong pulse level, a shifted grid or swing",
        ).judge(),
        Gate(
            "uncertain_notes",
            "Notes flagged uncertain",
            uncertain,
            0.06,
            0.20,
            False,
            "pitch trackers disagreed, or the octave was corrected",
        ).judge(),
    ]
    if second is None:
        for g in gates:
            if g.name.startswith("second_model"):
                g.note = second_reason or "not run"

    # ear-check list: (category, points, reason) per bar; a bar's severity sums the worst points per category
    checks = {}
    add = lambda bar, cat, pts, why: checks.setdefault(bar, []).append((cat, pts, why))
    bars = sorted({nt["bar"] for nt in notes})
    for bar in range(min(bars + [1]), max(bars + [1]) + 1):
        i0 = origin + 16 * (bar - 1)
        if i0 < 0 or i0 + 16 >= len(grid):
            continue
        a, b = int(grid[i0] * 100), min(n, int(grid[i0 + 16] * 100))
        if b - a < 20 or playing[a:b].mean() < 0.25:
            continue
        cov = (cover[a:b] & playing[a:b]).sum() / max(1, playing[a:b].sum())
        if cov < 0.8:
            add(bar, "missing", 2, f"notes missing: {cov:.0%} of the sounding bass is covered")
        bm = both[a : min(b, m)]
        if bm.sum() >= 20:
            pm = same_name[a : min(b, m)].sum() / bm.sum()
            if pm < 0.7:
                add(bar, "audio_pitch", 2, f"rendered notes match the recording's pitch only {pm:.0%} of the time")
        if fb is not None:
            sel = (fa[a:b] >= 0) & (fb[a:b] >= 0)
            if sel.sum() >= 20:
                ag = np.mean(fa[a:b][sel] == fb[a:b][sel])
                if ag < 0.7:
                    add(
                        bar,
                        "second_model",
                        2 if ag < 0.5 else 1,
                        f"Basic Pitch hears different notes ({ag:.0%} agreement)",
                    )
    for nt in notes:
        pos = f"{nt['beat']}.{nt['sixteenth']}"
        if nt["flag"] == "octave":
            add(nt["bar"], "octave_ambiguous", 1, f"octave was ambiguous at {pos} (decided from the spectrum)")
        elif nt["flag"] == "alone":  # muscriptor.ALONE
            add(
                nt["bar"],
                "uncertain",
                1,
                f"note at {pos} is MuScriptor's alone (bass2sheet's own finder does not hear it)",
            )
        elif nt["flag"]:
            add(nt["bar"], "uncertain", 1, f"uncertain pitch at {pos} (pitch trackers disagree)")
    for nt, why in doubts:
        add(nt["bar"], "octave_doubt", 2, f"note at {nt['beat']}.{nt['sixteenth']} {why}")
    for nt in notes:
        pos, name = f"{nt['beat']}.{nt['sixteenth']}", note_name(nt["pitch"])
        if "sounds_like" in nt:
            like = note_name(nt["sounds_like"])
            add(nt["bar"], "own_sound", 2, f"the {name} at {pos} sounds more like the song's {like}s than its {name}s")
        elif nt.get("own_sound", 1.0) < ownsound.UNLIKE:
            add(nt["bar"], "own_sound", 1, f"the {name} at {pos} does not sound like the song's other {name}s")
    severity = {
        bar: sum(max(p for c, p, _ in items if c == cat) for cat in {c for c, _, _ in items})
        for bar, items in checks.items()
    }
    n_bars = max(1, len({nt["bar"] for nt in notes}))
    gates.append(
        Gate(
            "check_bars",
            "Bars that need an ear check",
            sum(s >= CHECK_SEVERITY for s in severity.values()) / n_bars,
            0.06,
            0.25,
            False,
            f"share of bars with notes where independent checks disagree (severity ≥ {CHECK_SEVERITY})",
        ).judge()
    )
    checks = {bar: dict(severity=severity[bar], reasons=[w for _, _, w in items]) for bar, items in checks.items()}
    return gates, checks


LAYOUT_GATES = {"meter_fit", "grid_eighths"}  # about how the bars are laid out, not whether the notes are right
NOT_IN_VERDICT = {"own_sound"}  # reported, but the verdict's thresholds were set before this gate existed
MANY_WARNINGS = 5


def verdict(gates):
    """FAIL when a gate fails or when MANY_WARNINGS of the accuracy gates warn at once: in the evaluation runs,
    results with five or more warnings were as inaccurate as those with a failed gate. WARN for any warning."""
    gates = [g for g in gates if g.name not in NOT_IN_VERDICT]
    statuses = [g.status for g in gates]
    accuracy_warnings = sum(g.status == "warn" for g in gates if g.name not in LAYOUT_GATES)
    if "fail" in statuses or accuracy_warnings >= MANY_WARNINGS:
        return "FAIL"
    return "WARN" if "warn" in statuses else "PASS"


def gates_json(gates):
    return [asdict(g) for g in gates]


def check_list(checks, grid, origin):
    """Ear-check bars, most severe first."""
    rows = [
        dict(
            bar=b,
            time=clock(grid[min(origin + 16 * (b - 1), len(grid) - 1)]),
            severity=c["severity"],
            reasons=list(dict.fromkeys(c["reasons"])),
        )
        for b, c in checks.items()
    ]
    return sorted(rows, key=lambda r: (-r["severity"], r["bar"]))
