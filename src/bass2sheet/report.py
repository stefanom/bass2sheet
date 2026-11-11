"""Run report: a self-contained HTML page and the same facts as JSON."""

import html
import json

from .common import atomic_write_text

STATUS_STYLE = {
    "pass": ("#1a7f37", "#dafbe1"),
    "warn": ("#9a6700", "#fff8c5"),
    "fail": ("#d1242f", "#ffebe9"),
    "skipped": ("#59636e", "#eaeef2"),
}
VERDICT_TEXT = {
    "PASS": "Every quality gate passed. The sheet should be close; still check the bars listed below by ear.",
    "WARN": "Usable, with warnings. Read the flagged gates and check the listed bars by ear before relying on it.",
    "FAIL": "Failed quality gates. Treat the sheet as a draft at best; the failing gates say what went wrong.",
}


def fmt(g):
    v = g["value"]
    if v is None:
        return "–"
    if g["name"] in ("bass_level", "grid_offset_ms"):
        return f"{v:.1f}"
    return f"{v:.0%}"


def threshold(g):
    op = "≥" if g["higher_is_better"] else "≤"
    f = (lambda x: f"{x:.0%}") if g["name"] not in ("bass_level", "grid_offset_ms") else (lambda x: f"{x:g}")
    return f"pass {op} {f(g['warn'])}; fail beyond {f(g['fail'])}"


def write(path_html, path_json, data):
    atomic_write_text(path_json, json.dumps(data, indent=1, default=float))
    e = html.escape
    v = data["verdict"]
    color, bg = STATUS_STYLE[{"PASS": "pass", "WARN": "warn", "FAIL": "fail"}[v]]
    rows = []
    for g in data["gates"]:
        c, b = STATUS_STYLE[g["status"]]
        rows.append(
            f"<tr><td>{e(g['label'])}<div class=m>{e(g['meaning'])}{(' — ' + e(g['note'])) if g['note'] else ''}</div></td>"
            f"<td class=n>{fmt(g)}</td><td class=t>{e(threshold(g))}</td>"
            f"<td><span class=chip style='color:{c};background:{b}'>{g['status']}</span></td></tr>"
        )
    th = data["check_severity"]
    sev = lambda c: (
        f"<span class=chip style='color:#d1242f;background:#ffebe9'>{c['severity']}</span>"
        if c["severity"] >= th
        else f"<span class=chip style='color:#59636e;background:#eaeef2'>{c['severity']}</span>"
    )
    checks = (
        "".join(
            f"<tr><td class=n>{c['bar']}</td><td class=n>{e(c['time'])}</td><td>{sev(c)}</td>"
            f"<td>{'<br>'.join(e(r) for r in c['reasons'])}</td></tr>"
            for c in data["ear_checks"]
        )
        or "<tr><td colspan=4>None.</td></tr>"
    )
    n_marked = sum(c["severity"] >= th for c in data["ear_checks"])
    attempts = "".join(f"<li><b>{e(a['method'])}</b>: {e(a['outcome'])}</li>" for a in data["attempts"])
    files = "".join(f"<li><code>{e(k)}</code>: {e(v)}</li>" for k, v in data["files"].items())
    warns = "".join(f"<li>{e(w)}</li>" for w in data["warnings"]) or "<li>None.</li>"
    timings = ", ".join(f"{k} {v:.0f} s" for k, v in data["timings"].items())
    s = data["summary"]
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>bass2sheet report: {e(data["title"])}</title><style>
body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#1f2328;margin:0}}
main{{max-width:960px;margin:0 auto;padding:32px 20px 64px}} h1{{margin:0 0 4px;font-size:1.7rem}} h2{{margin:36px 0 10px;font-size:1.2rem}}
.sub{{color:#59636e}} .verdict{{margin:20px 0;padding:14px 18px;border-radius:10px;background:{bg};color:{color};font-weight:600}}
.verdict b{{font-size:1.3rem;margin-right:10px}} table{{border-collapse:collapse;width:100%;font-size:.92rem}}
td,th{{padding:7px 9px;border-bottom:1px solid #d1d9e0;text-align:left;vertical-align:top}} th{{color:#59636e;font-size:.8rem;text-transform:uppercase}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} td.t{{color:#59636e;font-size:.85rem;white-space:nowrap}}
.m{{color:#59636e;font-size:.82rem}} .chip{{padding:2px 8px;border-radius:999px;font-size:.8rem;font-weight:700;text-transform:uppercase}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}} .kv div{{background:#f6f8fa;border:1px solid #d1d9e0;border-radius:8px;padding:10px 12px}}
.kv b{{display:block;font-size:1.2rem}} code{{background:#f6f8fa;padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>{e(data["title"])}</h1><div class=sub>{e(data["artist"])} · {e(data["input"])} · {s["duration"]}</div>
<div class=verdict><b>{v}</b>{e(VERDICT_TEXT[v])}</div>
<div class=kv><div><b>{s["notes"]}</b>notes</div><div><b>{s["bars"]}</b>bars</div><div><b>{s["tempo"]:.0f} BPM</b>tempo</div>
<div><b>{e(s["key"])}</b>key (estimate)</div><div><b>{e(s["separator"])}</b>separator used</div><div><b>{n_marked}</b>bars to check by ear first</div></div>
<h2>Quality gates</h2><table><tr><th>Gate</th><th>Value</th><th>Threshold</th><th>Status</th></tr>{"".join(rows)}</table>
<h2>Bars to check by ear</h2><p class=sub>Most severe first. Severity adds up independent signals (missing notes or a pitch
mismatch with the recording: 2; Basic Pitch disagreeing: 1–2; a doubtful octave: 2; an ambiguous octave or uncertain note: 1). Bars with
severity ≥ {th} carry a red "check" above the staff in the PDF. Listen with <code>audio/bass_ab_lr.wav</code> (real bass left, transcription
right) or <code>audio/song_resynth.wav</code> (the song with the transcription in place of the bass).</p>
<table><tr><th>Bar</th><th>Time</th><th>Severity</th><th>Why</th></tr>{checks}</table>
<h2>Separation attempts</h2><ul>{attempts}</ul>
<h2>Warnings</h2><ul>{warns}</ul>
<h2>Files</h2><ul>{files}</ul>
<p class=sub>Timings: {e(timings)}. bass2sheet {e(data["version"])}.</p>
</main></body></html>"""
    atomic_write_text(path_html, page)
