"""A portable HTML report: inline styles, SVG plots and PNG camera previews."""

import base64
from html import escape
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np

from .trace import Trace


def _png(array):
    height, width, _ = array.shape

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = b"".join(b"\0" + row.tobytes() for row in array)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(image).decode()


def _cameras(ref, cand, result):
    first = result["first_recorded_difference"]
    pairs = []
    for key, rr in ref.records.items():
        if first and key[:3] != tuple(first[k] for k in ("session", "episode", "step")):
            continue
        cr = cand.records.get(key)
        if cr is None:
            continue
        for name in rr["fields"]:
            spec = ref.manifest["contracts"].get(key[3], {}).get(name, {})
            if spec.get("kind") != "image" or spec.get("layout") != "HWC":
                continue
            row = next(
                (
                    r
                    for r in result["rows"]
                    if all(r.get(k) == rr[k] for k in ("session", "episode", "step", "stage"))
                    and r["field"] == name
                ),
                {},
            )
            target = row.get("candidate_field", name)
            if target not in cr["fields"]:
                continue
            a, b = ref.array(rr, name), cand.array(cr, target)
            if any(x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3 for x in (a, b)):
                continue
            # Bound preview size; trace retains the full image.
            imgs = [
                _png(
                    x[
                        :: max(1, (x.shape[0] + 119) // 120), :: max(1, (x.shape[1] + 191) // 192)
                    ].copy()
                )
                for x in (a, b)
            ]
            caption = escape(f"{key[3]} / {name} · {key[1]}:{key[2]}")
            pairs.append(
                f'<div class="camera"><p>{caption}</p><div class="pair">'
                f'<figure><img alt="Reference {caption}" src="{imgs[0]}"><figcaption>Reference</figcaption></figure>'
                f'<figure><img alt="Candidate {caption}" src="{imgs[1]}"><figcaption>Candidate</figcaption></figure>'
                "</div></div>"
            )
        if len(pairs) >= 4:
            break
    return (
        "".join(pairs)
        or '<p class="muted">No declared HWC uint8 RGB camera preview at this coordinate.</p>'
    )


def _curves(ref, cand, result):
    plots = []
    for stage, fields in ref.manifest["contracts"].items():
        for field, spec in fields.items():
            if spec.get("kind") != "action":
                continue
            groups = {}
            for key, rr in ref.records.items():
                if key[3] != stage or field not in rr["fields"]:
                    continue
                cr = cand.records.get(key)
                row = next(
                    (
                        r
                        for r in result["rows"]
                        if r.get("session") == key[0]
                        and r.get("episode") == key[1]
                        and r.get("step") == key[2]
                        and r.get("stage") == stage
                        and r["field"] == field
                    ),
                    {},
                )
                target = row.get("candidate_field", field)
                if cr is None or target not in cr["fields"]:
                    continue
                a, b = ref.array(rr, field), cand.array(cr, target)
                if not a.size or not b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
                    continue
                groups.setdefault(key[:2], []).append((key[2], a.flat[0], b.flat[0]))
            for group, data in groups.items():
                data.sort()
                numbers = [float(x) for _, a, b in data for x in (a, b)]
                low, high = min(numbers), max(numbers)
                span = high - low or 1
                if not np.isfinite(span):
                    continue  # Full numeric evidence remains in JSON when plot scaling overflows.
                steps = [x[0] for x in data]
                width = max(steps) - min(steps) or 1

                def points(column):
                    return " ".join(
                        f"{40 + (x[0] - min(steps)) / width * 500:.1f},"
                        f"{130 - (x[column] - low) / span * 100:.1f}"
                        for x in data
                    )

                title = escape(f"{stage} / {field} [flat channel 0] · {group[0]} / {group[1]}")
                plots.append(
                    f'<div class="plot"><p>{title}</p><svg viewBox="0 0 580 163" role="img" aria-label="{title}">'
                    '<path d="M40 20V135H550" fill="none" stroke="#ccd6dd"/>'
                    f'<polyline points="{points(1)}" fill="none" stroke="#087e8b" stroke-width="3"/>'
                    f'<polyline points="{points(2)}" fill="none" stroke="#e47541" stroke-width="2.5" stroke-dasharray="6 4"/>'
                    f'<text x="4" y="25">{high:.3g}</text><text x="4" y="133">{low:.3g}</text>'
                    f'<text x="40" y="155">step {min(steps)}</text><text x="490" y="155">{max(steps)}</text></svg>'
                    f'<span class="muted">Declared units: {escape(str(spec.get("units", "unknown")))}</span></div>'
                )
    return "".join(plots) or '<p class="muted">No paired finite action samples to plot.</p>'


def write_report(result, reference, candidate, directory):
    ref = reference if isinstance(reference, Trace) else Trace(reference)
    cand = candidate if isinstance(candidate, Trace) else Trace(candidate)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    first = result["first_recorded_difference"]
    if first:
        headline = escape(f"{first['stage']} / {first['field']}")
        evidence = escape(
            f"{first['session']} · {first['episode']} · step {first['step']} — {first['reason']}"
        )
        values = escape(
            f"Reference {first.get('reference_value', '—')} → candidate {first.get('candidate_value', '—')} · "
            f"max |Δ| {first.get('max_abs_error', '—')}"
        )
    else:
        headline = "No recorded divergence" if result["status"] == "PASS" else "Evidence incomplete"
        evidence = "Review the stage timeline and evidence coverage below."
        values = "A PASS covers the declared recorded boundaries only."
    rows = []
    for row in result["rows"]:
        location = escape(
            f"{row.get('session', '—')} / {row.get('episode', '—')} / {row.get('step', '—')}"
        )
        detail = escape(json.dumps(row, indent=2))
        status = row["status"]
        rows.append(
            f'<tr data-status="{status}"><td>{location}</td><td>{escape(row.get("stage", "—"))}</td>'
            f'<td>{escape(row["field"])}</td><td><span class="badge {status}">{status}</span></td>'
            f"<td><details><summary>{escape(row['reason'])}</summary><pre>{detail}</pre></details></td></tr>"
        )
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'">
<title>VLA Preflight · Evidence report</title><style>
:root{color-scheme:light;--ink:#17353f;--muted:#5b7078;--line:#d9e3e7}*{box-sizing:border-box}body{margin:0;background:#f2f6f7;color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 34px 70px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:18px}.brand{font-weight:800;letter-spacing:-.7px;font-size:23px}.brand i{font-style:normal;color:#087e8b}.small,.muted{color:var(--muted);font-size:13px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#087e8b;font-weight:750}h1{font-size:40px;letter-spacing:-1.5px;line-height:1.12;margin:12px 0}h2{font-size:21px;letter-spacing:-.4px;margin:0 0 12px}h3{font-size:15px}p{margin:8px 0 14px}.hero{padding:29px 0 23px;display:grid;grid-template-columns:1fr auto;gap:24px}.badge{display:inline-block;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:800;letter-spacing:.8px}.PASS{background:#d7f1e8;color:#11634d}.FAIL{background:#fce0d5;color:#9f3c20}.UNKNOWN{background:#fff0cb;color:#805c08}.stats{display:flex;gap:10px;align-items:start}.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:14px 22px;min-width:92px}.stat strong{font-size:25px;display:block}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:20px}.focus{border-left:5px solid #e47541;background:#fffcfa}.focus h2{font-size:27px}.mono,pre{font-family:ui-monospace,Consolas,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:14px;border-radius:7px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.camera p,.plot p{font-size:12px;color:var(--muted);font-weight:600}.pair{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}figure img{width:100%;max-height:145px;object-fit:contain;border-radius:8px;background:#edf3f5;image-rendering:pixelated}figcaption{font-size:11px;color:var(--muted);margin-top:5px}.plot svg{width:100%;display:block}.plot text{font:10px system-ui;fill:#5b7078}.legend{font-size:12px;color:#087e8b}.legend span{color:#c55f31;margin-left:16px}.toolbar{display:flex;gap:12px;margin:15px 0}input,select{border:1px solid var(--line);border-radius:7px;padding:9px 12px;background:white;font:inherit;color:var(--ink)}input{flex:1;min-width:0}.tablewrap{overflow:auto;max-height:510px}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}th{background:#edf4f5;position:sticky;top:0}td,th{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}td:nth-child(1){white-space:nowrap}summary{cursor:pointer}footer{font-size:12px;color:var(--muted)}a{color:#087e8b}details.provenance{margin-top:18px}@media(max-width:760px){main{padding:18px}.hero,.grid{grid-template-columns:1fr}h1{font-size:32px}.stats{flex-wrap:wrap}.card{padding:16px}}
</style><main><header><div class="brand"><i>◈</i> VLA Preflight</div><div class="small">LOCAL EVIDENCE REPORT · v0.1.0rc1</div></header>
<section class="hero"><div><div class="eyebrow">Policy integration / recorded parity</div><h1>Find the first boundary<br>that changed.</h1><p class="muted">{{LABELS}}</p><span class="badge {{STATUS}}">{{STATUS}}</span></div><div class="stats">{{STATS}}</div></section>
<section class="card focus"><div class="eyebrow">First recorded difference</div><h2>{{HEADLINE}}</h2><p>{{EVIDENCE}}</p><p class="mono">{{VALUES}}</p><p class="small">This is the earliest failing recorded boundary. Uninstrumented code between boundaries remains unobserved.</p></section>
<div class="grid"><section class="card"><h2>Camera pairs</h2><p class="small">Preview at the first divergence coordinate; otherwise the first recorded frame. Up to four pairs.</p>{{CAMERAS}}</section><section class="card"><h2>Action traces</h2><p class="legend">━━ Reference <span>┄┄ Candidate</span></p><p class="small">Flat channel 0 only. Full arrays and all-channel comparison remain in the trace and JSON.</p>{{CURVES}}</section></div>
<section class="card"><h2>Episode / stage timeline</h2><p class="small">Ordered by session, episode, step, declared stage. Expand a reason for numeric evidence.</p><div class="toolbar"><input id="search" aria-label="Filter timeline" placeholder="Filter episode, stage, field…"><select id="status" aria-label="Filter status"><option value="">All evidence</option>FAIL</option>UNKNOWN</option>PASS</select></div><div class="tablewrap"><table><thead><tr><th>Session / episode / step</th><th>Stage</th><th>Field</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
<details class="provenance"><summary>Versions and comparison scope</summary><pre>{{PROVENANCE}}</pre></details></section>
<footer>VLA Preflight · {{SCOPE}}. Images and graphs are embedded; this report makes no network requests. <a href="result.json">Companion JSON</a> · NPY arrays remain in the source traces. Treat traces and reports as sensitive deployment data.</footer></main>
<script>const search=document.querySelector('#search'),status=document.querySelector('#status');function filter(){for(const row of document.querySelectorAll('tbody tr'))row.hidden=!(row.textContent.toLowerCase().includes(search.value.toLowerCase())&&(!status.value||row.dataset.status===status.value));}search.addEventListener('input',filter);status.addEventListener('change',filter);</script></html>"""
    replacements = {
        "LABELS": escape(f"{result['reference']} → {result['candidate']}"),
        "STATUS": result["status"],
        "HEADLINE": headline,
        "EVIDENCE": evidence,
        "VALUES": values,
        "STATS": "".join(
            f'<div class="stat"><strong>{result["counts"][s]}</strong><span class="badge {s}">{s}</span></div>'
            for s in ("PASS", "FAIL", "UNKNOWN")
        ),
        "CAMERAS": _cameras(ref, cand, result),
        "CURVES": _curves(ref, cand, result),
        "ROWS": "".join(rows),
        "PROVENANCE": escape(json.dumps(result["versions"], indent=2)),
        "SCOPE": escape(result["scope"]),
    }
    template = re.sub(r"\{\{([A-Z]+)\}\}", lambda match: replacements[match[1]], template)
    (path / "index.html").write_text(template, encoding="utf-8")
    return path / "index.html"
