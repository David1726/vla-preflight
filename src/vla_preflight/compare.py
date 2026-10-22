"""Pairwise evidence comparison. Passing is bounded by recorded contracts."""

import math

import numpy as np

from .trace import Trace

EXIT_CODES = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}


def _declared(spec):
    return all(spec.get(k) not in (None, "", "unknown") for k in ("semantic", "units", "layout"))


def compare(reference, candidate, *, atol=1e-6, rtol=1e-5, field_map=None, tolerances=None):
    """Compare same-coordinate arrays; field_map is stage -> reference -> candidate.

    No implicit transpose, unit conversion, camera identification, or normalization.
    tolerances is stage -> reference field -> {atol, rtol}.
    """
    ref = reference if isinstance(reference, Trace) else Trace(reference)
    cand = candidate if isinstance(candidate, Trace) else Trace(candidate)
    for value in (atol, rtol):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Tolerances must be finite and nonnegative")
    field_map, tolerances = field_map or {}, tolerances or {}
    rows = []
    stages = list(dict.fromkeys(ref.manifest["stages"] + cand.manifest["stages"]))
    stage_rank = {stage: i for i, stage in enumerate(stages)}
    # Preserve episode/session encounter order, then step and declared stage order.
    coordinates = list(dict.fromkeys(k[:3] for k in list(ref.records) + list(cand.records)))
    episodes = list(dict.fromkeys(k[:2] for k in coordinates))
    coordinates.sort(key=lambda k: (episodes.index(k[:2]), k[2]))
    keys = [(s, e, step, stage) for s, e, step in coordinates for stage in stages]
    deterministic = ref.manifest["deterministic"] and cand.manifest["deterministic"]
    for key in keys:
        rr, cr = ref.records.get(key), cand.records.get(key)
        base = dict(zip(("session", "episode", "step", "stage"), key))
        if rr is None or cr is None:
            rows.append(
                {
                    **base,
                    "field": "*",
                    "status": "UNKNOWN",
                    "reason": "Stage evidence missing in reference or candidate",
                }
            )
            continue
        mapping = field_map.get(key[3], {})
        if len(set(mapping.values())) != len(mapping.values()):
            raise ValueError("Field mappings must be one-to-one")
        consumed = set()
        for name in rr["fields"]:
            target = mapping.get(name, name)
            if target in consumed:
                raise ValueError("Field mappings collide with an unmapped field")
            consumed.add(target)
            row = {**base, "field": name, "candidate_field": target}
            if target not in cr["fields"]:
                rows.append({**row, "status": "UNKNOWN", "reason": "Field evidence missing"})
                continue
            rs = ref.manifest["contracts"].get(key[3], {}).get(name, {})
            cs = cand.manifest["contracts"].get(key[3], {}).get(target, {})
            a, b = ref.array(rr, name), cand.array(cr, target)
            limit = tolerances.get(key[3], {}).get(name, {})
            absolute, relative = limit.get("atol", atol), limit.get("rtol", rtol)
            if any(not math.isfinite(v) or v < 0 for v in (absolute, relative)):
                raise ValueError("Tolerances must be finite and nonnegative")
            row.update(
                atol=absolute,
                rtol=relative,
                reference_shape=list(a.shape),
                candidate_shape=list(b.shape),
            )
            if not _declared(rs) or not _declared(cs):
                row.update(status="UNKNOWN", reason="Semantic, units or layout declaration missing")
            elif rs != cs:
                row.update(status="FAIL", reason="Explicit semantic contracts differ")
            elif a.shape != b.shape:
                row.update(status="FAIL", reason="Array shapes differ")
            elif a.size == 0:
                row.update(status="UNKNOWN", reason="Empty array provides no numeric evidence")
            elif not np.isfinite(a).all() or not np.isfinite(b).all():
                row.update(status="FAIL", reason="Nonfinite numeric evidence")
            elif not deterministic:
                row.update(
                    status="UNKNOWN", reason="Randomness is not fixed; value parity is inconclusive"
                )
            else:
                # Preserve integer precision (notably timestamps/counters above 2**53).
                if a.dtype.kind in "bui" and b.dtype.kind in "bui":
                    delta = np.abs(a.astype(object) - b.astype(object))
                    equal = np.asarray(
                        delta <= absolute + relative * np.abs(a.astype(object)), dtype=bool
                    )
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                        equal = np.isclose(b, a, atol=absolute, rtol=relative)
                max_error = float(delta.max())
                row.update(
                    status="PASS" if equal.all() else "FAIL",
                    reason="Within declared tolerance"
                    if equal.all()
                    else "Values exceed tolerance",
                    max_abs_error=max_error if math.isfinite(max_error) else "overflow",
                    mismatch_count=int(np.count_nonzero(~equal)),
                )
                if not equal.all():
                    index = tuple(np.argwhere(~equal)[0])
                    row.update(
                        first_index=[int(i) for i in index],
                        reference_value=a[index].item(),
                        candidate_value=b[index].item(),
                    )
            rows.append(row)
        for extra in cr["fields"].keys() - consumed:
            rows.append(
                {
                    **base,
                    "field": extra,
                    "status": "UNKNOWN",
                    "reason": "Candidate field has no reference evidence",
                }
            )
    if not rows:
        rows.append({"status": "UNKNOWN", "reason": "No recorded evidence", "field": "*"})
    status = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in rows)
        else ("UNKNOWN" if any(r["status"] == "UNKNOWN" for r in rows) else "PASS")
    )
    first = next((r for r in rows if r["status"] == "FAIL"), None)
    return {
        "schema": "vla-preflight.result/1",
        "status": status,
        "exit_code": EXIT_CODES[status],
        "scope": "Recorded I/O parity only; no task-success or robot-safety claim",
        "reference": ref.manifest["label"],
        "candidate": cand.manifest["label"],
        "versions": {"reference": ref.manifest["versions"], "candidate": cand.manifest["versions"]},
        "first_recorded_difference": first,
        "rows": rows,
        "counts": {s: sum(r["status"] == s for r in rows) for s in EXIT_CODES},
        "stage_order": sorted(stages, key=stage_rank.get),
    }
