"""Seeded integration faults; none represents a discovered upstream bug."""

from copy import deepcopy
from html import escape
import json
from pathlib import Path
import time

import numpy as np

from .compare import compare
from .probes import ChunkPolicy, DemoBroker, lifecycle_probe
from .report import write_report
from .trace import Capture

STAGES = ["raw_observation", "model_input", "model_output", "command"]
IMAGE = {
    "semantic": "front_camera",
    "units": "uint8_0_255",
    "layout": "HWC",
    "kind": "image",
    "channels": ["R", "G", "B"],
}
STATE = {"semantic": "joint_position", "units": "rad", "layout": "D", "channels": ["joint0"]}
NORMALIZED = {
    "semantic": "joint_command",
    "units": "normalized",
    "layout": "D",
    "kind": "action",
    "channels": ["joint0"],
}
PHYSICAL = {**NORMALIZED, "units": "rad", "normalization": {"mean": [0.2], "std": [0.5]}}
CONTRACT = {
    "raw_observation": {
        "front": IMAGE,
        "wrist": {**IMAGE, "semantic": "wrist_camera"},
        "state": STATE,
    },
    "model_input": {"front": IMAGE, "wrist": {**IMAGE, "semantic": "wrist_camera"}, "state": STATE},
    "model_output": {"actions": NORMALIZED},
    "command": {"actions": PHYSICAL},
}


def observation(step):
    y, x = np.indices((40, 64))
    front = np.stack([(x * 4) % 256, (y * 6) % 256, np.full_like(x, 55 + step)], -1).astype("uint8")
    wrist = np.stack([np.full_like(x, 210), (x * 3 + step) % 256, (y * 5) % 256], -1).astype(
        "uint8"
    )
    return {"front": front, "wrist": wrist, "state": np.array([step / 10])}


def capture_demo(directory, fault=None, *, label=None, source=None):
    contract = deepcopy(CONTRACT)
    if fault == "unknown_semantics":
        contract["command"]["actions"].pop("units")
    capture = Capture(
        directory,
        stages=STAGES,
        contracts=contract,
        versions={"fixture": "seeded-demo/1", "pipeline": "numpy-demo/1"},
        deterministic=fault != "unfixed_randomness",
        label=label or (f"SEEDED FAULT: {fault}" if fault else "Healthy deterministic fixture"),
    )
    broker = DemoBroker(ChunkPolicy(), fault)
    # Candidate raw inputs are replayed from the saved reference, rather than regenerated.
    samples = (
        source.replay()
        if source
        else (
            ({"episode": episode, "step": step}, observation(step))
            for episode, length in (("episode-0", 7), ("episode-1", 5))
            for step in range(length)
        )
    )
    previous = None
    for record, raw in samples:
        episode, step = record["episode"], record["step"]
        if episode != previous:
            broker.reset()
            previous = episode
        coords = {"episode": episode, "step": step}
        capture.record("raw_observation", raw, **coords)
        model_input = {k: v.copy() for k, v in raw.items()}
        if fault == "camera_mapping":
            model_input["front"], model_input["wrist"] = model_input["wrist"], model_input["front"]
        if fault != "missing_stage":
            capture.record("model_input", model_input, **coords)
        actions = broker.infer({"marker": 1 if episode == "episode-0" else 2})["actions"] / 100
        capture.record("model_output", {"actions": actions}, **coords)
        command = actions * 0.5 + 0.2
        if fault == "wrong_denormalization":
            command = actions * 0.1 + 0.2
        if fault == "double_denormalization":
            command = command * 0.5 + 0.2
        capture.record("command", {"actions": command}, **coords)
    return capture.close()


def run_demo(directory):
    start = time.perf_counter()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    reference = capture_demo(path / "reference")
    scenarios = [
        ("healthy", None, "PASS", None),
        ("camera_mapping", "camera_mapping", "FAIL", "model_input"),
        ("wrong_denormalization", "wrong_denormalization", "FAIL", "command"),
        ("double_denormalization", "double_denormalization", "FAIL", "command"),
        ("chunk_index", "chunk_index", "FAIL", "model_output"),
        ("missing_reset", "missing_reset", "FAIL", "model_output"),
        ("fixed", None, "PASS", None),
        ("unknown_semantics", "unknown_semantics", "UNKNOWN", None),
        ("missing_stage", "missing_stage", "UNKNOWN", None),
        ("unfixed_randomness", "unfixed_randomness", "UNKNOWN", None),
    ]
    summary = []
    for name, fault, expected, stage in scenarios:
        candidate = capture_demo(
            path / name / "trace",
            fault,
            label=(
                "Fixed: camera mapping, denormalization, chunk index and reset restored"
                if name == "fixed"
                else None
            ),
            source=reference,
        )
        result = compare(reference, candidate)
        write_report(result, reference, candidate, path / name)
        first = result["first_recorded_difference"]
        actual_stage = first["stage"] if first else None
        assert result["status"] == expected and actual_stage == stage, (name, result)
        summary.append(
            {
                "scenario": name,
                "seeded_fault": fault if expected == "FAIL" else None,
                "expected": expected,
                "actual": result["status"],
                "first_recorded_stage": actual_stage,
                "report": f"{name}/index.html",
            }
        )
    probes = {
        name: lifecycle_probe(lambda factory, f=fault: DemoBroker(factory(), f))
        for name, fault in (
            ("healthy", None),
            ("chunk_index", "chunk_index"),
            ("missing_reset", "missing_reset"),
            ("fixed", None),
        )
    }
    assert [p["status"] for p in probes.values()] == ["PASS", "FAIL", "FAIL", "PASS"]
    evidence = {
        "schema": "vla-preflight.demo/1",
        "scenarios": summary,
        "probes": probes,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "seeded_faults": 5,
        "detected_seeded_faults": 5,
        "missed_seeded_faults": 0,
        "healthy_false_alarms": 0,
        "scope": "Synthetic fixtures only. No general detection-rate claim; no trained model or simulation.",
    }
    (path / "summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    cards = "".join(
        f'<a href="{r["report"]}"><b>{escape(r["scenario"].replace("_", " "))}</b>'
        f'<span class="{r["actual"]}">{r["actual"]}</span><small>First recorded difference: '
        f"{escape(r['first_recorded_stage'] or 'none')}</small></a>"
        for r in summary
    )
    (path / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA Preflight · Replay lab</title><style>
body{background:#f2f6f7;color:#17353f;font:16px/1.6 system-ui;margin:0}main{max-width:1020px;padding:40px 28px;margin:auto}header{font-size:23px;font-weight:800;border-bottom:1px solid #d9e3e7;padding-bottom:20px}h1{font-size:48px;line-height:1.1;letter-spacing:-2px;max-width:700px}p{max-width:700px;color:#5b7078}.tag{color:#087e8b;font-size:12px;letter-spacing:2px;margin-top:34px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:28px 0}a{display:block;background:white;border:1px solid #d9e3e7;border-radius:12px;padding:20px;color:inherit;text-decoration:none}a:hover{border-color:#087e8b;box-shadow:0 5px 20px #17353f12}b{text-transform:capitalize}span{float:right;font-size:12px;font-weight:800;padding:3px 9px;border-radius:5px}.PASS{background:#d7f1e8;color:#11634d}.FAIL{background:#fce0d5;color:#9f3c20}.UNKNOWN{background:#fff0cb;color:#805c08}small{display:block;margin-top:14px;color:#5b7078}code{background:#e4ecef;padding:3px 6px;border-radius:4px}footer{font-size:12px;color:#5b7078}@media(max-width:650px){.grid{grid-template-columns:1fr}h1{font-size:36px}}</style><main><header>◈ VLA Preflight</header><div class="tag">OFFLINE REPLAY LAB / SYNTHETIC FIXTURES</div><h1>The array shape is right.<br>Is the integration?</h1><p>Replay the same captured observation through healthy, deliberately broken and fixed pipelines. Inspect where recorded camera inputs, action scaling or chunk consumption first diverge.</p><p>Start with <b>camera mapping</b>, then inspect <b>missing reset</b> and the <b>fixed</b> control. Every fault below is injected by this demo; none is an upstream bug claim.</p><div class="grid">"""
        + cards
        + f"""</div><footer>Generated in {evidence["elapsed_seconds"]} seconds. Five seeded fault cases detected; zero missed seeded cases and zero healthy false alarms in these fixtures. Three incomplete-evidence controls remain UNKNOWN. No robot, model weights, API key or server. Lifecycle probe evidence: summary.json.</footer></main></html>""",
        encoding="utf-8",
    )
    return evidence
