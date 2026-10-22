from copy import deepcopy
import base64
from html.parser import HTMLParser
import json
import math
import struct
import zlib

import numpy as np
import pytest

from vla_preflight import Capture, Trace, capture_call, compare, write_report
from vla_preflight.cli import main
from vla_preflight.demo import run_demo
from vla_preflight.probes import DemoBroker, lifecycle_probe

SPEC = {"semantic": "position", "units": "rad", "layout": "D"}


def trace(path, data=(1.0, 2.0), spec=SPEC, deterministic=True, stages=None, name="x"):
    c = Capture(
        path,
        stages=stages or ["input"],
        contracts={"input": {name: spec}},
        versions={"fixture": "1"},
        deterministic=deterministic,
    )
    c.record("input", {name: np.asarray(data)}, episode="ep", step=0)
    return c.close()


def test_demo_expected_diagnoses(tmp_path):
    result = run_demo(tmp_path / "demo")
    assert result["detected_seeded_faults"] == 5
    assert result["missed_seeded_faults"] == result["healthy_false_alarms"] == 0
    reset = next(r for r in result["scenarios"] if r["scenario"] == "missing_reset")
    report = json.loads((tmp_path / "demo/missing_reset/result.json").read_text())
    first = report["first_recorded_difference"]
    assert (first["episode"], first["step"], first["stage"]) == ("episode-1", 0, "model_output")
    assert reset["actual"] == "FAIL"
    assert sum(r["actual"] == "UNKNOWN" for r in result["scenarios"]) == 3


def test_capture_freezes_input_and_replay_is_fresh(tmp_path):
    values = np.array([1.0, 2.0])
    c = Capture(
        tmp_path / "trace",
        stages=["input"],
        contracts={"input": {"x": SPEC}},
        versions={"fixture": "1"},
        deterministic=True,
    )
    c.record("input", {"x": values}, episode="ep", step=0)
    values[0] = 99
    t = c.close()
    _, replay = next(t.replay("input"))
    assert replay["x"].tolist() == [1.0, 2.0]
    replay["x"][0] = -10
    assert next(t.replay("input"))[1]["x"].tolist() == [1.0, 2.0]


def test_reject_ambiguous_determinism_declaration(tmp_path):
    with pytest.raises(ValueError, match="explicit boolean"):
        Capture(
            tmp_path / "a",
            stages=["input"],
            contracts={},
            versions={"fixture": "1"},
            deterministic="false",
        )


@pytest.mark.parametrize("delta,status", [(0.0, "PASS"), (1e-7, "PASS"), (0.01, "FAIL")])
def test_tolerances_and_numeric_evidence(tmp_path, delta, status):
    a = trace(tmp_path / "a")
    b = trace(tmp_path / "b", [1 + delta, 2])
    result = compare(a, b, rtol=0, atol=1e-6)
    assert result["status"] == status
    if status == "FAIL":
        assert result["first_recorded_difference"]["first_index"] == [0]


@pytest.mark.parametrize("spec", [{}, {"semantic": "position"}, {**SPEC, "units": "unknown"}])
def test_unknown_semantics_even_when_values_match(tmp_path, spec):
    assert compare(trace(tmp_path / "a"), trace(tmp_path / "b", spec=spec))["status"] == "UNKNOWN"


def test_explicit_contract_mismatch(tmp_path):
    r = compare(trace(tmp_path / "a"), trace(tmp_path / "b", spec={**SPEC, "units": "m"}))
    assert r["status"] == "FAIL"
    assert "contracts" in r["first_recorded_difference"]["reason"]


def test_unfixed_randomness_does_not_fail_value_parity(tmp_path):
    r = compare(trace(tmp_path / "a"), trace(tmp_path / "b", [300.0, 400.0], deterministic=False))
    assert r["status"] == "UNKNOWN"
    assert r["first_recorded_difference"] is None


def test_missing_declared_stage_is_unknown(tmp_path):
    a = trace(tmp_path / "a", stages=["input", "unobserved"])
    b = trace(tmp_path / "b", stages=["input", "unobserved"])
    assert compare(a, b)["status"] == "UNKNOWN"


def test_empty_trace_unknown(tmp_path):
    for name in ("a", "b"):
        Capture(
            tmp_path / name,
            stages=["input"],
            contracts={},
            versions={"fixture": "1"},
            deterministic=True,
        ).close()
    assert compare(tmp_path / "a", tmp_path / "b")["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "data,status",
    [([float("nan")], "FAIL"), ([float("inf")], "FAIL"), ([], "UNKNOWN"), ([[1.0, 2.0]], "FAIL")],
)
def test_invalid_or_incompatible_numeric_evidence(tmp_path, data, status):
    a = trace(
        tmp_path / "a", [] if data == [] else [1.0, 2.0] if isinstance(data[0], list) else [1.0]
    )
    assert compare(a, trace(tmp_path / "b", data))["status"] == status


def test_explicit_mapping_and_per_field_tolerance(tmp_path):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b", [1.01, 2.0], name="joint")
    result = compare(
        a,
        b,
        field_map={"input": {"x": "joint"}},
        tolerances={"input": {"x": {"atol": 0.02, "rtol": 0}}},
    )
    assert result["status"] == "PASS"
    assert result["rows"][0]["candidate_field"] == "joint"


def test_integer_precision_and_boolean_values(tmp_path):
    a = trace(tmp_path / "a", np.array([2**63 + 1], dtype="uint64"))
    b = trace(tmp_path / "b", np.array([2**63 + 2], dtype="uint64"))
    result = compare(a, b, atol=0, rtol=0)
    assert result["status"] == "FAIL"
    assert result["first_recorded_difference"]["reference_value"] == 2**63 + 1
    a, b = trace(tmp_path / "c", [True]), trace(tmp_path / "d", [False])
    assert compare(a, b, atol=0, rtol=0)["status"] == "FAIL"


def test_overflow_difference_still_produces_json(tmp_path):
    a, b = trace(tmp_path / "a", [1e308]), trace(tmp_path / "b", [-1e308])
    result = compare(a, b)
    assert result["status"] == "FAIL"
    assert result["first_recorded_difference"]["max_abs_error"] == "overflow"
    write_report(result, a, b, tmp_path / "report")


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tmp_path, tolerance):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b")
    with pytest.raises(ValueError):
        compare(a, b, atol=tolerance)


def test_array_integrity_and_path_validation(tmp_path):
    trace(tmp_path / "a")
    manifest_path = tmp_path / "a/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    meta = manifest["records"][0]["fields"]["x"]
    original = deepcopy(meta)
    meta["file"] = "../../escape.npy"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        Trace(tmp_path / "a")
    manifest["records"][0]["fields"]["x"] = original
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "a" / original["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        Trace(tmp_path / "a")


def test_reject_object_and_duplicate_capture(tmp_path):
    c = Capture(
        tmp_path / "a",
        stages=["input"],
        contracts={},
        versions={"fixture": "1"},
        deterministic=True,
    )
    with pytest.raises(ValueError, match="Only"):
        c.record("input", {"object": np.array([{}], dtype=object)}, episode="ep", step=0)
    c.record("input", {"x": [1.0]}, episode="ep", step=0)
    with pytest.raises(ValueError, match="Duplicate"):
        c.record("input", {"x": [1.0]}, episode="ep", step=0)


@pytest.mark.parametrize(
    "fault,status", [(None, "PASS"), ("chunk_index", "FAIL"), ("missing_reset", "FAIL")]
)
def test_lifecycle_probes(fault, status):
    result = lifecycle_probe(lambda policy_factory: DemoBroker(policy_factory(), fault))
    assert result["status"] == status
    assert result["checks"][-1]["status"] == "SKIP"


def test_multi_session_declared_and_broken_isolation():
    class Sessions:
        def __init__(self, factory, broken=False):
            self.factory, self.broken, self.sessions = factory, broken, {}

        def get(self, session):
            key = "shared" if self.broken else session
            if key not in self.sessions:
                self.sessions[key] = DemoBroker(self.factory())
            return self.sessions[key]

        def infer(self, obs, session):
            return self.get(session).infer(obs)

        def reset(self, session):
            self.get(session).reset()

    for broken, status in ((False, "PASS"), (True, "FAIL")):
        assert (
            lifecycle_probe(lambda f: Sessions(f, broken), supports_sessions=True)["status"]
            == status
        )


def test_wrapper_and_report_escape_untrusted_labels(tmp_path):
    c = Capture(
        tmp_path / "a",
        stages=["input", "output"],
        contracts={s: {"x": SPEC} for s in ("input", "output")},
        versions={"fixture": "</pre><script>bad()</script>"},
        deterministic=True,
        label="<img src=x onerror=bad()>",
    )
    out = capture_call(
        c,
        lambda x: {"x": x["x"] * 2},
        {"x": np.array([1.0])},
        input_stage="input",
        output_stage="output",
        episode="ep",
        step=0,
    )
    assert out["x"].tolist() == [2.0]
    a = c.close()
    report = write_report(compare(a, a), a, a, tmp_path / "report")
    html = report.read_text(encoding="utf-8")
    assert "<script>bad()" not in html and "<img src=x" not in html
    assert "connect-src 'none'" in html


def test_cli_status_codes(tmp_path):
    a, b = trace(tmp_path / "a"), trace(tmp_path / "b", [10.0, 20.0])
    assert main(["compare", str(a.path), str(a.path), "--out", str(tmp_path / "pass")]) == 0
    assert main(["compare", str(a.path), str(b.path), "--out", str(tmp_path / "fail")]) == 1
    unknown = trace(tmp_path / "unknown", spec={})
    assert (
        main(["compare", str(a.path), str(unknown.path), "--out", str(tmp_path / "unknown-report")])
        == 2
    )
    assert main(["compare", str(a.path), "missing", "--out", str(tmp_path / "error")]) == 3


def test_report_embedded_images_curves_links_and_placeholders(tmp_path):
    run_demo(tmp_path / "demo")

    class Artifacts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images, self.links, self.curves = [], [], []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "img":
                self.images.append(attrs["src"])
            if tag == "a":
                self.links.append(attrs["href"])
            if tag == "polyline":
                self.curves.append(attrs["points"])
            if tag == "script":
                assert "src" not in attrs

    report = tmp_path / "demo/camera_mapping/index.html"
    html = report.read_text(encoding="utf-8")
    assert "{{" not in html
    parsed = Artifacts()
    parsed.feed(html)
    assert parsed.images and parsed.curves and parsed.links
    for source in parsed.images:
        blob = base64.b64decode(source.split(",", 1)[1])
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", blob[16:24])
        compressed, offset = b"", 8
        while offset < len(blob):
            length = struct.unpack(">I", blob[offset : offset + 4])[0]
            kind = blob[offset + 4 : offset + 8]
            if kind == b"IDAT":
                compressed += blob[offset + 8 : offset + 8 + length]
            offset += length + 12
        assert len(zlib.decompress(compressed)) == height * (1 + width * 3)
    for curve in parsed.curves:
        assert all(math.isfinite(float(v)) for pair in curve.split() for v in pair.split(","))
    for link in parsed.links:
        assert (report.parent / link).is_file()
