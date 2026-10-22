"""JSON manifests and numeric NPY files; no pickle or executable trace objects."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "vla-preflight.trace/1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Capture:
    """Capture explicitly instrumented stages. Each destination must be new.

    Contracts map stage -> field -> {semantic, units, layout, ...}. A semantic
    descriptor is a declaration by the caller, never inferred from an array.
    """

    def __init__(self, directory, *, stages, contracts, versions, deterministic, label=""):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        if not stages or len(stages) != len(set(stages)) or not versions:
            raise ValueError("Declare distinct ordered stages and nonempty versions")
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be an explicit boolean")
        self.manifest = {
            "schema": SCHEMA,
            "stages": list(stages),
            "contracts": deepcopy(contracts),
            "versions": deepcopy(versions),
            "deterministic": bool(deterministic),
            "label": label,
            "records": [],
        }
        self.keys = set()

    def record(self, stage, values, *, episode, step, session="default"):
        if stage not in self.manifest["stages"] or not values:
            raise ValueError("Stage must be declared and contain at least one field")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isinstance(episode, str) or not isinstance(session, str):
            raise ValueError("episode and session must be strings")
        key = (session, episode, step, stage)
        if key in self.keys:
            raise ValueError(f"Duplicate stage coordinate: {key}")
        arrays = {}
        for name, value in values.items():
            if not isinstance(name, str):
                raise ValueError("Field names must be strings")
            array = np.array(value, copy=True)
            if array.dtype.kind not in "buif":
                raise ValueError("Only boolean, integer and real floating arrays are supported")
            arrays[name] = array
        fields = {}
        for index, (name, array) in enumerate(arrays.items()):
            file = f"array-{len(self.keys):06d}-{index:03d}.npy"
            np.save(self.path / file, array, allow_pickle=False)
            fields[name] = {
                "file": file,
                "sha256": digest(self.path / file),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        self.keys.add(key)
        self.manifest["records"].append(
            {
                "session": session,
                "episode": episode,
                "step": step,
                "stage": stage,
                "fields": fields,
            }
        )

    def close(self):
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        return Trace(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()


class Trace:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("schema") != SCHEMA:
            raise ValueError("Unsupported trace schema")
        if not m.get("stages") or len(set(m["stages"])) != len(m["stages"]):
            raise ValueError("Invalid stage order")
        if not m.get("versions") or type(m.get("deterministic")) is not bool:
            raise ValueError("Missing provenance or determinism declaration")
        self.records = {}
        for record in m["records"]:
            key = self.key(record)
            if key in self.records or record["stage"] not in m["stages"]:
                raise ValueError("Duplicate or undeclared stage coordinate")
            if not isinstance(record["step"], int) or record["step"] < 0:
                raise ValueError("Invalid step")
            self.records[key] = record
            for field in record["fields"]:
                self.array(record, field)  # Validate integrity before comparisons/reporting.

    @staticmethod
    def key(record):
        return tuple(record[k] for k in ("session", "episode", "step", "stage"))

    def array(self, record, field):
        meta = record["fields"][field]
        path = (self.path / meta["file"]).resolve()
        if not path.is_relative_to(self.path) or path.suffix != ".npy":
            raise ValueError("Array path escapes trace directory or is not NPY")
        if digest(path) != meta["sha256"]:
            raise ValueError("Array integrity mismatch")
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind not in "buif":
            raise ValueError("Unsafe or unsupported array dtype")
        if list(array.shape) != meta["shape"] or array.dtype.str != meta["dtype"]:
            raise ValueError("Array metadata mismatch")
        return array

    def replay(self, stage="raw_observation"):
        """Yield fresh arrays for a caller's pipeline; no robot commands are executed."""
        for record in self.manifest["records"]:
            if record["stage"] == stage:
                yield record.copy(), {name: self.array(record, name) for name in record["fields"]}


def capture_call(
    capture, function, values, *, input_stage, output_stage, episode, step, session="default"
):
    """Instrument a mapping -> mapping function without modifying its behavior."""
    coordinates = {"episode": episode, "step": step, "session": session}
    capture.record(input_stage, values, **coordinates)
    output = function(values)
    capture.record(output_stage, output, **coordinates)
    return output
