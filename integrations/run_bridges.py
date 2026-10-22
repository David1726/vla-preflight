"""Run pinned upstream source slices, with no trained model, server or simulation.

LeRobot package initializers are bypassed to avoid unrelated dataset/video/robot
dependencies. All types, converters, pipeline and numeric processor code are
unmodified upstream modules. See docs/integrations.md for this boundary.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import types

import numpy as np

from vla_preflight import Capture, compare, write_report
from vla_preflight.probes import ChunkPolicy, lifecycle_probe

HERE = Path(__file__).resolve().parent
OPENPI = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT = "53feaef37b67e8f4c23b6953e880ff535e39d6fe"
SPEC = {"semantic": "fixture_command", "units": "fixture_units", "layout": "D", "kind": "action"}


def verify_sources():
    lock = json.loads((HERE / "sources.json").read_text())
    for file, meta in lock["files"].items():
        actual = hashlib.sha256((HERE / "vendor" / file).read_bytes()).hexdigest()
        if actual != meta["sha256"]:
            raise ValueError(f"Pinned upstream file hash mismatch: {file}")
    return lock


def namespace(name, path):
    if name in sys.modules:
        raise RuntimeError(f"Run bridge checks in a fresh process: {name} already loaded")
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def openpi_checks(out):
    namespace("openpi_client", HERE / "vendor/openpi_client")
    from openpi_client.action_chunk_broker import ActionChunkBroker

    class Wrapper:
        def __init__(self, policy, fault):
            self.broker = ActionChunkBroker(policy, action_horizon=3)
            self.fault = fault

        def infer(self, obs):
            if self.fault == "chunk_index":
                self.broker.infer(obs)  # SEEDED integration fault: consume an extra action.
            return self.broker.infer(obs)

        def reset(self):
            if self.fault != "missing_reset":  # SEEDED: caller omits the real broker reset.
                self.broker.reset()

    def capture(path, fault=None):
        c = Capture(
            path,
            stages=["raw_observation", "command"],
            contracts={
                "raw_observation": {"marker": {**SPEC, "kind": "state"}},
                "command": {"actions": SPEC},
            },
            versions={"openpi": OPENPI, "model": "deterministic ChunkPolicy/1"},
            deterministic=True,
            label=f"OpenPI real broker / {fault or 'healthy'}",
        )
        adapter = Wrapper(ChunkPolicy(), fault)
        for ep, marker in (("A", 1), ("B", 2)):
            adapter.reset()
            for step in range(5):
                obs = {"marker": np.array([marker])}
                c.record("raw_observation", obs, episode=ep, step=step)
                action = adapter.infer({"marker": marker})
                c.record("command", action, episode=ep, step=step)
        return c.close()

    reference = capture(out / "reference")
    results = []
    for name, fault, expected in (
        ("healthy", None, "PASS"),
        ("chunk_index", "chunk_index", "FAIL"),
        ("missing_reset", "missing_reset", "FAIL"),
        ("fixed", None, "PASS"),
    ):
        candidate = capture(out / name / "trace", fault)
        r = compare(reference, candidate)
        write_report(r, reference, candidate, out / name)
        probe = lifecycle_probe(lambda factory: Wrapper(factory(), fault))
        if r["status"] != expected or probe["status"] != expected:
            raise AssertionError((name, r, probe))
        results.append(
            {
                "scenario": name,
                "status": r["status"],
                "expected": expected,
                "first": r["first_recorded_difference"],
                "probe": probe,
            }
        )
    return {
        "source_sha": OPENPI,
        "executed": "Original ActionChunkBroker.infer/reset and dm-tree",
        "trained_model": False,
        "simulation": False,
        "multi_session": "unsupported; probe skipped",
        "results": results,
    }


def lerobot_checks(out):
    import torch

    root = HERE / "vendor/src/lerobot"
    namespace("lerobot", root)
    config = namespace("lerobot.configs", root / "configs")
    types_module = importlib.import_module("lerobot.configs.types")
    for name in ("FeatureType", "NormalizationMode", "PipelineFeatureType", "PolicyFeature"):
        setattr(config, name, getattr(types_module, name))
    namespace("lerobot.utils", root / "utils")
    namespace("lerobot.processor", root / "processor")
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.converters import create_transition
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        "observation.state": {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
        "action": {"mean": [0.1, 0.3], "std": [0.5, 0.2]},
    }
    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats, device="cpu"
    )
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats, device="cpu"
    )
    stages = ["raw_observation", "model_input", "model_output", "command"]
    contracts = {
        s: {
            "state" if s in stages[:2] else "actions": {
                "semantic": "synthetic_joint_position",
                "units": "rad" if s in (stages[0], stages[3]) else "normalized",
                "layout": "D",
                "kind": "state" if s in stages[:2] else "action",
                "channels": ["joint0", "joint1"],
            }
        }
        for s in stages
    }

    def capture(path, fault=False):
        c = Capture(
            path,
            stages=stages,
            contracts=contracts,
            deterministic=True,
            versions={"lerobot": LEROBOT, "torch": torch.__version__, "stats": stats},
            label="LeRobot real processor / " + ("SEEDED double inverse" if fault else "healthy"),
        )
        for step in range(4):
            state = torch.tensor([3.0 + step, 6.0 + step])
            original = state.clone()
            c.record(stages[0], {"state": state.numpy()}, episode="fixture", step=step)
            processed = normalizer(create_transition(observation={"observation.state": state}))
            actual = processed["observation"]["observation.state"]
            np.testing.assert_allclose(actual.numpy(), (state.numpy() - [1, 2]) / [2, 4], atol=1e-6)
            torch.testing.assert_close(state, original)  # Real processor must not mutate raw input.
            c.record(stages[1], {"state": actual.numpy()}, episode="fixture", step=step)
            action = torch.tensor([-1.0 + step / 10, 0.5 + step / 10])
            c.record(stages[2], {"actions": action.numpy()}, episode="fixture", step=step)
            output = unnormalizer(create_transition(action=action))
            expected = action.numpy() * [0.5, 0.2] + [0.1, 0.3]
            np.testing.assert_allclose(output["action"].numpy(), expected, atol=1e-6)
            if fault:
                output = unnormalizer(output)  # SEEDED caller applies correct processor twice.
            c.record(stages[3], {"actions": output["action"].numpy()}, episode="fixture", step=step)
        return c.close()

    reference = capture(out / "reference")
    results = []
    for name, fault, expected in (
        ("healthy", False, "PASS"),
        ("double_inverse", True, "FAIL"),
        ("fixed", False, "PASS"),
    ):
        candidate = capture(out / name / "trace", fault)
        r = compare(reference, candidate)
        write_report(r, reference, candidate, out / name)
        if r["status"] != expected:
            raise AssertionError((name, r))
        results.append(
            {
                "scenario": name,
                "status": r["status"],
                "expected": expected,
                "first": r["first_recorded_difference"],
            }
        )
    return {
        "source_sha": LEROBOT,
        "executed": "Original MEAN_STD normalizer/unnormalizer, converters, base pipeline and types",
        "loading": "package initializer isolation; no numeric code replaced",
        "torch": torch.__version__,
        "trained_model": False,
        "simulation": False,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bridge-output")
    args = parser.parse_args()
    verify_sources()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "openpi": openpi_checks(out / "openpi"),
        "lerobot": lerobot_checks(out / "lerobot"),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
