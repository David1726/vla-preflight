"""Copy this pattern into a PR check; replace only Pipeline with your adapter."""

import argparse
from pathlib import Path

import numpy as np

from vla_preflight import Capture, Trace, compare, write_report


class Pipeline:
    def reset(self):
        self.offset = 0

    def infer(self, observation):
        return observation["state"] * 0.5 + self.offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="example-output")
    args = parser.parse_args()
    root = Path(args.out)
    spec = {"semantic": "joint_position", "units": "rad", "layout": "D"}
    contracts = {
        "raw_observation": {"state": spec},
        "command": {"actions": {**spec, "kind": "action"}},
    }
    for version in ("reference", "candidate"):
        samples = (
            Trace(root / "reference").replay()
            if version == "candidate"
            else (
                ({"episode": "episode-0", "step": step}, {"state": np.array([step / 10, 0.2])})
                for step in range(5)
            )
        )
        pipeline = Pipeline()
        pipeline.reset()
        with Capture(
            root / version,
            stages=list(contracts),
            contracts=contracts,
            versions={"example": "1", "role": version},
            deterministic=True,
            label=version,
        ) as c:
            for coordinate, observation in samples:
                coords = {k: coordinate[k] for k in ("episode", "step")}
                c.record("raw_observation", observation, **coords)
                c.record("command", {"actions": pipeline.infer(observation)}, **coords)
    result = compare(root / "reference", root / "candidate")
    write_report(result, root / "reference", root / "candidate", root / "report")
    print(result["status"])
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
