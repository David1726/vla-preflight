"""Deterministic chunk/reset probes, independent of trained model behavior."""

import numpy as np


class ChunkPolicy:
    """Three distinct commands per inference, resettable, observation dependent."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def infer(self, observation):
        base = float(observation["marker"]) * 100 + self.calls * 10
        self.calls += 1
        return {"actions": np.arange(3, dtype=float)[:, None] + base}

    def reset(self):
        self.calls = 0
        self.resets += 1


class DemoBroker:
    """Small demo adapter. This is not an OpenPI implementation or bridge."""

    def __init__(self, policy, fault=None):
        self.policy, self.fault = policy, fault
        self.cache, self.index = None, 0

    def infer(self, observation):
        if self.cache is None or self.index >= len(self.cache):
            self.cache = self.policy.infer(observation)["actions"]
            self.index = 1 if self.fault == "chunk_index" else 0
        output = self.cache[self.index].copy()
        self.index += 1
        return {"actions": output}

    def reset(self):
        if self.fault == "missing_reset":
            return
        self.cache, self.index = None, 0
        self.policy.reset()


def lifecycle_probe(factory, *, supports_sessions=False):
    """factory(policy_factory) -> adapter, exposing infer(obs, session) / reset(session).

    Single-session adapters may omit the session argument. An adapter declaring
    sessions must independently instantiate policy_factory for each session.
    """
    adapter = factory(ChunkPolicy)
    checks = []

    def infer(marker, session="A"):
        result = (
            adapter.infer({"marker": marker}, session=session)
            if supports_sessions
            else (adapter.infer({"marker": marker}))
        )
        return np.asarray(result["actions"]).reshape(-1).tolist()

    def reset(session="A"):
        adapter.reset(session=session) if supports_sessions else adapter.reset()

    def check(name, actual, expected):
        checks.append(
            {
                "name": name,
                "status": "PASS" if actual == expected else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    reset()
    check(
        "chunk_consumption",
        [infer(1) for _ in range(8)],
        [[100.0], [101.0], [102.0], [110.0], [111.0], [112.0], [120.0], [121.0]],
    )
    reset()
    check("reset_discards_partial_chunk_and_forwards", infer(2), [200.0])
    reset()
    check("reset_restarts_inner_policy", infer(2), [200.0])
    if supports_sessions:
        reset("A")
        reset("B")
        check(
            "session_interleaving",
            [infer(1, "A"), infer(9, "B"), infer(1, "A"), infer(9, "B")],
            [[100.0], [900.0], [101.0], [901.0]],
        )
        reset("A")
        check("session_reset_isolation", infer(9, "B"), [902.0])
    else:
        checks.append(
            {
                "name": "session_interleaving",
                "status": "SKIP",
                "reason": "Adapter does not declare multi-session support",
            }
        )
    return {
        "schema": "vla-preflight.probe/1",
        "status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        "checks": checks,
        "model": "deterministic synthetic ChunkPolicy",
        "simulation": False,
    }
