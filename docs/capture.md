# Capture, replay and compare your pipeline

## 1. Declare the boundaries you can actually observe

The trace API accepts a mapping of **flat field names to numeric arrays** at each
hook. Flatten nested observations explicitly, convert tensors with
`tensor.detach().cpu().numpy()`, and choose names that are stable across runs.
No arbitrary policy objects are serialized.

```python
import numpy as np
from vla_preflight import Capture, Trace, compare, write_report

state = {"semantic": "joint_position", "units": "rad", "layout": "D",
         "channels": ["shoulder", "elbow"]}
action = {"semantic": "joint_target", "units": "rad", "layout": "D",
          "channels": ["shoulder", "elbow"], "kind": "action"}
contracts = {"raw_observation": {"state": state}, "command": {"actions": action}}

def record_pipeline(destination, samples, pipeline, version):
    with Capture(destination, stages=["raw_observation", "command"],
                 contracts=contracts, versions={"pipeline": version},
                 deterministic=True, label=version) as capture:
        previous_episode = None
        for coordinate, observation in samples:
            if previous_episode != coordinate["episode"]:
                pipeline.reset()
                previous_episode = coordinate["episode"]
            coords = {k: coordinate[k] for k in ("episode", "step")}
            capture.record("raw_observation", observation, **coords)
            command = pipeline.infer(observation)
            capture.record("command", {"actions": command}, **coords)

# reference_samples: iterable of ({"episode": "episode-0", "step": 0},
#                                {"state": np.array([0.1, 0.2])})
# record_pipeline("traces/reference", reference_samples, known_good, "reference-sha")
# record_pipeline("traces/candidate", Trace("traces/reference").replay(), candidate, "candidate-sha")
# result = compare("traces/reference", "traces/candidate")
# write_report(result, "traces/reference", "traces/candidate", "parity-report")
```

`examples/check_parity.py` is a runnable version with a tiny deterministic policy.
The optional `capture_call()` helper wraps a mapping-to-mapping function; it
records its declared input/output stages and returns its original output. It
does not insert hooks inside the function. Capture copies values at record time,
and replay loads fresh arrays each time.

Declare only selected stages, in pipeline order. A declared stage missing at a
recorded coordinate becomes UNKNOWN, even when absent in both traces. A stage
neither trace declares is outside the comparison scope. An empty trace cannot
pass. The caller must record every step of the intended regression window;
the tool cannot detect steps absent from **both** manifests or invent an expected
episode length. Sparse step numbers are allowed.

## 2. Make semantics explicit

Every compared field needs `semantic`, `units` and `layout`. Use a meaningful
string such as `normalized` for normalized values or `dimensionless` where
appropriate; missing/empty/`unknown` is insufficient. Extra declaration fields
such as `channels`, `coordinate_frame`, `normalization` or `camera_id` participate
in exact contract equality. Caller declarations are not independently inferred
or certified.

For image previews, declare `kind: image`, `layout: HWC`, `channels: [R,G,B]`,
and provide an HWC uint8 RGB array. Other layouts can still be numerically
compared; explicitly adapt to HWC if you want previews. No automatic channel
conversion or image normalization is performed. For curves declare `kind: action`.

Examples of useful declarations:

```json
{
  "semantic": "end_effector_delta", "units": "m", "layout": "D",
  "channels": ["dx", "dy", "dz"], "coordinate_frame": "robot_base"
}
```

## 3. Configure renaming and tolerances without hidden conversions

`compare-config.json`:

```json
{
  "field_map": {"model_input": {"front": "observation.images.front"}},
  "tolerances": {"command": {"actions": {"atol": 0.00001, "rtol": 0.0001}}}
}
```

```sh
vla-preflight compare traces/reference traces/candidate --config compare-config.json --out parity-report
```

Maps rename reference fields to candidate fields **within the same stage**.
Mappings must be one-to-one, including unmapped names. Their descriptors must
still agree. Unknown config keys are errors. Numeric checks use
`abs(candidate - reference) <= atol + rtol * abs(reference)` elementwise, with
defaults `atol=1e-6`, `rtol=1e-5`. Integers retain exact difference precision.
Tolerance settings must be finite and nonnegative. Units, axes, channel order and
normalization are never silently converted.

For stochastic policies, set `deterministic=False` unless **the caller actually
fixed the randomness** (seed/model settings and backend determinism as needed).
Value parity then remains UNKNOWN, even for equal samples; structural/declared
contract problems or nonfinite arrays can still fail. No statistical equivalence
test is claimed.

## 4. Probe lifecycle independently

```python
from vla_preflight.probes import lifecycle_probe

# Existing adapter wraps a supplied synthetic policy and exposes infer/reset.
# factory receives a policy factory so session-aware adapters can create one
# independent policy instance per session.
# result = lifecycle_probe(lambda policy_factory: YourAdapter(policy_factory()),
#                          supports_sessions=False)
```

The policy emits three distinct action rows per call. Checks consume eight
steps across chunk boundaries, reset a partially consumed chunk, and verify that
inner policy state restarts. Only adapters explicitly declaring sessions run
interleaving and reset-isolation checks. A single-session SKIP does not invalidate
the chunk/reset PASS; it is visible in probe JSON. Exceptions from your adapter
propagate to the test runner as errors rather than being presented as parity PASS.

## Trace format and operational limits

`manifest.json` contains schema version, ordered selected stages, contracts,
software versions, caller determinism declaration, and records keyed by
`(session, episode, step, stage)`. Each field references an independent numeric NPY
file with shape, dtype and SHA-256. Event records inherit the manifest schema and
versions. Duplicate coordinates are rejected. Do not edit the manifest manually
to make a failing run pass; capture the actual repaired pipeline in a new folder.

Capture writes synchronously, suitable for offline replay and CI. An exception
inside a capture context leaves an incomplete folder without a finalized
manifest; preserve it for debugging or choose a new destination. There is no
crash recovery, compression, streaming database or hard memory budget. Do not
feed untrusted huge traces into this release. Digests detect accidental changes;
they are not authentication. Reports contain captured data: anonymize before sharing.
