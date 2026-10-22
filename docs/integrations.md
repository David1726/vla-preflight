# Two real upstream paths

These checks run original, fixed-version upstream source. Synthetic observation,
statistics and policy fixtures remove hardware/weight requirements. They are not
claims about model quality or upstream defects.

## Run both local checks

Use Python **3.12** for the LeRobot source slice (it uses Python 3.12 syntax).
The NumPy demo itself supports Python 3.10+. In an environment dedicated to this project:

```sh
python -m pip install --index-url https://pypi.org/simple -e .
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
python -m pip install --index-url https://pypi.org/simple -r integrations/requirements.txt
python integrations/run_bridges.py --out bridge-output
```

On Windows use `.venv/Scripts/python.exe` for each `python` above; on POSIX use
`.venv/bin/python`. Bridge dependencies are optional and considerably larger
than the quickstart (the Windows CPU Torch wheel downloaded here was 216 MB).
No model weights are downloaded. Use a new output folder on repeat runs.

All source files are present locally, checked against
[sources.json](../integrations/sources.json) before execution. After installing
dependencies the bridge runner is offline. Reports are under
`bridge-output/{openpi,lerobot}/<scenario>/index.html`; raw probe/comparison evidence
is in `bridge-output/verification.json`. The compact committed copy is
[bridge evidence](evidence/bridges.json).

## OpenPI: capture broker consumption and reset

Fixed revision: `215abfb217dbac7d5f1273282331b9b1866c0479`.
[Original broker source](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py).

The runner imports the **unmodified** `ActionChunkBroker` and `BasePolicy` from
the included source slice, using real `dm-tree` and `typing_extensions`. A
namespace package path is used because only these two modules are included.
`ChunkPolicy` is this project's deterministic synthetic inner policy, not an
OpenPI model. No mock broker, websocket service or simulation stands in for the
target code.

The known policy emits `[100, 101, 102]`, then `[110, 111, 112]`, and so on.
The horizon is explicitly 3. A partially consumed episode ends, then the caller
resets and changes the observation marker. Healthy and repaired wrapper calls
forward reset to the original broker and inner policy. Seeded wrapper faults
either consume an extra row or omit that reset. The original broker is unchanged.

For your installed `openpi_client`, insert capture around the broker:

```python
from openpi_client.action_chunk_broker import ActionChunkBroker

broker = ActionChunkBroker(your_policy, action_horizon=your_execution_horizon)
broker.reset()  # At the actual episode boundary, before the first infer.
# capture.record("raw_observation", flattened_observation, episode=episode, step=step)
result = broker.infer(your_observation)
# capture.record("command", result, episode=episode, step=step)
```

Declare your real action fields/units; the runner's `fixture_units` are explicitly
synthetic. To observe preprocessing or denormalization, instrument those actual
boundaries inside your caller too. A command-only difference identifies a recorded
boundary, not a particular internal operation. The broker's `action_horizon`
is a caller setting; Preflight does not infer it or validate all possible horizons.

The real broker is a single-policy instance with one cache. We **do not declare
multi-session support**, so interleaving probes are skipped with a reason. A
session router built around independent brokers may declare support and run
the isolation checks; that router needs its own tests.

## LeRobot: record real normalization and inverse output

Fixed revision: `53feaef37b67e8f4c23b6953e880ff535e39d6fe`.
[Original numeric processor](https://github.com/huggingface/lerobot/blob/53feaef37b67e8f4c23b6953e880ff535e39d6fe/src/lerobot/processor/normalize_processor.py).

The runner executes original `NormalizerProcessorStep.__call__` and
`UnnormalizerProcessorStep.__call__` with real CPU Torch tensors, real
`create_transition`, real config/types and base pipeline class definitions.
It supplies explicit two-joint MEAN_STD stats and checks expected values with an
independent arithmetic oracle, as well as raw tensor non-mutation.

For example, raw state `[3, 6]`, mean `[1, 2]`, std `[2, 4]` becomes `[1, 1]`
within tolerance; action `[-1, .5]`, mean `[.1, .3]`, std `[.5, .2]` becomes
`[-.4, .4]`. Deliberately invoking the same correct inverse processor twice
produces `[-.1, .38]` and fails at the command boundary. Removing the extra call
passes. This is a caller fault, not a LeRobot bug.

In a fully installed LeRobot project, hooks go around your existing processors:

```python
from lerobot.processor.converters import create_transition

# capture.record("raw_observation", flattened_numpy_observation, episode=episode, step=step)
processed = your_normalizer(create_transition(observation=your_tensor_observation))
# capture.record("model_input", flattened_numpy(processed["observation"]), episode=episode, step=step)
# normalized_action = your_policy(...)
# capture.record("model_output", {"actions": normalized_action.detach().cpu().numpy()}, ...)
physical = your_unnormalizer(create_transition(action=normalized_action))["action"]
# capture.record("command", {"actions": physical.detach().cpu().numpy()}, episode=episode, step=step)
```

The commented hooks show placement; use actual tensors and declarations from your
project, and the runnable runner for a complete copyable fixture.

### Precise source-loading limitation

The fixture uses namespace modules for `lerobot`, `lerobot.processor`,
`lerobot.utils` and `lerobot.configs`, and re-exports the original config types.
It **does not execute package `__init__.py` files**, which pull in unrelated
datasets/video/robot modules. Numeric processors, converters, pipeline, constants,
hub utilities and type implementations are unmodified. No stats/model/dataset
reader is needed; the runner supplies synthetic stats directly. Hub utilities
are imported but no Hub/network operation runs.

Consequently this validates that specific source slice, not full-package
dependency resolution, a pretrained checkpoint pipeline, all normalization modes
or all upstream releases. Existing LeRobot contribution work and environments
were not modified; the snapshot uses clean committed source, not a local patch.

## Compatibility matrix

| Component | Version tested locally | OS / execution | Result and scope |
| --- | --- | --- | --- |
| Core | 0.1.0rc1, Python 3.12.15, NumPy 2.2.6 (development), 2.5.4 (isolated installs) | Windows 11, CPU | Unit/CLI/demo/replay checks; packaged install smoke without Torch |
| OpenPI client source slice | `215abfb217dbac7d5f1273282331b9b1866c0479` | Windows 11, CPU; dm-tree 0.1.9 | Original broker infer/reset; healthy/fixed PASS, two seeded wrapper failures |
| LeRobot processor source slice | `53feaef37b67e8f4c23b6953e880ff535e39d6fe` | Windows 11; Torch 2.7.1+cpu | Original MEAN_STD normalization/inverse; healthy/fixed PASS, one seeded caller failure |
| Python 3.10/3.11/3.13 and Linux core | CI configuration supplied | Remote CI not executed | Not locally verified; intended core support |
| Full upstream install, model/transport, simulator, robot | None | Not run | No compatibility/task-success claim |
| StarVLA / other ecosystems | None | Not run | No adapter support claim |
