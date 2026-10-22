# Executed verification and remaining limits

Local release candidate: **0.1.0rc1**, Windows 11 / Python 3.12.15 / NumPy 2.2.6.
The separate source/wheel installs also ran with NumPy 2.5.4 and no Torch.
This work was performed locally on 2026-10-22 (Asia/Shanghai). See machine-readable
timestamps rather than interpreting repository file modification time as run time.

## Evidence you can inspect

- [Demo summary](evidence/demo-summary.json): ten cases with expected/actual
  outcomes and deterministic chunk/reset probe evidence.
- [Original upstream runs](evidence/bridges.json): real OpenPI broker and LeRobot
  processor-slice checks, revision identifiers, first recorded failures and limits.
- [Fresh source quickstart timing](evidence/quickstart.json): new environment,
  source install and demo, individual durations and outputs. Pip's inherited
  global cache was unwritable in this sandbox, so the recipe disables it;
  previous unsuccessful attempts are recorded. This is not a cold-network benchmark.
- [Local release checks](evidence/release-check.json): commands, exit codes,
  artifact checksums and execution durations. Logs and environment freeze remain
  beside it. No remote workflow run is claimed.
- [Wheel install smoke](evidence/wheel-smoke.json): wheel installation and
  installed CLI/demo execution in a separate NumPy-only environment.
- [Camera-swap HTML](reports/camera_mapping/index.html),
  [missed-reset HTML](reports/missing_reset/index.html),
  [fixed HTML](reports/fixed/index.html): standalone example reports with JSON
  next to each. Generate the full set with the demo command.

## What the cases established

| Set | Healthy | Seeded candidate | Repair | Interpretation |
| --- | --- | --- | --- | --- |
| Core replay | PASS | Camera swap → model_input; wrong/repeated scaling → command; skipped chunk → model_output; missing reset → episode-1 model_output | PASS | 5/5 seeded cases detected, 0 missed within this synthetic suite, 0 healthy false alarms |
| Evidence controls | Known semantics/determinism | Missing units, missing declared hook, unfixed randomness → UNKNOWN | Use complete evidence | No false PASS for those incomplete-evidence cases |
| Original OpenPI broker | PASS | Extra consumption / caller drops reset → command FAIL; probe FAIL | PASS | Original broker executed, synthetic inner policy; no upstream defect alleged |
| Original LeRobot processor | PASS | Caller repeats inverse → command FAIL | PASS | Original MEAN_STD processor executed with explicit expected output; no upstream defect alleged |

The camera fixture uses a policy independent of image content. It proves that a
camera boundary comparison catches the swap; it does not estimate the effect of
that swap on a trained model. Repairs restore all caller settings to the healthy
path and are captured again. They do not edit trace arrays or bypass comparison.

Core regression tests additionally exercise input capture/replay non-mutation,
float tolerances, uint64 exactness, booleans, nonfinite arrays, overflow evidence,
shape/semantic mismatch, missing/empty evidence, path traversal and corruption,
HTML escaping, explicit renaming, CLI exit codes, session-capable adapters with
both healthy and shared-state failure controls. Test counts and final output are
in `evidence/tests.log`; outcome evidence above is more useful than a test count alone.

## Visual and interactive QA limitation

The browser tool rejected `file://` navigation under its URL security policy.
No alternate browser surface or local-server workaround was used. The HTML file
was handed to the Codex file viewer; its browser appearance and interactive
filter behavior have **not been visually verified in this session**. Static
artifact checks validate embedded PNGs, local links, numeric SVG coordinates,
labels, escaped content and no unresolved report placeholders.

The README GIF is a separately rendered walkthrough made from actual trace and
JSON evidence. Its PNG frames were opened and inspected in the image tool. It is
clearly labeled as an illustration, not a browser screenshot. Before public
release, open the actual report in a local browser and check the filters,
expandable evidence, layout, images, charts and narrow-screen behavior.

## Unverified or unsupported

- Full upstream environment install, model weights/inference, websocket transport,
  dataset loading, simulation, robot commands and task performance were not run.
- LeRobot covers the pinned source slice and MEAN_STD mode only; its package
  initializers are isolated. No other normalization mode/version is certified.
- OpenPI's single broker does not claim multi-session routing. Its interleaving
  probe is skipped; a local session router test establishes only the probe's
  behavior with declared support.
- Linux and other Python versions have a normal CI configuration, not an executed
  remote CI result. Very long traces, memory budgets and realtime capture overhead
  have not been characterized.
- Semantic declarations, reference correctness, expected episode length and
  determinism settings come from the caller. Identical faults or missing steps
  absent from both traces cannot be independently discovered.
- PyPI name availability, external adoption and star potential have not been
  established. The local validation described here predates GitHub publication;
  no upstream issue/comment/PR or PyPI publication was part of that validation.
