# Contributing

Useful contributions make a deployment regression easier to reproduce. A new
adapter should execute a real upstream boundary, not only generate matching field
names. Small scoped cases are welcome.

## Local development

```sh
python -m pip install --index-url https://pypi.org/simple -e ".[dev]"
python -m pytest -q
ruff check src tests scripts examples integrations/run_bridges.py
ruff format --check src tests scripts examples integrations/run_bridges.py
python -m vla_preflight demo --out my-demo
```

Keep optional framework dependencies outside the core requirements. Core tests
must run without Torch, network, API key, robot or GPU. Do not edit the vendored
original upstream source to make an adapter pass.

## Add an adapter or regression case

1. Explain the actual boundary being checked and the upstream source/license.
   Pin a version/SHA and retain the original attribution. Declare whether the
   run uses a real model, synthetic policy, transport, simulation or hardware.
2. Define explicit stage/field semantics, units, layout, camera/channel order and
   normalization stats as applicable. State which stages remain unobserved.
3. Capture a healthy reference, replay its raw input into the candidate, and
   include a healthy control, a clearly labeled fault and a repaired capture.
   A real field report should cite its original issue/author and permission to
   share data; a seeded case must not be presented as an upstream bug.
4. If lifecycle matters, accept `policy_factory` in the probe factory, implement
   `infer`/`reset`, and set session support honestly. Use independent policies
   per session if claiming multi-session isolation. See `tests/test_preflight.py`.
5. Assert the earliest **recorded** failure location and an incomplete-evidence
   UNKNOWN case. Confirm inputs are unchanged and document tolerances.
6. Add runnable commands and actual evidence to the compatibility matrix. Mark
   unexecuted configurations as unverified. Include a short offline report or
   anonymized trace that a maintainer can reproduce without external credentials.

Do not add automatic physical-semantic inference, a general benchmark platform
or a model training requirement to the core. Discuss broader scope in your own
repository before implementing it; this local project does not automatically
send any issues or messages.

## Trace privacy and review

Use synthetic or permission-cleared arrays. Remove private camera frames,
locations, access tokens, hostnames and user identifiers before sharing. Numeric
files can contain sensitive data even when their filenames look harmless.
Review scripts and reports you share. No telemetry is collected by this project.

## Release ownership

The maintainer owns publication and release decisions. Keep release evidence
truthful: local checks are not remote CI, source-slice tests are not full model
evaluation, and seeded faults are not measured real-world bug prevalence.
