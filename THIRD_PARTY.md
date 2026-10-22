# Third-party attribution

The core `src/vla_preflight` is MIT licensed. The optional source verification fixtures
under `integrations/vendor` retain their upstream Apache-2.0 licenses and copyright
headers; they are **not relicensed under MIT**. They are included in the source
distribution, not the runtime wheel. No upstream code was edited.

| Source slice | Fixed revision | Preserved license |
| --- | --- | --- |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) `action_chunk_broker.py`, `base_policy.py` | `215abfb217dbac7d5f1273282331b9b1866c0479` | [Apache-2.0](integrations/vendor/OPENPI_LICENSE) |
| [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/53feaef37b67e8f4c23b6953e880ff535e39d6fe) types, normalization processors, converters, pipeline, constants, hub utilities | `53feaef37b67e8f4c23b6953e880ff535e39d6fe` | [Apache-2.0, copyright Hugging Face](integrations/vendor/LICENSE) |

[sources.json](integrations/sources.json) records every included upstream file's
original URL and SHA-256. Both fixed repository roots were checked for a NOTICE;
neither contains one. OpenPI's Gemma license and model code are not part of this
slice; no weights are included. Source verification rejects modified fixture files.

Runtime dependency: [NumPy](https://numpy.org/), BSD-3-Clause. Optional bridge
dependencies (PyTorch, dm-tree, Hugging Face Hub, safetensors and their transitive
dependencies) are installed separately and retain their own licenses. See the
environment freeze for the versions actually used.

The historical [StarVLA issue #501](https://github.com/starVLA/starVLA/issues/501)
and [PR #502](https://github.com/starVLA/starVLA/pull/502) motivated an integration
boundary example in the selection research. That contribution belongs to its
original author and upstream project. This project does not claim its discovery,
copy its patch, or advertise a StarVLA bridge as tested.
