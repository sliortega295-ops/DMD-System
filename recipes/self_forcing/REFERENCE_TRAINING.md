# Original Self-Forcing reference training

This opt-in route uses the existing original LiveWan baseline implementation,
not the C51 optimized role-pool pipeline. It starts from real supplied frozen
weights or a complete Stage1 Student and owns a finite 2..64-cycle input stream.
It is CPU/source-qualified; pretrained CUDA/NCCL training, capacity and restart
remain **NOT_RUN** in this framework integration.

## Exact owner and algorithm

| Item | Reference route |
| --- | --- |
| Model owners | Student 1.3B: world16 FSDP2, FP32 trainables; Teacher 14B: frozen BF16 world16 FSDP2; Critic 1.3B: replicated frozen BF16 base plus FP32 LoRA32 |
| Workload | 368x640, 81 frames, real global B16, original four-step causal rollout, continuous shifted-uniform sigma |
| Updates per complete cycle | G1/C5/EMA0; original losses, clipping, gradient synchronization and foreach AdamW |
| Initialization | Actual frozen model inventories, optional complete Stage1 Student, independent explicit Critic LoRA seed; genuine G0/C0/EMA0 and zero Adam steps |
| Execution | Original `build_models(mode="baseline")`, `_build_streamers`, `build_score_context` once, then unchanged `_run_baseline(step_offset=5*cycle)` |
| Restart | All-rank complete-cycle checkpoint, same source/assets/finite horizon/inputs/seed/precision/optimizer/topology |

This is **not all-role FULL_SHARD**: Critic LoRA remains replicated. It does
not inherit a historical systems-only bootstrap, relabel a C51 trace, merge
LoRA, enable optimized kernels or offer a new timing result. Selected C51 still
requires its separate saved-plan contract and remains bounded to two cycles.

## Prepare genuine inputs and assets

1. Prepare 80 distinct original TeacherLatents records per requested cycle,
   following [the native input guide](../../docs/guides/livewan_inputs.md).
   Use the same explicit seed and horizon in preparation and training.
2. Supply clean source checkouts at the repository's pinned LiveWan and Wan2.1
   gitlinks. Inspect those origins rather than substituting another package.
3. Supply original Student/Critic 1.3B and Teacher 14B config plus safetensors
   files. Configuration/index/full weight identities are inventoried. At
   execution, the actual original constructors on meta tensors verify every
   parameter key/shape before allocating physical model owners. No weights,
   CUDA kernels or packages are downloaded.

The following API builds the **real** asset document from your own files and
the already validated execution environment; it does not create dummy hashes:

```python
import importlib.metadata
import json
import os
from pathlib import Path
from distillschedule.training.livewan_reference_assets import (
    SCHEMA, BACKEND, DEPENDENCIES, source_identity, frozen_inventory,
    environment_snapshot, asset_config,
)

assets = {
    "schema_version": SCHEMA,
    "sources": {
        "livewan": source_identity("/absolute/project/LiveWan", project="livewan"),
        "wan21": source_identity("/absolute/project/Wan2.1", project="wan21"),
    },
    "models": {
        "student": frozen_inventory("/absolute/project/Wan1.3B"),
        "critic": frozen_inventory("/absolute/project/Wan1.3B"),
        "teacher": frozen_inventory("/absolute/project/Wan14B"),
    },
    "initial_student": {"kind": "base_weights"},
    "critic_lora_seed": 20260831,
    "initial_versions": {"generator": 0, "critic": 0, "ema": 0},
    "optimizer_steps": {"generator": 0, "critic": 0},
    "runtime": {
        "dependency_versions": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "environment": environment_snapshot(os.environ),
        "backend": BACKEND,
    },
}
with Path("/absolute/project/reference-assets.json").open("x") as stream:
    json.dump(asset_config(assets), stream, indent=2)
```

Prepare this inside the intended environment with the same effective launch
variables. The runtime observes rather than applies them. It requires the
original FlashAttention2 path and rejects inherited benchmark/optimization,
profiling and allocator overrides. The metadata API does not import Torch or
open weight payloads; `frozen_inventory` intentionally reads full file bytes
to compute identities. Actual payload/backend admission occurs on workers.
Environment snapshots include only declared nonsecret Hub/Transformers
settings; credential-shaped variables are excluded before reading their values.
Explicit asset documents containing credential keys reject. Do not put tokens
or other authentication material in an asset manifest.

For Stage1, replace `initial_student` with `kind: stage1_state`, an absolute
`path`, actual `bytes` and `sha256`. Only a complete plain tensor state or its
single `model` wrapper is supported. It is restricted-loaded and key/shape/
dtype/finite-checked before the original legacy reader runs. Do not supply
an arbitrary pickle or interpret Stage1 weights as optimizer/RNG restart state.

## Explicit capacity policy

Supply a JSON object with schema
`distillschedule.livewan_reference_capacity.v1`, `gpu_metric` equal to
`peak_allocated_bytes`, `memory_limits` for string rank IDs `0`..`15`,
`host_limits` for nodes `0` and `1`, and explicit nonnegative
`gpu_reserve_bytes` / `host_reserve_bytes`. Values are bytes under the existing
common capacity validator. Choose validated allocation budgets and reserves
for your resources; this recipe intentionally supplies **no universal capacity
default**. It does not assume that the reference fits a 40GB GPU.

Before payload/model construction, every local GPU must have the explicit
allocation budget plus reserve available. Original host/cgroup observation,
allocator peak and driver-free reserves are checked again at complete cycles.
OOM counter changes fail. A CPU test is not physical capacity qualification.
The worker binds its actual CUDA-device UUID to the local physical inventory,
then checks the same binding again before construction and at capacity checks.
This first route requires physical index `rank % 8`; reordered, duplicated,
unresolved or drifting device assignments reject rather than selecting another
rank's budget. Visibility text alone is not proof of device ownership.

## Train and resume

Copy [training.reference.json](training.reference.json), replace its explicit
paths, and run discovery/admission before requesting any GPU work:

```bash
distillctl doctor --config /absolute/project/training.reference.json
distillctl train /absolute/project/training.reference.json --dry-run
```

Actual execution enters `distillctl train` on **each rank** in an
operator-owned two-node/eight-rank-per-node world16 torchrun allocation. It
creates and owns fresh NCCL and Gloo groups, never borrows an embedding group,
reserves nodes, changes visibility or invents rendezvous settings. The clean
committed framework source, input and asset identities must agree across ranks.

`checkpoint_every` saves only after all five input rows and actual G1/C5
commits complete. The true FSDP/module/optimizer owners are quiesced and
resharded at this boundary; Parameter objects are not replaced. Named Adam
moments, trainables, actual versions, per-rank RNG and data cursor go through
the common atomic all-rank checkpoint writer. Frozen Teacher/base weights are
reconstructed from pinned assets; temporary streamer/KV caches are rebuilt.
The original rollout detaches Generator text projection. Its materialized Adam
slots therefore remain at step zero with zero moments; other Generator slots
must be at step k and Critic slots at step 5k. Admission checks named slot
ownership, not an assumption that every Generator tensor received a gradient.

To resume an interrupted incomplete target, use the **same** configuration,
source, assets, seed, finite input horizon and topology, add
`trainer.resume_from` pointing to a completed checkpoint, and select a **new**
output directory. Target extension, finished-target resume and partial/missing
rank shards reject. Do not point at sampled trajectory sketches or the final
weights of a different historical experiment.

Receipts retain actual losses/norms, five successful input operations, actual
G/C/EMA versions, RNG identity, source/runtime origins and capacity observations.
They are not a formal performance or cross-reference numerical qualification.
The separate [full C51 continuation](C51_CONTINUATION.md) implements its
physical drain, shadow and two-slot recovery. Its
[native output bridge](../../docs/guides/livewan_native_output.md) exports the
current Student and generates on new inputs. These are explicit adapters,
not optimizations silently enabled by this reference route; their native GPU
qualification is recorded separately.
