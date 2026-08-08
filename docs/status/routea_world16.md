# Route-A typed world16 status

**Status date:** 2026-08-08  
**Live tracker:** [Issue #2](https://github.com/sliortega295-ops/DMD-System/issues/2)  
**Active integration branch:** `codex/routea-world16-repository-hardening`
**Evidence state:** Attempt 005-002 functional/evidence PASS; formal repeated timing Not Run

This page records accepted evidence, not intent. The frozen workload is defined separately in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md).

## Attempt ledger

| Attempt | Final evidence label | What happened | Claim allowed |
|---|---|---|---|
| 001 | Failed before model construction | The typed wrapper invoked Accelerate without exposing the matching environment's `bin` directory; Accelerate could not resolve `deepspeed`. No training step ran and OOM deltas were zero. | Launcher failure only |
| 002 | Failed at checkpoint world-size gate | Each host formed an independent world8 group. The world16 ZeRO-1 checkpoint correctly rejected the topology before a training step. | Topology failure only |
| 003 | **Legacy PARTIAL** (historically `Not Run` at the Issue #2 snapshot) | The job later exited with 16 rank traces and checkpoint-15, but its mixed legacy launcher manifest and missing in-run Attempt-004 topology/evidence contract cannot be retroactively repaired. The hardened replay measured a diagnostic 646.7306-s window, but it is not claimable. | Functional candidate only; no accepted timing claim |
| 004 | **Not Run** | Dual-host N0 source/config smoke passed at commit `a625c09` (46 tests per host). Four immutable N1 launch IDs were preserved, but all stopped before model construction: helper execute bits, runtime-version probing, pinned-Accelerate config validation, then a 19-minute LeoFS permission/lookup stall. The last attempt never acquired the GPU locks and both worker transports ended 255; no training or OOM evidence was produced. | N0/pre-model infrastructure evidence only |
| 005-001 | **Not Run** | After LeoFS recovery, both fresh N0 smoke gates passed and the live 2x8 topology probe itself passed Gloo plus NCCL with ranks 0-15. The launcher then rejected the valid payload because it expected obsolete top-level summary keys instead of the emitted nested summary/rank-record schema. Both workers exited 1 before model construction, with no new OOM. | Topology success and pre-model consumer bug only |
| 005-002 | **PASS** | At raw execution commit `d4a27a3`, both workers returned 0 and all 16 ranks completed official 384×640 steps 10–14 with G2/C10/EMA10 → G3/C15/EMA15, one Generator update, five Critic updates, five EMA updates, zero lifecycle residue, checkpoint-15 and no OOM increment. A postprocess-only schema fix at `2ea09d4` read the producer's nested sparse-EMA fields, structured EMA predicate, trace-kind marker, spaced NVIDIA CSV, and scientific-notation resource rows; the preserved raw phase events were not changed. | One-run functional acceptance and diagnostic timing only; no formal speedup claim |

`Running` is not a result. Attempt 003's later files are preserved as legacy partial
evidence; they do not become an Attempt-004 PASS merely because both worker return
codes were zero.

## What the current branch establishes in source

The repository now has code and documentation for:

- a single frozen contract document and a canonical BAAI environment profile;
- explicit ZeRO-1/CPU-optimizer/no-parameter-offload validation;
- node-qualified Accelerate config rendering without a checked-in absolute DeepSpeed path;
- Python/Accelerate/DeepSpeed and Helios source identity recording;
- a pre-model 2×8/world16 topology probe with a tiny NCCL collective;
- fail-closed JSONL parsing, exact rank/trace/version checks and checkpoint inventory;
- actual-OOM-only memory acceptance with cgroup/host/CUDA observations retained;
- a central two-host coordinator with peer cancellation and strict finalization;
- a resumable No-GAN suite DAG, target-graph Generator offload search and paired timing driver.

These components are now cluster-validated for the frozen Route-A compatibility path by Attempt 005-002. This does not validate the still-external Route-B executors.

The N0 fixes are initialization-only and do not change the Full5 workload: node-local
Python bytecode caches avoid shared-filesystem races, GAS=1 remains explicit in the
launcher/Helios contracts, runtime versions come from package metadata, and the
pinned Accelerate release now parses the generated YAML during smoke. The LeoFS
failure signatures were `__leofs_get_access_perm` and `leofs_lookup_v1`; GPU memory
remained zero and the node locks remained available. A new N1 attempt is permitted
only after those uninterruptible waiters have exited and `/share/project` access is
healthy again.

A bounded live recovery audit at `2026-08-08T12:40:22Z` additionally found four
LeoFS D-state tasks and 444 zombies on node 0, and two LeoFS D-state tasks and 300
zombies on node 1. The Route-A tasks have pending `SIGKILL`, but an uninterruptible
kernel lookup cannot be reaped until LeoFS returns. Neither container has
`CAP_SYS_ADMIN`, and its Kubernetes service account receives HTTP 403 for pod
inspection, so a client remount or safe job restart cannot be performed from inside
the allocation. The exact audit is preserved in
`artifacts/manifests/routea_attempt004_leofs_recovery_20260808.json`. This remains an
external infrastructure block before model construction, not OOM and not Full5
evidence.

## Still open

Issue #2 remains open because the compatibility adapter still invokes the pinned external
`train_helios.py`, and the one-run diagnostic timing missed its consistency band.
The No-GAN suite's target G/S/T candidate executor, live
Step10-A→B redistribution, continuous Route-B steps10–19 and matched formal launch
commands are still explicit external seams; their results remain `Not Run`.

## Attempt 005-002 measured result

- Global steps-10–14 window: **767.166962 s**; weighted **153.433392 s/step**.
- Frozen reference: 688.725684 s; delta **+11.389%** (slower), outside the
  preregistered ±10% diagnostic band.
- Throughput: **0.104280 samples/s** for 80 logical samples in the five-step window.
- GPU peak: 33,804,603,904 allocated bytes and 36,335,255,552 reserved bytes.
- Node cgroup peaks: 798,765,285,376 bytes (node 0) and 782,242,111,488 bytes
  (node 1), with zero OOM/OOM-kill increments under the actual-OOM-only policy.
- Checkpoint-15: 97 files, 99,826,243,285 bytes.

The authoritative curated files are
`artifacts/manifests/routea_attempt005_002_result.json` and
`artifacts/manifests/routea_attempt005_002_acceptance.json`. Formal repeated timing
is still `Not Run`.

A successful topology probe will close the specific “two independent world8” failure mode, but it will not by itself close checkpoint continuation, Full5 semantics, EMA parity or timing consistency.

## Status vocabulary

- `PASS`: the exact functional/evidence contract passed; one-run timing remains diagnostic unless a separate formal driver passed.
- `FAIL`: complete Attempt 004-shaped evidence exists but an exact contract check failed.
- `CRASH`: workers completed evidence collection with a nonzero process return.
- `OOM`: cgroup OOM or OOM-kill counters increased.
- `PARTIAL`: required node/rank/checkpoint evidence is missing.
- `DRY_RUN`: commands were rendered only and may never satisfy a later real dependency.
- `NOT RUN`: no accepted final execution evidence exists.
