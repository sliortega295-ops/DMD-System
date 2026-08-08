# Route-A typed world16 status

**Status date:** 2026-08-08  
**Live tracker:** [Issue #2](https://github.com/sliortega295-ops/DMD-System/issues/2)  
**Active integration branch:** `codex/routea-world16-repository-hardening`
**Evidence state:** source hardening updated; new cluster acceptance still Not Run

This page records accepted evidence, not intent. The frozen workload is defined separately in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md).

## Attempt ledger

| Attempt | Final evidence label | What happened | Claim allowed |
|---|---|---|---|
| 001 | Failed before model construction | The typed wrapper invoked Accelerate without exposing the matching environment's `bin` directory; Accelerate could not resolve `deepspeed`. No training step ran and OOM deltas were zero. | Launcher failure only |
| 002 | Failed at checkpoint world-size gate | Each host formed an independent world8 group. The world16 ZeRO-1 checkpoint correctly rejected the topology before a training step. | Topology failure only |
| 003 | **Legacy PARTIAL** (historically `Not Run` at the Issue #2 snapshot) | The job later exited with 16 rank traces and checkpoint-15, but its mixed legacy launcher manifest and missing in-run Attempt-004 topology/evidence contract cannot be retroactively repaired. The hardened replay measured a diagnostic 646.7306-s window, but it is not claimable. | Functional candidate only; no accepted timing claim |
| 004 | **Not Run** | The hardened coordinator, node contracts and summarizer now exist in source, but no immutable two-node result has passed them yet. | None |

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

These are **source-complete, cluster-validation pending** until the updated smoke test and a real two-node run execute on the target environment.

## Still open

Issue #2 remains open because the new source has not passed a real Attempt 004 on the
cluster. The compatibility adapter still invokes the pinned external
`train_helios.py`. The No-GAN suite's target G/S/T candidate executor, live
Step10-A→B redistribution, continuous Route-B steps10–19 and matched formal launch
commands are still explicit external seams; their results remain `Not Run`.

A successful topology probe will close the specific “two independent world8” failure mode, but it will not by itself close checkpoint continuation, Full5 semantics, EMA parity or timing consistency.

## Status vocabulary

- `PASS`: the exact functional/evidence contract passed; one-run timing remains diagnostic unless a separate formal driver passed.
- `FAIL`: complete Attempt 004-shaped evidence exists but an exact contract check failed.
- `CRASH`: workers completed evidence collection with a nonzero process return.
- `OOM`: cgroup OOM or OOM-kill counters increased.
- `PARTIAL`: required node/rank/checkpoint evidence is missing.
- `DRY_RUN`: commands were rendered only and may never satisfy a later real dependency.
- `NOT RUN`: no accepted final execution evidence exists.
