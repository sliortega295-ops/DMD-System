# Route-A typed world16 status

**Status date:** 2026-08-08  
**Live tracker:** [Issue #2](https://github.com/sliortega295-ops/DMD-System/issues/2)  
**Adaptation branch at feedback time:** `server-adaptation-20260808`  
**Recorded HEAD:** `cc8fae3855a34f95022b9cbbe8f2a6cae8eefba7`

This page records accepted evidence, not intent. The frozen workload is defined separately in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md).

## Attempt ledger

| Attempt | Final evidence label | What happened | Claim allowed |
|---|---|---|---|
| 001 | Failed before model construction | The typed wrapper invoked Accelerate without exposing the matching environment's `bin` directory; Accelerate could not resolve `deepspeed`. No training step ran and OOM deltas were zero. | Launcher failure only |
| 002 | Failed at checkpoint world-size gate | Each host formed an independent world8 group. The world16 ZeRO-1 checkpoint correctly rejected the topology before a training step. | Topology failure only |
| 003 | **Not Run** at the Issue #2 snapshot | The process had reached real world16 checkpoint resume when the issue was written, but no final result, rank-complete evidence or summarizer output had been accepted. GitHub bookkeeping did not interrupt the run. | None |

`Running` is not a result. Unless a later immutable manifest is linked from Issue #2 and this page is updated, Attempt 003 remains `Not Run` for repository claims.

## What the cleanup establishes in source

The repository now has code and documentation for:

- a single frozen contract document and a canonical BAAI environment profile;
- explicit ZeRO-1/CPU-optimizer/no-parameter-offload validation;
- node-qualified Accelerate config rendering without a checked-in absolute DeepSpeed path;
- Python/Accelerate/DeepSpeed and Helios source identity recording;
- a pre-model 2×8/world16 topology probe with a tiny NCCL collective;
- fail-closed JSONL parsing, exact rank-set checks and checkpoint inventory in the summarizer; and
- separate functional, timing-band and host-reserve statuses.

These are **source-complete, cluster-validation pending** until the updated smoke test and a real two-node run execute on the target environment.

## Still open

Issue #2 remains open because the repository still lacks a central coordinator that owns both host workers, propagates peer failure/cancellation and performs one authoritative finalization. The compatibility adapter also still invokes the pinned external `train_helios.py`; normalized new-versus-legacy config parity must be extended beyond the current frozen fields before broader workloads are claimed.

A successful topology probe will close the specific “two independent world8” failure mode, but it will not by itself close checkpoint continuation, Full5 semantics, EMA parity or timing consistency.

## Status vocabulary

- `PASS`: functional contract passed and, when a matched baseline was supplied, timing remained inside the preregistered band.
- `FUNCTIONAL_PASS_TIMING_NOT_EVALUATED`: functional contract passed but no valid baseline comparison was available.
- `FUNCTIONAL_PASS_TIMING_OUTSIDE_BAND`: functional contract passed but the diagnostic timing hypothesis failed.
- `CONTRACT_FAIL`: complete evidence exists but topology/steps/versions/checkpoint/lifecycle do not match.
- `CRASH`: workers completed evidence collection with a nonzero process return.
- `OOM`: cgroup OOM or OOM-kill counters increased.
- `PARTIAL`: required node/rank/checkpoint evidence is missing.
- `EVIDENCE_INVALID`: present evidence is malformed or internally inconsistent.
- `NOT RUN`: no accepted final execution evidence exists.
