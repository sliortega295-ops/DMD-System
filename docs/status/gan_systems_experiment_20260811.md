# Systems-only GAN Baseline/Pipeline execution report

This is an evidence-backed systems-performance report, not a convergence or
official-checkpoint-1000 claim.  `Not Run` entries are intentionally retained.

## Environment

- Nodes: two BAAI jobs, eight A100-40GB GPUs per node (16 GPUs total).
- Resolution/precision: official 384x640, latent `[B,16,9,48,80]`, BF16.
- Python: 3.10.20.
- Accelerate: 1.12.0.
- DeepSpeed: 0.18.4, ZeRO-1 with CPU optimizer offload.
- CUDA/PyTorch versions: not revalidated in the curated result receipts; the
  job environment name is not treated as version evidence.
- Helios source: `c66e413856fda81ec280f0e8e0dd361da0c85830`.
- Executed DMD-System source identities are frozen in
  `gan_systems_formal_workload_contract_20260811.json`: baseline remote head
  `99aa0a31cd1fd5aebcb1d72bd721ece5fd09b683`, pipeline base head
  `03394bc6f58ef1001c759e993d06a32a3c23af3f`, plus executable SHA-256 values.

## Systems-only GAN bootstrap

- Source checkpoint:
  `/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/runs/iteration124_m0_b4fast_zero1_shared_cw4_x16_checkpoint10/checkpoint-10`.
- Source logical package:
  `/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/logical_world16_step10_cal_a_v3`.
- Source logical Generator BF16 payload SHA-256:
  `417827ee2e26a7bd5787f97aa70bd5cf2b257f02540eca79cf34dffb76e6f858`.
- GAN-augmented Critic model-state SHA-256:
  `870e6ff78789f53bcca64b50b9e698f07c62e516088641e2e51e6bc9a8878679`.
- Shared GAN-head FP32 master SHA-256:
  `19f9fa2dd46ac749f981c011f4447e16471639046531bab0b80e9f69cfe0815a`.
- Optimizer materialization: PASS. Generator/Critic state comes from the source;
  GAN-head FP32 master and zero Adam moments were materialized before timing.
  The corrected package has 16 fixed-width ZeRO-1 shards, including the
  required 27 zero-padding elements in the last Critic shard.
- Immutable bootstrap artifact hash:
  `c528dd62b72267570f42b61c36a0926da85e1e0176930efcf72649ef8da3776e`.
- Logical control window: 1000--1004. This directory is named
  `checkpoint-1000` only to activate official predicates; it is not an official
  step-1000 continuation.

## PR #4 Sequential GAN baseline

- Unit/contract/dry-run gates: PASS during bring-up. Current local no-Torch
  refactor suite: 116 PASS; two Torch-dependent builder tests were not rerun
  locally because the local test environment has no Torch.
- Real world16 Full5: PASS.
- CUDA/cgroup OOM: none in the accepted run.
- Update counts: Generator 1, Critic 5, EMA 1.
- Fairness receipt: PASS.
- Compute inventory: PASS.
- Exact unprofiled timing receipt: PASS.
- Fixed memory policy: Generator and Score frozen blocks 0--39, window 1,
  `HOST_SHARED_IMMUTABLE`, no prefetch; selected Generator activation offload
  threshold 32 MiB.
- Formal Full5 seconds: 656.125120, 596.234347, 658.338259; median 656.125120.

## PR #5 persistent 8G+8Score pipeline

- Unit/contract gates: PASS.
- Real world16 Full5: PASS.
- Topology: global ranks 0--7 persistent Generator, 8--15 persistent Score.
- Bidirectional latent/gradient P2P: PASS.
- Deadlock in unprofiled formal trials: none.
- Batch contract: two waves of 8 compose each logical batch16.
- Update counts: Generator 1 after both step-1000 waves; Critic 5 with one
  GAS=2 commit per logical step; EMA 1.
- Fairness receipt: PASS.
- Compute inventory: PASS.
- Exact unprofiled timing receipt: PASS.
- Maximum observed allocator state in the standalone gate: 33,652,875,264
  bytes allocated and 36,148,609,024 bytes reserved; no new OOM.
- Formal Full5 seconds: 1139.194009, 1043.215013, 1170.196848; median
  1139.194009.

## Fairness

- Actual-consumption stochastic parity: PASS for all 80 logical sample-step
  records, not merely same-seed parity.
- Logical trace internal bundle hash:
  `d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15`.
- DiT exit and complete-forward parity: PASS.
- DMD/GAN Score-call parity: PASS; four Generator-side Score forwards and two
  Critic-side Score forwards per logical sample under the frozen configuration.
- Fairness digest:
  `b7104c0729f44d28963dff90fca6d51b293798d9dc0ba718b1d3af7d6985de66`.
- Compute digest:
  `482dfd6cc0b27747adb0ef835cfea187dd6c7769d847e443853b46ab06d2af34`.

## Formal timing

| Pair | Launch order | Sequential (s) | Pipeline (s) | Sequential/Pipeline |
|---|---|---:|---:|---:|
| 0 | Sequential -> Pipeline | 656.125120 | 1139.194009 | 0.575956x |
| 1 | Pipeline -> Sequential | 596.234347 | 1043.215013 | 0.571535x |
| 2 | Sequential -> Pipeline | 658.338259 | 1170.196848 | 0.562588x |

- Median paired speedup: **0.571535x**.
- Geometric-mean paired speedup: **0.569999x**.
- Speedup CV: 1.195%; Sequential CV: 5.532%; Pipeline CV: 5.924%.
- Interpretation: at the same conservative fixed memory policy, this pipeline
  is a clear negative result: approximately 1.75x slower (+74.97% time).
- Mean throughput over 80 logical samples: Sequential 0.125609 samples/s;
  Pipeline 0.071586 samples/s.

## Nsight Sequential

- Exact Full5 range receipts: PASS for all 16 ranks.
- Node reports: two non-empty `.nsys-rep` files.
- Node-0 report SHA-256:
  `47d5efd0601eea34643f85e019cf155e61f9eff20ea93646649a2b2652cfaa5a`.
- Node-1 report SHA-256:
  `13f357526229f1c3972fe7af0249818804b92e6f14f4f48b50680b2235d8bda3`.
- Diagnostic indication on node 1: summed overlapping process-level CUDA API
  time is dominated by `cudaMemcpyAsync` (51.3%, 2808.654 s, 904,065 calls)
  and `cudaStreamSynchronize` (36.8%, 2016.051 s, 637,526 calls). These summed
  process times are not wall-clock fractions and are excluded from speedup.

## Nsight Pipeline

- Exact valid report: FAIL / no report.
- Fourteen non-controller ranks completed the logical window in
  1256.233--1256.234 s, but ranks 0 and 8 blocked in the already documented
  node-local `cudaProfilerStop` distributed-teardown failure.
- No `.nsys-rep` finalized, so no overlap or bottleneck claim is made from this
  attempt. The same mechanism will not be retried.

## Minimum-offload search

- `k*_baseline`: **Not Run**.
- Baseline capacity boundary (`k*-1` FAIL, `k*` PASS, fresh `k*` PASS):
  **Not Run**.
- `k*_pipeline`: **Not Run**.
- Pipeline capacity boundary: **Not Run**.
- Pipeline Level-C timing at measured `k*_pipeline`: **Not Run**.

## Current blocker

During bounded cleanup of the failed Pipeline Nsight attempt, an operator-side
SSH quoting error expanded a process-match variable locally and terminated the
two job containers more broadly than intended. Both old SSH endpoints now time
out. This is an operator incident, not a workload, capacity, or infrastructure
result. The completed formal timing evidence remains valid and immutable.

Recovery is limited and explicit: restart/recreate both GPU jobs, update SSH
aliases if IDs change, revalidate source/assets/GPU ownership/cgroup state, and
run only the already prepared Sequential and Pipeline actual-OOM-only capacity
searches. The failed Pipeline profiler mechanism must not be retried.

## 2026-08-12 role/lifecycle follow-up

The jobs were later recreated and the bounded Pipeline lifecycle sequence was
continued.  `G40/S10/A128` passed the full real world16 capacity/correctness
gate.  A subsequent low-overhead launch completed all 16 model workers and
produced a measured 824.685006 s exact Full5 window, but its aggregate receipt
postprocessor failed and only one launch was run.  The value is therefore
preliminary and not formal-speedup eligible.  GPU work is now stopped for an
implementation correction.  See
[`gan_pipeline_rolephase_20260812.md`](gan_pipeline_rolephase_20260812.md) for
the complete handoff and claim boundary.
