# GAN world16 execution status

## Scope

This status tracks the systems-only GAN experiment on two 8xA100-40GB nodes.
The immutable bootstrap is **not** an official checkpoint-1000: parameters and
optimizer state originate from the validated earlier Helios state, while logical
control executes GAN-active steps 1000--1004.

## Frozen inputs

- Bootstrap: `systems_only_gan_bootstrap_v2/checkpoint-1000`
- Artifact hash: `c528dd62b72267570f42b61c36a0926da85e1e0176930efcf72649ef8da3776e`
- GAN-aware Score pre-cast hash: `d31e8f364501c925e4b831e83d9ef3b23d2920aaf90852eb839e10f764d91709`
- Operation-trace artifact hash: `d31d4df15ccdf4790ee6239e0354f9d051cff2c9d2e55a83c771af4fa271c170`
- Operation-trace internal bundle hash used by execution receipts:
  `d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15`
- Shape/backend: official 384x640, BF16, world16 ZeRO-1 with CPU optimizer.

## Bounded bring-up attempts

| Run | Result | First failing boundary | Capacity failure |
|---|---|---|---|
| `...004` | FAIL | rank-15 Critic Adam state lacked 27 ZeRO padding elements | No |
| `...005` | FAIL | semantic observer began after official GAN low-VRAM backward/commit | No |
| `...006` | FAIL | step1002 Generator patch embedding remained on CPU after skipped EMA | No |
| `...007` | **PASS** | none; all steps 1000--1004 and checkpoint-1005 complete | No |
| `...008` | **PASS** | none; complete actual-consumption and physical-call receipts | No |

Run007 observed exactly one Generator update, five Critic updates and one EMA
update on every rank. Final observed versions were G201/C1005/EMA801. Both
activation and frozen-block lifecycle records ended at zero. The sampled node
cgroup peaks were 596,042,338,304 and 607,768,354,816 bytes; no new OOM or
OOM-kill event occurred. The detailed functional run's global five-step window
was 596.418 s, but this is diagnostic only because high-overhead correctness
instrumentation was enabled.

The initial run007 summarizer rejection was an evidence-parser bug: the runtime
uses categorical authority `operation_trace`, stores EMA version under the
nested EMA receipt, emits `optimization_step_before=800`, and emits final
component versions on their update receipts rather than the completion marker.
The corrected strict parser consumes those observed receipts and returns PASS;
it does not derive versions from the schedule or change training math.

## Functional and fairness gates

Sequential world16 functional execution: **PASS**.  Run008 additionally closes
the actual-consumption fairness and compute-inventory gate: 16 rank-qualified
access logs contain 2,432 observed tensor lookups; the normalized fairness and
compute receipts each contain all 80 logical sample-step entries.  Their
contract digests are respectively
`b7104c0729f44d28963dff90fca6d51b293798d9dc0ba718b1d3af7d6985de66`
and `482dfd6cc0b27747adb0ef835cfea187dd6c7769d847e443853b46ab06d2af34`.
Independent block events reconcile 31 logical Generator-backbone forwards and
14 logical Score-backbone forwards per rank, with checkpoint recomputation
reported separately (3 and 11 respectively) and balanced releases.

Run008's exact-range receipt is structurally valid (609.593 s), but it remains
diagnostic because the run enabled correctness and detailed lifecycle
instrumentation.  It is not the low-overhead formal denominator.

The persistent 8G+8Score implementation also passed its standalone real-14B
gate (`gan_pipeline_formal_gate_20260811_003`).  It completed the same five
logical steps with two waves per logical batch, exactly one Generator commit,
five Critic commits, one EMA commit, balanced lifecycle handles, and no new
CUDA/cgroup OOM.  Its normalized fairness and compute digests match the
sequential denominator.  Maximum observed allocator values were
33,652,875,264 allocated and 36,148,609,024 reserved bytes.

## Fixed-policy paired formal timing

The paired experiment keeps the same all-40/window-1/shared-immutable/no-
prefetch memory policy on both implementations.  All six runs restarted from
the same systems-only bootstrap and passed fairness, compute-inventory, and
exact unprofiled timing gates.

| Pair | Order | Sequential (s) | Pipeline (s) | Sequential/Pipeline |
|---|---|---:|---:|---:|
| 0 | Sequential then Pipeline | 656.125120 | 1139.194009 | 0.575956x |
| 1 | Pipeline then Sequential | 596.234347 | 1043.215013 | 0.571535x |
| 2 | Sequential then Pipeline | 658.338259 | 1170.196848 | 0.562588x |

Sequential median is 656.125120 s and Pipeline median is 1139.194009 s.
Median paired speedup is **0.571535x**; geometric-mean paired speedup is
**0.569999x**.  Thus the frozen persistent 8G+8Score candidate is a measured
negative result: it takes about **1.75x** as long as Sequential (+74.97%) at
the same memory policy.  Speedup CV is 1.195%; Sequential and Pipeline timing
CVs are 5.532% and 5.924%, respectively.  The authoritative artifact is
`artifacts/formal/gan_systems_paired_formal_20260811_002/result.json`.

## Nsight diagnostics

The Sequential diagnostic capture passed its exact Full5 range contract and
produced two non-empty node reports.  On node 1, summed process-level CUDA API
time was dominated by `cudaMemcpyAsync` (51.3%, 2808.654 s across 904,065
calls) and `cudaStreamSynchronize` (36.8%, 2016.051 s across 637,526 calls).
These are overlapping eight-process sums, not wall-clock fractions.  The
reports remain diagnostic and are excluded from the formal timing table.

The Pipeline diagnostic run completed the model window on 14 non-controller
ranks (1256.233--1256.234 s receipts), but ranks 0 and 8 blocked in the known
node-local `cudaProfilerStop` teardown failure.  No `.nsys-rep` finalized, so
Pipeline Nsight is **FAIL**, not a partial profile.  This failure matches the
preserved bounded negative result from Iteration 255 and must not be retried
through the same mechanism.  See
`artifacts/manifests/gan_pipeline_nsys_20260811_001_failure.json`.

## Current blocker and remaining work

During bounded cleanup of the failed profiler run, an operator-side SSH
quoting error expanded a remote process-match variable locally and terminated
the two job containers more broadly than intended.  Both SSH endpoints became
unavailable.  This is not a workload or capacity result.  The incident is
recorded in `artifacts/manifests/gan_remote_cleanup_incident_20260811_001.json`.

Minimum-offload searches are therefore **Not Run**:

- Sequential `k*_baseline`: Not Run;
- Pipeline `k*_pipeline`: Not Run;
- Pipeline Level-C timing at measured `k*_pipeline`: Not Run.

Recovery requires restarting or recreating both BAAI jobs, revalidating the
live substrate, then running the already-prepared actual-OOM-only searches.
The formal fixed-policy negative result above remains valid and immutable.
