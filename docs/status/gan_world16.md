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
- Operation-trace bundle hash: `d31d4df15ccdf4790ee6239e0354f9d051cff2c9d2e55a83c771af4fa271c170`
- Shape/backend: official 384x640, BF16, world16 ZeRO-1 with CPU optimizer.

## Bounded bring-up attempts

| Run | Result | First failing boundary | Capacity failure |
|---|---|---|---|
| `...004` | FAIL | rank-15 Critic Adam state lacked 27 ZeRO padding elements | No |
| `...005` | FAIL | semantic observer began after official GAN low-VRAM backward/commit | No |
| `...006` | FAIL | step1002 Generator patch embedding remained on CPU after skipped EMA | No |
| `...007` | **PASS** | none; all steps 1000--1004 and checkpoint-1005 complete | No |

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

## Current gate

Sequential world16 functional execution: **PASS**.

Still **Not Run**: baseline actual-consumption fairness receipt, compute
inventory receipt, low-overhead formal baseline timing, real persistent
8G+8Score pipeline, paired trials, Nsight, and minimum-offload search.
