# Bounded H100 Helios no-grad Generator retention experiment

The current No-GAN H100 winner is 98.213873 s versus the official-model-swap
GPU-ZeRO-1 baseline 138.371948 s (single diagnostic pair). The 1.5x target
would require 92.247966 s. This is a hypothesis, not a promised saving.

`h100-nograd-retain` changes only the existing
`generator_nograd_persistent_unshard` flag in a separately registered plan.
It preserves the current 2x4 replicate/shard mesh, B1 physical rows/global16,
trace, inputs, dtype, losses, five rollout batches, G1/C5/EMA5 and all kernels.
No explicit first-block prefetch or new batching is introduced. The existing
public-API scope restores all41 FSDP units before leaving no-grad production.

The old policy rejected all HSDP retention. A real8-rank H100 primitive now
demonstrates the existing public API on replicate2/shard4: 205 versus41 actual
all-gather submissions for five tiny forwards, bitwise outputs and post-scope
gradient/update parity, and all41 units restored. Receipt SHA256:
`52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966`.
This proves the library lifecycle only, **not14B capacity or speedup**.

The narrow policy extension accepts explicit H100 2x4 B1 only; default A100
HSDP and B2/B5 remain rejected. Hardware admission is an InitVar, not a new
serialized execution flag. H100 source/asset/topology preflight still applies.
The real launcher must supply
`DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT` and its SHA, rank records,
mesh, restore and observed all-gather counts are validated before model load.

At source commit: static/stdlib checks pass; remote focused pytest pending.
Real14B validate-only/correctness/timing: **NOT_RUN**. After CPU tests, admit
one validate-only, one correctness, and only on PASS one unprofiled timing.
Keep the prior winner and all failed runs. Numerical REVIEW is not strict PASS.
