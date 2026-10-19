# R2 Generator optimizer capacity fallback v1

This opt-in candidate is `only_A_with_capacity_offload`, not the immutable
`only_A` arm and not a complete orthogonal A/B factorial cell. All canonical
plans, default placement guards and previous OOM evidence remain unchanged.

It inherits the complete R2 only_A 8G+8Score layout, depth-1 four-cycle pipeline,
Critic shadow versions and original recomputation paths. All B caches/fusions
remain disabled. The sole capacity change uses the existing `CPUAdamW` for the
Generator optimizer: original FP32 GPU execution parameters, gradient mean and
clipping remain unchanged; CPU owns same-dtype parameter mirrors, gradient
buffers and Adam moments. Critic optimizer, Teacher and activation placement
remain unchanged. The global baseline optimizer flag remains `cuda`; this
source-bound candidate carries its independent role-specific capacity policy.

The existing optimizer executes gradient D2H, host-consumption synchronization,
ordinary CPU foreach AdamW, and updated-parameter H2D inside its `step()`. That
call remains inside the original update and original complete-cycle endpoints.
No CPU work/transfer is moved outside timing. CPU/CUDA floating-point kernels
can differ: native finite/census/version checks do not establish numerical
equivalence. Report the original REVIEW boundary, never strict PASS by fiat.

Before launch, audit the eight Generator processes together. The frozen model
has 1,418,996,800 FP32 parameters: each rank needs one CPU parameter mirror, one
gradient buffer and two moments (16 bytes/parameter persistently), with the two
transfer buffers pinned. Additional foreach temporary, model-loading/mmap,
runtime, transport and existing cgroup memory must be budgeted. A conservative
40-byte/parameter envelope plus a separately recorded host reserve is used for
this one candidate. Host capacity is not inferred from CUDA allocated memory.

After the final endpoint, actual optimizer receipts must prove four Generator
commits, CPU moments/mirror/buffers, CUDA execution parameters and paired
nonzero D2H/H2D bytes/events. Score ranks must show no CPU optimizer. Full
original call/shape/sample/version/finite/source and cleanup gates still apply.
CPU synthetic fixture timing remains unclaimable.

Authorization permits one native allocation, C1 warm-up followed by C2/C3/C4,
n=1. Failed old only_A remains OOM. The new result is labelled "layout and
scheduling + necessary optimizer offload". Its difference from full includes
both offload cost and state/operator mechanisms; it is not pure B benefit.
