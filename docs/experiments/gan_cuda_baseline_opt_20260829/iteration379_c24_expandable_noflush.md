# Iteration 379: C24 expandable allocator with persistence no-flush

## Contract

C24 is the one bounded recovery allowed after C23.  It keeps the exact C23
workload and no-flush persistence lifecycle, but starts every worker with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  It does not change a
tensor, model call, collective, FSDP reshard, loss, update, trace, RNG or timer
boundary.

## Results

| Gate | Run | Result |
|---|---|---|
| focused tests | exact remote Torch 2.10 environment | 119 PASS |
| validate-only | `iter379_gan_cuda_c24_expandable_noflush_validate_r1_20260830` | 16/16 PASS |
| correctness/capacity | `iter379_gan_cuda_c24_expandable_noflush_correctness_r1_20260830` | 16/16 PASS |
| unprofiled formal timing | `iter379_gan_cuda_c24_expandable_noflush_timing_r1_20260830` | **127.619803944 s** |

The correctness summary SHA-256 is
`dc7c8aa3fbc7ae33f4b511c5efb32d890722502f877e76a17fc9583eef7df205`.
The formal timing summary SHA-256 is
`b3bbe1a1bd513888eec519a60dea3f6e1f2702e33d0a7de8357c53216bb1f3d4`.
Every rank reports the exact requested allocator policy, zero pre/post
allocator-cache flushes, 41 explicit reshards and zero unsharded FSDP units
after the Generator rollout phase.  The existing semantic gate passes 16/16,
including G201/C1005/EMA801, and neither node records a CUDA or cgroup OOM.

## Decision

Expandable segments resolves the C23 fragmentation failure, but does not
improve the formal denominator.  C24 is `1.805799500 s` (`1.4353%`) slower than
C21 `125.814004444 s`.  It is therefore rejected as a performance promotion.
The safe C21 result remains authoritative, and the no-flush allocator path is
closed: no further allocator repair or repeat timing is authorized without a
materially different mechanism.

