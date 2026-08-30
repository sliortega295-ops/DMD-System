# Iteration 378: C23 persistence allocator no-flush

## Contract

C23 changes only the two `torch.cuda.empty_cache()` calls surrounding C21's
bounded Generator no-grad persistent-unshard scope.  All 41 FSDP2 units are
still materialized for the same five B1 rollouts, explicitly resharded, and
restored before Critic work.  The flag is opt-in; the default remains C21's
conservative pre/post flush behavior.

## Results

| Gate | Run | Result |
|---|---|---|
| focused tests | exact remote Torch 2.10 environment | 119 PASS |
| validate-only | `iter378_gan_cuda_c23_persist_noflush_validate_r1_20260830` | 16/16 PASS |
| correctness/capacity | `iter378_gan_cuda_c23_persist_noflush_correctness_r1_20260830` | 16/16 PASS |
| formal timing attempt | `iter378_gan_cuda_c23_persist_noflush_timing_r1_20260830` | **CUDA OOM; no timing result** |

The correctness run proves the intended semantics: G201/C1005/EMA801,
`pre_empty_cache_count=0`, `post_empty_cache_count=0`, 41 fully materialized
units before exit, zero unsharded units after exit, and all 41 explicit
reshards. Its maximum allocated HBM is about `36.403 GiB` and maximum reserved
HBM is `38.590 GiB`.

The formal timing arm is nevertheless a capacity failure. Ranks 3, 4, 11 and
12 report the same failure while allocating a `292 MiB` LoRA linear output:

- `36.43 GiB` allocated by PyTorch;
- `1.91 GiB` reserved but unallocated;
- only about `196--202 MiB` driver-free;
- no host/cgroup OOM.

The large reserved-but-unallocated pool plus a smaller failed allocation is
direct evidence of an allocator fragmentation/layout failure, not a workload
semantic failure. The run was stopped after preserving all four OOM receipts;
no profiled or partial latency is promoted.

## Decision

C23 is rejected and the safe formal denominator remains C21
`125.814004444 s`.  The only bounded recovery authorized next is C24: retain
the same no-flush code path but enable PyTorch's source-supported
`expandable_segments:True` allocator policy from process start. This tests the
observed fragmentation mechanism without changing tensors, calls, losses,
collectives, FSDP lifecycle, or timing boundaries. Any OOM rejects it and
returns immediately to C21.
