# Iteration 364: C11 post-fix GAN-on profile

## Identity and claim boundary

- Run: `iter364_gan_cuda_c11_profile_p1_20260830`.
- Workload HEAD: `f3c5ed480001fb0007914de25264edd8439e9f50`.
- Formal denominator remains the unprofiled C11 result
  `138.462140212 s`; the profiled controller envelope
  `217.229654622 s` is diagnostic-only.
- Workload: GAN-on steps 1000--1004, world16 FSDP2, B1/global-B16,
  Score packing off, CUDA Generator/Critic residuals and EMA, local FA2,
  chunk2048, Critic selective Flash replay stride2, Generator no-grad
  cross-KV cache, and exact modulated-norm backward.
- Correctness inventory: 16/16 ranks complete with G1/C5/EMA1 and final
  G201/C1005/EMA801. Both node reports and SQLite exports are non-empty and
  the exact range receipt gate passes for all 16 ranks.

Raw `.nsys-rep` and SQLite files remain in remote persistent storage under:

`/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/artifacts/runs/iter364_gan_cuda_c11_profile_p1_20260830/`

Only compact analysis is recorded here. Raw artifact identities are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| node0 `.nsys-rep` | 270,517,766 | `06c19fbc4fd4286205f620b3f319c7971055dd4c061f780640d1e55525f8fb3d` |
| node0 SQLite | 636,788,736 | `cdda8497cd6a0daa541a4515f7249fc628e8be5d752f158dc338ae88053c6c6a` |
| node1 `.nsys-rep` | 269,616,470 | `d83b91f70933602d6adb6f0b2a8c2d1583e3e304cc25879f3f97f84c13190107` |
| node1 SQLite | 636,829,696 | `7c13168785e85aa5e64b57052fc5ecae8c084e6eab62baf658dc80fd20ecb1df` |
| all-rank attribution | 546,432 | `cfcfb24ffbfd82ea6e25ce1456101dbb0c6752089e3beda0f58460c4fe29ac8f` |
| remote summary | 223,333 | `451ccf43f004d16e4e3fd89d92ac8f33de93d31e2179bfba901e8df6cd71934b` |

## Profile health

Remote `nsys-ai 0.3.0 doctor` passes both node SQLite files. Node0 records
8 GPUs, 158,166 NVTX events, 23,017 NCCL kernels, schema 3.4.0, and estimated
profiler overhead 1.7%; node1 records the same eight-device coverage with
158,170 NVTX events, 23,017 NCCL kernels, and 2.0% overhead. The profiler did
not export a GPU model, so this report intentionally makes no MFU claim.

The required universal evidence step was completed on node0 GPU0. Its remote
timeline is at `http://127.0.0.1:18164` (SSH port forwarding is required from a
workstation).

## Overlap-aware attribution

All values below are medians across 16 rank-local exact Full5 ranges. Compute,
NCCL and copies are interval unions and may overlap; `exposed` means not covered
by non-NCCL compute. They are GPU-seconds per rank, not world wall-clock sums.

| Phase | Wall (s) | Compute union | NCCL union | Compute+NCCL overlap | NCCL exposed | Copy exposed | No traced GPU activity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full5 | 139.448 | 126.112 | 61.524 | 54.422 | 7.121 | 0.501 | 5.903 |
| Generator update | 28.347 | 21.577 | 11.812 | 9.664 | 2.127 | 0.166 | 4.500 |
| G gradient forward | 6.575 | 3.976 | 3.225 | 2.744 | 0.490 | 0.019 | 2.104 |
| Score/DMD/GAN-G | 12.292 | 8.852 | 5.251 | 3.917 | 1.251 | 0.080 | 2.113 |
| G backward | 9.247 | 8.672 | 3.430 | 3.046 | 0.386 | 0.066 | 0.143 |
| G no-grad rollouts (5) | 31.913 | 28.993 | 21.985 | 19.475 | 2.485 | 0.098 | 0.388 |
| Critic phase (5 updates) | 79.110 | 75.574 | 26.987 | 24.504 | 2.483 | 0.232 | 0.899 |

For representative rank0, `nsys-ai critical_path` classifies the range as
GPU-compute-bound with confidence 0.846: 125.549 s compute-on-path,
7.592 s communication-only and 6.293 s complement. This is temporal coverage,
not SM utilization or MFU.

## Compute and communication evidence

The largest rank-local kernel families are repeated 14B block work:

- dominant BF16 FFN GEMM: about 28.4 kernel-seconds/GPU;
- FlashAttention forward: about 14.2 kernel-seconds/GPU;
- second BF16 FFN GEMM: about 13.7 kernel-seconds/GPU;
- FlashAttention backward: about 12.7 kernel-seconds/GPU;
- reduce-scatter: about 7.3 kernel-seconds/GPU;
- BF16 add: about 6.2 kernel-seconds/GPU.

The prior real-workload NCU capture showed the dominant FFN GEMM at 88.32%
SM/Tensor throughput and 25.23% DRAM throughput, so blindly replacing that one
cuBLAS kernel is not a credible route to the remaining target.

The launch-signature-controlled observational comparison shows that major
NCCL overlap changes matched FFN median duration by roughly 0.5--1.0% and
FlashAttention forward by roughly 1--4% for common signatures. This is not a
causal A/B; the unmatched FlashAttention-backward signature is discarded. It
supports only the bounded conclusion that contention is not large enough, by
itself, to explain the 17.35 s target gap.

## New source-qualified hot-path gap

`nsys-ai sync_cost_analysis` reports 10.416 s of stream/event synchronization
wall on rank0 (7.47% of its profiled range). This includes FSDP/framework
synchronization and therefore is not all removable. Source correlation finds
a smaller exact candidate that is unconditional inside the measured P0 path:

- `runtime/gst_pool_helios.py`: one `torch.cuda.empty_cache()` and one
  `torch.cuda.synchronize()` in the Generator-update rollout;
- `runtime/static_fsdp2_p0_helios.py`: one
  `torch.cuda.synchronize()` after each of the five no-grad rollout batches.

Shape, dtype, autograd and physical-call checks following these calls use host
metadata and do not require completion of the default stream. Downstream work
inherits PyTorch stream/event dependencies, and the Full5 timer retains its
existing start/end synchronization. C12 therefore tests only this explicit
synchronization/allocator cleanup:

```python
# before: every rollout blocks the host, and G update flushes the allocator
torch.cuda.empty_cache()
generated_x0 = run_generator(...)
torch.cuda.synchronize(device)

# C12: opt-in low-overhead path; formal boundary synchronization is unchanged
if empty_cache_before_rollout:
    torch.cuda.empty_cache()
generated_x0 = run_generator(...)
if synchronize_after_rollout:
    torch.cuda.synchronize(device)
```

This does not claim all 10.416 s is recoverable. The falsifiable question is
whether allowing the CPU to enqueue the next dependent phase, while preserving
the exact outer boundary and correctness gates, reduces one unprofiled Full5.

## Decision

The profile is **PASS_DIAGNOSTIC_ONLY**. It does not update the formal
`138.462140212 s` result. C12 is authorized as a single default-off mechanism:
omit only six explicit in-window rollout synchronizations and one
Generator-update `empty_cache`; retain outer timer synchronization, workload
math, physical calls, FSDP placement, and every parity/inventory gate.
