# Iteration 374: C19 production FlashAttention NCU diagnostic

## Frozen identity and validity

- Source HEAD: `f4f005435d49228eccd2238166edee1fdbc77366`.
- Run: `iter374_gan_cuda_c18_ncu_flashfwd_p1_20260830`.
- Workload: exact C18 GAN-on world16 Full5, diagnostic-only; every C18 policy,
  trace, model call, update and version receipt remained frozen.
- Target: one dense FA2 forward on node0 GPU0 in
  `FSDP_BLOCK_FORWARD/critic/block_32`.
- Kernel grid/block: `(87,1,40)` / `(128,1,1)`.
- NCU replay passes: `44`; exactly one target kernel was profiled.
- All 16 workload range receipts exist, both nodes completed, and CUDA/cgroup
  OOM deltas are zero. The NCU replay wall is not a formal timing result.

Raw evidence remains remote under:

```text
/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/
  artifacts/runs/iter374_gan_cuda_c18_ncu_flashfwd_p1_20260830/
```

The report SHA-256 is
`62b5765f67956d1a8ef53b8a61a3404733710bb2d0c5e07c6da6f29f6099a832`;
the details CSV SHA-256 is
`98ac435680de372376c271f153b002bc3a6f605a46f60f608136f03e458bcead`.

## Hardware result

| Metric | Value |
|---|---:|
| Kernel duration | `11.581504 ms` |
| Compute/SM throughput | `70.363613%` |
| Tensor HMMA pipeline active, elapsed | `70.363613%` |
| Shared/tensor pipeline active, active cycles | `71.865164%` |
| DRAM throughput | `2.865665%` |
| L1/TEX throughput | `50.726274%` |
| L2 throughput | `24.761111%` |
| L2 hit rate | `97.029243%` |
| Issue slots busy | `40.472322%` |
| No eligible warp | `59.524862%` |
| Achieved occupancy | `12.359914%` |
| Registers/thread | `255` |
| Dynamic shared memory/block | `65.536 KiB` |
| Waves/SM | `16.11` |

The largest warp-stall ratios per issued instruction are wait (`1.756819`),
math-pipe throttle (`1.031011`), not-selected (`0.323526`) and short-scoreboard
(`0.267004`). NCU classifies the kernel as compute-heavier-than-memory and
reports tensor/shared as its highest-used pipeline. Low occupancy is jointly
limited by registers and shared memory; it is not caused by DRAM bandwidth.

## Claim boundary and decision

This representative FA2 forward is not at the 88.3% Tensor ceiling measured
for the dominant BF16 GEMM, but it is already a mature compute/tensor kernel:
DRAM is below 3%, the tensor pipeline is around 70%, and the launch has more
than sixteen waves/SM. NCU's theoretical issue-slot suggestions are not an
end-to-end speedup estimate. The exact signature contributes about
`10.180 s/GPU` of kernel sum in the C15 profile and overlaps both NCCL and
other work; even an unrealistic 20% isolated improvement cannot close the
remaining `8.331 s` critical-path gap by itself.

No source-compatible flag in the pinned dispatcher changes the selected dense
kernel. FA3 is not an A100 path, SDPA was already slower in the matched C2/C3
comparison, and the installed projection fusion drops active PEFT LoRA math.
Therefore C19 does **not** authorize hand-replacing the production attention
kernel or changing backend. The next diagnostic must use the exact current
C18 timeline to find a larger source-owned gap or a repeated-compute mechanism;
it must not infer one from aggregate FA2 kernel duration.

Status: `PASS_DIAGNOSTIC_ONLY / NO_KERNEL_IMPLEMENTATION_AUTHORIZED`.
