# Iteration 370: C15 post-fix Nsight diagnosis

## Frozen diagnostic identity

- Source HEAD: `1ef0e907aff7a64d4b459f0f41af64c499d3689b`.
- Run: `iter370_gan_cuda_c15_profile_p1_20260830`.
- Workload: exact C15 GAN-on world16 Full5; B1/global-B16, Score packing off,
  world16 FSDP2, CUDA EMA, both residuals CUDA-resident, local pinned FA2,
  cross-K/V reuse, and the C14/C15 fused modulated-norm forward/backward path.
- The profile wall is diagnostic-only. The formal denominator remains
  `131.562027140 s` from Iteration 369.

Both node-outer reports and SQLite exports are nonempty. The SQLite files have
kernel events on devices `0..7`, zero unresolved kernel names, and no Nsight
post-processing error. All 16 exact range receipts passed.

## Overlap-aware result

The 16-rank median timeline attribution inside the exact Full5 range is:

| Category | Seconds |
|---|---:|
| Full5 range | 131.549899 |
| non-NCCL compute union | 119.347899 |
| NCCL union | 57.901986 |
| compute/NCCL overlap | 51.027846 |
| NCCL exposed versus compute | 6.874140 |
| copy exposed versus compute/NCCL | 0.377047 |
| no traced GPU activity | 5.040138 |

The mutually exclusive median partition reconciles to the Full5 wall:

- compute only: `68.290791 s`;
- compute + NCCL: `51.027764 s`;
- NCCL only: `6.749571 s`;
- NCCL + copy: `0.124569 s`;
- copy only: `0.251928 s`;
- no traced GPU activity: `5.040138 s`.

The profile is therefore compute-dominated on the timeline. This is temporal
coverage, not SM utilization or MFU. The dominant sampled BF16 GEMM was already
measured at about 88.32% Tensor/SM throughput in Iteration 358, so replacing
that one cuBLAS kernel remains closed.

## Remaining source-qualified work

- All-gather kernel union remains large (`50.93 s` median) but most is hidden;
  the exposed collective budget is only about `6.87 s`.
- Generator no-grad rollout is `30.99 s`, of which `28.01 s` is non-NCCL
  compute. Cross-K/V reuse is already active.
- Critic phase is `74.25 s`, of which `70.87 s` is non-NCCL compute.
- The BF16 add family contributes about `6.2 GPU-s/GPU`; source and receipt
  reconciliation attributes more than half its invocation inventory to exact
  scale-1 LoRA epilogues. The already-tested beta-add GEMM route remains
  rejected because it regressed registered B3 shapes.
- D2H is effectively absent. H2D is only about `0.227 GiB/rank`, and copy-only
  wall is too small to close the target gap.

Generic `nsys-ai` root-cause totals that combine startup, all eight devices, and
the captured Full5 must not be interpreted as one-GPU critical-path wall. The
range-qualified per-device analysis above is authoritative for this iteration.

## Decision boundary

C15 is `10.451889037 s` above the requested 30%-faster threshold
(`121.110138103 s`). Exposed-NCCL-only tuning cannot close that gap, but the
trace shows every large FSDP all-gather uses NCCL's `RING_LL` kernel. A bounded
two-node microbenchmark of the exact large-block all-gather/reduce-scatter
regime is authorized before changing the formal workload. This is a new
protocol-selection question, not a repeat of rejected HSDP or explicit FSDP
prefetch.
