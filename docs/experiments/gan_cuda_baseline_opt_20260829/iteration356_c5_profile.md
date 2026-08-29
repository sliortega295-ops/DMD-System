# Iteration 356: C5 post-optimization profile

## Claim boundary

- Formal denominator remains the unprofiled C5 run
  `iter355_gan_cuda_selectiveflash20_timing_r1_20260829` at
  `142.295341636 s`.
- This run is diagnostic only.  Its controller rank-clock envelope is not a
  performance result.
- Exact runtime HEAD on both nodes:
  `e75c80e95198690ee070a8d7f77c6db9bfb85ce9`.
- Run ID: `iter356_gan_cuda_selectiveflash20_profile_p1_20260829`.
- Both node reports and SQLite exports are non-empty, 16/16 exact range
  receipts pass, and Nsight reports no parser error.

The raw reports remain on the remote shared evidence filesystem.  Analysis was
performed in place over SSH; no raw SQLite or nsys-rep was copied locally.

## Overlap-aware result

The exact all-rank analyzer reports a `142.680913011 s` median captured Full5
range.  The small difference from the formal run is profiler overhead and is
not used as a speed claim.

| Scope | Wall | Compute union | NCCL union | Compute/NCCL overlap | NCCL exposed vs compute | Copy exposed | No traced activity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full5 median | 142.681 s | 128.837 s | 60.529 s | 53.170 s | 7.354 s | 0.488 s | 6.128 s |
| Generator update | 28.751 s | 21.939 s | -- | -- | 2.294 s | -- | 4.408 s |
| G gradient forward | 6.580 s | 3.983 s | -- | -- | 0.482 s | -- | 2.114 s |
| Score/DMD/GAN-G | 12.342 s | 8.924 s | -- | -- | 1.416 s | -- | 1.930 s |
| G backward | 9.582 s | 8.971 s | -- | -- | 0.412 s | -- | 0.164 s |
| G no-grad rollout | 32.176 s | 29.343 s | -- | -- | 2.385 s | -- | 0.400 s |
| Critic phase | 81.680 s | 77.602 s | -- | -- | 2.890 s | -- | 0.981 s |

Compute, NCCL, and copy unions overlap and must not be summed.  `no traced
activity` is not labelled CPU idle without a matching runtime/API owner.

Representative-device `nsys-ai` results agree with the all-rank analyzer:

- node0 device0: `65.337 s` compute-only, `7.329 s` NCCL-only,
  `62.918 s` compute+NCCL overlap, and `7.082 s` device complement;
- node1 device0: NCCL-only `8.727 s`, complement `5.770 s`, and NCCL overlap
  `87.9%`;
- D2H is effectively absent and H2D is only about `29 ms` union on the
  representative GPU.  Host offload is not the current bottleneck.

The universal evidence step produced 43 findings.  The remote timeline server
is bound to `127.0.0.1:18156` on node0; it is reachable only through an SSH
port forward.

## Kernel evidence and root cause

The top kernel inventory is aggregated over eight devices; those sums are not
wall time.  Dominant families are BF16 GEMMs, FlashAttention forward/backward,
pointwise adds, direct/split copies, and LayerNorm.  C5 already uses the exact
all-role BF16 scale-one LoRA add path: rank0 reports 12,679 Generator and 4,908
Critic fused invocations, all BF16, scale 1, with zero fallback.  Therefore the
remaining direct-copy family cannot be labelled a BF16/FP32 LoRA cast without
new operator correlation.

The primary root cause is repeated model compute, not host transfer or exposed
collectives.  Even an impossible removal of all `7.354 s` exposed NCCL and all
`6.128 s` no-traced activity would leave the measured compute union above the
`121.110138103 s` target.  At least about `7.73 s` of compute-union reduction
is also required.

## Next bounded candidate

C6 changes only FSDP placement from world16 FULL_SHARD to node-local HSDP
`(replicate=2, shard=8)`.  It preserves 16 rank-qualified B1 samples, global
batch16, every physical model call, RNG/trace, optimizer/EMA and C5 compute
kernels.  This does not claim to solve the whole remaining gap; it tests
whether shortening all-gather groups and avoiding cross-node parameter
materialization reduces the measured `60.529 s` NCCL union, `7.354 s` exposed
tail, or compute/collective contention.

Historical No-GAN evidence makes this a bounded low-risk question rather than
a promised large win: the plain node-HSDP control improved only about `1.24 s`.
The GAN graph and C5 residency are materially different, so C6 must first pass
capacity and 16-rank correctness.  No all-gather prefetch, batching, packing,
or additional kernel is combined.

## Remote evidence

`/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/artifacts/runs/iter356_gan_cuda_selectiveflash20_profile_p1_20260829/`

Key derived files:

- `analysis/nsys_ai_doctor_node0.json`
- `analysis/nsys_ai_doctor_node1.json`
- `analysis/nsys_ai_findings_node0_gpu0.json`
- `analysis/phase_attribution_all16_v2.json`
- `analysis/top_kernels.node0_gpu0.json`
- `analysis/nccl_breakdown.node0_gpu0.json`
- `analysis/kernel_overlap_matrix.node0_gpu0.json`

