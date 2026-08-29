# Iteration 357: C6 node-HSDP placement result

## Claim boundary

- Parent and current fastest formal arm: C5 world16 FSDP2,
  `142.295341636 s`.
- C6 changes only parameter placement to node-local HSDP
  `(replicate=2, shard=8)`.  The GAN-on steps, B1/global-B16 inputs, trace,
  physical model-call inventory, kernels, optimizer/EMA order, and Full5 timer
  are unchanged.
- Exact source: `b7b2f0356c85691338385277a6579c7f69bcdd7e`.

## Construction result

The first C6 correctness run failed before model compute because it constructed
the full Critic on CUDA before sharding.  With the Generator HSDP shard already
resident, that temporary copy reached about `38.27 GiB` allocated and failed a
`144 MiB` allocation.  The bounded C6.1 repair restores on CPU and invokes the
same 40-block bottom-up `fully_shard` policy directly, so each block moves and
shards before the next.  It does not change the persistent workload or timer.

The corrected path reports on every rank:

- `construction_order=CPU_RESTORE_THEN_BLOCKWISE_FSDP_DEVICE_SHARD`;
- `construction_only_compatibility=true`;
- FSDP2 backend `COMPOSABLE_HYBRID_SHARD`, mesh `[2,8]`;
- no parameter offload, streaming, model swap, or transition.

## Gates and formal result

| Gate | Run | Result |
|---|---|---|
| validate-only | `iter357_gan_cuda_hsdp_fsdpfirst_validate_r2_20260829` | PASS, SHA `7b715ef5...92fbd5` |
| correctness | `iter357_gan_cuda_hsdp_fsdpfirst_correctness_r2_20260829` | 16/16 PASS, SHA `ec2dd952...ce7c9a` |
| unprofiled timing | `iter357_gan_cuda_hsdp_fsdpfirst_timing_r1_20260829` | PASS, SHA `fe94b23a...e00f69` |

The timing summary reports `144.716706512 s`.  All 16 ranks complete G1/C5/EMA1
and end at G201/C1005/EMA801.  Rank0 reaches `31.183228970 GiB` max allocated,
`34.957031250 GiB` max reserved, and `0.037475586 GiB` final driver free; no
CUDA or cgroup OOM occurs.

## Decision

C6.1 regresses C5 by `2.421364876 s` (`1.701647326%`).  It remains
`23.606568409 s` above the requested 30%-reduction threshold.  Node-HSDP is
therefore **rejected as a performance candidate**.  The construction repair is
retained because it is a useful fail-closed capacity path, but subsequent
performance work returns to the C5 world16-FSDP substrate.

The evidence also closes the immediate communication-topology hypothesis:
halving the shard group does not improve matched GAN Full5 wall.  The next
bounded question is kernel-level compute efficiency on the measured dominant
BF16 GEMM and FlashAttention families, using NCU rather than treating temporal
kernel coverage as MFU.
