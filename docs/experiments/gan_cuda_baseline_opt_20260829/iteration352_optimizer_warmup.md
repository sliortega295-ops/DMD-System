# Iteration 352: world16 optimizer dispatch warmup

## Decision record

- Gate: reduce the first-use optimizer dispatch bubbles seen in the Iteration
  351 all-rank profile without changing the frozen GAN-on workload.
- Authorized hypothesis: run the existing zero-gradient, zero-learning-rate,
  zero-weight-decay optimizer dispatch warmup before the formal timer, then
  restore local parameters, Adam tensors, optimizer step, and gradients.
- Frozen contract: world16 FSDP2 FULL_SHARD, 384x640 BF16, B1/global B16,
  Score packing off, logical steps 1000-1004, G1/C5/EMA1, final
  G201/C1005/EMA801, CUDA residuals and EMA, no parameter offload/streaming or
  model swap, and unchanged trace/RNG/loss/update/inventory/timer boundaries.
- Prior evidence: Iteration 351 attributed about `3.811 s` no-traced activity
  to Generator optimizer first use and `4.173 s` to the first Critic update,
  while later Critic optimizer boundaries were only `0.291-0.416 s`.
- Decision: GO, as a setup-only single-variable arm.

## Preserved failed attempts

`iter352_gan_cuda_optwarm_correctness_r2_20260829` failed closed before the
workload because the Critic exposed two distinct DeviceMesh Python objects.
Inspection proved both objects describe the same logical `cuda`, rank 0-15,
shape `(16,)`, name `fsdp`, placement `Shard` mesh.  The repair validates this
logical signature while continuing to reject any material topology mismatch.

## Focused validation

- Focused tests: `75 passed`.
- Shell syntax: PASS.
- Generator warmup: one DeviceMesh wrapper; logical world16 Shard signature.
- Critic warmup: two equivalent wrappers; logical world16 Shard signature.
- Every rank reports parameter bytes restored, Adam-state bytes restored,
  optimizer step restored, gradients cleared, and warmup outside the timer.

## World16 evidence

Correctness:

- Run ID: `iter352_gan_cuda_optwarm_correctness_r3_20260829`.
- HEAD: `9bb21c0a833c7bc53d675df6be1792200ac5777f`.
- Summary SHA-256:
  `c84b839371aec84eedce4593a9ac21fa9f5d07b3d5abc4382303b66e66b8849b`.
- Result: 16/16 PASS, G1/C5/EMA1, final G201/C1005/EMA801, exact physical
  inventory, no CUDA OOM, and no cgroup OOM increment.

Formal unprofiled timing:

- Run ID: `iter352_gan_cuda_optwarm_timing_r1_20260829`.
- Summary SHA-256:
  `000742835774fd96bfe1996dff6a46b58ae94da2ec6f801cd7b36793d129fd4d`.
- Full5: `144.621745256 s`.
- Rank timing range: `144.601088321-144.613901685 s`.

## Outcome

| Comparison | Delta | Reduction | Speedup |
|---|---:|---:|---:|
| C2 vs C1.1 (`151.851949039 s`) | `-7.230203783 s` | `4.7614%` | `1.0500x` |
| C2 vs C0 (`173.014483004 s`) | `-28.392737748 s` | `16.4106%` | `1.1963x` |

C2 is promoted as the current intermediate substrate.  It does not complete
the requested 30% reduction: the threshold is `121.110138103 s`, leaving
`23.511607153 s` (`16.2573%` of C2) to remove.

## Next constraint

C3 may change only the attention backend to the pinned local FA2 build already
validated on the same Helios model family.  It must not simultaneously add
HSDP, selective checkpoint-output retention, or another kernel optimization.
