# Iteration 351 profile attribution

## Identity and claim boundary

- Run: `iter351_gan_cuda_fusions_b3_profile_p1_20260829`.
- Source: `42c0cc6aae6c7039ebcf9944eb1d821c38371671`.
- Workload: GAN-on world16 FSDP2, B1/global batch16, Score packing off,
  CUDA-resident Generator/Critic residuals, CUDA EMA, G1/C5/EMA1, final
  G201/C1005/EMA801.
- Correctness: 16/16 PASS; the profile is diagnostic only.
- Formal denominator remains the matched unprofiled C1.1 result:
  `151.851949039 s`.  The profiled range is not a speed denominator.
- Nsight Systems: `2024.6.2.225`; both node reports and SQLite exports are
  nonempty.  `nsys-ai doctor` found no blocking profile-health failure.  The
  GPU model is absent from CUPTI target metadata, so this profile makes no MFU
  claim.

## Exact 16-rank phase attribution

The bounded analyzer scans each node SQLite once.  Every rank uses its exact
Full5/NVTX phase ranges.  Values below are medians across all 16 ranks; interval
unions may overlap unless explicitly described as exposed.  The mutually
exclusive partition reconciles to each phase wall with maximum absolute
residual `2.84e-14 s`.

| Phase | Wall (s) | Compute union (s) | NCCL union (s) | NCCL exposed vs compute (s) | Copy union (s) | No traced GPU activity (s) |
|---|---:|---:|---:|---:|---:|---:|
| Full5 | 151.754 | 130.967 | 55.181 | 6.812 | 0.541 | 13.424 |
| Generator update | 31.171 | 22.049 | 10.834 | 1.856 | 0.191 | 7.196 |
| G gradient forward | 5.433 | 3.996 | 2.966 | 0.436 | 0.025 | 0.986 |
| Score/DMD/GAN-G | 12.005 | 9.012 | 4.928 | 0.947 | 0.091 | 2.031 |
| G backward | 9.824 | 9.008 | 2.883 | 0.457 | 0.075 | 0.298 |
| G optimizer | 3.905 | 0.033 | 0.051 | 0.048 | 0.000 | **3.811** |
| Five G no-grad rollouts | 32.543 | 29.548 | 20.243 | 2.362 | 0.105 | 0.564 |
| Critic phase | 87.949 | 79.404 | 24.273 | 2.667 | 0.239 | 5.623 |
| Critic update 0 | 20.665 | 15.869 | 4.801 | 0.579 | 0.047 | **4.173** |
| Critic update 1 | 16.959 | 15.869 | 5.146 | 0.633 | 0.048 | 0.416 |
| Critic update 2 | 16.780 | 15.878 | 4.828 | 0.549 | 0.048 | 0.291 |
| Critic update 3 | 16.681 | 15.893 | 4.749 | 0.446 | 0.048 | 0.292 |
| Critic update 4 | 16.675 | 15.888 | 4.709 | 0.441 | 0.048 | 0.292 |

H2D is small and D2H is effectively absent in the current CUDA-resident path.
The full-span `nsys-ai` analysis reports `6.833 s` exposed communication on
representative node0 GPU0 and classifies the critical path as GPU-compute
dominated.  Its stream-summed `48% idle` finding is not used as a device-wall
idle claim; the exact device-wall complement above is the authoritative
timeline metric.

## C2 preflight: world16 optimizer dispatch warmup

- **Current gate:** C1.1 passed and remains above the 30% target; C2 is the
  next single-variable correctness/timing iteration.
- **Authorized hypothesis:** PyTorch lazily builds DTensor/foreach AdamW
  dispatch during the first G and first C optimizer commits.  A setup-only,
  zero-gradient, zero-LR, zero-weight-decay warmup followed by exact local
  parameter/Adam-state restoration can move this one-time cost outside the
  formal timer.
- **Frozen contracts:** source workload, trace/RNG, model/input/precision,
  world16 FSDP topology, loss, Score/DiT inventory, update order/counts,
  optimizer hyperparameters, EMA, and formal timer boundaries remain C1.1.
- **Matching prior evidence:** the same mechanism is already bounded and
  validated for node-HSDP (`10a4a50`); the helper verifies exact local
  parameter and Adam-state restoration and clears synthetic gradients.  In the
  current profile, G optimizer has `3.811 s` no-traced time and Critic update 0
  has `4.173 s`, whereas Critic updates 1-4 have only `0.292-0.416 s`.
- **Rejected/superseded paths:** do not repeat the known B2 trace-control or
  Score-packing parity failures; do not combine HSDP, attention changes, or a
  new kernel with C2.  HSDP alone is not the first choice because current
  exposed NCCL is only `6.812 s` and changing 16-way to 8-way shard payload can
  remove only a small fraction of it.
- **Falsifiable question:** does the same exact-restoration warmup accept the
  world16 `(Shard(),)` DTensor layout, pass focused tests and 16/16 frozen
  correctness with unchanged fingerprints, then reduce matched unprofiled
  Full5 time?
- **Decision:** `GO` for C2 only.

The nsys-ai evidence timeline was built remotely at
`http://127.0.0.1:18145`; the server was stopped before the next GPU launch.

