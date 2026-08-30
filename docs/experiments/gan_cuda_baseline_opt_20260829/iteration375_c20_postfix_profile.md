# Iteration 375: C20 post-C18 diagnostic profile

## Claim boundary

This run is diagnostic-only.  The formal C18 denominator remains
`129.441379467 s`; the controller's profiled process wall is not used for a
speed claim.

## Identity and validity

- Run: `iter375_gan_cuda_c18_profile_p1_20260830`
- Code: `91ca0e007f64cfb857089fd73957669397754a8c`
- Workload: exact C18 GAN-on world16 FSDP2, B1/global-B16, Score packing off
- Range receipts: 16/16 PASS
- Nsight reports/SQLite: 2/2 nonempty
- Exact Full5 range median: `130.093331865 s`
- Nsight parser errors, CUDA OOM, cgroup OOM: zero

## All-rank median attribution

| Phase | Wall (s) | Compute union (s) | NCCL union (s) | NCCL hidden by compute (s) | NCCL exposed vs compute (s) | Copy exposed (s) | No traced GPU activity (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full5 | 130.093332 | 119.306060 | 57.804130 | 50.703842 | 6.947460 | 0.376638 | 3.470532 |
| Generator update | 23.104220 | 20.494940 | 10.234208 | 8.949983 | 1.280717 | 0.124812 | 1.236073 |
| Score/DMD/GAN-G | 9.527643 | 8.529923 | 4.217917 | 3.631842 | 0.593242 | 0.071168 | 0.340003 |
| Five Generator no-grad rollouts | 30.986322 | 28.010204 | 20.466938 | 17.894036 | 2.543321 | 0.095432 | 0.374100 |
| Five Critic updates | 75.552953 | 70.533811 | 26.547363 | 23.518464 | 2.995996 | 0.152158 | 1.857656 |

The sets overlap unless explicitly described as exposed-vs-compute.  Temporal
kernel coverage is not SM utilization or MFU.

## Post-fix result

Relative to C15, C17/C18 reduce the exact Generator-update profile wall by
`3.096269 s`, almost entirely by removing `2.569427 s` of no-traced time and
`0.528081 s` of exposed NCCL.  Score/DMD/GAN-G alone falls by `2.691273 s`.
Model compute is unchanged within profile noise.

The first Critic update contains a separate `~1.3 s` GPU-inactive interval
inside FSDP root post-backward/reduce-scatter.  It overlaps hundreds of
`cudaFree` calls and is not evidence of model offload.  Later Critic updates do
not reproduce this gap.

## Next bounded candidate

C21 will test one new mechanism only: keep the Generator's FSDP2 parameters
unsharded across the five already-existing B1 no-grad rollouts, then restore
the original per-block `reshard_after_forward=True` state before Critic work.
It does not batch samples or alter any DiT call.  Capacity is uncertain and is
therefore a fail-closed gate, not an assumed optimization.
