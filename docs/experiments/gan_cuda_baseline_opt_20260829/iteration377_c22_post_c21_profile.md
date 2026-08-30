# Iteration 377: C22 post-C21 diagnostic profile

## Workload identity

- Run: `iter377_gan_cuda_c21_profile_p1_20260830`
- Code: `59ae41c6a6a3d295117737ed6b26ef3e9a4102a9`
- Formal denominator (unchanged): C21 `125.814004444 s`
- Diagnostic exact-range median: `127.053260181 s`
- GAN world16 FSDP2, B1/global-B16, Score packing off, local FA2,
  Generator and Critic residuals CUDA-resident, GPU EMA, and the exact C21
  41-unit Generator no-grad persistence policy.

The capture passes the profile gate: 16/16 range receipts, two nonempty
reports and SQLite files, devices 0--7 on both nodes, no missing kernel names,
no Nsight parser error, no CUDA/cgroup OOM, and no training worker left on a
GPU. Profile wall time is diagnostic only.

## Overlap-aware attribution

All values below are the median over the 16 per-device exact ranges. Compute,
NCCL and copy unions overlap and must not be summed.

| Phase | Wall (s) | non-NCCL compute union (s) | NCCL union (s) | compute/NCCL overlap (s) | NCCL exposed vs compute (s) | no traced GPU activity (s) |
|---|---:|---:|---:|---:|---:|---:|
| Full5 | 127.053 | 116.565 | 38.595 | 33.228 | 5.337 | 4.900 |
| Generator update | 23.132 | 20.499 | 10.186 | 8.931 | 1.235 | 1.302 |
| Score/DMD/GAN-G | 9.520 | 8.529 | 4.193 | 3.608 | 0.587 | 0.340 |
| five Generator no-grad rollouts | 26.760 | 25.267 | 1.104 | 0.472 | 0.632 | 0.741 |
| Critic phase | 74.589 | 70.503 | 26.934 | 23.524 | 3.344 | 0.697 |

C21 removes the intended repeated materialization. Relative to C20/C18, the
five-rollout NCCL union falls by `19.362796 s`, exposed NCCL falls by
`1.911149 s`, and rollout wall falls by `4.225912 s` in the profiled paths.
Only rollout 0 retains parameter all-gather; rollouts 1--4 have zero NCCL.

## New bounded source finding

Process-scoped CUDA Runtime attribution assigns a median `2.1232 s` of the
Full5 GPU-inactive complement to `cudaFree`. On representative node0 rank 0,
the two C21 allocator-flush clusters are:

- before persistence: 892 `cudaFree` calls over a `1.258138 s` envelope;
- after explicit reshard: 429 calls over a `0.327522 s` envelope.

The C21 receipt shows why the conservative flushes were installed: reserved
memory changes from `30.3828 GiB` to `7.1855 GiB` before materialization, then
from about `38.0391 GiB` to `8.3359 GiB` after reshard. Allocated memory is
already only about `5.978 GiB` before entry and `6.071 GiB` after restore, so
the flushes release allocator cache rather than live model tensors.

This is a source-qualified candidate, not a guaranteed gain: retaining the
cache may fragment or exhaust 40-GiB HBM even though the subsequent Critic has
the same transformer geometry. Capacity/correctness must therefore precede
timing.

## Claim boundary and persistent artifacts

The raw reports and SQLite remain in remote persistent storage:

`/share/project/shuchenweng/yongyan_liu/DMD-System_gan_cuda_baseline_20260829/artifacts/runs/iter377_gan_cuda_c21_profile_p1_20260830/`

Derived evidence:

- `analysis/phase_attribution_all16_v2.json`
- `analysis/phase_runtime_gap_attribution.node0.json`

The remote host does not expose the `nsys-ai` CLI on the validated training
environment PATH, so the existing bounded one-pass SQLite analyzers were used
in place and the raw 494-MB databases were not copied locally. This does not
weaken the explicit profile-validity, union, or process-correlation checks;
it only means no `nsys-ai timeline-web` URL was produced.

