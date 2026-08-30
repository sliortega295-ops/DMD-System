# Iteration 373: C18 official Generator `free_memory` elision

## Preflight decision

- **Current gate:** C17 is the fastest valid substrate at
  `130.630471052 s`; the registered target remains `121.110138103 s`.
- **Authorized hypothesis:** remove only the four official
  `diffusers.training_utils.free_memory()` calls made inside the one GAN
  Generator Score/DMD/GAN-G operation, and only in formal timing/profile.
- **Frozen baseline and contracts:** C17 world16 FSDP2, B1/global-B16,
  steps 1000--1004, local FA2, trace/RNG/loss/call inventory, G1/C5/EMA1,
  final G201/C1005/EMA801, optimizer/EMA order, attention/checkpoint/cache,
  residency and Full5 timer are unchanged.
- **Matching evidence:** the valid C15 profile is still representative of the
  unchanged C17 Generator path.  Its Score/DMD/GAN-G median wall is
  `12.216890 s`, with `2.630651 s` no visible GPU activity.  Exact
  process-scoped Runtime attribution finds `1.277312 s` median memory-API
  overlap, dominated by `cudaFree`, and `1.253964 s` without a CUDA Runtime
  interval.  The installed official sources contain two outer calls plus two
  calls in `Gan_D_Loss_With_Cached_Grad.forward` under the configured
  `is_gan_low_vram_mode=True` path.
- **Exact source identities:**
  - `compute_distribution_matching_loss`:
    `c479e4f3eec2730d1e991ecb19e06588ec30510dabe48e2daf3846d9f7960db1`
  - `Gan_D_Loss_With_Cached_Grad`:
    `8455f5d265b189208491681e6372790f7b5809773953a6c855a668f10b632643`
  - Diffusers `free_memory`:
    `d16707d1b9935038a99f9a927043844d7eea8d22a9b6e008fbfc014c49885df3`
- **Remote attribution evidence:**
  `iter370_gan_cuda_c15_profile_p1_20260830/analysis/phase_runtime_gap_attribution.node0.json`,
  SHA256
  `f448f0ac8925659b148c81e3a4e88bd161d955ca45d248b0516e79c024237a39`.
- **Rejected/superseded paths:** B2/B5, Real-Score packing, HSDP, explicit
  prefetch, forced NCCL protocols, C13 static-context cache and C10 simple
  LoRA beta-add remain closed.  C18 is distinct from C12, which removed one
  application-owned flush before rollout rather than these four official GAN
  helper calls.
- **Falsifiable question:** can exact bounded elision remove allocator/GC
  critical-path time and produce a Full5 below `130.630471052 s` without
  increasing CUDA/cgroup OOM counters or changing any semantic receipt?
- **Decision:** **GO**, subject to source-pinned focused tests, validate-only,
  16-rank correctness/capacity, then one unprofiled timing.

## Claim boundary

The `2.630651 s` profile complement is not a promised speedup.  Runtime
categories may overlap, CPU garbage collection is not directly traced, and
retaining allocator cache can trade latency for HBM fragmentation.  Only the
unprofiled Full5 result may establish a gain.

## Results

| Gate | Run | Result |
|---|---|---:|
| focused tests | 24 focused tests plus compile/shell gates | PASS |
| validate-only | `iter373_gan_cuda_c18_nofreememory_validate_r1_20260830` | 16/16 PASS |
| correctness attempt 1 | `iter373_gan_cuda_c18_nofreememory_correctness_r1_20260830` | ENVIRONMENT FAIL |
| correctness | `iter373_gan_cuda_c18_nofreememory_correctness_r2_20260830` | 16/16 PASS |
| formal unprofiled Full5 | `iter373_gan_cuda_c18_nofreememory_timing_r1_20260830` | **129.441379467 s** |

The first correctness attempt was cancelled before construction completed
after both platform SSH transports broke at approximately two minutes.  The
remote wrappers propagated `SIGTERM`, released both locks, left no GPU process
and recorded no OOM.  This is preserved as launcher/environment evidence, not
as a workload failure.  Adding task-local SSH keepalives passed a 135-second
two-host transport canary; the fresh `r2` run then passed without code changes.

Correctness records `(scope=1, calls=4, executed=4, elided=0, restored=1)` on
all 16 ranks.  Timing records `(1,4,0,4,1)`.  Both retain the C17 Critic scalar
receipts, G1/C5/EMA1 and final G201/C1005/EMA801.  Timing maximum allocated and
reserved HBM are `24.962112904 GiB` and `38.009765625 GiB`; minimum final
driver free is `5.053100586 GiB`.  CUDA and cgroup OOM deltas are zero.

Summary SHA256 values:

- validate: `99ba4c189732ce1365bdfcaebb04774fd36a81b875a0d733c89a8d6c7e8b35a5`;
- correctness: `4e30847ada714f90031ced0970c77c0974ec96e1c5ce1a014bdfed8ad03c4a3d`;
- timing: `1242f40f9a41767e4f0be8b34fe24c1af8967cca1685d09519e6ca6eea6599a4`.

## Decision

C18 improves C17 by `1.189091585 s` (`0.910271222%`) and C0 by
`43.573103537 s` (`25.184656672%`, `1.336624221x`).  Promote C18 as the current
fastest exact substrate.  It is still `8.331241364 s` above the 30%-faster
threshold and needs another `6.436304525%` reduction; the goal remains active.
