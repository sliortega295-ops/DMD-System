# Nsight utilization and GAN-on role-phase evidence bundle

This directory contains the five figures embedded in the Obsidian note
`2026-08 Helios Nsight图`, renamed to stable descriptive filenames, plus a
curated machine-readable snapshot of the current GAN-on measurements.

## Figures

1. `01_nogan_all40_wave8_nsys_timeline.png`
   - No-GAN all-40, `swap-wave=8`.
   - Existing Nsight/CUPTI capture reanalysis.
   - Profiled PhaseTracer window: `737.430 s`; matched unprofiled native Full5:
     `688.726 s`.
2. `02_nogan_routea_score4_404_diagnostic_timeline.png`
   - No-GAN Route-A Score4.
   - `404.042 s` offline PhaseTracer/block-streamer diagnostic; this is not a
     CUPTI timeline and formal paired timing was not run.
3. `03_nogan_full5_iteration309_nsys_timeline.png`
   - No-GAN Full5 Iteration 309.
   - Profiled diagnostic: Step10-A `110.24 s`, transition `17.85 s`, Critic
     phase `164.22 s`, Full5 `288.98 s`.
   - Matched unprofiled Full5 median: `279.934 s`.
4. `04_gan_on_sequential_fine_nsys_timeline.png`
   - GAN-on sequential baseline with fine physical Critic/GAN ranges.
5. `05_gan_on_rolephase_reference_timeline.png`
   - GAN-on Role-Phase Full5.
   - All 16 workload/range receipts are present; exact CUDA/NVTX activity is
     available only for representative ranks 2 and 9. Some role splits are
     receipt-reconstructed and are visually distinguished in the figure.

The SHA-256 digest, dimensions, evidence coverage, and run identity for every
figure are recorded in `gan_on_pipeline_metrics.json`. Exact read-only
reanalysis of the existing No-GAN Nsight SQLite reports is recorded in
`no_gan_utilization_supplement.json`; no new GPU run was launched for that
supplement.

## Utilization results and denominator

The semantic bars should **not** be interpreted as GPU-compute utilization.
They include compute, NCCL, copies, synchronization, commit/receive waits,
transition work, and unclassified host/control time.

For both No-GAN captures below, each rank uses its node's closed profiler range
as denominator. The table reports the arithmetic mean over ranks. `Active` is
the interval union of compute, NCCL, and all CUDA memcpy directions;
`No traced GPU activity` is its complement, not proof of scheduler or CPU idle.

| Existing capture | Exact ranks | Compute | NCCL | All copy | Active | No traced GPU activity |
|---|---:|---:|---:|---:|---:|---:|
| No-GAN all-40, wave8 | 16 | **14.03%** | 25.69% | 47.21% | 86.93% | 13.07% |
| No-GAN Iteration 309, all roles | 16 | **26.24%** | 20.79% | 18.58% | 64.73% | 35.27% |

The all-40 trace therefore looks full for a real reason--traced GPU activity is
high--but only about `14%` of the capture wall is model-compute kernel union.
H2D alone averages `45.66%` of the per-node profiler range. In Iteration 309,
less streaming raises compute occupancy while exposing role imbalance:

| Iteration 309 target role | Ranks | Compute | NCCL | All copy | Active | No traced GPU activity |
|---|---:|---:|---:|---:|---:|---:|
| Generator | 0, 1, 8, 9 | **38.40%** | 8.43% | 8.31% | 54.64% | 45.36% |
| Critic | 2-7, 10-15 | **22.19%** | 24.91% | 22.00% | 68.10% | 31.90% |

Global rank 8 is the strongest outlier: `37.14%` active union and `62.86%`
no-traced-GPU-activity residual. This is consistent with the visible
role-allocation/scheduling bubble, but the residual still must not be labeled
physical GPU idle without a narrower causal trace.

For the GAN-on Role-Phase trace, exact compute-kernel interval union over each
representative rank's 0.5-second binned semantic span is:

| Representative rank | Compute | NCCL | H2D | D2H |
|---:|---:|---:|---:|---:|
| 2 (producer) | 20.89% | 6.44% | 1.42% | 3.71% |
| 9 (critic) | 19.27% | 42.56% | 5.01% | 13.61% |

These are independent interval unions and may overlap, so the columns are not
additive. The earlier rough statement of “about 25% compute” should therefore
be narrowed to approximately `19-21%` for the two exactly captured GAN-on
ranks. It must not be generalized to every figure: the exact No-GAN results
above range from `14.03%` overall for all-40 to `38.40%` for Iteration 309's
Generator role.

## Current GAN-on scheduling evidence

- Semantic spans: rank 2 `573.202 s`; rank 9 `576.142 s`.
- Rank 2: Phase A `99.741 s`, A-to-B `90.170 s`, Phase B `298.857 s`,
  B-to-A `84.309 s`.
- Rank 9: A-to-B `86.298 s`, Phase B `298.871 s`, B-to-A `84.556 s`.
- The two transitions consume roughly `29.7-30.4%` of the representative
  semantic spans.
- Producer rank 2 contains `202.883 s` summed Critic-commit-wait ranges during
  Phase B. This is evidence of a scheduling/critical-path bubble, but it is not
  by itself proof that the GPU was physically idle.

## ZeRO-3 status boundary

Iteration 299 attempted one bounded DP12 ZeRO-3 Critic update with 32 MiB
activation offload and zero streamed blocks. It stopped before checkpoint
restore or any Critic forward/backward/update because a fail-closed guard used
a ZeRO-1/2-only `partition_gradients` attribute against DeepSpeed 0.18.4
Stage-3.

Therefore:

- no CUDA OOM was observed;
- no Critic commit occurred;
- one-update ZeRO-3 capacity/correctness remains unresolved;
- Full5, reverse handoff, timing, Nsight, and performance comparison are all
  `NOT_RUN`.

This failure is not evidence that ZeRO-3 is slower, faster, correct, or unable
to fit.

## Claim boundary

All timelines in this bundle are diagnostic. Native latency or speedup claims
must come from matched unprofiled runs. White gaps are unclassified unless a
CUDA-activity track proves otherwise, and profiler receipt/finalization tails
must not be stretched into the semantic timeline.
