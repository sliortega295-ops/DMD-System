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
figure are recorded in `gan_on_pipeline_metrics.json`.

## Key utilization result

The semantic bars should **not** be interpreted as GPU-compute utilization.
They include compute, NCCL, copies, synchronization, commit/receive waits,
transition work, and unclassified host/control time.

For the GAN-on Role-Phase trace, exact compute-kernel interval union over each
representative rank's 0.5-second binned semantic span is:

| Representative rank | Compute | NCCL | H2D | D2H |
|---:|---:|---:|---:|---:|
| 2 (producer) | 20.89% | 6.44% | 1.42% | 3.71% |
| 9 (critic) | 19.27% | 42.56% | 5.01% | 13.61% |

These are independent interval unions and may overlap, so the columns are not
additive. The earlier rough statement of “about 25% compute” should therefore
be narrowed to approximately `19-21%` for the two exactly captured ranks.

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
