# LiveWan cross-scale status

## Current gate

`CONTINUOUS_B16_COMPLETE_B32_SYSTEMS_DIAGNOSTIC_COMPLETE`

## Authorized hypothesis

A persistent 8G + 8Score mixed GPU pool can overlap the resident Wan1.3B
Student work with the Wan1.3B Critic and FSDP2 Wan14B Teacher work without
changing the registered SF-DMD call and update inventory. Increasing the
continuous workload from global B16 to B32 may amortize only the remaining
pipeline fill and drain.

## Frozen contracts

See `docs/contracts/livewan_crossscale_g1c5_world16.md`. Source, model shapes,
random operations, optimizer order, continuous sigma, application call
inventory, and G1/C5/EMA0 update semantics remain frozen. The pool assignment
is persistent: ranks 0-7 are Generator ranks and ranks 8-15 are Score ranks.

## Current factual status

- C21 continuous B16 completed on 16/16 ranks with final `G3/C15/EMA0`.
  The unprofiled steady Full5 interval median is `55.095211267 s`.
- C21-C1 (begin the first B1 Critic backward as soon as wave0 arrives) completed
  but regressed to `55.150482594 s`; it is rejected.
- The Critic checkpoint RNG-state-elision candidate C22 completed but regressed
  to `55.153462575 s`; it is rejected.
- C23 continuous B32 completed correctness and timing on 16/16 ranks. Its real
  B32 steady Full5 interval is `108.583634309 s` for 160 logical samples,
  equivalent to `54.291817154 s` per 80 samples (`1.473518556 samples/s`).
- C23 reuses the 16 available B16 source rows for slots 16-31 within each step,
  while preserving distinct operation keys and RNG. It is therefore a bounded
  systems throughput diagnostic, not a convergence or training-quality result.
- Generator backpressure and Score receive wait are already negligible in the
  final B32 run (`0.008950 s` and `0.000143 s` medians respectively). More
  waves produce only a `1.48%` throughput gain; remaining improvement requires
  reducing Critic/Teacher compute rather than another transport tweak.
- All accepted runs had zero CUDA/cgroup OOM increments. Final B32 peak
  allocated was `35.175426 GiB/rank`, with `1.006226 GiB` minimum driver free.

## Claim boundary

The B16 and B32 observations are diagnostic-only. Historical strict
sequential/pipeline numerical parity remains outside the preregistered gate,
and C23 source reuse is not a training-quality contract. The actual B16 Full5
wall is `55.095211267 s`; `54.291817154 s` is only the B32 result normalized to
the amount of work in B16 and must not be described as the actual B32 wall.

## Current decision

`STOP_PIPELINE_BUBBLE_SEARCH`. The 50-55 second target is reached only in the
B32 per-sample/B16-equivalent sense, while the actual same-B16 winner is just
above the band at `55.095 s`. C21-C1 and C22 are rejected. Do not continue
last-step NCCL or wave-count tuning without new evidence; the next independent
hypothesis would have to target the dense Critic backward and step0 Teacher
compute.
