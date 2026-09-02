# LiveWan cross-scale status

## Current gate

`CONTINUOUS_B16_50_55_TARGET_MET_C24`

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
- C24 restores the exact B16 arm and adds only the source-pinned exact Score
  modulation fusion. Validate, correctness, and timing all passed at
  `c2a1ee389f7fe93dd9c289ab32068e41e5b9a90b` with 16/16 ranks and final
  `G3/C15/EMA0`. Its actual B16 steady intervals are `54.493145533 s` and
  `54.390061760 s`, median `54.441603646 s` (`1.469464429 samples/s`).
- C24 improves the C21 B16 control by `0.653607621 s` (`1.186%`, `1.01201x`)
  without changing global batch, pools, trace, logical/physical calls, losses,
  or update order. Every Score rank reports 2301 fused forward and 900 fused
  backward invocations with zero contract failure.
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
winner is C24 at `54.441603646 s`; `54.291817154 s` is only the B32 result normalized to
the amount of work in B16 and must not be described as the actual B32 wall.

## Current decision

`TARGET_MET_STOP`. C24 proves the 50-55 second objective on an actual same-B16
run. C21-C1 and C22 remain rejected, while C23 remains a separate B32 throughput
diagnostic. Do not continue last-step NCCL, wave-count, or unrelated kernel
search under this completed objective.
