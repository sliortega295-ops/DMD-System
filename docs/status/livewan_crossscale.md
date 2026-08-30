# LiveWan cross-scale status

## Current gate

`SERVER_ASSET_AND_INTEGRATION_PREFLIGHT`

## Authorized hypothesis

A persistent mixed GPU pool can overlap the resident Wan1.3B Student work with
the Wan1.3B Critic and FSDP2 Wan14B Teacher work and achieve at least 1.30x over
one sequential world16 FSDP2 denominator without changing SF-DMD math.

## Frozen contracts

See `docs/contracts/livewan_crossscale_g1c5_world16.md`. Source, model shapes,
random operations, global batch, G1/C5 updates, optimizer ordering, continuous
sigma, and application call inventory are frozen.

## Current factual status

- Source-pinned typed plans, adapter, operation trace/bootstrap builders,
  sequential worker, 8G+8Score worker, launch/controller, fail-closed summaries,
  correctness comparator, and model-free topology/P2P gates are implemented.
- Focused tests and the repository refactor suite passed before the final
  topology/receipt additions; those additions require a fresh test pass.
- The validated Torch 2.10 environment and two idle 8xA100 nodes were observed.
- Wan2.1-T2V-1.3B is present and pinned.
- Wan2.1-T2V-14B, LiveWan prompts, and the 80-sample input bundle are being
  prepared under persistent shared storage.
- Real model construction, correctness, timing, profiling, and 1.30x result are
  `NOT_RUN` until assets and the exact clean experimental HEAD are complete.

## GO / NO-GO

GO for bounded source tests, asset preparation, model-free world16 probes, and
then the single sequential capacity gate. NO-GO for performance claims or pool
search until baseline/pipeline correctness passes on the same revision.
