# W1 — Wan1.3B autoregressive DMD

**Original protocol:** Helios hierarchical AR; No-GAN; G1/C5/EMA5.

Original Wan adapter with compute-qualified retention. Fresh measured costs are required; r34 hand-written EMA packets are not promoted.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads native validate --task configs/distillschedule/original_wan/full5_reference.json
```

Continue through the existing [native runbook](../../docs/runbooks/wan_compute_state_planning_v1.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/strategies/wan_cli.py`](../../src/distillschedule/strategies/wan_cli.py)
- [`src/distillschedule/strategies/wan_worker.py`](../../src/distillschedule/strategies/wan_worker.py)
- [`src/distillschedule/strategies/wan_planning.py`](../../src/distillschedule/strategies/wan_planning.py)

The compatibility profile `wan13b_nogan_resident_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `ea29129342741b54e59c4773e07454f541fadcc7`.

H100 measured source: `sliortega295-ops/DMD-System@7a668bcb12ad37d831dced3aca1be4ec7be0a2fe`. Published source: `97b8ecc9cf7747d93619acad1a6083f98faae084`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.

To inspect the optimized compute/retention cohort, append `--include-incumbent-bundle --include-critic-retention` in the qualified native environment. This requires compatible Triton and intentionally rejects an unqualified environment; it does not install a backend.
