# W2 — Helios14B No-GAN

**Original protocol:** Hierarchical AR; G1/C5/EMA5.

Explicit safe-incumbent cohort, not the older locked 94-second recipe. Later partial-retention derivatives remain independently pinned.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads candidates --workload helios14b_nogan_best --include-helios-safe-incumbent
```

Continue through the existing [native runbook](../../docs/runbooks/native_planner_workflow.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/search/helios_model_adapter.py`](../../src/distillschedule/search/helios_model_adapter.py)
- [`src/distillschedule/search/workflow_cli.py`](../../src/distillschedule/search/workflow_cli.py)

The compatibility profile `helios14b_nogan_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `07fcc0ba1ec28b810b50cac92ec41c1a5232d3e1`.

H100 measured source: `sliortega295-ops/DMD-System@8fc70e8ce9f6e0243604812120119c9f4c078e55`. Published source: `8fc70e8ce9f6e0243604812120119c9f4c078e55`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.
