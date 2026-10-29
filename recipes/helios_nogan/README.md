# W2 — Helios14B No-GAN

**Original protocol:** Hierarchical AR; G1/C5/EMA5.

For native complete-checkpoint export and independent new-prompt decoded video,
see [Native Student output](NATIVE_OUTPUT.md).

Explicit safe-incumbent cohort, not the older locked 94-second recipe. Later partial-retention derivatives remain independently pinned.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads candidates --workload helios14b_nogan_best --include-helios-safe-incumbent
```

Continue through the existing [native runbook](../../docs/runbooks/native_planner_workflow.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- source checkout: `src/distillschedule/search/helios_model_adapter.py`
- source checkout: `src/distillschedule/search/workflow_cli.py`

The compatibility profile `helios14b_nogan_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: source-bound record (source checkout: `artifacts/publications/20261019/evidence/planner/six_workload_native_match.json`), status `MATCHED_DIAGNOSTIC_TARGET_MET`. Its fixed training source is `07fcc0ba1ec28b810b50cac92ec41c1a5232d3e1`.

H100 measured source: `sliortega295-ops/DMD-System@8fc70e8ce9f6e0243604812120119c9f4c078e55`. Published source: `8fc70e8ce9f6e0243604812120119c9f4c078e55`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived measurement index (source checkout: `artifacts/publications/20261019/h100/measured_results.csv`) defines the actual denominator and numerical status.

Read the six-workload correction (source checkout: `docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md`) before interpreting any number. No new GPU or quality result is created by this recipe.

## Public training configuration

For the complete original tiny model on CPU, use the separately versioned
[warm No-GAN Program reference](CPU_REFERENCE.md). It connects genuine original
G2/C10/EMA10 state, original subsequent updates, common safe recovery and
current full-Generator export/new-prompt latent inference. CPU1 source-rank0
execution is not world16 reduction or pretrained quality validation.

For fresh native cohorts without historical SSH/HF/flash/trace assets, use the
separate [`training.reference.json`](training.reference.json) and
[direct-reference guide](REFERENCE_TRAINING.md). This requires an actual pinned
original Step10 G2/C10/EMA10 package; it does not invent a default checkpoint.

For rank-local amplified exits produced by all-zero native history, use the
separate [native-history continuation](NATIVE_HISTORY.md). It retains the
original per-rank rollout rather than forcing normal-branch inputs; the aligned
FSDP2 phase implementation is CPU-verified, with full-size GPU capacity still
requiring admission.

[`training.json`](training.json) selects `helios_nogan.safe_ed8.v1` with explicit
`mode: manual` and `execution.preset`; supply your qualified original assets
and output directory, **not a historical saved plan**. The existing typed
implementation is bound against the current source/assets and enters the same
native runner and gates. This is exactly two fixed-contract cycles with no
checkpoint/resume, not continuous training. REVIEW; original warm No-GAN task, not fresh-stream optimized training.

Read the [direct preset guide](../../docs/guides/native_presets.md) for the
complete source/mechanism/runtime mapping and the
[native training guide](../../docs/guides/native_workloads.md) for separate
reference/fresh-input and saved-plan paths. CPU admission does not certify
GPU execution, numerical parity, quality or speedup.
