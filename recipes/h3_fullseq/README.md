# W5 — Audio-Video DMD (Full-Sequence / H3 Grid1000)

**Original protocol:** Full-sequence AV; approximate Grid1000 sigma; 4-step; G1/F5; LoRA128.

Historical A100/H100 timings exist. The latest manual/saved-plan pair is blocked by a sampled backward numerical FAIL. Do not bypass the gate.

## Inspect the actual adapter (no GPU execution)

Run from the repository root:

```bash
python scripts/distillctl.py workloads candidates --workload minimax_h3_grid1000_best
```

Continue through the existing [native runbook](../../docs/runbooks/native_planner_workflow.md) for asset qualification, measured calibration, planning and saved-plan execution. Source identity, original update counts, precision, capacity policy and numerical checks remain mandatory. This guide does not pre-fill a winner, create a new trainer or resume a stopped queue.

## Code ownership

- [`src/distillschedule/search/h3_action_model_adapter.py`](../../src/distillschedule/search/h3_action_model_adapter.py)
- [`src/distillschedule/backends/training/minimax_h3/trainer_loop.py`](../../src/distillschedule/backends/training/minimax_h3/trainer_loop.py)

The compatibility profile `minimax_h3_grid1000_best` is preserved but is not silently substituted for this guide's explicit cohort.

Shared integration origin: `ea29129342741b54e59c4773e07454f541fadcc7`. Subsequent organization commits are not historical training identities.

## Hardware/evidence

A100 manual/saved-plan evidence: [source-bound record](../../artifacts/publications/20261019/evidence/planner/six_workload_native_match.json), status `DIAGNOSED_FIRST_OBSERVED_BACKWARD_DIFFERENCE_SAMPLED_GATE_FAIL_PAIR_NOT_RUN`. Its fixed training source is `07fcc0ba1ec28b810b50cac92ec41c1a5232d3e1`.

H100 measured source: `sliortega295-ops/H3-DMD-for-A100@9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71`. Published source: `9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71`. This is a separately pinned implementation, not automatic proof of this checkout's H100 runtime. The archived [measurement index](../../artifacts/publications/20261019/h100/measured_results.csv) defines the actual denominator and numerical status.

Read the [six-workload correction](../../docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md) before interpreting any number. No new GPU or quality result is created by this recipe.

## Public training configuration

[`training.json`](training.json) selects `h3_fullseq.cpu_staged_trim.v1` with explicit
`mode: manual` and `execution.preset`; supply your qualified original assets
and output directory, **not a historical saved plan**. The existing typed
implementation is bound against the current source/assets and enters the same
native runner and gates. This is exactly two fixed-contract cycles with no
checkpoint/resume, not continuous training. FAIL unresolved; explicit research reproduction only, not stable or timing-eligible.

Read the [direct preset guide](../../docs/guides/native_presets.md) for the
complete source/mechanism/runtime mapping and the
[native training guide](../../docs/guides/native_workloads.md) for separate
reference/fresh-input and saved-plan paths. CPU admission does not certify
GPU execution, numerical parity, quality or speedup.

For finite fresh-data training and complete-cycle restart without a selected
benchmark plan, use the separate [original Grid1000 reference lifecycle](../h3-grid-original-training.md).
It retains the existing trainer and Grid objective, requires an explicit original
config and qualified assets, and keeps optional optimized paths OFF. CPU source
and restart mechanics do not promote the historical optimized numerical FAIL.

## Independent current-Student inference

The [converted Grid inference guide](../../docs/guides/h3_grid_inference.md)
describes the actual frozen-base/current-LoRA/four-table representation and
original four-step sampler. New-source CPU tests include strict common export
and public new-process generation of packed AV tokens. Native checkpoint
export now has an original-reference output-only owner bridge and an explicit
[`trainer.export` template](export.native.template.json). The media-codec ABI
remains separate; W6 output code is not substituted and the optimized backward
hold is unchanged.
