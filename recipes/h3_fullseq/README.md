# W5 — Audio-Video DMD (Full-Sequence / H3 Grid1000)

**Original protocol:** Full-sequence AV; approximate Grid1000 sigma; 4-step; G1/F5; LoRA128.

## Choose a training path

Use the [original Grid1000 lifecycle](../h3-grid-original-training.md) for
finite fresh-data training, complete-cycle checkpoint and restart.
[`training.json`](training.json) is the separate legacy optimization-reproduction
preset `h3_fullseq.cpu_staged_trim.v1`: exactly two fixed-contract cycles with
no checkpoint/resume interface.

The preset uses CPU-staged, trimmed FA3 checkpoint replay with native numerical checks.

## Inspect and run

```bash
python scripts/distillctl.py workloads candidates --workload minimax_h3_grid1000_best
```

The [native runbook](../../docs/runbooks/native_planner_workflow.md) and
[direct preset guide](../../docs/guides/native_workloads.md#native_presets) describe asset,
hardware and execution requirements. The code is in
`src/distillschedule/search/h3_action_model_adapter.py` and
`src/distillschedule/backends/training/minimax_h3/trainer_loop.py`.

## Current-Student output

The [Grid inference guide](../../docs/guides/training.md#h3_grid_inference) covers the
frozen-base/current-LoRA/four-table representation, original four-step sampler,
native checkpoint export and new-process generation of AV tokens. Use
[`export.native.template.json`](export.native.template.json) and a compatible
codec manifest for decoded audio/video. This uses the W5 sampler, not W6's.

## Results

[Performance and source records](../../docs/guides/performance_validation.md)
include the full-sequence result and its baseline. The original-reference
lifecycle and the two-cycle optimization preset retain separate configurations.
