# LiveWan C34: canonical persistent 8G + 8Score pipeline

Profile `livewan_crossscale_pipeline_best`, version 2. This is the tested C34
implementation integrated into the existing registry/worker, not another runtime.
The other four canonical profiles are unchanged.

## Source and configuration

The profile binds the reviewed worker and pinned LiveWan/Wan dependencies.
Use its assets template and source validation before launching. C34 uses
`diagnostic-timing`; the native trajectory comparison and timing gate bind the
same source, input and seed.

## What C34 contains

- Original actual B16, two B1 Generator waves, Critic physical B2, G1/C5 per cycle,
  EMA0, continuous sigma, unchanged input/noise/seed (20260830), losses and commits.
- Frozen Score RMSNorm and exact Score modulation kernels.
- Generator no-grad autocast weight-cache lifetime bounded to one rollout.
- Generator no-grad RMS cast/square, modulation and residual-gate kernels.
- Row-major internal no-grad residual output; public output contract retained.
- Critic exact FlashAttention output checkpoint cache on GPU, not CPU offload.
- All corresponding source hashes, runtime counters and fail-closed validators.

## Portable setup and CPU-only checks

Use `configs/distillschedule/best_known/assets_examples/livewan_crossscale_pipeline_best.json`
and replace every asset path/identity SHA. Both nodes must use the exact integrated
HEAD and the same verified Python/FlashAttention overlay. The prepared source,
bootstrap and trace receipts must agree. A **two-cycle** trace/bootstrap is
required, not the old three-cycle C24 bundle. Existing asset builders now accept
2 cycles. Do not change shared environments or regenerate random inputs silently.

```bash
python scripts/distillctl.py workloads list
python scripts/distillctl.py workloads validate \
  --workload livewan_crossscale_pipeline_best --assets-config "$ASSETS"
python scripts/distillctl.py workloads run \
  --workload livewan_crossscale_pipeline_best --assets-config "$ASSETS" \
  --mode correctness --run-id c34-correctness-unique --dry-run
```

`validate` is the CPU CLI action; there is no `run --mode validate` for this
2-cycle profile. The legacy worker's validate-mode uses its 3-cycle template.
No SSH/GPU is invoked by list/explain/validate/render/dry-run.

## Correctness and native timing

Remove `--dry-run` only after resource/source/asset acceptance. Canonical
`correctness` maps to the existing **two-cycle trajectory** mode. Native output
is summarized through `summarize_livewan_trajectory_world16.py`; the command
collects losses/state/census rather than presenting its wall as performance.

A fresh same-HEAD baseline trajectory is still required. Run the retained
`run_livewan_crossscale_world16.py` with `--execution-mode baseline
--pipeline-variant b1 --run-kind trajectory --trajectory-cycles 2` and matching
assets/seed, then use `compare_livewan_two_cycle_trajectories.py`. Do not bind old
4c994ed receipts to a new integration HEAD.

Build the unchanged matched native gate, optionally binding the canonical
coordinator receipt from the pipeline correctness run:

```bash
PYTHONPATH=src:. python scripts/distillschedule/build_livewan_two_cycle_timing_gate.py \
  --baseline-trajectory-summary "$BASELINE_SUMMARY" \
  --pipeline-trajectory-summary "$PIPELINE_SUMMARY" \
  --trajectory-comparison "$COMPARISON" \
  --pipeline-variant c34-continuous-b16-residual-layout \
  --expected-head "$(git rev-parse HEAD)" \
  --integration-receipt "$PIPELINE_RUN/coordinator-contract.json" \
  --output "$NEW_BOUND_GATE"
```

Put `$NEW_BOUND_GATE` and its SHA256 in the assets JSON `correctness` binding.
The immutable pipeline arm ID is `livewan_crossscale_pipeline_best-v2` in both
trajectory and timing. Run IDs remain separate and must never be reused.

```bash
python scripts/distillctl.py workloads run \
  --workload livewan_crossscale_pipeline_best --assets-config "$ASSETS" \
  --mode diagnostic-timing --run-id c34-native-c2-unique --dry-run
```

This maps to the tested `two-cycle-timing` worker and
`summarize_livewan_two_cycle_timing_world16.py`, not the legacy 3-cycle summarizer.
Cycle1 is warmup. Only cycle2 completion-endpoint delta is comparable; never
average cycles or compare against a differently initialized/diagnostic baseline.
Native source/trace/optimizer/replay/census/teardown gates remain mandatory.
