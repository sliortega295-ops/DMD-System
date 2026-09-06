# LiveWan C34: canonical persistent 8G + 8Score pipeline

Profile `livewan_crossscale_pipeline_best`, version 2. This is the tested C34
implementation integrated into the existing registry/worker, not another runtime.
The other four canonical profiles are unchanged.

## Frozen source and historical evidence

Source/tested HEAD: `4c994edca7d83ff553cab6025d8b93b279cd1634`
(`agent/livewan-steady130-20260905`). LiveWan and Wan submodules remain pinned at
`1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5` and
`9737cba9c1c3c4d04b33fcad41c111989865d315`.

| Historical arm | Cycle1, excluded | Cycle2, native |
|---|---:|---:|
| Same-head world16 FSDP2 baseline | 79.625242188 s | 62.427882812 s |
| C34 persistent 8G+8Score | 51.383207031 s | 47.018683594 s |

Single matched observation: 1.327725024x. Not a median, not an integration result.
16/16 ranks, 32 rank-cycle receipts, G2/C10/EMA0 and OOM delta 0. Strict numerical
comparison remains **REVIEW**, not strict parity PASS. The source's existing
`REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW` admission is unchanged. This profile
therefore exposes `diagnostic-timing`, not strict `timing`.
Evidence: `../evidence/livewan_c34_20260906.json`.
C24 remains in `../history/livewan_c24_profile_v1.json` and its original typed plan.

## What C34 contains

- Original actual B16, two B1 Generator waves, Critic physical B2, G1/C5 per cycle,
  EMA0, continuous sigma, unchanged input/noise/seed (20260830), losses and commits.
- Frozen Score RMSNorm and exact Score modulation kernels.
- Generator no-grad autocast weight-cache lifetime bounded to one rollout.
- Generator no-grad RMS cast/square, modulation and residual-gate kernels.
- Row-major internal no-grad residual output; public output contract retained.
- Critic exact FlashAttention output checkpoint cache on GPU, not CPU offload.
- All corresponding source hashes, runtime counters and fail-closed validators.

Peak allocated in that historical run: Generator 35.152 GiB, Score 18.200 GiB.
Allocator/driver headroom must still be checked live; OOM delta 0 is not proof of
large safety margin on every device. No additional KV, batching or HSDP change.

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

## Correctness and native timing (NOT RUN in this integration)

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

**Integrated GPU/model execution and performance reproduction: NOT_RUN.**
