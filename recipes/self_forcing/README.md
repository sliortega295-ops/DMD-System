# W4 — Self-Forcing DMD / LiveWan C51

**Original protocol:** 14B teacher -> 1.3B student; continuous sigma; G1/C5/EMA0.

This is the cross-scale LiveWan workflow, separate from
[official Self-Forcing](../self_forcing_official/README.md).
The full C51 implementation uses an 8G+8Score dual-wave pipeline.

## Choose a training path

- [`training.json`](training.json): `self_forcing.c51.v1`, the original
  two-cycle C51 comparison preset.
- [C51 continuation](C51_CONTINUATION.md): full-C51 fresh-cohort training,
  complete-cycle checkpoint and two-pool recovery.
- [Original reference](REFERENCE_TRAINING.md): 2..64 fresh teacher cohorts
  from genuine zero-step assets, using the original reference owners.
- [CPU Program](CPU_PUBLIC.md): single-device FP32 original-model lifecycle
  and current-Student output.

Select C51 explicitly: the older five-profile compatibility registry selects
C34. Use the continuation recipe when checkpoint/resume is needed.

## Inspect and run

```bash
python scripts/distillctl.py workloads candidates --workload livewan_crossscale_pipeline_best --include-livewan-c51
```

The [C51 runbook](../../docs/runbooks/livewan_c51_planner_v1.md) describes manual
and saved-plan execution. The [native output guide](../../docs/guides/livewan_native_output.md)
uses the current Student, original streamer and Wan VAE for new inputs.

## Implementation and results

The implementation is in `src/distillschedule/search/livewan_implementation.py`
and the native worker (source checkout: `src/distillschedule/runtime/livewan_crossscale_worker.py`).
See [direct presets](../../docs/guides/native_presets.md) and
[current performance and validation](../../docs/guides/performance_validation.md).
Historical C34, C51 and A100/H100 measurements retain their own configuration
and timing definitions in the research records.
