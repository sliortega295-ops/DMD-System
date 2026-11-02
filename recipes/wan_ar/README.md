# W1 — Wan1.3B autoregressive DMD

**Original protocol:** Helios hierarchical AR; No-GAN; G1/C5/EMA5.

The optimized bundle combines Resident execution, chunk2048, LoRA/norm fusion,
Generator rollout graphs and cast reuse, and Critic gradient graphs.

## Choose a training path

- [`training.json`](training.json): the original two-cycle fixed-input preset
  `wan_ar.incumbent21a.v1`, for quick comparisons and benchmark reproduction.
- [Incumbent stream](INCUMBENT_STREAM.md): the full bundle over 2..100 fresh
  cohorts, with complete-cycle native checkpoint and restart. Graph signatures
  must match the supported input shapes.
- [CPU reference](CPU_REFERENCE.md): original model and update logic through
  the common Program lifecycle, with a single source-rank CPU fixture.

Supply the recipe's model/bootstrap/trace assets and an output directory.
Use the continuation recipe for long training; the two-cycle preset itself
has no checkpoint/resume interface. The native bundle keeps retain4 disabled.

## Inspect and run

```bash
python scripts/distillctl.py workloads native validate --task configs/distillschedule/original_wan/full5_reference.json
```

The [native runbook](../../docs/runbooks/wan_compute_state_planning_v1.md) covers
manual and saved-plan execution. `--include-incumbent-bundle --include-critic-retention`
exposes the relevant candidates in the compatible native/Triton environment.
See [native output](NATIVE_OUTPUT.md) for current-Student export and Wan2.1 VAE decoding.

## Implementation and results

The entrypoints are `src/distillschedule/strategies/wan_cli.py`, `wan_worker.py`
and `wan_planning.py`. [Direct presets](../../docs/guides/native_presets.md)
map them to the public configuration. A100 and H100 use their corresponding
hardware configuration.

[Current performance and validation](../../docs/guides/performance_validation.md)
explains Cycle 2 screening and subsequent longer-cycle validation. Original
per-run measurements remain in `docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md`
and `artifacts/publications/20261019/`.
