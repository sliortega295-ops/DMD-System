# W2 — Helios14B No-GAN

**Original protocol:** Hierarchical AR; G1/C5/EMA5.

The SafeBundle combines node-HSDP, chunk512, rollout window5, normalization and
LoRA-B fusion, and checkpoint-result reuse.

## Choose a training path

- [`training.json`](training.json): `helios_nogan.safe_ed8.v1`, the original
  two-cycle warm-state comparison preset.
- [Safe stream](SAFE_STREAM.md): the optimized fresh-cohort continuation path
  with complete-cycle checkpoint and restart.
- [Native-history continuation](NATIVE_HISTORY.md): original per-rank history
  and amplified rollout exits with aligned FSDP2 phases.
- [Direct reference](REFERENCE_TRAINING.md): original reference execution from
  a pinned Step10 G2/C10/EMA10 package.
- [CPU reference](CPU_REFERENCE.md): original update logic and state recovery
  on a single source-rank fixture.

Use the matching prepared inputs, warm bootstrap and capacity configuration.
The two-cycle preset is for comparison; use the stream path for long training.

## Inspect and run

```bash
python scripts/distillctl.py workloads candidates --workload helios14b_nogan_best --include-helios-safe-incumbent
```

The [native runbook](../../docs/runbooks/native_planner_workflow.md) covers
manual and saved-plan execution. [Native Student output](NATIVE_OUTPUT.md)
covers checkpoint export and new-prompt decoded video.

## Implementation and results

The adapter entrypoints are `src/distillschedule/search/helios_model_adapter.py`
and `workflow_cli.py`; [direct presets](../../docs/guides/native_presets.md)
identify their implementation and input bindings.

See [current performance and validation](../../docs/guides/performance_validation.md).
The older locked 94-second profile and subsequent partial-retention experiments
retain their separate source/configuration identities in the research records.
