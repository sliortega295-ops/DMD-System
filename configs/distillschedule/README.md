# Typed DistillSchedule plans

This directory contains composable runtime plans rather than copied experiment launchers.

- `fragments/`: reusable model, shape, DMD schedule, and lifecycle fragments.
- `plans/`: complete, fail-closed `ExecutionPlan` documents.

Resolve and inspect a plan with:

```bash
python scripts/distillctl.py validate configs/distillschedule/plans/helios_14b_384x640_full5.json
python scripts/distillctl.py show configs/distillschedule/plans/helios_14b_384x640_full5.json
python scripts/distillctl.py run configs/distillschedule/plans/helios_14b_384x640_full5.json --dry-run
```

The Helios plan compiles to a generated `resolved_helios_config.yaml` and the existing
source-faithful `train_helios.py` path. New model families should add an adapter under
`src/distillschedule/adapters/` instead of adding another branch to `train_helios.py`.

Paths that differ across machines should use the `${ENV:NAME:-default}` form. The two
important remote overrides in the example are `DISTILLSCHEDULE_TRACE_BUNDLE` and
`DISTILLSCHEDULE_CHECKPOINT`.
