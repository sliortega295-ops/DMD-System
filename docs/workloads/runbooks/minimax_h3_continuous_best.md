# MiniMax-H3 shared backbone / continuous H3 / staged FA3 replay

Canonical profile: `minimax_h3_continuous_best`. The profile binds its model, update contract,
source and execution policy.

```bash
cp configs/distillschedule/best_known/assets_examples/minimax_h3_continuous_best.json /tmp/minimax_h3_continuous_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload minimax_h3_continuous_best --assets-config /tmp/minimax_h3_continuous_best.assets.json --run-id check-minimax_h3_continuous_best
python scripts/distillschedule/run_best_known_workload.py run --workload minimax_h3_continuous_best --assets-config /tmp/minimax_h3_continuous_best.assets.json --run-id run-minimax_h3_continuous_best --dry-run
```

See the [performance page](../../guides/performance_validation.md) for measured results.
