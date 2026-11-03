# Wan2.1-1.3B Resident / async128 Tree chunk2048

Canonical profile: `wan13b_nogan_resident_best`. The profile binds its model, update contract,
source and execution policy.

```bash
cp configs/distillschedule/best_known/assets_examples/wan13b_nogan_resident_best.json /tmp/wan13b_nogan_resident_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload wan13b_nogan_resident_best --assets-config /tmp/wan13b_nogan_resident_best.assets.json --run-id check-wan13b_nogan_resident_best
python scripts/distillschedule/run_best_known_workload.py run --workload wan13b_nogan_resident_best --assets-config /tmp/wan13b_nogan_resident_best.assets.json --run-id run-wan13b_nogan_resident_best --dry-run
```

See the [performance page](../../guides/performance_validation.md) for measured results.
