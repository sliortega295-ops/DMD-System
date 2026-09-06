# Wan2.1-1.3B Resident / async128 Tree chunk2048

Canonical profile: `wan13b_nogan_resident_best`. Semantics: `source_equivalent_numerical_drift_reported`. Historical result: `26.227822833 s` with scope `world-max No-GAN Full5 G1/C5/EMA5`; this is provenance only.

```bash
cp configs/distillschedule/best_known/assets_examples/wan13b_nogan_resident_best.json /tmp/wan13b_nogan_resident_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload wan13b_nogan_resident_best --assets-config /tmp/wan13b_nogan_resident_best.assets.json --run-id check-wan13b_nogan_resident_best
python scripts/distillschedule/run_best_known_workload.py run --workload wan13b_nogan_resident_best --assets-config /tmp/wan13b_nogan_resident_best.assets.json --run-id run-wan13b_nogan_resident_best --dry-run
```

GPU revalidation of the integrated code: **NOT_RUN_ON_INTEGRATION_BRANCH**.
