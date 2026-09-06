# LiveWan C24 continuous actual-B16 cross-scale pipeline

Canonical profile: `livewan_crossscale_pipeline_best`. Semantics: `continuous_sigma_source_equivalent`. Historical result: `54.441603646 s` with scope `actual-B16 measured middle Full5 of three continuous cycles; G3/C15/EMA0 final`; this is provenance only.

```bash
cp configs/distillschedule/best_known/assets_examples/livewan_crossscale_pipeline_best.json /tmp/livewan_crossscale_pipeline_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload livewan_crossscale_pipeline_best --assets-config /tmp/livewan_crossscale_pipeline_best.assets.json --run-id check-livewan_crossscale_pipeline_best
python scripts/distillschedule/run_best_known_workload.py run --workload livewan_crossscale_pipeline_best --assets-config /tmp/livewan_crossscale_pipeline_best.assets.json --run-id run-livewan_crossscale_pipeline_best --dry-run
```

GPU revalidation of the integrated code: **NOT_RUN_ON_INTEGRATION_BRANCH**.
