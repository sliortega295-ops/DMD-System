# Helios14B No-GAN node-HSDP / local FA2 / selective replay

Canonical profile: `helios14b_nogan_best`. Semantics: `source_equivalent_strict_correctness_pass`. Historical result: `94.461699722 s` with scope `world-max Full5 G1/C5/EMA5`; this is provenance only.

```bash
cp configs/distillschedule/best_known/assets_examples/helios14b_nogan_best.json /tmp/helios14b_nogan_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload helios14b_nogan_best --assets-config /tmp/helios14b_nogan_best.assets.json --run-id check-helios14b_nogan_best
python scripts/distillschedule/run_best_known_workload.py run --workload helios14b_nogan_best --assets-config /tmp/helios14b_nogan_best.assets.json --run-id run-helios14b_nogan_best --dry-run
```

GPU revalidation of the integrated code: **NOT_RUN_ON_INTEGRATION_BRANCH**.
