# MiniMax-H3 shared backbone / Grid1000 / staged FA3 replay

Canonical profile: `minimax_h3_grid1000_best`. Semantics: `approximate_grid1000`. Historical result: `772.332 s` with scope `rank0 train-start through post-world16 matched-workload receipt; Student1+Fake5`; this is provenance only.

```bash
cp configs/distillschedule/best_known/assets_examples/minimax_h3_grid1000_best.json /tmp/minimax_h3_grid1000_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload minimax_h3_grid1000_best --assets-config /tmp/minimax_h3_grid1000_best.assets.json --run-id check-minimax_h3_grid1000_best
python scripts/distillschedule/run_best_known_workload.py run --workload minimax_h3_grid1000_best --assets-config /tmp/minimax_h3_grid1000_best.assets.json --run-id run-minimax_h3_grid1000_best --dry-run
```

GPU revalidation of the integrated code: **NOT_RUN_ON_INTEGRATION_BRANCH**.
