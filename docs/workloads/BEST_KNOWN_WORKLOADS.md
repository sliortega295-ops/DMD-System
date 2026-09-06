# Canonical Best-Known Workloads

One registry exposes exactly five canonical profiles. The common entrypoint is:

```bash
python scripts/distillschedule/run_best_known_workload.py list
python scripts/distillctl.py workloads list
```

For any profile:

```bash
python scripts/distillschedule/run_best_known_workload.py explain --workload <id>
python scripts/distillschedule/run_best_known_workload.py validate --workload <id> --assets-config <assets.json>
python scripts/distillschedule/run_best_known_workload.py render --workload <id> --assets-config <assets.json> --output /tmp/resolved.json
python scripts/distillschedule/run_best_known_workload.py run --workload <id> --assets-config <assets.json> --run-id <id> --dry-run
```

Asset templates are in `configs/distillschedule/best_known/assets_examples/`. Models, checkpoints, traces, Grid manifests and Python environments remain explicit external assets.

| Profile | Semantics | Historical provenance | Timer boundary | Integrated GPU status |
|---|---|---:|---|---|
| `wan13b_nogan_resident_best` | `source_equivalent_numerical_drift_reported` | 26.227822833 s | world-max No-GAN Full5 G1/C5/EMA5 (single unprofiled diagnostic) | NOT_RUN_ON_INTEGRATION_BRANCH |
| `helios14b_nogan_best` | `source_equivalent_strict_correctness_pass` | 94.461699722 s | world-max Full5 G1/C5/EMA5 (single unprofiled correctness-gated trial) | NOT_RUN_ON_INTEGRATION_BRANCH |
| `minimax_h3_grid1000_best` | `approximate_grid1000` | 772.332 s | rank0 train-start through post-world16 matched-workload receipt; Student1+Fake5 (single unprofiled matched run; approximate Grid1000) | NOT_RUN_ON_INTEGRATION_BRANCH |
| `helios14b_gan_best` | `source_equivalent_strict_correctness_pass` | 125.258735931 s | world-max Full5 G1/C5/EMA1 (single unprofiled C25 trial) | NOT_RUN_ON_INTEGRATION_BRANCH |
| `livewan_crossscale_pipeline_best` | `continuous_sigma_source_equivalent_numerical_drift_reported` | 47.018683594 s | same-process cycle2 only; cycle1 excluded; G2/C10/EMA0 (one matched native observation; strict numeric REVIEW) | NOT_RUN_ON_INTEGRATION_BRANCH |

Do not compare the seconds across rows as model speed: models, topology and timer scopes differ. LiveWan 54.291817154 s is a B32-derived B16-equivalent normalization and is intentionally not the canonical actual-B16 number.

LiveWan v2 selects C34. The former C24 54.441603646 s profile is preserved in `history/livewan_c24_profile_v1.json`; its three-cycle timing is not an isolated C24-to-C34 A/B. C34 uses `--mode diagnostic-timing`, not strict `timing`; native two-cycle gate and numeric drift records remain required. See `runbooks/livewan_crossscale_pipeline_best.md`.
