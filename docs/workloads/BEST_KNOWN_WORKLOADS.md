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

| Profile | Execution policy | Workload guide |
| --- | --- | --- |
| `wan13b_nogan_resident_best` | Resident, asynchronous gradient reduction, chunk2048 | [Wan](runbooks/wan13b_nogan_resident_best.md) |
| `helios14b_nogan_best` | No-GAN sharding and fused modulation | [Helios No-GAN](runbooks/helios14b_nogan_best.md) |
| `helios14b_gan_best` | GAN calls, retained rollout weights and replay | [Helios GAN](runbooks/helios14b_gan_best.md) |
| `livewan_crossscale_pipeline_best` | C34 persistent 8G + 8Score pipeline | [LiveWan C34](runbooks/livewan_crossscale_pipeline_best.md) |
| `minimax_h3_grid1000_best` | Shared backbone, Grid1000 and staged attention cache | [H3 full-sequence](runbooks/minimax_h3_grid1000_best.md) |

These are versioned compatibility profiles. The current six-workload entry and
public continuous training are described in [native workloads](../guides/native_workloads.md).
C34 remains distinct from the C51 recipe and W6 is a separate AR workload.
Measurements are collected on the [performance page](../guides/performance_validation.md).

## Profile-guided search

The opt-in [registered-plan search](profile_guided_search.md) reuses this compiler and dispatcher. It preserves canonical defaults and separates measured search results from historical evidence.
