# Helios14B GAN world16 FSDP2 / C25

Canonical profile: `helios14b_gan_best`. The profile binds its model, update contract,
source and execution policy.

```bash
cp configs/distillschedule/best_known/assets_examples/helios14b_gan_best.json /tmp/helios14b_gan_best.assets.json
# Replace placeholder paths/SHA256 values.
python scripts/distillschedule/run_best_known_workload.py validate --workload helios14b_gan_best --assets-config /tmp/helios14b_gan_best.assets.json --run-id check-helios14b_gan_best
python scripts/distillschedule/run_best_known_workload.py run --workload helios14b_gan_best --assets-config /tmp/helios14b_gan_best.assets.json --run-id run-helios14b_gan_best --dry-run
```

See the [performance page](../../guides/performance_validation.md) for measured results.
