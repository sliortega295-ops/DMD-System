# Current workload recipes

Use `distillctl workloads list` for W1–W6 and `describe --workload Wn` for exact source identities. Each subdirectory points to the existing native CLI and contract. No recipe implements a second training loop or supplies old wall times as new calibration costs.

The six default `training.json` files explicitly select [versioned optimized
manual presets](../docs/guides/native_presets.md), with no user-created saved
plan. These are two-cycle research bindings, not continuous training; W5
retains its unresolved numerical FAIL. Reference and saved-plan paths remain
separate.

The machine-readable source of truth is `src/distillschedule/workloads/catalog.json`, included in wheels. A recipe is a guide, not a launch permission or a passing numerical certificate. Five old locked JSON profiles remain explicitly available with `workloads list --legacy`.


Separately versioned local CPU references preserve the actual original
[W1 No-GAN Full5](wan_ar/CPU_REFERENCE.md),
[Helios GAN](helios_gan/REFERENCE_TRAINING_CPU.md) and
[H3 cached previous/current AV](h3_ar/CPU_REFERENCE.md) models and training
helpers through Program/common checkpoints. They are not renamed benchmark
profiles, full pretrained/GPU qualification, or substitutes for the native
selected-plan recipe. The [support matrix](../docs/guides/support_matrix.md)
separates each operation and evidence level.
