# Wan equal-frozen-base Resident sharing, standalone experimental v1

Two role models with the SAME explicit frozen checkpoint identity may alias
byte-equal, shape/dtype/device/stride-equal frozen Parameter storages at setup.
All eligible comparisons complete before any aliasing. Preserve both Parameter
objects, independent FP32 LoRA trainables, optimizer/EMA, buffers, role mode and
the original forward/backward graph. Not W+BA or BF16 LoRA weight fusion.

Do not apply to Helios G/Score's different frozen checkpoints, FSDP/DTensor,
meta parameters, merged adapters or already-shared role trainables. New helper
has NO default caller, CLI, profile, search or scheduler installation. Real
model identities and raw storage-deduplicated savings must be measured before
claiming capacity. Pointer/version audit is metadata-only and does not detect
arbitrary untracked `.data` writes; source must forbid base writes/merge.

Sharing by itself is a capacity enabler, NOT a speed claim. A later native
candidate must bind its setup/runtime receipt, exercise original graph/input/
sample/version/update/numerical gates and account for any added packing or
transfer. Extra retained blocks/batching need their own admitted experiment;
do not silently enable them or inherit old GPU status.
