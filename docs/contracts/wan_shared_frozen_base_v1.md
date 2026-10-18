# Wan equal-frozen-base Resident sharing, standalone experimental v1

Two role models with the SAME explicit frozen checkpoint identity may alias
byte-equal, shape/dtype/device/stride-equal frozen Parameter storages at setup.
All eligible comparisons complete before any aliasing. Preserve both Parameter
objects, independent FP32 LoRA trainables, optimizer/EMA, buffers, role mode and
the original forward/backward graph. Not W+BA or BF16 LoRA weight fusion.

Do not apply to Helios G/Score's different frozen checkpoints, FSDP/DTensor,
meta parameters, merged adapters or already-shared role trainables. New helper
has NO default caller, profile, search or scheduler installation. Real
model identities and raw storage-deduplicated savings must be measured before
claiming capacity. Pointer/version audit is metadata-only and does not detect
arbitrary untracked `.data` writes; source must forbid base writes/merge.

Sharing by itself is a capacity enabler, NOT a speed claim. A later native
candidate must bind its setup/runtime receipt, exercise original graph/input/
sample/version/update/numerical gates and account for any added packing or
transfer. Extra retained blocks/batching need their own admitted experiment;
do not silently enable them or inherit old GPU status.

## Explicit native manual experiment (not registered in the planner)

`distillschedule.runtime.wan13b_shared_base_experiment` requires `--execute`
and an immutable SHA-bound packet. It reuses the original manual worker, two
native cycles, highwater collector and full-state exporter. Control is the
current private-base retainC26-29 path. The only admitted child shares frozen
G/C storage before ANY graph capture and retains C24-29 using the existing
`tail6.v1` original-autograd mechanism. This is a dependent capacity-enabler
combination, not independent additive savings and not an assertion of speed.

Original Critic graph construction snapshots addresses before any capture.
The experimental setup checks the controller is entirely cold (zero captures,
replays, backward replays, references and empty cache), verifies its old address
contract, aliases the equal frozen storages, then rebinds that cold snapshot
exactly once. Every subsequent original pointer/capture guard remains enabled.
No graph may predate the alias. Audits, equality scans and allocation metadata
are outside native cycle timers. Independent trainables/optimizer/EMA/buffers
and all canonical wrappers, losses, counts, RNG and policies are unchanged.

Physical storage equality on a real CPU checkpoint is not GPU correctness or
HBM evidence. Native eligibility still requires all 16 original receipts,
versions/calls/finite, original allocated-capacity reserve, actual graph and
retention census, frozen storage auditing and matched full-state comparison.
