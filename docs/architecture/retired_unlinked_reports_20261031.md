# Retired unlinked documentation snapshots (2026-10-31)

These fixed-date snapshots had no inbound references from the current checked-in source,
guide, Skill index, test, packaging manifest or workflow when reviewed.
They are not current operating instructions; the maintained [framework architecture](framework.md),
[current guides](../README.md) and immutable machine receipts remain authoritative.

The original bytes remain recoverable from Git at
[pre-cleanup main `447f261`](https://github.com/sliortega295-ops/DMD-System/tree/447f2613631a7271a0a4b294bc838df6ecdacd5e).
Removal from the current documentation tree is not a reinterpretation of their original
CPU/GPU validation status or training evidence.

| Retired path | SHA-256 |
| --- | --- |
| `docs/status/FRAMEWORK_LIVEWAN_INPUTS_20261021.md` | `72cb7bbfb862180949e25c1ebe0f812bbd46124f656755ffb3fe316c2d899d0c` |
| `docs/status/FRAMEWORK_LIVEWAN_STATE_20261021.md` | `1e25126daefa3d8e1f9fb68a56e871edbc36ab7bbfa984afeab97e15f18cc1eb` |
| `docs/status/HELIOS_NOGAN_CPU_ACCEPTANCE_20261023.md` | `6c256d9b8568b42d4b2c5cd96d469aaf73ebe8be19457cfa740e467b3e64136e` |
| `docs/status/original_h3_ar_native_20261017.md` | `291f6bc8dd26f44859cdc1dc8b9140f41c228f891d0884cddf85102e1298224b` |
| `docs/architecture/refactor_v1_status.json` | `4c9309d450e6cb1f873a88b810de9e97f5841ec4ee10304421a1b382beac9d4e` |
| `docs/architecture/v0.2_source_inventory.md` | `59dd0619135b46f4e0c95df36c7c67b6b465a5da1d759b59c3bda2ddb19d875b` |

The separate Route-A/Route-B retirement and its source inventory remain
documented in [legacy route retirement](legacy_route_retirement.md).
Reports linked by the public Skill experience index, currently maintained recipe docs,
frozen source maps, and machine-readable result receipts were not removed.

## Unreferenced standalone helpers

The following legacy, developer-only Python files had no imported symbol,
script caller, documentation link, CI reference or source-lock dependency in
the audited current tree. Their bytes remain in [pre-cleanup Git history](https://github.com/sliortega295-ops/DMD-System/tree/447f2613631a7271a0a4b294bc838df6ecdacd5e).
This does **not** retire shared Critic training, ZeRO/FSDP, checkpoint comparison
as a technique or any public runtime API.

| Former path | SHA-256 |
| --- | --- |
| `scripts/distillschedule/critic_only_protocol.py` | `9a46b3a87c6367e937c9c877befde4583beea10fb41b064911e2ac44789b18f9` |
| `scripts/distillschedule/compare_zero1_checkpoints.py` | `702ee0db15f57cbaaa566f6b1288102f83eba7ef0c2b3dbf7e4e92a24933e9f1` |
