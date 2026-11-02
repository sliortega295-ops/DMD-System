# Retired documentation and helpers

The first group below contains unreferenced fixed-date snapshots. The second
group consolidates earlier acceptance narratives into the maintained guides;
incoming guide and packaging references were updated at retirement.
They are not current operating instructions; the maintained [framework architecture](framework.md),
[current guides](../guides/support_matrix.md) and immutable machine receipts remain authoritative.

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

## Consolidated acceptance narratives (2026-11-02)

Current instructions and results are maintained in the [guides](../guides/support_matrix.md).
The following duplicate/process reports are recoverable from
[main `3ca90fa9`](https://github.com/sliortega295-ops/DMD-System/tree/3ca90fa91ca8e8bf433474c1acf3160ee1430635).
Machine-readable evidence and historical source indexes remain in the repository.

| Retired path | SHA-256 |
| --- | --- |
| `docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md` | `4da6d516b8fa2a4539fce96a825d7861e3bb652e4f59a5c53cb69d94989e8aab` |
| `docs/status/FRAMEWORK_CPU_ACCEPTANCE_20261021.md` | `e7e5aa0a3fbd672feb202a5a87cfade2df55b4b89104151198bf9baf36c90f1a` |
| `docs/status/FRAMEWORK_LIVEWAN_REFERENCE_20261021.md` | `069ddce3fca685138890bbbd4c8e26e83a315ed10faa34ef59c00127d7e85a21` |
| `docs/status/FRAMEWORK_MAIN_INTEGRATION_20261030.md` | `026a49a776ecec1ace27056d25cabaf6a08c4e918b26db42c0ba1168bded0c52` |
| `docs/status/FRAMEWORK_MODEL_EXPANSION_CPU_20261023.md` | `c12a9d94877839069bf6407c739b35a54843d548fc62332758c96b9e399f4724` |
| `docs/status/FRAMEWORK_NATIVE_LIFECYCLE_20261021.md` | `543e1e43482b6b12f3e2b32fb8218b7249ca3de67c319daa6eff3b3840f89725` |
| `docs/status/FRAMEWORK_ORIGINAL_CPU_ACCEPTANCE_20261022.md` | `35cc9d4b20e1f30a5f0c34b5ae6b54c4e5da4ef7dbd2c54555c96ea74225372b` |
| `docs/status/H3_CURRENT_STUDENT_CPU_ACCEPTANCE_20261022.md` | `4b84367f19009e84cd9aa9cfac43471ccdeb8c9d821435cfd38ccfdd42354d1e` |
| `docs/status/PLANNER_SCHEDULER_INTEGRATION_20261015.md` | `12f02b158e582e3e595d6c419bdf5cf289a41cae03e78c6b4fff2ea500a78808` |
| `docs/status/WAN_ORIGINAL_CPU_ACCEPTANCE_20261023.md` | `ba36747a20a75c958034b8c2fa1f64f54691f55c004c989b7cbf600bb59596e4` |
| `docs/status/WAN_PARAMETERIZED_BRIDGE_20261012.md` | `3e0c1a5937e9678692e32c87e0f5415f4ddfdf9ffc9a19270fb486dc9252997a` |
| `docs/guides/validation_records_20261030.md` | `8f06613d6f324ae978d94c9bda9b0c9e1e964ca68afd67b03493e89423c800c2` |
