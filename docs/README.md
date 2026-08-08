# Documentation map

DMD-System separates durable design, frozen experiment contracts, operational procedures and evidence status. Do not use a historical experiment note as the source of truth for a current launch.

## Architecture

- [`architecture/refactor_v1.md`](architecture/refactor_v1.md): typed runtime, adapter boundary and strangler migration order.
- [`architecture/v0.2_source_inventory.md`](architecture/v0.2_source_inventory.md): source versus raw-evidence retention policy.
- [`architecture/baai_server_adaptation_20260808.md`](architecture/baai_server_adaptation_20260808.md): historical CPU/source adaptation snapshot; not the live world16 status page.

## Frozen contracts

- [`contracts/routea_full5_world16.md`](contracts/routea_full5_world16.md): the exact Route-A Full5 world16 workload and prohibited semantic changes.

A contract answers **what must remain equivalent**. It must not contain mutable run IDs, launcher outcomes or temporary recovery steps.

## Runbooks

- [`runbooks/baai_routea_world16.md`](runbooks/baai_routea_world16.md): environment, preflight, two-host worker launch, topology probe and summarization.

A runbook answers **how to execute the current implementation**. It may change when tooling changes, but it may not silently change the frozen contract.

## Status and experiments

- [`status/routea_world16.md`](status/routea_world16.md): authoritative evidence ledger and current open gaps.
- [`experiments/routea_world16_typed_revalidation_20260808.md`](experiments/routea_world16_typed_revalidation_20260808.md): detailed preregistration and attempt history for the August 8 integration cycle.

A status page answers **what has actually been measured**. `Running`, `Partial`, `Failed` and `Not Run` are first-class outcomes; none may be rewritten as a successful result.

## Evidence

- [`../artifacts/manifests/README.md`](../artifacts/manifests/README.md): curated manifest policy, status labels and raw-evidence boundary.
- [`../profiling/nsight/README.md`](../profiling/nsight/README.md): diagnostic profiling conventions and formal-claim boundary.
