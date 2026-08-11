# Documentation map

DMD-System separates durable design, frozen experiment contracts, operational procedures and evidence status. Do not use a historical experiment note as the source of truth for a current launch.

## Architecture

- [`architecture/refactor_v1.md`](architecture/refactor_v1.md): typed runtime, adapter boundary and strangler migration order.
- [`architecture/v0.2_source_inventory.md`](architecture/v0.2_source_inventory.md): source versus raw-evidence retention policy.
- [`architecture/baai_server_adaptation_20260808.md`](architecture/baai_server_adaptation_20260808.md): historical CPU/source adaptation snapshot; not the live world16 status page.

## Frozen contracts

- [`contracts/experiment_names.md`](contracts/experiment_names.md): canonical experiment IDs; `world16` alone is never a result name.
- [`contracts/ra16_f5_pr4.md`](contracts/ra16_f5_pr4.md): enhanced Route-A Full5 lifecycle contract for the current partial-streaming target.
- [`contracts/ra16_f5_pr8.md`](contracts/ra16_f5_pr8.md): bounded g8 successor contract after the PR4 admission failure.
- [`contracts/ra16_f5_pr12.md`](contracts/ra16_f5_pr12.md): one-attempt g12 successor contract after the PR8 admission failure.
- [`contracts/routea_full5_world16.md`](contracts/routea_full5_world16.md): the exact Route-A Full5 world16 workload and prohibited semantic changes.

A contract answers **what must remain equivalent**. It must not contain mutable run IDs, launcher outcomes or temporary recovery steps.

## Runbooks

- [`runbooks/baai_routea_world16.md`](runbooks/baai_routea_world16.md): environment, preflight, two-host worker launch, topology probe and summarization.
- [`runbooks/baai_ra16_f5_pr4.md`](runbooks/baai_ra16_f5_pr4.md): current partial-streaming Full5 functional-gate command and launch guards.
- [`runbooks/baai_no_gan_suite.md`](runbooks/baai_no_gan_suite.md): central suite entrypoint, evidence reuse, external executor seams and `Not Run` boundaries.

A runbook answers **how to execute the current implementation**. It may change when tooling changes, but it may not silently change the frozen contract.

## Status and experiments

- [`status/ra16_f5_pr4.md`](status/ra16_f5_pr4.md): current enhanced Route-A target and its explicit `Not Run` boundary.
- [`status/ra16_f5_pr8.md`](status/ra16_f5_pr8.md): current PR8 admission and per-rank memory-analysis status.
- [`status/ra16_f5_pr12.md`](status/ra16_f5_pr12.md): current g12 preregistration and admission status.
- [`status/routea_world16.md`](status/routea_world16.md): authoritative evidence ledger and current open gaps.
- [`experiments/routea_world16_typed_revalidation_20260808.md`](experiments/routea_world16_typed_revalidation_20260808.md): detailed preregistration and attempt history for the August 8 integration cycle.

A status page answers **what has actually been measured**. `Running`, `Partial`, `Failed` and `Not Run` are first-class outcomes; none may be rewritten as a successful result.

## Evidence

- [`../artifacts/manifests/README.md`](../artifacts/manifests/README.md): curated manifest policy, status labels and raw-evidence boundary.
- [`../profiling/nsight/README.md`](../profiling/nsight/README.md): diagnostic profiling conventions and formal-claim boundary.
