# Integration and source map

Start at the target repository, not at a historical path in a conversation.
`dmd_perf.py inspect --repo /absolute/checkout` reads Git and optional DMD
metadata without importing model code. It also accepts an external Git repository.
For a source archive without Git, manually record its file/config hashes.

## Existing training repository

Reuse its command, environment, resource scheduler and checker. Record an explicit
binding between its fields and the workload contract: full cycle, role update
counts, warmup, timer boundaries, rank aggregation, input identity, gradient/EMA
state and continuation. Missing fields are integration work; do not invent PASS.
Use the project's run root, or a local ignored `.dmd-optimizer/` directory if
there is no convention. Keep source changes in a reviewable branch/worktree.

No framework port is needed for optimization. Add a DMD adapter only if the user
wants the unified lifecycle and the existing adapter/recipe fits.

## DMD-System source anchors

| Need | Repository-relative source |
| --- | --- |
| Current workload catalog | `src/distillschedule/workloads/catalog.json` |
| Support scope | `docs/guides/support_matrix.md` |
| Training command and recovery | `docs/guides/training.md`, `recipes/` |
| Physical capability seams | `docs/guides/optimization_capabilities.md` |
| Existing implementation registry | `src/distillschedule/search/implementation_library.py` |
| Saved-plan workflow | `docs/runbooks/native_planner_workflow.md` |
| Ownership and extension | `docs/architecture/framework.md`, `docs/guides/extensions.md` |
| Historical evidence interpretation | `docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md` |

At the packaged source snapshot there are six preserved native workloads:
W1 Wan No-GAN, W2 Helios No-GAN, W3 Helios GAN, W4 LiveWan cross-scale,
W5 H3 full-sequence approximate Grid1000, W6 H3 autoregressive continuous sigma.
Additional model integrations have their own support lanes. Discover current
capabilities from code; do not hardcode six as the limit.

`dmd_knowledge.py inventory --repo /checkout --store /project/knowledge`
extracts the literal registry without executing its imports. Entries have source
hashes and **unknown** shape/dtype/grad/topology constraints until code inspection.
Registry membership does not prove portability, current correctness or speed.
Resolve fast-path guards, third-party rights and tests before adapting an entry.

Native dry runs, CPU tests and saved-plan validity do not establish real GPU
effect or distributed correctness. Preserve the runner's effect/version receipts
and its actual acceptance status.
