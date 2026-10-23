# Bounded planner/native workflow integration — 2026-10-15

**Pending cluster validation.** No GPU work, push or merge was performed for
this integration. CPU receipt fixtures are not measurements or GPU correctness.

## Source and scope

The GitHub refresh before implementation confirmed main
`e69f4bf8b8bb64f54542fea74c452b53e090b526` and open, draft, unmerged PR #34
`0abc0b6d1435b6fe293dea59cf32345b590cd48a`. The latter is this change's base.
The capability stack remains #28 → #29 → #30 → #33 → #34; review/merge those
dependencies first, then this incremental integration, rerunning source-bound
acceptance on the eventual integration HEAD. No historical source identity
may be relabeled as the new HEAD.

Implemented:

- One CLI exposes the five existing native families' actual candidate bindings,
  explicit matched calibration collection and source-qualified assembly. It
  reuses the existing search, serialized plan and native dispatcher.
- Original H3 admits explicit 768x1344x124 or 480x832x124 request identities.
  CPU-staged/trim and CUDA/no-trim enter implementation IDs, semantic/config
  bindings, costs and installed-runtime receipts. Legacy CPU defaults and all
  five canonical profiles remain unchanged.
- H3 v3 macro costs require joined native source receipts. Missing costs,
  shape/storage substitution, stale observations and cross-workload selected
  plans reject. Lowering revalidates the selected cost model against the
  original native observations, not just a caller's self-consistent JSON.
- Source-level CPU fixtures force the synthetic CUDA candidate to be selected,
  serialize it and check its dispatcher/config binding; native receipt checks
  reject CPU cache activity, wrong registration and missing rank evidence.

## Validation at implementation time

- Targeted H3, native macro and model validation tests: **305 passed**.
- Whole `tests/test_refactor_*.py` collection: blocked by missing local
  `omegaconf`; this is not a full-suite PASS.
- With the one collection-blocked module explicitly excluded: **1177 passed,
  19 skipped, 13 failed**, all 13 failures from missing `omegaconf`. This broad
  run preceded the last additional targeted regression test. No dependency
  environment was installed; the existing pytest-only environment was reused.
- Required source compilation, shell syntax and whitespace checks passed.
- Reviewed-source import-map/integration verification: **PASS_CPU_INTEGRATION**,
  including 18 best-known integration tests. Adaptations preserve prior
  provenance hashes; canonical profile hashes are unchanged.

Tests needing the existing full environment are `test_refactor_adapter.py`,
`test_refactor_executor.py`, `test_refactor_routea_integration.py`, and
`test_refactor_routea_attempt004.py`. Run CPU-only with
`CUDA_VISIBLE_DEVICES=''`; do not infer GPU validation from them.

## Remaining boundaries

No current-HEAD target-GPU correctness, calibration, numerical comparison,
memory acceptance or selected-plan confirmation exists yet. The older
480x832 CPU/CUDA two-cycle observations (213.667630936/195.759008156 seconds)
are not imported as costs, are not 768x1344 measurements, and establish neither
continuous-exact nor long-horizon equivalence.

Nsight `workloads profile` remains a diagnostic importer, not an automatically
fitted cost library. Calibration selection is explicit and bounded. H3 macro
accounting observes native stages; the separately gated retained-action path
still requires its own action-qualified evidence. The five original families
are not rewritten onto the separate open-workload generic engine. Arbitrary
shape, world size, batch, update ratio, new kernels or new workload semantics
are unsupported, not silently approximated.

See `docs/contracts/minimax_h3_bounded_requests_v1.md` for the immutable
contract and `docs/runbooks/native_planner_workflow.md` for the exact command
sequence and resource budget. Full calibration → search → native confirmation
requires up to five serial two-cycle world16 runs for the two-anchor example.
A single CUDA candidate replay validates only candidate execution wiring, not
the complete measured search cycle or a new performance claim.
# Follow-up: deployment dependencies and CPU isolation

After the initial source delivery, a remote CPU fixture inherited an unsupported
`DISTILLSCHEDULE_RUN_KIND`. Commit `2f74af2` isolates that fixture explicitly and
adds polluted-environment and negative runtime-gate regressions; it does not
relax the observer's execution-mode check. The original remote outcome was
98 passed / 1 failed on each node, not a whole-suite PASS.

Commit `88fd19d` adds portable, explicit per-node dependency overlays to H3
assets. Only path/relative identity file/SHA256 are admitted; no site-specific
paths or implicit experiment hooks enter the library. Dependency identities
bind calibration, costs, replay and serialized plans. Default canonical profile
bytes/asset fingerprints remain unchanged. Local targeted validation including
schema/tamper/local-node/old-cost regressions: **355 passed**. Dependency import
and GPU execution on the final merged source still require separate real gates.

# Follow-up: explicit H3 child toolchain PATH

The parent executed the acdf48b operator-selected packet: both nodes reached
native setup, then Triton's CPU helper compilation failed with
`collect2: fatal error: cannot find 'ld'`. This was **not a completed GPU
correctness run**. The parent reports owned cleanup PASS and OOM deltas zero
on both nodes. Its two-node CPU reproduction failed with the original missing
PATH and passed compile/link/execute with only `PATH=/usr/bin:/bin` added.

Commit `b39766c` fixes only resolved H3 child PATH and checks serialized
binding; source import-map provenance records its exact bytes. No system
installation, CUDA-path guess, model or tolerance change. Targeted local CPU
suite: **89 passed**, including real CPU compile/link/execute with an explicitly
constructed child environment, no inherited toolchain PATH, no Torch or CUDA.
A new packet/source/run identity and fresh imports/toolchain/owner gates remain
required. The general R9 calibration/search transport adapter is deferred;
operator-selected smoke must not be described as searched execution.

## Follow-up: opt-in R9 calibrated selection transport

The frozen 1437839 packet was subsequently run by the GPU-owning operator:
its world collector reported `PASS_OPERATOR_SELECTED_NATIVE_CUDA_PLAN_EXECUTION`
(16 ranks, two cycles, C2 195.826461578 s). It explicitly reported matched
reference `NOT_RUN` and measured search closed `false`. That result is not a
calibration or numerical gate for this follow-up source.

The new independent follow-up provides a bounded H3 low-resolution R9
transport through unified calibration and saved-plan execution. Native
correctness, calibration and current-source cost verification reuse the
existing core. The portable controller does not read absent remote costs;
full rebinding/validation happens on the receipt host and in each worker.
R9 terminal/argv/env/rank/resource receipts are consumed as R9 evidence,
not manufactured legacy shell-launch files. Source/runtime lowering and
transport markers fail closed; explicit read-only admission retries retain
history, GPU execution cannot retry, and unknown owner terminal state blocks
new sessions. The external owner changes only its three workload-kind labels.

Targeted CPU regression: **273 passed**, including the synthetic full-chain
and negative R9 tests. No Torch installation, remote SSH or GPU launch was
performed by this implementation agent. New-source cluster compatibility,
matched numerical comparison, calibration costs and search-selected native
confirmation remain **Pending cluster validation**. See
`docs/runbooks/H3_R9_CALIBRATED_SEARCH.md`; no other workload or FA3 split
candidate is added and no prior source measurement is imported as a cost.
