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
