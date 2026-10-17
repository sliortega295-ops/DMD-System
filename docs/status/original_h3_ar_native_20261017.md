# W6 native adapter CPU handoff — 2026-10-17

Base: `07fcc0ba1ec28b810b50cac92ec41c1a5232d3e1`.
Status: **Pending cluster validation**. No SSH/GPU run, package installation,
push, canonical-profile modification or training-math rewrite was performed.

- Opt-in W6 schema on `workloads native --task`, plus read-only native adapter
  discovery. Canonical five-profile defaults and historical timings untouched.
- Whole original update envelope through common `service.select`, source-bound
  calibration, saved-plan independent lowering validation, and persistent
  original `H3ARSession` / `PreparedRolePlan` native dispatch.
- Source and AP helper/library pins taken from existing native provenance.
  Baseline and AP are genuinely different runtime mechanisms; no old shell
  filename or historical time serves as a planner candidate cost.
- CPU verification: 28 focused tests; 467 targeted H3/Grid/macro/Wan tests;
  compileall and shell syntax checks passed. The broader available refactor
  run had 2011 passed / 87 skipped / 15 failed, with failures due to absent
  OmegaConf or Triton in the specified local Python environment. One old
  Route-A test module could not collect because OmegaConf was unavailable.
  No dependencies were installed and those gates are not reported as PASS.
- Full-model/native math parity, two-node execution, actual AP mechanism
  census, physical capacity and <=3% manual matching remain **NOT_RUN**.
  Numerical status remains REVIEW, not exact-equivalence PASS. CPU fixture
  durations are synthetic tests only and never published as GPU costs.

Coordinator next: CPU-only source/asset admission (one checkpoint payload hash,
then stat-bound small receipts), preserve original entry64 GiB cgroup gate,
freeze explicit common capacity and source, run both real native calibration
arms, assemble all16 ranks, save common-planner selection, then run selected
and matched manual AP arms. Keep existing owner safety/cleanup/resource gates.
See the contract for historical capacity evidence and the runbook for CLI use.
