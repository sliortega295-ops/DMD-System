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

## CPU follow-up: unavailable cgroup lifetime peak

The first integration (`1683942`) unconditionally read `memory.peak` after
the native update. The known target kernel lacks that file. A CPU fixture
reproduced the actual worker closure's `FileNotFoundError`; the preserved red
log is `h3_ar_cpu/logs/red_missing_memory_peak.log` in the campaign artifacts.
This was a CPU reproduction, not a GPU result. The transient raw failure JSON
was reaped by pytest; the stack log, not that missing JSON, is the retained
failure evidence.

The repair reuses the common actual-cgroup hierarchy observation and identical
capacity evidence policy. Unavailable peak stays `None`/`UNAVAILABLE`, only
exact finite hardcap + maintained reserve 0 + unchanged OOM counters can pass,
and actual capability is bound through assembly, saved plan and runtime.
Observed peaks retain their reserve/highwater checks. Original entry 64 GiB
headroom remains a separate gate. No capacity number, training math, input
mapping, model/RNG progression or golden output was changed.

The actual production observation branch completes two CPU-fixture updates
with no peak file; changing actual capacity during execution is rejected and
preserves a failure receipt. Tests also enforce native GPU-info patch install
before `set_device`/`Accelerator`. Focused W6 + common resource verification:
61 passed / 11 subtests passed; expanded H3/macro/Wan regression:
500 passed / 11 subtests passed; compileall passed. Target Torch 2.10 execution,
full native math parity, two-node GPU work and manual timing matching remain
**NOT_RUN** by this worker; the coordinator owns those gates.
