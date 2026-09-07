# Joint DMD planning: implementation and acceptance ledger

## Scope

The registered-recipe search remains available and its published evidence is
unchanged. The new, opt-in `--strategy model-based` implements a separate search
algorithm over the **same** `core.graph.ExecutionGraph`, not a second training
loop. Its proposed objective is native **cycle 2** wall time after one in-process
warm-up. Initialization, correctness probes, calibration and profiling are not
part of that denominator.

The restricted **Wan placement/chunk/epilogue family now has a completed
measured-cost → search → native-validation loop**. LiveWan/H3 and the broader
sharing/reordering dimensions are not yet GPU-validated generated planners.
CPU simulation, native support, diagnostic validation and strict numerical
equivalence remain separate statuses.

Latest evidence: [Wan closed-loop native diagnostic](../experiments/DMD_JOINT_PLANNER_WAN_CLOSED_LOOP_20260907.md).
Runtime source is `4e3ffb59c21429f3e9e714f14006f5ae6afbd002`, not a later
report-publication commit. Six current-source native trials completed 16/16
rank gates; confirmed cycle-2 comparison is **35.975044 → 24.617367 s,
1.461369× diagnostic**. Numerical `REVIEW` remains; no strict/canonical winner
was promoted. The search rediscovered the incumbent rather than beating it.

## Method being implemented

1. An adapter freezes the workload, versions, sample/RNG identities and precision
   and emits operations plus supported physical implementations.
2. Sharing candidates are generated from equal frozen checkpoint content,
   architecture, dtype and placement, disjoint trainable storage, and explicit
   adapter/recompute scopes. Maximal sharing is **not** assumed fastest: it can
   prevent concurrent execution and incur handoff costs.
3. Existing kernels and their supported tuning knobs form an implementation
   library. Source inventory alone is not a correctness or composability proof.
4. Source/environment/hardware/assets/**precision**-bound native operation or
   microbenchmark receipts provide latency and workspace costs. Nsight unions
   are diagnostic features, never operation wall-time substitutes. Missing costs
   stay `UNCALIBRATED`; there are no made-up throughput estimates.
5. A dependency/resource simulator evaluates candidates, counts shared physical
   storage once and retains output tensors through their last consumer. It does
   not assume two operations can overlap simply because their DAG edges permit
   it. GPU/link reservations and shared-adapter mutual exclusion also apply.
6. Bounded Metropolis search mutates decisions; baseline and incumbent seeds are
   evaluated before spending the proposal budget. A bounded exhaustive CPU
   oracle checks the search on small fixtures.
7. A supported native lowerer must consume the selected decision vector exactly.
   It may not silently run the nearest old recipe. Only new native receipts can
   establish correctness, feasibility, latency or a winner.

The profile/cost-model/simulator/search decomposition is inspired by
[RealHF](https://arxiv.org/abs/2406.14088). The DMD-specific search dimensions are
frozen-base aliasing with independent LoRA/replay scopes, legal G/C reordering,
bounded cross-cycle lookahead, placement, and existing kernel implementations.
MCMC itself is not presented as a new research contribution. No RealHF code was
copied into this implementation.

## Implemented CPU core

| Component | Location | Current evidence |
|---|---|---|
| Versioned physical choices over existing graph | `search/plan_ir.py` | RAW/WAR/recompute, rank/state/resource and lifetime rejection tests |
| Frozen-backbone sharing generator | `search/sharing.py` | Disjoint adapter storage, immutable base identity; mutable-base sharing rejected |
| Operation costs | `search/cost_model.py` | Receipt hash/sample/identity reconciliation; profiled cost rejection |
| Resource and memory simulator | `search/simulator.py` | Hand-calculated serial, overlap, adapter mutex, memory tradeoff and link contention fixtures |
| Bounded search | `search/model_search.py` | New non-seed combination found; small-space exhaustive oracle; deterministic planner RNG |
| Existing implementation inventory | `search/implementation_library.py` | AST/source hashes without Torch imports; no generated kernels |
| Precision identity | `search/precision.py` | Explicit TF32 policy and runtime assertion; verification does not mutate Torch |
| Native validation controller | `search/model_validation.py` | Total 3..8 attempts including anchors/failures/confirmation; immutable resume and precision checks |
| Observed native calibration binding | `search/runtime_identity.py` | Fresh, sanitized worker Python on both nodes; versions/selected binaries and 16 GPU identities before/after execution |
| Unified CLI | `workloads search --strategy model-based` | CPU by default, explicit `--execute-native` after calibration; legacy search unchanged |

Simulator times in unit tests are **synthetic nanoseconds**, not real model or
GPU measurements. A resource envelope includes its internal overlap; adding its
NCCL/kernel union again would double count it. `critical_predecessors` are model
predictions, not proof of a measured CUDA critical path. The simulator reports
cycle-2 activity that begins before cycle-1 completion so that finite-pipeline
fill/drain effects remain visible; two cycles alone do not prove asymptotic
steady state.

## Commands (CPU only)

```bash
python scripts/distillctl.py workloads search \
  --strategy model-based --workload livewan_crossscale_pipeline_best --dry-run

# problem.json must be adapter-produced. Real costs require source receipts.
python scripts/distillctl.py workloads search \
  --strategy model-based --workload wan13b_nogan_resident_best \
  --problem problem.json --cost-model costs.json \
  --max-proposals 10000 --search-timeout 600 --search-seed 0 \
  --output new-search-result.json
```

The second command runs the simulator/search, **not GPU workers**. A CPU fixture
cannot be promoted to a native plan. Existing outputs are never overwritten.
The existing registered native search remains the default strategy.

## Wan native composition bridge

`wan_implementation.py` and `wan_model_adapter.py` now lower independently
selected **placement, chunk size and existing no-grad LoRA epilogue** into the
existing Wan workers/dispatcher. There are seven admitted combinations: one
FSDP control and six Resident combinations (512/1024/2048 × epilogue off/on).
An unsupported FSDP+epilogue or placement is rejected, not mapped to a nearby
recipe. Unit tests verify that 1024 actually reaches all 30 block attributes.

This adapter models the native worker's 14 composite stage envelopes over two
cycles: G update, five-rollout production, then five C-update+EMA stages per
cycle. **14 stages is not a DiT forward count.** The existing trace-based native
forward/backward/update census remains authoritative. The adapter does not
claim to lower arbitrary G/C reordering, HSDP or new Wan GPU pools yet.

All `joint-wan1-*` implementations, including the control, explicitly enable
the same matmul/cuDNN TF32 policy; BF16 base/FP32 LoRA/FP32 reduction remain
unchanged. Runtime observations are checked in both trajectory and native timing
receipts. These new controls must not be compared against legacy TF32-off runs.
Canonical defaults and the old registered IDs are not changed by this opt-in.

The layout input schema is `distillschedule.wan_stage_layout.v1`: current source,
precision and workload binding, sixteen rank owners for each implementation,
post-setup static allocation, and per-rank capacity. Stage cost envelopes include
**all additional live allocation**, including retained rollout outputs; the
adapter therefore does not add separate retained-tensor lifetimes a second time.
Per-rank maxima are conservative, not an assertion that all ranks use equal HBM.
`operation_contract()` supplies the signature that each calibration must match.

```bash
# Runs next to the small native calibration receipts (normally remotely).
# Requires the complete measured layout + costs. The initial pilot below had
# one fragment; the closed-loop report now records all seven at runtime 4e3ffb59.
python scripts/distillctl.py workloads search \
  --strategy model-based --workload wan13b_nogan_resident_best \
  --planning-layout native-stage-layout.json --cost-model native-stage-costs.json \
  --assets-config assets.json --run-id fresh-planner-canary \
  --emit-native-plan native-plan.json --output predicted-plans.json
```

This emits a plan; it does not launch it. Lowering rebuilds the adapter graph,
verifies calibration hashes, and checks the selected decision vector against the
actual native candidate. Saved synthetic fixtures or changed DAGs are rejected.
The existing native runner supports an explicit matching generated control;
it rejects mixing the generated precision family with legacy arms. The validation
controller now connects calibrated proposals to this runner. The native collector
has produced all seven current-source fragments (the original one-fragment
pilot is retained below as historical evidence). The controller
requires a complete measured layout/cost set; it does not invent missing
implementation costs or silently reuse historical results.

```bash
# Example for a separately authorized future epoch. The completed Wan experiment
# has spent its entire 8-attempt budget. Do not relaunch it or reset its budget.
# A fresh validation journal must not overwrite completed or failed runs.
python scripts/distillctl.py workloads search \
  --strategy model-based --workload wan13b_nogan_resident_best \
  --planning-layout native-stage-layout.json --cost-model native-stage-costs.json \
  --execute-native --assets-config assets.json --run-id new-generated-validation \
  --validation-dir new-validation-journal --max-trials 6 \
  --output new-native-validation-result.json
```

The total budget includes baseline, incumbent, new plans and independent
confirmation (eight means **1 + 1 + at most 5 + 1**, not eight candidates plus
anchors). The initial pilot spent two timing attempts; the closed-loop session
then spent the **six remaining**, including source-required replacement anchors
and independent confirmation. No Wan timing attempts remain in this epoch.
Do not reset the overall experiment budget when changing a commit.
Failed/aborted attempts consume budget. Resume rechecks the small receipts and
refuses an incomplete run; it never retries a possibly live worker. Numerical
`REVIEW` remains diagnostic and cannot produce a strict winner. Predictions and
measured errors are stored separately; native latency never gets replaced by a
model estimate. This does not update a canonical profile.

The new model-based runner does not accept expected environment/hardware
fingerprints simply echoed into measurements. It collects both nodes' actual
Python/Torch/CUDA/NCCL/package and selected-binary identities, GPU UUIDs and
topology in the same sanitized import environment as the native launcher.
Calibration and execution observations must agree before correctness workers,
before timing and after execution. Collection itself creates no CUDA context.
This is a versions/selected-binaries check, **not** a full-environment hash or a
replacement for the dispatcher's GPU ownership lock. The `be877cef` anchor
experiments predate this additional machine-readable gate and are not relabeled
as having used it.

## Acceptance remaining before a planner speed claim

- LiveWan/H3 planning adapters and generated native lowering. Wan's generated
  FSDP control, Resident incumbent and three model-ranked non-seed combinations
  now have native two-cycle validation, plus independent winner confirmation.
  Strict numerical equivalence remains unestablished; the current Wan timing
  budget is exhausted (two earlier anchors plus six current-source attempts).
- Finer calibrated operator/layer costs, G/C ordering/pool lowering, and shared
  backbone placement choices; no unsupported TP/PP, dynamic repartition or
  fabricated pool mappings.
- Native operation calibration and memory lifetimes for each admitted binding.
- Wan, then LiveWan, then H3 Grid1000 pilots under owned world16 locks. H3 keeps
  approximate Grid1000 semantics and FA3 CPU replay storage explicit; boundary
  offload `none` does not mean there are no CPU transfers.
- Per workload: at most eight timing runs (baseline, incumbent, up to five new
  plans, independent confirmation). Each uses two cycles, one initialization,
  unprofiled cycle 2 only. Numerical REVIEW is not strict equivalence.
- Canonical winner changes require measured evidence and separate review; this
  branch neither changes canonical profile hashes nor auto-merges main.

## Initial CPU acceptance, 2026-09-07

Inherited source: `d310597e774b18fea98118e00a58a4b912e89f19`, based on fetched main
`e69f4bf8b8bb64f54542fea74c452b53e090b526`. Branch:
`agent/dmd-joint-planner-20260907`.

- First core: **498 passed, 17 skipped**. With Wan composition:
  `tests/test_refactor_*.py`: **509 passed, 17 skipped**; skipped tests are
  not claimed as passing GPU checks.
- Existing five-workload integration/source-hash/compile/shell checks: **PASS**.
- Wan generated control/incumbent: native validation below. Operation
  calibration and automatic model-generated non-seed GPU search: **NOT_RUN**.
- No historical timing has been relabeled as a result of this branch.

Small local evidence:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/`.
Raw model/profile assets remain remote. Two-node read-only preflight found 16
idle A100s and the prior Torch 2.10/CUDA 12.8 environment, but this does not grant
a future GPU lock; ownership must be checked again at launch.

## Native Wan bridge acceptance

Experiment source: `be877cef1b3599266a4d8b75f304006653cae5b9`.
Both generated implementations completed two-cycle correctness on 16/16 ranks,
then one independent two-cycle unprofiled timing run each. Every rank ended at
G4/C20/EMA20. No new OOM; owned teardown and subsequent idle/lock audit passed.

| Generated implementation | Cycle 1 (warm-up) | Cycle 2 | Peak allocated |
|---|---:|---:|---:|
| world16 FSDP control, chunk512, epilogue off | 45.774857 s | 36.443519 s | 4.298 GiB |
| Resident incumbent, chunk2048, epilogue on | 27.660358 s | 24.628109 s | 21.899 GiB |

The diagnostic cycle-2 ratio is **1.479753x**. This is an **incumbent
revalidation**, not a new plan discovered by the model search and not a
placement-only ablation (the Resident implementation also selects existing
reduction/kernel policies). Both arms explicitly observed TF32 enabled; neither
is compared against historical TF32-off data. Coarse two-cycle trajectory
comparison passed; strict P0 numerical equivalence remains **NOT_ESTABLISHED**.
The native comparator's legacy status string contains `50C`, but this run
executed **two**, not fifty, cycles.

Evidence: `docs/experiments/DMD_JOINT_PLANNER_WAN_BRIDGE_20260907.md` and
`artifacts/manifests/dmd_joint_planner_wan_bridge_20260907.json`. Publication
commits after `be877cef` do not relabel the experiment's source revision.

## Native controller identity probe

Controller implementation `18e889ad3653103e7f1d1aebfb055d71cdc22205` passed
**521 CPU tests, 17 skipped**, plus the five-workload integration/compile/shell
checks. A bounded independent review reproduced and verified repairs for
expected-identity echo and inherited SSH import-path contamination. Full CE
review orchestration was unavailable; this is not a full review coverage claim.

The exact commit was then deployed to a **new** shared worktree and probed on
both actual A100 nodes through the native RPC and sanitized worker Python.
Both observed Python 3.10.20, Torch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5 and
eight GPU identities. Neither probe created a CUDA context. No model loading,
GPU experiment or new timing attempt occurred. Pre/post GPU process lists were
empty, both node locks were available, and cgroup OOM/oom_kill increments were
zero. The selected-version/binary identity scope remains explicit.

Receipt: `artifacts/manifests/dmd_joint_planner_runtime_identity_20260907.json`.
This proves the observation path on the real substrate, not a measured cost
model, automatically discovered GPU plan or completed LiveWan/H3 pilot.

## Opt-in native stage calibration bridge

The new collector is attached to the existing Wan two-cycle worker's actual
G update, five-rollout production and five C+EMA call sites. No duplicate
training loop, model arithmetic or altered call census is introduced. Only
generated `joint-wan1-*` profiles admit `calibration`; the canonical five
profiles and registered timing defaults are unchanged.

```bash
# Without --execute this only writes a NOT_RUN source-only plan.
python scripts/distillschedule/calibrate_wan_native_stages.py \
  --implementation joint-wan1-resident-c1024-e1 \
  --expected-head "$(git rev-parse HEAD)" \
  --run-id fresh-native-stage-canary --output "${EVIDENCE_ROOT}/new-controller-evidence"

# Explicit GPU opt-in additionally requires --assets-config assets.json --execute.
# Exactly ONE implementation, current-source correctness preparation, then two
# calibration cycles. This command never starts the seven-arm matrix or timing.
```

Each of the fourteen stages uses an entry barrier/device sync and trailing
device sync. Memory observations and entry synchronization are outside the
stage's duration. This perturbs cross-stage overlap: **neither cycle wall nor
sum of stage durations is a formal measurement**. Ordinary timing summaries
explicitly reject these receipts, including calibration hidden under the old
timing schema. Failure snapshots preserve partial stages and cannot become PASS.

Collection validates current-source correctness, all sixteen owners, fixed
sample/trace/versions/census/finite gates, both nodes' observed environment and
GPU identities before/after, OOM counters and owned teardown. Coarse numeric
`REVIEW` is retained, never upgraded to strict equivalence. Output consists of
fourteen content-hashed `operation_calibration.v1` receipts, operation costs and
a **partial** layout fragment. Unknown implementations remain `UNCALIBRATED`;
one fragment cannot masquerade as the complete seven-implementation layout.

The operation sample is the maximum rank-local synchronized envelope, not a
kernel-duration sum. Peak allocation is aggregated across all stage peak resets;
workspace includes all live bytes above post-setup static allocation, including
retained rollout outputs. Memory limits subtract boundary-observed non-allocator
usage plus 2 GiB safety. This is conservative allocator accounting with sampled
driver information, **not an exact continuously observed driver peak**.
The calibration context also observes peaks before the FSDP helpers' nested
`reset_peak_memory_stats` calls, invokes those original resets, and restores the
process-local wrapper on exit/failure; otherwise earlier rollout peaks vanish.

Calibration GPU expenses and native timing attempts are separate ledger fields.
This milestone authorizes one non-seed `resident/c1024/epilogue-on` diagnostic
canary, not a fresh eight-trial budget. Current-source correctness may have to
be collected again after instrumentation changes; old-source gates are never
relabelled. The two published timing attempts remain spent; six timing attempts
remain, and calibration does not establish a new speedup or canonical winner.

### Native single-implementation pilot: PASS

Executed source `4e3ffb59c21429f3e9e714f14006f5ae6afbd002`, run
`jointp_wan_stage_c1024_20260907_r1-calibration`: **16/16 ranks, 224 ordered
stage observations, 14 verified operation-cost receipts**, final G4/C20/EMA20.
Peak allocated was 21.90 GiB, with no new CUDA/cgroup OOM and owned teardown
verified on both nodes. The current-source two-cycle correctness preparations
passed the existing coarse trajectory gate, not strict numerical equivalence.

This proves actual native collection and receipt ingestion for
`joint-wan1-resident-c1024-e1`, not a complete search database. The two diagnostic
cycle envelopes include extra stage synchronization and cannot establish speedup.
Keep calibration execution at the recorded runtime source/import paths; later
report-only publication commits do not change the measured source identity.

Report: [native stage calibration](../experiments/DMD_JOINT_PLANNER_NATIVE_CALIBRATION_20260907.md).
Manifest: `artifacts/manifests/dmd_joint_planner_native_calibration_20260907.json`.
