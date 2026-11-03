# Unified bounded native planner workflow

Use existing `scripts/distillctl.py workloads`: source-only intent → matched
native correctness → component calibration → CPU assembly/search → fresh saved
execution. Only explicit execution flags launch under the existing operator-owned
runner/locks/gates; no new trainer/allocation is introduced. Use a clean common
source and fresh output/run IDs, with raw receipts/models in remote persistent
storage. Assembly runs beside actual returned files, not invented local copies.

| Binding family | Bounded legal inventory, not measured choices |
| --- | --- |
| Wan |35 action implementations; placement/chunk/pointwise/retention |
| Helios No-GAN |20 FSDP/HSDP/chunk/window actions |
| Helios GAN |4 world16 FSDP/chunk/persistence actions; five-rollout group |
| H3 Grid1000 |6 macros per request; CPU/CUDA cache/depth/tile, joined seven-stage costs |
| LiveWan |5 coupled FSDP/pipeline-commit implementations, not independent G/Score overlap |

These counts are source capability, not current-HEAD GPU qualification. Native
algorithms stay distinct from generic Program/RoleTask factories. For W6's task
adapter use [AR admission/calibration/run](../guides/native_workloads.md#original_h3_ar_native).

## H3 calibration, assembly, search and confirmation

Supply exact request-matching prompt cache, Grid1000 mmap, converted Diffusers,
FA3, LightX2V and assets; `HEAD`, `OUT`, `ASSETS` below are operator-supplied.
The examples include the existing Wan stage/manual commands and explicit H3
noise-diagnostic policy, each retaining its own scope:

```bash
python scripts/distillctl.py workloads candidates \
  --workload minimax_h3_grid1000_best \
  --request configs/distillschedule/requests/minimax_h3_480x832_grid1000.json

# Optional Nsight diagnostic import, not a cost model and never GPU MFU:
python scripts/distillctl.py workloads profile --input "$PROFILE_INPUT" --output "$PROFILE_OUTPUT"

# Omit --execute-native first to inspect the immutable intent without launching.
python scripts/distillctl.py workloads calibrate \
  --workload minimax_h3_grid1000_best --expected-head "$HEAD" \
  --run-id h3-v3-cal --output "$OUT/collection" --assets-config "$ASSETS" \
  --candidate joint-h3v3-h480-w832-f124-cpu_staged_trim-d2-n128 \
              joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128 \
  --execute-native

python scripts/distillctl.py workloads assemble-calibration \
  --workload minimax_h3_grid1000_best --expected-head "$HEAD" \
  --result "$OUT/collection/calibration-0/result.json" "$OUT/collection/calibration-1/result.json" \
  --output "$OUT/library"

python scripts/distillctl.py workloads search --strategy model-based \
  --workload minimax_h3_grid1000_best --problem "$OUT/library/problem.json" \
  --cost-model "$OUT/library/cost-model.json" --search-algorithm hierarchical \
  --run-id h3-v3-selected --assets-config "$ASSETS" \
  --emit-native-plan "$OUT/selected-plan.json" --output "$OUT/search.json"

python scripts/distillctl.py workloads run --workload minimax_h3_grid1000_best \
  --search-plan "$OUT/selected-plan.json" --dry-run
python scripts/distillctl.py workloads run --workload minimax_h3_grid1000_best \
  --search-plan "$OUT/selected-plan.json" --run-id h3-v3-confirm \
  --coordinator-output "$OUT/confirmation"
```

```bash
# Source-only unless --assets-config "$ASSETS" --execute is explicitly added.
python tools/assets/calibrate_wan_native_stages.py \
  --implementation joint-wan2-resident-c2048-e1-w5 --expected-head "$HEAD" \
  --run-id "$NEW_RUN_ID" --output "$OUT/new-collection"
python tools/assets/assemble_wan_action_calibrations.py \
  --calibration-result "$FSDP_COMPONENT_RESULT" \
  --calibration-result "$RESIDENT_COMPONENT_RESULT" \
  --expected-head "$HEAD" --output "$OUT/new-library"
python scripts/distillctl.py workloads search --strategy model-based \
  --workload wan13b_nogan_resident_best \
  --planning-layout "$LAYOUT" --cost-model "$COSTS" --assets-config "$ASSETS" \
  --run-id "$NEW_PLAN_ID" --emit-native-plan "$OUT/new-plan.json" \
  --output "$OUT/new-search.json"
```

```bash
python scripts/distillctl.py workloads native benchmark-manual \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --implementation "$RUN/manual-strategy.json" --cycles 2 \
  --output "$RUN/new-manual-native" --execute
```

```json
{
  "schema_version": "distillschedule.h3_noise_diagnostic_policy.v1",
  "source_head": "<exact current 40-character commit>",
  "reference_candidate_id": "joint-h3v3-h480-w832-f124-cpu_staged_trim-d2-n128",
  "candidate_id": "joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128",
  "max_cross_self_delta_l2_ratio": 2.0,
  "purpose": "PLANNER_DIAGNOSTIC_ONLY",
  "formal_timing_eligible": false,
  "default_gate_unchanged": true
}
```

### What the H3 runs prove

Minimal wiring takes up to two correctness trajectories, two instrumented
calibrations and one selected correctness replay, serial world16. Emitted plans
use correctness, not timing. A separate model-based `--execute-native` with fresh
`--validation-dir --max-trials 3` performs matched-control/incumbent/independent
confirmation diagnostics; budget its additional correctness gates separately.
Do not reuse a source-only output directory for execution.

Require both node launch/preflight manifests; exact source/request/semantics/
assets/runtime/hardware;16unique ranks/two cycles/G2F10EMA0; finite original
Grid1000/census/sample/RNG/state checks. Actual FA3 cache/trim/depth must match;
CUDA-cache CPU-transfer/trim counters must be zero. Resource evidence includes
setup/lifetime peaks, no CUDA/cgroup OOM and owned cleanup. Reopen raw hashes,
not environment echoes/exit codes. Predictions and measured errors are recorded separately.

### Wan action/cost and manual boundary

`joint-wan1-*` uses composite stage envelopes; prepared
`joint-wan2-{fsdp|resident}-c{chunk}-e{0|1}-w{1..5}` retains rollout-window
lifetime in original G/R/C/EMA execution. B1/global16 and G1C5EMA5 stay fixed;
window is not batching. Missing handlers/dependencies/cross-pool transport fail.
Cost keys bind actual work/shape/precision/placement/source/assets/hardware/
residency and warm/cold state. Component fences perturb overlap: calibrated
walls are not native speedup denominators; envelopes already include their
internal overlap/retained allocation, never add unions or memory twice.
Unknown costs remain `UNCALIBRATED`. [Helios clocks/costs](#helios_component_planning)
are separate.

The manual command above admits only two fixed-trace native cycles under the
existing world16 owner, not plans/fresh streams/world1. Its receipt says
`MANUAL_REFERENCE_DIRECT_NATIVE_ORDER`, `search_performed=false`,
`plan_sha256=null`. Compare separately saved `run --plan ... --execute` at
identical source/assets/precision/hardware/timer using world-maxC2.
Without execute, no assets/models load.

### H3 control-repeat diagnostic and toolchain

`DEFAULT_P0_PARITY` stays strict. Only after both cross-cache and repeated-control
comparisons fail may the explicit `h3_noise_diagnostic_policy.v1` above apply.
Only storage/trim differs; shape/checkpoint/tile/depth/schedule/precision match.
Fresh control/candidate/control-repeat must pass all16/two-cycle/sample/RNG/replay
contracts. Each Student/Fake loss and four parameter/Adam sketches needs
cross-L2≤limit×self-L2, limit≤2; self0 admits only cross0. Assembly rejects
aliased/stale/missing/edited witnesses. Record the measured cross-run noise and the original comparator result;
noise-diagnostic results remain separate from timing admission.

The exact H3 child `PATH=/usr/bin:/bin` is serialized, not inherited or prepended
with environment/CUDA bins. Before retry qualify GCC+ld by compiling/linking/
running a minimal CPU program on both nodes under that child env; record hashes
and returns. Imports alone are insufficient; CPU linker success is not CUDA/JIT
workload proof. Preserve failed attempts and create fresh bound plan/run/seal.

## Metadata transport and H100

Asset-node `rpc_transport:argv-json-tty-v1` is an explicit alternative to stdin
when a gateway loses EOF. Compressed bounded JSON goes in argv, the same private
worker input stream parses it and exactly one framed result is required.
Journal request/raw stdout/stderr; duplicate/missing frames fail. It is not GPU
retry permission: an interrupted R9 needs both owner streams terminal before any
new SSH/query; never edit an immutable failed run to switch transports.

`distillctl workloads hardware` connects the pinned8-H100 Wan bridge to the
same search/cost model while keeping physical workers distinct from logical
B1/global16 samples. [Hardware planning](../workloads/hardware_native_planning.md)
defines its admitted scope; CPU tests are not H100 GPU execution or permission
to preempt another reservation.

<a id="H3_R9_CALIBRATED_SEARCH"></a>
## Opt-in H3 R9 calibrated search transport

R9 is an opt-in transport/evidence adapter for480×832×124 Grid1000,
CPU-staged/trim versus CUDA/no-trim, depth2/LoRA128. Canonical profiles,
FA3 splits/numeric thresholds/training are unchanged. Existing
`calibrate --execute-native --r9-transport DEPLOYMENT.json` performs matched
correctness before calibration; `run --search-plan PLAN.json --r9-transport ...`
executes the saved selection, never default-transport fallback.

Deployment fields are exactly `schema_version:distillschedule.r9_transport.v1`
and absolute `{path,sha256}` pins for `owner`, `template`, `dependency_provenance`.
The owner SHA is reviewed in `search/r9_contract.py`; the template carries
assignment/safety/runtime, never old cost authority. Only its two explicit
Torch/Transformers overlays enter imports; no portable site paths are embedded.

The controller recompiles current node commands/config/env. A remote saved
`model_plan` is a portable envelope, not local cost validation. CPU RPC
`r9_rebind`/`r9_prepare` run full `validated_profile` beside raw receipts;
workers repeat it plus `validate_local` before models. Rebinding preserves
`model_plan`; finite sorted/indented JSON and exact sealed-manifest bytes bind.

Each run: prepare → two hidden-CUDA dependency/import/linker qualifications →
seal → both owner admissions/executions → world collection. Only two owner SSH
sessions exist during execution; CPU RPC resumes after both EOFs. Preserve
lock/UUID/FD/PID-birth/permit/peer/reserve/OOM/deadline/lease/signal/reap/cleanup
gates, original entrypoint and tracked node sampler. R9 receipts positively bind
unprofiled argv/env/ranks/cache/plan/terminal/resources, not fabricated legacy
launch/teardown text. Missing proof cannot fall back to legacy evidence.

| Required fresh run on one clean source | Stop condition |
| --- | --- |
|1CPU-cache correctness,2CUDA correctness + comparison |Comparison must accept **all** sampled state/loss/sample/RNG fields; any numerical failure stops the sequence |
|3CPU calibration,4CUDA calibration + CPU assembly/search/save |Source/runtime/resource/cost/selection invalidity stops before5 |
|5actual confirmation of measured winner |Do not force CUDA or admit unmeasured variants |

Old HEAD times, including1437839 operator-selected smoke, are not costs.
Calibration uses max-rank contextual CUDA-stream stage envelopes, not cross-rank
wall unions. Five runs close bounded selection execution, not the full
`ModelValidationSession` journal (≥3diagnostic timing trials), robust speedup,
continuous-exact semantics or long-run parity. Work/lease bounds remain explicit;
no profiler/install/automatic retry.

Failed admission attempts are immutable numbered records. Read-only
`workloads r9-owner --plan ... --admission-attempt N` retry requires prior EOFs;
one exclusive execution-start forbids GPU retry. Unknown terminal state forbids
new sessions, even queries. New-source cluster validation remains NOT_RUN until
real site receipts, not synthetic CPU parser/path tests, satisfy the sequence.

<a id="livewan_c51_planner_v1"></a>
## LiveWan C51 full-bundle planner opt-in v1

The explicit C51 decision cohort preserves the locked C34/five default candidates
and the separate 69-frame request. Source origin `aa30527db5d048e2ab23222eef587cb7cfe909f6`, expanded
template SHA `a50e4e3a969a513dae0399878f7e4f25e23085ee119e1175cad83192c0c6f7bb`.
C51 includes C34 G modulation/residual and Score gate/LayerNorm, pinned Score
FA3 + exact per-checkpoint SAC, G read-only FA2 split-KV and no-grad fused RMS.
It excludes C37step1/C39step0 overlap, C40LoRA sharing, C44down cache, merge,
changed thresholds/math. Actual globalB16/368×640×81/two cycles/G1C5EMA0,
continuous shifted sigma/sample/RNG/precision/version/transport remain original.

```bash
python scripts/distillctl.py workloads candidates \
  --workload livewan_crossscale_pipeline_best --include-livewan-c51
```

```json
{
  "path": "/absolute/pinned/flash-attn3/artifact",
  "identity_file": "metadata.json",
  "sha256": "f1b4b0ef991c0966f7d848777154c0591c4580968fb6da33f6b2a2829a109309"
}
```

```bash
# Source-only by default. This command does not contact nodes or launch GPUs.
python scripts/distillctl.py workloads calibrate \
  --workload livewan_crossscale_pipeline_best \
  --candidate joint-livewan1-fsdp-m0-g0-r0-bundle1-control \
              joint-livewan1-pipeline-m1-g1-r1-bundle1-c51 \
  --expected-head "$(git rev-parse HEAD)" --run-id c51-v1-source-only \
  --output /absolute/project/evidence/c51-v1-source-only

# Only the authorized runner may execute this, with fresh output/run ID.
# The existing workflow owns matching correctness, serial calibration,
# native node locks, runtime identities, capacity/driver evidence and teardown.
python scripts/distillctl.py workloads calibrate \
  --workload livewan_crossscale_pipeline_best \
  --candidate joint-livewan1-fsdp-m0-g0-r0-bundle1-control \
              joint-livewan1-pipeline-m1-g1-r1-bundle1-c51 \
  --expected-head "$(git rev-parse HEAD)" --run-id c51-v1-calibration \
  --assets-config /absolute/project/c51-v1-assets.json \
  --output /absolute/project/evidence/c51-v1-calibration --execute-native

python scripts/distillctl.py workloads assemble-calibration \
  --workload livewan_crossscale_pipeline_best \
  --expected-head "$(git rev-parse HEAD)" \
  --result /absolute/project/evidence/c51-v1-calibration/calibration-0/result.json \
           /absolute/project/evidence/c51-v1-calibration/calibration-1/result.json \
  --output /absolute/project/evidence/c51-v1-assembled
```

```bash
python scripts/distillctl.py workloads search --strategy model-based \
  --workload livewan_crossscale_pipeline_best \
  --problem /absolute/project/evidence/c51-v1-assembled/problem.json \
  --cost-model /absolute/project/evidence/c51-v1-assembled/cost-model.json \
  --assets-config /absolute/project/c51-v1-assets.json \
  --run-id c51-v1-selected --exhaustive --max-proposals 32 \
  --output /absolute/project/evidence/c51-v1-search.json \
  --emit-native-plan /absolute/project/evidence/c51-v1-selected.json
```

```bash
python scripts/distillctl.py workloads run \
  --workload livewan_crossscale_pipeline_best \
  --search-plan /absolute/project/evidence/c51-v1-selected.json \
  --assets-config /absolute/project/c51-v1-assets.json --run-id c51-v1-replay \
  --coordinator-output /absolute/project/evidence/c51-v1-replay
```

```bash
CUDA_VISIBLE_DEVICES='' LIVEWAN_REQUIRE_TORCH210_CPU=1 PYTHONPATH=src \
  python -m pytest -q \
  tests/test_refactor_livewan_c51_planner.py \
  tests/test_refactor_livewan_c51_artifact_chain.py \
  tests/test_refactor_livewan_c51_checkpoint_cpu.py \
  tests/test_refactor_livewan_generator_fused_rms.py \
  tests/test_refactor_livewan_generator_splitkv.py \
  tests/test_refactor_livewan_c38_score_fa3.py \
  tests/test_refactor_livewan_c31_integration.py \
  tests/test_refactor_livewan_c32_integration.py \
  tests/test_refactor_livewan_c33_integration.py \
  tests/test_refactor_livewan_c34_integration.py \
  tests/test_refactor_livewan_request.py \
  tests/test_refactor_livewan_action_observation.py \
  tests/test_refactor_livewan_attention_checkpoint.py \
  tests/test_refactor_native_macro_adapter.py \
  tests/test_refactor_livewan_calibration_summary.py
```

The opt-in control is `joint-livewan1-fsdp-m0-g0-r0-bundle1-control`; C51 is
`joint-livewan1-pipeline-m1-g1-r1-bundle1-c51`. Four other controls retainC31-C34.
Omitting `--include-livewan-c51` preserves the original IDs. Treat C51 as an
indivisible bundle, not factor-wise scheduling/ablation proof.

Both control/C51 assets add the pinned `score_flash_attn3_kernel` directory
shown above, mapped to `DISTILLSCHEDULE_SCORE_FA3_KERNEL`. The actual helper
checks five hashes and imported custom-op schema; no FA2/differentFA3/HEAD/env
substitute. The historical artifact location is provenance only (source checkout:
`src/distillschedule/runtime/livewan_crossscale_worker.py`); revalidate real
local kernel/environment/ownership on clean same-source nodes.

Existing calibration `_prepare` runs both native correctness trajectories,
summary/comparison/timing-gate-v2 before costs. Native trajectory/timing producers
atomically publish lowered flat plan+digest once per node-local rank0 before
models/timers; summary/compare/gate/header/loader bind those exact shape/source/
plan files. Canonical C31-C34 do not inherit new opt-in requirements.
Joined real source-qualified observations feed the common static-plus-additional
all-rank capacity model; no handwritten/historical/nearby timings fill gaps.
Use authorized `--execute-native --validation-dir ...` only with a fresh budget/
ID, or the saved replay command. Source-only/default commands launch nothing.

The CPU gate requires existing validated Torch2.10 (install nothing): actual
checkpoint/autograd/RNG/Adam and two production SAC replay reads, with only
CUDA discovery metadata substituted. Artifact-chain tensor/receipt fixtures
remain explicitly synthetic. Recorded local characterization was255passed,
0skipped,18.02s on2.14.0+cpu **without** require2.10, not the production gate.

Target proof still needs same-source all16 two-cycle state/sample/RNG/numeric
parity, real `planner_c51_bundle` FA3/SAC/Score/split-KV/RMS/modulation/cast/grad
receipts, asynchronous operation calibration, static/reserved/driver peaks,
unchanged OOM and teardown, then matched unprofiledC2/repeat-CV checks.
Historical42s observations/CPU fixtures are not those proofs. No strict winner,
formal speedup or arbitrary independent G/Score/link/pool-DAG admission follows.

<a id="wan_compute_state_planning_v1"></a>
## Wan bounded checkpoint-retention planning, v1

Retention/scratch choices need a full `compute_policy` plus backend-bound
`compute_key`: backend alone would alias distinct work/residency. Window depth
stays a DAG decision, not a compute key. Every state-qualified component context
pins policy, including persistent graph effects; a rehashed saved plan cannot
substitute an unmeasured variant.

```bash
python scripts/distillctl.py workloads native validate \
  --task configs/distillschedule/original_wan/full5_reference.json \
  --include-incumbent-bundle --include-critic-retention
```

Only exact original `OriginalWanTask` and immutable incumbent geometry/assets/
precision/bootstrap admit `planner-retain4.v1`, Critic26–29. The inventory does
not add failed tail6/tail8, attention replay, merge, changed batch/optimizer/math.
Old `manual-only.v1` retains its calibration/run refusal; default/legacy JSON
are unchanged and streaming-campaign inventory is not extended.

Calibrate each explicit strategy, assemble actual all-rank receipts, then use
existing `plan`/`run --plan`. Measured action costs/workspace/output lifetimes
and frontier search decide, not a forced winner or historical~20s cost fill.
`OriginalWanSession` installs existing `checkpoint_block_retention.install`:
selected4blocks use original normal autograd, other26 original checkpoints;
Generator dispatch/source/census/versions/capacity/numerics remain unchanged.
CPU planner/30block/Adam/RNG tests use synthetic costs, not native GPU proof.
Changed source needs fresh matched calibration and manual-versus-savedC2;
current-source qualification remains NOT_RUN without those receipts.

<a id="helios_component_planning"></a>
## Helios No-GAN / GAN: measured components and native plans

Existing model-based search uses original P0 losses/helpers through a versioned
graph, not another trainer. Legacy `native0` fixes canonical kernels/B1/global16,
unpacked Score, residuals/EMA/FA2; full manual GAN is separately pinned
SM80FA3/down40/N70-OFF with its own source-qualified costs.

| Variant | Admitted physical decisions | Native order/version mapping |
| --- | --- | --- |
| No-GAN |world16FSDP or2×8HSDP; chunk512/2048; window1..5; G persistenceOFF |G,R/C/EMA after eachC; cohort10..14; G2/C10/EMA10→4/20/20 |
| GAN |world16FSDP; chunk512/2048; window5; persistenceOFF/ON |G, one persistent `r_all` for5actualB1 rollouts,5C commits, EMA afterC0; logical200/1000/800→202/1010/802, physical2/10/800→4/20/802 |

GAN packed fake/real/R1 is physicalB3/logicalB1, R2inactive. Never split its
persistent scope or write logical offsets198/990 into optimizer/scheduler.
No TP/PP/sharing aliases are introduced. Both launches initialize once and
carry real state through two fixed-cohort replay cycles, explicitly
`fixed_source_cohort_replay`, not fresh15..19/1005..1009 inputs.

```bash
python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_nogan_best --dry-run
python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_gan_best --dry-run

# After NativeTrialRunner has collected matched correctness and component costs:
python scripts/distillctl.py workloads assemble-calibration \
  --workload helios14b_gan_best --result <baseline-result.json> <manual-result.json> \
  --expected-head <same-source-head> --include-helios-gan-manual \
  --helios-allocator-margin-mib 2048 --output <new-assembled-directory>

python scripts/distillctl.py workloads search --strategy model-based \
  --workload helios14b_nogan_best --planning-layout <layout.json> \
  --cost-model <cost-model.json> --assets-config <assets.json> \
  --run-id <fresh-id> --emit-native-plan <native-plan.json> \
  --output <new-search-result.json>
```

`helios_implementation` validates typed assignment/launcher;
`helios_planner_cycle` owns shared graph/runner and original helper dispatch,
forward/trace/commit/version/output-lifetime/effective-policy receipts.
Correctness precedes same-source/assets/assignment calibration and native timing.
Loss reads/global-coordinate parameter/Adam/EMA sketches stay outside timers;
the sampled comparison status is retained. `NativeTrialRunner.calibrate()` collects and
`helios_model_adapter.assemble_calibrations()` joins real observations.

Costs are max-rank fenced native action walls, including internal NCCL/compute
overlap. Keys bind chunk/placement/persistence/shape/calls/C1-vs-C2 state;
matching windows may reuse components, missing observations stay UNCALIBRATED.
Memory is post-setup static + per-action peaks + retained-latent lifetimes;
other growth is conservative workspace, not construction-peak qualification or
invented inter-action overlap.

Default allocator margin is exactly2GiB. Explicit
`--helios-allocator-margin-mib N` / `allocator_margin_bytes` is a nonnegative
integer (not boolean), omission preserving old layout fields/default. Rebuild
rank caps from actual driver total minus maximum `max(total-free-reserved,0)`
minus margin, minimum across fragments. Native/manual-match budgets must equal
that hash-bound layout, not arbitrary caller limits. This is distinct from
`gpu_reserve_bytes`/sampled free-driver gates. A new margin needs authorization
and fresh preregistration; it cannot relabel a prior rejection or authorize a run.

`helios_model_adapter` builds/rechecks common source-qualified planning;
`helios_native` validates16rank source/runtime/resource/OOM/teardown receipts.
NativeC2 is maximum rank-local CUDA-fenced wall, C1warmup; profile/calibration
walls are never speedup denominators.
