# LiveWan C51 full-bundle planner opt-in v1

Status: **CPU/source integration only; Pending cluster validation**. The locked
C34 canonical profile, its five existing planner implementations and the
69-frame request remain unchanged. No historical latency is a cost on this
source. C51 reduction/backend numerical status is not promoted from REVIEW.

## Scope and provenance

The source origin is `aa30527db5d048e2ab23222eef587cb7cfe909f6`. Only the
accepted C51 path is transplanted: C34 Generator modulation/residual layout,
Score residual gate and LayerNorm, pinned Score FA3 plus exact per-checkpoint
SAC, Generator current-input read-only FA2 split-KV, and no-grad fused RMS.
It does **not** enable C37 step1 overlap, C39 step0 delivery, C40 shared-input
LoRA, C44 FFN-down caching, merged LoRA, new capacity thresholds or training
math. Current geometry, native action/transport observers and asynchronous
commit calibration hooks are retained, not replaced with the older worker.

The contract stays actual global B16, 368x640x81, two continuous cycles,
G1/C5/EMA0 per cycle, continuous shifted sigma, original sample/RNG/version
identities and pinned precision/backends. The C51 source template has expanded
SHA256 `a50e4e3a969a513dae0399878f7e4f25e23085ee119e1175cad83192c0c6f7bb`.
Unrequested shape, source, sampler/RNG, backend, placement or mixed candidate
changes fail before model construction.

## Existing public workflow

Use the existing `scripts/distillctl.py workloads` interface; no second runner
or training stack is introduced. All commands below run from the checkout.

```bash
python scripts/distillctl.py workloads candidates \
  --workload livewan_crossscale_pipeline_best --include-livewan-c51
```

This is a separate uniform decision cohort. The matched FSDP control is
`joint-livewan1-fsdp-m0-g0-r0-bundle1-control`; C51 is
`joint-livewan1-pipeline-m1-g1-r1-bundle1-c51`. Its four other controls retain
the C31-C34 runtime paths. Omitting the flag preserves the original IDs and
five candidates exactly. C51 is an indivisible accepted bundle: it is not
evidence that its individual factors can be independently scheduled/ablated.

Prepare a **new, source-bound** assets JSON using the existing canonical
LiveWan schema and actual authorized node/source/run locations. Both control
and C51 must include the extra `score_flash_attn3_kernel` directory asset:

```json
{
  "path": "/absolute/pinned/flash-attn3/artifact",
  "identity_file": "metadata.json",
  "sha256": "f1b4b0ef991c0966f7d848777154c0591c4580968fb6da33f6b2a2829a109309"
}
```

It is mapped to `DISTILLSCHEDULE_SCORE_FA3_KERNEL`; the real helper still
checks all five fixed artifact hashes and the imported custom-op schema. Do
not substitute FA2, another FA3 version, a performance environment override or
a source tree from a different HEAD. Keep both nodes clean at the same new
HEAD and revalidate the exact existing environment, assets and ownership.
The old source's default artifact directory was
`/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/hf_home/hub/models--kernels-community--flash-attn3/snapshots/acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2/build/torch210-cxx11-cu128-x86_64-linux`.
This is source provenance, not a live existence/identity check.

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

Qualification is part of that existing calibration workflow: `_prepare` first
runs both same-source two-cycle native trajectories, the real summary and
comparison, and the unchanged canonical timing-gate v2 policy. It does not
reuse a historical trajectory as fresh qualification. Native trajectory and
timing producers publish the actual lowered flat plan plus digest atomically,
once per node's local rank zero, before models/Full5 timers. C51 summaries,
comparison, gate, actual node header and worker gate loader verify the producer
files and original shape/source/plan hashes; canonical C31-C34 gates do not
inherit these new opt-in requirements.

Search and lower with the existing common interface (this does not launch):

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

A bundle layout requires joined, source-qualified calibration results;
hand-written or historical costs are rejected. Uncalibrated controls remain
rejected, not assigned nearby recipe timings. The common simulator enforces
the existing all-rank static-plus-conservative-additional-peak capacity model.
Only a separately authorized native validation run may use
`--execute-native --validation-dir /absolute/project/evidence/c51-v1-validation`
instead of `--emit-native-plan`, with its explicit budget and new run ID.
Alternatively replay the saved plan through the existing CLI:

```bash
python scripts/distillctl.py workloads run \
  --workload livewan_crossscale_pipeline_best \
  --search-plan /absolute/project/evidence/c51-v1-selected.json \
  --assets-config /absolute/project/c51-v1-assets.json --run-id c51-v1-replay \
  --coordinator-output /absolute/project/evidence/c51-v1-replay
```

## Target-runtime CPU gate before any GPU work

Use the already validated Torch 2.10 interpreter; install nothing. This gate
has no skips on that runtime. It exercises the real CPU Torch checkpoint,
autograd, RNG and Adam state, with only CUDA custom-op discovery/metadata
substituted. The actual production proxy must report its two independent SAC
replay reads; a newer local Torch result does not establish the 2.10 census.
The artifact test executes the actual worker metadata-writer statements and
actual summary/compare/gate/node-header/worker-loader chain, with explicit
synthetic CPU tensor-state/receipt fixtures rather than GPU claims.

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

Local characterization: 255 passed, zero skipped, 18.02 seconds on Torch
2.14.0+cpu **without** the require-2.10 flag. The actual production-2.10
receipt gate and target GPU validation are still NOT_RUN in this CPU lane.

## Required target evidence still missing

- Fresh two-cycle, same-source, all-16-rank trajectory parity including
  samples/RNG and G/C/EMA versions, outputs/loss and trainable/optimizer state.
- Actual `planner_c51_bundle` identity and operator receipts: Score FA3/SAC,
  Score gate/LayerNorm, split-KV eligible/reference call reconciliation,
  restored RMS/modulation/gate/cast-cache scopes and original grad fallbacks.
- Both anchors' all-rank asynchronous operation calibration, post-setup
  allocations, reserved/driver peaks, unchanged cgroup OOM and clean teardown;
  current source/environment/assets/hardware bindings must agree.
- Matched unprofiled cycle2 timing and repeat/CV required by the experiment
  contract. CPU fixtures and historical 42-second observations establish none
  of these. Strict winner, formal speedup and independent G/Score/link service
  or arbitrary pool-DAG admission remain unavailable.
