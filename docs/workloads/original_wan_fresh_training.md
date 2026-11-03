# Original Wan: reuse a selected strategy with fresh input cohorts

`auto-run` prepares and searches on a fixed, paired calibration cohort. The new
`prepare-stream` / `train-stream` path instead advances input cohorts and absolute
source steps while retaining the selected original engines and optimizer/EMA
state. Both use the original Wan/Helios-pyramid no-GAN mathematics, not the
separate Euler/Diffusers fixture protocol.

## Preparation

Provide a JSON object with one `cohorts` array. Each array item is a complete
`WanInputSources` object as documented in `original_wan_auto_run.md`: sixteen
feature paths, a seed and cohort index. There must be 2..64 entries with distinct
(seed, cohort) identities. Source clips can recur if intentionally declared;
the system does not claim that videos are globally unique. The task keeps
geometry, rank, horizons, update ratio and EMA interval fixed for this run.
Only absolute steps advance: cycle i starts at first_step + i * fake_updates.

```bash
python scripts/distillctl.py workloads native prepare-stream \
  --task task.json --sources cohort-sources.json \
  --training-config training.yaml --output /shared/wan-input-stream --execute
```

Without --execute this validates only. --resume reuses complete, matching input
captures; partial captures are not silently overwritten or resampled. This is
input-preparation resume, NOT a resumable training checkpoint.

## Training

After validating a selected plan and its matched-reference result, launch the
same original worker on each already allocated node:

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank=0 --master-addr=NODE0_HOST --master-port=29402 --max-restarts=0 \
  scripts/distillctl.py workloads native train-stream \
  --task /shared/new-wan-task/campaign/task.json \
  --assets /shared/new-wan-task/campaign/assets.json \
  --plan /shared/new-wan-task/campaign/selected-plan.json \
  --stream /shared/wan-input-stream/stream.json --output /shared/wan-training --execute
# Other node: node-rank=1, otherwise identical paths and rendezvous.
```

The complete stream is validated before model construction. Engines are built
once from the supplied original initialization. At completed-cycle boundaries,
only trace/task bindings and input-derived Python caches change. Parameters,
Adam moments, gradient buckets and EMA persist. Every rank agrees on cohort,
order and versions before running its original handlers. Selected G/R/C order is
retained; EMA operations are inserted using the current absolute-step cadence,
not blindly copied from the first cycle. Current graph validation checks every
operation and retained-window constraint.

The plan is reused as an execution POLICY, not as an accurate latency/capacity
prediction for new random inputs and exits. New data can still expose an OOM or
numerical issue; these fail rather than shrinking inputs or relabeling timings.
The output distinguishes fresh cohorts from fixed-trace diagnostics, reports
model-cycle time and input-plus-execution wall time separately, and saves final
trainable/Adam/EMA evidence. Comparisons reject different stream identities.

CPU tests execute real Adam/autograd on explicitly small scalar helper fixtures
through the original dispatcher and show parity across legal window orders.
They are NOT original-Wan model quality, CUDA/FSDP correctness, full GPU capacity
or throughput evidence. The full 16-GPU stream/restart path requires its own
execution receipt. This does not support mid-run geometry changes, automatic
checkpoint recovery, GAN, or other model protocols.

## Native zero-history branch (explicit Resident scope)

Set `stage_schedule_policy: source_native_amplified_v1` in a **new task** when
preparing fresh cohorts that can contain zero history. The existing default
`source_normal_only` and immutable benchmark traces are unchanged. Neither
history dropout nor seeds are altered to avoid this branch.

The new policy preserves the original three synchronized initial exit draws.
For a rank whose three history sums are all zero, the original scheduler uses
twice the stage horizon and replaces the initial exits with rank-local native
draws. Other ranks retain their initial exits. Capture records both decisions,
the actual rank, doubled budgets, and actual call counts; the worker checks the
predicate against the payload and validates each rank's own executed census.
There are no padded or discarded model calls. Capture uses reproducible isolated
CPU input streams, **not** the RNG continuation of an earlier upstream run.

This scope requires Resident forward execution without parameter collectives.
FSDP is explicitly rejected: native rank-local exit counts can differ and its
forward all-gathers do not yet have a collective-alignment implementation.
The stream-capable `wan_ar.incumbent21a_stream.v2` preset accepts this policy at
its original geometry. Graph limits, parameter/input version checks, optimizer,
EMA and capacity admission are not relaxed. Fixed-trace v1 is not relabeled.

Use the public `distillctl train` entry with `data.input_kind` set to
`original_wan_cohort_stream.v1`, `data.prepared` pointing to `stream.json`, and
an explicit checkpoint interval. Only complete-cycle, same-source, same-stream,
same-topology checkpoints can resume; transient graphs are rebuilt. CPU capture
and replay tests establish branch/census behavior, not model quality or GPU
restart equivalence.

<a id="original_wan_auto_run"></a>
## Original Wan: new task inputs to existing optimization strategies

This entry continues PR #33 on the original Wan1.3B / Helios-pyramid no-GAN
protocol. It does not use the newer single-device Euler test protocol and does
not invent kernels. The old five canonical profiles remain unchanged.

### What now happens automatically

Given a valid `OriginalWanTask`, matching model initialization assets and new
pre-encoded features, one command per preallocated node validates source geometry,
prepares native history and sampled inputs, writes an immutable rank16 trace,
creates the existing seven-arm strategy campaign, calibrates admitted original
backends, selects a plan and runs native confirmation. No manual trace hash or
sequence of calibration/assembly/plan commands is needed.

Candidates are FSDP control or Resident + async128/Tree, chunks 512/1024/2048 and
eligible no-grad LoRA epilogue. FSDP remains chunk512/non-fused. All original
Generator, Score, Critic and EMA functions are retained.

### Inputs

`--task`: existing original-Wan task JSON. Resolution must be a multiple of 64,
frames 4k+1, G1/Fn 1..64, LoRA rank must match the initialization, and three stage
horizons are explicit. Actual exits remain sampled: eight total horizon positions
do not imply eight model calls on every rollout. Hardware remains 2x8 GPUs,
physical B1/global16. These are task requirements, not parameters changed to
obtain speedups.

`--sources`: see `source_features.template.json`. Sixteen distinct local `.pt`
or safetensors files, one per global rank, plus seed and cohort. Required keys:
`vae_latent` of shape `[sections,16,latent_frames,H/8,W/8]`, `prompt_embed` of
shape `[tokens,4096]`, and explicit binary `attention_mask` of shape `[tokens]`.
Text and mask can have a leading B1. Torch files use safe `weights_only` loading.
No VAE encoding, raw-video resizing, missing-mask reconstruction, old-trace
resizing or replacement-clip sampling is performed. New resolution requires
features actually encoded at that resolution.

`--model-assets`: `model_root`, `bootstrap`, `negative_prompt`, as in
`model_assets.template.json`. Existing original asset validators compute the
hashes. Changing rank needs a matching explicit initialization; trained adapters
are not silently reset or converted.

`--training-config`: the original template's input-preparation settings. Only
implemented history dropout/corruption and timestep sampling settings can differ;
optimizer, loss and scheduler changes that the worker would ignore are rejected.
Shape, frequency, rank and stage horizons come from the task, not inactive YAML
fields. GT-history does not apply the non-GT saturation augmentation.

### Native branch boundary

The existing measured world16 backend admits the normal globally synchronized
pyramid schedule. The upstream all-zero GT-history branch samples amplified exits
independently per rank. Capture refuses that branch without resampling, changing
dropout or relabeling it as the normal path. Empty initial history and T2V history
dropout can therefore fail admission. Do not search for a passing seed and present
it as the original data distribution. Supporting this branch requires a separate
collective-safe contract; it is not fixed by dropping validation.

The explicit operation-qualified CPU/Python RNG streams define a new paired
experiment, not byte-identical continuation of an old CUDA RNG stream. Comparison
arms consume the same captured inputs.

### One command per node

Both already allocated nodes must see identical source and shared absolute data
and output paths. Run on each node, changing only `--node-rank`:

```bash
python scripts/distillctl.py workloads native auto-run \
  --task task.json --sources source-features.json --model-assets model-assets.json \
  --training-config configs/distillschedule/templates/helios_stage3_384x640.yaml \
  --output /shared/new-wan-task \
  --node-rank 0 --master-addr NODE0_HOST --master-port 29401 --execute
# Second node: --node-rank 1, same host, port, input and output paths.
```

Without `--execute`, inputs are validated but no campaign/GPU is started. The
default budget is seven component arms plus one selected-plan confirmation,
eight distributed launches total. `--without-fusion --max-launches 5` requests
four non-fused arms plus confirmation. An insufficient budget is rejected before
capture, not used to truncate candidates silently.

`--resume` retains campaign journals and budgets. Unknown interrupted native
launches are not automatically repeated. Failed input preparation is preserved;
no automatic resampling or trial-counter reset. Readiness is atomic. There is
no SSH, scheduler submission or allocation of new GPUs.

Separate `capture-inputs`, `prepare-task`, `prepare-campaign`, `calibrate`,
`assemble`, `plan` and `run` commands remain available. Input capture/preparation
uses the same flags above without rendezvous arguments, and requires `--execute`
to write new captured inputs.

### Evidence

CPU tests call the actual native dataset/history/scheduler functions and replay
captured schedules through the original sampler with an injected model boundary.
Controller tests inject workers. Neither is a pretrained-model, GPU-collective,
quality or speedup result. Full-model GPU validation remains NOT_RUN. Numerical
REVIEW remains REVIEW, not PASS.

Campaign cycles are paired fixed-cohort diagnostic runs, not streaming fresh
long-training data. Helios14B, H3, LiveWan, GAN, arbitrary GPU pool layouts and
automatic shared-backbone choices are not parameterized by this entry.

<a id="parameterized_native_backend"></a>
## Parameterized native backend: measured actions to real training

This is the next opt-in layer on PR #29. It connects the common parameterized
ExecutionGraph, CostModel and event-frontier solver to actual model
forward/backward/Adam/EMA calls through PreparedRolePlan. It does not replace
main or any of the five canonical best-known profiles.

### Explicit scope

Two model adapters are provided:

* `torch-reference`: small randomly initialized flow network, **test fixture only**.
  This runs real Torch autograd/LoRA/Adam, not mock handlers or synthetic times.
* `diffusers-wan`: the actual `diffusers.WanTransformer3DModel` (optional pinned
  diffusers 0.35.2). Tiny random configurations validate the architecture on CPU;
  `fixture=false` requires three explicit local config/safetensors checkpoint
  directories and a local pre-encoded condition dataset. Nothing is downloaded
  from Hugging Face and no checkpoint is guessed from the model name.

The new protocol is
`euler_last_step_detached_dmd_post_g_independent_fake_v1`:

1. Euler flow rollout with explicit step count and shifted sigma schedule;
   only the final denoising call carries the Generator gradient.
2. Frozen real/fake score inference constructs a normalized detached DMD input
   gradient. The normalizer floor is an explicit algorithm parameter. Teacher
   guidance, when requested, uses cond + guidance*(cond-uncond).
3. One Generator AdamW commit, then n independently bound post-commit rollouts,
   n ordered Critic flow-matching commits and declared EMA positions.
4. Every logical action/microbatch has deterministically bound initial noise,
   score noise, sigma and sample identity. Scheduling does not alter consumption
   of a shared global RNG stream. This is the declared new protocol, not a
   relabeling of previous experiment traces.

The existing `runtime/dmd_boundary.py` surrogate and critic-loss primitives are
reused. This is **not** the historical Helios-pyramid Wan contract, LightX2V H3
Grid1000, or DiffSynth autoregressive H3. GAN, stochastic early exits and
previous/current history need their original training-protocol adapters.
A missing DiffSynth training source is not replaced with a different algorithm.

### Parameters that now reach native computation

Resolution and frame count follow the declared compression lattice (Wan2.1:
4 temporal / 8 spatial, first-frame convention), then the native patch lattice.
Invalid sizes are rejected, not silently rounded. The latent shape, number of
rollout forwards, Fake updates, microbatch/accumulation, text dimensions, LoRA
rank/alpha and EMA positions derive from NativeTask rather than a locked recipe.
Each task instance has a new semantic identity. Changing it requires matching
calibration; old Full5 evidence is never promoted to the new task.

Native implementation choices currently include ordinary/full-forward
non-reentrant checkpointing, an exact retained-output budget, and supported
reference or existing CUDA BF16/FP32 no-grad LoRA epilogues. Autograd calls use
the reference epilogue; no unsupported training kernel is substituted. Fallback
reasons are exposed by `validate`. A rank/shape-specific fusion need not exist
for the rest of the native flow to run.

Placement is **single process, single device, separate resident role models**.
WORLD_SIZE>1 fails before constructing the model. This layer does not yet
provide FSDP/HSDP, automatic shared backbones, CPU activation offload or distributed
pool scheduling. Do not represent the serial event planner as having executed
multi-GPU overlap. Large checkpoints still need sufficient host/device memory;
a parameter/optimizer lower-bound gate is not a workspace capacity proof.

### Commands

Run from a clean source checkout (the root runtime primitives are required):

```bash
python -m pip install -e '.[test,native]'
# For actual Wan architecture/local checkpoint support:
python -m pip install -e '.[test,native-wan]'

TASK=configs/distillschedule/native_examples/torch_g1f3.json
python scripts/distillctl.py workloads native validate --task "$TASK"
python scripts/distillctl.py workloads native calibrate --task "$TASK" \
  --output /tmp/native-calibration-UNIQUE --execute
python scripts/distillctl.py workloads native plan --task "$TASK" \
  --calibration /tmp/native-calibration-UNIQUE/calibration.json \
  --output /tmp/native-plan-UNIQUE.json
python scripts/distillctl.py workloads native run --task "$TASK" \
  --plan /tmp/native-plan-UNIQUE.json --output /tmp/native-run-UNIQUE \
  --cycles 3 --execute
```

Use `wan_tiny_cpu.json` for the real tiny Wan model and
`wan_local_checkpoint.template.json` as an explicitly incomplete local-checkpoint
example. Production conditions use safetensors keys `prompt_embeds` and, for
nonzero teacher guidance, `negative_prompt_embeds`. Each is
[samples, text_tokens, text_dim]. A cycle consumes `(1+n)*effective_batch` samples
in G/r0/r1/... order. Cohort exhaustion stops the run; data are never recycled
silently. Image conditioning and autoregressive history are not accepted as
unrecognized extra fields.

`validate` and native commands without `--execute` do not import a model or
start training. `plan` validates task/code/artifact identities and operates on
CPU metadata; it does not allocate a GPU model. `run` reloads and rechecks actual
model/data/device/software identities before accepting the plan. It also
reconciles selected order, per-action timings and memory against measured costs;
re-hashing edited predicted numbers is insufficient to admit them.

All outputs are new immutable directories/files. Failed runs keep failure
records. No model weights, conditions or raw large tensors are committed.
The output final-state.pt is an inspectable weights-only tensor artifact, **not
an advertised resumable training checkpoint**.

### Calibration and native validation

Calibration first runs an independent serial reference for two cycles, then
compares every implementation's complete trainable/Adam/EMA state with the
reference. Frozen weights must remain unchanged. G and C adapters have disjoint
trainables. Each admitted implementation collects real cold and warm action
observations; only warm components enter the current cost model. These are
synchronized calibration observations, not native end-to-end speedups.

The common frontier solver chooses a serial topological order. Shared/static
storage is accounted once and rollout output storage remains live through the
owning Critic return. A separate `retained_limits` constraint in the common
solver enforces the exact output budget without inventing a second GPU rank.
Non-greedy orders can actually reach the common native dispatcher. The original
solver's event-family/beam/state-budget limitations still apply.

On CPU, action time is measured but memory is tensor accounting, **not RSS**.
On CUDA, action workspace is a measured allocator peak delta, not a guarantee
about driver-free headroom/fragmentation. Search reuses calibrated components;
execution rechecks current capacity and preserves the reservation. The current
serial model does not predict asynchronous contention or new GPU layouts.

`run` performs fresh matched two-cycle correctness before separate timed
execution, restoring original trainables and Adam/EMA state between phases.
The timed process has one warm-up cycle and then real consecutive updates. No
state probes/final tensor serialization are in that native timing boundary.
Gradient clipping/optimizer work remain inside it. No quality or speedup claim
is inferred from a PASS receipt. Calibration/confirmation work is explicitly
additional to the requested timed cycles and must be included in a user's new
epoch budget; no completed historical search budget is reset.

### Validation boundary

Dependency-light task tests cover invalid/unknown parameters and no-Torch
validation. Torch tests execute real updates across G1/F1/F2/F5, 1/4/8 denoising
steps, different video shapes and LoRA ranks, microbatch/accumulation, EMA,
checkpointing, stale source/task/cost rejection and edited-plan rejection.
Optional Wan tests execute real tiny Wan forwards/backwards, full-state parity,
calibrate-search-native execution and loading local generated checkpoints.
Tiny random checkpoints are integration fixtures, not pretrained quality data.

GitHub Actions runs these on CPU with pinned Torch/diffusers. A100 execution,
full-size pretrained training, CUDA epilogue canary, long-horizon quality,
DiffSynth H3 and multi-GPU parameterized execution remain separately unvalidated.

<a id="parameterized_original_wan"></a>
## Parameterized ORIGINAL Wan optimization bridge

This opt-in layer joins event-frontier search to the **existing** Wan1.3B
Helios-pyramid no-GAN implementation. It does not substitute the separate
single-device Euler/Diffusers training protocol. The five canonical profiles
and the original model-based search interface remain unchanged.

### Implemented versus validated

The original task schema routes through `workloads native`. It generates
world16 FSDP or Resident/async128/Tree candidates, modulation chunks 512/1024/2048,
and the existing Generator no-grad LoRA epilogue. FSDP control remains
chunk512/epilogue-off. Choices reach original constructors, optimizer, loss,
score functions, pyramid sampler and EMA helpers.

Input capture and task preparation are now connected: see
[Original Wan auto-run](#original_wan_auto_run). It starts from NEW pre-encoded
features, invokes original model-free input primitives, generates a task-bound
trace and prepares the existing budgeted calibration/search/confirmation.
No manual trace hash entry or old-trace resizing is required. VAE/text encoding
itself is not automated, and the all-zero rank-local amplified branch remains
unsupported without changing/resampling the source distribution.

Fresh-cohort training is a separate opt-in path described in
[Fresh training](original_wan_fresh_training.md). It advances absolute steps and
EMA cadence while retaining the selected engines and optimizer state. Existing
calibration/confirmation still repeat their fixed cohort for paired diagnosis;
those results must not be called fresh-data training.

CPU tests exercise capability generation, plan selection, capacity rejection,
original-helper binding, per-block chunk effects, actual native input/scheduler
primitives, and persistent dispatcher/Adam fixture state. They are **not**
full-Wan or CUDA tests. Multi-GPU execution, full-model numerical parity, speed
and generation quality remain NOT_RUN for this integration.

### Configurable contract

`OriginalWanTask` describes height/width multiples of 64, frames 4k+1, G1/Fn
(1..64), LoRA rank (1..256 matching bootstrap), original Generator-update
boundaries, three stage horizons and EMA cadence. Active pyramid sampling uses
`stage2_inference_steps`; the ordinary denoising list is inactive and cannot be
changed as if it controlled compute. Random exits are still sampled/captured,
so a horizon does not imply an identical call count on every rollout.

Still fixed: Wan1.3B 30-block model, world16/two nodes of eight, B1/global16,
original precision/history semantics, three stages and no-GAN. This does not
parameterize Helios14B, H3, LiveWan, arbitrary pool splits, automatic backbone
sharing or arbitrary intermediate-state offload.

#### Explicit historical benchmark input scope

`stage_schedule_policy="benchmark_v4_strict_global"` is an opt-in extension for
parameter-generalization systems experiments using the previously declared v4
strict-global exit scope. It preserves sampled history dropout, zero-history
tensors and operation-qualified RNG; it does not resample passing inputs. For
all-zero history it retains the native doubled timestep grid, but the exits are
still the rank-zero synchronized exits declared by this benchmark. This is not
the upstream rank-local amplified-exit distribution, and must not be reported as
full-distribution training or quality equivalence. Every changed task needs new
trace identities and measured costs.

The default `source_normal_only` capture remains unchanged and rejects all-zero
history. Its serialized task identity is unchanged. No default profile, original
training helper, loss, gradient or optimizer is modified by this input scope.

### Search and execution

`validate` inventories candidates without model construction. `auto-run` is
one command per allocated node for new input preparation and the existing
campaign. Individual `capture-inputs`, `prepare-task`, `prepare-campaign`,
`calibrate`, `assemble`, `plan`, `run`, `prepare-stream` and `train-stream`
commands remain available for controlled use.

The default campaign performs seven component calibration arms and one native
confirmation, eight distributed launches total. `--without-fusion
--max-launches 5` requests four non-fused arms plus confirmation. Insufficient
budgets are rejected, failed arms remain failed, and unknown interrupted
launches are not automatically repeated. --resume preserves journals and
counts. CPU fixture costs cannot launch native GPU execution.

Plans only rank source-, hardware-, task- and asset-matched observed choices.
The selected layout is installed before NCCL/model construction, and worker
checks cover actual reduction, block chunks, fusion invocation and trace-derived
call counts. Final trainable, Adam and EMA states are compared with a matched
FSDP control; numerical REVIEW never becomes PASS merely because a process
completed. Calibration is diagnostic, not a formal speed baseline.

### Remaining scope gates

Real 16-rank original-Wan smoke/parity and performance, support for the upstream
amplified-history branch, raw-video/text encoding if needed, resumable long
training, and the other four native protocol/capability adapters remain to be
validated or implemented. Do not promote CPU integration to those claims.

<a id="wan_incumbent_bundle_v1"></a>
## Wan incumbent21a-v1: task-bound opt-in bundle

This is a bounded port of an existing mechanism, **not a new optimization or a
promotion of historical timing**. Existing independent placement/chunk/no-grad
LoRA-epilogue/window choices, IDs, defaults and campaigns remain unchanged.

### Public route

The existing `workloads native` front door remains the execution route:

1. `validate --include-incumbent-bundle` advertises the typed candidate only when
   Triton is available and the task is exactly `OriginalWanTask()`.
2. `calibrate --implementation <strategy.json> --execute` runs the original
   native world16 handlers for two fixed-trace cycles and emits per-rank action,
   storage-lifetime, backend-effect and precision receipts.
3. `assemble --rank-receipts ...` requires sixteen completed current-source rank
   files, reconciles the bundle effects, and emits a measured collection/costs.
4. `plan --calibration <control-collection> <bundle-collection>` calls the SAME
   `distillschedule.search.service.select` and event-frontier scheduler used by
   the existing adapter. It saves a source/asset/cost/graph-bound plan. The bundle
   can lose to the control; no historical duration is a search cost.
5. `run --plan <saved-plan> --execute --cycles 2` independently validates the
   plan/costs, then uses `OriginalWanSession` and the original prepared action
   handlers. It does not dispatch to the historical runner. GPU work is still
   explicit and requires two nodes of eight GPUs; this document launches none.

CPU factory example, suitable for writing explicit strategy files:

```python
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import implementations
choices = implementations(OriginalWanTask(), triton_available=True,
                          incumbent_bundle_available=True)
control = choices[0].to_dict()  # Original FSDP control, all bundle fields off.
bundle = choices[-1].to_dict() # resident/chunk2048/no-grad epilogue + typed v1.
```

Example command forms after source review/application, using explicit paths and
an already validated environment (do not execute without allocation authority):

```bash
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native validate \
  --task "$RUN/task.json" --include-incumbent-bundle --output "$RUN/capabilities.json"
# On EACH existing node, under its already authorized 2-node torchrun allocation:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native calibrate \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --implementation "$RUN/bundle-strategy.json" --output "$RUN/bundle-calibration" --execute
# Repeat the same command with the independently declared FSDP control strategy.
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native assemble \
  --task "$RUN/task.json" --rank-receipts "$RUN/bundle-calibration"/rank_*.json \
  --output "$RUN/bundle-costs"
# Assemble the control separately, then:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native plan \
  --task "$RUN/task.json" --calibration "$RUN/control-costs/collection.json" \
  "$RUN/bundle-costs/collection.json" --output "$RUN/selected-plan.json"
# On EACH existing node, under its authorized allocation:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native run \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --plan "$RUN/selected-plan.json" --cycles 2 --output "$RUN/confirmation" --execute
```

A CPU fixture collection cannot launch the worker. Current-source GPU numerical
acceptance remains required against matched reference trainables/Adam/EMA;
`compare_runs` retains REVIEW on failure. The existing worker does not promote a
completed run or its action-calibration timings to formal speedup eligibility.

### Capability and semantic boundary

`WanCandidateBundle` v1 has five explicit booleans, all default-off:
`grad_lora_fused_epilogue`, `modulated_norm_fusion`,
`generator_rollout_cuda_graph`, `generator_nograd_lora_cast_cache`, and
`critic_grad_cuda_graph`. Only the all-off or this coupled all-on bundle is
admitted; arbitrary five-axis cross-products are rejected. The old no-grad
LoRA epilogue is still its existing separate axis and must be enabled here.

Admission is restricted to the original 384x640/33-frame, B1/world16, rank256,
G1/C5/EMA5 task, steps10-14, original bootstrap G2/C10/EMA10, fixed legacy trace
bundle and binding, negative-prompt manifest, and actual base-model file hashes
reconciled with the immutable bootstrap inventory. New task geometries, new
traces, fresh-cohort streams, more than two cycles and another bootstrap fail
closed. Rollout signatures, pointer/layout guards, dropout restrictions, first-
cycle capture limits and literal G3/G4/27-calls/5-entries/three-shape census stay
unchanged. The common planner adds only a legal schedule restriction r0->r4;
window/memory choices remain measured decisions, not physical batching.

The exact original reducer, gradient convention, clipping, Adam, checkpointing,
EMA and FP32 trainable/BF16 frozen-storage policies remain outside graph replay.
Critic graphs return parameter gradients to original accumulators/hooks. Norm
counters are process-shared and are NOT added across duplicate role receipts.
Cast reuse retains the original Tensor.to result and ordinary LoRA-A GEMM.
Merged LoRA, rejected input GEMM, detached Score graphs, G-grad graphs, fused
optimizer and foreach EMA are not registered or activated by this bundle.

### Precision and evidence identity

The port uses the CURRENT generated Wan precision contract, equally for control
and bundle: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, declared matmul/cuDNN TF32 policy,
and no `NVIDIA_TF32_OVERRIDE`. Runtime flags, matmul precision, package/runtime
versions and observed parameter dtypes remain bound in actual receipts. No
candidate-only precision override is introduced.

The historical 21a observation was 20.31478561926633 s C2 world-max on A100 and
used a nonempty driver override. It is provenance only; this source/precision/
hardware combination has no measured time yet and no 20.3-second promise.
Historical comparison status contains `50C`, but that particular evidence had
only TWO cycles and one seed. It is not 50-cycle equivalence or quality proof.

Every bundle-bearing component cost includes all five capability fields and
actual SHA256s of the imported mechanisms plus existing cast/epilogue code.
Current source fingerprints, task, full assets, runtime, rank hardware and raw
rank receipts remain bound by the existing collection and saved-plan gates.
Neither old eager costs nor historical timing is borrowed.

### Explicitly not admitted

- Legacy five-profile/macro trajectory/timing lowering and old summarizers:
  `registry.candidate_profile` rejects this new ID rather than weakening eager
  counter assertions. Registry identity/parse works, but execution uses the
  task-bound public route above.
- Source-pinned world8 H100 hardware adapter and world8 logical-slot bridge:
  unchanged and NOT_ADMITTED for this bundle.
- General task-derived graph census: not implemented; literal guards retained.
- CUDA/numerical/current-source timing validation: Pending cluster validation.
