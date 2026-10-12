# Parameterized native backend: measured actions to real training

This is the next opt-in layer on PR #29. It connects the common parameterized
ExecutionGraph, CostModel and event-frontier solver to actual model
forward/backward/Adam/EMA calls through PreparedRolePlan. It does not replace
main or any of the five canonical best-known profiles.

## Explicit scope

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

## Parameters that now reach native computation

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

## Commands

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

## Calibration and native validation

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

## Validation boundary

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
