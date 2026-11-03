# Official native owners: execution policies

Official DMD2 and Self-Forcing own backward, clipping, Adam and native
scheduler/EMA clocks inside an atomic Program call. The common lifecycle does
not replay that update or add a second gradient reducer. Visible pure-Call
policies are a separate interface.

## Complete official profiles

For SD1.5 GAN, [native FSDP2](native_workloads.md#dmd2_fsdp2_reference) is the primary reference;
FP32 role-DDP below is supplemental. The full optimized profiles are
`optimization_profile: "dmd2-sd15-resident-tree-v1"` (Resident/DDP, fused AdamW,
activation retention, classifier packing and update-scoped casts) and
`"self-forcing-wan-fsdp2-r6-v1"` (native FSDP2, phase/text-KV retention,
checkpoint reuse, fused operations and paired Teacher CFG blocks).
Official Self-Forcing is not W4 LiveWan. Use [the complete configurations and
launch policy](native_workloads.md#official_optimized_training) for these A100 world16 profiles;
source locks, geometry, precision, updates and same-topology recovery remain
owner-enforced.

## Per-role DMD2 checkpoint policy

On the pinned Torch2.10/Diffusers0.37.1/Transformers4.51.3/PEFT0.17.1 CPU1 FP32
stack, the SD1.5/SDXL owner exposes scoped UNet checkpoint options. Default
`coupled-v1` uses `owner_execution.native_cycle.gradient_checkpointing` for
both roles. For independent flags, set
`model.parameters.native_checkpoint_policy: "component-v1"` and retain the
original `settings.gradient_checkpointing: true`. The reference checkpoints
both roles; this manual `execution` retains Generator checkpointing and Fake
epsilon activations:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {
      "native_cycle": {
        "generator_checkpointing": true,
        "fake_checkpointing": false
      }
    }
  }
}
```

See [the SD1.5 template](../../recipes/dmd2_sd15/training.component-checkpoint.json)
and [SDXL recipe](../../recipes/dmd2_sdxl/README.md). With `coupled-v1`, replace
the native option with `{"gradient_checkpointing": true}`. CPU manual,
bounded auto and saved-plan execution use the same descriptor, paired preupdate
reference comparison and first-resumed-cycle admission.

Actual Generator requests/recomputes and Fake grad-enabled epsilon retention
are checked separately; no-grad Teacher/classifier-only calls do not count.
Unused or contradictory counters fail, and flags/callbacks/hooks are restored
on success or exceptions. This CPU policy is separate from mixed-precision,
distributed and production-GAN profile admission. For source-qualified tiny
single-device snapshot planning, use [resident budgets](#resident_owner_training),
not an opaque-owner CUDA copy.

Official Self-Forcing exposes only
`owner_execution.native_cycle.fake_gradient_checkpointing` on its exact tiny
CPU1 FP32 profile. Keep `settings.gradient_checkpointing: true`: Generator
ON/OFF references have distinct identities because their causal
cross-attention-cache/gradient behavior differs; this is not a searched toggle.
The [Fake-only checkpoint recipe](../../recipes/self_forcing_official/README.md#explicit-checkpoint-on-reference-and-fake-only-physical-policy)
specifies the source/runtime/profile and actual replay counters.

## Persistent native DMD2 role DDP

`model.parameters.native_role_backend: "ddp-fp32-v1"` installs persistent actual
DDP on Generator and the complete guidance role. Native backward/clip/Adam
remain the only update loop. Rank0 bootstrap synchronizes role weights/buffers;
pretrained frozen tensors must already agree. The backend admits CPU/Gloo or
owned CUDA/NCCL with FP32 state, not the optimized mixed-precision bundle.

Start from [the CPU/Gloo template](../../recipes/dmd2_sd15/training.ddp-fp32.json),
replace local paths and provide independent rank-strided prepared inputs:

```sh
torchrun --standalone --nproc-per-node=2 -m distillschedule.cli train training.ddp-fp32.json
```

Model/trainer devices must agree; CUDA uses `trainer.device: "cuda"` and
`trainer.distributed.backend: "nccl"` with actual local-rank ownership. Global
batch is native per-rank batch × world size, never duplicated samples. Each
rank saves complete role/optimizer state, native/outer RNG, input progress and
clocks. Same-source/topology resume reinstalls wrappers before loading and
rebuilds buckets; `.module` is not part of the checkpoint ABI. Adam commits
once per native update, while both schedulers keep the original Accelerate
world-size multiplier.

Only reference or the exact direct-manual `torch-callable-native-owner-v1`
policy is admitted. Clone-based auto/plan, owner/checkpoint overrides, GAN,
mixed precision, FSDP2 and topology-changing restart are rejected by this
FP32 backend. Its native state/restart tests use CPU/Gloo; CUDA/NCCL capacity
needs its own qualification.

## Official Self-Forcing: release consumed rollout KV

The [explicit template](../../recipes/self_forcing_official/training.consumed-kv.json)
sets `model.parameters.native_cache_lifetime: "release-after-commit-v1"`;
the default is `native-v1`. This resident-owner profile is not a searched toggle.
FSDP2 reference/R6 require `native-v1`; R6's earlier Critic release is separate.

The owner releases its two self-attention KV caches and cross-attention cache
only after native backward/clip and the corresponding Adam/EMA commit. It does
not shorten causal attention or release tensors during replay/backward. The
next native rollout rebuilds caches; source/checkpoint identity binds the
policy, while checkpoints omit transient caches. Release does not read tensor
values on CPU, clear the allocator or synchronize CUDA.

Receipts require one Generator release and the configured Critic count per
cycle (five by default). Bytes count deduplicated owner storage references
relinquished per commit, not allocator-freed memory; other observers may hold
aliases. Native checkpoint-ON/OFF state and restart coverage is CPU-scoped.

See [model environments and test lanes](model_families.md#verification-lanes)
and [performance/source records](performance_validation.md) for validation of
each selected recipe.

<a id="fsdp_boundaries"></a>
## Model-owned FSDP parameter groups

An adapter with a registered outer model method can explicitly name disjoint
child modules as native FSDP groups:

```python
ReplicaSemantics(
    module_bindings={"student_forward": "student"},
    backward_calls=("student_backward",),
    local_batch_size=1,
    forward_interface="registered_methods",
    forward_methods={"student_forward": "dmd_forward"},
    shard_boundaries={"student": ("blocks.0", "blocks.1", "blocks.2")},
)
```

The existing uniform FSDP2/HSDP installation applies `fully_shard` to these
children in the declared order, then to the remaining outer root. The root
registers its single outer method. Child groups use their ordinary `forward`
entries, so the model must call each child as a module rather than directly
reading that child's sharded parameters from an outer private method.

The declaration adds no model-specific training loop. Optimizer objects, groups
and initialized moments migrate to the replacement parameters once. Explicit
all-parameter shadow maps follow the final layout by full parameter name.
Declared immutable frozen roots can use the same grouping without an optimizer
or parameter gradients. A frozen forward can still preserve an input VJP when
its original Program declares a differentiable call.

Paths must resolve to distinct, nonempty child modules with real `forward`
implementations. Ancestor/descendant cuts and cross-group parameter/storage
aliases are rejected. An empty mapping retains existing whole-root identities.
Path order is part of the workload identity; changing a declared path, child
object or parameter after installation fails before the next cycle. Mixed
per-root layouts and role-pool sharding do not accept this declaration.

With `fsdp2_reshard`, native groups can free full parameters between child calls.
With `fsdp2_keep_forward`, visited groups can all stay materialized until their
backward or the complete-cycle boundary. Finalization explicitly reshards each
child and outer root because native `reshard()` is not recursive.

Validation uses small real Torch modules on CPU/Gloo, covering losses,
gradients, Adam moments, all-name EMA, immutable Teacher state, local input VJPs,
and fresh-process checkpoint continuation. Physical parameter-storage samples
are CPU observations. CUDA/HSDP topology, large-model construction peaks,
generation entries and export staging still need their own qualification;
this generic capability does not open an unqualified public model factory.

<a id="resident_owner_training"></a>
## Budgeted resident-owner training

The public `distillctl train` lifecycle can keep a native model owner in place
while admitting an explicit checkpoint policy, calibrating alternatives, or
executing a saved plan. It uses the same serial snapshot/restore protocol for
all three modes. There is no whole-owner CUDA clone and no separate trainer.

### Supported scope

This initial portfolio is the source-qualified, single-device FP32 **tiny
native** SD1.5, SDXL and official Self-Forcing configurations produced by their
recipe fixture builders. They instantiate the actual upstream model classes.
SD uses B1, resolution 16, latent resolution 8 and Fake:Student=2; Self-Forcing
uses 21×4×4 latents, three-frame blocks and Fake:Student=5. Self-Forcing must
keep Generator checkpointing ON; only the noncausal Fake checkpoint policy is
an alternative. The native sampling and update rules are unchanged.

Use each recipe's pinned dependency stack and explicit local source/assets.
Pretrained-size constructors, AMP, distributed owners and other model families
do not inherit this admission. Their reference and optimized-profile entry
points retain their existing contracts. CPU and CUDA qualification are recorded
separately in the [support matrix](model_families.md#support_matrix).

### Configure an explicit budget

Add `execution.resident_budget` to the normal training configuration. All
amounts are positive integer **bytes**. This example describes a 4 GiB CUDA
trial envelope, not a recommended budget for pretrained models:

```json
{
  "schema_version": "distillschedule.training_resident.v1",
  "construction_limit_bytes": 134217728,
  "host_staging_limit_bytes": 134217728,
  "trial": {
    "host_limit_bytes": 2147483648,
    "snapshot_limit_bytes": 536870912,
    "result_limit_bytes": 536870912,
    "scratch_bytes": 134217728,
    "reserve_bytes": 134217728,
    "host_growth_bytes": 134217728,
    "device_limit_bytes": 4294967296,
    "device_state_growth_bytes": 134217728,
    "device_gradient_bytes": 67108864,
    "device_activation_bytes": 536870912,
    "device_workspace_bytes": 134217728,
    "device_restore_bytes": 67108864,
    "device_reserve_bytes": 134217728
  }
}
```

For CPU, the `trial` object contains only `host_limit_bytes`,
`snapshot_limit_bytes`, `result_limit_bytes`, `scratch_bytes` and
`reserve_bytes`. Do not copy CUDA fields into a CPU configuration. Host limits
cover managed tensors and staging, not the entire Python process RSS.

Configuration inspection and budget validation use only the core package;
they do not import Torch, construct a model or probe a GPU. The provider's
optional dependencies are needed when training begins, not to read the request.

The factory checks source, architecture, weight headers, precision, topology
and construction capacity **before allocating the model**. Training also checks
fresh host/cgroup and CUDA headroom. A budget does not turn an unsupported
model into a supported one. `construction-admission.json` records the forecast
and actual managed storage; these variable observations are not checkpoint
identity.

### Direct policy, automatic planning and saved plans

For direct SD1.5/SDXL checkpoint execution, use:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {
      "native_cycle": {"gradient_checkpointing": true}
    }
  }
}
```

Add the budget object above to this execution object. For official
Self-Forcing, replace the native option with
`{"fake_gradient_checkpointing": false}`; retain its Generator-ON model
setting. Direct mode needs no calibration file.

Automatic mode uses `"mode": "auto"`, an explicit
`"memory_limit_bytes": 2147483648`, `"max_candidates": 2`, `"samples": 1`
and `"max_program_runs": 3`, together with the same resident budget. It writes
real measured costs and `planning/selected-plan.json`. To reuse that plan from
its bound initial state, use `"mode": "plan"` and
`"plan": "/path/to/selected-plan.json"`. A source, state, input or policy
mismatch is an error, not a fallback to a hand-written command.

```bash
distillctl train train.json --cycles 3 --output runs/first
distillctl train train.json --cycles 6 --resume /path/to/checkpoint --output runs/resumed
distillctl export train.json --checkpoint /path/to/checkpoint --output exports/student
```

The first policy admission compares reference and candidate from one snapshot
and accepts exactly **one** real cycle. Calibration trials restore their initial
state and do not advance the data iterator. Later cycles reuse the admitted
owner and bind fresh inputs and parameter versions. Native effect counters are
written into the cycle metrics.

### Resume and failure behavior

Use the same source, device, topology, mathematical configuration and resident
budget when restoring a complete-cycle checkpoint. The lifecycle reinstalls
the selected physical policy before restoring roles, Adam, schedulers, EMA,
RNG, data positions and native clocks. Transient snapshots and caches are
rebuilt. A resumed selected owner re-admits its next actual cycle; it does not
consume an extra sample or search again.

Admission receipts are published before a speculative cycle is accepted.
Publication or numerical failure restores the snapshot. A failure during an
ordinary continuation invalidates that live owner and prevents publishing a
new checkpoint; restart from the last complete checkpoint. Closing the public
model lifetime revokes its resident admission.

Inspect `construction-admission.json`, `owner-admission.json` or
`plan-admission/execution.json`, per-cycle metrics and the complete checkpoint.
Accepted inputs advance exactly one logical cycle and its native clocks while
retaining parameter ownership and the declared capacity reserves.

The inference export contains the current Student (not the optimizer or a
resumable training state). Existing recipe generation and decoding commands
consume it unchanged.

### Validation

The public path has run on A100 with the actual tiny SD1.5, SDXL and official
Self-Forcing classes. For each family, three consecutive cycles through
automatic selection match the native reference state, and direct execution
matches an uninterrupted run after a fresh-process checkpoint restore. Current
Student exports decode to images or video. The shared saved-plan/restart path
is additionally exercised with SD1.5, including changed-plan rejection.

The tests retain one native owner, check parameters/Adam/scheduler/EMA/RNG/data
positions and clocks, and leave the unrelated device's RNG unchanged.
`tests/test_training_resident_lifecycle.py` covers the CPU path;
`tests/gpu/test_training_resident_public_gpu.py` is the opt-in CUDA check.
These checks use small randomly initialized native models; they do not measure
pretrained quality or performance.

The `official-models-cpu` workflow runs the public resident lifecycle in separate
DMD2 and Self-Forcing jobs. Their required receipts cover real native updates,
automatic selection, saved plans, rollback and fresh-process restore; missing
or skipped required cases fail the job. CUDA qualification remains separate.
