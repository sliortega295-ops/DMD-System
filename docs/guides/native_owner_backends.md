# Official native owners: execution policies

Official DMD2 and Self-Forcing own backward, clipping, Adam and native
scheduler/EMA clocks inside an atomic Program call. The common lifecycle does
not replay that update or add a second gradient reducer. Visible pure-Call
policies are a separate interface.

## Complete official profiles

For SD1.5 GAN, [native FSDP2](dmd2_fsdp2_reference.md) is the primary reference;
FP32 role-DDP below is supplemental. The full optimized profiles are
`optimization_profile: "dmd2-sd15-resident-tree-v1"` (Resident/DDP, fused AdamW,
activation retention, classifier packing and update-scoped casts) and
`"self-forcing-wan-fsdp2-r6-v1"` (native FSDP2, phase/text-KV retention,
checkpoint reuse, fused operations and paired Teacher CFG blocks).
Official Self-Forcing is not W4 LiveWan. Use [the complete configurations and
launch policy](official_optimized_training.md) for these A100 world16 profiles;
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
single-device snapshot planning, use [resident budgets](resident_owner_training.md),
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
