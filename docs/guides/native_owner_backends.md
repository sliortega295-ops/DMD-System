# Official native owners: checkpoint policy and role replicas

The official DMD2 flow owns backward, clipping, Adam and native scheduler
updates inside one atomic `Program` call. Replaying that whole call would
duplicate training. The common runtime therefore preserves its existing owner
and cycle/checkpoint lifecycle rather than treating it as an ordinary pure
forward or adding a second gradient reducer.

The independent opt-in integrations below do **not** change
existing recipes or promote the separately measured mixed-precision native
benchmark portfolio into a fully integrated public training backend.

## Per-role DMD2 checkpoint policy

Use `model.parameters.native_checkpoint_policy: "component-v1"` and the
original `settings.gradient_checkpointing: true`. The reference retains
checkpointing on both Generator and Fake. A manual policy can keep Generator
checkpointing but retain the Fake epsilon forward's activations:

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

See [the complete SD1.5 template](../../recipes/dmd2_sd15/training.component-checkpoint.json).
The same model option applies to the actual SDXL owner, not a substitute UNet.
The existing bounded CPU calibration/search and saved-plan path use this same
descriptor. Independent preupdate admission and first-resumed-cycle admission
compare native loss, gradients, Adam, schedulers, clocks and RNG.

Effects are checked separately: Generator must actually request checkpointing
and recompute; Fake must actually execute grad-enabled epsilon forwards with
zero checkpoint/recompute calls and retained activations. Unused, malformed or
contradictory counters fail closed. No-grad Teacher and classifier-only calls
are not counted as Fake epsilon retention. Original flags, callbacks and hooks
are restored on success and exceptions. No full-loop replay is introduced.

This profile is qualified on the pinned Torch2.10/Diffusers0.37.1 CPU1/FP32
closure with actual tiny SD1.5/SDXL classes and fresh-process continuation.
GPU owner overrides, AMP, GAN production geometry, native role replicas and
FSDP2 are **not** qualified by these CPU results. The default `coupled-v1`
checkpoint ABI and its old boolean selection remain unchanged.

## Persistent native DMD2 role DDP

`model.parameters.native_role_backend: "ddp-fp32-v1"` is a separate explicit
reference backend. It installs actual persistent DDP wrappers on the native
Generator and complete guidance role, once per process. The native owner still
performs backward/clip/optimizer updates; the generic runtime does not reduce
those gradients a second time. Native trainable/frozen ownership remains
checked, and the original rank0 DDP bootstrap synchronizes role weights and
buffers before training. Only the native random unused DummyNetwork may differ
before that bootstrap; pretrained frozen tensors must already agree.

Start from [the CPU/Gloo template](../../recipes/dmd2_sd15/training.ddp-fp32.json),
replace all explicit local paths and supply enough independent prepared input
records for every rank. For example:

```sh
torchrun --standalone --nproc-per-node=2 -m distillschedule.cli train training.ddp-fp32.json
```

The template uses CPU/Gloo, FP32, original iteration order and rank-strided
sampling; actual global batch is per-rank native batch multiplied by world
size, **not** duplicated samples. The model and trainer devices must agree.
The source also admits explicit owned CUDA/NCCL placement (`trainer.device:
"cuda"` with `trainer.distributed.backend: "nccl"`); that new public path has
not received its own GPU qualification. It is not the measured native BF16,
FSDP2 or fused-Adam speed configuration. Do not infer production capacity or a
speedup from CPU tests.

The common lifecycle saves one complete canonical role/optimizer checkpoint
per rank plus native/outer RNG, input progress and both clocks. DDP's `.module`
prefix is not part of the checkpoint ABI. Resume uses the same topology and
source, reinstalls actual wrappers before loading state, and rebuilds temporary
buckets. Native Adam updates remain once per actual update; both schedulers
retain the original Accelerate world-size multiplier. Wrapper forward counters
cover the current process, whereas optimizer counters cover persistent native
training clocks; receipts identify that distinction.

Reference mode or the exact direct manual native-owner backend are supported.
Auto/plan clone trials, arbitrary owner overrides, per-role checkpoint
overrides, GAN ownership, mixed precision, FSDP2 and topology-changing restart
are rejected, not silently downgraded. The visible-module FSDP/replica
capabilities do not automatically apply to an opaque optimizer-containing owner.

## Official Self-Forcing: release consumed rollout KV

The [explicit template](../../recipes/self_forcing_official/training.consumed-kv.json)
adds only `model.parameters.native_cache_lifetime: "release-after-commit-v1"`.
The unchanged default is `native-v1`. This is a bound model profile, not a
searched owner-execution option or an automatic CUDA optimization.

The owner drops its two self-attention KV caches and cross-attention cache
only **after** the corresponding native forward/backward/clip and Adam/EMA
commit. It does not shorten a causal sequence or release saved tensors during
checkpoint replay. The next original native rollout allocates fresh caches.
The release helper does not collect tensor values on CPU, clear the allocator,
or synchronize CUDA. Cache lifetime is part of checkpoint/source identity;
resume rebuilds temporary caches rather than serializing them as training state.

Per-cycle receipts require one actual Generator release and the configured
number of Critic releases (five by default). The reported byte count sums
deduplicated **owner storage references relinquished per commit**, not memory
freed by the allocator or a measured HBM reduction; an observer can retain
aliases. Metrics use the common float-only collector contract.

Actual tiny native CPU tests compare three cycles with checkpointing OFF and
ON, including losses, gradients, Adam, EMA, native clocks, input progress and
RNG. A separate public new-process resume matches uninterrupted full state.
These tests qualify the CPU integration, not pretrained quality, performance
or this new public policy's CUDA capacity. The separately measured native GPU
implementation and the complete native speed portfolio remain distinct.

## Evidence boundary

- Actual native SD1.5 and SDXL CPU2/Gloo training completed three cycles and exact
  fresh-process resume vs uninterrupted state comparison, including Adam,
  schedulers, RNG and input progress. This is implementation evidence, not
  pretrained quality or distributed GPU evidence.
- The component policy completed actual SD1.5/SDXL native three-cycle and
  restart checks on Torch2.10 without CUDA initialization.
- Full measured native optimization migration remains incomplete: production
  mixed precision/GAN, fused Adam, FSDP2 and the remaining native Self-Forcing portfolio
  still need their own continuous-lifecycle integration and qualification.

These opt-ins are capability/source-bound implementations within the existing
framework, not extra public trainers or historical run-id launchers.
