# Original LiveWan reference training v1

Schema: `distillschedule.livewan_reference_training.v1`.
Asset schema: `distillschedule.livewan_reference_assets.v1`.
Input: [`original_livewan_teacher_cohorts.v1`](original_livewan_teacher_cohorts_v1.md).
Checkpoint policy: `distillschedule.livewan_reference_checkpoint.v1`.

This independently versioned, opt-in reference route does not change any
canonical C51 candidate, historical trace, source digest or two-cycle worker.

## Training ownership

The original `ModelBundle` constructor owns model loading, original causal
attachment, original FP32 Generator/world16 FSDP2, replicated FP32 Critic
LoRA32 over a frozen BF16 base and frozen BF16 Teacher/world16 FSDP2. It owns
both actual foreach AdamW optimizers. The unchanged original baseline helper
owns four-step rollout, continuous sigma, DMD/denoising losses, clipping,
synchronization and exactly G1/C5/EMA0 per complete cycle.

Fixed input geometry is 368x640, 81 frames, B1/rank and logical global B16.
Each cycle consumes five successful real input loads for every logical rank.
Operation-specific input seeds are distinct from the original upstream
sequential RNG continuation. Finite prepared inputs do not silently recycle.

## Admission

- Require 2..64 complete finite cycles and an explicit matching seed.
- Exact clean LiveWan/Wan2.1 source identities, actual import origins, used
  model config/index/weight inventories and optional complete Stage1 state.
- No historical systems-only bootstrap or implicit winner. Fresh state starts
  at G0/C0/EMA0 and both zero-step Adam owners.
- Explicit dependency, backend/precision/environment and physical host/GPU
  allocation/reserve policy. No automatic environment changes or installation.
- Original safetensors inventories match actual original meta-model key/shape
  layouts before physical model allocation; Stage1 is restricted-admitted.
- Fresh owned world16 two-node NCCL and Gloo groups, with rank-local metadata
  failures exchanged before entering subsequent model collectives.

## Complete-cycle state

The common atomic all-rank checkpoint stores actual trainables, named Adam
moments and steps, commit versions, native cursor and per-rank RNG. It binds
source/assets/config/precision/optimizer/topology/input identity and the fixed
target horizon. A complete cycle k requires actual Gk/C5k/EMA0, cursor 5k and
no pending gradients on original model or optimizer owners.

All required rank files and completion marker must exist. Hash/layout/name/
dtype/mode/cursor/version/Adam-step mismatches reject before mutation. Loading
preserves Parameter/optimizer owners and hooks. Frozen Teacher/base state and
temporary streamer caches are reconstructed, not mistaken for trainable state.
Only an unfinished same-target, same-topology boundary may resume; extending
the target or converting a C51 physical role-pool checkpoint is unsupported.

## Verification and nonclaims

Actual CPU input/LoRA/Adam/DTensor state and Gloo failure tests, original
meta-constructors and explicitly labeled dispatch fixtures qualify source and
owner contracts only. CUDA/pretrained FSDP2/NCCL execution, physical capacity,
numerical equivalence, generated quality and performance remain NOT_RUN.
This contract supplies neither C51 transport drain/shadow reconstruction nor
native generation/export, and cannot relax their existing rejection gates.
