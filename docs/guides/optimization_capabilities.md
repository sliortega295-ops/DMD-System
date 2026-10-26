# Physical capabilities and executable plans

The training algorithm owns loss, RNG draws, native update clocks and samples.
A physical optimization may only change admitted implementation details. Model,
workflow and strategy names are not interchangeable dimensions.

```sh
distillctl models list
distillctl algorithms list
distillctl strategies list
distillctl capabilities list
```

These commands read descriptions, not models, and never initialize CUDA. A
catalog entry is **not** a claim that every model supports that optimization.
Actual candidates are constructed from the live Program, owned modules,
precision, dependencies, topology and measured effects during calibration.
Unsupported or nonexercised candidates are refused, not silently substituted.

## Reuse the existing runtime

| Mechanism | Real execution path | Applicability and evidence boundary |
| --- | --- | --- |
| Frozen parameter sharing | Existing `sharing_inventory` and Session storage aliasing | Explicit immutable module roots with identical finite weights, layout and dtype; no mutable/trainable storage alias |
| Activation checkpointing | Native nonreentrant pure `Call` replay | A `replay_safe` gradient forward with declared reads/writes; never replay a native optimizer/update macrocall |
| Selective result retention | Existing native matmul/attention checkpoint contexts | Only exercised native operations and version-valid tensors; real save/recompute counters, not an estimated flag |
| Resident/replicated/FSDP2/HSDP2 | Existing resident and distributed Program lifecycle implementations | Visible supported modules, admitted optimizer/gradient semantics and actual groups; opaque owners do not acquire FSDP support by advertisement |
| Role pools and legal reordering | Existing dependency/version graph, role transport and native lowerings | Explicit executable pool/transport/capacity and sample/update contract; no model-name winner fallback |
| Model-native KV/cast/replay and fused kernels | Existing original-recipe implementations | Native per-recipe parameter/input/precision/version lifetime checks; no blanket transfer from a DiT to an SD UNet |

The W1–W6 native paths keep their existing source/backend and hardware gates.
Their prior GPU records are not new GPU validation of the SD, official
Self-Forcing or LTX integrations. The new providers always have an independent
CPU reference path; a specialized GPU mechanism is never required for that
reference to run. There is no universal `all_optimizations=true` switch.

## Four uses of one executable policy

* `reference`: original admitted implementation, no calibration required.
* `manual`: explicit supported implementation and optional legal call order.
* `auto`: bounded real calibration → existing shared solver → saved selected
  plan → actual runtime, with a paired reference gate.
* `plan`: consume that source/input/configuration-bound plan; no substitution
  of an old hand-tuned launcher or silent unsupported-backend fallback.

The common lifecycle retains the selected policy across complete cycles while
parameters, input positions and logical versions actually advance. An altered
source, effective model contract, owner capability descriptor, topology or
cycle structure invalidates the saved policy. Temporary caches are rebuilt on
restart. A checkpoint restores progress; an inference export deliberately does
not carry optimizer/history/RNG continuation state.

For a new backend or capability, implement its applicability predicate, real
execution and measurable effects through these existing interfaces. Then test
output, gradient, optimizer, RNG, version and invalidation behavior against its
reference before supplying costs. A predicted cost or configuration field by
itself is not an implementation.

## Model adaptation, not optimization-script copying

A new flow model can reuse `training.dmd:build_algorithm` by providing the six
actual role methods: `validate_feeds(feeds)`, `sample_template(feeds)`,
`rollout(feeds, sigmas, gradient_policy)`,
`denoise(sample, sigma, feeds, conditional=...)`, `loss_masks(feeds, sample)`,
and `apply_conditioning(sample, feeds)`. Payloads are named tensor dictionaries;
there is no fixed video rank or model name in the generic algorithm. Return a
`ModelComponents` with explicit flow prediction and `dmd_flow_v1` capability,
visible Student/Fake/Teacher state, optimizers and declared clocks. Both LTX AV
and an independent rank-two model exercise that same implementation.

Official DMD2 and official Self-Forcing retain their different native losses,
rollouts and update clocks in their own actual upstream owners. They are not
forced into the generic flow policy. See [model families](model_families.md)
and [extension contracts](extensions.md).

## Scoped native owner example: DMD2 UNet checkpointing

The actual SD1.5/SDXL owner offers a finite source-bound
`owner_execution.native_cycle.gradient_checkpointing` option on the qualified
Torch2.10/Diffusers0.37.1/Transformers4.51.3/PEFT0.17.1 CPU FP32 stack:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {"native_cycle": {"gradient_checkpointing": true}}
  }
}
```

This changes supported original UNet checkpoint flags/callbacks within a
complete native macrocycle and restores them afterward, including exceptions.
It never replays the optimizer-containing `external_grad` Call, changes the
bound mathematical settings or consumes training RNG during discovery. Real
separate Generator/Fake request/recompute counters must be observed; an enabled
but unexercised option is pruned/rejected. Unqualified physical stacks expose
baseline only and do not lose an otherwise admitted reference path.

The same option is actually calibrated/selected under a measured tracked-CPU
storage limit and executed through manual, auto and saved-plan modes. This is
functional planning evidence, **not a GPU-memory or speedup measurement**.
Explicit optimized manual execution and the first continued update after
process restore receive a paired same-preupdate-state reference gate on
independent clones. Only the admitted selected session becomes live state.
Its native SD15/SDXL loss/gradient/Adam/scheduler/RNG and fresh-process
continuation tests execute actual upstream classes; they do not qualify FSDP,
AMP or arbitrary opaque owners.

## Official Self-Forcing: Fake-only checkpoint policy

The explicit native `settings.gradient_checkpointing: true` reference has a
separate source identity from the unchanged default OFF reference. The causal
Generator's ON/OFF branches differ in cross-attention-cache/gradient behavior;
**Generator checkpoint is not a searched physical toggle**. On the exact
qualified tiny CPU1 FP32 profile, the same `owner_execution.v1` protocol exposes
only `owner_execution.native_cycle.fake_gradient_checkpointing` for the actual
noncausal Fake model. Source/runtime/profile predicates, not a model-name branch
in search, admit the alternative. Other profiles retain baseline only.

Original native checkpoint and recompute entries are observed with read-only
block hooks and source/Torch code-object checks. OFF is verified by zero entries
plus real direct grad-enabled execution, not the requested boolean alone. The
atomic G/C macrocall is never replayed; Generator policy, caches, modes, native
RNG/data/update order and Adam/EMA remain source-owned. Three tiny CPU cycles
match the independent original Trainer oracle, and public manual new-process
restart and measured auto/saved-plan use the same capability. Unqualified CUDA
trial copying still refuses; pretrained/GPU/performance remain NOT_RUN. See the
[checkpoint recipe](../../recipes/self_forcing_official/README.md#explicit-checkpoint-on-reference-and-fake-only-physical-policy)
for configuration, instrumentation and evidence boundaries.

## Modern native flow planning is a separately qualified CPU slice

The FLUX.2 klein Base, original Qwen-Image, Wan2.2 T2V and HunyuanVideo-1.5 tiny actual-class
fixtures use the same common reference/manual/auto/saved-plan runner. Their
real measured candidates and selected sessions preserve losses, gradients,
updates, optimizer/scheduler/EMA, input/RNG clocks and fresh-process restore.
No model-name winner, historical cost, kernel timing or speedup is substituted.

LTX-2.5's actual tiny AV fixture now exercises that same measured planner in
its **separate pinned environment**, including LoRA-base immutable declarations,
three complete G1/F5 updates and public saved-plan/new-process recovery. The
original `RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED` label is retained;
only that declared fixture is qualified for whole-role CPU trial copies. Actual
frozen tensors must match exactly before sharing; mutable A/B and Adam state
remain separate. The reference mathematical program and common searcher do not
branch on the model name. Full-parameter reference training remains independent
of LoRA-specific sharing.

Clone-based trials admit only the declared **SYNTHETIC_TEST_ONLY CPU1 FP32**
layout (or the explicit LTX random-native fixture above) and verified owned
alias/module state. A local asset manifest labeled
`LOCAL_PRETRAINED_UNVERIFIED` is not proof of model size, source authenticity or
snapshot safety: auto/plan trial cloning refuses it **before copying or
executing**. Its independently admitted non-clone reference/manual lifecycle
remains available; changing a trust label is not a capacity qualification.
`memory_limit_bytes` constrains selected **tracked tensor storage**, not host
RSS, temporary paired-clone construction, CPU staging or large-model OOM risk.

Saved plans bind source/runtime, complete effective architecture and graph,
parameter values, algorithm, input geometry/dtype, hardware and cache state;
drift refuses before a formal update. Transient caches are rebuilt on restore,
never treated as portable immutable model state. Opaque native CUDA owners
still refuse clone-based planning until capacity-bounded backend-native
snapshot/restore and paired comparison are qualified. No whole-owner deepcopy
is an admitted CUDA fallback. See [bounded release notes](release_notes.md).

## Direct single-CUDA immutable sharing

The five modern flow adapters (LTX-2.5, FLUX.2 klein Base, original Qwen-Image,
Wan2.2 T2V and HunyuanVideo-1.5 T2V) also admit this **exact** explicit policy
with an already-supported single-CUDA FP32 configuration:

```json
{
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": true,
    "checkpoint_bindings": []
  }
}
```

This is the existing persistent Session's real immutable-parameter aliasing,
not a second trainer or a CUDA trial copy. Equal frozen LoRA base parameters
may share storage; trainable A/B, gradients and Adam ownership remain separate.
The runtime verifies the actual immutable inventory, precision and values,
then checks that immutable state has not changed. A mismatched asset does not
silently become shared. Normal loss, input/RNG order and update clocks remain
unchanged. Complete-cycle restart reconstructs the same aliases before loading
training state; export still owns the current Student, not a merged weight.

**Capacity boundary:** all original roles are constructed before aliasing.
Reduced tracked live storage is not a reduced construction peak, allocator
capacity guarantee or a speedup claim. Admit the full original model construction
on the target device first. This policy does not enable AMP, distributed roles,
checkpoint replay, reordering or CUDA auto/saved-plan calibration. Such requests
still fail rather than dropping fields or falling back to a fixed recipe.

Manual training retains `initial_reference_gate.status=NOT_RUN`: it does not
manufacture a per-user paired numerical admission. The separately recorded
native tiny-model reference/manual and restart tests qualify only their tested
scope; use the [hardware acceptance record](../status/FRAMEWORK_A100_ACCEPTANCE_20261024.md)
for actual evidence, not a catalog flag or a skipped optional CUDA test.

### Explicit image replication with immutable sharing

The [FLUX/Qwen replica recipe](../../recipes/flow_image/README.md#explicit-replicated-training-not-model-sharding)
uses the existing post-backward replica backend with distinct local-B1 samples
and rank-specific noise. It averages gradients before clipping/Adam and keeps
every role resident on each rank. Exact manual immutable sharing composes with
this path; real alias/reduction counters and all-rank checkpoints are required.
Actual tiny-class Gloo2 and A100 NCCL2 oracle/restart/export evidence is recorded
separately. World16 is not yet accepted. This is not model sharding or support
for DDP-forward hooks around private denoise methods. CUDA planning, AMP and
full pretrained capacity remain unsupported or unqualified as documented.

## Atomic native-owner policies

The official DMD2 [owner profiles](native_owner_backends.md) are explicit
exceptions to ordinary pure-Call checkpointing and visible-module replicas.
`component-v1` independently binds Generator/Fake checkpoint flags to actual
replay/retention counters on CPU1. `ddp-fp32-v1` retains the native internal
backward/clip/Adam loop under persistent actual role DDP, with same-topology
per-rank restart. The latter is reference/direct-manual only; no second generic
reducer, opaque-owner FSDP support or clone-based search is implied. CPU tests
and separately archived native GPU timings are different evidence scopes.

Official Self-Forcing also exposes the explicit bound model profile
`native_cache_lifetime: "release-after-commit-v1"`. It releases only consumed
rollout cache ownership after native backward/clip and Adam/EMA commit; the
next original rollout rebuilds caches. This is not an automatic search
dimension or a checkpoint-ON/OFF Generator rewrite. Per-cycle actual release
counts and deduplicated owner-reference bytes are observable, with CPU native
state/restart qualification and separate GPU/performance NOT_RUN status.
