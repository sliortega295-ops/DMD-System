# Physical capabilities and executable plans

The algorithm owns losses, RNG draws, samples and native update clocks. Physical
execution changes only admitted implementation details; a model name is not an
optimization policy. Inspect the available interfaces without loading models:

```sh
distillctl models list
distillctl algorithms list
distillctl strategies list
distillctl capabilities list
```

## Choose an execution path

| Path | Supported mechanisms and conditions | Start here |
| --- | --- | --- |
| Resident visible Program | Immutable parameter sharing; nonreentrant checkpointing of pure `replay_safe` gradient Calls; selective retention of exercised native matmul/attention results with valid tensor versions | [Training](training.md), [flow policies below](#direct-single-cuda-immutable-sharing) |
| Visible distributed Program | Replicas, FSDP2/HSDP2 and per-root mixed layouts with explicit module/optimizer ownership, replica semantics, dtype and initialized groups | [Extension contracts](extensions.md) |
| Official SD1.5 DMD2 | Native FSDP2 reference or the complete optimized Resident/DDP profile; A100 world16, B1/global16, 512×512, G1/C10/EMA0 and native mixed-precision exceptions | [FSDP2 reference](dmd2_fsdp2_reference.md), [optimized training](official_optimized_training.md) |
| Official Self-Forcing Wan | Native FSDP2 reference or complete R6 profile; A100 world16, B1/global16, 368×640/81 frames, G1/C5/EMA0, BF16 compute/FP32 masters; distinct from W4 LiveWan | [Official workflow](../../recipes/self_forcing_official/README.md), [optimized training](official_optimized_training.md) |
| Atomic native-owner alternatives | Source-bound SD checkpoint policies, FP32 native role DDP and official Self-Forcing post-commit KV release; each has its own device/precision scope | [Owner policies](native_owner_backends.md), [budgeted resident planning](resident_owner_training.md) |
| Role pools and legal reordering | Existing dependency/version graph, executable transport/pool topology and capacity; preserve sample and update contracts | [Framework ownership](../architecture/framework.md), [native workloads](native_workloads.md) |
| W1–W6 native recipes | Recipe-specific cache/cast/replay/fused kernels and layouts, including W6 ZeRO-3; retain their source, asset, shape, precision and hardware predicates | [Native commands](native_workloads.md), [support matrix](support_matrix.md) |

Candidates come from live owned state, dependencies, topology and observed
effects—not a catalog flag or model-name winner. Unexercised checkpoint/retention
options are rejected. An opaque DMD2/Self-Forcing macrocall owns backward,
clipping and optimizer updates: never replay it or add a second generic reducer.
Its native FSDP2/DDP implementation is separate from visible-module sharding.

## Four uses of one executable policy

Where the selected backend supports them:

* `reference`: use the original admitted implementation without calibration.
* `manual`: select an explicit supported implementation and legal call order.
* `auto`: measure bounded candidates, compare with the paired reference, then
  execute the shared solver's selected plan through the same runtime.
* `plan`: execute that saved, source/input/configuration-bound policy.

The persistent lifecycle advances real parameters, data positions and versions
across complete cycles. Plans bind source/runtime, effective architecture and
weights, algorithm, input geometry/dtype, owner capabilities, topology and cache
state; drift is an error. Restore rebuilds transient caches. A training checkpoint
restores optimizer/history/RNG progress; an inference export carries the current
Student instead. See [training and recovery](training.md).

## Direct single-CUDA immutable sharing

LTX-2.5, FLUX.2 klein Base, original Qwen-Image, Wan2.2 T2V and HunyuanVideo-1.5
T2V admit this exact single-device manual `execution` policy with FP32 owned state:

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

The persistent Session aliases equal, finite, contiguous declared frozen
parameters with matching dtype/layout for the live module lifetime. Trainable
LoRA A/B, gradients and Adam state stay separate; immutable contents remain
checked. Complete-cycle restore reconstructs the aliases before loading state.
This changes neither role placement, checkpoint replay nor call order, and
export keeps the unmerged current Student. Direct sharing does not run paired
calibration.

**All original roles are constructed before sharing.** The full construction
must fit the target device; lower tracked live storage does not lower that peak.
LTX uses its separate pinned runtime. Its optional
[all-component parameter budgets](../../recipes/ltx25/README.md#optional-all-component-cpu-parameter-budgets)
admit larger declared architectures through native meta sizing before loading
roles/codecs. Budgets remain bound through recovery and current-Student export;
device memory and execution placement are configured separately.
All five adapters accept
`model.parameters.compute_precision: "fp32"` or `"bf16-mixed"` on supported
CPU/CUDA devices: native Transformer autocast keeps FP32 master/Adam/EMA state
and flow/sampling math, without changing model-native norm/rotary precision.

### Local-B1 image, video and AV replicas

LTX, FLUX/Qwen, Wan2.2 and HunyuanVideo use the same explicit replica contract:
CPU/Gloo or CUDA/NCCL, rank-strided distinct inputs, local B1 and global batch
equal to world size. Every rank holds complete roles; gradients are averaged
after backward and before clipping/Adam. This is not model sharding or DDP
forward hooks around private role methods. Both FP32 and explicit BF16 compute
compose with reference/direct-manual replicas; exact manual immutable sharing
is applied inside each replica. Recovery requires the same topology and all
rank shards. See the [replica configuration and launch sequence](../../recipes/flow_image/README.md#explicit-replicated-training-not-model-sharding),
[Wan2.2](../../recipes/wan22_dmd/README.md) and
[HunyuanVideo](../../recipes/hunyuan_video_dmd/README.md) and
[LTX AV](../../recipes/ltx25/README.md#explicit-local-b1-replicas) recipes.

LTX separately admits [native CPU block FSDP](../../recipes/ltx25/README.md#opt-in-native-cpu-block-fsdp):
an explicit FP32/Gloo manual recipe, with real Student/Fake/Teacher child/root
groups and owned optional EMA shards. It uses dense CPU initialization,
resident codecs and the common training, recovery and export lifecycle.

## Measured planning and clone safety

The four image/video flow adapters admit whole-role calibration copies only for
declared `SYNTHETIC_TEST_ONLY` CPU1 fixtures with FP32 owned tensors; the chosen
FP32/BF16 compute policy is retained. LTX uses its explicit
`RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED` CPU fixture in the isolated
LTX environment, retaining the configured FP32/BF16 compute policy.
Alias/module-state checks still apply. A
`LOCAL_PRETRAINED_UNVERIFIED` manifest does not establish size, authenticity or
copy safety and is refused before trial copying; independently admitted
non-clone reference/manual execution remains available.

`memory_limit_bytes` bounds selected tracked tensor storage, not process RSS,
simultaneous trial clones, staging or construction peaks. Flow CUDA execution
uses reference or exact direct sharing, not whole-role CUDA copies. Native
SD/Self-Forcing owners likewise cannot use whole-owner CUDA deepcopy: the
separate [budgeted resident protocol](resident_owner_training.md) admits serial
snapshot/restore only for its source-qualified tiny FP32 single-device profiles,
not pretrained-size or distributed owners.

## Model adaptation, not optimization-script copying

Reuse `training.dmd:build_algorithm` for flow DMD by providing six role methods:
`validate_feeds(feeds)`, `sample_template(feeds)`,
`rollout(feeds, sigmas, gradient_policy)`,
`denoise(sample, sigma, feeds, conditional=...)`, `loss_masks(feeds, sample)`
and `apply_conditioning(sample, feeds)`. Named tensor payloads impose no fixed
video rank or model name. `ModelComponents` declares flow prediction,
`dmd_flow_v1`, visible Student/Fake/Teacher state, optimizers and clocks.
Official DMD2 and official Self-Forcing keep their distinct upstream losses,
rollouts and native clocks rather than adopting this flow policy.

New capabilities need an applicability predicate, real execution and observable
effects, plus output/gradient/optimizer/RNG/version checks against the reference.
See [extensions](extensions.md), [model environments](model_families.md),
[performance and validation](performance_validation.md) and
[source and asset licenses](../../THIRD_PARTY.md).
