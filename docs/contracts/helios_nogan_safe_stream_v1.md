# W2 native SafeBundle fresh stream v1 (explicit research opt-in)

Preset: `helios_nogan.safe_stream_ed8.v1`. Policy:
`distillschedule.helios_safe_stream_policy.v1`. This is a new direct strategy,
not a revision of `helios_nogan.safe_ed8.v1`, a saved benchmark plan, or the
original reference policy. It is **source/CPU connected, GPU NOT_RUN, numerical
REVIEW**, never stable/default or a performance claim.

## Immutable workload and physical policy

- Original No-GAN 33-frame, 384x640, 222-stage, physical B1/global16 helpers.
  G1/C5/EMA5 per complete cohort; the source-qualified Step10 bootstrap remains
  G2/C10/EMA10, with original named AdamW, LR schedulers and EMA recurrence.
- Exactly 2..64 distinct, prepared `original_helios_cohort_stream.v1` cohorts.
  Each supplies five actual rank/step-qualified feature sets. Source steps
  advance from 10..14, 15..19, etc.; a new seed or relabelled old trace is not
  fresh training. There is no arbitrary epoch, cold bootstrap or GAN support.
- Exact typed `HeliosImplementation(no-gan,node-hsdp,512,5,False,SafeBundle.safe())`.
  Same node-HSDP row-major 2x8 mesh, `(Replicate(), Shard(0))` FP32 trainables,
  BF16 compute/FP32 reductions, no parameter offload, persistent unshard OFF.
- Full all-or-none SafeBundle: five exact-modulated-norm switches, rollout
  fence elision, LoRA-B GEMM and captured LoRA checkpoint replay scope. Original
  all-role scale1 add, dispatch warmup, no-grad/exact-grad residual fusions
  remain ON; every other optional flag is explicitly OFF. Selective attention
  replay is **stride0**, never the systematically drifting old stride2 path.
  Merged LoRA, KV/static-context caches and GAN controls are not admitted.

`training.helios_safe_stream` binds these physical controls directly, without
calling `HeliosImplementation.environment()` or inheriting historical model,
trace, negative-prompt paths or hashes. Unknown/mismatched inherited framework
environment controls reject. The original reference remains world16 FSDP,
SDPA, window1, all optimizations OFF; old two-cycle preset/plan gates are intact.

## Assets, dependencies and execution

Only an explicit local `distillschedule.helios_reference_assets.v1` manifest
is admitted. Original loaders verify model/precast/bootstrap/negative identities
and payloads. The first freshly prepared cohort, not a historical trace,
supplies startup controls. Source, shape, all-rank inputs and effective policy
are admitted before constructing models.

The operator must explicitly supply an existing
`LOCAL_KERNELS=kernels-community/flash-attn2=/absolute/local/path`. Missing
extension is a real GPU dependency **NOT_RUN**; there is no installation,
download, private fallback path or CPU fake acceptance. Original P0 still
checks the exact extension SHA256 and the actually imported FA2 callable.
Allocator overrides are refused. Actual setup, transient rollout, checkpoint
staging and restore capacity/reserve remain unqualified; no generic multi-model
trial clone or CPU parity gate is used as a substitute.

The public wrapper enters the existing world16 NCCL/Gloo owner and hardened P0
construction once. The original `_Observer` and `_prepare_cycle` receive the
full typed implementation for every fresh cohort. Actual installed-kernel
counter snapshots and positive deltas are required **per cycle**, along with
native operation/update/version census and release of all retained rollouts.

## Complete-boundary checkpoint and restore

Checkpoint identity includes current source HEAD/fingerprint, exact finite
stream/horizon, verified immutable assets, effective SafeBundle policy and
native owner global trainable-layout digest plus 2x8 topology. Native local
shard records preserve names, shape/dtype/placements/mesh; common checkpoint
publishes only after every rank shard is present and hash verified.

The actual G/C trainables, buffers/modes, named Adam states, schedulers, EMA,
native clocks, per-rank RNG and exact next-cohort cursor are restored into the
same owners. A resumed process rebuilds original models/kernels/caches once,
validates its actual shard layout against saved identity, and restores state
and RNG before preparing the next cohort. Replay/rollout/capture caches are
not serialized. Pending gradients, retained rollouts, wrong source/topology,
incomplete G1/C5/EMA5 boundaries and missing/corrupt rank shards reject.

## Evidence boundary

Tests exercise public producer/config/worker/typed-policy routing and original
rollout/loss/Adam/LR/EMA helpers using scalar CPU DTensor owners. Four fresh
cohorts versus save-at-two/new-owner resume-to-four compare actual state and
RNG. Separate two-rank Gloo tests exercise local-shard state publication and
fault refusal. CUDA telemetry and kernel effect counters in tiny tests are
**synthetic**, not native kernel installation/numerical/capacity acceptance.

Pretrained world16 GPU execution, strict optimized/reference numerical parity,
long-run reliability/quality, capacity, decoded generation/export and speedup
remain NOT_RUN/REVIEW. This contract does not complete U3/U7 for all six
workloads, alter the frozen benchmark, or qualify a stable long lifecycle.
