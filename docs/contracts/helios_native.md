# Helios native input, execution and checkpoint contracts

These versioned opt-ins reuse the original No-GAN/GAN owners, not a new
algorithm or an extension of the immutable two-cycle benchmark. Native math,
source/assets, rank ownership, update clocks and complete all-rank checkpoint
rules remain binding. [Native recipes](../guides/native_workloads.md) select the actual policy.

<a id="helios_gan_native_history_v1"></a>
## W3 GAN native-history v1 — explicit research opt-in

This ABI is independent of W2 native history. It retains the **same original
GAN training owner**, full FA3/down40/no-merge bundle, world16 FSDP2, C2048,
physical B1, five detached rollouts, joint GAN gradients and G1/C5/EMA1 clocks.
It is not a benchmark-request extension, stable recipe, No-GAN fallback,
numerical-equivalence result, CUDA-capacity result or performance result.

### Explicit input and execution binding

- Public preparation: `distillctl data prepare-helios-gan-stream` with
  `--history-policy source_native_amplified_v1`.
- Training `data.input_kind`:
  `original_helios_gan_native_history_cohort_stream.v1`.
- Execution remains explicit manual preset
  `helios_gan.fa3_down40_nomerge_stream.v1`, with the original eight local asset
  roles and pinned FA3 implementation. N70/merged execution remains OFF.
- The normal `original_helios_gan_cohort_stream.v1` default is unchanged. Its
  producer still rejects zero history. Its default loader rejects this new
  stream, and the native loader rejects normal/W2/old-request relabeling.

Use `helios_gan_inputs.portable_config()` to remove only the five unused legacy
system locators. Write/hash one portable configuration and bind that exact file
in both preparation and assets. Native math/configuration validation remains
unchanged. A finite record pool additionally requires the existing explicit
`distillschedule.helios_record_epochs.v1` input policy; it does not create a new
input sampler or change its seed/order/draw identities.

GAN capture remains separate from W2 capture. Original dataset/history,
corruption, scheduler, rollout noise, Score timestep, GAN crop and R1 epsilon
owners are unchanged. The actual original history predicate decides whether
each stage has its original budget2 or doubled budget4. Normal initial exits
remain common to all ranks. Amplified exits use the actual rank-local native
draws, with no seed search, resampling, padding, dummy calls, forced normal exit
or shortened rollout. The trace records and revalidates the exact grids.

The independent execution schema is
`distillschedule.helios_gan_native_history_execution.v1`, and semantics are
`original_helios_gan_rank_local_amplified.v1`. W2 success is not GAN admission.

### Physical owners and safe boundaries

1. G: validate the actual root plus40 complete `HeliosTransformerBlock` FSDP
   owners, one active original checkpoint function, and original sharded
   reshard-after-forward/backward entry state. Reuse the existing
   `fsdp2_variable_forward_phase`, materializing all41 units before the original
   `_run_gan_generator_update`. Retain them through its original GAN/Score
   latent-gradient boundary and G backward. Release and restore before the
   **original** engine clipping/Adam/scheduler/zero-grad step. Exceptions release
   the units without fabricating a commit.
2. Detached `r_all`: do not enter the G phase. Keep the original
   `generator_nograd_persistent_unshard_scope` exactly. Every rank performs at
   least one complete forward, which materializes the same41 units; all repeat
   forwards retain those units. Preserve five actual B1 rollouts, original
   allocator flushes and the original release before Critic updates. The native
   census requires all41 actual full owners before exit,41 releases, restored
   policy and zero retained unsharded units. Nested/materialized entry is still
   rejected.
3. C/GAN head, GAN/R1/crops, frozen-down40 and FA3 replay, joint-gradient/adapter
   boundaries, update order and EMA use the unchanged original owners. Dynamic
   G prefixes do not change fixed C replay work:400 original/replay frozen-down
   operations per cycle remain required. No counter is warmed or synthesized.

The G phase materializes the entire Generator while its activations and Score
state coexist. CPU sharding/component tests do **not** establish that this fits
the assigned GPUs. Fresh full-size CUDA/host/cgroup/HBM reserve admission is
required; source support never authorizes lowering memory or numeric gates.

### Continuation and source identity

Native G/C/GAN-head/Adam/scheduler/EMA/RNG state still uses the same original
`engine_state`, `save_checkpoint`, `restore_checkpoint` and atomic all-rank
publication owners at drained complete-cycle boundaries. The independent
checkpoint policy pins the GAN capability, execution source, source HEAD,
full transitive source fingerprint, immutable assets, exact stream SHA, actual
native owner layout and existing epoch/draw identity. Any mismatch rejects.

Bootstrap remains physical G2/C10/EMA800 and curriculum G200/C1000/EMA800;
after two new cycles it is G4/C20/EMA802, curriculum G202/C1010/EMA802. This is
not a recovered post50 checkpoint or an official checkpoint1000. New source
requires new source-bound W3 preparation from the existing feature pool, not
relabeling W2 traces or changing an old stream's fingerprint.

Capture RNG checks assume the original helper closure is already loaded. A cold
Rich import consumed Python `getrandbits`; warm helpers and the input-stream owner
preserved Python/CPU Torch RNG. Do not conceal that setup boundary by changing seeds.

<a id="helios_gan_native_history_checkpoint_cpu_v1"></a>
## W3 native-history G checkpoint hidden-input CPU policy v1

**Explicit default-OFF research candidate; Pending cluster validation.**
No native fit/GO, numerical promotion, quality, timing or long-trajectory claim.
The original W3 stream/default recipes and W2 policies remain separate.

### Identity and unchanged owners

Preset `helios_gan.native_history_fa3_down40_checkpoint_cpu.v1` requires
`original_helios_gan_native_history_cohort_stream.v1`, an explicit capacity file,
current source/assets/stream identity and the original fresh two-node world16
owner. No plan, reference substitution, node-HSDP factory or replay fallback.
The original G40/root41 and C40/five-head/root46 FP32 Shard(0) owners, FS16,
C2048, B1/globalB16, persistent-unshard/window5, FA3/down40/full bundle,
N70-OFF, joint DMD/GAN/R1 losses, G1/C5/EMA1, Adam/LR/RNG/dtypes are unchanged.
Physical versions C1 G3/C15/EMA801, fresh C2 G4/C20/EMA802 retain curriculum
offsets198/990 and source steps1000/1005. Caches are rebuilt, never serialized.

The adapter authenticates real loaded source/code and exact closures from
`NativeDownStreamSAC.bind`, the captured original G checkpoint partial, original
LoRA replay wrappers and C's original stride2 selective checkpoint selector.
There is no `__wrapped__` traversal, replacement checkpoint or new SAC backend.
Default W3's original callable validator/refusal is not weakened.

### Only changed storage and exact liveness

Only the exact first hidden Tensor object saved by each original G whole-block
non-reentrant checkpoint uses synchronous, same-dtype CPU storage. Parameters,
buffers, aliases and shared encoder/RoPE/timestep inputs are not offloaded.
Hooks cover only the original checkpoint call, never its block recomputation,
whole GAN helper or C call. Mutation/version sentinels, lease bounds,
restored/unused release and error cleanup reuse the W2 session primitives;
the new policy does not reuse W2 topology or its execution identity.

G shapes are `[1,2940,5120]`, `[1,4560,5120]`, `[1,11040,5120]`, BF16, 40 blocks.
All120 copies total7,593,984,000 D2H bytes/action. Actual source-validated
rank-local stage exits determine `j=max({0} union {s in {1,2}: exit_s>0})`.
Only suffix j..2 restores:40/80/120 inputs and4,521,984,000 /6,389,760,000 /
7,593,984,000 H2D bytes. Every copied boundary releases exactly once; dead
prefix graphs release unused, never fabricated replay. Both original history
horizons2/4 and all original calls remain unchanged.

The original cropped GAN-G C call keeps its `[1,16,9,24,40]` latent and full
histories: `[1,4560,5120]` BF16 hidden input,40 blocks,1,867,776,000 logical
checkpoint-input bytes on GPU. The four original patch checkpoints
(`patch_embedding`, `patch_short`, `patch_mid`, `patch_long`) remain on the
original callable, in source order, with no CPU copies. Together with40 block
and five auxiliary head checkpoints this is49 checkpoint dispatches. The20 original
selective contexts (two FA3 results each) remain unchanged. down40 is inactive
here (`core.current is None`), not disabled for later original C updates.
The original cached `autograd.grad` must complete its one input derivative and
empty all target-Torch2.10 SAC FIFOs before original G backward. The observer
adds no loss, detach, parameter gradient, backward, optimizer or EMA operation.

### Bounded capacity, observations not reservations

The capacity JSON must declare exact per-rank7,593,984,000 bytes and per-node
60,751,872,000 bytes (eight disjoint ranks), pageable XOR pinned. No unbounded,
mixed or silently enlarged declaration. Original startup512GiB and running64GiB
host reserves remain; fresh physical/cgroup observations include the additional
node payload before construction and during every copy.

GPU floor remains512MiB. Before GAN-G C forward the known C hidden-input subset
plus floor requires2,404,646,912 free bytes. Each largest G restore separately
requires113,049,600+536,870,912=649,920,512 free bytes. Neither check proves total
HBM fit: FA3 saves, transient all-gathers, gradients, casts, FFN/head work,
allocator fragmentation, driver and EMA storage remain additional demands.
The earlier unmodified W3 conditional logical lower-bound deficit is not an
observed W3 OOM. Removing G hidden-input residency is a necessary candidate,
not sufficient evidence to admit a native launch.

New identity/checkpoint schemas bind source, complete stream, capacity, exact
world16 native-owner layout, full joint G/C/five-head state, Adam/scheduler/EMA,
rank RNG and cursor. Only complete all-rank cycle publication is resumable;
missing/corrupt/stale/W2/old-policy checkpoints must refuse before mutation.
G/C windows cannot cross or share a live CPU lease cache. Complete G action
receipts require41 acquire/release,1 backward/commit, all120 boundary copies,
the exact source-derived suffix census and the original drained GAN-G C census.
Existing full joint-GAN action/effect checks still govern every cycle.

Production requires target Torch2.10 exact SAC FIFO/replay callbacks; the named
CPU Torch2.14 layout substitution is component evidence only. The recipe remains
a placeholder, not an admitted launch packet.

<a id="helios_nogan_native_history_v1"></a>
## W2 native rank-local amplified history, version 1

This is an opt-in **continuation input/execution contract**, not a new
`OriginalHeliosRequest`, canonical benchmark, saved-plan dimension or trainer.
The normal `original_helios_cohort_stream.v1` route is unchanged.

### Identity

- Input kind: `original_helios_native_history_cohort_stream.v1`.
- Stream schema: `distillschedule.original_helios_native_history_cohort_stream.v1`.
- Cohort schema: `distillschedule.helios_native_history_cohort.v1`; explicitly
  carries `native_policy=source_native_amplified_v1`.
- Execution: `distillschedule.helios_native_history_execution.v1`.
- Checkpoint policy: `distillschedule.helios_native_history_checkpoint_policy.v1`.
- Fixed original W2 geometry: 384×640, 33 frames, BF16 input, physical B1,
  global B16, world16, G1/C5/EMA5; original Step10 G2/C10/EMA10 bootstrap.
- Same complete-cycle checkpoint and same-source/assets/stream/topology resume
  rules as the existing original W2 lifecycle. Phase materializations are
  transient, are released before a checkpoint, and are rebuilt on restore.

A stream's kind, policy, source inventory, source-code fingerprint, samples,
seeds, per-rank controls and full task digest are bound together. Old input kinds
reject new stream/task schemas. Original benchmark request enums are not widened.

### Original sampling and work

The existing native dataset, `wan_capture.prepare_history`, isolated input RNG
streams, block noise and `wan_capture.stage_control` remain authoritative.
Preparation supplies **actual** post-corruption history zero predicates and the
actual rank to the latter. Initial normal exits still agree across ranks. A
zero-history rank then keeps its original independent amplified exits and
*doubled scheduler budget*: 4 rather than 2 at each pyramid stage. Neither the
number of calls nor the rank-local exception is synchronized, padded, dropped,
resampled or replaced by a normal branch.

The shared pure all-rank schedule validator is reused, but its old Resident-only
execution label is not: this contract requires
`FSDP2_ALIGNED_MATERIALIZATION_WITH_ORIGINAL_REDUCE_AND_STEP`. Before model
construction, each worker independently checks its real CPU tensor history
predicate, original seeded exits and exact executed scheduler grid. Runtime
receipts check that worker's actual calls, shapes, BF16-projected timesteps,
backward/commit counts and original update versions.

### Actual execution

Only the existing reference backend is admitted: world16 FSDP2, chunk512, SDPA,
B1/window1, original AdamW/scheduler/EMA, old persistent-unshard flag OFF.
Manual SafeBundle, saved plans, unbound policy overrides and GAN are rejected.

At each aligned Generator graph action all ranks materialize the same original
FSDP root and **40 complete `HeliosTransformerBlock` checkpoint units** in the
same order. Interior PEFT Linear sharding or an equal count of other units is
not equivalent. The installed active checkpoint owner must use the original
`fsdp2_metadata_tolerant_checkpoint` or its admitted original partial.

For G update the phase encloses the entire existing helper: native rollout,
Score calls and G backward. Its engine view releases full parameters before
calling the **original** engine step, clipping, AdamW, scheduler and zero-grad.
Detached Critic-input rollout actions get their own aligned no-grad phase;
the old no-grad-only persistence scope is not stretched over a G backward.
Critic updates and EMA keep their original helpers and order.

Each actual action receipt records acquire/release/forward/backward/commit and
restoration counts with the native task SHA. Exceptions release owned full
parameters and cannot publish a successful phase/complete checkpoint.

<a id="helios_nogan_native_history_safe_v1"></a>
## W2 native-history + full SafeBundle continuation v1

Explicit preset: `helios_nogan.native_history_safe_ed8.v1`.
Example: `recipes/helios_nogan/training.native-history-safe.json`.

This version composes the existing original rank-local history input ABI with
node-HSDP 2×8 and the **entire** `safe-nogan-ed8-v1` bundle. It does not change
`safe_stream_ed8.v1`, the reference/native-history policy, a benchmark plan, or
the original sampler/training helpers. No persistent-unshard, merged LoRA, old
stride2 attention replay, extra padding, truncated exits or new random draws.

### Physical and algorithm contract

- Original 384×640×33, BF16 inputs, physical B1/global16, G1/C5/EMA5 per cohort.
  The source-qualified Step10 G2/C10/EMA10 bootstrap and original Adam/LR/EMA
  remain mandatory; this is not a cold bootstrap or quality-training claim.
- Source-qualified `original_helios_native_history_cohort_stream.v1`, including
  normal synchronized initial exits and rank-local amplified exits with the
  original doubled denoising budgets. The prepared finite horizon is immutable.
- Exactly the existing SafeBundle implementation: node-HSDP row-major 2×8,
  C512, rollout window5, p0, stride0, pinned local FA2, original scale1/add,
  norm/residual/LoRA fusion, dispatch warmup and LoRA checkpoint replay scope.
  All original installation checks and **per-cycle positive effect deltas**
  remain required. Merely selecting the preset cannot satisfy that census.

### Collective-safe actions

The existing `fsdp2_variable_forward_phase` owns the same 41 FSDP units: root
and 40 **complete original Helios checkpoint blocks**, never interior Linears.
Before each aligned G action or detached `r0`…`r4`, every rank explicitly
materializes these units in the same order on its original HSDP shard mesh.
Unequal rank-local no-grad prefix lengths therefore do not issue parameter
all-gathers within that variable region. The original checkpoint/backward and
HSDP gradient reduction remain authoritative. G has one original backward and
commit; full parameters are released **before** original clipping/Adam/LR.
Detached actions do not update and release at action exit, including failures.
The ordered window5 graph is unchanged: all five rollouts precede the first C
update, and each output is freed after its actual consumer. No `r_all`/GAN
persistent owner is borrowed or nested.

The runtime additionally verifies the real FP32 DTensor 2×8 ownership before
using this capability. It records new phase schema, mesh, task digest and actual
acquire/release/backward/commit census. Full-safe native actions reject a
reference-phase receipt, even if its numerical counters happen to match.

### Restart and current output

New training identity:
`distillschedule.helios_native_safe_stream_identity.v1`.
New policy:
`distillschedule.helios_native_safe_stream_checkpoint_policy.v1`.
Both bind the exact new execution descriptor, source fingerprint, immutable
assets, input stream/horizon, per-rank state/layout and cursor. The existing
all-rank atomic checkpoint, state restore, stop control and output-only owner
are reused. No temporary full materialization, rollout or kernel cache is
serialized. Old reference/SafeBundle checkpoint identities are not relabelled.
Regenerate only source-bound stream metadata/controls through the existing
producer from retained inputs and **the same** original config and seeds; never
rehash old-source traces into a current stream or resample exceptions.

<a id="helios_nogan_native_checkpoint_inputs_cpu_v1"></a>
## W2 native-history full SafeBundle checkpoint-hidden CPU preservation v1

Public opt-in: `helios_nogan.native_history_safe_ed8_checkpoint_cpu.v1`.
Training identity: `distillschedule.helios_checkpoint_hidden_cpu_identity.v1`.
Checkpoint policy: `distillschedule.helios_checkpoint_hidden_cpu_checkpoint.v1`.
Executed boundary receipt:
`distillschedule.helios_g_checkpoint_hidden_cpu_execution.v1`.

This separately versioned capacity candidate composes the unchanged
[native-history SafeBundle contract](#helios_nogan_native_history_safe_v1).
It does not widen or replace old presets, original history geometry or original
checkpoint callable admission. The mathematical algorithm, all true gradient
chains, RNG and call/update inventory are unchanged. No parameter/optimizer
offload or kernel tuning is authorized by this contract.

The only eligible saves are the exact first hidden tensor of each of the pinned
40 original whole Generator checkpoint blocks, once for each of the three
differentiable B1 pyramid stages. Shapes, BF16 dtype and capacity are exact;
extra inputs/stages and altered ownership/callables fail closed. Eight disjoint
rank partitions bound pinned or pageable node storage before allocation.

Registration is admitted while the original sharded FP32 DTensor owners are
visible, before FSDP materialization substitutes compute Parameters. The
already-admitted wrapper is installed only inside the complete-block phase.
All 120 updating-exit boundaries must copy and release their inputs, but not
every exit remains connected to the sole final-x0 loss: later original no-grad
prefixes sever earlier stages. The original validated executed stage schedule,
not configured exits or a restore-count range, determines the required suffix.
Each stage/block/shape record must show exactly one required restore and release,
or an actual unused release for a dead prefix. No extra backward is introduced.

Current policy/source, full SafeBundle, asset/input/mesh/version identities and
the declared capacity are admission requirements before original construction.
The unchanged startup/running host reserves are 512/64 GiB, and restore preserves
the unchanged 512 MiB GPU reserve. Fresh observations are not resource leases
against unrelated applications. Complete native checkpoint publication and
same-topology restore retain their existing all-rank/RNG/data/state gates.
Temporary input leases are drained and rebuilt, never saved as training state.

See the [public recipe and exact byte accounting](../../recipes/helios_nogan/CHECKPOINT_INPUT_CPU.md)
for the actual runtime boundary, host files, executed counters and refusal rules.
CPU synthetic/component proof cannot promote native HSDP capacity, numerical
parity, timing or quality. Target Torch 2.10 and target GPU evidence remain
**NOT_RUN** unless independently executed; `native_GO` remains false.

<a id="helios_nogan_safe_stream_v1"></a>
## W2 native SafeBundle fresh stream v1 (explicit research opt-in)

Preset: `helios_nogan.safe_stream_ed8.v1`. Policy:
`distillschedule.helios_safe_stream_policy.v1`. This is a new direct strategy,
not a revision of `helios_nogan.safe_ed8.v1`, a saved benchmark plan, or the
original reference policy. Select this strategy explicitly through its versioned policy.

### Immutable workload and physical policy

- Original No-GAN 33-frame, 384x640, 222-stage, physical B1/global16 helpers.
  G1/C5/EMA5 per complete cohort; the source-qualified Step10 bootstrap remains
  G2/C10/EMA10, with original named AdamW, LR schedulers and EMA recurrence.
- Exactly 2..100 distinct, prepared `original_helios_cohort_stream.v1` cohorts.
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

### Assets, dependencies and execution

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

### Complete-boundary checkpoint and restore

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

The original 64-cohort ceiling is superseded only by the
[bounded fresh-stream horizon](native_output.md#W1_W2_NATIVE_STREAM_HORIZON_V1).
Native current-G output has its own [output ABI](native_output.md#HELIOS_NATIVE_OUTPUT_V1);
the frozen two-cycle benchmark and repeated-trace bounds are unchanged.

<a id="helios_original_bounded_requests_v1"></a>
## Original Helios 14B bounded requests (opt-in)

This adapter executes the existing No-GAN/GAN training helpers, not Wan1.3B or
a generic DMD/Euler replacement. Canonical profiles and old generated identities
are unchanged. A request has its own identity, input bundle, measured costs and
serialized native plan; historical seconds cannot be reused as its costs.

Admitted dimensions: 384×640, 17 or33 video frames, active three-stage budgets
(1,1,1) or(2,2,2), world16, physicalB1/global16, G1/C5. No-GAN has EMA5;
GAN has EMA1 and retains its physical/logical checkpoint-version mapping.
Different ratios and physical8/global16 are rejected until their original
training implementations are supported, not relabelled as ordinary world8.

`OriginalHeliosRequest` is propagated through candidate IDs/assignments,
profile semantics, component cost keys, fresh trace metadata, actual helper
geometry, generated model inputs, observations and final receipts. GAN's
logical B1 fake/real/R1 pack remains physical B3 with half-spatial crops.

The exact asset identity is `trace/bundle.json` file SHA256. Its embedded
content hash is independently verified against16 request-qualified rank
manifests and bound to launch arguments/environment. Local-launch admission
compares against that verified hash, never an unconstrained environment
exception or an old canonical trace sentinel.

Default scope `source_normal_only` rejects all-zero-history amplified traces.
Explicit `benchmark_v4_strict_global` retains original sampled history and
per-rank native (possibly doubled) scheduler grids, but uses the historical
rank0-synchronized physical exit/call-count benchmark scope. This is not a
claim to reproduce upstream rank-local amplified exit probabilities. Each
trace records the complete native shifted grid, executed prefix, history
predicate and request hash. Before model construction, the worker recomputes
the grids from pinned original source and verifies hash-checked history sums.
Actual model timesteps/shapes are checked after fenced cycles, outside timers.

Native GPU validation is a separate gate. CPU tests, plans and this document do
not establish capacity, numerical equivalence or speedup for a new request.

<a id="original_helios_record_epochs_v1"></a>
## Original Helios finite feature-pool epochs v1

**Explicit policy:** `distillschedule.helios_record_epochs.v1`. This is a new
input identity policy inside the existing W2/W3 stream kinds, not a trainer,
optimization preset, algorithm or native precision change. Absence of the
policy preserves the old v1 globally-distinct-path/payload refusals and output
fields. Old captured traces cannot be relabeled as new draws.

### Public preparation

Use the existing `distillctl data prepare-helios-stream` (W2) or
`prepare-helios-gan-stream` (W3), with `--input-policy
distillschedule.helios_record_epochs.v1`. The existing `--sources` file is then
an object with exactly these fields:

- `schema_version`: `distillschedule.helios_record_pool.v1`;
- `seed`: nonnegative 63-bit integer, not a boolean;
- `cycles`: explicit finite integer horizon, 2 through 100;
- `evidence_kind`: `ENCODED_FEATURES_UNVERIFIED_QUALITY` or
  `CPU_SYNTHETIC_TEST_ONLY` (neither attests pretrained quality);
- `records`: at least 80 records, each exactly `id`, absolute `path`, and
  `sha256`, pinned to an existing original `.pt` or `.safetensors` feature.

No feature files are created, copied, hard-linked or aliased. ID, resolved path,
file payload and semantic native tensor content must be distinct in the
immutable inventory. The existing safe `load_feature` owns latent/text/mask
geometry, finiteness, padding semantics and source-hash checks; no resize,
projection, replacement data or unsafe pickle fallback is introduced.
The duplicate-content check normalizes validated binary masks to boolean only
for that check. The original mask dtype and tensor are retained for capture;
an integer/float encoding of the same mask is not another distinct record.

One local `random.Random(seed)` permutation of the inventory repeats until
`80 * cycles` logical draws have been assigned. No record is dropped; the last
epoch may be partial. Every consecutive complete 80-draw cohort has 80 distinct
records, even when it crosses an epoch boundary. Extra pool records are allowed.
Training horizon and native steps never reset at an epoch boundary.

### Record, draw and native operation identity

Prepared identity separates immutable record descriptors and exact order from
each row's `record_id`, `draw_index`, `epoch_index`, `epoch_offset` and `draw_id`.
Draw IDs bind policy, source inventory/hash, seed, cohort, native step and rank.
The policy hash binds all finite draw IDs. Each cohort and rank trace provenance
also binds the policy, cohort and exact record/draw census. Both native producer
and consumer recompute source inventory and exact ordering; rehashing edited
metadata is not an alternative sampler or authorization to weaken these gates.

The existing `capture_rank` bodies remain byte-identical. They call the original
dataset with `base_seed=source.seed`, `_epoch=task.index` and rank-qualified
indices. Existing operation-keyed `input_stream(seed, cohort, rank, step,
phase, stream)` contexts generate fresh history, noises, stage decisions,
timesteps, GAN crops and R1 epsilon as before. Reused original feature data is
not replayed captured rollout/noise. This is not a claim of upstream sequential
training-RNG continuation. Preparing/revalidating this policy uses local RNG
only and must not advance formal training RNG.

### Existing lifecycle and evidence boundary

The native training identity and receipts expose the policy/inventory/draw
hashes; the full stream hash also authenticates every row and rank trace.
The existing complete-cycle checkpoint cursor `C` means the next logical draw
is `80*C`, under that exact immutable finite policy; policy or order changes
invalidate restore. The new policy requires an integer cursor at the same
safe cycle boundary. No extra sample is read or operation RNG consumed merely
to reconstruct that cursor. Receipts distinguish actual cross-cohort record
reuse from fresh logical draws and report the next draw cursor explicitly.

Original W2/W3 source, asset, numerical, capacity, branch, precision, update,
physical/curriculum/EMA clock and rank-own gates are unchanged. W3 merged-LoRA
N70 stays OFF. This policy does not extend graph/branch or trainer capabilities,
qualify model assets, import old checkpoints, or supply missing dependencies.
CPU native-format loading, metadata-only 100-cohort admission and checkpoint/RNG
fixtures are protocol evidence, not actual native capture, target Torch 2.10,
world16 GPU, pretrained quality, numerical parity or performance evidence.
