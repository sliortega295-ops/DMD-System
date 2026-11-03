# H3 request, native AR and diagnostic contracts

W5 Grid1000 and W6 previous/current AR are distinct algorithms and input owners.
A task/metadata admission is not new GPU numerical or performance evidence.
Keep native source receipts, original measurements and FAIL/REVIEW boundaries;
recalibrate on a changed admitted source. [Native commands](../guides/native_workloads.md)
and [AR output](../guides/training.md#h3_ar_native_inference) use these contracts.

<a id="h3_attention_tuple_capture_v2"></a>
## H3 actual-attention tuple capture v2

This opt-in diagnostic observes the original first Fake update at cycle 1,
rank 0, block 10. It calls the original backward once and saves its inputs and
gradient outputs. Ordinary training and performance measurement do not enable it.

### Bind the sample, not the cache capacity

H3 packs `[text | stereo audio | video]` without sequence padding. The text
length depends on the selected condition. A maximum-sized replay cache is not
the shape of every attention call.

Version 2 uses:

- schema: `distillschedule.h3_native_attention_capture_opt_in.v2`;
- selector: `h3.w5.c1.fake0.block10.actualtuple.v2`;
- a sealed, exact Q/K/V/output shape and corresponding LSE shape;
- the existing sample, native-config, source, provider, owner and capacity
  bindings.

After the original Fake input has been prepared, the observer reads its text,
audio and video **shape metadata**, and the loaded model's head count and head
dimension. Their packed shape must equal the sealed geometry before entering
the selected Fake forward/backward. No tensor values are read and no extra
layout, device copy or random sample is produced.

For example, 37,296 video rows, 414 stereo-audio rows and 17 text rows yield
37,727 attention rows, not the 37,760-row maximum. The original input is neither
padded nor cropped to make the diagnostic fit.

The reviewed maximum shape and byte ceilings remain unchanged. Version 1
retains its original fixed geometry. Schema/selector combinations cannot be
mixed. At capture, every tensor must still match its exact admitted dtype,
shape, device, layout and storage constraints.

### Evidence consumers

`native_capture_geometry(packet)` provides the same versioned geometry check to
the worker and tuple consumers. A v2 tuple includes `prepared_geometry` in its
binding; readers must compare it with the sealed packet and validate all nine
tensor files against that geometry rather than a maximum-size constant.

Capture receipts remain correctness diagnostics, excluded from timing/cost
claims. An intact tuple is not a numerical-equivalence result: the original
native completion checks and subsequent same-tuple analysis remain separate.

<a id="minimax_h3_bounded_requests_v1"></a>
## Bounded original H3 Grid1000 requests, v1

`distillschedule.h3_request.v1` admits exactly 768x1344 and 480x832, each with
56 or 124 frames. The default remains 768x1344x124. This is a versioned request contract, **not** a replacement or
sixth canonical best-known profile. The five canonical JSON files and their
lock hashes remain unchanged. Each noncanonical request has a distinct semantic
fingerprint and resolved profile. Its historical section contains no inherited
canonical seconds.

Pinned LightX2V `d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be` H3
`packing.py::video_latent_num_frames` requires `num_frames % 17 == 5` and
returns `(num_frames - 5) // 17 * 5 + 2`. Thus 56 and 124 are valid native
packing lengths; 60 is invalid. This contract admits only the two bounded
lengths, not arbitrary native lengths or Wan's frame lattice. Audio packing
retains the native `round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND)` rule.

Everything except the explicitly selected resolution/frame count and physical cache policy
retains the original canonical H3 protocol: approximate discrete Grid1000,
world16 on 2x8 A100, B1/global16, BF16, SP1, LoRA128, four rollout evaluations,
fixed end step 3, Student1/Fake5, segment1, activation policy none, FA3 splits2,
all 50 replay blocks, seed20260817, no EMA. The native two-cycle adapter runs
one process lifetime, advances G2/F10, and does not reset models/RNG between
cycles. This does not support autoregressive H3 or continuous-exact sigma.
Eight-GPU topology requests and arbitrary shapes remain unsupported. The
CPU-staged matched control is still approximate Grid1000, not a continuous
reference baseline.

### Physical implementation identities

Legacy `joint-h3v1`/`joint-h3v2` retain CPU-staged cache plus trim, their old IDs,
and their default behavior. New `joint-h3v3` IDs include the request and either:

- `cpu_staged_trim`: pageable reservoir and bounded pinned staging, trim on;
  inflight depth 1 or 2, LoRA epilogue N64 or N128.
- `cuda_no_trim`: existing GPU-resident FA3 out/LSE replay, trim off;
  inactive D2H depth fixed at 2, LoRA epilogue N64 or N128.

Other storage/trim combinations are not admitted. The CLI inventory exposes
the native macro family (no new rollout reordering). A v3 optional `-r1..5`
retains the separately gated action family; it requires action-qualified
calibration and prior action proof. It cannot enter the macro collector.

### Invariants and evidence

The selected ID, request, resolved model config and environment must agree
before model construction. The selected frame count enters both
`contract.frames` and `training.dmd.num_frames`; a 124-frame loaded config
cannot satisfy a 56-frame candidate. After native setup the actual installed cache
registration must match; v3 per-cycle receipts include its policy and native
counters. CUDA must have zero CPU cache/staging/transfer and trim counters and
nonzero native replay forward/backward execution. The original workload-owned
per-cycle FA3/Grid1000/census/version gates remain unchanged.

Request/shape, storage/trim and kernel choices enter cost keys. H3 v3 macro
planning requires joined, hash-bound native calibration results. Missing costs
remain unavailable; neither parent canonical seconds nor short external
480x832 observations are imported as calibrated costs. Serialized lowering
rechecks original rank observations and the selected cost model. Cross-request
comparison and cross-workload dispatch fail closed.

CPU fixtures establish wiring and rejection behavior only. A new source HEAD
needs matched target-GPU correctness and calibration, then independent native
confirmation. Sampled two-cycle numerical REVIEW is not full-state or
long-horizon equivalence, and diagnostic timing is not formal speedup proof.

<a id="original_h3_ar_capacity_v2"></a>
## W6 native capacity policy v2: explicit node host limits

This versions only capacity metadata for the original H3 AR adapter. The task,
training math, geometry, update ratio, timer, source/asset identity gates and
other workload defaults do not change. New calibration and saved execution
must use one unchanged source HEAD; older GPU evidence is not rewritten.

The existing unversioned v1 policy remains accepted with exactly four scalar
integer fields: `gpu_limit_bytes`, `gpu_reserve_bytes`, `host_limit_bytes` and
`host_reserve_bytes`. It still rejects unsupported capacity observations.

V2 has exactly five top-level fields:

- `schema_version`: `distillschedule.original_h3_ar_capacity.v2`;
- shared `gpu_limit_bytes` and `gpu_reserve_bytes`;
- `host_limits`: exactly string keys `"0"` and `"1"`, positive integer bytes;
- shared `host_reserve_bytes`.

No default limit or reserve is supplied. Every rank resolves its node using
`rank // 8`; an absent, invalid or mismatched node is rejected. Receipts retain
the complete identical policy on all16 ranks and record explicit node/host-limit
bindings inside memory evidence. Assembly, calibrated hardware/capacity binding,
saved-plan validation and execution preserve the full map, including the peer
node's limit; node-local resolution never replaces the saved common policy.

A missing `memory.peak` remains `None`/`UNAVAILABLE`, never current-as-peak. Each
node must independently prove its declared limit equals its **fresh observed
finite hierarchy hard limit**, with maintained host reserve 0 and unchanged
OOM/oom_kill counters. A historical process `memory.max` snapshot, physical
capacity alone, or `min(node0,node1)` is not a replacement for that observation.
Available lifetime peaks retain their existing current/highwater/reserve checks.
The separate original 64 GiB entry headroom gate remains unchanged. GPU capacity,
shared reserves and every numeric gate retain their meaning; v2 does not relax
or infer any number.

CPU tests cover distinct exact limits with unavailable peaks, explicit wrong-node
rejection even for equal limits, v1 compatibility, schema rejection, full-policy
assembly/plan tamper rejection, and the actual worker saved-plan branch on both
node ranks. No target GPU or full-model numerical result is established here.

<a id="original_h3_ar_continuation_v1"></a>
## W6 native finite continuation v1

Opt-in preset: `h3_ar.zero3_ap_stream.v1`. Input kind:
`original_h3_ar_fresh_stream.v1`. This is the actual original native
`H3ARSession` / `DMD2Trainer` / `AlternatingAdamW` / DeepSpeed ZeRO-3 owner,
not the independent cached-AV CPU reference. Its reference choice is explicitly
`zero3-native` through this same owner. No fallback occurs.

### Unchanged mathematics and bounded scope

The original 50-block model, previous56/current56 AV, 768x1344/24fps,
eight-step rollout, G1/F1/EMA0, LoRA64, world16 B1, native BF16 with autocast
cache disabled, optimizer ownership and FP32 masters are unchanged. Native
GAN start remains 500: it must remain inactive throughout this 2..100-cycle
contract. Rank6 retains its original 94-frame audio schema; other ranks retain
their own SHA-bound original parser metadata. No crop, padding, normalization,
fake condition, replacement sampler or changed loss is admitted.

The old immutable rank-input manifest and `native_graph` two-cycle entry retain
their exact guards and reject this manifest. A new finite graph emits the same
whole native update, operation order, census and version transitions.

### Fresh input authority

The manifest binds the original source receipt, rank-input contract, one seed,
an immutable finite horizon, and exactly 16 distinct records per cycle. Records
are explicitly ordered cycle-major, then rank0..15. Each supplies `cycle`,
`rank`, unique `record_id`, absolute cached-record `path` and `sha256`, plus a
nonempty `sources` mapping of named original source/encoding file references
(`path`, `sha256`). All used input bytes are checked. Preparation derives tensor
shape/dtype/scalar metadata from the original deterministic parser; training
checks it again. No model or media is downloaded or encoded by this command.

Every rank validates its next original parser result before collective model
allocation. That same CPU result is consumed by the shared constructor.
Subsequent fresh inputs are prepared before dispatch and committed only after
the actual update/census/clock checks succeed. A failed update is not replayed
or represented as a completed cycle.

### Complete native checkpoint

The common `training.checkpoint` publisher remains authoritative. Native engine
files are sidecars inside the same all-rank, SHA-bound, `_SUCCESS` publication.
Each actual engine calls `save_checkpoint(tag='state', save_latest=False,
exclude_frozen_parameters=True)`. Registered permanent trainables include both
LoRAs and every inactive discriminator parameter. A scoped `requires_grad`
policy includes them while excluding the frozen backbone; `finally` restores
every original flag on both success and failure. No optimizer codec is invented.

The boundary stores trainer/student/fake versions, native `global_steps=2*cycle`,
`active_role`/`train_heads`, named trainability and module modes, per-rank
Python/NumPy/Torch/owned-CUDA RNG and the fresh stream cursor. The manifest binds
current framework/source-layout/native/helper/checkpoint/input identities,
precision, topology, capacity, horizon, seed and effective DeepSpeed config.
Native ordered `_get_zero_param_shapes` partitions and actual writer flags and
file-name getters are recorded, not inferred from current trainability.

Restore requires a pristine new-process native engine, the same topology and
all common/native shard hashes. Wrong identity, missing/corrupt files, named or
ordered layout/config drift refuse before restore mutation. Native model-shard
keys, dtypes, source-version/config and client-state fields are audited first;
`load_module_strict=False` permits only intentionally absent frozen parameters.
The actual `load_checkpoint` restores optimizer states and FP32 masters.
DeepSpeed0.18.4's engine source is pinned to
`ef3b4797809453f4941eaa769c9cd4e397e3b52085131e8916990ea78d4f0f6c`.
Native checkpoints are trusted pickle artifacts: never restore untrusted files.
An I/O/backend failure during native mutation is not advertised as rollback-safe.

Before publication the unchanged epoch helper proves consumption/event drain,
advances only its read-only view and clears result/FIFO/prefetch metadata.
These transient objects and pinned buffers are never checkpoint state. On
resume, actual trainer and optimizer clocks are restored; only `view.origin` and
process-local observer baselines are rebased. An empty new pool never receives
`begin_next_cycle`. First resumed staging is honestly cold, with the frozen
helper's historical origin-based metadata retained separately. Later drained
cycles reuse only process-local pinned storage, never stale results.

Source/CPU protocol fixtures do not qualify native world16 continuation, full-model
capacity or numerical/quality behavior. The separate output consumer is documented
in [AR output](../guides/training.md#h3_ar_native_inference); it is not a continuation proof.

<a id="original_h3_ar_native_v1"></a>
## W6 original DiffSynth H3 AR native adapter v1

This **opt-in task adapter** uses the common `distillctl workloads native --task`
front door, `search.service.select`, source-bound `CostModel`, and
`PreparedRolePlan`. It is not a sixth canonical frozen profile. The existing
five-profile registry, defaults, historical receipts and reported times are
unchanged. `workloads native list-adapters` discovers task schemas without
loading models.

### Frozen workload, not a protocol replacement

- Exact DiffSynth KS3 object snapshot: source-receipt SHA256
  `9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989`.
  This is not a verified Git commit. Admission reconciles the **complete Python
  source inventory** against that immutable receipt, not just four entry files.
- Original previous/current AV chunks: current 56 frames at 24 fps (~2.33 s),
  clean previous AV reference, native 768x1344 cached geometry. Actual target
  video `[1,24,17,48,84]`, audio `[2,32,93]`, previous video rows `[17136,96]`,
  previous audio rows `[186,32]`. Text length remains sample-qualified.
- Eight-step original DMD2; per update Student9/Fake2/Teacher1 forwards,
  Student1/Fake1 backwards and real Adam updates; G1/Fake1/EMA0; LoRA64;
  original BF16 autocast with `cache_enabled=False`, original FP32 master Adam.
  The dormant GAN branch remains configured with its original start step500;
  it must be inactive in both fresh cycles. No GAN/EMA/source-step substitution.
- Original `h3_dmd2.DMD2Trainer.update`, `H3Adapter`, `prepare_batch`,
  `dmd2_zero3.configure/initialize/step`, checkpoint helper and RNG progression.
  `H3ARSession` binds those functions; it does not copy/rewrite training math.
- One world16 group, two nodes x eight A100-SXM4-40GB; real validated
  Torch2.10.0+cu128. No world1/world8/Grid1000/fullseq/Diffusers fallback.
  The pinned native startup metadata helper intentionally rejects other GPUs.

### Admitted physical choices

Only two existing native mechanisms can be calibrated:

| ID | Actual native implementation |
|---|---|
| `zero3-native` | Original ZeRO3 communication/checkpoint policy, no added attention-result cache/fusions |
| `zero3-ap` | Accepted AP bundle: native async comm, block leaf/replay release, unit prefetch, pointwise/SwiGLU/RMS/LoRA implementations, CPU-staged selected-attention replay cache with bounded H2D prefetch and version-qualified epoch invalidation, host-register4 pinned allocator |

Both retain no parameter/optimizer offload and no **checkpoint-boundary**
offload. AP is **not copy-free or all-GPU-resident**: its attention replay
cache performs real D2H/H2D transfers and retains pinned host memory. Those
costs remain inside the opaque native update and common host capacity policy.
The saved plan binds every bundle choice, exact helper/library SHA256, source,
checkpoint admission, rank cache manifest, topology, runtime and capacity.

Historical native first-update control (~316.095 s) and accepted AP Cycle2
255.6706443373114 s are provenance only. Neither is embedded as a cost or an
automatic winner. They cannot establish timing on this new integration source.

### Honest planner scope

The single graph operation is **one complete original native update**. Its
version contract after warmup is Student1/Fake1/EMA0 -> Student2/Fake2/EMA0.
EMA0 is read/preserved, never falsely reported as an EMA write.

Each implementation needs fresh unprofiled two-cycle, all16-rank native
calibration receipts. Cost is max16 actual fenced Cycle2 duration; capacity is
post-setup static allocation plus a conservative all-rank observed additional
allocation envelope. No fake G/F components, overlap discounts, historical
scalar costs, or inferred new layouts enter the common planner. A candidate
that is unmeasured or outside common capacity is rejected. Search is limited
to these admitted observed implementations, not a global optimum claim.

`run` revalidates and independently recomputes saved cost/graph/order/timeline/
memory lowering before model construction. It takes its actual implementation
from the saved plan; no recipe override or old shell/worker filename is accepted.
The same persistent native session executes both updates through the common
prepared handler. Original native census/Adam/version/cache epoch and backend
execution witnesses must pass. C1 is warmup, C2 is the real second native
update with no RNG/optimizer reset; native CUDA fences/barriers remain in the
timer. State/RNG hashing and epoch bookkeeping remain outside timers.

### Source/assets and safety

Large checkpoint files are hashed **once in owner CPU/I/O admission before GPU
reservation**, with unchanged before/after stat. The hash-bound audit records
per-file SHA256 plus size/mtime/ctime and origin-host dev/inode. Workers strongly
hash the small audit/config/index/source, check complete payload inventory/stat,
and record a local stat witness; they do not re-read all weights to hash them.
Cross-host stat checks use size/mtime/ctime, never compare mount-local dev/inode.
This is explicit stat-bound reuse of the same payload identity, not a fresh
cryptographic full-payload read on every worker. Any stat/inventory change fails
closed and needs owner re-admission. Same-host dev/inode are also checked.

All source/helper/library pins, cache identity, actual sample geometry and
current capacity are validated before or during the original native path.
No SSH, GPU allocation, cleanup of foreign processes, download, installation,
or new launch scheduler exists in the adapter. An external authorized owner
must coordinate the two-node torchrun and resource sampling. Internal cgroup
`memory.peak`, when available, is conservative lifetime highwater, not an
isolated workload host peak. Missing `memory.peak` is **None/UNAVAILABLE**,
never sampled `memory.current` relabeled as peak. The identical common
ResourceObserver policy admits that capability only with the **exact observed
finite cgroup hard limit**, zero maintained host reserve and unchanged observed
OOM/oom_kill counters. A stricter budget or nonzero reserve without a real peak
is rejected before model construction. Each receipt records the observed
cgroup hierarchy/physical capacity, current separately, peak capability and
evidence scope; assembly and the saved plan bind that capability. Sampled
driver-free checks do not prove an unsampled reserve minimum.

### Validation status

Pending matched cluster validation; numerical status is
`REVIEW_REQUIRES_MATCHED_REFERENCE`.



### Capacity provenance and new policy relation

The historical C2 closeout reported peak allocated **33.728214263916016 GiB**
and peak reserved **38.416015625 GiB**. Node0/node1 owner lifecycle receipts
reported sampled host peaks **463624593408 / 464590323712 bytes**, sampled
minimum driver-free **2097152 / 60817408 bytes**. Sampling was about 5 seconds;
these minima do not certify an unsampled reserve minimum.

The original two-cycle owner explicitly required **64 GiB cgroup headroom at
entry** and zero OOM/oom_kill increments throughout. No explicit sustained GPU
reserve or sustained host reserve floor was found in that owner/manifest;
those historical policy fields remain **UNKNOWN**, not inferred to be 1 GiB.

The new worker has **no default capacity policy**: all four policy byte fields
must be supplied identically for calibration/manual/selected runs. It retains
the original 64 GiB **entry-only** gate independently. `host_reserve_bytes`
means a separately declared maintained check, not that historical entry gate.
Do not populate it with 64 GiB merely because the original owner had an entry
gate, or import the unrelated GAN 1 GiB GPU reserve. Preserve the authorized
original policy; any additional reserve is a new explicit constraint and may
reject a previously accepted implementation. The new receipt records the
actual startup current/maximum and does not reset native GPU peak counters
between setup, C1 and C2.

The four-field scalar form remains supported. For nodes with different host
hard limits, use the explicit W6-only capacity-v2 map described in
`original_h3_ar_capacity_v2.md`; it preserves the same numeric gates and binds
each rank's actual node while retaining one complete common policy.

Sources: `h3_diffsynth_dmd2_world16_20261010/zero3_200s_r2/steady_two_cycle_20261012_r1/reports/remote_closeout_r1/{world16_two_cycle_result_r2,node_0_lifecycle,node_1_lifecycle}.json`
and `scripts/native_zero3_node_owner_two_cycle_r1.py:89-90` in that same run.

<a id="original_h3_ar_native_v2"></a>
## Original H3 AR native task v2: exact rank input metadata

This version corrects the v1 adapter's **metadata gate**, not native training.
The unchanged original cache/parser produces `[2,32,94]` current audio and
`[188,32]` previous audio on **rank 6 only**; the other 15 original samples
produce `[2,32,93]` / `[186,32]`. All ranks retain 56 current + 56 previous
video frames, 768×1344, original video latents `[1,24,17,48,84]`, previous
visual anchors `[17136,96]`, and the exact semantics in the v1 contract.
Prompt/packed sequence lengths also remain sample-specific.

`distillschedule.original_h3_ar_task.v2` binds the SHA256 of
`configs/distillschedule/original_h3_ar/rank_inputs_v1.json`. That curated
small file contains no tensor values: it binds each rank's original path/SHA,
raw/prepared tensor shapes/dtypes/packed layout, the original clean-reference
conditioning scalars, cache manifest, both original preparation source SHAs,
full native source receipt, and the CPU observation's SHA/source provenance.
The observation is **not timing/calibration evidence**.

The same validator is used immediately after unchanged original
`prepare_batch` in the live Session and in the mandatory owner CPU preflight
of **all 16 caches on both nodes**. It forbids wrong-rank identities, audio
93/94 substitution, packed/dtype drift, changed masks/conditioning/parser,
and alternate samples. There is no crop, pad, reshape, cast or new parser.
The worker serializes the exact input witness; the thin controller,
calibration assembler, component contract, saved-plan loader and live saved
run all bind the same rank-qualified authority. Missing old witnesses fail.

The common `workloads native --task` router exposes task v2. Canonical five
frozen profiles and other native adapters are unchanged. Explicit v1 requests
reach the H3 adapter only to fail closed as obsolete, not to silently upgrade.

New source/task identities require **new matched native/AP calibration and
manual/selected runs**; v1/56c receipts and historical C2 seconds cannot be
reused as costs. Both prior pre-model failures remain preserved. GPU policy,
capacity v1/v2 gates, optimizer/EMA/census, helper/library/source assets,
BF16 autocast, timing fences and unreset C2 state/RNG are unchanged.

### Validation status

Local CPU tests cover all 16 exact metadata rows, rank6 regression, negative
path/SHA/shape/dtype/packing/conditioning cases, actual Session constructor
handoff before model construction, parser file pins, common measured-plan
consumers and rehashed task tampering. Synthetic meta tensors in tests are
not native values or GPU evidence. **Pending target CPU admission and cluster
validation**: the root owner must run the actual original parser over all16
original remote caches on both nodes under the new source before reserving
any remaining native start. No performance or numerical parity is claimed.

<a id="original_h3_ar_record_epochs_v1"></a>
## Original H3 AR cached-record epochs v1

Policy: `distillschedule.h3_ar_record_epochs.v1`. Prepared kind:
`original_h3_ar_epoch_stream.v1`. This is an explicit finite-data policy for
the existing W6 native DMD2/ZeRO-3 lifecycle; the original fresh-stream v1
remains the default and still rejects repeated records.

### Sources and logical draws

The source list contains genuine, already cached previous/current audio-video
records, each qualified for its original rank. Rows contain exactly `rank`,
`record_id`, `path`, `sha256` and `sources`. Preparation supplies and checks the
original parser metadata; callers must not author that proof. Each rank needs
one or more records, no more than the requested horizon. Source IDs, canonical
paths and file hashes are globally unique. Provenance references are content
pinned. No new record files, tensors, padding or shape conversions are created.

The prepared document stores `source_records` independently of its `draws`.
For each rank, a private `random.Random(seed + rank)` permutes its source
indices once. This order repeats to the finite horizon, including a partial
last epoch. Draws are cycle-major/rank-major, and record the source index,
logical draw index, epoch/offset and a draw hash. Thus 16 source records used
for 50 cycles are **16 sources and 800 logical draws**, not 800 fresh archives.
Constructing this schedule does not consume the training RNG.

### Admission and runtime

The policy, source inventory and complete draw schedule are regenerated and
hash checked. The existing original parser qualifies each immutable source
once during preparation and validates actual records again during consumption.
Startup staging, later cycles, joint rank receipts and output-only restoration
use one shared record resolver and input identity. Rank-specific prompt lengths,
rank 6 audio geometry, masks, scalar conditions and precision remain unchanged.

Training must explicitly declare the epoch input kind. Pairing an epoch
manifest with the fresh kind, changing the seed/order/source, or using an
unknown policy fails before model construction. All model, loss, G1/F1/EMA0,
optimizer and ZeRO-3 rules remain those of the original native owner.

### Checkpoint boundary

The existing complete-cycle checkpoint binds `input_sha256`, horizon, seed,
source identity and cursor. These fields cover the epoch schedule; no new
native checkpoint ABI is needed. Resume uses the identical prepared document
and topology. An output-only restore resolves the saved draw without consuming
a next training draw. Caches are reconstructed, not serialized as training state.

Source/CPU behavior is validated independently of full-size GPU training.
This policy does not alter W5 Grid1000 or its outstanding numerical diagnosis.
