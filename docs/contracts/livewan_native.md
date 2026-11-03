# LiveWan native training and Teacher input contracts

The original-reference owner, legacy finite cohorts and C51 finite-pool epochs
have separate identities. A logical data cursor alone does not certify C51's
physical dual-wave drain. Use the [input operations](../guides/data.md#livewan_inputs)
and the [C51 recipe](../../recipes/self_forcing/C51_CONTINUATION.md).

<a id="original_livewan_reference_training_v1"></a>
## Original LiveWan reference training v1

Schema: `distillschedule.livewan_reference_training.v1`.
Asset schema: `distillschedule.livewan_reference_assets.v1`.
Input: [`original_livewan_teacher_cohorts.v1`](#original_livewan_teacher_cohorts_v1).
Checkpoint policy: `distillschedule.livewan_reference_checkpoint.v1`.

This independently versioned, opt-in reference route does not change any
canonical C51 candidate, historical trace, source digest or two-cycle worker.

### Training ownership

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

### Admission

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
- Actual owned CUDA UUID resolves uniquely to the same physical inventory entry
  at admission and capacity checks. This route requires physical index
  `rank % 8`; unsupported device reorderings and drift reject.
- Runtime environment identity omits credential values and unknown Hub keys.
  Explicit credential fields reject; declared execution controls remain bound.

### Complete-cycle state

The common atomic all-rank checkpoint stores actual trainables, named Adam
moments and steps, commit versions, native cursor and per-rank RNG. It binds
source/assets/config/precision/optimizer/topology/input identity and the fixed
target horizon. A complete cycle k requires actual Gk/C5k/EMA0, cursor 5k and
no pending gradients on original model or optimizer owners.
Original detached Generator text-projection slots remain zero-step/zero-moment;
other Generator slots are at step k, Critic slots at 5k. These checks bind the
actual optimizer's parameter order to canonical names before accepting state.

All required rank files and completion marker must exist. Hash/layout/name/
dtype/mode/cursor/version/Adam-step mismatches reject before mutation. Loading
preserves Parameter/optimizer owners and hooks. Frozen Teacher/base state and
temporary streamer caches are reconstructed, not mistaken for trainable state.
Only an unfinished same-target, same-topology boundary may resume; extending
the target or converting a C51 physical role-pool checkpoint is unsupported.

### Verification and nonclaims

Actual CPU input/LoRA/Adam/DTensor state and Gloo failure tests, original
meta-constructors and explicitly labeled dispatch fixtures qualify source and
owner contracts only. CUDA/pretrained FSDP2/NCCL execution, physical capacity,
numerical equivalence, generated quality and performance remain NOT_RUN.
This contract supplies neither C51 transport drain/shadow reconstruction nor
native generation/export, and cannot relax their existing rejection gates.

<a id="original_livewan_teacher_cohorts_v1"></a>
## Original LiveWan finite teacher cohorts, version 1

Input kind: `original_livewan_teacher_cohorts.v1`.
Manifest schema: `distillschedule.original_livewan_teacher_cohorts.v1`.
Cursor schema: `distillschedule.livewan_teacher_cursor.v1`.

This contract describes input ownership only, not a new training algorithm,
C51 checkpoint, model export, or quality measurement. Historical FormalTrace,
C27 source hashes and C51 selected-plan contracts remain unchanged.

### Source and record identity

- Original LiveWan gitlink `1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`, clean
  tracked tree, exact committed dataset/generator/prompt-generator/license bytes.
- Exactly `80 * N` distinct original teacher records for `1 <= N <= 64` finite
  cohorts. Canonical paths, archive bytes and latent tensor content must be
  distinct. Copies, replay, missing records, padding and silent repair reject.
- Each archive has only `latents`, `prompt_idx`, `seed`, `size`, `frames`,
  `steps`, `cfg`: finite FP16 latent `[16,21,46,80]`, 81 frames, `640*368`,
  40 teacher-generation steps, nonnegative integer seed and positive finite CFG.
- The original prompt bundle has FP16 `pos[P,512,4096]`, `neg[1,512,4096]`,
  matching nonempty prompt strings and a negative-prompt string. Its archive
  and positive tensor identities are bound. A record index is not a prompt
  index; independent records may share one prompt.
- Restricted tensor loading precedes original dataset loading, without an
  unrestricted fallback. The actual original TeacherLatents consumes the
  admitted canonical files. Source and archives remain operator-owned and
  immutable; concurrent local filesystem writers are outside this guarantee.
- Only complete manifest bytes publish, with exclusive atomic publication.
  Existing output files and historical evidence are never replaced.

### Logical schedule and RNG

The unchanged workload shape is 368x640, 81 frames, logical world16/globalB16,
G1/C5/EMA0 per complete cycle, continuous shifted-uniform sigma. Each cohort
owns five steps and sixteen logical B1 slots. Cohorts do not share records.
Teacher generation steps in an input archive are not Student rollout steps.

A local RNG seeded by the explicit 63-bit seed determines record order. Keys
are `step=S/slot=R` for a single cohort, otherwise
`cycle=C/step=K/global_step=S/slot=R`. World-frame and RoPE-gap controls and the
six existing operation seed streams are derived from those keys. Construction,
seeking and consumption do not draw from Python/Torch training RNG. This
independent operation-keyed stream is **not** upstream sequential-RNG
continuation; the manifest states that distinction explicitly.

### Cursor and failure behavior

A logical rank cursor advances only after actual original dataset consumption.
Save/restore admits only complete five-step boundaries and binds source,
input/prompt/order/seed identity, rank, world16 topology and next operation.
Foreign identities, incomplete cycles and changed payloads reject before cursor
mutation. Exhaustion stops; there is no implicit next epoch or trace replay.

This logical cursor is not acknowledgement of C51 physical two-wave consumption
or a drained role-pool pipeline. Fresh public W4 training remains rejected until
those native lifecycle owners are connected and qualified. No GPU execution,
pretrained quality or full native model restart follows from data admission.

See the [operator guide](../guides/data.md#livewan_inputs) for preparation and use.

<a id="original_livewan_teacher_epochs_v1"></a>
## Original LiveWan finite teacher-pool epochs v1

**Policy:** `distillschedule.livewan_teacher_epochs.v1`. This is an explicit,
versioned input policy inside `original_livewan_c51_teacher_stream.v1`, not a
new trainer, algorithm, precision mode or optimized preset. Its absence retains
the frozen v1 producer, manifest and no-record-reuse refusals unchanged.

### Immutable records and finite draws

The real pinned `TeacherLatents` dataset must contain at least 80 distinct
original teacher archives. All existing source, license, restricted loading,
prompt, dtype, geometry, metadata, path, archive hash and semantic latent hash
gates apply. Copies, hard-link aliases and reserialized identical latents do
not qualify as additional records. Preparation never writes teacher records.
Neither archive format nor this policy attests generation provenance or
pretrained quality.

The existing local `random.Random(seed).shuffle` produces one permutation of
the immutable record inventory. The policy records the exact record order,
record count, `80 * cycles` draw count and finite epoch count. That same seeded
permutation repeats without dropping records until the declared horizon ends;
the final epoch may be partial. No implicit horizon extension is supported.
For any pool size at least 80, every consecutive complete 80-draw G1/C5 cycle
has 80 distinct records, including a cycle crossing an epoch boundary. Record
reuse is allowed only across cycles under this declared policy.

Each row keeps `record_id`, archive/prompt bindings and dataset index separate
from `draw_index`, `epoch_index`, `epoch_offset`, `draw_id` and `cohort_id`.
Logical cycle/step/global-step/slot and operation keys never restart at an
epoch boundary. The draw identity binds policy, seed, operation, immutable
record and epoch position. Producer, native consumer and C51 metadata admission
recompute the exact order, row identities and controls; a rehashed edited
manifest does not bypass them.

### Unchanged native computation

All six existing operation-keyed seed streams remain derived from seed,
operation key and stream: rollout, Generator DMD sigma/noise, gradient blocks,
Critic sigma/noise. The existing separate `control` derivation still determines
world frames and rope gap. Local preparation, input reconstruction, cursor
restore and operation RNG do not consume Python/Torch's global training RNG.
Fresh operation RNG does not mean fresh teacher archives or upstream sequential
RNG equivalence. Continuous shifted-uniform sigma, native precision,
368x640/81-frame geometry, global B16, physical G B1 / Score B2, G1/C5/EMA0,
dual-wave transport and all parameter/cache version rules are unchanged.

### Existing C51 ownership and restart

The public C51 preparation command explicitly opts in with `--input-policy`.
The existing `self_forcing.c51_stream.v1` preset and native `InputLoader` consume
the manifest; no generic model-name or second training path is added. The
physical `PoolInputs` descriptors carry logical draw identities as well as
record identities. Every cycle's existing joint census still requires 80
distinct record IDs and operation keys across both role pools.

The existing drained complete-cycle checkpoint binds the full manifest input
hash, finite horizon, source/asset/seed/topology identity, physical progress,
dual-wave history and next operation keys. Policy/order/draw-count changes
therefore invalidate restore. Restoring the data cursor does not read the next
original sample or consume operation/training RNG. The next actual operation
uses its next logical key, even when its immutable record was used before.
Old historical training checkpoints are not imported by this policy.

Runtime receipts must report actual cross-cycle record reuse and the policy
schema/hash. The old no-replay receipt still requires `False` without policy
metadata. None of this relaxes mechanism counters, physical clocks, source
gates, capacity checks or native math.
