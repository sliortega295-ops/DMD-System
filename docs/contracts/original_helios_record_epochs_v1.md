# Original Helios finite feature-pool epochs v1

**Explicit policy:** `distillschedule.helios_record_epochs.v1`. This is a new
input identity policy inside the existing W2/W3 stream kinds, not a trainer,
optimization preset, algorithm or native precision change. Absence of the
policy preserves the old v1 globally-distinct-path/payload refusals and output
fields. Old captured traces cannot be relabeled as new draws.

## Public preparation

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

## Record, draw and native operation identity

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

## Existing lifecycle and evidence boundary

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
