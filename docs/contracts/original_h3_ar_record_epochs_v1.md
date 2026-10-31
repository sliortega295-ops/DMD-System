# Original H3 AR cached-record epochs v1

Policy: `distillschedule.h3_ar_record_epochs.v1`. Prepared kind:
`original_h3_ar_epoch_stream.v1`. This is an explicit finite-data policy for
the existing W6 native DMD2/ZeRO-3 lifecycle; the original fresh-stream v1
remains the default and still rejects repeated records.

## Sources and logical draws

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

## Admission and runtime

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

## Checkpoint boundary

The existing complete-cycle checkpoint binds `input_sha256`, horizon, seed,
source identity and cursor. These fields cover the epoch schedule; no new
native checkpoint ABI is needed. Resume uses the identical prepared document
and topology. An output-only restore resolves the saved draw without consuming
a next training draw. Caches are reconstructed, not serialized as training state.

Source/CPU behavior is validated independently of full-size GPU training.
This policy does not alter W5 Grid1000 or its outstanding numerical diagnosis.
