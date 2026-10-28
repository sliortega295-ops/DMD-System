# Original LiveWan finite teacher-pool epochs v1

**Policy:** `distillschedule.livewan_teacher_epochs.v1`. This is an explicit,
versioned input policy inside `original_livewan_c51_teacher_stream.v1`, not a
new trainer, algorithm, precision mode or optimized preset. Its absence retains
the frozen v1 producer, manifest and no-record-reuse refusals unchanged.

## Immutable records and finite draws

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

## Unchanged native computation

All six existing operation-keyed seed streams remain derived from seed,
operation key and stream: rollout, Generator DMD sigma/noise, gradient blocks,
Critic sigma/noise. The existing separate `control` derivation still determines
world frames and rope gap. Local preparation, input reconstruction, cursor
restore and operation RNG do not consume Python/Torch's global training RNG.
Fresh operation RNG does not mean fresh teacher archives or upstream sequential
RNG equivalence. Continuous shifted-uniform sigma, native precision,
368x640/81-frame geometry, global B16, physical G B1 / Score B2, G1/C5/EMA0,
dual-wave transport and all parameter/cache version rules are unchanged.

## Existing C51 ownership and restart

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
gates, capacity checks, native math or quality status (`REVIEW` / `NOT_RUN`
until independently qualified). CPU native-format fixtures are protocol
evidence only, never pretrained/GPU/world16 numerical or performance evidence.
