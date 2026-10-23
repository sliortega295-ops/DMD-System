# Original H3 AR native task v2: exact rank input metadata

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

## Validation status

Local CPU tests cover all 16 exact metadata rows, rank6 regression, negative
path/SHA/shape/dtype/packing/conditioning cases, actual Session constructor
handoff before model construction, parser file pins, common measured-plan
consumers and rehashed task tampering. Synthetic meta tensors in tests are
not native values or GPU evidence. **Pending target CPU admission and cluster
validation**: the root owner must run the actual original parser over all16
original remote caches on both nodes under the new source before reserving
any remaining native start. No performance or numerical parity is claimed.
