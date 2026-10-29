# W1 Generator weights initialization (not resume)

`distillctl import-weights initialization.json --output portable` is an explicit,
versioned **weights initialization/import** path. It executes **zero training
cycles** and publishes the existing portable inference format, plus
`portable.inference.json`. It never enters the native training worker or loads
optimizers, EMA, Teacher, clocks, streams, historical RNG or a training checkpoint.
Existing `train --resume` and `export --checkpoint` keep their strict complete
state/source/ABI/topology contracts unchanged.

Generate with NEW prepared prompt features through the existing entry point:

```
distillctl generate portable.inference.json --weights portable \
  --prepared NEW/prepared/manifest.json --output-kind video --output NEW-output
```

A successful import is neither quality validation nor evidence of 50 completed
training cycles. `checkpoint_cycle: 0` denotes this NEW initialization boundary,
not a fabricated current-source cycle or full-state continuation. The original
source, receipt and observed end-cycle/version counters remain under
`identity.weights_initialization`; CURRENT model source/dependency/constructor
identity stays in the existing `model` and `model.parameters` fields.
Original `numerical_comparison` and `quality_validation` are copied verbatim as
`origin_numerical_comparison` and `origin_quality_validation`, never promoted to
REVIEW, PASS or quality. The current export's own quality status stays NOT_RUN.

## Explicit request contract

The request has exactly these keys (no resume/checkpoint/config inheritance):

- `schema_version`: `distillschedule.wan_generator_weights_initialization.v1`.
- `kind`: `GENERATOR_WEIGHTS_INITIALIZATION_ONLY`.
- `source`: `wan_generator_import.source_identity()` (exact current importer and
  existing inference factory file SHA256).
- `parameters`: complete existing `wan_native_inference.v2` parameters, as used
  by `project_current_generator` / the existing W1 portable factory. Every
  constructor argument, original PEFT target/exclusion/rank setting, mixed BF16
  base/FP32 unmerged LoRA policy, task, native rollout, fresh inference noise
  protocol, runtime/source identity, codec and capacity limit is explicit. No
  `device` override is accepted by import. This is a CPU/single-owner operation.
- `base`: `{config: PIN, weights: [PIN, ...], indexes: [PIN, ...], pyramid: POLICY, key_policy: KEY_POLICY}`.
  At most 32 safe shards and one index. The exact config/shard/index basename →
  SHA inventory must equal the pinned original receipt's `identity.assets.model_files`.
  The raw config is resolved with the exact original class defaults and existing
  fixed `TRANSFORMER_ADDITIONAL_KWARGS`; the resulting complete constructor must
  equal `parameters.architecture`. Unknown raw fields fail closed.
- `trainables`: `{payload: PIN, parameter_names: [EXACT_ORDERED_NAMES, ...]}`.
  Only the native rank0 `state_evidence.generator` final-trainables safetensors
  producer contract is supported. No optimizer/EMA/checkpoint-envelope parser,
  suffix-map, merge, partial load or trainable state-key rewrite is provided.
- `codec`: existing native output codec `{config: PIN, weights: [PIN, ...],
  provenance: {status, origin, license}}`. The real AutoencoderKLWan loader and
  its strict precision/normalization/full-state admission are reused.
- `origin`: `{contract: "distillschedule.original_wan_rank0_final_trainables.v1",
  receipt: PIN, source_head: ORIGINAL_40_HEX, source_fingerprint: ORIGINAL_64_HEX}`.
  This is an explicit producer-format capability, **not a private historical
  commit allowlist**. Source pins must equal the old receipt unchanged. Production
  task digest, complete finite rank0 receipt, contiguous cycle/version progress,
  writer0/count/exact payload path/size/hash, original task and frozen-base
  inventory are all checked. Receipt pins protect integrity, not authenticity;
  callers must supply a trusted original receipt and original base/payload.

Each PIN is exactly `{path: ABSOLUTE_CANONICAL_LOCAL_FILE, bytes: POSITIVE_INT,
sha256: LOWERCASE_64_HEX}`. Requests/config/receipt/files are bounded and fully
rehashed before publication; symlinks, duplicate JSON keys and nonfinite JSON
are refused. The base must cover the exact bare original model state. Accepted
source base tensors are finite BF16/FP32, converted only by the original mixed
precision policy, with finite results required. Trainables must already be
finite dense FP32, matching every actual trainable name/order/shape/dtype before
any live trainable copy. EMA and arbitrary missing/extra fields are refused.

KEY_POLICY is explicitly `{kind: "EXACT_NATIVE_STATE_KEYS"}` or
`{kind: "ORIGINAL_WAN_FROM_PRETRAINED_KEY_REMAP_V1", source_sha256: ORIGINAL_TRANSFORMER_SHA}`.
The latter pins the existing complete original Transformer source and reproduces
only its exact from_pretrained `scale_shift_table` → `norm_out.scale_shift_table`
base-key compatibility. It requires that raw alias, refuses alias collisions,
and never applies suffix matching or any remap to the trained FP32 LoRA payload.
Raw shard-index ownership is checked before the exact original base-key policy.

POLICY is either `{kind: "COMPLETE_SAFE_BASE"}` or
`{kind: "SOURCE_SEEDED_WAN_MISSING_PATCHES", seed: 20260819}`. The latter permits
only the exact six weight/bias keys of patch_short/mid/long to be absent and
calls the existing original deterministic materializer. No other random base
fallback exists. The callback preserves original **FP32 meta** construction:
only loaded base keys enter the native mixed dtype policy; missing patches stay
FP32 meta, then the original materializer resets FP32 before casting BF16.
The original nonpersistent RoPE constructor is reused too.

## CPU verification and evidence boundaries

`tests/test_training_wan_generator_import.py` covers strict metadata, complete
base and FP32 tensor admission, no-mutation refusals and the real shared
publisher → new-process strict loader using plain Torch (not a W1 substitute).
`tests/test_training_wan_generator_import_upstream_cpu.py` uses the actual full
original tiny Helios Transformer, actual PEFT, native precision, complete,
original seeded-pyramid or raw-Wan-key base loaded by the actual original
from_pretrained producer (not the importer), real existing publisher/factory, independent
strict reload, NEW prompt, original native rollout and actual tiny Wan VAE
decode. Old source files are removed before the independent process, and
optimizer construction is forbidden. Fixture LoRA values are constructed,
**not trained**, and are labelled as such; the tests prove wiring, not quality,
production model capacity, old ≥50-trained weights or GPU qualification.

For a required upstream run, set `R217_REQUIRE_UPSTREAM=1`; absent dependencies
raise a collection error instead of skipping. Reuse an existing environment;
never install or fetch weights as part of testing. The R217 artifact package's
`scripts/run_target_cpu.py` checks exact staged source bytes, requires this mode,
and rejects any skipped case. No new training is needed to test or import.
