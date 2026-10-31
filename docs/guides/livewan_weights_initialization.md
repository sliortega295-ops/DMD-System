# Import an existing Self-Forcing Student

Use an existing, complete FP32 Student to create a portable inference artifact
without running more training. This path supports the original assembled
LiveWan Student format. For a complete framework training checkpoint, use
[checkpoint export](livewan_native_output.md) instead.

## Import and generate

The independent consumer uses CPU FP32 and one CPU thread:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl import-weights \
  /PROJECT/initialization.json --output /PROJECT/current-student

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl generate \
  /PROJECT/current-student.inference.json --weights /PROJECT/current-student \
  --prepared /PROJECT/new-inputs/manifest.json --output-kind video \
  --output /PROJECT/new-video
```

The command publishes the standard `weights.pt`, `export.json`, `_SUCCESS` and
companion inference configuration. It includes the full Student and Wan2.1 VAE;
subsequent generation does not need the original training data or checkpoints.
A portable artifact is not a resumable training checkpoint.

## Initialization request

The request is an explicit JSON object:

| Field | Value |
|---|---|
| `schema_version` | `distillschedule.livewan_generator_weights_initialization.v1` |
| `kind` | `GENERATOR_WEIGHTS_INITIALIZATION_ONLY` |
| `source` | `livewan_generator_import.source_identity()` from the installed implementation |
| `parameters` | Complete existing `livewan_native_current_generator.v1` inference parameters, without a `device` override |
| `student` | `{ "payload": PIN }` for the assembled full Student |
| `codec` | `{ "config": PIN, "weights": [PIN, ...], "provenance": { "status", "origin", "license" } }` |
| `origin` | `{ "contract", "assembly", "selected_summary", "reference_summary", "numerical_comparison", "model_source" }` |

Each `PIN` is `{ "path": "/absolute/canonical/file", "bytes": N,
"sha256": "64 lowercase hex digits" }`. The origin contract is
`distillschedule.original_livewan_assembled_student.v1`; its remaining fields
are pins of the original assembly receipt, both execution summaries, comparison
report and original `wan/modules/model.py` source.

The model parameters use the same architecture, geometry, runtime, codec and
capacity contract as the [native output consumer](livewan_native_output.md).
The existing [export policy template](../../recipes/self_forcing/export.native.template.json)
provides geometry and capacity fields. Supply actual local codec pins and its
license; the command does not download assets.

The original payload contains exactly `model`, `step`, `generator_updates`,
`source_head`, `input_binding` and `scope`. Every parameter must be present in
its original full name, contiguous FP32 shape and finite value. The importer
does not fill missing parameters from a base model, average replicas, select
EMA, merge LoRA, or infer a prefix mapping. Source files and payloads are checked
before publication; existing output paths are never overwritten.

## State and provenance

- Original source, counters, input identity, execution summaries and comparison
  results remain in `identity.weights_initialization`. New initialization uses
  cycle zero; it does not relabel old training as current-source continuation.
- Optimizers, schedulers, EMA, Teacher, Critic, training RNG and data cursors are
  not loaded. The caller's current RNG is preserved.
- The assembled parameter payload lacks Wan's nonpersistent `freqs`. The
  importer reconstructs it with the exact original `rope_params(1024, ...)`
  constructor expression, records that policy, and includes the result in the
  portable artifact. Later consumers strictly restore that exported state.
- Capacity admission covers the full model, codec, attention/KV workspace and
  initialization copies. Loading uses the existing safe tensor/checkpoint
  readers and common atomic publisher.

## New input format

A prepared sample contains exactly two detached, finite CPU FP32 tensors:

- `context`: `[1,512,4096]` positive text features;
- `world`: `[1,16,N,46,80]` clean opening latents, with `N` explicitly `3`, `6`
  or `9`.

The unchanged four-step `FewStepStreamer` generates the remaining blocks;
Wan2.1 VAE decoding yields 81 RGB frames at 368×640. The consumer never silently
crops a longer history or reuses a training row. Use `--output-kind latents`
only when decoded media is not required.
