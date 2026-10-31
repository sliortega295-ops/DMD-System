# Current Student export from the full C51 pipeline

W4 LiveWan's full C51 continuation can export the **current trained Student**
from a complete checkpoint, then generate from new text features and a supplied
clean opening world. This is the native dual-pool pipeline, not the separate
CPU Program training recipe or official Self-Forcing.

An existing assembled full Student can instead enter through
[explicit weights initialization](livewan_weights_initialization.md).
That command publishes the same portable format without repeating training;
it does not restore an old optimizer or replace the complete-checkpoint route.

## Export

Start with the same `training.c51-stream.json`, source, assets, input manifest,
seed, horizon and checkpoint cadence that produced the checkpoint. Add the
`trainer.export` object from
[`export.native.template.json`](../../recipes/self_forcing/export.native.template.json).
Its codec config and complete FP32 safetensors inventory are local file pins
(`path`, `bytes`, `sha256`); no weights are downloaded. The codec is the actual
Wan2.1 `AutoencoderKLWan`, with its native mean/std normalization.

Run on the original two-node, eight-ranks-per-node topology:

```sh
torchrun <your original world16 launch arguments> -m distillschedule export \
  /PROJECT/training-with-export.json \
  --checkpoint /PROJECT/train/checkpoints/cycle-000100 \
  --output /PROJECT/outputs/current-student
```

The command restores every required role/Adam/data/RNG shard, including Score
ranks. It exits **before** streamer/KV creation, text projection, transport,
lookahead or an update. A terminal checkpoint is legal for output, but remains
illegal for continuing training past the configured horizon. Cadence and
horizon are not rewritten to admit an otherwise different checkpoint.

Only rank 0's complete resident Student and the separately admitted codec are
staged to CPU and published. Score ranks participate in metadata/error checks;
they do not acquire a Generator copy. Parameters, gradients, modes, clocks and
training RNG are left unchanged. The actual nonpersistent complex RoPE state is
included, not approximated or reconstructed from guessed defaults.

Outputs:

- `current-student/`: full inference weights and hash-bound `_SUCCESS`;
- `current-student.inference.json`: independent generation configuration;
- `current-student.native-owner/`: separate output-owner receipts.

These weights do **not** contain optimizer or training state. Use the original
complete checkpoint, not the inference export, to resume training.

## Generate with new inputs

Create a safe preencoded record with exactly:

| Tensor | Shape | Meaning |
|---|---|---|
| `context` | `[1,512,4096]`, CPU FP32 | new raw positive text embeddings |
| `world` | `[1,16,N,46,80]`, CPU FP32, `N` in `3,6,9` | supplied clean opening context |

Import it with `distillctl data import`; the generic preencoded format preserves
the complete tensors and source identity. No training trace, negative prompt,
fabricated history, resizing or implicit prefix crop is used.

If the clean opening is RGB rather than already-normalized latents, use the
explicit `distillschedule.livewan_native_new_inputs.v1` request and preparer:

```sh
CUDA_VISIBLE_DEVICES='' distillctl data prepare /PROJECT/new-opening.json \
  --preparer distillschedule.data.livewan_native_inputs:prepare_inputs \
  --output /PROJECT/prepared-new
```

The request has exactly `schema_version`, `evidence_kind`, `geometry`, `budget`,
`codec`, `positive` and `opening`. Use the native evidence kind and unchanged
geometry/budget/codec from the export request. `positive` contains a complete
`manifest` file pin (`path`, `bytes`, `sha256`) and one explicit `sample_id` from
an already-prepared **new raw positive prompt**. Its sole `prompt_embeds` tensor
must already be exact FP32 `[1,512,4096]`; this step neither reruns T5 nor pads,
casts or substitutes a training prompt. `opening` contains `world_frames`
(`3`, `6` or `9`) and a `frames` list with exactly `(world_frames-1)*4+1`
distinct, complete RGB PNG file pins. Every supplied image must already be
368×640; this list is the explicit clean history, not an implicit crop of a
latent training trajectory. Supplying the first nine images of an existing
video is legal only when those nine files are explicitly chosen and pinned;
their origin is retained and is not relabeled as this Student's output.

Only the pinned original VAE is loaded, on CPU FP32. All supplied frames are
encoded deterministically using posterior **mode**, then normalized as
`(z-latents_mean)/latents_std`. No Student is loaded during preparation and no
training RNG is consumed. The published dataset is create-only and records the
exact prompt feature, opening images, codec files and request/source identities.

Choose **one** preparation route; the following import is only for an already
normalized preencoded record. Both routes use the same independent generator:

```sh
distillctl data import /PROJECT/new-inputs.json --output /PROJECT/prepared-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 distillctl generate /PROJECT/outputs/current-student.inference.json \
  --weights /PROJECT/outputs/current-student \
  --prepared /PROJECT/prepared-new/manifest.json \
  --output /PROJECT/generated --output-kind video
```

The independent consumer uses the original four-step `FewStepStreamer`:
block3, renoise, clean `t=0` KV commit, cache22, no noisy context and no graphs.
Its explicit independent generator stream does not consume training RNG.
The world prefix plus generated suffix yields 21 normalized latent frames;
the original VAE decodes them to 81 RGB frames at 368×640. `--output-kind latents`
skips decoding while keeping exactly the same sampling path.

## Validation and capacity

The portable consumer currently runs **CPU FP32/MATH attention**, preserving
the exact FP32 native Student storage. This is a separate inference precision
from mixed-precision CUDA training; it does not claim bitwise GPU output parity.
The explicit budget covers full Student/codec loading, complex RoPE, KV,
attention workspace and host/cgroup reserve. Score weights are never broadcast
to manufacture a uniform export owner. Sharded or DTensor primary weights are
rejected by the complete-replica publication mode.
The consumer enters an explicit context-local, no-grad inference capacity
scope after host admission. The existing tiny CPU reference/training attention
limits are unchanged outside that scope; SDPA math and masks are unchanged.

Tests use the actual original Wan class, real block backward/Adam, original
streamer, original VAE, a fresh-process generation path and real Gloo role
partitioning. Tiny fixture geometry is explicitly 32×32 and cannot be selected
by the native production export. Production world16 CUDA output and pretrained
quality remain later acceptance gates.

## Collecting an independent output

The generic generation `_SUCCESS` binds `generation.json`, not each separate
PNG. A read-only native output collector checks the actual public export and
prepared dataset, finite FP32 latent output and unchanged clean prefix, exact
81-frame RGB geometry, and pixel equality between every PNG and the actual
decoded tensor. It publishes a fresh separate hash inventory, without repairing
or rewriting the original output or marker:

```sh
CUDA_VISIBLE_DEVICES='' python -m distillschedule.evidence.livewan_native_output \
  --weights /PROJECT/outputs/current-student \
  --prepared /PROJECT/prepared-new/manifest.json \
  --generated /PROJECT/generated \
  --receipt /PROJECT/reports/current-student-output.json
```

Use a new generation directory and receipt for every attempt. A failed or
occupied attempt is never reused. The collector's PASS is an **output component
only**: it does not accept a failed native owner, rerun numerical validation,
establish generation quality or claim a speedup. Weights-initialization exports
retain their original cycle/source and numerical result at current cycle zero;
an assembled C100 Student output must not be called a C2 checkpoint Student
output. Separately validated C1/fresh-C2 checkpoint components do not repair an
original owner/finalization failure either.
