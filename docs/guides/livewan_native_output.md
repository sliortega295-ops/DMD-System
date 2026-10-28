# Current Student export from the full C51 pipeline

W4 LiveWan's full C51 continuation can export the **current trained Student**
from a complete checkpoint, then generate from new text features and a supplied
clean opening world. This is the native dual-pool pipeline, not the separate
CPU Program training recipe or official Self-Forcing.

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

```sh
distillctl data import /PROJECT/new-inputs.json --output /PROJECT/prepared-new
distillctl generate /PROJECT/outputs/current-student.inference.json \
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
