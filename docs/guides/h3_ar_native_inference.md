# Independent H3 Student inference

The native-layout consumer preserves the original base tensors and **unmerged
Student LoRA in its actual forward storage dtype**. It supports the full original 50-block H3 architecture;
small original-class configurations are explicitly labeled fixtures. It never
loads a Fake Score adapter, discriminator, optimizer or old training cache.

## Execution and assets

Use an existing compatible Torch 2.10 CPU environment with the
`h3-cached-reference` dependencies and one compute thread. This is an explicit
CPU dense-attention output path, not an automatic fallback from distributed
CUDA training. BF16/FP32 tensor storage is preserved per key; computation uses
CPU BF16 autocast with the original eight-step continuous schedule. The native
BF16 engine's LoRA tensors remain BF16; an FP32 stored adapter remains FP32.
FP32 optimizer masters are not substituted for current forward weights. There is
no merged LoRA, weight quantization or alternate sampler.

The model and DMD2 numerical source are the existing pinned original closure:
the DiT is byte-identical to native H3 and DMD2 changes only its import paths.
The inference artifact contains the full frozen base and current Student, so
independent generation does not reopen a training checkpoint or its assets.
All tensor keys, shapes and dtypes must match before strict restoration.
Random or uninitialized weights cannot generate before restoration.

Provide explicit parameter, attention, working-set and host-reserve limits.
Full H3 is large even for inference. The consumer checks a meta model and fresh
physical/cgroup headroom before allocating real weights; it does not silently
shrink the architecture, lower precision or change the input shape to fit.

## New previous/current conditions

Prepare a new record through the existing
[original H3 cached-AV input path](../../recipes/h3_ar/CPU_REFERENCE.md), retaining its prompt,
clean previous video/audio and original packing information. The record's
declared geometry must match the artifact. Current latent values are shape
templates; the original rollout draws fresh noise. No old training sample,
precomputed rollout or training RNG state is used for independent generation.

Given the common inference artifact and its matching inference configuration:

```sh
distillctl generate inference.json --weights /project/current-student \
  --prepared /project/new-conditions/manifest.json \
  --output /project/new-output --output-kind latents
```

The output contains named `video_latents`, `audio_latents`, exact layout
metadata and `_SUCCESS`. Decode media separately using
[`distillctl decode`](h3_audio_video_decode.md) and actual original codec
weights. Do not reshape latents into pixels or trim the audio/video streams.

## Export a native W6 checkpoint

Start from your original fresh-stream training configuration and add the
explicit `trainer.export` section in
[`export.native.template.json`](../../recipes/h3_ar/export.native.template.json).
Keep its original horizon, seed, checkpoint cadence, assets, execution strategy,
capacity policy and source identity. Both the reference and full optimized
native owners use the same output-only restoration path.

On each already-launched world16 rank, with the same admitted native runtime:

```sh
distillctl export native-export.json --checkpoint /project/checkpoints/cycle-000100 \
  --output /project/current-student
```

This restores the complete native checkpoint, including original FP32 master
Adam and rank RNG, before selecting the current Student. It performs **zero
training updates**. A terminal checkpoint is allowed only for export; ordinary
training resume still requires a nonterminal boundary. Construction reads the
actual saved rank input without advancing its cursor or asking for cycle 101.

The existing ZeRO-3 owner gathers one layer at a time. Only the primary retains
the full frozen base and Student weights on CPU. Neither Fake Score nor the
discriminator enters the inference artifact, and LoRA is not merged. The
original checkpoint config, complete tensor layout and storage dtypes must
match. Output budgets cannot reduce the native host or device reserve.

The command publishes `current-student/` and `current-student.inference.json`.
Use the latter with the independent `generate` command above. Its geometry and
seed are explicit inference settings, not changes to the training stream.
Full-size CPU dense attention needs substantial host memory; export and
inference each check their own declared budget. Export is not a replacement for
a resumable training checkpoint.

Source/CPU tests cover the restore-to-export control chain and actual original
model consumer. Native world16 CUDA export and pretrained output quality still
require their corresponding hardware acceptance. W5 Grid1000 remains a
different model/workload binding.

## Explicit native-output reproduction (not a generate fallback)

The separate opt-in [`native-output` reproduction tool](../../recipes/h3_ar/NATIVE_OUTPUT_REPRODUCTION.md)
can hand an explicitly pinned R375 C50/AP40-history/new232-text request to its
original two-node owner and historical runtime. It does not generalize the
portable input consumer, retag old checkpoints or authorize GPU ownership.
