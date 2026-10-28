# Independent H3 Student inference

The native-layout consumer preserves the original base tensors and **unmerged
FP32 Student LoRA**. It supports the full original 50-block H3 architecture;
small original-class configurations are explicitly labeled fixtures. It never
loads a Fake Score adapter, discriminator, optimizer or old training cache.

## Execution and assets

Use an existing compatible Torch 2.10 CPU environment with the
`h3-cached-reference` dependencies and one compute thread. This is an explicit
CPU dense-attention output path, not an automatic fallback from distributed
CUDA training. BF16/FP32 tensor storage is preserved per key; computation uses
CPU BF16 autocast with the original eight-step continuous schedule. There is
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

## Current integration boundary

`h3_ar_portable` supplies the real independent model/rollout consumer and the
common artifact carrier. `h3_ar_inference` supplies the backend-native
current-Student consolidation primitive. Connecting the native W6 checkpoint
restoration to the public `export` command is a separate integration; this
guide does not claim that producer is already exposed. Source/CPU coverage is
distinct from full native CUDA, pretrained generation quality and W5 Grid1000
compatibility.
