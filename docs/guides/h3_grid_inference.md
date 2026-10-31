# W5: current Grid1000 Student output

W5 uses the **converted Diffusers MiniMax-H3 model**, the original LightX2V
four-evaluation sampler, and separate video/audio flow shifts **6/3**. It is
not the W6 DiffSynth eight-evaluation autoregressive sampler. W5 keeps
LoRA128; W6 keeps its distinct LoRA64 and previous/current conditioning.

## Why the export includes a rollout table

Grid1000 training physically removes each block's frozen AdaLN projection.
Its remaining model state is therefore **not a complete ordinary H3 checkpoint**.
Rebuilding the deleted projections with random weights, substituting continuous
AdaLN, or passing it to W6's inference code would change the model.

The independent W5 representation contains:

- all remaining frozen Transformer weights;
- the **current, unmerged Student** LoRA, without Fake Score or optimizer state;
- exactly four frozen BF16 rollout-table entries from the admitted Grid mmap;
- the actual otherwise-nonpersistent RoPE buffer, including its native dtype.

No LoRA merge, projection reconstruction, sigma re-quantization, or old
training sample is used. The first rollout step has one unique timestep;
the original Grid ABI stores two equal copies, of which only the addressed
rows are read. Later steps retain their distinct video/audio timesteps.

## Implemented components

`distillschedule.training.h3_grid_inference` supplies:

1. `current_student_view`: a read-only view of the real current Student plus
   those four table entries; sharded weights remain DTensors. It does not
   perform an all-gather or claim capacity admission for that collective.
2. `export_components`: strict projection of an explicit, already consolidated
   CPU snapshot into the common inference artifact format.
3. `build_inference_model`: a separate source/runtime-bound consumer using
   the actual converted Diffusers class and PEFT adapters. It preserves every
   supplied weight dtype, loads all buffers strictly, and checks host capacity
   before concrete allocation.
4. The common `distillctl generate` path on **new prepared text features**, with
   the original LightX2V sampler, original noise order, FP32 latents, data-ward
   velocity convention and four evaluations.

The constructor and sampler come from hash-pinned operator-supplied source.
There is no model download or fallback to another H3 implementation. Use the
existing W5 dependency environment: Diffusers0.40.0.dev0 with the converted H3
class and Torch2.10. The operator must supply source terms and exact assets.
Discovery of the adapter itself imports no Torch, model or CUDA.

## Inputs and outputs

Use the existing `distillctl data import` command to prepare a new
condition. The per-sample tensor tree is exactly:

```text
prompt_embeds:   BF16 CPU [1, text_tokens, 5120]
text_token_tags: int64 CPU [text_tokens], every tag 1
```

The geometry, independent inference seed, model layout, source and original
four-step schedule are bound into the exported configuration; `generate`
does not accept unbound overrides. Training RNG is neither restored nor used.
Missing or mis-shaped conditions and previous-chunk fields are rejected.

`--output-kind latents` returns **packed** `video_tokens` and `audio_tokens`,
each with an explicit `BLD` schema. To generate media, provide an explicit
original H3 codec manifest through `--parameters`:

```json
{
  "decoder_manifest": {
    "path": "/PROJECT/h3-codec.json",
    "bytes": 12345,
    "sha256": "REPLACE_WITH_ACTUAL_SHA256"
  }
}
```

```sh
distillctl generate /PROJECT/current-grid-student.inference.json \
  --weights /PROJECT/current-grid-student \
  --prepared /PROJECT/new-text/manifest.json \
  --parameters /PROJECT/media-parameters.json \
  --output-kind audio_video --output /PROJECT/generated-av
```

The [original codec manifest](h3_audio_video_decode.md) binds both safe weight
inventories, source terms, CPU FP32 execution and capacity. It must match the
original Grid geometry: video `[1,24,T,H/16,W/16]`, stereo audio `[2,32,A]`,
FPS24, 32kHz, `T=(frames-5)/17*5+2` for `17*n+5` frames and
`A=round(frames/24*40)`. The native decoder's temporal planner performs the
final frame check. No cropping, interpolation or duration resampling is added.

The bridge executes the original pinned `unpatchify_video` and `unpack_audio`
definitions: spatial 1×2×2 video patches and **stereo channel-major** audio
rows. Packing conversion changes no values or normalization; the original VAE
applies its mean/std exactly once. W6's eight-step sampler is never used.
`decoder_manifest` changes decoding only, not rollout or Student parameters.
The bridge is also a factory for `distillctl decode` on supplied safe packed
token inputs (`manifest`, `grid_source` and `geometry` factory parameters).

## Verification and remaining binding

New CPU tests use the actual pinned converted model in a bounded random
configuration, actual LoRA gradients/Adam update, actual frozen AdaLN results,
the unmodified original sampler, and strict common export/restore. The
four-step packed output matches the original projection path exactly on this
CPU fixture. A separate process executes the public generation command on a
new condition without importing either training owner or W6's sampler.
These random weights establish implementation behavior, not pretrained quality.
The optional media path additionally checks original packing round trips on
non-square data and runs the public command in a new process through the actual
ViT3D and full BigVGAN decoder. Native GPU and pretrained quality are separate
acceptance stages.

## Export an original-reference native checkpoint

Copy your original fresh-data W5 reference training configuration, keeping
its native config file, horizon, seed, source, assets, checkpoint cadence and
input inventory unchanged. Add the `trainer.export` object from
[`export.native.template.json`](../../recipes/h3_fullseq/export.native.template.json).
The independent inference geometry/seed need not consume training data.

On each already-launched world16 rank, in the same admitted environment:

```sh
distillctl export native-export.json --checkpoint /project/checkpoints/cycle-000064 \
  --output /project/current-grid-student
```

The original owner restores **both role weights, Adam/schedulers, clocks,
sampler position and rank RNG** before output. The output-only cursor validates
its boundary without creating a loader iterator or reading the next sample.
A terminal checkpoint is allowed for export only; normal training resume still
requires a nonterminal boundary. Training performs zero additional calls or
updates. Only output metadata is redirected to a fresh `.native-owner` folder;
no identity-bearing native config is rewritten and no old run is overwritten.

Snapshot admission checks exact table entries and current unmerged LoRA.
All DTensor owners participate in full-tensor materialization; only primary
retains the complete CPU weights. Explicit host/cgroup and largest-tensor
GPU reserves are checked before gathering. Unused CUDA allocator blocks may
be released **in this output-only process, outside training timers** before
the driver-free check; this does not change a training hot path. Source,
request, layout or owner-state failures reject publication. A failed companion
configuration invalidates the new export's completion marker while retaining
the payload for diagnosis.

The command emits a matching `.inference.json` beside the artifact. Use that
with new prepared text conditions and `distillctl generate --output-kind latents`.
Neither Fake Score nor optimizer state is copied into the inference artifact.
This native bridge supports the **original reference** owner; it does not
bypass the selected optimized path's existing backward numerical failure.

Target CPU checks include the real original scalar-edge training checkpoint
restore, zero-update output exit, actual original converted-model publication,
and real two-rank Gloo DTensor gathering. Only the bounded model admission and
cluster edges are CPU fixtures; these do not establish world16 FSDP/CUDA,
pretrained quality, the codec ABI or the later 100-cycle reliability gate.
