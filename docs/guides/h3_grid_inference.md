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

Generation returns **packed** `video_tokens` and `audio_tokens`, each with an
explicit `BLD` latent schema. These are not RGB/video/audio media, nor W6's
`BCTHW`/`SCT` VAE interface. The converted-token-to-codec ABI remains a separate
item; there is no guessed reshape, frame count, decoder, or normalization.

## Verification and remaining binding

New CPU tests use the actual pinned converted model in a bounded random
configuration, actual LoRA gradients/Adam update, actual frozen AdaLN results,
the unmodified original sampler, and strict common export/restore. The
four-step packed output matches the original projection path exactly on this
CPU fixture. A separate process executes the public generation command on a
new condition without importing either training owner or W6's sampler.
These random weights establish implementation behavior, not pretrained quality.

**The public native W5 checkpoint-to-export owner bridge is not yet bound.**
Until it is, the three pieces above do not upgrade `distillctl export` for a
native W5 training checkpoint. The existing original-reference checkpoint and
resume route remains unchanged; the optimized backward numerical hold is
unchanged. Actual world16 consolidation, pretrained output, codec binding,
and the later 100-cycle reliability checks are not supplied by CPU tests.
