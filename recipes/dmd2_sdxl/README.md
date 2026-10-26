# Official DMD2 / Stable Diffusion XL

Read [the shared source, license, preparation and evidence contract](../dmd2_sd15/README.md).
The same exact pinned upstream source is used; restricted source is external,
and model assets must be explicit local pretrained bundles.

SDXL additionally requires local `text_encoder_2/` and `tokenizer_2/`. The
original `SDXLTextEncoder` joins both penultimate CLIP hidden states, uses the
second encoder's pooled projection, passes native size/crop/target time IDs,
and uses zero unconditional token/pooled embeddings for classifier-free
guidance. This is not single-CLIP SD1.5 conditioning with different shapes.

`training.json` uses native four-step backward simulation, schedule
`[999, 749, 499, 249]`, and ratio five. One common macrocycle consumes
`generator: 1`, `guidance: 4`, and independently sampled `denoising: 5` prompt
batches; original prefix simulation and renoising remain inside SDUniModel.
The initially drawn denoising indices are still consumed before backward
simulation overwrites them, and t999 is overwritten with pure outer noise.
Both LR schedulers step five times; G commits once and guidance five times.

To train from real preencoded denoising latents instead, set
`backward_simulation: false` and provide their exact FP32 B4HW tensors in every
`denoising` record. The source manifest must explicitly declare
`"preencoded_vae_compatibility": "PREENCODED_LATENTS_MATCH_LOCAL_VAE"`;
both tokenizer inventories and the asserted VAE bytes must match the actual
owner. This assertion does not claim that the preparer encoded supplied latents.
Full-GAN adds `real: 5` native latent/prompt batches, the
actual 1280-channel, 104,874,241-parameter head and explicitly chosen loss
weights. Nothing resizes that head for CPU tests. Generator LoRA uses the
upstream target modules and masks, not a substitute adapter architecture.

Single-step SDXL is **not** inferred from a base SDXL bundle: an explicit
`generator_checkpoint` is mandatory. The original `conditioning_timestep:399`
case is admitted only together with that actual single-step checkpoint; set
`denoising:false`, `backward_simulation:false`, `num_denoising_step:1` for that
pure-generation path. `training.single_step.json` supplies this actual
configuration template: independent `generator: 1` and `guidance: 4` streams,
no denoising stream, the explicit local initializing generator checkpoint,
and native timestep399. The checkpoint must contain the actual strict native
generator state for that case, not just a base SDXL bundle or an owner restart
checkpoint. Its bytes are stream-hashed before and after loading. No checkpoint
is downloaded or invented, and no pretrained-quality claim is made.

`images.template.json` uses the shared explicit offline actual-VAE intake for
1024-square RGB batches (or other explicitly declared compatible geometry).
Both native local CLIP tokenizers are mandatory even for raw-image records;
omitting `tokenizer_2` fails rather than substituting a single CLIP tokenizer.
Pinned preencoded latents remain an unchanged alternative.

```sh
python recipes/dmd2_sd15/create_cpu_fixture.py /PROJECT/random-sdxl \
  --source-root /PROJECT/dependencies/DMD2 --variant sdxl
distillctl train /PROJECT/random-sdxl/training.json
```

This explicitly builds random tiny **actual** dual-CLIP/UNet/VAE assets. The
CPU tests cover both variants, native-source loss/gradient/cache/RNG parity,
real Program new-process restart, and inference-only current-G/CLIP/VAE export
on new prompt inputs after removal of training/base assets. The original GAN
heads are tested separately, not claimed as tiny full-GAN end-to-end training.
CPU integration is not pretrained quality, CUDA, or distributed validation.
The shared [single-device CUDA FP32 reference source path](../dmd2_sd15/README.md#cpu-starter-and-evidence-boundary)
also places SDXL's actual dual CLIP, time IDs, denoising schedule and restored
optimizer moments on the selected device. Its actual random-tiny CUDA
oracle and lifecycle tests passed on A100/Torch 2.10 with an explicit
deterministic test policy. They remain NOT_RUN on CPU-only Torch. No AMP,
distributed, production-capacity, pretrained-quality or full-GAN GPU result
is implied. See [acceptance scope](../../docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

The shared [native UNet checkpoint policy](../dmd2_sd15/README.md#optional-native-unet-checkpoint-policy)
also applies to SDXL's actual generator/fake UNets, including backward-simulation
prefixes and native non-G disable/re-enable turns. It changes no denoising
schedule, conditioning, update ratio, optimizer or RNG semantics. The same
manual/auto/saved-plan admission and first-postrestart paired reference gate
apply **on the qualified CPU FP32 stack only**; Teacher, dual CLIP and VAE remain
outside the option. CUDA optimized trials are rejected until capacity-bounded
resident snapshot/restore also covers frozen state and CPU staging.

The independent `component-v1` checkpoint profile also applies to actual SDXL:
Generator retains checkpointing while Fake epsilon activations may be retained,
with separate per-role effect evidence. See the
[native owner guide](../../docs/guides/native_owner_backends.md). Role-DDP
implementation is separate and does not inherit the single-device CUDA or
component-profile qualification.
