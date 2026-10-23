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
`denoising` record. Full-GAN adds `real: 5` native latent/prompt batches, the
actual 1280-channel, 104,874,241-parameter head and explicitly chosen loss
weights. Nothing resizes that head for CPU tests. Generator LoRA uses the
upstream target modules and masks, not a substitute adapter architecture.

Single-step SDXL is **not** inferred from a base SDXL bundle: an explicit
`generator_checkpoint` is mandatory. The original `conditioning_timestep:399`
case is admitted only together with that actual single-step checkpoint; set
`denoising:false`, `backward_simulation:false`, `num_denoising_step:1` for that
pure-generation path. No checkpoint is downloaded or invented.

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
