# Official DMD2 / Stable Diffusion XL

Train actual upstream SDXL `SDUniModel`/`SDGuidance` with full Generator parameters
or native Generator LoRA, native epsilon losses and optional full-size GAN head.
Single-device CPU/`cuda:N` uses FP32. Source, isolated `train,dmd2` dependencies,
CC-BY-NC-SA-4.0 license, pinned revision and explicit local/Hub asset preparation
are shared with [SD1.5](../dmd2_sd15/README.md#source-environment-and-local-assets).

## SDXL assets and native conditioning

The local bundle additionally requires `text_encoder_2/` and `tokenizer_2/`.
Native `SDXLTextEncoder` joins both penultimate CLIP hidden states, uses the
second encoder's pooled projection and passes size/crop/target time IDs. CFG
uses zero unconditional token/pooled embeddings. Both tokenizers are required
for prompt and raw-image preparation.

## Choose a training policy

[training.json](training.json) uses four-step backward simulation with schedule
`[999, 749, 499, 249]` and ratio five. Each macrocycle consumes independent
`generator: 1`, `guidance: 4` and `denoising: 5` prompt batches. Native prefix
simulation/renoising remain inside SDUniModel: initial denoising-index draws
are consumed before simulation overwrites them, and t999 becomes pure outer
noise. Both LR schedulers step five times; G commits once and guidance five times.

For real preencoded denoising latents, set `backward_simulation: false` and put
exact FP32 B4HW latents in every denoising record. The source manifest must assert
`"preencoded_vae_compatibility":"PREENCODED_LATENTS_MATCH_LOCAL_VAE"`, binding
both tokenizer inventories and matching VAE bytes. This asserts compatibility,
not that preparation encoded those latents.

Full-GAN adds `real: 5` native latent/prompt batches and the actual 1280-channel,
104,874,241-parameter head, with explicitly chosen classifier loss weights.
Generator LoRA uses upstream target modules/masks and separate unmerged base/A/B
parameters; the classifier is not resized for tiny models.

[training.single_step.json](training.single_step.json) requires an actual strict
native `generator_checkpoint` for `conditioning_timestep:399`, not just base
SDXL weights or a framework restart checkpoint. It sets `denoising:false`,
`backward_simulation:false`, `num_denoising_step:1`, with independent
`generator: 1`/`guidance: 4` streams and no denoising stream.

## Prepare, train and export

Use [inputs.template.json](inputs.template.json) for prompts/pinned latents or
[images.template.json](images.template.json) for explicit actual-VAE encoding
at 1024-square or another declared compatible geometry. Native RGB encoding,
posterior seed and exact preprocessing binding follow the [shared SD input
contract](../dmd2_sd15/README.md#prepare-inputs-and-choose-the-workload).
Fill all local paths in a copy of the selected training template:

```sh
distillctl data prepare /PROJECT/inputs.json --output /PROJECT/prepared \
  --preparer distillschedule.data.sd_inputs:prepare_inputs
distillctl train /PROJECT/train.json --cycles 1 --output /PROJECT/partial
distillctl train /PROJECT/train.json --cycles 3 \
  --resume /PROJECT/partial/checkpoints/cycle-000001 --output /PROJECT/resumed
distillctl export /PROJECT/train.json \
  --checkpoint /PROJECT/resumed/checkpoints/cycle-000003 --output /PROJECT/export
```

Use returned checkpoint paths. [Native UNet checkpoint policy](../dmd2_sd15/README.md#optional-native-unet-checkpoint-policy)
and [component-v1](../../docs/guides/native_owner_backends.md#per-role-dmd2-checkpoint-policy)
also apply to qualified CPU FP32 SDXL: Generator checkpointing may coexist with
Fake epsilon activation retention. They preserve denoising/conditioning/update
semantics; Teacher, dual CLIP and VAE stay outside the option.

Export contains current Generator, dual CLIP and native VAE. Prepare new prompts
with matching tokenizer files and the native SD preparer, then use the
[common inference-request builder](../../docs/guides/training.md#generate-and-export):

```sh
distillctl generate /PROJECT/inference.json --weights /PROJECT/export \
  --prepared /PROJECT/new-prepared/manifest.json --output /PROJECT/generated --output-kind image
```

For random tiny actual dual-CLIP/UNet/VAE assets, use the shared fixture helper
with `--variant sdxl`; its generated training request is directly runnable:

```sh
python recipes/dmd2_sd15/create_cpu_fixture.py /PROJECT/random-sdxl \
  --source-root /PROJECT/dependencies/DMD2 --variant sdxl
distillctl train /PROJECT/random-sdxl/training.json
```
