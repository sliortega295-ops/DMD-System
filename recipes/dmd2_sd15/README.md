# Official DMD2 / Stable Diffusion 1.5

Train actual upstream `SDUniModel`/`SDGuidance` Generator and Fake Score with
native epsilon losses and optional classifier/GAN. The primary distributed
reference is [world16 FSDP2](training.fsdp2.json); the
[optimized Resident/DDP profile](training.optimized.json) preserves the same
SD1.5 GAN workload. Single-device FP32 and native FP32 role-DDP are additional
profiles, not substitutes for the FSDP2 performance baseline.

## Source, environment and local assets

Use a separate `train,dmd2` environment; see [installation](../../docs/guides/installation.md)
for dependency setup. The complete extra includes native CLIP's `ftfy==6.3.1`.
[tianweiy/DMD2](https://github.com/tianweiy/DMD2) is pinned at
`8d8fa55633d47cfb81bbc7a892e7248f9518763f`, under **CC-BY-NC-SA-4.0**, not
framework Apache-2.0. Its five numerical source files and license stay external.
This optional helper fetches and hash-verifies source only:

```sh
python recipes/dmd2_sd15/fetch_source.py /PROJECT/dependencies/DMD2 --with-test-oracle
```

Runtime imports use a private namespace; the optional trainer file is a test
oracle, not the runtime trainer. Set `model.parameters.source_root` explicitly.
Set `model_id` to a local bundle with `unet/`, `scheduler/`, `vae/`,
`text_encoder/` and `tokenizer/`. Preserve model/dataset licenses separately.
Set `BUNDLE` to that directory and `RUN` to a separate project-local run directory.
Inventory the bundle before training:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
```

For a new Hub bundle, select its immutable revision through the [explicit
prepare/materialize workflow](../../docs/guides/model_families.md#prepare-model-assets).
The native SD loader then binds actual component files, source and library
versions; it does not resolve remote model IDs or require a separate locker.

## Prepare inputs and choose the workload

Use [inputs.template.json](inputs.template.json) for native prompt batches or
[images.template.json](images.template.json) for raw RGB:

```sh
distillctl data prepare /PROJECT/inputs.json --output /PROJECT/prepared \
  --preparer distillschedule.data.sd_inputs:prepare_inputs
```

Each record's `prompt` is a string or string list; tokens are int64 B1L. Optional
`latents: {path, sha256}` supplies exact FP32 B4HW tensors unchanged and requires
`"preencoded_vae_compatibility":"PREENCODED_LATENTS_MATCH_LOCAL_VAE"` in the
source manifest. This is a caller compatibility assertion, not an encoding step.
Raw `images: [{path, sha256}, ...]` aligns with prompts and is mutually exclusive
with `latents`. Declare exact RGB/latent geometry, `posterior: sample` or `mode`,
independent integer seed and FP32 in `vae_encoding`. Native VAE encoding uses
[-1,1] pixels and the actual `scaling_factor`, with no implicit resize, crop,
rotation or color conversion. Posterior sampling uses a separate record-ID-bound
CPU RNG. Prepared schema v2 binds every row to tokenizer/VAE/source bytes;
older unbound rows must be prepared again.

[training.json](training.json) is the single-device, full-parameter, GAN-off
starter: ten native iterations per macrocycle, G at step zero/once per ten,
guidance and both schedulers every iteration, EMA off. Independent streams
consume `generator: 1` and `guidance: 9` batches; fresh noise/prompts are drawn
each native iteration. Guidance uses detached images cached **before** the G
commit, although the G optimizer commits first.

[training.gan.json](training.gan.json) adds the original 1280-channel,
52,437,761-parameter classifier and `real: 10` native latent/prompt batches.
Its classifier loss weights and diffusion-GAN options are explicit template
choices. Prepare the real stream from RGB or pinned compatible latents.
SD1.5 trains full Generator parameters; upstream Generator LoRA is
[SDXL-only](../dmd2_sdxl/README.md).

## Train, save and resume

Fill the chosen recipe's source, bundle, prepared streams and output paths.
Single-device recipes use `trainer.device: cpu` or an explicit owned `cuda:N`
and FP32 reference execution. FSDP2 uses FP32 masters/Adam, BF16 gradient compute
and Teacher, FP16 VAE, FP32 text/classifier heads. See the [FSDP2 launch,
collective export and same-topology recovery guide](../../docs/guides/dmd2_fsdp2_reference.md).
The [optimized guide](../../docs/guides/official_optimized_training.md) supplies
its Resident/DDP launch policy, mechanisms and unchanged performance results.

```sh
# Single-device example; use the profile's distributed launcher for FSDP2/DDP.
distillctl train /PROJECT/train.json --cycles 1 --output /PROJECT/partial
distillctl train /PROJECT/train.json --cycles 3 \
  --resume /PROJECT/partial/checkpoints/cycle-000001 --output /PROJECT/resumed
distillctl export /PROJECT/train.json \
  --checkpoint /PROJECT/resumed/checkpoints/cycle-000003 --output /PROJECT/export
```

Use checkpoint paths returned by `train`; `checkpoint_every` controls saves.
See the [shared lifecycle](../../docs/guides/training.md) for complete-cycle state.

<a id="cpu-starter-and-evidence-boundary"></a>
## Small native starter

```sh
python recipes/dmd2_sd15/create_cpu_fixture.py /PROJECT/random-sd15 \
  --source-root /PROJECT/dependencies/DMD2
distillctl train /PROJECT/random-sd15/training.json
```

The helper builds random tiny actual Diffusers UNet/CLIP/VAE assets and local
tokenizers; it does not download pretrained weights. The same single-device
entry accepts explicit CUDA placement. Source-sensitive numerical checks and
hardware-specific tests are described in [validation](../../docs/guides/performance_validation.md).

## Optional native UNet checkpoint policy

The common executor can control native Generator/Fake UNet checkpointing;
the optimizer-containing macrocall itself remains atomic and is not replayed.
For the qualified CPU1 FP32 Torch2.10/Diffusers0.37.1/Transformers4.51.3/PEFT0.17.1
profile, set `execution` explicitly:

```json
{"adapter":"program","mode":"manual","implementation":{
  "backend":"torch-callable-resident-v1","share_immutable":false,"checkpoint_bindings":[],
  "owner_execution":{"native_cycle":{"gradient_checkpointing":true}}}}
```

Omit `owner_execution` to retain the model's declared baseline. Explicit `false`
temporarily disables checkpointing. Flags/callbacks are restored after each
macrocycle, including exceptions; Teacher/CLIP/VAE are outside the option.
The same source-bound option participates in measured `auto`, saved `plan`
and paired-reference admission after restart. [Budgeted resident
execution](../../docs/guides/resident_owner_training.md) documents construction,
snapshot/restore and device admission rather than whole-owner CUDA copies.

Additional independent profiles are [component checkpointing](training.component-checkpoint.json)
(Generator checkpointed, Fake epsilon activations retained) and
[native FP32 role-DDP](training.ddp-fp32.json). Their [owner guide](../../docs/guides/native_owner_backends.md)
defines per-role effects and restart behavior.

## Generate and reuse Generator weights

Export contains **current G + CLIP + VAE**, native configs and tokenizer/VAE
hashes, without Teacher/Fake/Adam/EMA. Use byte-identical tokenizer files at any
local path to prepare new prompts with this recipe's SD preparer; raw/preencoded
image inputs additionally require matching VAE bytes. The common export-only
input preparer is not used for SD. Follow the [inference-request builder](../../docs/guides/training.md#generate-and-export)
with the completed export and new prepared manifest:

```sh
distillctl generate /PROJECT/inference.json --weights /PROJECT/export \
  --prepared /PROJECT/new-prepared/manifest.json --output /PROJECT/generated --output-kind image
```

Inference uses the export's exact factory/parameters and external source root,
not the training `sd15` factory; old base weights/checkpoints are not needed.
`--output-kind latents` selects latent diagnostics.

### Official generator-weight mapping

The hash-verified export's `weights.pt` maps `generator` to the unmodified native
`feedforward_model.state_dict()`, `text_encoder` to CLIP and `vae` to the VAE.
To initialize a native generator checkpoint without merging LoRA or substituting EMA:

```python
from pathlib import Path
import torch
from distillschedule.training.inference import read_export

root, manifest = read_export('/PROJECT/export')
weights = torch.load(root / 'weights.pt', map_location='cpu', weights_only=True)
torch.save(weights['generator'], Path('/PROJECT/native_generator.pt'))
```

Set `model.parameters.generator_checkpoint` to that file for strict native
weight initialization. SDXL LoRA requires matching native LoRA settings. This
starts new optimizer/scheduler/clocks; only `train --resume` restores the full
training trajectory, including Fake/RNG/data positions.
