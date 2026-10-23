# Official DMD2 / Stable Diffusion 1.5

This recipe uses the actual `SDUniModel`, `SDGuidance`, epsilon losses and
classifier implementation from [tianweiy/DMD2](https://github.com/tianweiy/DMD2)
at `8d8fa55633d47cfb81bbc7a892e7248f9518763f`. It does not use the framework's
toy DMD example, Helios/Wan mathematics, or a shape-compatible substitute.

## Source and local assets

Upstream DMD2 is CC-BY-NC-SA-4.0, **not Apache-2.0**. Its five complete numerical
source files and original license remain outside the package. Review that
license before use. This explicit helper fetches pinned source only:

```sh
python recipes/dmd2_sd15/fetch_source.py /PROJECT/dependencies/DMD2 --with-test-oracle
```

It verifies each complete file's SHA256, preserves the upstream bytes, and
never fetches weights. Runtime imports rewrite only upstream import roots into
a private namespace. Only the exact original `get_x0_from_noise`, `NoOpContext`
and `DummyNetwork` definitions are extracted from `main/utils.py`, avoiding its
unrelated plotting/LMDB imports. The optional trainer file is test-only oracle
source; it is not part of the runtime numerical closure.

Set `source_root` explicitly. Set `model_id` to your **local** pretrained
SD1.5 bundle containing `unet/`, `scheduler/`, `vae/`, `text_encoder/` and
`tokenizer/`. No model IDs are resolved remotely by this adapter. Assets,
source hashes, settings and actual model-library versions enter restart
identity. Model/dataset licenses remain separate.

## Input preparation and training

`inputs.template.json` contains new prompt records, not embeddings copied from
training. Use the same public data entrypoint:

```sh
distillctl data prepare inputs.json --output /PROJECT/prepared \
  --preparer distillschedule.data.sd_inputs:prepare_inputs
```

Each record is a native batch: `prompt`
may be one string or a list of strings. Tokens remain int64 B1L. Optional
`latents: {path, sha256}` loads an existing FP32 B4HW tensor unchanged; this
adapter never secretly encodes, resizes or renormalizes image latents.
Such supplied latents require the source manifest's explicit
`"preencoded_vae_compatibility": "PREENCODED_LATENTS_MATCH_LOCAL_VAE"`
assertion: the caller declares compatibility with this local bundle's actual
VAE weights/configuration/scaling. The preparer pins those bytes but **does
not claim to have encoded or independently verified the supplied latents**.

For raw local images, use `images.template.json` with the same preparer. Its
explicit `vae_encoding` declares `[height, width]`, `[channels, latent_height,
latent_width]`, `posterior: sample` or `mode`, independent integer `seed`, and
`dtype: float32`. Each record's `images: [{path, sha256}, ...]` aligns exactly
with its prompt batch; `images` and `latents` are mutually exclusive. Only
single-frame eight-bit RGB files at the **exact** declared size are admitted:
no resizing, cropping, EXIF rotation or RGB conversion is implicit. Pixels
normalize to [-1,1], then the actual local `AutoencoderKL` posterior is scaled
by its actual `scaling_factor`. Declared geometry is checked against both the
native VAE configuration and actual encoded output, never an assumed scale8.
Sampling uses a separate CPU generator seeded by the explicit seed plus
unique record ID; reordering records does not change samples or consume the
training RNG. Local tokenizer/VAE weights, encoder source, image bytes and
source manifest are hashed into provenance and rechecked before publication.
Only VAE and native CLIP tokenizers are loaded; no Teacher/Fake is constructed.
SD input schema v2 requires a path-independent preprocessing binding in both
prepared metadata and every tensor row. Training admission checks every stream;
the native owner checks again before any update. Checkpoint evaluation and
inference-only generation check each input row before any forward/RNG draw.
Same-shaped different CLIP vocabulary mappings or VAE weights/configuration
are incompatible, not silently accepted. Old prepared v1 rows without these
bindings must be prepared again; supplied latent tensors remain bitwise unchanged.
Raw-image encoding and caller-declared preencoded compatibility have distinct
`latent_origin` labels, never a fabricated encoder receipt.

Fill the explicit paths in `training.json`, then use the existing common
`distillctl train`, `export` and `generate` lifecycle. A cycle performs ten
native iterations: G at step zero then once per ten steps, guidance every
iteration, both LR schedulers every iteration, and no EMA. Independent
`generator` (1) and `guidance` (9) records are consumed per cycle. Noise and
prompts are fresh every native iteration. Guidance uses the detached image
cached **before** the G commit, even though the G optimizer commits first.

Full-GAN is the actual upstream optional head: set `cls_on_clean_image`, supply
`real` stream with ten native latent/prompt records per cycle, and explicitly
choose `gen_cls_loss`, `gen_cls_loss_weight`, `guidance_cls_loss_weight` and
optional diffusion-GAN settings. The 1280-channel, 52,437,761-parameter SD1.5
head is not resized to fit a tiny fixture. The default recipe leaves GAN off;
it is not a reproduction claim for a paper's complete training configuration.
`training.gan.json` is a concrete GAN-enabled template with independent
`generator: 1`, `guidance: 9`, `real: 10` native streams and explicit nonzero
generator/guidance classifier weights. These example weights are declared
choices, not a claim to reproduce a paper configuration. Prepare the `real`
stream from `images.template.json` or pinned latents, and replace all paths.
The separate 1280-channel GAN-head forward/backward test is **not** full-GAN
training acceptance; full-GAN/pretrained quality and GPU execution are NOT_RUN.

## CPU starter and evidence boundary

```sh
python recipes/dmd2_sd15/create_cpu_fixture.py /PROJECT/random-sd15 \
  --source-root /PROJECT/dependencies/DMD2
distillctl train /PROJECT/random-sd15/training.json
DMD2_SOURCE_ROOT=/PROJECT/dependencies/DMD2 pytest -q tests/test_training_dmd2*.py
```

The helper constructs local **random tiny real Diffusers UNet/CLIP/VAE** assets
and tokenizers. It does not download a pretrained model and does not establish
image quality, training convergence, throughput, or production parity.

CPU1 FP32 is the validated integration boundary. A single-device resident
CUDA FP32 **reference source path** is also implemented: explicitly select
`trainer.device: "cuda:0"` (or the owned CUDA index) and `execution.mode:
"reference"`. It places all native modules, scheduler/conditioning tensors,
inputs and restored Adam moments on that device; RNG and noncapturable Adam
step metadata stay on the host. Current-G generation and portable inference
reload use their explicitly selected CPU/CUDA placement. This is **not CUDA
validation**: GPU execution, capacity, pretrained quality and performance
remain NOT_RUN; AMP and distributed/FSDP ownership remain rejected.
The common backend consumes the owner's live
`runtime_tensors()` placement seam, excluding host RNG/Adam-step metadata and
including native plain conditioning/scheduler tensors.

CUDA auto, saved-plan and `owner_execution` admission are explicitly rejected,
as are whole-owner CUDA deep copies. The existing paired trial path cannot yet
snapshot/restore real resident roles within a capacity budget covering frozen
Teacher/CLIP/VAE, optimizer, comparison and CPU staging. CPU trial behavior is
unchanged; no unsupported CUDA request silently falls back to it. Optional
`tests/gpu/test_training_dmd2_lifecycle.py` uses the existing random tiny actual
fixtures for native-source parity, host-checkpoint restore, new-process resume
and current-G export/reload. It skips on CPU-only Torch and is not pretrained
or full-GAN GPU evidence.

Before update or checkpoint save, live mathematical settings, initially
frozen Teacher/CLIP/VAE/base-LoRA tensors and native conditioning/scheduler
tables must match their initial fingerprints. Upstream's post-first-guidance
`requires_grad=True` behavior is preserved; the Teacher still never enters
the optimizer. Mutable G/Fake weights are not incorrectly treated as frozen.
Install the complete `dmd2` extra: it explicitly pins `ftfy==6.3.1`, used in
the qualified native CLIP preprocessing. Installing Transformers alone may
silently select its different BasicTokenizer fallback; that is not the
documented reference environment.

## Optional native UNet checkpoint policy

The common capability inventory can measure the original DMD2 generator and
fake-UNet checkpoint option. The optimizer-containing `external_grad` macrocall
remains atomic and **not replay safe**. This is not ordinary call checkpointing:

```json
{
  "adapter": "program",
  "mode": "manual",
  "implementation": {
    "backend": "torch-callable-resident-v1",
    "share_immutable": false,
    "checkpoint_bindings": [],
    "owner_execution": {"native_cycle": {"gradient_checkpointing": true}}
  }
}
```

Omit `owner_execution` to keep the model's declared baseline unchanged. An
explicit `false` can temporarily disable a declared checkpoint-enabled baseline.
Neither option rewrites model settings, identity, native RNG, masks, modes or
optimizer ownership. Both original model/guidance flags and all affected UNet
checkpoint flags/callbacks are restored after each complete macrocycle, also on
exceptions. Teacher, CLIP and VAE are never checkpointed by this option.

`auto` uses the same bounded candidate budget, measured calibration and shared
solver; `plan` reexecutes the same source-bound measured option. A lower tracked
storage budget may select checkpointing even when its measured CPU time is
longer. Enabled options require observed generator **and** fake checkpoint
requests and real callable recomputations on every run; explicit disabled
options require verified disabled flags and zero requests. Unexercised options
are pruned during calibration and rejected for manual/saved-plan execution.

New manual owner policies receive full preupdate reference admission on
independent clones, including model/Adam/schedulers/native clocks/RNG and
outputs. Only the selected clone is adopted. The first continued optimized
update after process restart is similarly re-admitted; ordinary steady-state
cycles do not execute duplicate reference updates. Provider descriptors, owner
class code, native sources and helper-file hashes bind plans and restart policy.
Malformed, changed or incompatible selections fail closed.

This capability is limited to actual checkpoint-capable native UNets on the
verified Torch 2.10.0 / Diffusers 0.37.1 / Transformers 4.51.3 / PEFT 0.17.1
CPU1/FP32 closure. Tiny SD1.5/SDXL tests validate native effect and restart
semantics; CPU tracked-storage and timing are diagnostics, **not** GPU memory,
throughput, pretrained quality, AMP/FSDP or production-validation claims.
An otherwise admitted reference on a different numerical-library stack keeps
its declared baseline: its descriptor marks this optional capability unavailable
and offers no owner override. This does not silently qualify checkpointing on
that stack. Runtime drift relative to an already constructed owner still fails.

Export projects **current G + CLIP + VAE only**, with architecture configs and
named module weights. `distillschedule.training.sd:build_inference_model`
does not construct a teacher/fake UNet or optimizer and never calls
`from_pretrained`. It can generate on new prepared prompts after deleting the
original base bundle, restart checkpoints, and training inputs. The explicit
external source root remains necessary. Select `--output-kind image` to write
the native VAE decode as an RGB image; `latents` remains available.
Exports retain tokenizer/VAE file hashes, not their old asset paths. New prompt
preparation needs byte-identical local `tokenizer/` (and SDXL `tokenizer_2/`)
files, which may be copied to a different directory independently of the old
base weights. Tokenizer-only input preparation does not load or require a VAE.
New raw/preencoded image inputs additionally bind the matching VAE bytes.

### Official generator-weight mapping

The common export's `weights.pt` is a named mapping: `generator` contains the
**unmodified native `feedforward_model.state_dict()`**, `text_encoder` contains
native CLIP (dual CLIP on SDXL), and `vae` contains the native VAE. There is no
framework prefix renaming, dtype conversion, EMA substitution or LoRA merge.
To initialize an upstream-compatible generator checkpoint, extract this exact
mapping from a completed, hash-verified export:

```python
from pathlib import Path
import torch
from distillschedule.training.inference import read_export

root, manifest = read_export('/PROJECT/export')
weights = torch.load(root / 'weights.pt', map_location='cpu', weights_only=True)
torch.save(weights['generator'], Path('/PROJECT/native_generator.pt'))
```

Set `model.parameters.generator_checkpoint` to that file to initialize the
same native architecture through `distillctl train`. SDXL LoRA needs the same
native LoRA settings; its separate base/A/B tensors remain separate. Strict
key and shape checking rejects incompatible models. This mapping is tested
with real updated SD1.5, SDXL and SDXL-LoRA exports and new native owners.
It imports **weights only**, starts new optimizer/scheduler/native clocks and
does not resume Fake Score, RNG or data positions. Only a complete framework
checkpoint supplied to `train --resume` restores the training trajectory.
