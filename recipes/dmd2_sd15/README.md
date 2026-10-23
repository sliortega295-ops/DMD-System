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

CPU1 FP32 is the validated integration boundary. CUDA/AMP, distributed/FSDP and
GPU scheduling are NOT_RUN. Current generic stateful tensor traversal mixes
CUDA model state with host RNG/Adam step metadata, so this adapter rejects
CUDA rather than pretending it works. `DMD2Owner.runtime_tensors()` exposes a
narrow future placement seam; it is not yet common-backend wiring.

Export projects **current G + CLIP + VAE only**, with architecture configs and
named module weights. `distillschedule.training.sd:build_inference_model`
does not construct a teacher/fake UNet or optimizer and never calls
`from_pretrained`. It can generate on new prepared prompts after deleting the
original base bundle, restart checkpoints, and training inputs. The explicit
external source root remains necessary. Select `--output-kind image` to write
the native VAE decode as an RGB image; `latents` remains available.
