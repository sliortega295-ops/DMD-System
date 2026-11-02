# HunyuanVideo-1.5: text-to-video flow DMD

This recipe uses the native `HunyuanVideo15Transformer3DModel`, temporal4
`AutoencoderKLHunyuanVideo15`, Qwen2.5 primary text encoder and T5/ByT5 glyph
conditioning through the common DMD-System lifecycle. Student, Teacher and Fake
Score use the same model family, unmerged LoRA, four Euler steps, G1/F5, local
batch one and CFG=1.

## Environment, source and license

Use the separate `modern-flow` dependency group and [environment.json](environment.json).
The pinned model/config source is
[HunyuanVideo-1.5-Diffusers-480p_t2v at 286be7c](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/tree/286be7ce72277246578a3e3cc2487e95ddae5bcf).
[source.json](source.json) records its exact identity. Model assets retain the
**Tencent Hunyuan Community License** and their text/tokenizer terms; these are
separate from Apache-licensed framework/dependency implementations. No pretrained
weights are bundled or relicensed.

## Local assets and inputs

Provide native `transformer/`, `vae/`, `text_encoder/`, `text_encoder_2/`,
`tokenizer/` and `tokenizer_2/` directories in one bundle. Inventory them with:

```sh
python recipes/hunyuan_video_dmd/lock_local_assets.py /ABSOLUTE/local-hunyuan15 \
  --cpu-max-parameters YOUR_PER_COMPONENT_LIMIT
```

This bounds all role/codec constructors before weight loading and records safe
shards, hashes and shapes. Distinct Student/Fake/Teacher subdirectories are
selectable with `--student`, `--fake` and `--teacher`. For BF16-stored checkpoints,
add `--dtype-policy bf16_or_fp32_to_fp32` to convert weights to FP32 owned state.
Native config bytes, T5 tied ownership and both tokenizers' options are retained.

Choose the [raw](inputs.raw.template.json), [prompt](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) template. RGB directories require
Pillow; video containers also require `ffmpeg` and `ffprobe`. Keep source geometry:
`Fraw=(Flatent-1)*4+1`, with height/width divisible by the actual VAE spatial ratio.
The VAE uses its posterior mode times `scaling_factor`, with the exact inverse at
decode. Prompt geometry uses output/VAE channels, not the concatenated Transformer
input width. Preparation retains source hashes, IDs and codec/conditioning binding.

Primary text uses the native Qwen chat template, crop108 and max1000 features.
Glyph text uses the native quoted-text extraction and slow ByT5 tokenizer,
max256. With no quoted glyphs, native zero features plus an all-false glyph mask
are valid. Empty-prompt conditions come from the actual encoders.

```sh
distillctl data prepare /ABSOLUTE/inputs.json --output /ABSOLUTE/prepared \
  --preparer distillschedule.data.hunyuan_video_inputs:prepare_inputs
```

## Train, resume and export

Copy [training.lora.json](training.lora.json) and fill the asset, prepared-data
and output paths. Set `trainer.device` to `cpu` or the owned `cuda:N`. Native
compute defaults to FP32; add `"compute_precision":"bf16-mixed"` to
`model.parameters` for BF16 Transformer autocast. Master parameters, Adam/EMA,
codecs and flow/sampling math stay FP32; checkpoints, saved plans and portable
exports retain the compute setting.

```sh
distillctl train /ABSOLUTE/train.json --cycles 2 --output /ABSOLUTE/run
distillctl train /ABSOLUTE/train.json --cycles 3 \
  --resume /ABSOLUTE/COMPLETE_CHECKPOINT --output /ABSOLUTE/resumed
distillctl export /ABSOLUTE/train.json --checkpoint /ABSOLUTE/FINAL_CHECKPOINT \
  --output /ABSOLUTE/export
```

Use the checkpoint path returned by `train`. Recovery retains all role parameters,
Adam, schedulers, EMA, RNG, sample position and native update clocks. The velocity
is `v=epsilon-x0`; native T2V receives the original zero image channels/mask and
semantic tokens, dual-text conditions and `1000*sigma`. The Euler sign is unchanged.

## Execution strategies

The reference policy is `{"mode":"reference"}`. For single-device immutable-base
sharing, use:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

Explicit equal-rank B1 replicas use `data.sharding="rank_strided"`, the actual
local device and a matching `trainer.distributed.backend` (`gloo` on CPU or
`nccl` on CUDA). Reference mode performs post-backward gradient averaging before
clipping/Adam. The replicated sharing policy is:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-replicated-v1","reduction":"post_backward","bucket_cap_mb":0,"local":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}}
```

Each rank retains independent records and RNG. Sharing applies to immutable
bases after construction, not mutable LoRA parameters. The actual policy and
capacity conditions are in [execution capabilities](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing).
The CPU tiny fixture also exercises calibration and saved-plan execution through
this same lifecycle.

## New-prompt output

Export projects the current Student, VAE, both text encoders and tokenizers.
Use the returned factory/parameters in an inference config, prepare a new prompt
with the [export-only preparer](../../docs/guides/export_inputs.md), then generate:

```sh
distillctl generate /ABSOLUTE/inference.json --weights /ABSOLUTE/export \
  --prepared /ABSOLUTE/new-prepared/manifest.json --output /ABSOLUTE/generated --output-kind video
```

Output is decoded THWC uint8 RGB plus a PNG frame sequence. CPU portable inference
uses the exported current Student/codecs without the original training assets or
checkpoints, including native quoted-glyph conditioning.

## Tests and provenance

```sh
DMD_MODERN_FLOW_CPU=1 python recipes/hunyuan_video_dmd/create_cpu_fixture.py /ABSOLUTE/NEW-fixture
DMD_MODERN_FLOW_CPU=1 python -m pytest -q tests/test_training_hunyuan_video_cpu.py tests/test_training_flow_precision.py
DMD_MODERN_FLOW_CPU=1 DMD_NATIVE_FLOW_REPLICATED=1 python -m pytest -q tests/test_training_flow_video_replicated.py
```

Tests use actual upstream small configs and compare native text/codec/velocity,
real updates, reference/sharing state, saved-plan execution, fresh-process recovery
and current-Student output. Gloo2 uses an independent sequential B1 gradient-mean
oracle. Historical native FP32 CUDA acceptance is at
`artifacts/framework/20261024/native_video_cuda_acceptance.json`; CPU fixture
results remain distinct from pretrained-model GPU measurements.
