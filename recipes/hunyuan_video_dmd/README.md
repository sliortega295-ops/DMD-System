# HunyuanVideo-1.5: text-to-video flow DMD

Train native `HunyuanVideo15Transformer3DModel` Student/Fake LoRA with a frozen
Teacher, temporal4 `AutoencoderKLHunyuanVideo15`, Qwen2.5 primary text and T5/ByT5
glyph conditioning. Common No-GAN flow DMD uses four Euler steps
`[1,.75,.5,.25,0]`, G1/F5, local B1 and CFG=1. Compute is FP32 by default;
`model.parameters.compute_precision: "bf16-mixed"` enables native Transformer
autocast while master weights, Adam/EMA, codecs and flow/sampling math stay FP32.
Checkpoints and exports retain this setting.

## Environment, source and assets

Use the isolated `train,modern-flow` environment in [environment.json](environment.json).
[source.json](source.json) pins
[HunyuanVideo-1.5-Diffusers-480p_t2v at 286be7c](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/tree/286be7ce72277246578a3e3cc2487e95ddae5bcf).
Model assets retain the **Tencent Hunyuan Community License** and their
text/tokenizer terms, separately from Apache-licensed framework/dependencies.
Provide native `transformer/`, `vae/`, `text_encoder/`, `text_encoder_2/`,
`tokenizer/` and `tokenizer_2/` in `BUNDLE`; set `RUN` to a separate project-local
run directory:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
python recipes/hunyuan_video_dmd/lock_local_assets.py "$BUNDLE" \
  --cpu-max-parameters YOUR_PER_COMPONENT_LIMIT
```

For Hub files, use the pinned revision with the [explicit prepare/materialize
workflow](../../docs/guides/model_families.md#prepare-model-assets), verify it,
then run the locker. Its `assets.json` retains native configs, tied T5 ownership
and both tokenizers' options. `--student`, `--fake`, `--teacher` select distinct
same-architecture directories within the bundle. Add
`--dtype-policy bf16_or_fp32_to_fp32` for BF16 storage explicitly converted to
FP32 owned state. Constructor budgets bound each component, not total capacity.

## Prepare native inputs

Choose the [raw](inputs.raw.template.json), [prompt](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) template. RGB directories need
Pillow; video containers also need `ffmpeg`/`ffprobe`. Preserve source geometry:
`Fraw=(Flatent-1)*4+1`, height/width divisible by the actual VAE spatial ratio.
Prompt geometry uses VAE output channels, not concatenated Transformer width.

```sh
distillctl data prepare /ABSOLUTE/inputs.json --output /ABSOLUTE/prepared \
  --preparer distillschedule.data.hunyuan_video_inputs:prepare_inputs
```

VAE posterior mode is multiplied by `scaling_factor`, inverted exactly on decode.
Primary text uses native Qwen chat/crop108/max1000; glyph text uses native quoted
text extraction and slow ByT5/max256. No quoted glyphs means native zero features
with an all-false glyph mask. Empty-prompt features come from the actual encoders;
IDs, source hashes and codec/conditioning bindings remain attached to inputs.
Native T2V uses zero image channels/mask/semantic tokens, dual text and
`1000*sigma`; velocity is `v=epsilon-x0` with the original Euler sign.

## Train, resume and export

Copy [training.lora.json](training.lora.json), fill asset/prepared/output paths,
and set `trainer.device` to `cpu` or the owned `cuda:N`:

```sh
distillctl train /ABSOLUTE/train.json --cycles 2 --output /ABSOLUTE/run
distillctl train /ABSOLUTE/train.json --cycles 3 \
  --resume /ABSOLUTE/COMPLETE_CHECKPOINT --output /ABSOLUTE/resumed
distillctl export /ABSOLUTE/train.json --checkpoint /ABSOLUTE/FINAL_CHECKPOINT \
  --output /ABSOLUTE/export
```

Use returned checkpoint paths and the [common lifecycle](../../docs/guides/training.md).
Reference mode is `{"mode":"reference"}`. For exact immutable-base sharing:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

Sharing leaves Student/Fake LoRA independent and occurs after construction.
Equal-B1 replication uses the shared [rank-strided data, Gloo/NCCL launch and
replicated sharing policy](../flow_image/README.md#explicit-replicated-training-not-model-sharding).
See [capacity conditions](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing).

## Generate new prompts

Export contains current Student, VAE, both text encoders and tokenizers. Prepare
new prompts with the [export-only preparer](../../docs/guides/export_inputs.md),
then create an inference request with the [common builder](../../docs/guides/training.md#generate-and-export):

```sh
distillctl generate /ABSOLUTE/inference.json --weights /ABSOLUTE/export \
  --prepared /ABSOLUTE/new-prepared/manifest.json --output /ABSOLUTE/generated --output-kind video
```

Output is decoded THWC uint8 RGB plus PNG frames, including native quoted-glyph
conditioning, without training assets/checkpoints. A small random native-class
starter is available through
`python recipes/hunyuan_video_dmd/create_cpu_fixture.py /ABSOLUTE/NEW-fixture`.
