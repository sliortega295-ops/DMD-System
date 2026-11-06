# Wan2.2 TI2V-5B: text-to-video flow DMD

Train native `WanTransformer3DModel` Student/Fake LoRA with frozen Teacher,
`AutoencoderKLWan`, UMT5 and T5 tokenizer. This is the T2V subpath with common
No-GAN flow DMD: four Euler steps `[1,.75,.5,.25,0]`, G1/F5, local B1 and CFG=1.
FP32 is the default; `model.parameters.compute_precision: "bf16-mixed"` selects
native Transformer autocast while master weights, Adam/EMA, codecs and
flow/sampling math remain FP32. The setting is checkpoint/export-bound.

## Environment and assets

Use the isolated `train,modern-flow` environment in [shared environment](../flow_image/environment.json).
[source.json](source.json) pins `Wan-AI/Wan2.2-TI2V-5B-Diffusers` and its
Apache-2.0 model/config attribution; preserve tokenizer, ftfy and sentencepiece
notices. Training uses explicit local files. Provide `transformer/`, `vae/`,
`text_encoder/` and `tokenizer/` in `BUNDLE`; set `RUN` to a separate project-local
run directory:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
python recipes/wan22_dmd/lock_local_assets.py "$BUNDLE" \
  --cpu-max-parameters YOUR_PER_COMPONENT_LIMIT
```

For Hub files, use the revision in `source.json` with the [explicit pinned
prepare/materialize workflow](../../docs/guides/model_families.md#prepare-model-assets),
verify it, then run the locker. Its `assets.json` adds native component admission.
`--student`, `--fake`, `--teacher` select distinct same-architecture directories
inside the bundle. Add `--dtype-policy bf16_or_fp32_to_fp32` for BF16-stored
weights explicitly converted to FP32 owned state. Constructor budgets are
per-component limits, not total host/device capacity estimates.

## Prepare inputs

Choose the [raw](inputs.raw.template.json), [prompt](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) template. RGB directories use
Pillow; video containers also need `ffmpeg`/`ffprobe`. Preserve source geometry:
`Fraw=(Flatent-1)*4+1`, height/width divisible by32. The spatial16 VAE and spatial2
Transformer patches operate on BCFHW latents; do not patchify RGB twice.

```sh
distillctl data prepare /ABSOLUTE/inputs.json --output /ABSOLUTE/prepared \
  --preparer distillschedule.data.wan22_inputs:prepare_inputs
```

Preparation uses native posterior mode, VAE mean/std normalization, 226-row UMT5
features, native zero padding and genuinely encoded empty-prompt conditions.
Prompt-only geometry defines a noise template; preencoded rows retain exact
contexts, masks, geometry and codec/source binding. Native velocity is
`v=epsilon-x0`; the Transformer receives `1000*sigma` expanded over patches,
with the original Euler sign.

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

Use returned checkpoint paths. The [common lifecycle](../../docs/guides/training.md)
handles complete-cycle save/restore. Reference mode is `{"mode":"reference"}`;
exact immutable-base sharing uses:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

Sharing retains disjoint Student/Fake trainables and occurs after role
construction. For equal-B1 replicas, use the shared [rank-strided data,
Gloo/NCCL launch and replicated sharing policy](../flow_image/README.md#explicit-replicated-training-not-model-sharding).
See [capacity conditions](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing).

## Generate with the current Student

Export contains current Student, VAE, UMT5 and tokenizer. Prepare new prompts
with the [export-only preparer](../../docs/guides/data.md#export_inputs), and create an
inference request with the [common builder](../../docs/guides/training.md#generate-and-export):

```sh
distillctl generate /ABSOLUTE/inference.json --weights /ABSOLUTE/export \
  --prepared /ABSOLUTE/new-prepared/manifest.json --output /ABSOLUTE/generated --output-kind video
```

Output is decoded THWC uint8 RGB plus PNG frames, independently of the original
training bundle/checkpoints. For a small random native-class starter, run
`python recipes/wan22_dmd/create_cpu_fixture.py /ABSOLUTE/NEW-fixture` in the
isolated environment.
