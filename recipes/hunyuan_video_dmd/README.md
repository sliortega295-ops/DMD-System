# HunyuanVideo-1.5 T2V: bounded native CPU flow-DMD reference

**Not pretrained/GPU/quality acceptance.** The admitted slice is CPU1 FP32,
batch1, same-family unmerged LoRA, four Euler steps, G1/F5 and CFG=1, through
the existing common `training.dmd` Program, checkpoint and export interfaces.
This is not official DMD2, Self-Forcing, I2V, meanflow or a distilled variant.
No model-name branch was added to the trainer, planner or controller.

## Source and license

The source-only candidate is
[HunyuanVideo-1.5-Diffusers-480p_t2v at 286be7c](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/tree/286be7ce72277246578a3e3cc2487e95ddae5bcf).
Its model assets use **Tencent Hunyuan Community terms**, repository license
tag `other`, **not Apache-2.0**. Review the original terms and all text/tokenizer
and data obligations before use. Framework/dependency implementation licenses
do not grant asset redistribution. Full upstream configs/weights are not
vendored here; small constructor choices are source-derived test fixtures.
[source.json](source.json) retains hashes and the stale upstream example ID's
401 audit failure; it is not a reason to retry/authenticate another account.

Actual imports: Diffusers0.37.1 `HunyuanVideo15Transformer3DModel`,
`AutoencoderKLHunyuanVideo15`, `HunyuanVideo15Pipeline`; Transformers5.14.1
`Qwen2_5_VLTextModel`, `T5EncoderModel`, `Qwen2TokenizerFast`, **slow**
`ByT5Tokenizer`. This uses the separate [modern-flow environment](environment.json),
not a Transformers4/PEFT0.17 environment. Use Python3.12 and install the
`modern-flow`, `train`, `test` extras only in a new isolated environment; never
upgrade a historical experiment lane in place. Construction installs nothing.

## Already-local assets only

Provide complete native safe-serialized directories `transformer`, `vae`,
`text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2` in one local bundle.
Separate Student/Fake/Teacher native directories may be selected explicitly.
Inventory creation does not download, load or authenticate pretrained weights:

```sh
python recipes/hunyuan_video_dmd/lock_local_assets.py /ABSOLUTE/local-hunyuan15 \
  --cpu-max-parameters EXPLICIT_PER_COMPONENT_PARAMETER_LIMIT
```

The explicit parameter limit is mandatory. Runtime sizes **all three roles
and every VAE/text encoder on meta before any concrete model/weight load**;
portable reconstruction repeats that gate. This is not full-model capacity
qualification: large native CPU layouts and pretrained execution are NOT_RUN.
Strict F32 storage is default; the explicit `--dtype-policy
bf16_or_fp32_to_fp32` admits only exact BF16/F32-storage to FP32-compute
conversion, not BF16 runtime/AMP/GPU qualification. FP16, missing/contradictory
tied aliases, missing shards, unrecorded files, unknown configs, unsafe paths
and hash/source/version drift reject. Production never substitutes random
initialization. Raw legacy Qwen RoPE config bytes and native T5 tied ownership
are retained, not normalized back through a lossy serialized config.

## Native math, conditions and data

The common contract is `v=epsilon-x0`,
`z=(1-sigma)*x0+sigma*epsilon`, `x0=z-sigma*v`, with no sign change.
The actual Transformer receives `[z, zeros_like(z), one-channel zero mask]`,
`timestep=sigma*1000` and legitimate native zero image semantic embeddings
`[1,729,image_embed_dim]`. No reference image, extra condition, forced attention
backend, causal cache or inference-only hook is invented.

Primary conditioning calls the native `_get_mllm_prompt_embeds` with the native
system/chat template, crop108, hidden layer-3, max1000. Its mask **must** be
nonempty, including the actual empty-prompt encoding. Secondary conditioning
calls native `_get_byt5_prompt_embeds`, including native quoted-glyph extraction,
max256. With no quoted text, native **zeros + allfalse mask** is a valid
sentinel. Glyph features are never fabricated; Unicode quotes/bytes execute
the real encoder. Both native tokenizers, options, added-token state and source
identities travel with export. Overflow rejects rather than silently truncates.

Copy one [raw](inputs.raw.template.json), [prompts](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) input manifest and replace
every placeholder. Raw RGB videos/frame directories must have
`Fraw=(Flatent-1)*4+1`, H/W divisible by the actual VAE spatial compression;
no crop/resize/frame truncation/resampling. The native VAE encodes its posterior
mode times `scaling_factor` and decodes the exact inverse. Tiny CPU fixture:
`[1,3,5,16,16] -> [1,4,2,4,4] -> RGB5`, spatial4/**temporal4**, not temporal1.
Prompt geometry uses **output/VAE** channels, never the concatenated native
Transformer input width. The template's 32 channels refer to the public source
config, not a tested full pretrained run; use 4 for the tiny fixture.

```sh
distillctl data prepare /ABSOLUTE/inputs.json --output /ABSOLUTE/prepared \
  --preparer distillschedule.data.hunyuan_video_inputs:prepare_inputs
```

Prepared records preserve stable IDs, media/file SHA256s and native source,
text/tokenizer/VAE/dtype binding. Preencoded `.pt` intake is weights-only and
requires all real primary/glyph and genuine empty-prompt fields; invented or
stale features reject. Raw, prompts and preencoded share the same validator.

## Training, recovery, sharing and portable output

Copy [training.lora.json](training.lora.json), set assets/prepared/output paths,
then use the same public lifecycle as existing models:

```sh
distillctl train /ABSOLUTE/train.json --cycles 1 --output /ABSOLUTE/first
distillctl train /ABSOLUTE/train.json --cycles 3 --resume /ABSOLUTE/COMPLETE_CHECKPOINT_FROM_FIRST \
  --output /ABSOLUTE/resumed
distillctl export /ABSOLUTE/train.json --checkpoint /ABSOLUTE/FINAL_COMPLETE_CHECKPOINT \
  --output /ABSOLUTE/current-student-export
```

Use the exact checkpoint path returned by `train`. Complete boundaries preserve real Adam,
schedulers, EMA, RNG, data position/IDs, clocks and role versions. Teacher and
all codecs remain frozen; Student/Fake own distinct trainable LoRA parameters.
To execute the existing immutable-sharing optimization, explicitly replace
`execution` with the following; do not share mutable LoRA parameters:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

The native CPU tests actually execute shared and unshared cycles and compare
all gradients/Adam/schedulers/EMA/outputs, measure actual shared aliases and
bytes, resume sharing in a fresh process and reject immutable mutation.

Prepare a **new** prompt manifest with a novel ID/prompt/geometry before
removing training assets. Export projects current Student + all native codecs
and both tokenizers, not EMA/Teacher/Fake/base-reload fallback. The returned
export factory/parameters form the inference model configuration; then:

```sh
distillctl generate /ABSOLUTE/inference.json --weights /ABSOLUTE/current-student-export \
  --prepared /ABSOLUTE/new-prepared/manifest.json --output /ABSOLUTE/generated --output-kind video
```

Portable generation requires neither original assets nor training checkpoints;
it outputs decoded THWC uint8 RGB plus frame files. The current data preparer
still requires the original explicit assets for preparing new inputs. There is
no export-only public preparer in this unit, and no such workflow is claimed.

## Reproduce the synthetic CPU proofs

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DMD_MODERN_FLOW_CPU=1
python recipes/hunyuan_video_dmd/create_cpu_fixture.py /ABSOLUTE/NEW_TEST_ONLY_DIRECTORY
python -m pytest -q tests/test_training_hunyuan_video_cpu.py
```

The explicitly random actual-class fixture is **SYNTHETIC_TEST_ONLY**. Native
forward/gradient/text/codec/Euler oracles, exact three-cycle fresh-process
recovery, real sharing and portable novel-input decoding are bounded interface
and lifecycle evidence, not long training or convergence. Full pretrained CPU,
GPU/CUDA/distributed/AMP, full-model capacity, production video geometries,
performance and quality gates V3–V5 remain **NOT_RUN**. This does not complete
the mature-framework plan or full U15 acceptance.
