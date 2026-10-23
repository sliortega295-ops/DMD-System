# LTX-2.5: native AV + reusable No-GAN flow DMD

**Implemented boundary:** real upstream tiny native architectures, exact local
FP32 state dictionaries, raw/preencoded AV and real Gemma4 text, full or
unmerged trainable LoRA, G1/F5 training, common complete-cycle checkpoint/resume,
and current-Student native AV generation/export. **Not a pretrained 22B quality,
GPU, distributed, performance or official LTX distillation reproduction claim.**

## Source and environment

The external source is [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) at
`e98f6d927d83c92ef2be6aec76c8440b19306e39` (ltx-core 1.5 / LTX-2.5).
`training/ltx_source.py` locks 114 source/license files; the reviewed
`source-manifest.json` SHA256 is
`6e3539b8134e81a3aeb5815f46c8a6a8af147120f1b9d25eb0050635de92cf7b`.
The source stays external under **LTX-2.x Community License**, not this project's
Apache license. The adapter does not bundle, fetch or relicense weights.

Use an isolated environment, not the SD/Self-Forcing Transformers 4.51 process.
The CPU acceptance environment is pinned in `environment.json`. Source imports
use `PinnedSource(python_roots=('packages/ltx-core/src',),
import_roots=('ltx_core',))`; no global `sys.path` installation or CUDA context is
needed. Both native transformer attention paths are explicitly MATH SDPA.

```sh
# PY is the Python executable in that isolated environment.
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src
# Optional source-only fetch, no model download; skip if already present.
"$PY" recipes/ltx25/fetch_source.py "$LTX_SOURCE_ROOT"
```

## Executable random native fixture

```sh
"$PY" recipes/ltx25/create_cpu_fixture.py "$RUN/fixture" \
  --source-root "$LTX_SOURCE_ROOT" --cycles 3
"$PY" -m distillschedule train "$RUN/fixture/training.json"
# Same actual native bases, but independent trainable low-rank A/B parameters:
"$PY" -m distillschedule train "$RUN/fixture/training.lora.json"
```

The fixture writes procedural PNG image/3-frame video, stereo PCM WAV and text.
Actual native VideoEncoder/ConvVideoDecoder map `[1,3,3,8,8]` to
`[1,128,2,2,2]`; AudioEncoder/AudioDecoder + Vocoder map the 16 kHz stereo input
to `[1,8,11,16]` and decode a native 24 kHz waveform. Native Gemma4 (hidden24,
one layer), LTXGemmaTokenizer, FeatureExtractorV2 and two connectors produce
distinct video/audio contexts. The **empty prompt is genuinely encoded** by the
same owned modules, never a zero tensor. Random weights mean no quality claim.

The portable path is a strict explicit constructor-config + native-state-dict
bundle, **or an original-format safetensors transformer entry** as described
below; neither is an automatic full monolithic pretrained 22B pipeline. Every role and
codec config is written in `assets.json`, every local weight has a streaming
SHA256, every key/shape/dtype is strictly checked, and empty/meta/nonfinite or
changed checkpoints fail. CPU model bounds prevent accidental 22B allocation.

### Original LTX-2.5 safetensors transformer import

`original-role.template.json` can replace each `roles.student/fake/teacher`
entry in the local bundle. This is a distinct, genuine original-format path,
not arbitrary `.pt` state dictionaries relabeled as original checkpoints.
The file must carry native `__metadata__.config.transformer`, explicit native
geometry/behavior, `model_version=2.5`, and original
`model.diffusion_model.*` keys.

`load_original_transformer` executes the pinned native
**SafetensorsModelStateDictLoader → LTXModelConfigurator →
LTXV_MODEL_COMFY_RENAMING_MAP → SingleGPUModelBuilder(device=cpu)**. It does
not reconstruct weights with a substitute model. Native assignment is normally
`strict=False`, so this adapter first verifies the full selected key set,
shapes, precision and finite values against the original configurator's internal
meta layout, and refuses an incomplete/uninitialized/meta return. Metadata
defaults must not invent missing production geometry. The exact native weights
are compared again after builder loading; source and file hashes are rechecked.
Inference LoRA fusion is disabled and the native registry has no model/weight
cache. Both real native attention paths are then explicitly pinned to MATH SDPA.

The caller must supply an explicit CPU `cpu_max_parameters` budget.
`strict_fp32` rejects other precisions; `cast_native_float_to_fp32` explicitly
permits native FP16/BF16 and uses the original builder's FP32 cast, never an
implicit cast. Locally generated tiny **original-format** safetensors test the
entire importer and common G1/F5/current-G projection path. Real pretrained 22B
loading/quality, all monolithic Diffusion-VAE/BWE/text variants and GPU execution
remain **NOT_RUN / not an integrated production pipeline claim**. The current
end-to-end CPU codec/Gemma route remains the explicit bounded local media bundle.

## Exact workload contract

The registered `dmd` factory uses the existing `DMDHooks/build_dmd_workload`,
Program runtime and common lifecycle. It is genuinely model-independent: role
modules supply `sample_template`, `validate_feeds`, `rollout`, `denoise`,
`loss_masks`, and `apply_conditioning`. A separate non-LTX, rank-2 payload test
uses the same factory, with no model-name branch and no hardcoded 5D boundary.

The first recipe fixes:

* Flow `z=(1-sigma)*x0+sigma*epsilon`, `v=epsilon-x0`, `x0=z-sigma*v`.
* Four deterministic Euler steps with **five explicit sigmas**
  `[1,.75,.5,.25,0]`, differentiating all steps (explicit `last_step` is also
  supported, a different policy).
* G1/F5, `reuse_student`: all five Fake updates use the detached **pre-G-update**
  Student sample. `fresh_student` is separately explicit, never inferred.
* Teacher CFG 2 with actual empty-prompt contexts; Fake conditional CFG 1.
* Uniform score sigma `[.1,.9)`. The Student target is detached
  `(fake_x0-teacher_x0)/clamp(masked_mean(abs(student-teacher_x0)),1e-5)` per
  example/modality; loss is half MSE to `detach(student-target)`. Fake regresses
  actual `epsilon-detached_student` flow velocity with MSE. Both modality
  weights are 1, normalized by the configured weight sum.
* Native VideoLatentPatchifier(1)/AudioPatchifier(1), BNC width128. Video
  coordinates use the actual VAE scale, causal first-frame correction and
  explicit FPS; audio positions are native seconds. Native `TokenTimesteps`
  wraps BN values, with **zero sigma on clean conditioning tokens**.
* First image/latent-video frame and first audio latent frame are explicit clean
  conditions, excluded from both losses. Remaining raw video/audio encodings
  supply exact native geometry; they are not silently used as a supervised DMD
  target. Prompt-only input has no fabricated media conditions.
* Ordinary visible Student/Fake/Teacher modules and AdamW optimizers,
  beta=(.9,.999), eps=1e-8, weight_decay=0, clip norm1. Each scheduler advances
  once per **actual role update**, not once per nominal cycle. Optional
  post-G-update EMA and actual `[cycles,G_updates,F_updates]` clocks are saved.

This is an explicit No-GAN flow-DMD policy, **not** official DMD2 or official
Self-Forcing loss/normalizer/gradient semantics.

LoRA is exactly `W*x + (alpha/rank)*B(A*x)`: separate trainable A/B on explicitly
named native Linear targets, zero initial B, frozen unmerged base. Teacher and
codecs remain frozen. No PEFT dependency, inference `builder.loras`, or weight
merging is used. Unknown/non-Linear targets fail before training.

## Data preparation

`inputs.raw.template.json`, `inputs.prompts.template.json`, and
`inputs.preencoded.template.json` describe the three explicit intake modes.

```python
from distillschedule.data.ltx_inputs import prepare_inputs
prepare_inputs("inputs.json", "prepared")
```

No resize, frame padding, guessed FPS, waveform resampling or implicit latent
conversion occurs. Prompt-only needs explicit native AV latent shapes, FPS and
zero conditioning-prefix lengths, plus the actual local Gemma assets.
Preencoded rows preserve every tensor exactly, but must carry both native
projected/null contexts, physical coords, clean conditioning masks, native
shapes and matching codec/source identity. Changed source/codec bindings,
invented zero unconditional features, wrong modality shapes, floating 0/1
attention masks, and stale coordinates fail closed. Bool/int64 BL masks are
binary 1=valid; floating B11L masks are native additive 0/-finfo.max only.

## Common checkpoint and portable AV output

```sh
"$PY" -m distillschedule train "$RUN/fixture/training.json" --cycles 1 --output "$RUN/partial"
"$PY" -m distillschedule train "$RUN/fixture/training.json" --cycles 3 \
  --resume "$RUN/partial/checkpoints/cycle-000001" --output "$RUN/resumed"
"$PY" -m distillschedule evaluate "$RUN/fixture/training.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/evaluation"
"$PY" -m distillschedule export "$RUN/fixture/training.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/export"
```

The export projects **current trained G + all actual native codecs/Gemma/
connectors** and inline local tokenizer config. It contains neither Teacher,
Fake, optimizers, EMA nor a training-assets path. Use the `model.factory` and
`model.parameters` in `export.json` as an inference-only model config; generate
on a new complete prepared native input with `--output-kind audio_video`.
The original training weights/inputs/checkpoints are not required to load it;
the pinned external source and isolated runtime remain required.

Decoded video comes from actual `decode_video` FHWC `[0,1]` chunks, then explicit
uint8 conversion. Audio comes from actual `decode_audio` + Vocoder, exported as
native-rate PCM WAV. The outputs are a frame directory + WAV + safe tensor
artifact, **not a muxed or quality-validated production movie**. Explicit latent
diagnostics use `--output-kind latents`; no latent reshape is called a decoder.

## Verification and boundaries

```sh
LTX_SOURCE_ROOT="$LTX_SOURCE_ROOT" "$PY" -m pytest -q \
  tests/test_training_dmd_flow.py tests/test_training_ltx_source.py \
  tests/test_training_ltx_original_checkpoint.py tests/test_training_ltx_cpu.py
```

Tests exercise actual native raw codecs and text, AV forwards/backwards, loss
masking, real updates, LoRA ownership, and **fresh-process** restart equality of
all state/RNG/Adam/schedulers/EMA/data/versions versus uninterrupted training,
followed by current-G AV export/generation without training base assets.
Missing optional upstream source skips; once the source is explicitly supplied,
missing dependencies are failures, not silent skips.

CPU/native source acceptance is not GPU/distributed parity, a benchmark,
pretrained quality, or full production checkpoint compatibility. Those remain
`NOT_RUN`; nonapplicable strategies never silently substitute another model.
