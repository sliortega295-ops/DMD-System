# Native image flow-DMD: shared CPU reference lifecycle

**Implemented, not a production GPU claim:** actual local Diffusers Transformer,
native Qwen text encoder/tokenizer and native VAE; unmerged same-family LoRA;
the existing `training.dmd` Program; real G1/F5 AdamW, per-role schedulers and EMA;
complete-cycle restart; current-Student export and decoded new-input images.
No second trainer, kernel search, inference LoRA merge, or approximate cache.
The first admitted recipe is batch-one for raw, prompt-only and preencoded
inputs alike; unsupported preencoded batching is rejected before training.
The tests use randomly initialized **actual upstream tiny classes**, explicitly
`SYNTHETIC_TEST_ONLY`. Full pretrained execution, V3–V5 GPU/long-run/quality,
AMP/distributed and performance are **NOT_RUN**, not inferred from those tests.

Read the family contracts: [FLUX.2 klein Base](../flux_dmd/README.md) and
[original Qwen-Image](../qwen_image_dmd/README.md).

## Isolated dependency lane

Use Python 3.12 (the qualified CPU interpreter), CPU Torch 2.10.0 and the
`modern-flow` extra in a separate environment. Exact pins are recorded in
[environment.json](environment.json). Do not upgrade/combine existing DMD2,
Self-Forcing or Helios Transformers4 / PEFT0.17 lanes. PEFT0.18.1 is required
for the qualified Diffusers/Transformers5 import lane; trainable LoRA itself
uses the framework's explicit native Linear `W*x + (alpha/r)*B(A*x)`.

Set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Nothing in construction, preparation,
training, export or generation downloads assets, installs dependencies or
initializes a CUDA generator. Unsupported CUDA/distributed/autocast requests
are rejected. Catalog and known-compatibility queries do not import Torch.

## First run with actual tiny native classes

`PY` denotes the Python executable in that isolated environment; `RUN` must be
a new project-local/scratch run directory. Choose exactly one family:

```sh
"$PY" recipes/flow_image/create_cpu_fixture.py "$RUN/fixture" \
  --family flux2-klein-base --cycles 3
# Alternative: --family qwen-image
"$PY" -m distillschedule train "$RUN/fixture/train.json" --cycles 1 --output "$RUN/first"
"$PY" -m distillschedule train "$RUN/fixture/train.json" --cycles 3 \
  --resume "$RUN/first/checkpoints/cycle-000001" --output "$RUN/resumed"
"$PY" -m distillschedule export "$RUN/fixture/train.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/export"
```

Use the **exact checkpoint path returned by each command** if the installed
lifecycle's output naming differs. The fixture creates procedural RGB inputs,
real native text/empty-prompt features, strictly saved native safe-format assets
and an explicit training recipe. It changes Fake's saved output projection only
to exercise nonzero first-cycle gradients; that is fixture evidence, not a
pretrained initialization recipe or speedup claim.

## Already-present pretrained-format assets

Obtain and license assets independently; these recipes never fetch them. The
bundle must contain the exact family's `transformer/`, `vae/`, `text_encoder/`
and `tokenizer/` folders, with actual native `config.json`, complete safetensors
(or complete indexed shards), and tokenizer files. Check the family’s fixed
repository revision yourself. A declared repository name plus file hashes does
**not** authenticate remote pretrained provenance, which remains
`LOCAL_PRETRAINED_UNVERIFIED` until independently established.

```sh
"$PY" recipes/flow_image/lock_local_assets.py "$BUNDLE" --family flux2-klein-base \
  --cpu-max-parameters "$EXPLICIT_PER_COMPONENT_CPU_BUDGET" --text-max-length 512
```

The default `strict_fp32` rejects non-FP32 floating storage. An explicitly chosen
`--dtype-policy bf16_or_fp32_to_fp32` permits only BF16/F32 stored floating
weights and converts BF16 **exactly** to FP32 compute. Source hashes and policy
are manifest-bound; FP16, FP8, quantized, missing, malformed and nonfinite
weights have no fallback. This is not BF16 compute or GPU qualification.
Budgets are parameter-count limits, **not** a promise that three large models,
Adam, EMA, text/VAE and temporary loading state fit your RAM. Full pretrained
multi-billion-parameter CPU construction has not been exercised.

Copy the family's `training.template.json`, replace the three absolute asset/data/output
paths, and select actual existing native Linear LoRA targets. Its explicit
G1/F5/four-step policy is a CPU reference recipe, not tuned pretrained quality.

Optional `--student`, `--fake`, `--teacher` choose separate folders **inside**
the bundle. The first recipe requires matching native role architectures.
With identical base checkpoints and zero-initialized LoRA B, the first G
surrogate may be zero before Fake diverges; no random production fallback is
introduced to hide that fact. CPU1 LoRA is admitted; full-parameter image
training and other family variants are not.

## Native offline preparation

Use the existing trusted preparer plugin, not a new global CLI option:

```sh
"$PY" -m distillschedule data prepare "$RUN/inputs.json" --output "$RUN/prepared" \
  --preparer distillschedule.data.flow_image_inputs:prepare_inputs
```

A raw manifest is:

```json
{
  "schema_version": "distillschedule.flow_image_inputs.v1",
  "family": "flux2-klein-base",
  "assets": "/absolute/bundle/assets.json",
  "mode": "raw",
  "records": [{"id": "sample-0", "prompt": "a red bird", "image": {
    "path": "input.png", "sha256": "REPLACE_WITH_ACTUAL_IMAGE_SHA256"
  }}]
}
```

Raw images must be actual RGB with geometry divisible by native VAE scale times
2; no resizing, color conversion, padding or sampled VAE posterior is silently
introduced. Encoding uses the native posterior **mode** and each family's exact
normalization/packing. There is one T2I record per complete cycle; use a real
finite cohort or an explicitly supported common sampling policy.

For `mode: "prompts"`, replace `image` with `latent_shape: [1,C,H,W]` in **native
unpatched VAE BCHW units**. The zeros created for that mode are explicit
noise-generation geometry templates, never conditional features or real images.
Prompt and empty-prompt features are always encoded by the actual native text
model. For `mode: "preencoded"`, records instead contain only `id` and
`features: {"path": "row.pt", "sha256": "..."}`. Complete tensor-only rows,
family/config/source/codec/dtype binding, masks, geometry and the exact genuine
empty-prompt features must match. Wrong-family same-shape tensors, stale codecs,
unknown/cache/edit fields and invented null contexts are refused.

Use one explicit text width for a cohort. FLUX retains native fixed-length
Qwen3 hidden states without an arbitrary denoiser mask. Qwen-Image uses native
crop34 features plus **fixed right padding with a native boolean mask**; only
masked padding is zero. Neither path silently truncates an overlength prompt.
The tests compare the padded Qwen native forward against the native unpadded
forward and train three different-length prompt records through the common
shape-bound lifecycle. Changing the padding/text/codec/source policy changes
the binding and invalidates old prepared features.

## Mathematics and execution contract

This is explicitly configured **No-GAN rectified-flow DMD**, not official DMD2,
Self-Forcing, or an official FLUX/Qwen distillation reproduction. For
`z=(1-sigma)*x0+sigma*epsilon`, native output is `v=epsilon-x0` and
`x0=z-sigma*v`; there is **no output negation**. The first recipe uses four
explicit Euler steps `[1,.75,.5,.25,0]`, G1/F5, teacher/fake CFG=1, actual
AdamW (`betas=.9,.999`, `eps=1e-8`, no weight decay), per-actual-update exponential
LR clocks and explicit optional EMA. Loss/noise/normalizer/gradient/Fake policies
are all present in the generated training JSON and owned by common `dmd`.

Reference mode needs no optimization. To exercise the **existing exact frozen
parameter sharing** mechanism, set:

```json
{"mode": "manual", "implementation": {
  "backend": "torch-callable-resident-v1", "share_immutable": true,
  "checkpoint_bindings": []
}}
```

Replace only the training JSON’s `execution` section. Frozen Teacher/media and
unmerged LoRA bases declare immutable subtrees. All named immutable **slots**
remain content-checked even when equal norm/bias parameters alias inside one
subtree; optimizer ownership still counts unique trainable Parameters. Tests
check real storage reduction, actual aliases and exact loss/gradient/update/
Adam/scheduler/EMA/Teacher equality against an unshared control. No approximate
cache, guessed normalization kernel or inference weight merging is added.

## New input and portable output

Prepare genuinely new prompts with the same codec binding, then export the
restored current Student. The returned export manifest supplies the exact
`build_inference_model` factory and parameters for an inference config. Keep
those model parameters unchanged; choose an independent inference RNG seed in
the trainer section. Run:

```sh
"$PY" -m distillschedule generate "$RUN/inference.json" --weights "$RUN/export" \
  --prepared "$RUN/new-prepared/manifest.json" --output "$RUN/generated" --output-kind image
```

The export contains current unmerged Student plus actual native VAE/text/tokenizer
codecs, not Teacher, Fake, Adam, EMA, a base checkpoint path or training inputs.
The tests hide all training assets and checkpoints during a fresh-process
portable generation and compare it exactly against the current restored Student.
Actual RGB PNG output is a decoded **diagnostic**, not pretrained quality proof.
`--output-kind latents` instead produces explicitly labeled native normalized
latent diagnostics (FLUX is still BN-normalized patchified BCHW at that boundary).

## Focused validation

```sh
DMD_MODERN_FLOW_CPU=1 "$PY" -m pytest -q \
  tests/test_training_flow_images_cpu.py tests/test_training_model_algorithm_compatibility.py
```

Opt-in missing dependencies are failures, not hidden skips. Outside the isolated
lane, the CPU-native test file is explicitly skipped; lightweight compatibility
checks remain runnable. V1/source and synthetic CPU lifecycle evidence does not
complete the plan's U12/U13 V3–V5 pretrained/GPU/quality gates.
