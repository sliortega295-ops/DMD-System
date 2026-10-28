# Native image flow-DMD: shared reference and replicated lifecycle

**Implemented, not a production GPU claim:** actual local Diffusers Transformer,
native Qwen text encoder/tokenizer and native VAE; unmerged same-family LoRA;
the existing `training.dmd` Program; real G1/F5 AdamW, per-role schedulers and EMA;
complete-cycle restart; current-Student export and decoded new-input images.
No second trainer, kernel search, inference LoRA merge, or approximate cache.
The first admitted recipe is batch-one for raw, prompt-only and preencoded
inputs alike; unsupported preencoded batching is rejected before training.
The tests use randomly initialized **actual upstream tiny classes**, explicitly
`SYNTHETIC_TEST_ONLY`. Both families have actual A100 FP32 reference update,
restart and new-input output evidence. Full pretrained execution/capacity,
long-run quality, AMP/sharded ownership and performance are **NOT_RUN**, not inferred
from those tests.

Read the family contracts: [FLUX.2 klein Base](../flux_dmd/README.md) and
[original Qwen-Image](../qwen_image_dmd/README.md).

## Isolated dependency lane

Use Python 3.12 (the qualified CPU interpreter), CPU Torch 2.10.0 and the
`modern-flow` extra in a separate environment. Exact pins are recorded in
[environment.json](environment.json). Do not upgrade/combine existing DMD2,
Self-Forcing or Helios Transformers4 / PEFT0.17 lanes. PEFT0.18.1 is required
for the qualified Diffusers/Transformers5 import lane; trainable LoRA itself
uses the framework's explicit native Linear `W*x + (alpha/r)*B(A*x)`.

For the CPU path set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
That path does not initialize CUDA. Neither path downloads assets or installs
dependencies. Autocast and unsupported distributed policies are rejected. Catalog and
known-compatibility queries do not import Torch.

### Explicit single-GPU reference

Use the same pinned packages with a compatible CUDA Torch2.10 build, keep the
offline settings, and expose the GPU you own. The acceptance environment was
Python3.13.11 / Torch2.10.0+cu128 on one A100-40GB. Prepare inputs on CPU as below;
set `trainer.device` to `cuda:0` in a copy of the recipe and retain
`execution.mode: reference`. If `model.parameters.device` is also specified,
it must agree. The same `distillctl train`, resume, export and generate commands
then use the owned device; no separate trainer or experimental launcher is needed.

CUDA reference owns all role/media tensors, Adam state and EMA, including
Qwen's complex RoPE buffers without a lossy dtype conversion. Automatic
planning, saved plans and manual policies other than exact immutable sharing still
**fail closed**: the current whole-role trial copies are not capacity-bounded.
CPU planning remains independently supported for the documented tiny fixtures.
Do not infer full pretrained models fit merely from the parameter-count budget.

[The exact acceptance receipt](../../artifacts/framework/20261024/modern_image_cuda_acceptance.json)
records native-oracle loss/gradient/Adam/EMA checks, new-process C1-to-C3 restart,
and current-Student new-prompt PNG generation with training assets removed.
Exact CUDA comparisons use an explicit deterministic **test** policy, not a
changed production default or a promise of default-kernel determinism.

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

The strict text loader admits the pinned public legacy JSON ABI through the
actual native Config parser: Qwen RoPE aliases, Qwen2.5-VL flat text fields and
the audited T5/UMT5 encoder metadata. Unknown fields, conflicting aliases and
unsupported metadata values are refused before weight payload loading. Widths
come from parsed native configs, not assumed JSON nesting. Original config bytes
and hashes remain asset-bound; export reconstruction retains the raw document,
including T5's legacy tying/decoder-scaling intent, rather than normalizing and
reparsing `Config.to_dict()`. Native config classes and compatibility source are
also runtime-bound, so old prepared data/exports need regeneration when that
source identity changes.

Native safe saves can omit redundant tied keys (for example T5/UMT5
`encoder.embed_tokens.weight`). Completion is allowed only when the actual
constructor proves the slots hold the **identical Parameter object**. At least
one tensor per alias group must exist, and all supplied aliases must agree
exactly. Independent equal-shaped/equal-valued tensors are never filled. Exact
indices, headers, dtype policy and finite payload checks still apply; ordinary
optimizer enumeration remains unique. This format compatibility is not video
training, pretrained-weight execution, GPU or quality qualification.

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

That exact policy is also admitted on the single-CUDA FP32 path; it installs
sharing on the persistent live owner without CUDA trial cloning. See the
[shared CUDA policy and capacity boundary](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing).
All roles still materialize before sharing; this does not lower construction
capacity requirements. CUDA auto/saved-plan trials remain refused.

## Explicit replicated training, not model sharding

FLUX and Qwen-Image can use the existing post-backward replica backend. Each
rank owns complete Student, Fake Score and Teacher models; each consumes one
distinct prepared record per cycle. Gradients are averaged across ranks **before**
clipping and the single optimizer commit. Per-call noise has a deterministic
rank/world namespace, without advancing the outer training RNG. The original
single-rank seed stream is unchanged. This is equal-global-batch replication,
not FSDP, tensor parallelism or a way to fit an oversized model.

Copy a normal reference recipe and change only these sections:

```json
{
  "data": {"prepared": "/absolute/prepared/manifest.json", "sharding": "rank_strided"},
  "execution": {"mode": "reference"},
  "trainer": {
    "cycles": 3, "seed": 17, "device": "cuda",
    "distributed": {"backend": "nccl", "timeout_seconds": 180},
    "output_dir": "/absolute/run"
  }
}
```

Retain the recipe's model and algorithm sections. Prepare at least
`world_size * cycles` distinct records; incomplete cohorts fail rather than
padding, duplicating or dropping samples. Each record remains local B1; the
global batch is the number of ranks. `cuda` resolves to the actual local rank;
do not force every process to `cuda:0`.

For example, run the following on **each** of two allocated eight-GPU nodes,
with `NODE_RANK` set to 0/1 and a shared configuration/output filesystem:

```sh
"$PY" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --module distillschedule train "$RUN/train.json" --cycles 1 --output "$RUN/partial"
# Repeat the same torchrun prefix in new processes:
# ... --module distillschedule train "$RUN/train.json" --cycles 3 \
#       --resume "$RUN/partial/checkpoints/cycle-000001" --output "$RUN/resumed"
# ... --module distillschedule export "$RUN/train.json" \
#       --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/export"
```

Use the checkpoint path returned by the CLI. Resume/export require the **same
topology** and all rank shards; `generate --weights` uses the exported Student
in its separate single-device inference owner. CPU/Gloo follows the same path
with `device: cpu` and `backend: gloo`.

To combine replication with the already-supported exact frozen-state sharing,
replace `execution` with this explicit policy:

```json
{"mode": "manual", "implementation": {
  "backend": "torch-callable-replicated-v1",
  "reduction": "post_backward", "bucket_cap_mb": 0,
  "local": {"backend": "torch-callable-resident-v1",
            "share_immutable": true, "checkpoint_bindings": []}
}}
```

The effective receipt must show completed gradient reductions, replica state
consensus and, for manual mode, real shared aliases. Private native rollout and
denoise methods do not pass through a role's `forward`; DDP-overlap wrappers,
FSDP/HSDP, auto/saved-plan trials, autocast and unrelated caches remain explicitly
refused. Construction still materializes all roles. Tiny-model replica tests
cannot establish that pretrained 4B/20B configurations fit or converge; see the
[acceptance status](../../docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

## New input and portable output

After exporting, the [common export-only preparer](../../docs/guides/export_inputs.md)
can encode new prompts/raw RGB using restored native codecs even after deleting
all training assets/checkpoints. The existing image asset-based preparer below
remains available and still explicitly requires its original asset bundle.

Prepare genuinely new prompts with the same codec binding, then export the
restored current Student. Use the
[common inference-request builder](../../docs/guides/training.md#generate-and-export)
with your training request, completed export and new prepared manifest to create
`inference.json`. It copies the export's exact `build_inference_model` factory
and parameters, with an independent inference RNG seed in the trainer section,
instead of reusing the training factory or its assets/streams. Run:

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
  tests/test_training_flow_images_cpu.py tests/test_training_native_text_assets_cpu.py \
  tests/test_training_model_algorithm_compatibility.py
```

Opt-in missing dependencies are failures, not hidden skips. Outside the isolated
lane, the CPU-native test file is explicitly skipped; lightweight compatibility
checks remain runnable. V1/source and synthetic CPU lifecycle evidence does not
complete the plan's U12/U13 V3–V5 pretrained/GPU/quality gates.
