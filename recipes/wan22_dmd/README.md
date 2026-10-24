# Wan2.2 TI2V-5B, T2V-only common-flow DMD

This independently named recipe uses **actual** `WanTransformer3DModel`, residual
`AutoencoderKLWan`, `UMT5EncoderModel`, native T5 fast tokenizer and
`WanPipeline._get_t5_prompt_embeds` through the existing common Program lifecycle.
It is not Wan2.1/W1, official Self-Forcing or an official DMD2 equivalence claim.
Source/revision/Apache-2.0 attribution is in [source.json](source.json).

## Verified boundary

CPU1 or single-CUDA FP32 reference, batch one; same-family Teacher/Student/Fake; exact unmerged LoRA;
four explicit Euler steps; one Student/five Fake updates per complete cycle;
Adam, scheduler, EMA, RNG/data order/clocks; independent current-Student export;
THWC uint8 video decoded by the actual residual VAE. Tests create random tiny
native weights under **SYNTHETIC_TEST_ONLY**, not pretrained or quality evidence.
The example learning rates/rank are fixture settings, not quality recommendations.

Full pretrained memory capacity/training/quality, AMP and distributed execution
are **NOT_RUN**. CUDA manual/auto/saved-plan execution and resident trial cloning,
expert/high-low/A14B, Wan2.1, I2V, image/causal/cache fields,
other codecs and incompatible algorithms explicitly reject. No automatic download,
weight conversion to another family, implicit image condition or fallback trainer.

## Isolated environment and local assets

Use the separate `modern-flow` extra and [environment.json](environment.json).
Do not merge it into the Transformers4/PEFT0.17 DMD2/Self-Forcing lanes. Native
cleaning uses pinned ftfy, not a replacement cleaner. Install/setup is an operator
step; this recipe never installs dependencies or downloads models at runtime.

Place already-acquired native safetensors/config directories inside one local
bundle: `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`. The operator is
responsible for source authenticity and aggregate RAM. Inventory/hash everything:

```sh
python recipes/wan22_dmd/lock_local_assets.py /ABSOLUTE/local-wan22 \
  --cpu-max-parameters YOUR_EXPLICIT_PER_COMPONENT_LIMIT
```

The locker neither loads weights nor grants capacity. Runtime validates each actual
native constructor's meta parameter budget **before any concrete component is
allocated**, then validates all safe shard names/headers/shapes/finite values.
The default storage policy is strict FP32. `--dtype-policy
bf16_or_fp32_to_fp32` admits only exact BF16→FP32 storage conversion, not BF16
compute/GPU qualification. Pick a budget for available host RAM; no claim that
three full 5B roles plus optimizer/text/VAE fit on the current host. Student,
Fake and Teacher may use distinct `--student/--fake/--teacher` subdirectories,
but architecture/latent domain must match. Source identity remains explicitly
`LOCAL_PRETRAINED_UNVERIFIED`, not automatically authenticated to the public pin.

## Inputs and public lifecycle

RGB frame directories require Pillow. Container video files additionally require
**both ffmpeg and ffprobe** already on PATH; a missing tool is an error, not a
download, guessed frame rate or optional skip.

Copy/edit the raw, prompt or preencoded templates. Raw input is a self-contained
video supported by existing ffmpeg/ffprobe, or a sorted RGB frame directory; no
resize/crop/frame resample. Raw `F=(Flatent-1)*4+1`; H/W divisible by32 (native
spatial16 codec and spatial2 Transformer patches). The VAE patchifies internally:
never patchify RGB twice. Latents are native BCFHW, normalized by actual VAE
mean/std; mode encoding is deterministic. Prompt-only zeros are explicitly
geometry/noise templates, **not** fabricated conditions. Real UMT5 text is fixed
226 rows with the native zero padding; overflow rejects, not silent truncation.
Empty-prompt features are genuine native encodings. Preencoded rows must include
all native context/mask/geometry/codec-binding fields from this preparer.

File SHA256 is required. For a frame directory, obtain its sorted inventory
fingerprint with `distillschedule.data.manifest.source_identity(path, frames=True)`.
Payload hashes, original source identities, sample IDs and cursor are retained.

```sh
python -m distillschedule models show wan22-ti2v5b-t2v
python -m distillschedule data prepare /ABSOLUTE/inputs.raw.json \
  --preparer distillschedule.data.wan22_inputs:prepare_inputs --output /ABSOLUTE/prepared
# Edit training.lora.json paths before running; the fixed common recipe is G1/F5.
python -m distillschedule train /ABSOLUTE/training.lora.json --output /ABSOLUTE/run
python -m distillschedule train /ABSOLUTE/training.lora.json \
  --resume /ABSOLUTE/run/checkpoints/cycle-000001 --cycles 3 --output /ABSOLUTE/resumed
python -m distillschedule export /ABSOLUTE/training.lora.json \
  --checkpoint /ABSOLUTE/resumed/checkpoints/cycle-000003 --output /ABSOLUTE/export
```

Use the actual checkpoint path returned by the command (do not assume its name).
For the single-device CUDA reference path, set `trainer.device` explicitly to
`cuda:0` (or another available local CUDA index), keep `execution.mode` as
`reference`, and use the same device for training and complete-cycle resume.
Adam, scheduler, EMA and all native media/role tensors retain their owned device;
the common DMD algorithm and update clocks are unchanged. Offline data preparation
remains CPU-based. CUDA inference can also relocate a portable export to CPU;
this is different from cross-device training-checkpoint resume, which is not
admitted. No automatic precision conversion or distributed launch is implied.

An inference config uses the returned export `model.factory`/`model.parameters`;
prepare a **new** prompt with explicit compatible latent geometry, then generate
with `--weights /ABSOLUTE/export --prepared /ABSOLUTE/new-prepared/manifest.json
--output /ABSOLUTE/generated --output-kind video`. The portable export owns the
CURRENT Student (not base, Teacher or EMA), native VAE/UMT5 weights and the exact
native tokenizer backend/class/extra IDs/options. It does not need the original
training asset bundle, Teacher/Fake or a training checkpoint. Source/runtime
fingerprints must still match. A resumable complete-cycle checkpoint and a
portable inference export are distinct artifacts, not interchangeable.

To execute the existing **CPU** immutable-base optimization, explicitly select:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

This uses existing frozen-base capability inventory and runtime alias checks,
not fixture hand-aliasing or merged LoRA. Teacher is frozen and Student/Fake
optimizers own disjoint trainables. Immutable mutation invalidates execution;
there is no silent downgrade to a different strategy.

## Reproducible CPU-only synthetic check

The [common export-only preparer](../../docs/guides/export_inputs.md) additionally
encodes new prompts/raw video using this export's restored native codecs after
all training assets/checkpoints are deleted. The Wan asset-based preparer remains
separate and still requires its original bundle. Output is a PNG frame sequence,
not an inferred-FPS playable container or pretrained-quality result.

```sh
DMD_MODERN_FLOW_CPU=1 python recipes/wan22_dmd/create_cpu_fixture.py /ABSOLUTE/NEW-fixture
DMD_MODERN_FLOW_CPU=1 python -m pytest -q tests/test_training_wan22_cpu.py
```

The tests use actual native tiny models, raw5×32×32 RGB → latent2×2×2 → raw5;
manual native velocity/Euler/text/codec equality; real gradients and optimizer
updates; shared/unshared parity; fresh-process exact complete restart; new-prompt
portable decode while original assets are hidden and training checkpoints deleted;
and explicit refusal cases. No fake denoiser/codec stands in for end-to-end proof.

The separate [native CUDA acceptance](../../artifacts/framework/20261024/native_video_cuda_acceptance.json)
covers three actual tiny Wan2.2 tests on a non-default owned CUDA device:
native forward/gradient/Euler/text/VAE comparison; complete DMD/Adam/EMA/RNG
state; and new-process restart plus export-only new-prompt RGB video generation.
CPU inference relocation matches an independent CPU current-Student reference.
The same explicit deterministic test policy applies to oracle and CLI processes;
it does not change production defaults or establish pretrained quality/capacity.

## Third-party notice

Model/config source: Wan-AI/Wan2.2-TI2V-5B-Diffusers at the exact revision in
`source.json` (Apache-2.0). Implementations come from pinned Hugging Face Diffusers
and Transformers (Apache-2.0), native tokenizer/tokenizers and ftfy/sentencepiece;
retain their licenses when redistributing. No pretrained weights are bundled.
