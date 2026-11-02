# LTX-2.5: native AV + reusable No-GAN flow DMD

Train actual LTX audio–video Transformers with full parameters or independent,
unmerged Student/Fake LoRA. The common `dmd` algorithm supplies G1/F5 flow DMD;
native video/audio VAEs, vocoder, Gemma4 and connectors supply inputs and decoded
output. This integration uses FP32 on one CPU or explicit `cuda:N` device and
bounded native architectures (1..16 layers, video/audio head inner widths <=1024).

## Source, environment and assets

Use an isolated `train,ltx25` environment with Python >=3.11 and the pins in
[environment.json](environment.json). The external
[Lightricks/LTX-2 source](https://github.com/Lightricks/LTX-2) is fixed at
`e98f6d927d83c92ef2be6aec76c8440b19306e39` (ltx-core 1.5 / LTX-2.5), under the
**LTX-2.x Community License**, separately from framework Apache-2.0.
The source loader verifies 114 source/license files in a private import namespace.
Set `PY` to this environment's Python, `LTX_SOURCE_ROOT` to that checkout,
`BUNDLE` to your asset directory and `RUN` to a separate project-local run directory:

```sh
# Optional source-only fetch; skip when the pinned source is already present.
"$PY" recipes/ltx25/fetch_source.py "$LTX_SOURCE_ROOT"
```

Supply an explicit `assets.json` containing native constructor configs and exact
local FP32 state dictionaries for every role and codec; the fixture below is a
complete example. Inventory existing files before use:

```sh
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
```

For an explicitly selected Hub source, use the [pinned download/materialization
workflow](../../docs/guides/model_families.md#prepare-model-assets) first.
The source receipt does not replace the native asset manifest. Every local
weight is hash-bound and checked for exact keys, shapes, dtype and finite values;
native architecture bounds are checked before loading.

### Original safetensors transformer import

[original-role.template.json](original-role.template.json) can replace any
`roles.student/fake/teacher` entry. It requires native
`__metadata__.config.transformer`, explicit geometry/behavior, `model_version=2.5`
and original `model.diffusion_model.*` keys. Loading uses the pinned native
**SafetensorsModelStateDictLoader → LTXModelConfigurator →
LTXV_MODEL_COMFY_RENAMING_MAP → SingleGPUModelBuilder(device=cpu)**, with complete
key/shape/value checks and independently owned parameter storage.

Set `cpu_max_parameters` explicitly. The same bounded architecture contract
applies to local/original roles and exports.
`strict_fp32` requires FP32 storage; `cast_native_float_to_fp32` explicitly allows
native FP16/BF16 through the builder's FP32 cast. Both native Transformer
attention paths use MATH SDPA. Codecs/Gemma remain explicit local bundle entries.

## First run with random native assets

This helper creates small actual upstream classes, procedural PNG/video/WAV
inputs and local text assets in a new directory:

```sh
"$PY" recipes/ltx25/create_cpu_fixture.py "$RUN/fixture" \
  --source-root "$LTX_SOURCE_ROOT" --cycles 3
distillctl train "$RUN/fixture/training.lora.json"
# Full-parameter training uses the same algorithm and lifecycle.
distillctl train "$RUN/fixture/training.json"
```

## Exact workload contract

The registered `dmd` factory uses the existing `DMDHooks/build_dmd_workload`,
Program runtime and common lifecycle. Role modules supply `sample_template`,
`validate_feeds`, `rollout`, `denoise`, `loss_masks`, and `apply_conditioning`.
This is an explicit No-GAN flow-DMD policy, distinct from official DMD2 and
official Self-Forcing.

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

Complete-cycle save/restore uses `distillschedule.dmd_clocks.v1` and native
`distillschedule.dmd_exponential_lr.v1` schedulers. Counters must equal `[c,c,c*N]`,
scheduler `last_epoch`/`_step_count` must equal actual updates/updates plus one,
and logical versions must equal Program write counts times `c`. Admission checks
these before updates, restore and save publication. Plain-tensor clock
checkpoints predate this resumable schema.

LoRA is `W*x + (alpha/rank)*B(A*x)`: explicit native Linear targets, separate
trainable A/B, zero initial B and frozen unmerged bases. Teacher/codecs stay
frozen; training does not use PEFT, inference `builder.loras` or weight merging.

## Prepare inputs

Choose the [raw](inputs.raw.template.json), [prompt](inputs.prompts.template.json)
or [preencoded](inputs.preencoded.template.json) template:

```sh
distillctl data prepare "$RUN/inputs.json" --output "$RUN/prepared" \
  --preparer distillschedule.data.ltx_inputs:prepare_inputs
```

Preparation uses the native media encoders and actual Gemma/tokenizer/connectors,
including genuinely encoded empty-prompt video/audio contexts. It never resizes,
pads frames, guesses FPS, resamples audio or implicitly converts latents.
Prompt-only records specify AV latent shapes, FPS and zero conditioning-prefix
lengths. Image conditioning covers only image-backed encoded frames, not a
zero-filled tail of a requested longer video. Preencoded rows retain exact
projected/null contexts, coordinates, clean masks, native shapes and codec/source
binding. Bool/int64 BL masks use binary 1=valid; floating B11L masks use native
additive 0/-finfo.max.

## Execution and checkpoint lifecycle

Copy [training.json](training.json) or [training.lora.json](training.lora.json),
fill source/assets/data/output paths, and select `trainer.device`. Reference
mode works on CPU or single-device CUDA FP32. The LoRA owner also supports exact
immutable sharing with this `execution` object:

```json
{"mode":"manual","implementation":{"backend":"torch-callable-resident-v1","share_immutable":true,"checkpoint_bindings":[]}}
```

Only equal frozen tensors alias; Student/Fake trainables and optimizer state
remain independent. Roles initially load before sharing; see the shared
[capacity conditions](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing).
For the random native CPU fixture, measured planning is available with
`{"mode":"auto","memory_limit_bytes":100000000,"max_candidates":4,"samples":1,"max_program_runs":5}`.
Reuse its selected policy with `{"mode":"plan","plan":"/ABSOLUTE/selected-plan.json"}`.
Whole-role trial copies are restricted to the CPU1 FP32 random fixture; CUDA
uses reference or the direct sharing policy, without clone-based trials.

```sh
distillctl train "$RUN/train.json" --cycles 1 --output "$RUN/partial"
distillctl train "$RUN/train.json" --cycles 3 \
  --resume "$RUN/partial/checkpoints/cycle-000001" --output "$RUN/resumed"
distillctl evaluate "$RUN/train.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/evaluation"
distillctl export "$RUN/train.json" \
  --checkpoint "$RUN/resumed/checkpoints/cycle-000003" --output "$RUN/export"
```

Use the checkpoint paths returned by `train`. The [common lifecycle
guide](../../docs/guides/training.md) covers save/resume and measured-plan identity.

## Generate new audio–video

The export contains current Student + native codecs/Gemma/connectors and inline
tokenizer config. The pinned external source and isolated runtime remain needed;
training weights, Teacher/Fake, optimizer and old inputs are not needed.
Use the [inference-request builder](../../docs/guides/training.md#generate-and-export)
and [export-only preparer](../../docs/guides/export_inputs.md) for new inputs.
Save `$RUN/new-inputs.json` beside `export/` with this complete prompt manifest
for the tiny fixture:

```json
{
  "schema_version": "distillschedule.export_inputs.v1",
  "weights": "export",
  "mode": "prompts",
  "records": [{
    "id": "new-prompt",
    "prompt": "a dog rests",
    "geometry": {
      "fps": 8.0,
      "video_shape": [1, 128, 2, 2, 2],
      "audio_shape": [1, 8, 11, 16],
      "conditioning": {
        "video_prefix_latent_frames": 0,
        "audio_prefix_latent_frames": 0
      }
    }
  }]
}
```

For `mode: raw`, use the [native raw record fields](inputs.raw.template.json)
(`image`/`video`/`audio`/`text`/`metadata.geometry`) in the same export envelope.
Media paths are relative to this manifest. Missing media needs explicit latent
shapes; clean conditioning comes only from actual supplied media.

```sh
distillctl data prepare "$RUN/new-inputs.json" --output "$RUN/new-prepared" \
  --preparer distillschedule.data.export_inputs:prepare_inputs
distillctl generate "$RUN/inference.json" --weights "$RUN/export" \
  --prepared "$RUN/new-prepared/manifest.json" --output "$RUN/new-av" --output-kind audio_video
```

Use your supported native geometry for other assets. Output is native-decoded
RGB frames, native-rate PCM WAV and a safe tensor artifact, not a muxed movie.
`--output-kind latents` selects latent diagnostics instead.
