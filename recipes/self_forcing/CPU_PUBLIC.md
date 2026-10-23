# LiveWan public CPU1 training → current-G export → new-input generation

This is **original LiveWan/Wan2.1 CPU1 FP32, 32×32, 81 frames / 21
latents**, not world16/BF16, C51 role-pool restart, pretrained quality or a
performance result. It is independent of the separately pinned official
`guandeh17/Self-Forcing` route. Random tiny full-class tests are not pretrained.

Use the `livewan-wan21-reference` plus `train` extras (Torch≥2.10,
Diffusers **0.35.2**, NumPy and safetensors); add `media` for PNG frames and
the optional original codec (the extra includes `einops==0.8.1`). Do not mix in Helios'
Diffusers0.37.1 extra and call it the qualified Wan runtime. Run with one CPU
thread, CPU FP32/default dtype, no outer autocast, no initialized CUDA or
distributed process group. No asset or model download is automatic.

## Supply and prepare training assets

`distillschedule.training.livewan_public` defines the exact versioned `ASSETS`,
`SCOPE`, `PRECISION`, `GEOMETRY`, `BACKEND` and runtime/source identities. The
asset manifest must pin complete original Student/Teacher/Critic JSON configs
and all-FP32 safetensors, separate raw negative `[1,512,4096]`, original Critic
LoRA32 seed, truthful evidence label and CPU construction/load/cycle/attention
budgets. All models use original `WanModel`, not replacement dummy modules.
The geometry requires full raw text `[512,4096]` and full latent `[16,21,4,4]`
for **each of five distinct records per cycle**. Source and content copies do
not count as fresh records. There is no pickle, resize, cast or native crop.

Prepare the separately versioned five-record source manifest using the public
explicit preparer (its schema/constants are in `data/livewan_reference_inputs.py`):

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
distillctl data prepare /absolute/training-inputs.json \
  --preparer distillschedule.data.livewan_reference_inputs:prepare_inputs \
  --output /absolute/training-prepared
distillctl train /absolute/cpu-training.json
distillctl export /absolute/cpu-training.json \
  --checkpoint /absolute/run/checkpoints/cycle-000001 \
  --output /absolute/current-G-export
```

The training request uses model factory
`distillschedule.training.livewan_public:build_model` with only its explicit
`asset_manifest` pin/path, and algorithm
`distillschedule.training.livewan_public:build_algorithm`, Program/reference,
finite unshuffled single-process data, CPU, integer seed, positive finite
cycles and checkpoint interval1. Use the actual checkpoint path returned by
training, rather than guessing a directory name. Complete-cycle restart owns
G/C/Teacher, named Adam state, RNG and finite cursor together; partial owners
are poisoned and cannot export. Each cycle is original G1/C5/EMA0.

Export contains **only current full Student** under role `student`; never
Teacher, Critic, either Adam, private training RNG/cursor or a pre-G rollout.
Inference construction has architecture/source/runtime/geometry contents, not
training asset paths. Training assets, prepared cohorts and checkpoints are
not needed for subsequent generation. Export provenance may name historical
training paths but the inference factory never resolves them.

## Supply a new inference record

Use `data/livewan_inference_inputs.py`'s `SCHEMA`, `FORMAT` and
`producer_identity()`. Its source manifest has exact fields `schema_version`,
`format`, `scope`, `precision`, `geometry`, `backend`, `producer`,
`evidence_kind`, `records`. Each record is `{id, source, world_frames}` with
an absolute `{path,bytes,sha256}` safetensors pin and `world_frames`3,6 or9.
The safe archive has **only** `world:[1,16,W,4,4]` and
`context:[1,512,4096]`, finite detached CPU FP32, with the truthful matching
`evidence_kind` file metadata. Supply an actual clean opening world and raw
positive text; no negative, fabricated zero history, implicit teacher-clip
crop, future target frames or prompt encoder is provided.

```bash
distillctl data prepare /absolute/NEW-inputs.json \
  --preparer distillschedule.data.livewan_inference_inputs:prepare_inputs \
  --output /absolute/NEW-prepared
```

Create `/absolute/inference.json` using the exact `model.factory` and
`model.parameters` returned in `current-G-export/export.json`, algorithm
`distillschedule.training.livewan_inference:refuse_training`, the NEW prepared
manifest, Program/reference, CPU, cycles1 and an explicit runtime seed.
The factory seed is metadata, not a reseed of each streaming block. Generation
uses a fresh native cache, current text projection, clean world K/V, RoPE gap0,
all four original renoise steps with fresh intermediate noise and the final
clean t0 K/V commit. It never calls CUDA-timed `stream()`.

```bash
distillctl generate /absolute/inference.json \
  --weights /absolute/current-G-export --prepared /absolute/NEW-prepared/manifest.json \
  --output-kind latents --output /absolute/generated-latents
```

Raw `latents` are explicitly **BCTHW `[1,16,21,4,4]` FP32**. They concatenate
the supplied W-frame world and generated `21-W` suffix; input metadata also
records `(W-1)*4+1` conditioning pixel frames. Latent mode emits no frames and
is labeled `LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY`. Requested video without a
real admitted decoder fails before output publication.

## Optional original Wan2.1 decode, not Diffusers or official Self-Forcing VAE

`training/wan21_codec.py` defines the independent codec schema and constants.
Use only pinned original Wan2.1 revision
`9737cba9c1c3c4d04b33fcad41c111989865d315`, `wan/modules/vae.py` SHA256
`4912ffd7a9fbb17ceef9a17d4a9c4b8aee36ec37c32594f54e57693b29909498` and
`LICENSE.txt` SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
Supply a local canonical root with these two files. They import through a
synthetic namespace; no native `wan` alias or frozen v1 inventory is modified.
The source license does not grant a pretrained-weight or dataset license.

The decoder manifest has exact fields `schema_version`, `scope`, `precision`,
`source`, `runtime`, `architecture`, `normalization`, `decode_policy`,
`geometry`, `evidence_kind`, `weight_provenance`, `weights`, `budget`.
Use the public codec constants and `source_spec(root)`/`runtime_identity()`.
Native production architecture is dim96/z16, multipliers `[1,2,4,4]`, two
residual blocks, temporal flags `[false,true,true]`, no attention scales,
dropout0. Safe pinned all-FP32 safetensors must have every original state key,
shape and finite value; weight provenance/license is mandatory. Admission
requires explicit parameter/load/activation/attention budgets. No production
random initialization is accepted; the tiny-width RANDOM_NOT_PRETRAINED schema
is solely for random complete-class tests.

The original constructor's exact 16-channel mean/std is verified and applied
once inside native `WanVAE_.decode`. The versioned demo policy clamps a
**decode-only copy** to[-4,4], clamps native RGB to[-1,1], then maps to THWC
uint8. Raw generated latents are unchanged. No resize, reshape-as-RGB,
Diffusers fallback or implicit CUDA autocast is used. Decoder identity and all
source/weight/policy pins are saved in `generation.pt` under `decoded.decoder`;
the explicit manifest pin is also in `generation.json.generation_parameters`.

Write `generation-parameters.json` as
`{"decoder_manifest":{"path":"/absolute/decoder.json","bytes":123,"sha256":"ACTUAL_SHA256"}}`
using the actual size/hash, then run:

```bash
distillctl generate /absolute/inference.json \
  --weights /absolute/current-G-export --prepared /absolute/NEW-prepared/manifest.json \
  --parameters /absolute/generation-parameters.json --output-kind video \
  --output /absolute/decoded-frames
```

This creates81 actual32×32 PNG RGB frames (sample fps16), not an encoded MP4.
Missing, changed, corrupt, wrong-dtype/layout/config/runtime or out-of-budget
codec assets fail closed, including when a broken codec was explicitly passed
to latent mode. No pretrained decoder assets have been supplied or qualified;
real dim96 capacity, visual quality, GPU/BF16/world16/C51 and performance remain
**NOT_RUN**.

## CPU qualification tests

Set `DMD_WAN21_TEST_UPSTREAM` to the above ORIGINAL local Wan2.1 root to run
the codec tests rather than skip them. The tests copy only the two pinned
source/license files to disposable scratch and generate tiny random full-class
weights; they download nothing. Run:

```bash
python -m pytest -q tests/test_training_livewan_reference_inputs.py \
  tests/test_training_livewan_public_cpu.py tests/test_training_livewan_inference_cpu.py
```

Coverage includes current-G export, pure projection/next-update equality,
fresh-process generation after all training payloads are deleted, actual
native full-step/cache census and exact native control, runtime seed and new
conditioning sensitivity, explicit latent schema, CLI latent/video paths,
actual original random codec21→81 pixels and strict source/asset admission.
