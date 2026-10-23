# Current Student export and original cached-AV latent inference

This opt-in **CPU1/world1/FP32** route extends [CPU_REFERENCE.md](CPU_REFERENCE.md).
It exports the **actual restored current Student: complete original frozen base
plus its unmerged LoRA64 tensors**. It does not export fake-score LoRA,
discriminator, optimizer, RNG, history, or input-cursor weights. It does not use
the original `export_lora` helper (which scales B for a different loader), merge
LoRA, cast weights, select EMA, change the canonical GAN start500, or modify the
frozen distributed/native W6 contract.

The inference path uses the maintained original MiniMaxH3DiT, H3Adapter,
cache/packing parser, `make_schedule` and unchanged eight-step `rollout`. This is
not full DiffSynth, an independent solver, a codec, a pretrained-model result,
quality validation, GPU validation, or a performance claim.

## Export a complete current checkpoint

Use the same explicitly admitted local environment and training configuration
that produced the checkpoint. Export restoration still needs its original
training assets/data; **generation from the resulting export does not**.

```sh
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
distillctl export /ABS/training.json \
  --checkpoint /ABS/run/checkpoints/cycle-000003 --output /ABS/student-export
```

The common named-module format writes full `weights.pt`, `export.json` and a
hash-bound `_SUCCESS`. It is **not a resumable training checkpoint**. Training
identity stays under the separate `identity.training_provenance` field, not
inside inference model parameters/metadata. No historical checkpoint is
retroactively relabeled: current source/helper/runtime identities must match.

Projection makes structure-only shallow copies of the original modules, with
independent registration dictionaries and the same live base/Student Parameter
and buffer objects. Only the copied fake-score dictionaries are pruned. It
never calls `eval`, `requires_grad_`, `select`, `to` or a forward on shared
parameters, and the projected components have no generation callback. Training
modes, flags, roles, optimizer state, object identities and RNG are preserved.

## Prepare NEW genuine cached conditions

Prepare a **separate** input inventory as described in CPU_REFERENCE.md, with
matching explicit current/previous/prompt geometry and actual detached finite
FP32 caches. Both the original flat cache and official split/packed triple are
admitted. Preparation invokes the original parser and preserves its tensor-only
canonical representation. Do not synthesize prompts or previous AV to stand in
for actual conditions.

```sh
distillctl data prepare-h3-cached-reference \
  --sources /ABS/NEW-inputs.json --output /ABS/NEW-prepared
```

Current `input_latents` and `audio_input_latents` provide **geometry templates
only**: their values are not an inference target or a starting sample. The
original rollout draws its own `randn_like` AV noise. Prompt embeddings and the
single clean previous AV chunk are genuine model conditions. Stereo audio order
is preserved. Masks, dirty/noise-augmented anchors, padding/control, keyframes,
retake conditioning, unknown fields, shape padding and dtype casts are refused.

## Bind a separate inference request

Copy `inference.cpu-latents.template.json` into your own run directory. Replace
its entire `model` object with the **exact** `model.factory` and
`model.parameters` from the completed export, and replace all absolute path
placeholders. The deliberately invalid placeholder parameters are not defaults.

Inference parameters explicitly bind architecture, original LoRA64 and
8-step/shift2.22 schedule, AV/prompt geometry, CPU parameter/byte/input limits,
evidence label, source16/current-helper content and dependency identity. They
contain no training paths, base-shard list, dataset or optimizer configuration.
Keep exported model parameters (including factory seed) unchanged; the separate
`trainer.seed` is the **generation noise seed** and may be chosen explicitly.

```sh
distillctl generate /ABS/inference.json \
  --weights /ABS/student-export --prepared /ABS/NEW-prepared/manifest.json \
  --output-kind latents --output /ABS/generated-latents
```

A fresh process constructs only a bounded original DiT/H3Adapter strict-load
target, prunes its own fake-score structure, then the existing common loader
stages every named tensor before strict restore. A serialization-only load
completion guard refuses generation from random constructor values. Only this
independently owned inference object is frozen/evaluated. It constructs no
discriminator or Adam and needs no training dataset, restart checkpoint,
old assets manifest or base safetensors shards. It refuses training and
checkpoint evaluation; use exported-weight generation, not `train`/`evaluate`.

The completed `generation.pt`, `generation.json` and `_SUCCESS` contain:

- `video_latents`: finite FP32 `[1,24,T,H,W]`, **BCTHW**.
- `audio_latents`: finite FP32 `[2,32,T]`, **SCT**, where S=2 is stereo.
- Exact named latent schema; `LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY` and
  `quality_validation=NOT_RUN`. No decoded frames/audio are claimed or invented.

The default output kind remains **video**. Without a real decoder that request
fails, leaving no partial output directory. No fake decoder is provided.

## Evidence and exclusions

`tests/test_training_h3_cached_inference_cpu.py` exercises actual original
training, fresh-process resume/export across the separately labeled
`GAN_START_2_TEST_FIXTURE`, full restored-current Student weight equality,
non-mutating projection/export and an exact subsequent original update. A
second fresh process runs with all old training payloads removed and trapped,
and rejects discriminator/Adam construction. Both flat and packed caches match
direct original eight-step rollout exactly at the same seed. Tests separately
exercise template-value independence and genuine prompt/previous-AV conditions,
strict staged weight rejection and source/runtime/config/input refusal.

The GAN2 fixture remains **tiny, not pretrained and not quality evidence**;
canonical training still uses GAN500. CUDA/distributed, auto/manual/saved-plan
modes, changes of source/dependency/precision, alternate samplers, media
decoding, production pretrained capacity and performance remain unsupported or
NOT_RUN. No numerical/source-manifest changes are made by this inference route.
