# Opt-in original cached-AV CPU reference

This **separate, bounded CPU1/world1/FP32 reference** connects the maintained
cached-only original source to `TrainingConfig`, Program execution and common
complete-cycle checkpoints. It does not change `training.json`, the native W6
ZeRO3/FSDP contracts, source16/source83 pins or historical results.

It uses the actual original MiniMaxH3DiT, independent unmerged student/fake
LoRA64 adapters, velocity discriminator, `DMD2Trainer.update`, `make_optimizers`
and `optimizer_step`. A single nonreplayable `external_grad` Call owns a complete
8-step rollout, one critic update and one student update. The continuous
logit-normal sigma and nine-boundary schedule ending at zero are not Grid1000.

## Install and supply explicit local assets

From an installed checkout, use the independent extra:

```sh
python -m pip install -e '.[h3-cached-reference]'
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

No model weights or caches are shipped or downloaded. Copy
`assets.cpu-reference.template.json`, `inputs.cpu-reference.template.json` and
`training.cpu-reference.json` into your own run directory. The placeholders are
deliberately **not executable defaults**. Fill every absolute, canonical local
file path, byte count and full SHA256. List every full base safetensors shard;
LoRA-only, missing, duplicate, partial, nonfinite or wrong-dtype bases are refused
before real model parameter allocation. No random fallback is available.

For the explicitly labeled tiny-model fixture, the actual source architecture
is in `architecture.cpu-tiny-fixture.json`. A locally created deterministic
full base must be labeled in the safetensors metadata with
`evidence_kind=FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY`.
The tests create and pin such a fixture; it is **not pretrained or quality
evidence**. The other admitted label,
`LOCAL_PRETRAINED_CACHED_AV_UNVERIFIED_QUALITY`, requires operator-supplied
payloads with that immutable metadata label and the same strict CPU capacity;
it is not a claim that a production pretrained model fits or was tested.

Populate the exact source/runtime objects explicitly in your local manifests:

```python
from distillschedule.reference import diffsynth_h3_cached
from distillschedule.training.h3_cached_assets import runtime_identity

source = diffsynth_h3_cached.source_identity()  # no Torch/model/network import
runtime = runtime_identity()                 # installed CPU closure identity
```

Use `source` in both manifests and `runtime` in the assets manifest. Runtime
versions, imported paths, entry hashes, FP32 default and single-thread CPU
settings are bound. The factory accepts only `asset_manifest`, `device` and
`seed`, never a dataset or cached tensors. The fixed config is fully explicit;
the original default `critic_updates=5` is never silently used for W6 G1F1.
The common session seeds setup once. Admission/meta inspection preserves caller
RNG, and the factory never resets an already advanced caller stream; the
original adapter/head construction consumes that stream naturally.

## Prepare actual original caches

Each inventory entry pins one unique local cache, in finite consumption order.
Accepted caches are the original flat dict or official
`(shared, positive, negative)` triple with current AV, one clean previous AV
chunk and width5120 prompt embeddings. AV/prompt tensors must already be
detached finite CPU FP32. Packed previous anchors need the original positional
grid and clean coefficients of1. The original parser's derived positional
grid is float64; it is preserved exactly, not treated as a model precision cast.

Preparation calls the original parser first and proves that its canonical raw
tensor-only representation reparses identically. It preserves stereo order,
previous/current packing and generated-only output semantics. It retains only
the packed grid fields required to reconstruct positions; it never serializes
parsed Python `seq_len`, noise-augmentation floats or `None` as prepared leaves.
All prepared leaves remain tensors under the existing `PreparedDataset` format.
Preparation is two-pass and streams one row at a time, retaining only metadata
between passes. It consumes no training RNG and computes no DMD noise, sigma,
timestep or rollout.

```sh
distillctl data prepare-h3-cached-reference --sources /ABS/inputs.json --output /ABS/prepared
distillctl data validate /ABS/prepared/manifest.json --sources
```

Masks, padding/control, keyframe/retake conditioning, unknown fields, dirty
anchors, duplicate source/content, cropping, padding and dtype casts are not
supported. The reference also requires fixed current/previous/prompt geometry.
Generic raw media preparation can use a separately supplied legitimate encoder;
this cached-only closure does **not** provide an AV encoder/decoder or fake one.

## Train, save and resume in a new process

Point the copied training config at the admitted assets and prepared manifest.
Choose target cycles no greater than both the finite inventory and the fixed
capacity. Then use the ordinary public command:

```sh
distillctl train /ABS/training.json
```

For continuation, start another process with the same identity-bound config,
a new `trainer.output_dir`, a larger `trainer.cycles`, and
`trainer.resume_from=/ABS/train/checkpoints/cycle-000002`. The common safe v2
checkpoint stages all LoRAs, head tensors, named AdamW groups/options/moments
and per-parameter clocks, sigma/config, native RNG, G/C clocks, modes, selected
role/trainability and consumed-source chain before any live restore mutation.
The frozen full base is reloaded from immutable local pins, not duplicated in
every checkpoint. Initial teacher/frozen-adapter/head-trainable flags differ
from the completed student-trainable/fake-and-head-frozen flags and are checked
separately. All owned gradients must be `None` at the complete boundary.

Native Python/NumPy/Torch RNG persists privately across Program RNG scopes;
outer RNG is saved/restored separately by the common checkpoint. Completed
outer `cycle=cursor=Gsteps=Csteps` and `versions.native_update=cycle` must agree.
The exact original trusted-pickle checkpoint helpers remain unchanged and are
**not called** by this public route. No automatic checkpoint recycling or
partial-cycle continuation is permitted.

Equivalently, the public CLI accepts
`distillctl train /ABS/training.json --resume /ABS/train/checkpoints/cycle-000002 --cycles 3 --output /ABS/resumed`.

The canonical protocol keeps GAN enabled at `gan_start_step=500`. The focused
tests additionally use the separately labeled `GAN_START_2_TEST_FIXTURE` with
`gan_start_step=2`, crossing the actual GAN branch on output3. This label is
accepted only for the non-pretrained fixture; it never changes canonical500.

## Evidence and limits

`tests/test_training_h3_cached_public_cpu.py` and
`tests/test_data_h3_cached_inputs.py` exercise actual CPU models through the
public chain, direct-original exact update equivalence, tensor-only flat/split
parser equivalence, all-owner tamper rejection and a **new OS process** exact
uninterrupted3 versus2+resume3 comparison across the GAN-start2 transition.
The independent original source-provider/CPU tests remain applicable unchanged.

Auto/manual/saved-plan execution, distributed/CUDA, epochs/shuffle/GAS,
backend substitution, portability across dependency/source/precision changes,
decoded-media generation and portable generator export are not admitted here.
Export/generation depends on the separate generic inference projection unit
and legitimate AV decoder assets. These are cached latent training results,
not video/audio files, pretrained quality results, GPU validation or speedups.
