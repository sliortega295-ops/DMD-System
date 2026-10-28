# Native W1 checkpoint → current G → NEW-input decoded output

This fills only the native W1 output gap. Do not repeat already-qualified
training/restore or CPU-reference work. This path has source/CPU evidence;
**full pretrained 1.3B native GPU output and quality are not yet qualified**.
W2/W3/W5/W6 output work is separate. See the
[versioned contract](../../docs/contracts/WAN_NATIVE_OUTPUT_V1.md).

## 1. Keep the original complete native binding

Use the exact W1 fresh-stream task/assets/bootstrap/source/runtime/topology and
strategy or selected plan that own the complete checkpoint. A checkpoint at
the terminal horizon (including 100) is legal for export without another
training cycle. Do not rename weights to a restart checkpoint, alter its source
identity, use EMA, merge LoRA, or substitute another preset. Older-source
checkpoints need a separately reviewed compatibility decision, not forged pins.

## 2. Supply local Wan2.1 VAE and capacity policy

Copy [export.native-policy.template.json](export.native-policy.template.json)
as the `trainer.export` object in a **copy** of the native training config.
Replace every placeholder with the local actual Wan2.1 VAE config/safe shard
path, exact byte count/SHA256, real model revision and license. List all shards
once. No asset is downloaded by this feature. Weights must already be finite
FP32 native Wan2.1, z16/RGB3/spatial8/temporal4; not Wan2.2 TI2V.

Match prompt_shape to your actual offline T5 features, and choose the explicit
inference seed/fps. The template's 64GiB construction/inference reserve and
16GiB per-owner / 256GiB aggregate export reserve are **admission ceilings,
not measured consumption or a memory reservation**. Inspect actual node host
capacity (all eight local owners gather), codec inventory, output geometry and
standalone device before running. Too-small budgets fail; no silent downscale,
latent clamp, precision conversion, or hidden offload is attempted.

This is an export-command supplement. Ordinary `train` continues to refuse
`trainer.export`; remove the supplement before any later training invocation.
The previous reference/literal/incumbent training recipes are unchanged.

## 3. Export on the same existing native world16 owner

On every rank of the original, explicitly admitted two-node/eight-ranks-per-node
launcher, execute the normal public export command:

```sh
distillctl export /project/native-training-with-output-policy.json \
  --checkpoint /project/training/checkpoints/cycle-000100 \
  --output /project/outputs/current-g
```

`distillctl` does not launch nodes or reserve GPUs. The existing native owner
restores complete state and branches straight into output, with zero training
cycles/cursor/optimizer/EMA advancement and no graph capture. Output paths must
not already exist. All ranks materialize full G/codec weights; only rank0
publishes the standard hash-bound inference artifact. Per-rank output-owner
receipts are in the separate `current-g.native-owner` sibling directory.

Successful publication produces `current-g/weights.pt`, `export.json`,
`_SUCCESS`, and companion `current-g.inference.json`. Preserve the manifest and
completion marker together. The artifact includes full current BF16/FP32 G
with unmerged FP32 LoRA plus the actual VAE, not only adapter or latent tensors.

## 4. Generate independently on NEW prompt features

Prepare a new `distillschedule.data.preencoded.v1` manifest containing only
BF16 `prompt_embeds` of the declared shape, with real feature hashes and new
sample IDs. Import it using the existing public data path:

```sh
distillctl data import /project/new-prompts/inputs.json --output /project/new-prompts/prepared
distillctl generate /project/outputs/current-g.inference.json \
  --weights /project/outputs/current-g \
  --prepared /project/new-prompts/prepared/manifest.json \
  --output-kind video --output /project/outputs/new-video
```

Generation is **one independent process**, not world16 and not training resume.
The emitted config defaults to CPU; an explicitly capacity-admitted user may
set only `trainer.device` to a visible single CUDA device. Do not alter model
parameters, source/runtime identities or precision to bypass strict loading.
It constructs no optimizer/Critic/EMA and does not need old training/base/VAE
directories. Keep seed and NEW prompt provenance with the result.

The emitted model uses native inference factory ABI **v2** and explicit
`distillschedule.wan_fresh_inference_noise.v1`. Four independent labelled CPU
streams supply fresh BF16 entry/rollout noise and FP32 exact-PSD gamma=1/3
block noise through the original operation-input seam. Original helpers,
training/default/reference behavior and global compatibility flags are not
changed. The old gamma=1/3 covariance is singular, so its default Cholesky path
can fail; this uses the existing centered four-vector PSD law instead, not
diagonal jitter or iid replacement. This is a new inference RNG contract,
**not old Cholesky draw-order/bitwise replay or training RNG continuation**.
Pyramid last-step horizons, native forward, schedule, flow and shapes remain
unchanged. See the versioned contract for exact substream derivation and order.

The original native three-stage sampler consumes those fresh inputs to produce
latents; the real bundled VAE decodes them with native std/mean applied once and
**no latent clamp**. Success yields THWC uint8 PNG frames in `frames/`, plus
`generation.pt`, `generation.json` and `_SUCCESS`. Declared geometry/fps and
decoder evidence are retained. This is a decoded frame sequence, not an MP4
container or a quality score. `--output-kind latents` is a separate explicit
diagnostic, never a substitute for decoded-output acceptance.

## Focused validation

- `tests/test_training_wan_native_outputs.py`: real small installed VAE,
  safe codec/RNG/refusals and actual native restore/output-only control seams
  with explicitly synthetic CUDA/rank/scalar owners; new singular-PSD/noise
  substream/control/RNG/refusal and operation-input routing cases.
- `tests/test_training_wan_native_output_owners.py`: actual two-process CPU
  Gloo/DTensor all-owner materialization, rank0 publication and failed writes.
- `tests/test_training_wan_native_outputs_upstream_cpu.py`: optional original
  Diffusers0.37.1/PEFT0.17.1 stack; actual tiny original G/current full export,
  independent native sampler and real VAE after old assets are removed. This
  random marker-checkpoint proof does not rerun training or qualify world16.

Publication failure must not be treated as a successful output. Distributed
unpublished `.tmp-*` staging is deliberately retained for failure audit; never
manually add `_SUCCESS`. Existing exports are never overwritten. Full native
pretrained GPU, world16 and quality gates remain separately `NOT_RUN`.
