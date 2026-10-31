# Models, algorithms and execution recipes

A model name selects checkpoints, prediction conventions, encoders/decoders and
trainable state. An algorithm selects actual rollout/loss/update mathematics.
An execution policy selects physical placement, communication and state reuse. The command remains
`distillctl` for every family; reference mode is not a second training CLI.

## Choose a compatible integration

| Model / workflow | Native mathematics | Data and output | Environment |
| --- | --- | --- | --- |
| SD1.5 / official DMD2 | Native `SDUniModel`/`SDGuidance`; single-step, optional original GAN | Prompt tokens; independent guidance/real/denoising streams when required; native VAE image output | `train,dmd2` |
| SDXL / official DMD2 | Dual CLIP, added conditions, backward simulation; four-step or explicitly initialized single-step | Both tokenizers; native VAE, pooled and unconditional conditions | `train,dmd2` |
| Official Self-Forcing Wan | Causal random-exit block rollout, KV state, DMD loss, native G/C and EMA clocks | Native T5 prompt encoding or pinned embeddings; native full-step causal inference and VAE video | `train,self-forcing-official` |
| LTX-2.5 / generic No-GAN DMD | Native AV LTX Transformer; explicit rectified-flow surrogate and Fake regression policy | Gemma4/connectors, native video/audio VAEs, vocoder; AV or explicit latents | isolated `train,ltx25`, Python >=3.11 |
| FLUX.2 klein Base 4B / explicit flow DMD | Flux2 Transformer with normalized sigma | Native Qwen3 layers 9/18/27, patch/BN VAE, exact immutable sharing, restart and current-Student decoded images | isolated `train,modern-flow`; CPU/CUDA FP32, local B1 |
| Original Qwen-Image T2I / explicit flow DMD | QwenImage Transformer, native text mask/img_shapes; CFG=1 only | Qwen2.5-VL template/crop34, fixed masked right padding, native VAE, exact restart/current-Student images | isolated `train,modern-flow`; CPU/CUDA FP32, local B1 |
| Wan2.2 TI2V-5B T2V subpath / explicit flow DMD | Wan Transformer, expanded normalized timestep, CFG=1 | UMT5/T5 and residual patch2/temporal4 VAE, raw video/features, exact restart/current-Student RGB | isolated `modern-flow`; CPU/CUDA FP32, local B1 |
| HunyuanVideo-1.5 T2V / explicit flow DMD | Hunyuan15 Transformer, dual text encoders and temporal4 VAE | Native text-only Qwen2.5 + T5/ByT5, video/features and current-Student RGB; Tencent asset terms | isolated `modern-flow`; CPU/CUDA FP32, local B1 |
| Existing W1–W6 | Their respective native rollout, loss and update rules | See each recipe and six-workload support matrix | Model-specific environments |

Choose the model and algorithm together, then an execution policy:

- **Common flow DMD:** LTX and the four image/video families support
  single-device FP32 reference and direct manual frozen-state sharing.
  FLUX/Qwen also provide equal-B1 post-backward replicas on Gloo/NCCL;
  video recipes expose their explicit local-B1 replica interface.
- **Official DMD2:** SD1.5 includes a [native FSDP2 reference](dmd2_fsdp2_reference.md)
  and an optimized Resident/DDP profile.
- **Official Self-Forcing:** native FSDP2 reference and optimized profiles
  preserve causal rollout and native update clocks.
- **Planning:** use the [execution guide](training.md#explicit-or-automatic-strategy)
  and [capability matrix](support_matrix.md#additional-model-and-workflow-integrations)
  to choose a reference, explicit optimized or measured saved-plan configuration.

Official Self-Forcing and W4 LiveWan have separate recipes. W4 uses a
14B Teacher and 1.3B Student with its own cross-scale sampling/update rules.
W5 is full-sequence H3 with an approximate Grid1000 schedule;
W6 is autoregressive previous/current H3 with continuous sigma.

## Explicit local assets

The optional providers use hash-locked upstream source, supplied locally, with
its original license. They never download model weights. The source-only fetch
helpers are optional, explicit operator actions, not training side effects.
DMD2 is CC-BY-NC-SA-4.0; LTX is LTX-2.x Community License; their sources are not
bundled or relicensed as framework Apache-2.0.

A model manifest binds actual architecture, every required component, all local
checkpoint bytes and relevant dependency versions. Missing or changed assets,
shape/precision disagreement, wrong prediction meaning and incomplete restore
state fail before training or output publication. A compatible shape alone is
not permission to substitute epsilon, velocity, flow or x0.

The modern-flow native loader checks every safetensors shard header before
constructing the model, then copies and releases one shard at a time instead
of accumulating all shards in a second CPU checkpoint. A single-file checkpoint
is still one full-size shard. Native tied parameters and
constructor-owned buffers are preserved; conflicting aliases or nonfinite
weights are rejected. Explicit BF16 storage-to-FP32 loading remains a storage
conversion, not a change to runtime precision. Allow memory for the concrete
model, the current shard and conversion temporaries; parameter-count admission
alone is not a total host-memory or multi-role capacity estimate.

Original LTX safetensors use the upstream loader and model builder with one
load-scoped weight registry. Validation and construction share one payload;
exact comparisons cast one tensor at a time, and the registry is cleared before
training owns independent tensor storage. This avoids loading the checkpoint
twice or keeping another full FP32 validation copy. It does not shard the
original single-file checkpoint: its payload, the constructed model and the
largest conversion/copy temporary must still fit in host memory.

See the runnable source-specific guides:

- [SD1.5 DMD2](../../recipes/dmd2_sd15/README.md)
- [SDXL DMD2](../../recipes/dmd2_sdxl/README.md)
- [Official Self-Forcing](../../recipes/self_forcing_official/README.md)
- [LTX-2.5](../../recipes/ltx25/README.md)
- [FLUX.2 klein Base 4B](../../recipes/flux_dmd/README.md)
- [Original Qwen-Image](../../recipes/qwen_image_dmd/README.md)
- [Wan2.2 TI2V-5B T2V](../../recipes/wan22_dmd/README.md)
- [HunyuanVideo-1.5 T2V](../../recipes/hunyuan_video_dmd/README.md)

## Native iterations and public cycles

The persistent lifecycle saves **both** the public complete-cycle index and the
algorithm's actual native step/update counters. DMD2's macrocycle contains one
Generator commit and its configured guidance updates. Its two LR schedulers
advance on the original native iteration clock, not an invented common clock.
Official Self-Forcing preserves independent G and C sample consumption and its
post-increment EMA initialization. LTX's explicit No-GAN recipe advances each
role's scheduler on that role's actual update. These are not interchangeable
losses or update schedules.

`train --resume` requires a complete checkpoint at a full cycle boundary and the
same topology/source/input contract. It restores model/optimizer/scheduler/EMA,
inner and outer RNG, every data-stream position and logical versions. Starting
from supplied pretrained weights is **initialization**, not continuation.
Transient KV/replay/cast caches are reconstructed, not portable training state.

## Current-Student generation and export

`export` saves the current Student/Generator and its inference codecs and
conditioners. Trainable LoRA remains unmerged. `generate --weights` accepts
new prepared inputs independently of the original training data and checkpoints.

Select the adapter's `--output-kind image|video|audio|audio_video|latents`:
image adapters write PNGs; video adapters write decoded frames; audio adapters
write native-rate WAVs. Audio-video output includes both frame files and audio.
Latent export retains an explicitly named native tensor layout.

The [integration records](validation_records_20261030.md) link the model-specific
source and execution checks, and the [official workflow records](../status/official_optimized_framework.md)
cover continuous optimized training, recovery and new-input generation.

## Separate installation lanes

In a clean, model-specific environment, install CPU Torch first for CPU
qualification; a native CUDA environment instead keeps its independently
admitted runtime. Nothing in normal model construction installs dependencies.

```sh
# Choose ONE lane in this environment, never all extras together.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,dmd2]'                 # SD1.5 or SDXL
# Alternatives in separate environments:
# python -m pip install -e '.[train,self-forcing-official]'
# python -m pip install -e '.[train,livewan-wan21-reference]'
```

LTX requires **Python >=3.11** (qualification/CI: 3.12) and a separate
Transformers5 environment. First install `torch==2.10.0`, `torchaudio==2.10.0`
and `torchvision==0.25.0` from the CPU index, then `.[train,ltx25]`.
Its exact optional dependencies include Transformers5.14.1, tokenizers0.22.2,
huggingface-hub1.7.1, scipy1.16.3 and colour-science0.4.7. An installed PEFT
from the SD environment is not a substitute for native LTX unmerged LoRA.
The core lightweight package remains Python>=3.10; LTX's transitive Python
requirements are not silently omitted on an older interpreter.

Original Helios helper `.34`, Wan/LiveWan `.35.2`, and full-original Helios
`.37.1` pins remain separate. These new extras do not upgrade an existing
experiment environment, and no combined `all` extra is advertised.

## Verification lanes

Each model family has its own CPU test environment. The source checkouts are
explicit, revision-pinned inputs; the tests use upstream model classes and
small fixtures. Run with `CUDA_VISIBLE_DEVICES=''`, offline Hub modes and one
CPU thread.

| Environment / CI lane | Verified components |
| --- | --- |
| `train,test,dmd2` | `DMD2_SOURCE_ROOT`; SD1.5/SDXL model, native update clocks, checkpoint and image output |
| `train,test,self-forcing-official` | `SELF_FORCING_SOURCE_ROOT`; causal rollout, role state and video output |
| `train,test,ltx25` | `LTX_SOURCE_ROOT`; AV models/codecs and common flow-DMD lifecycle |
| `train,test,livewan-wan21-reference` | Original Wan2.1/LiveWan public input, training and inference paths |
| `train,test,test-helios-model` | Original Helios model classes, G/C/EMA owners and fresh-process recovery |
| `test-helios-native-history` with the Helios model environment | Original native 40-block/FSDP topology in an independent test process |
| `train,test,test-h3` | Original H3 loop, data progress, complete-cycle state and Gloo recovery |
| `train,test,test-h3-output` plus pinned Diffusers source | Original Grid sampler, current-Student export and distributed state gathering |
| `test-h3-consolidation` | Installed DeepSpeed Student-state serialization method |

The workflows under `.github/workflows/` pin each source/dependency closure and
check the required named test receipts. Model-specific tests run in their own
processes so their dependency and module namespaces remain independent.
The common `.34` helper lane uses `scripts/run_framework_cpu_tests.py` to select
its compatible inventory. Provider-specific tests are assigned to dedicated
jobs. Operator-supplied H3 codec assets can also be checked explicitly through
`tests/test_h3_grid_codec_target_cpu.py`.

`python scripts/check_framework_install.py` builds a wheel and tests it in a
new isolated environment, reusing the invoking numerical dependency roots in
their import order. The installed-package check covers training, recovery,
saved-plan execution, export and new-input generation outside the checkout.

## Image models: FLUX and Qwen-Image

FLUX.2 klein Base and Qwen-Image use common flow DMD with unmerged LoRA,
four explicit Euler rollout steps and G1/F5 updates. The recipes provide
CPU/single-CUDA FP32 reference training, manual frozen-state sharing and
post-backward replicas through the shared lifecycle. The initial recipes use
local B1 and Teacher/Fake CFG=1.

Each adapter owns the native tokenizer/text-conditioning layout and VAE.
The asset manifest identifies the model configuration and every checkpoint
shard. Shared and reference configurations use the same checkpoint/recovery
and current-Student output interfaces. See the [shared image guide](../../recipes/flow_image/README.md).

## Video models: Wan2.2 and HunyuanVideo

Wan2.2 TI2V-5B's T2V recipe uses its native Transformer, UMT5 conditions and
patch2/temporal4 VAE. HunyuanVideo-1.5 T2V uses its native Transformer,
Qwen2.5 text conditions, T5/ByT5 glyph conditions and temporal4 VAE.

Both recipes reuse the common flow-DMD lifecycle: unmerged LoRA, four Euler
rollout steps, G1/F5, local B1 and CFG=1. They expose CPU/single-CUDA FP32
reference training, manual frozen-state sharing and explicit local-B1 replica
interfaces. Raw videos or frames and prompts are prepared into source-bound
features; generation decodes the current Student through the native VAE.

The [export-only input preparer](export_inputs.md) encodes new prompts/media
using exported inference components. See the [Wan2.2 recipe](../../recipes/wan22_dmd/README.md)
and [HunyuanVideo recipe](../../recipes/hunyuan_video_dmd/README.md) for local
asset manifests, memory budgets and commands. Hunyuan model assets retain the
Tencent Hunyuan terms.
