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
| FLUX.2 klein Base 4B / explicit flow DMD | Flux2 Transformer with normalized sigma | Native Qwen3 layers 9/18/27, patch/BN VAE, exact immutable sharing, restart and current-Student decoded images | isolated `train,modern-flow`; FP32 or BF16 autocast, local B1 |
| Original Qwen-Image T2I / explicit flow DMD | QwenImage Transformer, native text mask/img_shapes; CFG=1 only | Qwen2.5-VL template/crop34, fixed masked right padding, native VAE, exact restart/current-Student images | isolated `train,modern-flow`; FP32 or BF16 autocast, local B1 |
| Wan2.2 TI2V-5B T2V subpath / explicit flow DMD | Wan Transformer, expanded normalized timestep, CFG=1 | UMT5/T5 and residual patch2/temporal4 VAE, raw video/features, exact restart/current-Student RGB | isolated `modern-flow`; FP32 or BF16 autocast, local B1 |
| HunyuanVideo-1.5 T2V / explicit flow DMD | Hunyuan15 Transformer, dual text encoders and temporal4 VAE | Native text-only Qwen2.5 + T5/ByT5, video/features and current-Student RGB; Tencent asset terms | isolated `modern-flow`; FP32 or BF16 autocast, local B1 |
| Existing W1–W6 | Their respective native rollout, loss and update rules | See each recipe and six-workload support matrix | Model-specific environments |

Choose the model and algorithm together, then an execution policy:

- **Common flow DMD:** LTX and the four image/video families support
  single-device FP32 reference and direct manual frozen-state sharing.
  FLUX/Qwen also provide equal-B1 post-backward replicas on Gloo/NCCL;
  video recipes expose their explicit local-B1 replica interface.
- **Official DMD2:** SD1.5 includes a [native FSDP2 reference](native_workloads.md#dmd2_fsdp2_reference)
  and an optimized Resident/DDP profile.
- **Official Self-Forcing:** native FSDP2 reference and optimized profiles
  preserve causal rollout and native update clocks.
- **Planning:** use the [execution guide](training.md#explicit-or-automatic-strategy)
  and [capability matrix](#support_matrix--additional-model-and-workflow-integrations)
  to choose a reference, explicit optimized or measured saved-plan configuration.

Official Self-Forcing and W4 LiveWan have separate recipes. W4 uses a
14B Teacher and 1.3B Student with its own cross-scale sampling/update rules.
W5 is full-sequence H3 with an approximate Grid1000 schedule;
W6 is autoregressive previous/current H3 with continuous sigma.

## Prepare model assets

Use an existing model directory, or explicitly download a pinned Hugging Face
revision into the shared cache:

```sh
# Existing files: hash in place, without copying weights.
distillctl assets prepare --local-dir "$BUNDLE" --output "$RUN/source.json"

# Hub source: install the optional assets dependency group, then prepare once.
distillctl assets prepare --hf-repo "$MODEL_ID" --revision "$COMMIT_SHA" \
  --cache-dir "$CACHE" --materialize-dir "$BUNDLE" --output "$RUN/source.json"
distillctl assets verify "$RUN/source.json"
```

`COMMIT_SHA` is the full 40-character revision from the selected recipe.
Materialization creates regular files for native loaders, using hardlinks when
possible and copies across filesystems. Keep linked weights unchanged. Verify
the downloaded source first, then run the recipe's native asset locker on
`$BUNDLE` and prepare inputs. The locker adds its model manifest; training uses
that manifest and the local bundle. The source receipt records the exact file
set at preparation time. Existing directories and receipts are never overwritten.
Training and model discovery do not download weights.
`--tree-timeout SECONDS` bounds the complete paginated Hub file tree (default 60).
Exact pinned files then use the Hub's finite metadata/streaming timeouts and
retries, with no whole-payload transfer deadline or duplicate repository discovery.

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

The [performance and source records](performance_validation.md) link the model-specific
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
CPU/single-CUDA FP32/BF16-autocast reference training, manual frozen-state sharing and
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
rollout steps, G1/F5, local B1 and CFG=1. They expose CPU/single-CUDA FP32/BF16-autocast
reference training, manual frozen-state sharing and explicit local-B1 replica
interfaces. Raw videos or frames and prompts are prepared into source-bound
features; generation decodes the current Student through the native VAE.
Set `model.parameters.compute_precision: bf16-mixed` for native BF16 compute;
master parameters, optimizer/EMA state and flow/sampling math stay FP32. The
same setting is retained by checkpoints, saved plans and portable exports.

The [export-only input preparer](data.md#export_inputs) encodes new prompts/media
using exported inference components. See the [Wan2.2 recipe](../../recipes/wan22_dmd/README.md)
and [HunyuanVideo recipe](../../recipes/hunyuan_video_dmd/README.md) for local
asset manifests, memory budgets and commands. Hunyuan model assets retain the
Tencent Hunyuan terms.

<a id="support_matrix"></a>
<a id="support_matrix--additional-model-and-workflow-integrations"></a>
## Supported models, workflows and execution

DMD-System uses one **`distillctl`** interface for data preparation, persistent
training, complete-cycle recovery and current-Student generation. Choose the
model/workflow below, then use its recipe's source, assets, environment and
hardware contract. [Training](training.md), [data preparation](data.md) and
[extension interfaces](extensions.md) describe the common lifecycle; the
[execution capability map](optimization_capabilities.md#choose-an-execution-path)
lists physical strategies without implying every model supports each one.

### Six native workloads

| Workload | Workflow and execution | Continuous training and output |
| --- | --- | --- |
| W1 Wan1.3B | Hierarchical No-GAN DMD; native FSDP2 or Resident; complete Graph/cast/fusion preset | [Fresh incumbent stream](../../recipes/wan_ar/INCUMBENT_STREAM.md), complete-cycle recovery and [current-G video](../../recipes/wan_ar/NATIVE_OUTPUT.md) |
| W2 Helios No-GAN | G1/C5/EMA5; FSDP2 reference and node-HSDP SafeBundle | [Safe stream](../../recipes/helios_nogan/SAFE_STREAM.md), G/C/EMA recovery and [current-G video](../../recipes/helios_nogan/NATIVE_OUTPUT.md) |
| W3 Helios GAN | G1/C5/EMA1; full world16 FSDP2, chunk2048/window5, attention reuse and down40 | [Full native stream](../../recipes/helios_gan/NATIVE_STREAM.md), GAN-head/role state recovery and [current-G video](../../recipes/helios_gan/NATIVE_OUTPUT.md) |
| W4 LiveWan / cross-scale Self-Forcing | 14B Teacher → 1.3B Student; C51 8G+8Score dual-wave pipeline | [C51 continuation](../../recipes/self_forcing/C51_CONTINUATION.md), drained two-pool recovery and [current-Student output](training.md#livewan_native_output) |
| W5 full-sequence H3 | Full-sequence AV DMD; Grid1000 approximate schedule; original reference and selected FSDP2/cache execution | [Original-reference training](../../recipes/h3-grid-original-training.md), cycle recovery and [four-step AV export](training.md#h3_grid_inference) |
| W6 autoregressive H3 | DiffSynth previous/current AV DMD2; eight rollout steps, G1/F1; native ZeRO-3/AP | [Native continuation](../../recipes/h3_ar/NATIVE_CONTINUATION.md), engine/LoRA/Adam recovery and [current-Student AV export](training.md#h3_ar_native_inference) |

[Native workload commands](native_workloads.md) link the complete configuration
and launch sequence. Original H3 video/audio encoding and decoding are available
through [raw AV preparation](data.md#h3_audio_video_encode) and [AV codecs](training.md#h3_audio_video_decode).

### Native continuity and output

Fresh-stream recipes retain role state and save at complete-cycle boundaries.
The **two-cycle** presets remain the quick performance-reproduction window,
not a training-length limit. The [performance and correctness summary](performance_validation.md)
links the **100-cycle** official-workflow runs, cycle-50 fresh-process restart,
current-Student export and longer-cycle measurements.

Portable inference weights carry the current Student and its required components.
Training checkpoints additionally retain optimizer, scheduler, EMA, RNG, input
progress and execution versions. Use `train --resume` for training recovery and
`generate --weights` for inference from new inputs.

### Additional model and workflow integrations

| Integration | Model and distillation support | Execution and output |
| --- | --- | --- |
| [Official SD1.5 DMD2](../../recipes/dmd2_sd15/README.md) | Native UNet/CLIP/VAE, single-step DMD2, optional original GAN branch and independent generator/guidance/real-image streams | CPU/CUDA reference; native FSDP2 and optimized Resident/DDP; recovery and current-G image export |
| [Official SDXL DMD2](../../recipes/dmd2_sdxl/README.md) | Dual CLIP, additional conditioning, four-step and initialized single-step399; full or unmerged-LoRA Student | FP32 reference, native role-DDP and budgeted tiny resident planning; recovery and image generation |
| [Official Self-Forcing](../../recipes/self_forcing_official/README.md) | Causal Wan chunk rollout, random gradient exit, native KV cache and DMD/EMA clocks; not W4 LiveWan | Native FSDP2 reference/R6 optimized; budgeted tiny resident policies; recovery and current-G video |
| [LTX-2.5](../../recipes/ltx25/README.md) | Native AV Transformer, Gemma/connectors/codecs; parameter-budgeted loading; common G1/F5 flow DMD; full parameters or LoRA | FP32 or native BF16 CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; opt-in CPU/Gloo FP32 manual block FSDP with dense initialization; restart and current-Student AV generation |
| [FLUX.2 klein Base 4B T2I](../../recipes/flux_dmd/README.md) | Flux2/Qwen3/VAE, common flow DMD and unmerged LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; PNG output |
| [Original Qwen-Image T2I](../../recipes/qwen_image_dmd/README.md) | QwenImage/Qwen2.5-VL/VAE, native text masks, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; opt-in CPU/Gloo FP32 manual frozen-Teacher shard-first construction; restart and current-Student PNG output |
| [Wan2.2 TI2V-5B T2V](../../recipes/wan22_dmd/README.md) | Wan/UMT5/temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; decoded video |
| [HunyuanVideo-1.5 T2V](../../recipes/hunyuan_video_dmd/README.md) | Hunyuan15, dual text encoders/temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; decoded video |

LTX and the four image/video flow families support `fp32` or explicit
`bf16-mixed` compute with FP32 master/Adam/EMA state, CPU/Gloo or CUDA/NCCL
local-B1 replicas. LTX retains its separate pinned dependency environment.
See the [sharing, replica and
clone-safety conditions](optimization_capabilities.md#direct-single-cuda-immutable-sharing)
and [compatible environments](#choose-a-compatible-integration).
The [export-only preparer](data.md#export_inputs) encodes new modern-flow prompts/media.

### Integrated official optimization profiles

- **SD1.5:** [FSDP2 primary reference](../../recipes/dmd2_sd15/training.fsdp2.json)
  and [complete optimized Resident/DDP](../../recipes/dmd2_sd15/training.optimized.json);
  FP32 role-DDP is a supplemental reference, not the FSDP2 baseline.
- **Official Self-Forcing:** [FSDP2 reference](../../recipes/self_forcing_official/training.fsdp2-reference.json)
  and [complete R6 optimization](../../recipes/self_forcing_official/training.optimized.json).

The [official workflow guide](native_workloads.md#official_optimized_training) provides launch,
save, resume and output commands. [Native-owner alternatives](native_owner_backends.md)
cover CPU component checkpointing, FP32 role-DDP and post-commit cache release;
[resident planning](native_owner_backends.md#resident_owner_training) covers tiny snapshot-safe owners.

### Hardware

A100/H100 recipe identities can be inspected with
`distillctl workloads describe --workload Wn --hardware a100|h100`.
Execution uses each recipe's device, dtype, topology and capacity settings.

### Results and verification

- [Performance and source records](performance_validation.md)
- [Official optimized training and recovery](../status/official_optimized_framework.md)
- [Model environments and test lanes](#verification-lanes)
- [Public resident lifecycle](native_owner_backends.md#resident_owner_training)
- [Source and asset licenses](../../THIRD_PARTY.md)
