# Supported models, workflows and execution

DMD-System uses one **`distillctl`** interface for data preparation, persistent
training, complete-cycle recovery and current-Student generation. Choose the
model/workflow below, then use its recipe's source, assets, environment and
hardware contract. [Training](training.md), [data preparation](data.md) and
[extension interfaces](extensions.md) describe the common lifecycle; the
[execution capability map](optimization_capabilities.md#choose-an-execution-path)
lists physical strategies without implying every model supports each one.

## Six native workloads

| Workload | Workflow and execution | Continuous training and output |
| --- | --- | --- |
| W1 Wan1.3B | Hierarchical No-GAN DMD; native FSDP2 or Resident; complete Graph/cast/fusion preset | [Fresh incumbent stream](../../recipes/wan_ar/INCUMBENT_STREAM.md), complete-cycle recovery and [current-G video](../../recipes/wan_ar/NATIVE_OUTPUT.md) |
| W2 Helios No-GAN | G1/C5/EMA5; FSDP2 reference and node-HSDP SafeBundle | [Safe stream](../../recipes/helios_nogan/SAFE_STREAM.md), G/C/EMA recovery and [current-G video](../../recipes/helios_nogan/NATIVE_OUTPUT.md) |
| W3 Helios GAN | G1/C5/EMA1; full world16 FSDP2, chunk2048/window5, attention reuse and down40 | [Full native stream](../../recipes/helios_gan/NATIVE_STREAM.md), GAN-head/role state recovery and [current-G video](../../recipes/helios_gan/NATIVE_OUTPUT.md) |
| W4 LiveWan / cross-scale Self-Forcing | 14B Teacher → 1.3B Student; C51 8G+8Score dual-wave pipeline | [C51 continuation](../../recipes/self_forcing/C51_CONTINUATION.md), drained two-pool recovery and [current-Student output](livewan_native_output.md) |
| W5 full-sequence H3 | Full-sequence AV DMD; Grid1000 approximate schedule; original reference and selected FSDP2/cache execution | [Original-reference training](../../recipes/h3-grid-original-training.md), cycle recovery and [four-step AV export](h3_grid_inference.md) |
| W6 autoregressive H3 | DiffSynth previous/current AV DMD2; eight rollout steps, G1/F1; native ZeRO-3/AP | [Native continuation](../../recipes/h3_ar/NATIVE_CONTINUATION.md), engine/LoRA/Adam recovery and [current-Student AV export](h3_ar_native_inference.md) |

[Native workload commands](native_workloads.md) link the complete configuration
and launch sequence. Original H3 video/audio encoding and decoding are available
through [raw AV preparation](h3_audio_video_encode.md) and [AV codecs](h3_audio_video_decode.md).

## Native continuity and output

Fresh-stream recipes retain role state and save at complete-cycle boundaries.
The **two-cycle** presets remain the quick performance-reproduction window,
not a training-length limit. The [performance and correctness summary](performance_validation.md)
links the **100-cycle** official-workflow runs, cycle-50 fresh-process restart,
current-Student export and longer-cycle measurements.

Portable inference weights carry the current Student and its required components.
Training checkpoints additionally retain optimizer, scheduler, EMA, RNG, input
progress and execution versions. Use `train --resume` for training recovery and
`generate --weights` for inference from new inputs.

## Additional model and workflow integrations

| Integration | Model and distillation support | Execution and output |
| --- | --- | --- |
| [Official SD1.5 DMD2](../../recipes/dmd2_sd15/README.md) | Native UNet/CLIP/VAE, single-step DMD2, optional original GAN branch and independent generator/guidance/real-image streams | CPU/CUDA reference; native FSDP2 and optimized Resident/DDP; recovery and current-G image export |
| [Official SDXL DMD2](../../recipes/dmd2_sdxl/README.md) | Dual CLIP, additional conditioning, four-step and initialized single-step399; full or unmerged-LoRA Student | FP32 reference, native role-DDP and budgeted tiny resident planning; recovery and image generation |
| [Official Self-Forcing](../../recipes/self_forcing_official/README.md) | Causal Wan chunk rollout, random gradient exit, native KV cache and DMD/EMA clocks; not W4 LiveWan | Native FSDP2 reference/R6 optimized; budgeted tiny resident policies; recovery and current-G video |
| [LTX-2.5](../../recipes/ltx25/README.md) | Bounded native AV Transformer, Gemma/connectors/codecs; common G1/F5 flow DMD; full parameters or LoRA | FP32 or native BF16 CPU/single-CUDA reference, direct immutable sharing and CPU tiny-fixture planning; restart and AV generation |
| [FLUX.2 klein Base 4B T2I](../../recipes/flux_dmd/README.md) | Flux2/Qwen3/VAE, common flow DMD and unmerged LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; PNG output |
| [Original Qwen-Image T2I](../../recipes/qwen_image_dmd/README.md) | QwenImage/Qwen2.5-VL/VAE, native text masks, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; PNG output |
| [Wan2.2 TI2V-5B T2V](../../recipes/wan22_dmd/README.md) | Wan/UMT5/temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; decoded video |
| [HunyuanVideo-1.5 T2V](../../recipes/hunyuan_video_dmd/README.md) | Hunyuan15, dual text encoders/temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, sharing, local-B1 replicas and CPU tiny-fixture planning; decoded video |

LTX and the four image/video flow families support `fp32` or explicit
`bf16-mixed` compute with FP32 master/Adam/EMA state. The four image/video
families also admit CPU/Gloo and CUDA/NCCL replicas; LTX uses its separate
single-device runtime. See the [sharing, replica and
clone-safety conditions](optimization_capabilities.md#direct-single-cuda-immutable-sharing)
and [compatible environments](model_families.md#choose-a-compatible-integration).
The [export-only preparer](export_inputs.md) encodes new modern-flow prompts/media.

## Integrated official optimization profiles

- **SD1.5:** [FSDP2 primary reference](../../recipes/dmd2_sd15/training.fsdp2.json)
  and [complete optimized Resident/DDP](../../recipes/dmd2_sd15/training.optimized.json);
  FP32 role-DDP is a supplemental reference, not the FSDP2 baseline.
- **Official Self-Forcing:** [FSDP2 reference](../../recipes/self_forcing_official/training.fsdp2-reference.json)
  and [complete R6 optimization](../../recipes/self_forcing_official/training.optimized.json).

The [official workflow guide](official_optimized_training.md) provides launch,
save, resume and output commands. [Native-owner alternatives](native_owner_backends.md)
cover CPU component checkpointing, FP32 role-DDP and post-commit cache release;
[resident planning](resident_owner_training.md) covers tiny snapshot-safe owners.

## Hardware

A100/H100 recipe identities can be inspected with
`distillctl workloads describe --workload Wn --hardware a100|h100`.
Execution uses each recipe's device, dtype, topology and capacity settings.

## Results and verification

- [Performance and source records](performance_validation.md)
- [Official optimized training and recovery](../status/official_optimized_framework.md)
- [Model environments and test lanes](model_families.md#verification-lanes)
- [Public resident lifecycle](resident_owner_training.md)
- [Source and asset licenses](../../THIRD_PARTY.md)
