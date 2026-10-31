# Supported models, workflows and execution

DMD-System uses one **`distillctl`** interface for input preparation, training,
checkpoint recovery, strategy selection and Student generation.
Choose a model family and its execution recipe below.

## Common framework capabilities

| Capability | Available functionality |
| --- | --- |
| Configuration and discovery | Model, data, algorithm, execution and trainer sections; lightweight model/workflow/capability discovery |
| Data | Raw images, video, audio and text; prompt-only inputs; preencoded features; independent data streams and sample cursors |
| Training | Persistent Student, Teacher and Fake Score; algorithm-owned native clocks; optimizer, scheduler and EMA updates |
| Save and recover | Complete-cycle, same-topology checkpoints; atomic all-rank shard publication; parameters, optimizer, RNG, data progress and versions |
| Generation | Current-Student evaluation, portable weight export and new-input image/video/audio generation through model-specific codecs |
| Execution | Explicit reference/manual strategies and measured automatic/saved-plan execution through the same runtime |
| Extension | Model/data adapters, distillation Programs and execution capabilities registered independently |

See [training and recovery](training.md), [data preparation](data.md),
[model environments](model_families.md) and [extension interfaces](extensions.md).

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

Fresh-stream recipes retain role state across cycles and save at complete-cycle
boundaries. The original **two-cycle** presets remain available for performance
reproduction. Existing **100-cycle** official-workflow runs include a fresh-process
restart at cycle 50 and current-Student export; their records are linked below.

Portable inference weights carry the current Student and its required components.
Training checkpoints additionally retain optimizer, scheduler, EMA, RNG, input
progress and execution versions. Use `train --resume` for training recovery and
`generate --weights` for inference from new inputs.

## Additional model and workflow integrations

| Integration | Model and distillation support | Execution and output |
| --- | --- | --- |
| Official SD1.5 DMD2 | Native UNet/CLIP/VAE, single-step DMD2, optional original GAN branch and independent generator/guidance/real-image streams | CPU/CUDA reference; native FSDP2 and optimized Resident/DDP recipes; checkpoint recovery and current-G image export |
| Official SDXL DMD2 | Dual CLIP, additional conditioning, four-step and initialized single-step399; full or unmerged-LoRA Student | FP32 reference, explicit role-DDP and budgeted resident planning; complete-state recovery and image generation |
| Official Self-Forcing | Causal Wan chunk rollout, random gradient exit, native KV cache, DMD loss and EMA/update clocks | FSDP2 reference/optimized recipes; budgeted resident policies; checkpoint recovery and current-G video |
| LTX-2.5 | Native AV Transformer, Gemma/connectors and AV codecs; common G1/F5 flow DMD; full parameters or LoRA | CPU/CUDA reference, single-CUDA manual immutable sharing and CPU measured planning; raw-media input, restart and AV generation |
| FLUX.2 klein Base 4B T2I | Flux2/Qwen3/VAE, common flow DMD and unmerged LoRA | CPU/CUDA reference, immutable sharing, post-backward replicas and CPU measured planning; current-Student PNG output |
| Original Qwen-Image T2I | QwenImage/Qwen2.5-VL/VAE, native text masks, common flow DMD and LoRA | CPU/CUDA reference, immutable sharing, post-backward replicas and CPU measured planning; current-Student PNG output |
| Wan2.2 TI2V-5B T2V | Wan/UMT5/temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, single-CUDA manual sharing, explicit local-B1 replica interface and CPU measured planning; decoded video |
| HunyuanVideo-1.5 T2V | Hunyuan15, dual text encoders and temporal VAE, common flow DMD and LoRA | CPU/CUDA reference, single-CUDA manual sharing, explicit local-B1 replica interface and CPU measured planning; decoded video |

Use each [family recipe](model_families.md#choose-a-compatible-integration) with
its pinned environment and local assets. The common [export-only preparer](export_inputs.md)
encodes new prompts/media for the modern-flow image and video adapters.

## Integrated official optimization profiles

- **SD1.5 FSDP2:** `recipes/dmd2_sd15/training.fsdp2.json` — native block/root
  sharding, G1/C10/EMA0, mixed compute precision and FP32 master/Adam state.
- **SD1.5 optimized:** `recipes/dmd2_sd15/training.optimized.json` —
  Resident/DDP execution, fused Adam, activation retention, classifier packing
  and weight-cast scopes.
- **Official Self-Forcing:** `recipes/self_forcing_official/training.fsdp2-reference.json`
  and `training.optimized.json` — native FSDP2 reference and complete R6
  optimization, preserving causal rollout and native updates.

The [official workflow guide](official_optimized_training.md) provides the
launch, save, resume and output commands. [Native-owner policies](native_owner_backends.md)
and [resident planning](resident_owner_training.md) describe additional explicit
checkpoint, lifetime and planning choices.

## Hardware

A100/H100 recipe identities can be inspected with
`distillctl workloads describe --workload Wn --hardware a100|h100`.
Execution uses each recipe's device, dtype, topology and capacity settings.

## Integration records

- [Framework feature overview](../status/FRAMEWORK_MAIN_INTEGRATION_20261030.md)
- [Official optimized 100-cycle, restart and generation records](../status/official_optimized_framework.md)
- [Public resident lifecycle records](../status/FRAMEWORK_PUBLIC_RESIDENT_ACCEPTANCE_20261028.md)
- [Detailed model/backend validation records](validation_records_20261030.md)

The detailed records retain the exact model sizes, source revisions and test
conditions for the integrations listed here.
