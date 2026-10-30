# DMD-System framework integration — 2026-10-30

DMD-System brings model adapters, distillation workflows, execution strategies
and the training lifecycle together behind **`distillctl`**.

## User workflow

**Choose a model → choose a distillation workflow → prepare inputs → select a
strategy → train → save and resume → generate and export.**

| Function | Public entry |
| --- | --- |
| Discover models, algorithms and strategies | `distillctl models list`, `algorithms list`, `strategies list` |
| Prepare media, prompts or native features | `distillctl data prepare` and model-specific recipe preparers |
| Run a reference or optimized configuration | `distillctl train` |
| Continue from a complete-cycle checkpoint | `distillctl train --resume` |
| Generate from the current Student | `distillctl evaluate`, `distillctl generate --weights` |
| Publish inference weights | `distillctl export` |
| Calibrate, search and execute a saved plan | The common planning commands and `execution.mode: plan` |

## Model and workflow integrations

| Integration | Included functionality |
| --- | --- |
| Wan1.3B, Helios No-GAN and GAN | Native hierarchical rollout, role updates, optimization presets, fresh-input streams and checkpoint/output owners |
| LiveWan / cross-scale Self-Forcing | 14B Teacher → 1.3B Student, two-pool C51 pipeline, cycle-boundary save/recovery and Student export |
| H3 full-sequence and H3 AR | Separate full-sequence Grid1000 and autoregressive previous/current AV workflows, native state and codec interfaces |
| Official Self-Forcing | Native causal block rollout, KV cache, random gradient exit, role updates and EMA; reference and optimized FSDP2 configurations |
| Official DMD2 SD1.5 / SDXL | Native model, loss and update clocks; prompt, guidance and image streams; SD1.5 FSDP2 and optimized Resident/DDP configurations |
| LTX-2.5 | Native AV model, text conditioning and codecs; common flow-DMD lifecycle and LoRA |
| FLUX.2 klein Base / Qwen-Image | Native image model, text encoder and VAE adapters; flow DMD, LoRA and shared frozen-state policies |
| Wan2.2 TI2V-5B T2V / HunyuanVideo-1.5 T2V | Native video/text/VAE adapters; flow DMD, LoRA, prepared inputs and decoded Student output |

The [model-family guide](../guides/model_families.md) provides installation
and recipe selection. The [capability matrix](../guides/support_matrix.md)
records each available backend, precision and model configuration.

## Training state and execution

- Complete-cycle checkpoints preserve role parameters, optimizer/scheduler/EMA,
  RNG, data cursors and update versions, with same-topology recovery.
- Portable exports contain the current Student and its inference components.
- Explicit strategies and automatic plans use the existing runtime and its
  capability checks, parameter ownership and cache-version rules.
- New model/data adapters, workflows and physical capabilities register through
  separate extension points; see the [extension guide](../guides/extensions.md).

## Existing integration records

Wan has public world16 continuation/restart and independent new-prompt video
records. Official optimized DMD2 and Self-Forcing each completed 100 cycles,
a fresh-process restart at cycle 50, Student export and independent new-input
image/video generation. See the [official workflow records](official_optimized_framework.md).

LiveWan C51's same-source C1 checkpoint and fresh-process C2 **native
training/checkpoint** completed on 16 ranks, with 16 persistent checkpoint
shards. The compact [recovery record](../../artifacts/framework/20261030/livewan_native_recovery_scope.json)
retains the original source and component outcome. Model-specific tests and
validation records remain available through the capability matrix.
