# DMD-System 0.3.0

DMD-System brings model adapters, distillation workflows, execution strategies
and the training lifecycle together behind **`distillctl`**.

## Main integration — 2026-10-30

### One complete training workflow

Prepare raw media, prompts or native features; select a recipe and execution
strategy; train continuously; save and resume; generate from the current Student.
Checkpoints retain all role, optimizer, scheduler, EMA, RNG, data-cursor and
version state, with complete-cycle, same-topology recovery.

### Models and distillation algorithms

- Six native workloads: Wan1.3B, Helios No-GAN/GAN, cross-scale LiveWan,
  full-sequence H3 **Grid1000**, and autoregressive H3 audio–video DMD2.
- Official Self-Forcing and SD1.5/SDXL DMD2, preserving native rollout,
  conditioning, loss and update clocks.
- LTX-2.5, FLUX.2 klein Base, Qwen-Image, Wan2.2 and HunyuanVideo model adapters
  sharing the common flow-DMD lifecycle and extension interfaces.

### Reference, direct optimization and planning

Reference and optimized recipes use the same public training command.
The framework includes native FSDP2/Resident/HSDP/ZeRO-3 integrations,
role-pool pipelines, state sharing and reuse, fusion and capture policies.
Users can select an explicit strategy or calibrate, search, save and execute
a plan through the common capability and runtime interfaces.

Original **two-cycle** presets remain available alongside continuous-training
recipes. Existing **100-cycle** official DMD2 and Self-Forcing records include
cycle-50 fresh-process recovery and independent new-input Student generation.

### Data, generation and export

- Image, video, audio, text, prompt-only and preencoded input paths.
- Independent generator, guidance and real-image streams with saved positions.
- Current-Student inference exports with model-native image/video/audio codecs.
- [LiveWan current-Student output](livewan_native_output.md).
- [H3 full-sequence output](h3_grid_inference.md), [H3 AR output](h3_ar_native_inference.md)
  and [raw H3 AV preparation](h3_audio_video_encode.md).

### Installation and extension

Wheels and source packages contain curated recipes, configuration templates,
helpers, guides and third-party notices. Installed recipes are discoverable
through `distillschedule.recipes.directory()`.

Model, algorithm and execution capability extensions have separate registration
interfaces. Family-specific dependency groups retain the original model stacks;
see [installation](installation.md), [model families](model_families.md) and
[extension tutorials](extensions.md).

## Usage and integration records

- [Feature overview](../status/FRAMEWORK_MAIN_INTEGRATION_20261030.md)
- [Supported models and execution](support_matrix.md)
- [Training and checkpoint recovery](training.md)
- [Official optimized workflow records](../status/official_optimized_framework.md)
- [Detailed validation and migration records](validation_records_20261030.md)
- [Licenses and third-party notices](../../THIRD_PARTY.md)
