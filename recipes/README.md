# Training recipes

Choose a model and workflow, then follow the linked guide to install its
compatible environment and supply model weights and input data. Configuration
files contain placeholders; copy and edit them before launching training.

## Official workflows

| Recipe | Workflow |
| --- | --- |
| [SD1.5 DMD2](dmd2_sd15/README.md) | Single-step DMD2 with optional GAN; FSDP2 reference and Resident/DDP optimized configurations |
| [SDXL DMD2](dmd2_sdxl/README.md) | Four-step or explicitly initialized single-step DMD2 |
| [Official Self-Forcing](self_forcing_official/README.md) | Causal Wan rollout, native KV cache and update schedule |

The [official optimization guide](../docs/guides/native_workloads.md#official_optimized_training)
contains the SD1.5 and Self-Forcing configurations.

## Flow DMD models

| Modality | Recipe |
| --- | --- |
| Image | [FLUX.2 klein Base 4B](flux_dmd/README.md), [Qwen-Image](qwen_image_dmd/README.md) |
| Video | [Wan2.2 TI2V-5B](wan22_dmd/README.md), [HunyuanVideo-1.5](hunyuan_video_dmd/README.md) |
| Audio–video | [LTX-2.5](ltx25/README.md) |

These recipes reuse the flow-DMD workflow with model-specific prediction,
conditioning and codec adapters. The [support matrix](../docs/guides/model_families.md#support_matrix)
records which execution and output paths are available for each model.

## Native benchmark workflows

| ID | Recipe |
| --- | --- |
| W1 | [MiniMax H3 PDMD](h3_fullseq/README.md) |
| W2 | [MiniMax H3 DMD2](h3_ar/README.md) |
| W3 | [Wan2.1 DMD](wan_ar/README.md) |
| W4 | [Wan2.1 SF-DMD](self_forcing/README.md) |
| W5 | [Helios Hier-DMD](helios_nogan/README.md) |
| W6 | [Helios Hier-DMD2](helios_gan/README.md) |

Inspect a workflow by its recipe name, for example
`distillctl workloads describe --workload h3_fullseq`. Use the linked stream or
continuation recipes for continuous training and recovery.

## Before running

Use the [model-family guide](../docs/guides/model_families.md) to select dependencies.
Run `distillctl config validate YOUR_CONFIG.json`, followed by
`distillctl train YOUR_CONFIG.json --dry-run` to inspect launch settings. A dry run
checks configuration, not numerical correctness or GPU capacity.

For a small CPU introduction without model downloads, start with the
[framework tutorial](../examples/framework/README.md). Historical locked profiles
remain available through `distillctl workloads list --legacy` for reproduction.
