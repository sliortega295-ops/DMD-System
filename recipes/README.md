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

The [official optimization guide](../docs/guides/official_optimized_training.md)
contains the measured SD1.5 and Self-Forcing configurations.

## Flow DMD models

| Modality | Recipe |
| --- | --- |
| Image | [FLUX.2 klein Base 4B](flux_dmd/README.md), [Qwen-Image](qwen_image_dmd/README.md) |
| Video | [Wan2.2 TI2V-5B](wan22_dmd/README.md), [HunyuanVideo-1.5](hunyuan_video_dmd/README.md) |
| Audio–video | [LTX-2.5](ltx25/README.md) |

These recipes reuse the flow-DMD workflow with model-specific prediction,
conditioning and codec adapters. The [support matrix](../docs/guides/support_matrix.md)
records which execution and output paths are available for each model.

## Native benchmark workflows

| ID | Recipe |
| --- | --- |
| W1 | [Wan1.3B hierarchical No-GAN](wan_ar/README.md) |
| W2 | [Helios14B No-GAN](helios_nogan/README.md) |
| W3 | [Helios14B GAN](helios_gan/README.md) |
| W4 | [LiveWan cross-scale Self-Forcing](self_forcing/README.md) |
| W5 | [H3 full-sequence AV](h3_fullseq/README.md), with approximate Grid1000 sampling |
| W6 | [H3 autoregressive AV](h3_ar/README.md), with continuous sigma sampling |

Use `distillctl workloads list` and `describe --workload Wn` to inspect these
six protocols; `distillctl models list` also lists the additional models above.
The default native `training.json` files select two-cycle performance presets.
For continuous training and recovery, use the stream/continuation recipe linked
from each guide. W5's native optimized preset still has an unresolved numerical
failure; see [native presets](../docs/guides/native_presets.md).

## Before running

Use the [model-family guide](../docs/guides/model_families.md) to select dependencies.
Run `distillctl config validate YOUR_CONFIG.json`, followed by
`distillctl train YOUR_CONFIG.json --dry-run` to inspect launch settings. A dry run
checks configuration, not numerical correctness or GPU capacity.

For a small CPU introduction without model downloads, start with the
[framework tutorial](../examples/framework/README.md). Historical locked profiles
remain available through `distillctl workloads list --legacy` for reproduction.
