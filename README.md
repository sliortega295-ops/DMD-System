# DMD-System

A unified framework for **image, video and audio–video distribution-matching
distillation**. Choose a model, a distillation workflow and an execution
policy; use one lifecycle for data preparation, real continuous
updates, checkpoint recovery, generation and inference export.

The Python package is **`distillschedule`** and the user command is
**`distillctl`**. Algorithm mathematics and native iteration clocks remain
algorithm-owned. Physical placement, state lifetimes and legal execution order
use the existing Program, distributed backends and measured planner—not another
trainer or a hardcoded model-name winner.

**Current integration:** the six native workloads, official Self-Forcing and
SD1.5/SDXL DMD2, plus LTX, FLUX, Qwen-Image, Wan2.2 and HunyuanVideo adapters
are shipped together. See the [framework feature overview](docs/status/FRAMEWORK_MAIN_INTEGRATION_20261030.md)
and choose a recipe below.

## Install and discover

```sh
python -m pip install -e '.[train]'
distillctl doctor
distillctl models list
distillctl algorithms list
distillctl strategies list
distillctl capabilities list
distillctl workloads list
```

A base `pip install .` supports lightweight JSON configuration and discovery
without Torch, CUDA initialization or model downloads. `.[config]` adds YAML;
`.[media]` adds NumPy/Pillow; `.[train]` adds the common training dependencies.
Existing FFmpeg/FFprobe decode local video containers. Models and datasets are
always explicit operator-supplied assets.

**Use separate model environments.** SD/DMD2, official Self-Forcing, original
Wan/LiveWan, Helios helpers and LTX/Gemma have distinct pinned dependency lanes.
Do not install all extras together. See [installation and model families](docs/guides/model_families.md).

## A complete offline training lifecycle

This executable example uses RGB frame files, WAV audio, text and tiny models
to walk through real gradient/AdamW updates, recovery and generation.

```sh
DEMO="$(mktemp -d -t dmd-system.XXXXXX)"
python -m distillschedule.examples.tiny_av --output "$DEMO" --samples 6

distillctl data prepare "$DEMO/raw/manifest.json" --output "$DEMO/prepared"
distillctl doctor --config "$DEMO/train.json"
distillctl train "$DEMO/train.json"

CP="$DEMO/run/checkpoints/cycle-000004"
distillctl train "$DEMO/train.json" --resume "$CP" --cycles 6 --output "$DEMO/resumed"
distillctl evaluate "$DEMO/train.json" --checkpoint "$CP" --output "$DEMO/generated"
distillctl export "$DEMO/train.json" --checkpoint "$CP" --output "$DEMO/exported"

# A genuinely new seventh record, not part of the six-record training input.
NEW="$DEMO/new-inputs"
python -m distillschedule.examples.tiny_av --output "$NEW" --samples 7
distillctl data prepare "$NEW/raw/manifest.json" --output "$NEW/prepared"
distillctl generate "$DEMO/train.json" --weights "$DEMO/exported" \
  --prepared "$NEW/prepared/manifest.json" --sample-index 6 --output "$DEMO/new-generated"
```

Configuration has five independent sections: **model, data, algorithm,
execution, trainer**. Data can be raw images/video/audio/text, prompt-only, or
already encoded native features; model-specific encoders prepare features
offline. Independent generator/guidance/real-image streams keep their own
sample progress. Missing conditioning never falls back to synthetic tensors.

A training checkpoint saves actual roles, optimizer/scheduler/EMA state, native
and public clocks, RNG, input progress and versions. Recovery is at complete
cycle boundaries with the same topology; it is not weight-only initialization.
Portable inference export contains current Student/Generator weights and the
required owned inference components, **not** resumable optimizer/data state.
`generate --weights` uses new explicit inputs and does not need old training
artifacts. Outputs are actual images, video frames, WAV audio or AV where the
adapter supplies real decoders; explicit native latents are never called media.

Read [data formats](docs/guides/data.md), [training and recovery](docs/guides/training.md)
and [the end-to-end tutorial](examples/framework/README.md).

## Models and algorithms

| Integration | Reference workflow | Entry and scope |
| --- | --- | --- |
| SD1.5 | Official DMD2 single-step; optional original GAN branch | [Recipe and local image/prompt intake](recipes/dmd2_sd15/README.md) |
| SDXL | Official DMD2 four-step and explicitly initialized single-step; dual CLIP/added conditions | [Recipe](recipes/dmd2_sdxl/README.md) |
| Official Self-Forcing Wan | Native causal chunk rollout, random gradient exit, KV cache, DMD loss and EMA/update clocks | [Recipe](recipes/self_forcing_official/README.md) |
| LTX-2.5 | Actual LTX AV model/codecs/Gemma with reusable No-GAN flow DMD; full parameters or unmerged LoRA | [Recipe; start with LoRA](recipes/ltx25/README.md) |
| FLUX.2 klein Base 4B T2I | Native Flux2/Qwen3/VAE; explicit flow DMD with unmerged LoRA | [Recipe](recipes/flux_dmd/README.md) |
| Original Qwen-Image T2I | Native QwenImage/Qwen2.5-VL/VAE and masked text conditions; explicit flow DMD | [Recipe](recipes/qwen_image_dmd/README.md) |
| Wan2.2 TI2V-5B T2V subpath | Native Wan/UMT5/VAE; explicit flow DMD, not A14B experts or I2V | [Recipe](recipes/wan22_dmd/README.md) |
| HunyuanVideo-1.5 T2V | Native Hunyuan15, dual text and temporal VAE; explicit flow DMD | [Recipe](recipes/hunyuan_video_dmd/README.md) |

Start with a family recipe and choose its execution policy; data
preparation, `train`, `train --resume`, `export` and `generate` use the same
lifecycle. The common-flow LoRA recipes also expose
direct single-CUDA immutable sharing; FLUX/Qwen have separately qualified
post-backward replicas. [Model families](docs/guides/model_families.md) explains
the choices; the [support matrix](docs/guides/support_matrix.md#additional-model-and-workflow-integrations)
lists the available model, data and execution paths. Supply local checkpoints
and use the recipe's model-specific environment. Original DMD2 and LTX sources
use their respective licenses.

Official Self-Forcing is **not** the existing W4 LiveWan cross-scale workflow.

**Official distributed workflows:** SD1.5 provides a [native FSDP2 reference](docs/guides/dmd2_fsdp2_reference.md) and an optimized Resident/DDP configuration. Official Self-Forcing provides native FSDP2 reference and optimized configurations. [Choose and launch a recipe](docs/guides/official_optimized_training.md) through `distillctl train`. Each workflow keeps its own loss, sampling and update rules.

## Six preserved native workloads

| ID | Distinct protocol | Recipe |
| --- | --- | --- |
| W1 | Wan1.3B hierarchical No-GAN DMD | [Wan](recipes/wan_ar/README.md) |
| W2 | Helios14B No-GAN, G1/C5/EMA5 | [Helios No-GAN](recipes/helios_nogan/README.md) |
| W3 | Helios14B DMD + GAN, G1/C5/EMA1 | [Helios GAN](recipes/helios_gan/README.md) |
| W4 | LiveWan cross-scale Self-Forcing, 14B Teacher → 1.3B Student | [LiveWan](recipes/self_forcing/README.md), [current-G CPU lifecycle](recipes/self_forcing/CPU_PUBLIC.md) |
| W5 | Full-sequence H3 audio–video, explicitly approximate Grid1000 | [Full-sequence H3](recipes/h3_fullseq/README.md) |
| W6 | DiffSynth autoregressive previous/current audio–video, continuous sigma | [H3 AR](recipes/h3_ar/README.md) |

Use the [native guide](docs/guides/native_workloads.md) for continuous-training
recipes, checkpoint recovery and output commands, and the
[support matrix](docs/guides/support_matrix.md) to select a backend.

```sh
distillctl workloads describe --workload W4
distillctl workloads describe --workload W6 --hardware h100
distillctl config validate recipes/helios_nogan/training.json
distillctl train recipes/helios_nogan/training.json --dry-run
```

Replace recipe asset placeholders before execution. The old five locked
profiles remain available with `distillctl workloads list --legacy`.
Existing A100/H100 implementations and results are indexed in the
[source overview](docs/status/GITHUB_SOURCE_SNAPSHOT_20261019.md) and
[H100 overview](docs/status/H100_SOURCE_SNAPSHOT_20261019.md).

## Execution and extension

`reference` is a usable starting point; automatic planning is optional.
`manual` selects an admitted implementation; `auto` measures real candidates
and runs the existing constrained solver; `plan` executes that saved,
source-bound selection. The implementation and actual operation/version/effect
receipts are recorded. Layout and cache-version checks bind the saved plan to
the execution. The runtime has no Ray dependency.

See [physical capabilities](docs/guides/optimization_capabilities.md),
[framework architecture](docs/architecture/framework.md) and
[native planning](docs/runbooks/native_planner_workflow.md).

Three independent extension seams are supported:

1. **Model/data adapter:** checkpoint, prediction meaning, trainables, encoders,
   modalities and generation; reuse a compatible algorithm.
2. **Workflow:** native rollout/loss/update dependencies through existing DMD
   hooks or Program, reusing models and backends.
3. **Physical capability:** actual execution, applicability, effect counters and
   measured costs; no model-name branch in the general solver.

Start with [the new-model template](examples/framework/newmodel.py),
[the stateful-workflow template](examples/framework/newworkflow.py) and
[extension contracts](docs/guides/extensions.md). Installed packages may
advertise model/algorithm entry points; discovery reads metadata without
executing their trusted Python code.

## Development and licenses

Run the [CPU verification lanes](docs/guides/model_families.md#verification-lanes)
in their separate environments. The package's installed-wheel acceptance runs
outside the checkout and verifies fresh-process training/recovery, a measured
saved plan and genuinely new-input decoded generation after removing training
artifacts. See [the current integration receipt](docs/status/FRAMEWORK_MODEL_EXPANSION_CPU_20261023.md)
for the corresponding validation records.

Contributions must retain semantics and include real update/recovery or effect
proof as appropriate; see [CONTRIBUTING](CONTRIBUTING.md). Framework-authored
code is Apache-2.0. [Third-party notices](THIRD_PARTY.md), original source and
model/data licenses remain authoritative.
