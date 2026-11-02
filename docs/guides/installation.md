# Installation

## Install the framework

Use Python 3.10+ and a compatible PyTorch installation. From a checkout:

```sh
python -m pip install -e '.[train]'
distillctl doctor
distillctl models list
distillctl workloads list
distillctl capabilities list
```

For catalog and configuration tools alone, install `distillschedule` without
training extras. These commands do not download weights or initialize CUDA.
For CPU training, install a CPU Torch wheel first. CUDA installations use the
PyTorch build appropriate for the device and driver.

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,test]'
```

## Choose a model environment

Install the relevant dependency group rather than combining every model stack.
The [model-family guide](model_families.md) and each recipe list the matching
versions and environment files. Public distributed training uses Torch 2.10+.

| Workflow | Dependency group |
| --- | --- |
| Common Program training | `train` |
| Helios references and W1–W3 | `helios-reference` |
| Wan/native Self-Forcing helpers | `native-wan` |
| H3 cached audio–video | `h3-cached-reference` |
| Official SD1.5 / SDXL DMD2 | `dmd2` |
| Official causal Self-Forcing | `self-forcing-official` |
| FLUX, Qwen-Image, Wan2.2, HunyuanVideo | `modern-flow` |
| LTX-2.5 | See [LTX recipe](../../recipes/ltx25/README.md) |

H3 Grid1000 uses its source-qualified environment (source checkout: `.github/workflows/h3-grid-native-cpu.yml`).

For example, `python -m pip install -e '.[dmd2]'` selects the DMD2 stack.
DMD2 and official Self-Forcing use the Transformers 4 lane; modern-flow uses
Transformers 5. Keep these environments separate. Reproduction recipes retain
locked source, CUDA and attention-library versions.

## Prepare inputs and train

Model weights can be prepared independently of training. Follow the selected
[recipe](../../recipes/README.md) for its local checkpoint and encoder assets.
Use [data preparation](data.md) to encode media or prompts once; training reads
those features. DMD2 retains its upstream CC-BY-NC-SA-4.0 license; external model
and source terms are listed in [third-party notices](../../THIRD_PARTY.md).

Start with the [complete CPU tutorial](../../examples/framework/README.md), then
use the same lifecycle with a model recipe:

```sh
distillctl config validate /project/train.json
distillctl doctor --config /project/train.json
distillctl train /project/train.json --dry-run
distillctl train /project/train.json
```

Select a reference or an explicit optimized strategy directly. Calibration and
search are optional; see [execution capabilities](optimization_capabilities.md).
[Training, recovery and output](training.md) covers the remaining commands.

## Use installed recipes

A wheel includes recipes, helpers, environment templates and this documentation:

```sh
RECIPES=$(python -c 'from distillschedule.recipes import directory; print(directory())')
distillctl config validate "$RECIPES/flux_dmd/training.template.json"
```

Copy templates to your project before editing; keep datasets and runs there.
Native recipes that invoke audited root workers use a matching source checkout.
Source-only references point to the [repository](https://github.com/sliortega295-ops/DMD-System),
not files inside the installed documentation directory. For an unpublished build,
use its matching source checkout.

## Developer checks

Install `.[test]` in the relevant model environment. The affected CI workflow
and [contribution guide](https://github.com/sliortega295-ops/DMD-System/blob/main/CONTRIBUTING.md) list focused checks.
The source-only `scripts/check_framework_install.py` verifies wheel installation,
real tiny-model updates, new-process recovery and output outside the checkout;
its `--help` lists the common, modern-flow and official-workflow lanes.
