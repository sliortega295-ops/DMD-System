# Installation and environment selection

DMD-System has a lightweight command layer and optional training/model runtimes.
Install only the group needed for your entry point. Installation never includes
model weights. Inspection does not import model factories or initialize CUDA.

## Common lifecycle

From a source checkout, use an existing compatible Python environment:

```sh
python -m pip install -e '.[train,test]'
distillctl doctor
distillctl workloads list
python -m pytest -q tests/test_training_config.py tests/test_training_end_to_end.py
```

Python3.10 or later is required. The public training/distributed API requires
Torch2.10 or later; the older `native` extra's floor is not permission to run
public FSDP2 on an earlier Torch. For minimum-version **CPU** qualification,
install the CPU Torch wheel first in your chosen test environment:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,test]'
```

For lightweight configuration/catalog tools alone, `python -m pip install .`
is sufficient. Add `.[config]` for YAML. `.[media]` provides NumPy/Pillow;
video-container preparation additionally needs the operator's `ffmpeg` and
`ffprobe`. The [offline tutorial](../../examples/framework/README.md) uses RGB
frames, WAV and text and needs no network model download or video codec.

A wheel supports common Program factories outside the checkout. Native
recipes which load audited root scripts or upstream source still require the
checkout and their documented assets. An installed wheel is not a substitute
for a native source/environment receipt.

## Installed-only native tiny image/video tutorial (CPU, not pretrained)

Use a **new Python3.12 environment**, separate from original-model TF4 and LTX
environments. Install CPU Torch2.10 first, then the built wheel with only
`modern-flow` (or install its pinned dependencies offline from your own approved
cache). Never upgrade a previously qualified environment in place.

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'distillschedule[modern-flow] @ file:///ABSOLUTE/PATH/distillschedule-0.3.0-py3-none-any.whl'
# Work in your own empty project directory, not the source checkout.
RECIPES=$(python -c 'from distillschedule.recipes import directory; print(directory())')
distillctl doctor
distillctl models list
distillctl capabilities list
distillctl config validate "$RECIPES/flux_dmd/training.template.json"
python "$RECIPES/flow_image/create_cpu_fixture.py" image --family flux2-klein-base
python "$RECIPES/wan22_dmd/create_cpu_fixture.py" video
distillctl train image/train.json --cycles 1 --output image-first
distillctl train image/train.json --cycles 3 --resume image-first/checkpoints/cycle-000001 --output image-resumed
distillctl train video/train.json --cycles 3 --output video-trained
```

Use the exact `checkpoint` path returned by the first command if its checkpoint
name differs; resume is a new process and restores the complete cycle, not just
Student weights. The helper-created random actual upstream classes are clearly
**SYNTHETIC_TEST_ONLY**, CPU1/FP32/B1. They include native text/tokenizers/VAE,
real raw RGB data, G1/F5 updates, Adam, LR clocks and EMA; they are not a fake
surrogate network, pretrained quality, GPU or capacity qualification. Generated
image/video fixtures are small by design. The production templates require
your independently licensed, locked local assets and explicit parameter budgets.

Wheel recipes, templates, local asset-lockers and source-only fetch helpers are
installed under the distribution's `share/distillschedule/recipes`. The lazy
accessor above reads that installed inventory, never the cwd, an editable path,
or another checkout. Run helpers directly there; keep user configs/data/outputs
in your own project directory. Copy a template before editing it. The wheel
contains no pretrained weights, raw training features or external source closure.

For export and actual new-prompt/raw-media native preparation after deleting
training assets, follow [export-only inputs](export_inputs.md). Images decode
to uint8 HWC PNG; video to uint8 THWC RGB frame directories (no implicit video
container or frame-rate conversion). RGB frame directories require Pillow;
container video files require **both ffmpeg and ffprobe** on PATH. Neither is
downloaded/installed by data preparation; absence is an error, not a skip.

From a source checkout, the maintainer's offline verifier is:

```sh
python scripts/check_framework_install.py --modern-flow \
  --wheel /project/artifacts/distillschedule-0.3.0-py3-none-any.whl \
  --work-dir /project/acceptance-new --output /project/installed-modern.json
```

It installs the wheel offline into a **new isolated environment** and runs
queries, actual native image/video preparation, three-cycle training, exact
fresh-process continuation, export, asset/checkpoint deletion, export-only new
inputs and current-Student decoded output. Only dependency-only roots are reused
from the invoking pinned environment. No repo imports/PYTHONPATH/editable hooks,
model downloads or dependency mutations are allowed. `--work-dir` must not exist
and must be outside the checkout; it preserves media and subprocess receipts.

## Separate original-model dependency lanes

These groups are **alternatives**, not extras to combine into one environment.
The version differences come from the original source implementations.

| Purpose | Install group | Important boundary |
| --- | --- | --- |
| Common Program training and external model template | `train` (plus `test` for tests) | Model-specific encoders/checkpoints are supplied separately |
| Licensed original Helios helper CPU regression | `train,test,test-helios` | Diffusers0.34.0; Triton definitions may import on Linux/x86, but these CPU tests do not launch kernels |
| Native Wan/source-qualified Self-Forcing regression | `train,test,native-wan` | Diffusers0.35.2; not the Helios0.34 or full-model0.37 lane |
| Full original Helios CPU model and W1/W2/W3 Program references | `train,helios-reference` | Diffusers0.37.1; add `test` separately, not a test-only runtime dependency |
| Maintained original cached H3 AV CPU reference | `train,h3-cached-reference` | Independent small cached-only source closure; not the full DiffSynth/SFT distribution |
| Original Grid H3 CPU source qualification | Dedicated [workflow](../../.github/workflows/h3-grid-native-cpu.yml) | Additional exact upstream source/dependency pins; no pretrained-model validation |
| New native image/video flow CPU reference | `modern-flow` | Torch2.10/Diffusers0.37.1/Transformers5.14.1/PEFT0.18.1; separate Python3.12 lane, not a TF4/LTX upgrade |

Do not resolve a conflict by replacing an admitted native backend or editing its
source hashes. Reuse a compatible environment, or prepare a separate one you
own. Native A100/H100 recipes retain their own CUDA, attention, compiler and
weight identities; the table above is not a cluster installation recipe.

## Check the intended route before loading weights

```sh
distillctl config validate recipes/h3_ar/training.json
distillctl doctor --config recipes/h3_ar/training.json
distillctl train recipes/h3_ar/training.json --dry-run
```

`config validate` checks configuration structure. `doctor` describes installed
packages and declared capability without running factory code. A native dry-run
can require real source/asset metadata and will reject unfilled placeholders.
None of these proves that a large model fits or that a selected GPU kernel is
available. Native execution performs the remaining source, data, topology,
capacity and numerical checks before real work.

For actual runnable CPU reference configurations, use the recipe-specific
reference guides in the [support matrix](support_matrix.md), not the bounded
historical benchmark request. For errors, see [troubleshooting](troubleshooting.md).
