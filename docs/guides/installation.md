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
