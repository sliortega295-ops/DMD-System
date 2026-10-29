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

A wheel supports common Program factories outside the checkout. The official
DMD2/Self-Forcing factories also work installed-only with their explicit pinned
external source trees and local assets; they do not need the framework checkout
at runtime. Legacy native recipes which load audited root scripts still require
that checkout. An installed wheel is not a substitute for a native
source/environment receipt.

## Installed documentation and source references

Relative navigation from public guides and recipes stays within the
installed `share/distillschedule` documentation. A reference labeled
**source checkout:** gives a path relative to the matching DMD-System source
checkout, rather than a file under that installed directory. Source-only code,
CI definitions and archived result summaries are not duplicated as documentation
payloads in the wheel.

Browse the [DMD-System source repository](https://github.com/sliortega295-ops/DMD-System)
at the release or source revision recorded for your build/run. For an unpublished
development build, use its original local checkout; the default `main` branch
need not contain that build's files. Source references preserve their original
paths and, when present, section anchors.

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

## Installed official DMD2 and Self-Forcing acceptance (CPU, not pretrained)

Use an **already-qualified original-model Python environment** with the
dependencies declared by `dmd2` and `self-forcing-official`: CPU Torch2.10,
Diffusers0.37.1, Transformers4.51.3 and PEFT0.17.1, plus their respective
preparation dependencies. This compatible pair is deliberately separate from
the Transformers5/modern-flow and LTX stacks. The verifier checks actual imported
dependency versions and origins; it does not install or upgrade dependencies.

Supply both local upstream source trees explicitly:

- DMD2: `tianweiy/DMD2`, revision
  `8d8fa55633d47cfb81bbc7a892e7248f9518763f`, under its original
  **CC-BY-NC-SA-4.0** license, not this framework's Apache license.
- Official Self-Forcing: `guandeh17/Self-Forcing`, revision
  `33593df3e81fa3ec10239271dd2c100facac6de1`, under its original Apache-2.0
  license. This is not the separate LiveWan adapter.

The installed pinned-source loaders validate their exact original file hashes.
Missing/changed files are errors, not an invitation to replace the source locks.
External sources are neither downloaded nor bundled/transplanted into the wheel.
System Git is not required by the installed framework's acceptance path.

From a framework checkout, invoke the existing maintainer verifier with a built
wheel and a **new directory outside that checkout**:

```sh
python scripts/check_framework_install.py --official-workflows \
  --dmd2-source /project/sources/DMD2 \
  --self-forcing-source /project/sources/Self-Forcing \
  --wheel /project/artifacts/distillschedule-0.3.0-py3-none-any.whl \
  --work-dir /project/official-acceptance-new \
  --output /project/installed-official.json
```

Omit `--wheel` only to build from the current framework source using its
already-installed build backend. The helper installs **only that framework
wheel**, offline and without dependencies, in a new isolated environment. It
reuses dependency-only roots from the invoking environment, removes inherited
`PYTHONPATH`/`PYTHONHOME`, and runs outside the checkout. Every fresh CLI/fixture
child uses `-I` and checks its actual framework/runtime/transitive module origins
and installed recipe inventory before and after execution. Network connections
are refused. Neither `--official-workflows --modern-flow` nor a missing source
argument is accepted; these errors precede dependency/build/model work.

The opt-in path exercises **all three** real native tiny workflows, SD1.5,
SDXL and official causal Self-Forcing, using the existing installed
`dmd2_sd15/create_cpu_fixture.py` and
`self_forcing_official/create_cpu_fixture.py` builders. It verifies:

- Query/catalog/config commands without importing Torch/model implementations,
  initializing CUDA, opening weights or reading upstream model code.
- Public data preparation/validation and real native losses, backward and Adam
  updates. Three complete cycles use an explicit tiny **ratio2** contract: six
  native iterations, not six complete cycles. Independent stream sampling and
  input-progress clocks must match the uninterrupted run. The installed SD
  fixture uses distinguishable short prompts that survive its eight-token
  limit; the verifier checks the actual native iteration's per-role input
  tensor digests against independently sampled prepared records, not IDs alone.
- One-cycle save, fresh-process resume to three cycles, and strict equality of
  **every packed checkpoint field**, including native/outer RNG, models,
  optimizers, input progress, versions and the actually configured schedulers
  or EMA. DMD2 has both LR schedulers and **no EMA**; Self-Forcing has native
  EMA (tiny-test start step2, two updates) and **no LR scheduler**. Production
  recipe ratios and EMA defaults are not changed.
- Export of the **updated current Student**, not EMA/base weights, followed by
  fresh-process generation on genuinely different native conditions. Full
  generated tensors, decoded pixels and written PNGs must match that current
  Student exactly.
  For this CPU-only output reference, parameter values are checked exactly and
  rematerialized into fresh storage before inference, matching the export
  loader. This removes safetensor-view alignment as a source of one-ulp CPU
  reduction differences; no training state, RNG or comparison tolerance changes.

**Asset-retirement boundary:** new SD prompt conditions are prepared with the
actual local tokenizer bundle before its deletion. New Self-Forcing conditions
are explicit random native embeddings, **not native T5 text-encoder outputs**.
After preparing these new inputs, the verifier deletes all original model
bundles, raw/prepared training inputs, training configs, runs and checkpoints
before the final fresh generation. Only the export, new prepared conditions and
the exact external source integrations are needed for inference. This does not
claim SD export-only tokenization or Self-Forcing new-text/T5 preparation.

Success reports `PASS_INSTALLED_OFFICIAL_WORKFLOWS_CPU` with per-workflow class,
source/recipe hash, full-state, sampling, clock, export and child-origin
receipts. Media are random tiny actual native outputs: SD16×16 RGB images and
81-frame32×32 Self-Forcing videos (the native21-latent-frame contract). This is
**CPU integration evidence only**, not pretrained weights, quality, GPU,
capacity, long-run stability or performance evidence. Those remain `NOT_RUN`.

The matching mandatory-when-enabled integration test is:

```sh
DMD_OFFICIAL_INSTALLED_CPU=1 \
DMD2_SOURCE_ROOT=/project/sources/DMD2 \
SELF_FORCING_SOURCE_ROOT=/project/sources/Self-Forcing \
DMD_OFFICIAL_INSTALLED_WHEEL=/project/artifacts/distillschedule-0.3.0-py3-none-any.whl \
python -m pytest -q tests/test_framework_official_install.py
```

Without the opt-in, only pure admission/provenance tests run; the native test is
explicitly skipped. Once enabled, missing sources/dependencies or missing real
evidence fail the test rather than becoming skips. Without either opt-in mode,
the existing tiny-media verifier behavior is unchanged. `--help` remains a
stdlib-only query and never installs, imports Torch or loads sources/assets.

## Separate original-model dependency lanes

These groups select independently qualified stacks, not an install-everything
environment. The version differences come from original source implementations.
Only the compatible `dmd2`/`self-forcing-official` pair above is combined for that
specific installed acceptance; do not combine it with modern-flow or LTX.

| Purpose | Install group | Important boundary |
| --- | --- | --- |
| Common Program training and external model template | `train` (plus `test` for tests) | Model-specific encoders/checkpoints are supplied separately |
| Licensed original Helios helper CPU regression | `train,test,test-helios` | Diffusers0.34.0; Triton definitions may import on Linux/x86, but these CPU tests do not launch kernels |
| Native Wan/source-qualified Self-Forcing regression | `train,test,native-wan` | Diffusers0.35.2; not the Helios0.34 or full-model0.37 lane |
| Full original Helios CPU model and W1/W2/W3 Program references | `train,helios-reference` | Diffusers0.37.1; add `test` separately, not a test-only runtime dependency |
| Maintained original cached H3 AV CPU reference | `train,h3-cached-reference` | Independent small cached-only source closure; not the full DiffSynth/SFT distribution |
| Original Grid H3 CPU source qualification | Dedicated workflow (source checkout: `.github/workflows/h3-grid-native-cpu.yml`) | Additional exact upstream source/dependency pins; no pretrained-model validation |
| New native image/video flow CPU reference | `modern-flow` | Torch2.10/Diffusers0.37.1/Transformers5.14.1/PEFT0.18.1; separate Python3.12 lane, not a TF4/LTX upgrade |
| Official DMD2 SD1.5/SDXL | `dmd2` | Explicit original noncommercial source; Torch2.10/Diffusers0.37.1/Transformers4.51.3/PEFT0.17.1 |
| Official causal Self-Forcing | `self-forcing-official` | Explicit official source/native conditions; compatible TF4 original-model stack, not LiveWan |

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
