# DMD-System

An extensible framework for **video and audio-video distribution-matching
distillation**. DMD-System connects data preparation, model and algorithm
adapters, continuous training, checkpoint recovery and execution planning.

Model mathematics stays in the algorithm. Placement, state retention and
execution order are selected explicitly or from measured capabilities, then
executed by the existing runtime. The package is `distillschedule`; the common
command is `distillctl`.

## Install

```sh
python -m pip install -e '.[train,test]'
distillctl doctor
distillctl workloads list
```

Choose a compatible existing environment before installing. `pip install .`
provides lightweight JSON configuration and six-workload discovery;
`.[config]` adds YAML, `.[media]` adds NumPy/Pillow and `.[train]` adds public
training dependencies. Video-file decoding uses existing `ffmpeg`/`ffprobe`.
Original distributed recipes retain their pinned native environments/assets.

## A complete offline training example

This example uses actual RGB frame files, WAV audio and text, tiny randomly
initialized models and real DMD gradient/AdamW updates. It demonstrates the
lifecycle; it is not pretrained generation or a quality benchmark.

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
```

Configuration has five sections: **model, data, algorithm, execution, trainer**.
Data and adapter paths are explicit, with no implicit model downloads or
synthetic-data fallback. Read [the tutorial](examples/framework/README.md),
[data guide](docs/guides/data.md) and [training/recovery](docs/guides/training.md).

## Six original workloads

| ID | Protocol | Native recipe |
| --- | --- | --- |
| W1 | Wan1.3B hierarchical AR No-GAN DMD | [Wan](recipes/wan_ar/README.md) |
| W2 | Helios14B No-GAN, G1/C5/EMA5 | [Helios No-GAN](recipes/helios_nogan/README.md) |
| W3 | Helios14B DMD + GAN, G1/C5/EMA1 | [Helios GAN](recipes/helios_gan/README.md) |
| W4 | Self-Forcing / LiveWan, 14B Teacher → 1.3B Student | [C51 Self-Forcing](recipes/self_forcing/README.md) |
| W5 | Full-sequence audio-video H3, approximate Grid1000 | [Full-sequence H3](recipes/h3_fullseq/README.md) |
| W6 | Original DiffSynth AR audio-video H3, continuous sigma | [H3 AR](recipes/h3_ar/README.md) |

Each recipe contains a public `training.json` that lowers to the original
runtime and source/capacity/input gates. Saved selections actually reach that
runtime. Native execution requires a checkout and qualified local assets;
existing APIs retain their actual cycle and lifecycle limits. Consult the
[native guide](docs/guides/native_workloads.md) and
[support matrix](docs/guides/support_matrix.md).

```sh
distillctl workloads describe --workload W4
distillctl workloads describe --workload W6 --hardware h100
distillctl config validate recipes/helios_nogan/training.json
distillctl train recipes/helios_nogan/training.json --dry-run
```

Replace explicit recipe asset/plan placeholders before admission. Discovery
does not import models or initialize CUDA. Older immutable profiles remain
available with `distillctl workloads list --legacy`.

## Add a model or workflow

A model factory constructs named roles, optimizers and optional state; an
encoder prepares native features. An algorithm factory describes one cycle
with existing DMD hooks or `Program`/`Call`. The trainer owns continuous data
progress, metrics and complete-cycle save/recovery.

Start with [the external model template](examples/framework/newmodel.py) and
[extension guide](docs/guides/extensions.md). Adding that model needs no solver
branch. Loss, history, GAN and update logic remain algorithm-owned; supported
backends and optimization capabilities are declared separately.

## Strategies and automatic planning

Use `execution.mode: reference` first, or supply an admitted manual policy.
`auto` measures supported candidates, runs the shared constrained search and
executes its selection after the original reference gate. `plan` replays a
source-bound saved selection. Training records the implementation, executed
operations and logical versions.

Existing advanced entrypoints remain available:

```sh
distillctl workloads native program --help
distillctl workloads native validate --task configs/distillschedule/original_h3_ar/w6_reference.json
```

See [architecture](docs/architecture/framework.md),
[native planning](docs/runbooks/native_planner_workflow.md) and
[role pools](docs/workloads/open_role_pools.md). Distributed backends retain
their independent admission; Ray is not required.

## Development

```sh
python -m pytest -q tests/test_data_*.py tests/test_training_*.py
python scripts/check_framework_install.py
python scripts/check_wheel_install.py
distillctl workloads verify-catalog --repo-root .
python scripts/distillschedule/check_best_known_integration.py
```

CPU checks cover actual media, real autograd, optimizer state, fresh-process
restart, selected-plan execution and an external model. Hardware-qualified
records remain in the [source index](docs/status/GITHUB_SOURCE_SNAPSHOT_20261019.md)
and [H100 index](docs/status/H100_SOURCE_SNAPSHOT_20261019.md); this integration
does not remeasure large-model performance or quality.

## Organization

```text
src/distillschedule/  data, model interfaces, algorithms, training, backends, runtime and planner
recipes/             six original-model requests and guides
examples/            tutorials and new-model templates
runtime/             packaged, audited compatibility mechanisms
configs/             typed tasks, backend fragments and locked historical profiles
scripts/             entrypoints, native launchers and verification
tools/               standalone preparation and profiling tools
benchmarks/          developer kernel/state canaries
docs/                architecture, guides, contracts, runbooks and evidence
tests/               lifecycle, protocol, source and distributed regressions
third_party/         pinned upstream source and original notices
```

[Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) ·
[License](LICENSE) · [Third-party notices](THIRD_PARTY.md)
