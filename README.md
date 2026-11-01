# DMD-System

Distribution-matching distillation for image, video and audio–video models.
Train, resume, export and generate through the `distillctl` command; choose a
model-specific execution strategy without changing the training algorithm.
The Python package is named `distillschedule`.

[Documentation](docs/README.md) · [Recipes](recipes/README.md) ·
[Supported configurations](docs/guides/support_matrix.md) ·
[Optimizer Skill](docs/guides/optimizer_skill.md) · [Contributing](CONTRIBUTING.md)

**Research software:** model support, hardware requirements and validation vary
by recipe. Start with the [support matrix](docs/guides/support_matrix.md) and its
linked validation records before a large training run.

## Get started

Use Python 3.10 or later in a dedicated environment. From a source checkout:

```sh
git clone https://github.com/sliortega295-ops/DMD-System.git
cd DMD-System
python -m pip install -e '.[train,media]'
distillctl doctor
distillctl models list
```

This installs the common training dependencies. For a particular pretrained
model, use its [recipe](recipes/README.md) and
[model-family environment](docs/guides/model_families.md); some dependency
versions conflict and need separate environments. Weights and datasets are
supplied separately. The public training API requires PyTorch 2.10 or later.

### Try the training lifecycle on CPU

The following example needs no model download. It trains small random networks
on generated media to demonstrate the API, not pretrained generation quality.

```sh
DEMO="$(mktemp -d -t dmd-system.XXXXXX)"
python -m distillschedule.examples.tiny_av --output "$DEMO" --samples 6
distillctl data prepare "$DEMO/raw/manifest.json" --output "$DEMO/prepared"
distillctl train "$DEMO/train.json"
CP="$DEMO/run/checkpoints/cycle-000004"
distillctl train "$DEMO/train.json" --resume "$CP" --cycles 6 --output "$DEMO/resumed"
distillctl export "$DEMO/train.json" --checkpoint "$CP" --output "$DEMO/exported"
```

Continue with the [end-to-end tutorial](examples/framework/README.md) for
new-input generation, or choose a model below. See
[installation](docs/guides/installation.md) for CPU-only dependencies and
[troubleshooting](docs/guides/troubleshooting.md) for common setup issues.

## Models and workflows

| Workflow | Recipes |
| --- | --- |
| Official DMD2 | [SD1.5](recipes/dmd2_sd15/README.md), [SDXL](recipes/dmd2_sdxl/README.md) |
| Official Self-Forcing | [Causal Wan](recipes/self_forcing_official/README.md) |
| Flow DMD: images | [FLUX.2 klein Base 4B](recipes/flux_dmd/README.md), [Qwen-Image](recipes/qwen_image_dmd/README.md) |
| Flow DMD: video | [Wan2.2 TI2V-5B](recipes/wan22_dmd/README.md), [HunyuanVideo-1.5](recipes/hunyuan_video_dmd/README.md) |
| Flow DMD: audio–video | [LTX-2.5](recipes/ltx25/README.md) |
| Hierarchical video DMD | [Wan1.3B / W1](recipes/wan_ar/README.md), [Helios No-GAN / W2](recipes/helios_nogan/README.md), [Helios GAN / W3](recipes/helios_gan/README.md) |
| Cross-scale Self-Forcing | [LiveWan / W4](recipes/self_forcing/README.md) |
| H3 audio–video DMD | [Full-sequence Grid1000 / W5](recipes/h3_fullseq/README.md), [Autoregressive / W6](recipes/h3_ar/README.md) |

W1–W6 identify the original benchmark protocols, not the limit of model support.
Official Self-Forcing and cross-scale LiveWan have separate recipes. H3
Grid1000 uses an approximate schedule; H3 AR keeps continuous sigma sampling.
The [support matrix](docs/guides/support_matrix.md) distinguishes available
execution paths and the scope of their tests.

For official SD1.5 DMD2 and Self-Forcing, the
[optimized training guide](docs/guides/official_optimized_training.md) includes
reference/optimized configurations and source-bound performance results.
Other recipes document their own configurations and measurements.

## Optimize an existing project with Codex

The standalone `dmd-system-optimizer` Skill helps Codex locate existing
implementations, inspect bottlenecks and test changes using your project's
commands. Installing the Skill needs only the Python standard library; it does
not require installing the training framework or moving your trainer.

```sh
python3 tools/install_optimizer_skill.py
```

Then ask Codex: **Use $dmd-system-optimizer to optimize this workload. Reuse
existing implementations first, measure end-to-end changes, and keep a WORK_LOG.**

The Skill includes a public library of successful, failed and unresolved
experiments. Personal continuity uses available, authorized Codex history and
project logs. See the [Skill guide](docs/guides/optimizer_skill.md) for updates
and lookup commands.

## Development

The main code boundaries are [models/data, algorithms, training state and
execution](docs/architecture/framework.md). Model factories and workflow
interfaces are documented in the [extension guide](docs/guides/extensions.md).
Automatic planning is optional; supported recipes also provide reference or
manual execution policies.

See [CONTRIBUTING](CONTRIBUTING.md) for development environments, focused tests
and performance-change requirements. Report problems through
[GitHub Issues](https://github.com/sliortega295-ops/DMD-System/issues), with a
reproduction command and relevant logs.

Framework code is [Apache-2.0](LICENSE). Third-party source, model and dataset
licenses remain separate; see [THIRD_PARTY.md](THIRD_PARTY.md).
