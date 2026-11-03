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

H3 full-sequence training uses the environment and source configuration in its
[recipe](../../recipes/h3-fullseq-original-training.md).

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

<a id="troubleshooting"></a>
## Troubleshooting and evidence boundaries

Start with the configuration, dependency lane and last complete run receipt.
Do not replace the model, data, precision, backend or source pin merely to make
an error disappear. Keep failed output and use a new run directory for a retry.

| Symptom | What to check | Safe next action |
| --- | --- | --- |
| `doctor` reports missing prepared input | Resolved config paths and actual manifest | Prepare/import the required genuine inputs; do not fill missing conditions with random tensors |
| Optional package/import failure | [Environment lane](#choose-a-model-environment), actual import/version and original recipe | Reuse a matching environment; do not combine incompatible Diffusers pins |
| Recipe has placeholders | Required model/source/input/capacity/plan fields | Supply your explicitly owned assets before launch; no automatic download or guessed historical path |
| Source/layout/hash mismatch | Request, installed factory bytes, source index, saved plan and artifact hashes | Use the exact admitted source or create a separately reviewed new configuration/plan; do not edit receipts/goldens |
| Missing shard or `_SUCCESS` | Manifest, every named rank payload and complete-cycle marker | Preserve the partial directory; resume only a preceding complete checkpoint |
| Resume changes topology/data/horizon | Original request, scheduler contract, sample inventory, seed and rank layout | Use a supported same-topology complete-cycle boundary; arbitrary topology migration is unsupported |
| Owner is poisoned after a partial update | `failure.json`, last valid checkpoint and native step receipt | Stop that owner; reconstruct from the last complete checkpoint, not a partially committed in-memory model |
| Video generation rejects a native latent result | Export factory, `generate`/`decode` capability and output kind | Add an actual model decoder, or explicitly request `--output-kind latents`; latents are not generated media |
| Planning selects an unsupported capability | Measured candidate/source identity, effective policy and executed operations | Reject rather than falling back to a fixed recipe or reference backend |
| Peer exits or hangs in a distributed CPU check | First failing rank, process exit/signal, Gloo stage and framework/Torch/Python identity | Collect the original error on all ranks; a barrier/sleep or skipped test is not a demonstrated fix |

### What to retain

A Program run writes configuration/source identity, `effective-policy.json`,
`metrics.jsonl`, `summary.json` and complete checkpoint manifests; failures use
`failure.json` where the owning runtime can publish one. Native workers retain
their own rank/node receipts and actual source/asset/capacity gates. Check the
recipe before assuming a field or capability exists.

Keep the exact command (without credentials), source revision/diff, resolved
configuration, imported runtime, input identities, final versions and raw error.
When sharing a problem, small JSON/CSV/log excerpts are usually enough. Model
weights, full datasets, profilers and optimizer shards should remain in the
operator's persistent storage unless transfer is actually required.

### Checkpoint versus inference weights

A training checkpoint includes optimizer/scheduler/EMA, RNG, data progress and
logical versions under the exact restore contract. An inference export instead
contains the declared model state and source/config binding for new inputs.
They are not interchangeable. Full distributed export can require substantial
CPU materialization; tiny CPU qualification does not establish large-model host
capacity. See [training and output contracts](training.md#generate-and-export).
