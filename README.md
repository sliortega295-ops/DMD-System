# DMD-System

Execution planning and runtime support for **video and audio-video distribution-matching distillation**. DMD-System separates the original model/loss protocol from hardware placement, state lifetimes and execution order.

The package is `distillschedule`; the common entrypoint is `distillctl`. This research branch contains the shared six-workload planner integration. It is not a merge into `main`, a new GPU benchmark, or proof that all numerical/quality gates have passed.

## Install and explore

```bash
python -m pip install -e '.[test]'
distillctl workloads list
distillctl workloads describe --workload W4
distillctl workloads describe --workload W6 --hardware h100
```

These commands are read-only: no model imports, CUDA initialization, network calls or training. The catalog works in an installed wheel. Model recipes and source-bound validation still require a source checkout, explicitly supplied assets and the matching native environment.

## Six workloads

| ID | Protocol | Current execution guide |
|---|---|---|
| W1 | Wan1.3B hierarchical AR No-GAN DMD | [Original Wan + compute/state planning](recipes/wan_ar/README.md) |
| W2 | Helios14B No-GAN, G1/C5/EMA5 | [Safe-incumbent cohort](recipes/helios_nogan/README.md) |
| W3 | Helios14B GAN-on, G1/C5/EMA1 | [DMD + GAN, no merged-LoRA promotion](recipes/helios_gan/README.md) |
| W4 | Self-Forcing / LiveWan, 14B → 1.3B | [C51 full-bundle planning](recipes/self_forcing/README.md) |
| W5 | Full-sequence audio-video, H3 Grid1000 | [Approximate sigma, 4-step, G1/F5](recipes/h3_fullseq/README.md) |
| W6 | Original DiffSynth autoregressive audio-video | [Continuous sigma, 8-step, G1/F1](recipes/h3_ar/README.md) |

W5 and W6 are different training protocols, not two hardware variants of one workload. The catalog points to existing implementations; it does not add a second trainer or use historical endpoint times as planner costs.

The **five older locked profiles** remain for compatibility and historical reproduction:

```bash
distillctl workloads list --legacy
```

They are not the latest project inventory. In particular, the old LiveWan profile is C34; the W4 guide explicitly selects the C51 cohort. Existing long profile IDs, saved-plan validation, asset/source/capacity checks and legacy commands retain their original behavior.

## Plan and execute

A workload describes its original computation and state/version dependencies. Hardware-qualified calibration feeds the shared cost model and constrained frontier search. The serialized plan is then executed by the existing native adapter; operators must supply fresh source-bound inputs and capacity policies.

```bash
# Read-only inventory for the integrated C51 cohort:
distillctl workloads candidates \
  --workload livewan_crossscale_pipeline_best --include-livewan-c51

# Read-only validation of the original W6 protocol:
distillctl workloads native validate \
  --task configs/distillschedule/original_h3_ar/w6_reference.json
```

Use each recipe's runbook for calibrate → assemble → search/plan → saved-plan execution. No automatic GPU launch is added by this organization. The open `Program/Role` API remains available through `distillctl workloads native program --help`; its admitted capability combinations are distinct from the model-native adapters.

## Results and source identities

Start with the [six-workload correction](docs/status/SIX_WORKLOAD_RESULTS_CORRECTION_20261019.md), then follow the [A100/planner](docs/status/GITHUB_SOURCE_SNAPSHOT_20261019.md) and [H100](docs/status/H100_SOURCE_SNAPSHOT_20261019.md) source indices. Paper snapshots, same-source timing pairs, per-feature gains and manual/saved-plan comparisons have different denominators and must remain separate.

The archived evidence includes H100 timings for all six workloads and A100 saved-plan execution pairs for W1/W2/W3/W4/W6. W5's latest saved-plan timing pair remains blocked by its sampled backward numerical FAIL. W3's no-merge retime preserves its post-timer finalization failure. These are historical source-bound records, not GPU revalidation of this organization commit.

Later manual and H100 derivatives remain separately pinned where they have not been transplanted into the shared runtime. Discovery shows both measured and published commits rather than silently substituting a newer unmeasured candidate. H100 queues remain user-stopped.

## Repository layout

```text
src/distillschedule/    public IR, profiling, search, model bindings and runtime
runtime/                packaged compatibility mechanisms; audited paths unchanged
recipes/                six current workflow guides, no duplicate training code
configs/distillschedule/ original protocol tasks, backend fragments, locked legacy profiles
scripts/                common CLI and native launch/evidence dependencies
tools/assets/           standalone data/trace/checkpoint preparation
tools/profiling/        trace analysis, profiler commands and plots
benchmarks/kernels/     standalone kernel and state-lifetime canaries
tests/                 unit, protocol, source and distributed regression tests
docs/                  architecture, runbooks, source indices and evidence boundaries
artifacts/             immutable small receipts/publication packets, not live outputs
third_party/           pinned upstream implementations and licenses
```

[Organization and migration map](docs/architecture/repository_organization.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md)

## Source-only checks

```bash
distillctl workloads verify-catalog --repo-root .
python scripts/distillschedule/check_best_known_integration.py
python -m pytest -q tests/test_repository_organization.py
python scripts/check_wheel_install.py
```

`check_best_known_integration` retains the original import receipts and verifies an explicit current-source overlay; it does not silently accept a changed file. The wheel check installs into a fresh environment outside the checkout with isolated Python and asserts real site-packages imports. CPU/source tests do not replace CUDA/NCCL, complete training-state or long-run quality validation.
