# DMD-System

**DMD-System** is a research runtime for scheduling and executing multi-model diffusion distillation workloads.

The current prototype grows out of our Helios Stage-3 DMD work, but the runtime boundary is deliberately model-agnostic: model family, model size, spatial/temporal shape, update frequency, training backend, rollout engine, placement, residency, and lifecycle policy are represented explicitly instead of being hard-coded into one training script.

> Status: **research prototype / refactor v1**. The typed runtime is available and the existing Helios training path is retained behind a compatibility adapter while operations are migrated incrementally.

## Motivation

DMD-style training is a compound workload rather than a single-model training loop. A typical cycle may involve a trainable Generator, shared Real/Fake Score or Critic modes, optional Teacher/GAN/Reward/VAE branches, different update frequencies, parameter-version barriers, and different memory/placement policies across logical steps.

The systems problem is therefore not only “how do we fit one model?”, but **how do we schedule several physical model states and their lifetimes without changing training semantics?**

## Architecture

```text
ExecutionPlan
    ↓
DMD protocol compiler
    ↓
Model Adapter
    ↓
Training / Rollout Backend
    ↓
DistillExecutor
```

The main package is under `src/distillschedule/`:

```text
src/distillschedule/
├── core/                 # typed specs, plans, contracts, capabilities
├── workloads/dmd/        # DMD logical schedule and protocol compiler
├── adapters/             # Helios and generic model-family adapters
├── backends/
│   ├── training/         # training backend interfaces
│   └── rollout/          # native / remote / vLLM / SGLang facades
└── runtime/              # executor, role runner, version synchronization
```

Two specifications are kept separate:

- **WorkloadSpec**: algorithm semantics — models, shapes, branches, update frequencies.
- **ExecutionPlan**: physical execution — rank groups, backend, residency, streaming, offload, construction/swap waves, rollout engine, and evidence contract.

That separation lets the same DMD workload be evaluated at different model sizes, resolutions, hardware configurations, or rollout backends without duplicating the training logic.

## Quick start

```bash
python -m pip install -e .

python scripts/distillctl.py validate \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

python scripts/distillctl.py show \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

python scripts/distillctl.py run \
  configs/distillschedule/plans/helios_14b_384x640_full5.json \
  --dry-run
```

Every compiled run writes immutable `resolved_plan.json` and `compiled_run.json` artifacts before entering the model-specific path.

## Composable configuration

The Helios example composes reusable fragments instead of copying an entire launcher:

```text
helios_14b_models.json
+ video_384x640_f81.json
+ dmd_full5.json
+ helios_a_local_lifecycle.json
```

This makes model size, resolution, update schedule, and lifecycle policy independently replaceable.

## Helios integration

`HeliosDMDAdapter` currently follows a **strangler migration**:

```text
ExecutionPlan
    → HeliosDMDAdapter
    → validated runtime configuration
    → source-faithful Helios training path
```

We intentionally do not rewrite the validated Helios DMD loss, rollout, optimizer, EMA, operation-trace, or memory mechanisms in one step. Native operations are migrated only after parity checks against the existing path.

The first parity targets are one Critic-only operation and one complete Full5 cycle, followed by tensor/loss/version/optimizer/EMA equivalence.

## Rollout engines

Critic-only rollout is separated from differentiable Generator training:

```text
Differentiable Generator path   # autograd required
RolloutProvider path            # no-grad sample production
```

The rollout interface carries an explicit `ModelVersion`; the default contract is exact synchronization (`max_version_staleness = 0`). Facades for remote, vLLM, and SGLang backends are included, but the current Helios diffusion/history protocol is **not** claimed to be directly compatible with text-LLM serving APIs; a model-specific transport adapter is required.

## Repository layout

```text
configs/distillschedule/     composable plans and fragments
docs/architecture/           refactor design and migration notes
src/distillschedule/         new typed runtime
runtime/                     validated legacy/runtime mechanisms retained for migration
legacy/                      archived Level-3 prototype workloads
scripts/distillctl.py        lightweight CLI entry point
tests/                       runtime/refactor regression tests
```

## Design principles

1. **Semantics first.** Placement or overlap must not cross optimizer/version barriers.
2. **Physical state is explicit.** Logical Real/Fake/Critic roles may share one physical Score state.
3. **Shape is part of the plan.** Memory policies are profiled for model/precision/batch/frame/resolution buckets rather than assuming arbitrary dynamic shapes.
4. **Backends are pluggable.** DeepSpeed/FSDP-style training and native/remote rollout are execution choices, not workload semantics.
5. **Evidence is immutable.** Formal runs resolve their effective plan before launch.
6. **No silent fallback.** Unsupported rollout/backend combinations fail closed.

## MLSys roadmap

The current direction is to turn the hand-tuned Helios experiments into a scheduler that jointly reasons about Generator/Score/Teacher/Critic placement, parameter and activation residency, frozen-block streaming and prefetch, construction and swap concurrency, node/NUMA budgets, update frequency and parameter versions, microbatch wavefronts, and external rollout providers.

The intended evaluation is the **performance–memory Pareto frontier** across model sizes, resolutions, workloads, and hardware rather than a single peak-speed number.

## Development

```bash
python -m pip install -e '.[test]'
pytest -q
```

The refactor-specific regression tests cover configuration resolution, adapter validation, DMD protocol compilation, executor behavior, and rollout/version contracts.

## Upstream and license

The research prototype originated from work on the open-source [Helios](https://github.com/PKU-YuanGroup/Helios) codebase. Upstream model code and retained compatibility components remain subject to their corresponding notices and licenses.

This repository is distributed under the Apache-2.0 license unless a subcomponent states otherwise.
