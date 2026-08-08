# DMD-System

**DMD-System** is a research runtime for scheduling and executing multi-model diffusion-distillation workloads.

The project grows out of our Helios Stage-3 DMD work, but the runtime boundary is deliberately model-agnostic: model family, model size, spatial/temporal shape, update frequency, training backend, rollout engine, placement, residency and lifecycle policy are explicit inputs rather than branches inside one giant training script.

> **Status: research prototype / refactor v0.2.** The typed runtime is the target architecture. Validated Helios/A100 runtime mechanisms are retained in a compatibility layer and are being migrated operation by operation under numerical/version parity gates.

## Why DMD needs a runtime

A DMD training cycle is a compound workload rather than a single-model loop. It may involve a trainable Generator, shared Real/Fake Score or Critic modes, Teacher/EMA state, optional GAN/Reward/VAE branches, different update frequencies, parameter-version barriers, and different memory/placement policies at different logical steps.

The systems problem is therefore not only “how do we fit one model?” but:

> **How should several physical model states be placed, materialized, pipelined and transitioned without changing the DMD training semantics?**

## Architecture

```text
WorkloadSpec + ModelSpec + ShapeSpec
                ↓
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

The stable architecture lives under `src/distillschedule/`:

```text
src/distillschedule/
├── core/                 # typed specs, plans, contracts, capabilities
├── workloads/dmd/        # DMD schedule and logical operation graph
├── adapters/             # Helios and generic model-family adapters
├── backends/
│   ├── training/         # training backend interfaces
│   └── rollout/          # native / remote / vLLM / SGLang facades
└── runtime/              # executor, role runner, version synchronization
```

`WorkloadSpec` describes algorithm semantics. `ExecutionPlan` describes physical execution: rank groups, backend, residency, block streaming, activation offload, construction/swap waves, rollout provider and evidence contract. Keeping them separate lets the same DMD workload be evaluated at different model sizes, resolutions and hardware configurations without duplicating the training logic.

## Validated mechanism layer

The root `runtime/` package is intentionally retained during the strangler migration. These are not mock interfaces: they are mechanisms extracted from the real Helios/A100 research path and protected by correctness tests.

```text
runtime/
├── activation_offload.py       # audited saved-tensor CPU offload
├── frozen_block_streaming.py   # immutable block streaming + checkpoint interlock
├── immutable_model_backing.py  # node-shared frozen-state backing
├── residency.py                # physical-state ownership/version safety
├── host_admission.py           # node budget and reserve admission
├── ema.py                      # trainable-state-only EMA
├── dmd_boundary.py             # DMD and Critic gradient boundaries
├── deepspeed_backend.py        # ZeRO backend/offload contract
├── zero3_checkpoint.py
├── zero_partition.py
├── precast_checkpoint.py
├── operation_trace.py          # operation-keyed exact replay
├── rollout_trace.py            # nested rollout/autograd tracing
├── trace.py                    # phase/NVTX tracing
├── critic_pool*.py             # Critic-only pool/transport/checkpoint/trace
├── generator_update_gate.py
└── m1_action_library.py
```

The long-term direction is to move these mechanisms behind the typed package once the new execution path matches the validated implementation in tensor identity, loss, parameter-version order, optimizer update and EMA transition.

## Quick start

The typed plan/compiler path has no heavyweight model dependency:

```bash
python -m pip install -e '.[test]'

python scripts/distillctl.py validate \
  configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json

python scripts/distillctl.py show \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

python scripts/distillctl.py run \
  configs/distillschedule/plans/helios_14b_384x640_full5.json \
  --dry-run
```

Every compiled run writes immutable `resolved_plan.json` and `compiled_run.json` artifacts before entering the model-specific path.

## Helios integration

`HeliosDMDAdapter` follows a strangler migration:

```text
ExecutionPlan
    → HeliosDMDAdapter
    → validated runtime configuration
    → pinned Helios training checkout
```

The full upstream Helios model tree, weights and private cluster assets are **not vendored** here. On the cluster the adapter points to a pinned Helios checkout. Reusable DMD-System mechanisms are kept in this repository; one-off historical launchers and raw experiment output are not.

For the current two-node BAAI A100 environment, copy and source
`configs/distillschedule/server/baai_2x8_a100.env.example`, then run
`scripts/distillschedule/baai_server_smoke.sh`.  The smoke gate validates the
external Helios checkout, assets, typed plan, rendered configuration and command
without launching a model or claiming a GPU result.  Multi-node execution still
uses the validated external launcher until the typed executor gains a coordinator;
the single-process `distillctl run` command must not be presented as a world16
launcher.

Current parity order:

```text
Critic-only operation
→ Generator-update operation
→ complete Full5 cycle
→ tensor/loss/version/optimizer/EMA parity
→ migrate Helios batch/rollout/loss/checkpoint boundaries
```

## Critic-only and rollout engines

Differentiable Generator training and no-grad rollout are separate interfaces:

```text
Differentiable Generator     # autograd required
RolloutProvider              # no-grad sample production
```

`RolloutProvider` carries an explicit `ModelVersion`; the default contract is exact synchronization (`max_version_staleness = 0`). Native, remote, vLLM and SGLang facades are included. A serving engine is only used when a model-specific adapter actually supports the corresponding generation protocol; there is no silent fallback from Helios diffusion rollout to an LLM serving API.

The validated compatibility layer also includes the real Critic-pool contracts: rank topology, bounded queue, fixed-width P2P metadata, commit ACK/version checks, rank-qualified operation traces and Helios no-grad rollout primitives.

## Nsight / NVTX profiling

Profiling code is first-class source because it is needed to explain system behavior:

```text
profiling/nsight/
scripts/distillschedule/
├── run_nsys_locked.sh
├── nsys_profile_command.sh
├── analyze_w16_pool_nsys.py
├── plot_w16_nsys_latest.py
└── bench_transfer.py
```

`run_nsys_locked.sh` locks an idle GPU node, records source/environment metadata, captures an explicit NVTX/CUDA-profiler range and validates the report. Nsight output is diagnostic evidence; formal speedups come from separate unprofiled matched runs.

Raw `.nsys-rep`, SQLite exports and generated figures are intentionally excluded from normal Git history.

## Configuration

The Helios example composes reusable fragments rather than cloning launchers:

```text
helios_14b_models.json
+ video_384x640_f81.json
+ dmd_full5.json
+ helios_a_local_lifecycle.json
```

This makes model size, resolution, update schedule and lifecycle policy independently replaceable. Shape-sensitive policies are profiled for explicit `(model, precision, batch, frames, resolution)` buckets rather than pretending all shapes are interchangeable.

## Repository layout

```text
configs/distillschedule/       composable typed plans and fragments
docs/architecture/             design, migration boundary, source inventory
src/distillschedule/           stable typed runtime/API
runtime/                       validated compatibility mechanisms
scripts/distillschedule/       reusable experiment/profiling utilities
profiling/nsight/              profiler conventions and claim boundary
tests/                         refactor + mechanism correctness tests
```

See `docs/architecture/v0.2_source_inventory.md` for the explicit keep/archive policy.

## What is intentionally not in Git

The cleaned repository does **not** contain model weights, 100-GB checkpoints, `.nsys-rep` files, raw rank logs, tensor dumps, generated videos, private datasets, or hundreds of iteration-specific failed launcher/manifests. Those are immutable experiment artifacts rather than reusable source code.

The repository also avoids copying the complete upstream Helios tree. DMD-System remains a DMD runtime with a Helios adapter, not a permanent fork of one workload.

## Design principles

1. **Semantics first.** Placement and overlap cannot cross optimizer/version barriers.
2. **Physical state is explicit.** Logical Real/Fake/Critic modes may share one physical Score state.
3. **Lifecycle is phase-specific.** Construction admission, steady residency and transition peaks are different constraints.
4. **Shape is part of the plan.** Memory and performance profiles are keyed by workload shape.
5. **Backends are pluggable.** DeepSpeed/FSDP and native/remote rollout are execution choices, not workload semantics.
6. **Evidence is immutable.** Formal runs bind resolved plan, trace, checkpoint and source identity before launch.
7. **No silent fallback.** Unsupported model/backend combinations fail closed.

## MLSys direction

The current paper direction is a **frequency-, version-, and lifecycle-aware scheduler for multi-model diffusion distillation**. The scheduler should jointly reason about:

- Generator / Score / Teacher / Critic placement;
- parameter residency, sharding, streaming and offload;
- activation lifetime and gradient boundaries;
- construction and swap concurrency under node/NUMA budgets;
- microbatch producer/consumer overlap;
- phase transitions across different DMD update frequencies; and
- native or external rollout providers.

The intended evaluation is the end-to-end performance–memory Pareto frontier across model sizes, resolutions, workloads and hardware, not a single isolated microbenchmark.

Current engineering sequence:

```text
new/validated-path parity
→ reserve-admitted Full5 baseline
→ Critic-only Plan C + Generator-update Plan G + transition
→ lifecycle/placement planner
→ second real DMD workload
→ GAN stress path / external rollout engine / H100 evaluation
```

## Development

```bash
python -m pip install -e '.[test]'
pytest -q tests/test_refactor_*.py
```

Mechanism tests that import PyTorch/Helios are expected to run in the research environment with those optional dependencies installed. CI keeps the typed package lightweight and separately compile-checks the restored compatibility/runtime source.

## Upstream and license

The research prototype originated from work on the open-source Helios codebase. Upstream model code and retained compatibility components remain subject to their corresponding notices and licenses.

The DistillSchedule refactor follows Apache-2.0 unless a subcomponent states otherwise.
