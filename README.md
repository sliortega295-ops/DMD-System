# DMD-System

**DMD-System** is a research runtime for scheduling and executing compound diffusion-distillation workloads. It grew out of Helios Stage-3 DMD training, but its public boundary is model-agnostic: workload semantics, model roles, shapes, update frequencies, rank topology, training backend, rollout provider and memory lifecycle are represented explicitly rather than hidden inside one training script.

> **Current status — research prototype / refactor v0.2.** The typed runtime is the target architecture. Real Helios model math still runs through a pinned compatibility checkout. Route-A Attempt 005-002 has now passed the world16 functional/evidence contract, but its one-run diagnostic window was 11.389% slower than the frozen reference and is not a formal timing or speedup result. Route-B target-graph execution remains `Not Run`; live integration work is tracked in [Issue #2](https://github.com/sliortega295-ops/DMD-System/issues/2).

## Start here

| Need | Authoritative document |
|---|---|
| Understand the code and migration boundary | [`docs/architecture/refactor_v1.md`](docs/architecture/refactor_v1.md) |
| See what belongs in Git versus experiment storage | [`docs/architecture/v0.2_source_inventory.md`](docs/architecture/v0.2_source_inventory.md) |
| Read the frozen Route-A Full5 world16 semantics | [`docs/contracts/routea_full5_world16.md`](docs/contracts/routea_full5_world16.md) |
| Run the BAAI two-node compatibility path | [`docs/runbooks/baai_routea_world16.md`](docs/runbooks/baai_routea_world16.md) |
| Inspect or launch the staged No-GAN suite | [`docs/runbooks/baai_no_gan_suite.md`](docs/runbooks/baai_no_gan_suite.md) |
| Check the latest accepted/not-run status | [`docs/status/routea_world16.md`](docs/status/routea_world16.md) |
| Understand result manifests and claim labels | [`artifacts/manifests/README.md`](artifacts/manifests/README.md) |

The complete documentation index is in [`docs/README.md`](docs/README.md).

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

Two specifications are intentionally separate:

- `WorkloadSpec` describes algorithm semantics: Generator and Score roles, physical model identities, video/latent shape, DMD branches and G/C/EMA update rules.
- `ExecutionPlan` describes physical execution: rank groups, backend, placement, streaming, activation offload, construction/swap waves, rollout synchronization and evidence requirements.

The stable package is `src/distillschedule/`. The root `runtime/` package is a validated compatibility-mechanism layer retained during the strangler migration; it is not a second public API.

## Current Route-A world16 boundary

The frozen integration target is 2 nodes × 8 A100-40GB, world16, 384×640, steps 10–14, ZeRO-1 with CPU optimizer offload and exact G2/C10/EMA10 → G3/C15/EMA15 continuation. The exact contract is deliberately kept outside launcher prose in [`docs/contracts/routea_full5_world16.md`](docs/contracts/routea_full5_world16.md).

The repository now contains fail-closed building blocks for that path:

- typed plan and inner DeepSpeed contract validation;
- node-qualified Accelerate config rendering with explicit `standard` multi-node launcher;
- executable/source identity recording;
- a cheap 16-rank Gloo/NCCL topology probe before any 14B model construction;
- node-local locking, resource evidence and immutable compiled artifacts;
- a central two-host coordinator with peer-failure cancellation; and
- a summarizer that rejects actual CUDA/cgroup OOM while retaining host pressure as diagnostic evidence.

The coordinator and evidence path have passed one real 2×8 cluster acceptance. The No-GAN suite additionally contains a resumable DAG and bounded search/timing drivers; its target G/S/T candidate, live redistribution and continuous Route-B executors remain explicit external command seams and are therefore `Not Run`, not implemented results.

## Quick start

The typed compiler and generic plans do not require model weights:

```bash
python -m pip install -e '.[test]'

python scripts/distillctl.py validate \
  configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json

python scripts/distillctl.py show \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

pytest -q tests/test_refactor_*.py
```

A Helios dry run renders immutable `resolved_plan*.json`, `compiled_run*.json` and a resolved Helios config without launching a model:

```bash
python scripts/distillctl.py run \
  configs/distillschedule/plans/helios_14b_384x640_full5.json \
  --dry-run
```

The real BAAI world16 procedure has additional source, asset, executable, topology and evidence gates. Follow the runbook rather than assembling a command from old experiment notes.

## Repository layout

```text
src/distillschedule/           stable typed runtime and API
runtime/                       validated compatibility mechanisms
configs/distillschedule/       composable plans, templates and site profiles
scripts/distillschedule/       reusable launch, preflight, profiling and analysis tools
profiling/nsight/              profiler conventions and claim boundary
docs/contracts/                immutable workload semantics
docs/runbooks/                 operational procedures
docs/status/                   dated evidence/status ledgers
artifacts/manifests/            curated small result summaries only
tests/                          contract and mechanism regression tests
```

Weights, checkpoints, private datasets, raw rank logs, `.nsys-rep` files, SQLite exports, tensor dumps and full run directories remain outside normal Git history.

## Claim discipline

A result is not accepted merely because a process returned zero. Route-A evidence must establish the declared topology, exact logical steps and optimizer counts, parameter-version transition, checkpoint inventory, lifecycle cleanup and absence of OOM. Timing is claimable only after functional success and a valid matched baseline. Host and cgroup pressure remain measured diagnostics; only an actual CUDA OOM, cgroup OOM/OOM-kill, or memory-attributed nonzero exit rejects memory admission.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a frozen plan, compatibility seam or evidence schema.
