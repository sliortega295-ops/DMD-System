# DistillSchedule refactor v1

## Goal

This refactor changes the project boundary from “a modified Helios training script” to
“a typed DMD runtime with a Helios compatibility adapter.” It deliberately does **not**
rewrite the already-validated Helios loss, rollout, optimizer, EMA, operation-trace, or
memory primitives in one step.

The stable dependency direction is now:

```text
ExecutionPlan
    -> DMD protocol compiler
    -> model adapter
    -> training / rollout backend
    -> executor
```

Helios currently terminates that chain in the source-faithful legacy entrypoint:

```text
ExecutionPlan
    -> HeliosDMDAdapter
    -> validated DISTILLSCHEDULE_* environment
    -> train_helios.py
```

This is a strangler migration. New orchestration and configuration belong in
`src/distillschedule/`; model operations move out of `train_helios.py` only after an
operation-trace and checkpoint parity gate.

## Directory structure

```text
src/distillschedule/
├── core/
│   ├── specs.py          # ModelSpec / ShapeSpec / WorkloadSpec / HardwareSpec
│   ├── plan.py           # physical execution and lifecycle policy
│   ├── graph.py          # model-independent logical operation graph
│   ├── config.py         # deterministic include/merge/env resolution
│   ├── contracts.py      # fail-closed validation
│   └── capabilities.py   # adapter/backend capability declarations
├── workloads/dmd/
│   ├── schedule.py       # update-frequency semantics
│   └── protocol.py       # G / Score / Critic / EMA logical order
├── adapters/
│   ├── helios/           # typed plan -> source-faithful Helios invocation
│   └── generic/          # plan-only adapter for a new model family
├── backends/
│   ├── rollout/          # native, remote, vLLM, and SGLang facades
│   └── training/         # native backend protocol
├── runtime/
│   ├── executor.py       # compile/run and immutable resolved artifacts
│   ├── role_runner.py    # operation-by-operation migration target
│   └── version_sync.py   # monotonic component versions
└── cli.py                # distillctl
```

The old `runtime/` tree remains a compatibility surface for measured mechanisms. The
Level-3 synthetic workloads have moved to `legacy/level3_prototypes/`; their historical
`runtime.workloads.*` paths are import shims.

## The two specifications

`WorkloadSpec` describes algorithm semantics that should not change when hardware or
backends change:

- Generator and Score model family, size, precision, checkpoint, and adapter;
- batch, frame, latent, resolution, and pyramid shape;
- Generator/Critic/EMA update frequencies;
- DMD, GAN, Reward, VAE-decode, EMA, and CFG branches.

`ExecutionPlan` describes physical execution choices:

- hardware and rank groups;
- native, DeepSpeed, FSDP, or HSDP training backend;
- native, remote, vLLM, or SGLang rollout provider;
- exact model-version/staleness contract;
- activation offload, block streaming, host policy, construction wave, swap wave, and
  prefetch distance;
- operation trace, checkpoint, run range, and instrumentation mode.

A model-size or resolution experiment should therefore compose fragments rather than
copy an entire launcher. The example plan combines:

```text
helios_14b_models.json
+ video_384x640_f81.json
+ dmd_full5.json
+ helios_a_local_lifecycle.json
```

## Commands

Repository-local use does not require installation:

```bash
python scripts/distillctl.py validate \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

python scripts/distillctl.py show \
  configs/distillschedule/plans/helios_14b_384x640_full5.json

python scripts/distillctl.py run \
  configs/distillschedule/plans/helios_14b_384x640_full5.json \
  --dry-run
```

For an editable install:

```bash
python -m pip install -e .
distillctl validate configs/distillschedule/plans/helios_14b_384x640_full5.json
```

Every run writes `resolved_plan.json` and `compiled_run.json` before launching the model
entrypoint.

## Adding a model family

Create an adapter implementing `DMDModelAdapter`:

1. declare supported training and rollout backends;
2. validate model-family and shape-specific constraints;
3. compile the generic DMD protocol;
4. either return a legacy invocation or implement native operation handlers.

Register it in `distillschedule.adapters.default_registry`.

A native migration should proceed operation by operation:

```text
critic-only rollout
-> shared Score modes
-> Critic backward/update
-> Generator DMD boundary
-> Generator backward/update
-> EMA/checkpoint
```

Each migration step must compare operation trace, tensor identities, losses, versions,
optimizer state, and EMA against the legacy path.

## Adding vLLM or SGLang rollout

The Critic-only interface is separated from differentiable Generator training:

```text
Differentiable Generator path  # autograd, Generator update
RolloutProvider path           # no-grad samples for Critic training
```

`RolloutProvider` requires an explicit `ModelVersion`. With the default
`max_version_staleness=0`, the service must acknowledge `G@v` before it may produce any
sample tagged as `G@v`. vLLM and SGLang facades accept a transport client and do not
import either dependency into the core package.

Current Helios diffusion rollout is not declared directly compatible with the text-LLM
vLLM/SGLang APIs. A future Helios diffusion backend should either use a remote adapter
or explicitly declare compatibility after implementing its tensor/history protocol.

## Archive policy

Code belongs in `legacy/` when it is experiment-specific and no longer participates in
an accepted execution path. Moving code must not break manifest-referenced imports;
thin compatibility wrappers remain until the evidence retention window expires.

The next archive wave should cover one-off iteration launchers and superseded analysis
scripts, but only after the remote repository has a durable Git bundle/tag and their
raw evidence is independently addressable.

## Next refactor slice

The first remote parity target is one Critic-only operation and one Full5 cycle. Once
both match the legacy path, extract these sections from `train_helios.py` into:

```text
adapters/helios/batch.py
adapters/helios/rollout.py
adapters/helios/losses.py
adapters/helios/checkpoint.py
```

Do not add new model-family branches directly to `train_helios.py`.
