# Profile-guided DistillSchedule Planner v0

## Current conclusion

The repository now contains enough measured evidence to build the planner
architecture and validate **component accounting**, but not enough primitive
coverage to claim accurate prediction of unseen FSDP2, mesh-size, activation,
or arbitrary block-offload plans.

The strongest existing reconstruction is the validated No-GAN Role-Phase run:

```text
Phase A median        104.687 s
A→B transition         17.776 s
Critic phase          158.494 s
component sum         280.957 s
observed Full5 median 279.934 s
absolute error          1.023 s
relative error          0.365%
```

This proves that the simulator's cycle composition matches the measured phase
accounting. It is **not leave-one-out validation** because all three phase
components came from the same measured execution family.

The existing GAN observations establish useful ordering anchors:

```text
Sequential all40 median        656.125 s
fixed 8G+8Score median        1139.194 s
G40/S10/A128 detailed          868.489 s (diagnostic)
G40/S10/A128 low-overhead      824.685 s (single partial run)
```

They do not identify per-stage compute, NCCL, H2D, activation, or transition
coefficients independently. Consequently the simulator correctly reports the
new static FSDP2 candidate as `PROFILE_INCOMPLETE` until PR #13 emits real
primitive or whole-update calibration records.

## Planner decomposition

A plan is represented as:

- sequential phases;
- producer/consumer phases with explicit Q depth and credit semantics;
- sequence-dependent layout transitions; and
- GPU/CPU memory policies.

Every primitive key fixes:

```text
workload, operation, role, pyramid stage, grad mode,
backend, mesh size, local batch, residency,
activation policy, offload block set, dtype
```

Strict-family lookup never crosses workload, operation, role, backend, stage,
grad mode, residency, activation policy, block set, or dtype. Missing data is
not silently filled from a semantically different run.

## Hybrid memory model

Parameter, buffer, trainable-gradient, optimizer, EMA, block and transfer bytes
are analytical. The model explicitly includes:

- replicated versus ZeRO-3/FSDP2 parameter shards;
- largest current/prefetched all-gather unit;
- optimizer offload;
- activation GPU and CPU residency;
- persistent-pinned future targets;
- pageable→pinned construction duplication; and
- node-shared immutable frozen backing.

This is necessary to explain the recent GAN Role-Phase failures: serializing
construction removed the concurrent transient, but eight full private pinned
future targets still left insufficient CPU headroom for selected activations.

## Offload search

Once per-block bytes and measured transfer penalties are available, block
selection is solved as an exact Pareto-frontier knapsack:

```text
minimize added critical-path seconds
subject to released GPU bytes >= required headroom
```

The planner therefore need not assume a tail-k set. Blocks with lower call
frequency, lower criticality, or better overlap may be selected first.

## Initial profile protocol

The bounded calibration matrix is emitted by:

```bash
python scripts/distillschedule/print_planner_calibration_matrix.py \
  --profile configs/distillschedule/planner/helios_a100_profile_v0.json \
  --output /tmp/planner_calibration_matrix.json
```

The runtime should wrap small canaries with
`PlannerProfileRecorder.span()`. Required records are:

1. **Model inventory**
   - exact parameter/frozen/trainable/buffer bytes;
   - per-block bytes;
   - optimizer and EMA bytes.
2. **Per-operation service measurements**
   - Generator and Score DiT stages 0/1/2;
   - no-grad and selected-grad paths;
   - Fake/Real conditional/unconditional Score calls;
   - Critic denoise, GAN-G, GAN-D/R1;
   - optimizer and EMA commits.
3. **Resource measurements**
   - wall and optional CUDA-event time;
   - exact compute/NCCL/copy interval unions from Nsight;
   - H2D/D2H and collective bytes/events;
   - HBM allocated/reserved peaks;
   - cgroup stable and transient peaks;
   - saved-tensor activation peak/liveness.
4. **Hardware calibration**
   - pinned/pageable H2D and D2H bandwidth/latency;
   - intra-node and inter-node NCCL bandwidth/latency.
5. **Transitions**
   - source release, target prepare/materialize, optimizer restore,
     reshard/broadcast, process groups, barrier and cleanup.
6. **Validation anchors**
   - at least one full correctness/timing plan per backend family.

Profiled canaries remain diagnostic. Formal Full5 latency continues to come
from separate unprofiled runs.

## Building and checking a profile

```bash
python scripts/distillschedule/build_planner_profile.py \
  --seed-profile configs/distillschedule/planner/helios_a100_profile_v0.json \
  --event /path/to/rank0.jsonl \
  --event /path/to/rank1.jsonl \
  --output /tmp/helios_profile.json \
  --coverage-output /tmp/helios_profile_coverage.json

python scripts/distillschedule/simulate_full5_plans.py \
  --profile /tmp/helios_profile.json \
  --plan configs/distillschedule/planner/plans/no_gan_rolephase_replay.json \
  --plan configs/distillschedule/planner/plans/gan_static_fsdp2_candidate.json
```

## v0 acceptance criteria

Before using the planner to choose a paper result, target:

```text
GPU peak prediction error  < 5%
CPU peak prediction error  < 10%
Full5 latency error         < 10–15% on held-out plans
measured-plan ordering      correct
measured best plan          present in predicted Top-3
```

Until held-out plans exist, the output is a calibration/search aid rather than
an automatic-performance claim.
