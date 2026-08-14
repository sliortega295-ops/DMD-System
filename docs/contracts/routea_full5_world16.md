# Frozen contract: Route-A Full5 world16

This document is the semantic source of truth for the typed Route-A compatibility revalidation. Launchers, server paths and recovery scripts may change; the workload below may not change without creating a new contract and baseline.

## Workload identity

| Field | Frozen value |
|---|---|
| Hardware topology | 2 nodes × 8 A100-40GB; `world_size=16` |
| Resolution | 384×640 source video |
| Pyramid | 12×20 → 24×40 → 48×80 |
| Full latent | `[B, 16, 9, 48, 80]`, batch size 1 per rank |
| Logical range | continuous steps 10, 11, 12, 13, 14 |
| Branches | DMD and EMA enabled; GAN, Reward and DMD VAE decode disabled |
| Update ratio | Critic every step; Generator every fifth step |
| Initial versions | Generator 2, Critic 10, EMA 10 |
| Required final versions | Generator 3, Critic 15, EMA 15 |
| Backend | DeepSpeed ZeRO-1, CPU optimizer offload, no parameter offload |
| Trace | rank-qualified world16 trace v3 for the frozen samples |
| Resume point | canonical world16 checkpoint at logical step 10 |
| Lifecycle | Generator/Score 40-block `HOST_SHARED_IMMUTABLE` streaming, window 1 |
| Activation offload | Generator-selected tensors, threshold 32 MiB |
| Waves | construction wave 0; swap wave 8 |
| Comparison | one diagnostic Full5 run against the frozen Route-A baseline; ±10% timing band |

The typed plan is [`configs/distillschedule/plans/helios_routea_full5_world16_typed.json`](../../configs/distillschedule/plans/helios_routea_full5_world16_typed.json). The plan composes model, shape, DMD schedule and lifecycle fragments; the resolved plan digest, not only the top-level file hash, belongs in launch evidence.

## Physical model semantics

There are two 14B backbones:

- Generator: frozen Helios-Distilled backbone plus trainable Generator LoRA.
- Score: frozen Helios-Base backbone plus trainable Fake-Score/Critic LoRA.

Real Score and Fake Score share the same Score backbone and differ by Critic-LoRA state. Generator and Score do not share a backbone because their base checkpoints differ. This compatibility run must not replace either checkpoint, merge their semantics or enable extra GAN/Reward/VAE branches.

## EMA semantics

The legacy Helios field named `ema_start_step=750` controls the decay-regime transition. It is **not** an operation-enable gate for this checkpoint continuation. The frozen workload performs EMA bootstrap updates on steps 10–14, after each Critic update, and therefore requires five EMA updates.

The current typed representation expresses this as:

```text
ema_start_step = 750
ema_bootstrap_before_start = true
ema_update_every = 1
ema_after_critic = true
```

These names are compatibility schema fields. A future schema may rename them to separate `decay_start_step` from `update_enabled_from_step`, but the five-step behavior and final EMA version must remain identical.

## Required source and runtime identities

A real launch must record and compare, on both nodes:

- DMD-System HEAD, clean-tree state, top-level plan SHA and resolved plan digest;
- pinned Helios HEAD, clean-tree state and `train_helios.py` SHA;
- Python, Accelerate and DeepSpeed executable paths and SHA-256 values;
- rendered Accelerate config SHA and referenced DeepSpeed config SHA;
- model, data, trace, checkpoint, pre-cast and offline-kernel paths; and
- master address/ports plus node rank.

Path existence alone does not establish semantic identity. Where an asset has its own manifest or digest, that identity should be included in the preregistration/result record.

## Acceptance conditions

Functional acceptance requires all of the following:

1. The cheap pre-model probe establishes exactly 16 unique global ranks, two unique hosts, local ranks 0–7 on each host and a successful 16-rank NCCL collective.
2. Both node workers use the same resolved plan digest and pinned source/runtime identities.
3. Every rank executes exactly steps 10–14 with one Generator optimizer step, five Critic optimizer steps and five EMA steps.
4. The version transition is G2/C10/EMA10 → G3/C15/EMA15.
5. `checkpoint-15` exists and has a non-empty inventory.
6. Activation-offload and block-streaming live residency return to zero at the end of every rank.
7. Both node workers return zero and cgroup OOM/OOM-kill counters do not increase.

Timing consistency is evaluated only after functional acceptance. The 64-GiB host-reserve admission result is reported separately and cannot turn a functional failure into success or vice versa.

## Explicit non-goals and forbidden substitutions

This contract does not authorize:

- falling back to ZeRO-2;
- changing Full5 steps, batch, trace, checkpoint, optimizer, EMA order or branch semantics;
- replacing the real Helios workload with a mini/synthetic substitute for the final claim;
- treating two independent world8 groups as world16;
- publishing a timing result from a partial/failed run; or
- claiming that the typed runtime has natively reimplemented Helios model math.

Any intentional semantic change must receive a new plan ID, contract document, checkpoint/trace identity and matched baseline.
