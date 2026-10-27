# Official Self-Forcing and DMD2 optimized training

DMD-System provides reference and optimized configurations for both official
workflows. All configurations use `distillctl`: data preparation, continuous
updates, cycle-boundary checkpointing, resume and training metrics share the
framework lifecycle.

## Implementations and measured performance

16 × A100-40GB, global batch 16. Each measurement executes two consecutive
cycles after one initialization; the table reports cycle 2 without a profiler.
A DMD2 cycle contains G1/C10, and a Self-Forcing cycle contains G1/C5.

FSDP2 is the primary baseline for both workflows.

| Workflow | FSDP2 baseline (s) | Optimized (s) | Speedup |
|---|---:|---:|---:|
| Official DMD2, SD1.5 GAN, 512 × 512 | 7.514 | 3.475 | 2.162× |
| Official Self-Forcing, Wan, 368 × 640, 81 frames | 50.739 | 30.959 | 1.639× |

DMD2 also has an auxiliary Resident/DDP comparison: 6.114 → 3.475 s (1.759×).

These are the latest optimization-run results. The implementation mapping and
framework-integration checks are recorded in the [validation appendix](../status/official_optimized_framework.md).

## Choose a configuration

| Workflow | Reference | Optimized |
|---|---|---|
| SD1.5 DMD2 | FSDP2: `recipes/dmd2_sd15/training.fsdp2.json`; single-device FP32: `training.gan.json`; FP32 DDP: `training.ddp-fp32.json` | `recipes/dmd2_sd15/training.optimized.json` |
| Official Self-Forcing | `recipes/self_forcing_official/training.json`; native FSDP2: `training.fsdp2-reference.json` | `recipes/self_forcing_official/training.optimized.json` |

Fill in explicit local source, model/asset, prepared-data and output paths.
Install each model's optional dependencies in its own environment:

```bash
pip install -e '.[dmd2]'
# Or, in the Self-Forcing environment:
pip install -e '.[self-forcing-official]'
```

Self-Forcing uses the upstream FlashAttention-2 provider and the compatible
Triton runtime already used by its native backend. Prepare prompt embeddings
offline with the model's text encoder. Training workers read the prepared
conditions and do not load T5. The existing recipe guides describe local asset
and input manifests:
[DMD2](../../recipes/dmd2_sd15/README.md),
[Self-Forcing](../../recipes/self_forcing_official/README.md).

The [DMD2 FSDP2 reference guide](dmd2_fsdp2_reference.md) covers its native
precision policy, sharded checkpoints and distributed export. This reference
uses the same native algorithm and workload as the optimized SD1.5 recipe;
the existing single-device and FP32 DDP recipes remain separate choices.

Check and inspect a request before model construction:

```bash
distillctl config validate /path/to/training.optimized.json
distillctl doctor --config /path/to/training.optimized.json
distillctl capabilities show official-dmd2-resident
distillctl capabilities show official-self-forcing-fsdp2
```

### Distributed launch

Start the same command on each of two eight-GPU nodes, supplying its node rank
and a shared rendezvous address. All ranks use the same resolved configuration.

```bash
# DMD2: set these before torchrun initializes the NCCL group.
export NCCL_ALGO=allreduce:Tree
export NCCL_MIN_CTAS=16

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m distillschedule train /path/to/dmd2.training.optimized.json
```

Self-Forcing's native FSDP2 reference and optimized configurations share the
native `NCCL_CROSS_NIC=1` launch policy. Use a separate launch environment without
the DMD2-specific Tree/min16 settings. The library requires the explicit value
without changing the process environment; this carries over the original policy,
not a separately measured optimization gain.

```bash
# Set this before torchrun initializes the NCCL group, for either FSDP2 recipe.
export NCCL_CROSS_NIC=1

torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m distillschedule train /path/to/self-forcing.training.optimized.json
```

Set `trainer.cycles` to the desired training length. `trainer.checkpoint_every`
controls checkpoint frequency. Resume the complete training state at a cycle
boundary on the same topology:

```bash
# Use the same torchrun topology/environment as the initial invocation.
# Replace the final `train ...` arguments with:
distillctl train /path/to/training.optimized.json \
  --resume /path/to/run/checkpoints/cycle-000100 \
  --cycles 200 --output /path/to/resumed-run
```

Each rank records losses, native iteration/update clocks, input progress and
execution counters. Checkpoints include model roles, Adam state, native RNG and
data progress. Self-Forcing's temporary KV/replay caches are rebuilt on resume.

## What the optimized implementations do

### DMD2

- Keep trainable roles resident with native ready-bucket DDP.
- Use fused AdamW while keeping FP32 master parameters and moments.
- Retain Fake Score epsilon activations instead of replaying that forward.
- Pack real/fake classifier trunks, then evaluate the two original B1 heads and
  losses separately.
- Reuse BF16 casts within one Critic update, invalidating them before the next
  optimizer version.
- Preserve the native precision exceptions: no-grad Generator and classifier
  heads use FP32, gradient compute uses BF16, Teacher uses BF16 and VAE uses FP16.

### Official Self-Forcing

- Shard Generator, Fake Score and Teacher with FSDP2; hold gathered trainable
  weights within a phase and release them before clipping and Adam updates.
- Retain rollout text K/V and reuse Critic attention, FFN-down and projection
  results during checkpoint replay.
- Release Critic rollout KV before Score computation, and Generator KV after
  backward and commit; keep integer cache positions on the host.
- Hoist rollout context, fuse applicable RMS/norm/modulation/pointwise operations,
  and share each Teacher block's gathered weights across the conditional and
  unconditional CFG calls.

The model parameter `optimization_profile` binds this complete implementation.
The normal Program executor calls the native owner directly; no historical
benchmark script, run directory or saved plan is required for direct training.
Use the reference recipe when choosing the original execution path. The six
existing W1–W6 recipes and their saved-plan execution remain available.
