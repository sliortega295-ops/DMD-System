# Static world16 FSDP2 Full5 baselines

## Goal

These baselines answer a narrow systems question: **what happens if both Helios
14B physical backbones stay sharded on all 16 A100s for the entire Full5, with
no model switching or transition?**

They are intentionally simpler than Route-B/Role-Phase. Every rank owns a
world16 FSDP2 shard of both Generator and Score/Critic. The execution order may
switch from Generator compute to Score compute, but the physical parameter
representation never switches.

```text
world16, every rank

Generator 14B  -- FSDP2 FULL_SHARD, static
Score/Critic   -- FSDP2 FULL_SHARD, static

Full5:
G compute -> Score compute -> G backward/update -> Critic compute/update ...

model swaps        = 0
role transitions   = 0
CPU param streaming= 0
parameter offload  = 0
```

FSDP2 is applied bottom-up to all 40 Helios Transformer blocks and then to the
root module. `reshard_after_forward=True` keeps FULL_SHARD semantics while
using a Transformer block as the transient all-gather/reduce-scatter unit.
Mixed precision is BF16 parameter compute with FP32 gradient reduction. The
optimizer is created after `fully_shard`, so Adam state follows the sharded
DTensor parameters.

## No-GAN baseline

The No-GAN baseline remains the rank16/B16 workload from steps 10--14:

- Step10 performs one differentiable Generator rollout and DMD Generator update;
- every step performs one Critic denoising update;
- EMA executes on steps 10, 11, 12, 13, and 14;
- G2/C10/EMA10 becomes G3/C15/EMA15.

Each rank consumes its own authoritative rank-qualified trace slot. During the
Step10 DMD boundary, the same FSDP2 Score model runs Fake Score with the Critic
adapter enabled, then Real Score conditional/unconditional with the adapter
disabled. No third 14B Teacher copy is introduced.

## GAN-on baseline

The GAN baseline stays B16 so it is comparable to the existing Sequential GAN
denominator. It does **not** inherit Role-Phase's B12 Critic phase.

- logical steps are 1000--1004;
- Step1000 performs official DMD + GAN-G and exactly one Generator update;
- every step performs official Critic denoise + GAN-D + R1 and one Critic update;
- EMA executes only at Step1000;
- logical G200/C1000/EMA800 becomes G201/C1005/EMA801;
- physical optimizer source state remains G2/C10 from the systems-only bootstrap.

For GAN-G, the Score model remains the same trainable FSDP2 model that is later
used by the Critic. The latent derivative is obtained with `autograd.grad` for
the generated latent only; the implementation does not toggle FSDP2 parameter
`requires_grad` state and fails if the read-only boundary creates parameter
gradients.

## Checkpoint and EMA representation

Logical FP32 trainable masters are loaded before FSDP2 sharding. Adam first and
second moments are restored directly into DTensors matching each parameter's
`device_mesh` and `placements`. There is no optimizer CPU offload.

EMA stores only the fixed local FSDP2 shards on CPU. Since EMA is elementwise,
updating a fixed shard is mathematically equivalent to updating the full tensor
and then sharding; no EMA collective is needed per step.

## Measurement

Correctness, formal timing, and profiling are separate runs.

1. `validate`: trace/checkpoint/world16 contract only.
2. `correctness`: one real Full5, requiring 16/16 ranks, exact G/C/EMA update
   inventory, no transition/model swap, and no OOM.
3. `timing`: at least three matched unprofiled Full5 trials after a correctness
   summary has been pinned. Report the median global Full5 wall time.
4. `profile`: a separate Nsight run. Every torchrun child gets its own
   rank-qualified `.nsys-rep`/SQLite. Profiled wall time is diagnostic only.

The Full5 NVTX range is `STATIC_FSDP2_FULL5/<variant>`. GPU compute-kernel
occupancy is reported as:

```text
sum over 16 ranks of compute-kernel interval-union seconds
---------------------------------------------------------
sum over 16 ranks of exact Full5 NVTX wall seconds
```

`analyze_static_fsdp2_nsys.py` also reports NCCL, all CUDA copies, and
no-traced-GPU-activity. Those categories can overlap and must not be added.
No-traced activity is not labeled CPU or scheduler idle.

## Entry points

Per-node launch:

```bash
bash scripts/distillschedule/launch_static_fsdp2_full5_world16_node.sh \
  <node_rank> <run_id> <no-gan|gan> <validate|correctness|timing|profile> \
  <expected_head> <master_addr> <master_port>
```

Summarize one run:

```bash
python scripts/distillschedule/summarize_static_fsdp2_full5_world16.py \
  --run-root <run_root> --variant <no-gan|gan> --output <summary.json>
```

Analyze a profile after both nodes have exported all 16 SQLite files into the
shared profile directory:

```bash
python scripts/distillschedule/analyze_static_fsdp2_nsys.py \
  --profile-dir <run_root>/nsight \
  --variant <no-gan|gan> \
  --output <run_root>/nsight_summary.json
```

## Measured world16 evidence (2026-08-14)

Both variants subsequently passed 16/16-rank correctness and three formal
unprofiled Full5 trials:

| Variant | Formal Full5 trials | Median |
|---|---:|---:|
| No-GAN | 214.066 / 219.849 / 210.786 s | **214.066 s** |
| GAN-on | 348.074 / 347.244 / 368.709 s | **348.074 s** |

The corresponding clean diagnostic profiles, source-attributed Nsight
timelines, D2H analysis, and exact SQLite reconciliation are published in
[`docs/issue-assets/fsdp2-static-full5-20260814/`](../issue-assets/fsdp2-static-full5-20260814/README.md).

The dominant D2H activity is issued from named FSDP phases and
`pt_autograd_*` threads. It is not produced by configured parameter or
activation offload, but the trace does not resolve the internal operation
subsource. Profile copy sums overlap compute/NCCL and are not added to formal
wall time.

## Claim boundary

The completed correctness, formal timing, and diagnostic-profile runs establish
the measurements above for the frozen contracts. They do not establish a
speedup over a role-transition pipeline with a different placement/backend
contract. Profiled latency is diagnostic, compute/NCCL/copy intervals overlap,
and the remaining autograd-owned transfer operation is explicitly unresolved.
