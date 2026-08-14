# Proposed GAN pipeline: persistent 8G + 8Score

This document describes the first **performance-oriented GAN pipeline candidate**. It is intentionally stacked on the official-semantics GAN Route-A baseline and is not a new training algorithm.

## Core idea

Keep the physical placement fixed for the whole cycle:

```text
ranks 0-7   : Generator (Helios-Distilled + G LoRA)
ranks 8-15  : Score (Helios-Base + Critic LoRA + GAN heads)
```

Do not create a third 14B GAN model and do not move the Score backbone between a "teacher" pool and a "critic" pool. The same eight Score copies switch **mode**, not physical location.

This matches upstream structure: GAN heads live on the Score backbone, and upstream GAN-G low-VRAM already computes a cached gradient with respect to the fake latent rather than retaining the Score graph through Generator backward. The pipeline generalizes that boundary to the complete Score-side DMD + GAN-G objective.

## Generator-update step

At step `s % 5 == 0`, first flush the Critic rollout queue. The Score version used by Generator DMD/GAN-G must be the committed `C_s` state.

Because each role has eight ranks while the matched logical global batch is 16, the step has two micro-waves.

For each micro-wave:

```text
G rank:
  differentiable official rollout -> x_fake with local autograd graph
  send detach(x_fake) + trace/version identity to paired Score rank

Score rank (read-only parameters):
  official compute_distribution_matching_loss(
      DMD real/fake score + GAN-G cached-grad path
  )
  compute dL_score / dx_fake
  send latent gradient + scalar logs back to paired G rank

G rank:
  backward(x_fake, returned dL/dx_fake)
  accumulate G gradients
```

After exactly two waves, perform **one** Generator optimizer/scheduler step. Score/Critic/GAN parameters are frozen during this service and their version must not change.

The important consequence is that Score never has to move onto Generator GPUs. This removes the largest transition that a time-sharing GAN implementation would otherwise introduce.

The branch now has a concrete bidirectional P2P wire contract for this boundary: request/reply headers bind run nonce, logical step, micro-wave, paired ranks, G/C versions, trace SHA and BF16 latent/gradient checksums. The transport refuses version/trace/checksum drift and requires all outstanding P2P work to be flushed at barriers.

## Same-step Critic phase

The official order is Generator update first, then Critic. Therefore the Critic rollout for the Generator-update step must use the newly committed Generator version.

Immediately after the G commit, the same eight Score ranks re-enable Critic LoRA/GAN-head gradients. There is no 14B state transfer or process-group redistribution.

For each of two micro-waves:

```text
Generator lane                       Score/Critic lane
--------------                       -----------------
no-grad rollout (G_new)  --------->  denoising forward/backward
                                      GAN-D fake/real + R1/R2 forward/backward
```

The two lanes use a bounded Q2 producer/consumer queue. For one logical step, the Score lane sees two micro-waves but must produce exactly **one** Critic optimizer/scheduler commit.

Upstream low-VRAM GAN already requires denoising backward and GAN/R1/R2 backward to feed one Critic update. With the 8-rank matched layout, the generalized logical-step rule is:

```text
wave0 denoise backward
wave0 GAN/R1/R2 backward
wave1 denoise backward
wave1 GAN/R1/R2 backward
-----------------------
one Critic optimizer/scheduler commit
```

The DeepSpeed boundary helper for this four-backward/one-step rule is implemented. The remaining GPU gate is to prove on the target stack that GAS=2 loss normalization is numerically matched to the official world16 Route-A denominator.

## EMA boundary

The official GAN window is after `ema_start_step=750`. In DMD mode, Helios calls EMA only when `TRAIN_GENERATOR` is true. Therefore:

```text
checkpoint-1000: EMA optimization_step = 800
step1000:         EMA 800 -> 801
steps1001-1004:  no EMA update
step1005:         next EMA update
```

So Critic-only steps do not need a Generator/EMA residency transition. EMA remains an after-Critic barrier only on Generator-update steps. This is both source-faithful and slightly simpler than the earlier provisional every-step EMA schedule.

## Critic-only steps

For steps between Generator updates, `G` is version-stable. This allows the normal producer-consumer pipeline:

```text
time --->

G:      rollout s,w0 | rollout s,w1 | rollout s+1,w0 | rollout s+1,w1 | ...
Score:                C s,w0        | C s,w1        | C s+1,w0      | ...
```

A Critic version advances only after both micro-waves for that logical step. Generated samples can be prefetched because rollout does not depend on the current Critic version; the queue is nevertheless bounded at two payloads to prevent unbounded latent/trace lifetime.

Before the next Generator-update step (`s+5`), the queue must be fully flushed so Generator DMD/GAN-G observes the committed Score state.

## Why this is the primary candidate

Compared with a dynamic `8G + 4FakeScore + 4RealScore -> 8G + 8Critic` layout, persistent 8G+8Score has three advantages:

1. **No 14B redistribution at the 5-step boundary.** Only Critic/GAN-head grad mode changes.
2. **The upstream DMD implementation remains the Score-side math owner.** A Score rank can execute the official Real/Fake Score switches and GAN-G path instead of reimplementing those internals across two physical pools.
3. **The matched global batch is clean.** Two waves × eight ranks × microbatch1 = 16 for both G and Score.

A split 4+4 Score service can still be evaluated later if profiling shows the Generator-update score service dominates. It should be a separate placement experiment, not the first correctness target.

A second performance optimization can also be tested after correctness: double-buffer the two Generator-update micro-waves so G wave1 forward overlaps Score wave0 gradient computation, and then G wave0 backward overlaps Score wave1 computation. This is not enabled by default because retaining two differentiable Generator wave lifetimes may raise A100-40GB activation pressure; it needs its own memory/parity gate.

## Hard correctness barriers

The pipeline is rejected if any of the following occurs:

- Score/Critic parameters or optimizer step change during the Generator score-gradient service;
- Generator commit occurs before both score-gradient replies arrive;
- same-step Critic rollout uses the pre-update Generator version;
- Critic steps once per micro-wave instead of once per logical batch16;
- EMA occurs on post-750 Critic-only steps or before the same-step Critic commit;
- a Generator-update step starts with a non-empty rollout queue or outstanding P2P work;
- trace/RNG identity, batch16, GAN/R1/R2 settings, ZeRO-1, loss weights, or official update predicates differ from the matched GAN Route-A contract.

## Current implementation boundary

Implemented in this branch:

- `src/distillschedule/runtime/gan_pipeline.py`: topology, batch, queue, version and optimizer/EMA barrier state machines;
- `src/distillschedule/runtime/gan_pipeline_deepspeed.py`: two-wave/four-backward/one-Critic-step DeepSpeed boundary;
- `runtime/gan_pipeline_helios.py`: source-faithful Score-side latent-gradient extraction using upstream `compute_distribution_matching_loss`;
- `runtime/gan_pipeline_transport.py`: bidirectional Generator↔Score latent/gradient P2P transport contract;
- `configs/distillschedule/pipelines/helios_gan_world16_8g8score.json`: frozen candidate contract;
- `scripts/distillschedule/inspect_gan_pipeline.py`: exact schedule/version renderer;
- focused refactor regressions for the five-step, two-cycle and DeepSpeed accumulation rules.

Still required before a performance claim:

- integrate the P2P primitives into real world16 Generator and Score model workers;
- extend rank-qualified GAN trace coverage to every crop/R1/noise/random input consumed by the split boundary;
- prove DeepSpeed GAS=2 loss normalization and G/C gradient parity against matched sequential GAN Route-A;
- continuous steps1000–1009 parity, including both Generator boundaries;
- at least three matched unprofiled Route-A vs pipeline timing pairs.
