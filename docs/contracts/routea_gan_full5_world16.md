# Frozen contract: Route-A GAN-on Full5 world16 baseline

This contract defines the conservative **GAN-on baseline** used before GAN pipeline work. Model math and update order remain owned by a pinned Helios `train_helios.py`; DMD-System supplies world16 launch, offload lifecycle, trace/checkpoint binding and evidence gates.

## Why the window is 1000–1004

The official Helios GAN configuration keeps `gan_start_step=1000` and defines `TRAIN_GENERATOR = global_step % dfake_gen_update_ratio == 0` with `dfake_gen_update_ratio=5`. Therefore the first complete GAN-active frequency cycle is steps **1000, 1001, 1002, 1003, 1004**. The baseline must not rewrite `gan_start_step` merely to reuse an earlier checkpoint.

A source-faithful run requires a canonical GAN-on `checkpoint-1000`.  The
systems experiment may instead use one immutable artifact explicitly marked
`systems_only_gan_bootstrap`: real Generator/Critic/Adam/scheduler tensors come
from a validated earlier checkpoint, official GAN heads are initialized once
from a fixed seed, and all optimizer buffers are materialized before timing.
It is never described as an official checkpoint-1000 continuation.  Both
baseline and pipeline must fresh-restart the same artifact.

A rank-qualified operation trace covering 1000–1004 is required in either
case.  For the systems bootstrap, source-shaped steps10–14 tensors may be
remapped to logical1000–1004 only when GAN noise, crop, and R1 epsilon values
are added as explicit operation-keyed tensors and actual consumption is
receipted.

## Frozen workload

| Field | Value |
|---|---|
| Hardware | 2 nodes × 8 A100-40GB; world16 |
| Resolution | 384×640 |
| Latent | `[B,16,9,48,80]`, micro-batch 1/rank |
| Global batch | 16, GAS=1 |
| Precision | BF16 |
| Logical steps | 1000–1004 |
| Branches | DMD + GAN + EMA on; Reward and DMD-VAE-decode off |
| Generator frequency | every 5 steps |
| Critic frequency | every step |
| GAN activation | official `gan_start_step=1000` |
| EMA predicate | before step750 every step; at/after step750 in DMD mode only when `TRAIN_GENERATOR` |
| Initial versions | G200 / C1000 / EMA optimization_step 800 |
| Final versions | G201 / C1005 / EMA optimization_step 801 |
| Training backend | ZeRO-1 + CPU optimizer offload; no parameter offload fallback |
| Pipeline overlap | none |
| Prefetch | 0 |

The Generator version corresponds to its every-five-step update count and Critic version to its every-step update count. EMA must use its actual `EMAModel_Zero3.optimization_step`; after the post-750 predicate this is no longer equal to the global logical step. For a source-faithful continuation from step0, `checkpoint-1000` has 750 pre-start EMA calls plus 50 Generator-gated calls from steps750–999, hence EMA optimization_step=800.

## Official GAN schedule

Step 1000 is the Generator-update step. The pinned Helios control flow is:

```text
Generator rollout
→ DMD Generator loss
→ GAN-G loss through the Score/GAN discriminator path
→ Generator backward/update (G200→G201)
→ Critic denoising loss + GAN-D real/fake + R1/R2
→ one joint Critic optimizer update (C1000→C1001)
→ EMA update (800→801)
```

Steps 1001–1004 skip both Generator and EMA updates after `ema_start_step=750` but keep the official Critic/GAN-D path:

```text
Critic denoising loss + GAN-D real/fake + R1/R2
→ one Critic optimizer update
→ no EMA update
```

GAN is **not a third 14B backbone**. The discriminator heads live on the Helios-Base Score backbone. During the Generator GAN-G phase the official code freezes GAN-Critic trainables; it re-enables them before the Critic/GAN-D phase.

In official GAN low-VRAM mode the denoising backward and GAN/R1/R2 backward are accumulated into **one** Critic optimizer update. DMD-System must not split this into two optimizer steps or reorder the losses.

## Conservative offload baseline

The default functional denominator intentionally optimizes for feasibility and semantic clarity rather than speed:

- all 40 frozen Generator blocks use host-shared immutable streaming, window 1;
- all 40 frozen Score blocks use the same policy;
- CPU optimizer offload is enabled for Generator and Score/Critic trainables;
- Generator-selected saved activations use the existing 32-MiB offload threshold;
- prefetch distance is zero and no G/C/GAN overlap is allowed;
- VAE temporary residency follows the official GAN low-VRAM path for real-latent preparation and is returned to CPU afterward.

A separate capacity/correctness search may vary only the frozen streaming set to determine the measured-minimum GAN block-offload baseline. Search candidates are never formal timing results and may not change loss, batch, activation policy, ZeRO stage, trace, version or EMA predicates.

The public Helios GAN example uses its own DeepSpeed configuration. This systems baseline deliberately keeps the already validated Route-A **ZeRO-1 CPU-optimizer substrate**; source-faithfulness applies to model math, branch predicates, loss composition, optimizer-update count/order and EMA order. A known-bad ZeRO-2 fallback is not authorized.

## Trace and timing fairness scope

Formal baseline/pipeline comparison uses logical sample slots 0–15 rather than raw physical-rank identity. A shared RNG seed alone is not accepted after rescheduling into two eight-rank waves. The real workers must emit receipts proving that each logical sample consumed the same source/history, rollout noise sequence, DMD timestep/noise, GAN-G crop/noise, Critic timestep/noise, GAN-D/R1 crop/epsilon, and the same three-stage exits/complete DiT-forward counts. Fixed Score-call inventory is also frozen: step1000 uses Fake-cond + Real-cond + Real-uncond + GAN-G; every Critic step uses one denoise Score forward plus one packed GAN-D forward containing fake+real+R1 (R2 is disabled).

Formal traces cover only the common production path: every rank must use the same frozen pyramid/denoising exit schedule for the same logical operation. Any rank-local DiT-step schedule is out of scope and fails preregistration.

## Acceptance

A run is accepted only when:

1. both nodes normalize to the same GAN contract and pinned Helios source identity;
2. world16 topology/NCCL passes before 14B construction;
3. rendered config matches the official GAN fields including `gan_start_step=1000`, `ema_start_step=750`, low-VRAM GAN mode, GAN weights/hooks and R1/R2 settings;
4. all 16 ranks bind to the preregistered synchronized trace and execute steps1000–1004;
5. observed versions are exactly G200/C1000/EMA800 → G201/C1005/EMA801 with 1 Generator, 5 Critic and **1 EMA** update, at step1000 only;
6. stochastic/DiT and model-call receipts satisfy the matched fairness contract before any speedup is claimable;
7. `checkpoint-1005` is non-empty;
8. activation/streaming final live residency is zero;
9. both workers return zero and no CUDA/cgroup OOM/OOM-kill is observed.

The first accepted run is a functional/diagnostic baseline. Formal timing requires repeated **unprofiled** matched pairs. Nsight Systems is diagnostic attribution only and must use the exact step1000-before-first-model-compute → step1004-after-Critic-commit capture range.