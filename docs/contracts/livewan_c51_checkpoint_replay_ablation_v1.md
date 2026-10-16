# LiveWan C51 conditional checkpoint replay ablation v1

Status: CPU/source implementation. Pending pinned-runtime CPU validation and
world16 capacity/numerical/timing validation. No GPU result is attached to this
source revision. This is **not** a layout-and-scheduling-only arm.

## Identity and frozen parent

- Parent: `aa30527db5d048e2ab23222eef587cb7cfe909f6`, variant
  `c51-continuous-b16-generator-fused-rms`.
- Opt-in variant: `c51-critic-fa3-recompute-v1`.
- Plan: `configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c51_critic_fa3_recompute_v1.json`.
- Parent plans/contracts/source digests remain unchanged. The child has a new
  plan ID, explicit effective checkpoint policy and separate source identity.
- The registered native Cxx route is extended here; this does not add C51 to
  the separate generic search registry or merge it into main.

Frozen per-cycle contract: 368×640, 81 pixel frames (latent B1×16×21×46×80),
actual global B16, continuous shifted sigma, original trace/RNG/seed, G1/C5/EMA0.
Two consecutive cycles in one process, original checkpoint granularity, FA3
operator/options/dtypes, LoRA/optimizer/shadow versions and update order.
The runtime compares the complete resolved child with its resolved parent,
allowing only the registered checkpoint policy, child ID and contract marker.
The existing trajectory/native two-cycle lowering is applied to both alike.
All canonical C19/C30/C51 refusal cases stay in force.

## Single changed mechanism

On Score ranks only, for Critic grad self-attention inside the same non-reentrant
block checkpoint:

- Full C51: `MUST_SAVE` the pinned FA3 forward outputs in GPU SAC storage.
- Child: `PREFER_RECOMPUTE` the **same** pinned FA3 forward operator. No output
  reuse, CPU offload, substituted backend, skipped forward or broken gradient.

Q/K/V, projections, attention backward, cross-attention recomputation and
checkpoint RNG preservation are unchanged. Teacher/no-grad and Generator
paths are unchanged. Each checkpoint has independent storage; audit ledgers
retain metadata, not tensors. This condition isolates replay-output reuse only
**given** all other C51 optimizations. It does not isolate text K/V, rollout
cast cache, kernels, fused optimizer, role placement or pipeline scheduling.
Host/Gloo transport still exists; this is not a no-CPU-transfer claim.

The installed proxy is chosen in the worker's actual FA3 installation seam.
The serialized effective plan retains `cache_location=none_recompute` and the
child ID; it is never rewritten to claim cached execution. Both node-effective
plans and every rank receipt must agree. Source/operator/option/RNG guards and
ordered replay metadata checks remain strict.

## Required observed receipts (two cycles, per Score rank)

- Checkpoint calls: 300 (10 Critic grad forwards × 30 blocks).
- Original FA3 calls: 300.
- `replay_fa_cache_reads`: **0**.
- `replay_fa_recompute_calls`: **300**.
- Metadata/contract failures: 0; no CPU offload; default RNG preservation.
- Generator ranks: NOT_APPLICABLE, zero Critic checkpoint calls.

Full C51 must instead retain its canonical 300 replay cache reads. The child
never uses the canonical cache-receipt checker to declare fake cache hits.
Counters are checked after real work; plan configuration alone is insufficient.

## Qualification and native C2 entry

Only `trajectory` (exactly two cycles) and `two-cycle-timing` are registered.
No capacity shortcut, profiler, one-cycle or longer-trajectory entry is added.
Continue using the existing world16 controller/node lock/cleanup workflow.

On the **same new clean source HEAD**, first collect a C51 control trajectory
and child trajectory, each with two cycles. Use the existing comparator with:

```
--reference-variant c51-continuous-b16-generator-fused-rms
--pipeline-variant c51-critic-fa3-recompute-v1
```

The existing two-cycle timing-gate builder accepts the same pair. Its new
`distillschedule.livewan_conditional_replay_timing_gate.v1` schema identifies
reference/candidate as two pipeline implementations, not a fake baseline.
It keeps the original comparator, all 32 rank-cycle receipts/arm, exact source,
assets, seed, state anchors, numerical REVIEW and extreme-drift thresholds.
It cannot authorize an unrelated variant. Old v2 gates/defaults are unchanged.
The child cannot use a canonical v2 gate or a historical source's proof.

Only after the qualification/capacity gates pass, use one fresh unprofiled
native two-cycle control and one child timing arm, consuming the same gate,
with their matching arm IDs. Extra replay compute is inside the original timer.
C1 is warm-up; compare C2. Report the conditional marginal `T_child - T_full`,
not a separately additive share of total acceleration. Old aa30527 values
62.4535078125/42.38958984375 seconds are historical references only.

## Evidence boundary and budget

The current package authorizes **zero GPU launches**. The original package has
at most four starts remaining, not six new starts; night total remains 20.
A possible later sequence is two qualification starts, CPU comparison/gate,
then one native timing pair. Supervisor acceptance and fresh node ownership,
flock, environment and original capacity admission are prerequisites. Failures
count against the same budget; no loosened margin, numerical gate or retry reset.

CPU tests use real non-reentrant checkpoint/autograd and compare forward,
gradient, Adam update/state, RNG and actual operator execution counts. CUDA
metadata/operator loading alone is substituted in the CPU fixture. This does
not prove the real FA3 CUDA operator or world16 parity/capacity. Production
replay-ledger auditing is tied to pinned Torch 2.10. A newer CPU SAC API must not
silently be treated as that validation.
