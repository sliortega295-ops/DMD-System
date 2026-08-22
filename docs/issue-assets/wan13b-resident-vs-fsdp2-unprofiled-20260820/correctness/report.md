# Iteration 321: Wan1.3B resident vs world16 FSDP2 correctness gate

## Decision

**NO-GO for formal timing.** Both placements individually completed the frozen world16 No-GAN Full5 contract, but the pinned cross-placement numerical gate failed. No threshold was relaxed and no formal timing/Nsight was launched.

## Exact substrate

- HEAD: `a2cdefbebc90adca97fc3a9b4b90d26f0b1b5841`
- bootstrap: `6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d`
- trace: `654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`
- model identity: `d52cf9e2d04451c018ac50c4ee1043c4ceba4014e51549bfda2ea5a123c65f54`
- workload: world16, B1/rank, global batch16, G1/C5/EMA5, G2/C10/EMA10 -> G3/C15/EMA15, Score packing off
- model: real Wan2.1-T2V-1.3B, 30 blocks, Generator/Critic LoRA rank256

## Capacity and exact contracts

| Arm | 16 ranks | Versions/updates | Unique rollout identities | Max allocated | Max reserved | Min driver free |
|---|---:|---|---:|---:|---:|---:|
| Resident | PASS | PASS | 80 | 18.434 GiB | 19.332 GiB | 19.245 GiB |
| world16 FSDP2 | PASS | PASS | 80 | 3.533 GiB | 6.014 GiB | 32.555 GiB |

No CUDA or cgroup OOM delta occurred.

## Numerical evidence

Pinned P0 thresholds failed for 80/80 rollout rows and all four final-state classes. A bounded same-layout Resident repeat also fails the absolute P0 threshold, proving the default threshold is tighter than the substrate's repeat noise for rollout tensors. However cross-placement state drift is materially larger than same-layout drift:

| Aggregate class | Resident repeat normalized L2 | Resident vs FSDP normalized L2 | Ratio |
|---|---:|---:|---:|
| rollout | 0.0138253 | 0.0161246 | 1.17x |
| generator | 2.10465e-06 | 9.77778e-05 | 46.46x |
| critic | 4.54647e-06 | 6.92047e-05 | 15.22x |
| generator_optimizer | 0.000194008 | 0.0116406 | 60.00x |
| critic_optimizer | 0.000586834 | 0.0237021 | 40.39x |

Rollout aggregate cosine: self `0.999904431`, cross `0.999870627`. Cross-placement mismatch therefore cannot be accepted merely by pointing to same-layout nondeterminism.

## Diagnostic-only elapsed windows

The correctness runs included diagnostics/evidence writing and are **not formal timing**:

- Resident: `55.868545 s`
- Resident repeat: `57.595814 s`
- world16 FSDP2: `66.762061 s`
- raw diagnostic ratio: `1.194985x` (FSDP `19.50%` slower)

These values are retained only as a clue; the formal three-pair driver was fail-closed by correctness.

## Failures and limits

1. The public Wan checkpoint lacks Helios pyramid patch modules; both arms used the same deterministic systems-only BF16 initialization under seed 20260819. This is not an official Wan training checkpoint.
2. Two integration bugs were fixed before this gate: missing patch modules initially remained FP32, and rollout evidence was constructed but never persisted due misplaced dead code.
3. The current experiment answers capacity and exposes a layout-induced numerical difference. It does not yet provide a claimable FSDP efficiency number or a Helios14B extrapolation.

## Next highest-value question

Identify whether the cross-layout drift comes from the Resident explicit LoRA gradient mean versus FSDP2 reduction/recompute order, without relaxing the frozen numerical contract. Until that is resolved or a separately preregistered same-layout noise contract is authorized, formal paired timing remains `NOT_RUN`.
