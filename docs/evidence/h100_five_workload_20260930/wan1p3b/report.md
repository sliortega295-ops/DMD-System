# Wan1.3B H100 matched results — controlled diagnostic

Measured HEAD: `11dd0b155cc72572c907a79abc1a66cbdca25392`. Run: `wan_h100_paired_timing_r1`.
Eight H100-80GB GPUs; logical global batch16; original independent B1 slots
`r,r+8`, each loss/2 before one commit. BF16 frozen base/FP32 LoRA, rollout B1,
Score packing OFF, identical immutable checkpoint/trace/negative prompt.

Each fresh process executes two cycles. Cycle1 is excluded warmup; cycle2
contains G1/C5/EMA5. The denominator is maximum physical-rank native cycle2
wall including the registered completion boundary, not GPU-seconds or Nsight.
Final version: G4/C20/EMA20 from G2/C10/EMA10. AB / BA / AB order.

| Pair | FSDP8 baseline (s) | Resident async128/tree/chunk2048 (s) | Ratio |
|---|---:|---:|---:|
| 0 | 56.797855115 | 37.582740035 | 1.511275x |
| 1 | 56.788417506 | 37.509206853 | 1.513986x |
| 2 | 58.960118638 | 37.511148368 | 1.571803x |
| Median | 56.797855115 | 37.511148368 | 1.514159x |

- Baseline CV: 2.175%; candidate CV: 0.112% (sample standard deviation).
- Median paired ratio: 1.513986x; geometric mean paired ratio: 1.532102x.
- 8/8 physical workers covering all16 logical slots, dtype/call/trace/update/finite gates PASS in all six runs. CUDA/host OOM: no observed event; cgroup OOM delta0; no residual GPU contexts.
- Numerical **REVIEW**, not exact-equivalence PASS: 160/160 rollouts fail unchanged strict thresholds, worst normalized L2=0.01582547, minimum cosine=0.999815226. Five sampled final-state components PASS, but sampled coordinates are not complete state. Full-state parity NOT_RUN; cause unresolved.
- Thus **PASS_CONTROLLED_DIAGNOSTIC** under the existing trajectory study policy. Do not use this as strict-equivalent paper evidence without closing the numerical gate.
- Reported memory counters and construction times are in `trials.csv`; baseline peak resets mean they are not a matched full-run peak comparison. Host peak NOT_MEASURED. Driver-free values are end snapshots.
- No profile or H100-specific kernel optimization was used. The existing A100 candidate was reused. Different physical GPU counts prohibit comparing absolute A100/H100 latency as a hardware speedup claim.

## Evidence

Remote raw evidence: `/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/`.
The comparison, all eight original correctness/timing summaries, 64 rank receipts,
commands, health records and SHA index remain in the small local evidence subtree
`artifacts/wan_h100_raw_receipts_r1/`; model/tensor payloads remain remote.
`result.json` and `trials.csv` are derived from, and hash-bound to, the actual receipts.

## Remaining matrix

Only Wan1.3B is measured so far. Helios No-GAN/GAN and LiveWan are waiting for
complete asset gates; H3 additionally lacks original canonical 16-slot initial
LoRA/RNG replay evidence. No result is inferred for those four workloads.
