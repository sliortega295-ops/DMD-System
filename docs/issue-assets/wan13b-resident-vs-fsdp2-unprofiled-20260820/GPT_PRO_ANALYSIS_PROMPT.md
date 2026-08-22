# Prompt for GPT Pro: Wan1.3B Resident vs world16 FSDP2 analysis

You are reviewing a real 2-node x 8-A100 systems experiment in DMD-System PR
#21. Analyze the evidence; do not redesign the workload and do not turn a
failed correctness gate or a profiled diagnostic into a formal speed claim.

Repository and evidence:

- PR: https://github.com/sliortega295-ops/DMD-System/pull/21
- Unprofiled Iteration 321 bundle:
  `docs/issue-assets/wan13b-resident-vs-fsdp2-unprofiled-20260820/`
- Matched Nsight Iterations 326--327 bundle:
  `docs/issue-assets/wan13b-resident-vs-fsdp2-nsight-20260822/`
- Runtime implementation:
  `src/distillschedule/runtime/wan13b_upper_bound.py`
- Worker:
  `scripts/distillschedule/probe_wan13b_upper_bound_world16.py`
- Nsight analyzer:
  `scripts/distillschedule/analyze_wan13b_nsys.py`

Frozen workload:

- real Wan2.1-T2V-1.3B, 30 blocks;
- world16, 2 nodes x 8 A100-40GB;
- B1/rank, global B16, rollout B1, Score packing off;
- one Generator update, five Critic updates, five EMA updates;
- G2/C10/EMA10 -> G3/C15/EMA15;
- LoRA rank256, BF16, GPU residuals/EMA/optimizer;
- no parameter/activation/optimizer offload, streaming, or model swap;
- Resident keeps both complete backbones on every GPU and mean-all-reduces
  trainable LoRA gradients over world16;
- FSDP2 FULL_SHARD wraps 30 blocks plus root with
  `reshard_after_forward=True`.

Unprofiled diagnostic:

- Resident: 55.8685452938 s;
- Resident same-layout repeat: 57.5958141028 s;
- world16 FSDP2: 66.7620610059 s;
- raw ratio: 1.194985x, FSDP2 19.50% slower;
- Resident peak allocated/reserved: 18.434/19.332 GiB;
- FSDP2 peak allocated/reserved: 3.533/6.014 GiB;
- each arm individually passed 16/16 workload and capacity gates;
- cross-placement numerical gate failed, so formal timing was NOT RUN.

Numerical evidence, normalized L2:

| State class | Resident repeat | Resident vs FSDP2 | Cross/self ratio |
|---|---:|---:|---:|
| rollout | 0.0138253 | 0.0161246 | 1.17x |
| Generator | 2.10465e-6 | 9.77778e-5 | 46.46x |
| Critic | 4.54647e-6 | 6.92047e-5 | 15.22x |
| Generator Adam | 1.94008e-4 | 1.16406e-2 | 60.00x |
| Critic Adam | 5.86834e-4 | 2.37021e-2 | 40.39x |

Matched Nsight diagnostic, mean exact-range wall per GPU:

| Metric | FSDP2 | Resident |
|---|---:|---:|
| Profile wall | 77.336 s | 70.650 s |
| non-NCCL kernel union | 21.852 s | 25.171 s |
| NCCL union | 41.441 s | 6.336 s |
| NCCL not overlapped with same-GPU non-NCCL kernels | 29.502 s | 6.336 s |
| memcpy union | 9.831 s | 9.526 s |
| no traced GPU activity | 20.369 s | 29.617 s |
| D2H | 25.3166 GiB/rank | 25.3166 GiB/rank |
| H2D | 12.8332 GiB/rank | 12.8193 GiB/rank |

Collective not-overlapped values:

- FSDP2 all-gather: 18.192 s/GPU;
- FSDP2 reduce-scatter: 10.965 s/GPU;
- FSDP2 all-reduce: 0.345 s/GPU;
- Resident all-reduce: 6.336 s/GPU;
- Resident all-gather/reduce-scatter: zero.

Exact exclusive reconciliation:

- FSDP2 removes 35.105 s/GPU of NCCL union when changed to Resident;
- Resident has 15.258 s more compute-only, 3.914 s more copy-only, and 9.247 s
  more no-traced time;
- `35.105 - 15.258 - 3.914 - 9.247 = 6.686 s`, matching the profiled wall gap.

Important definitions:

- `non-NCCL kernel union` is not FLOPs, SM utilization, Tensor utilization, or
  MFU. It includes all non-NCCL CUDA kernels, including optimizer, EMA,
  pack/cast/copy/layout helpers and framework kernels.
- `NCCL not overlapped with compute` is not causal removable critical-path
  time.
- `no traced GPU activity` is not automatically CPU idle.
- Nsight wall must not be used as the unprofiled speed denominator.

Please answer:

1. Is the 19.50% unprofiled gap scientifically usable now, and what exact claim
   boundary is valid?
2. Why can Resident show more non-NCCL kernel union even though it removes
   FSDP2 all-gather/reduce-scatter? Separate model math from full replicated
   LoRA optimizer/EMA, gradient scaling, parameter layout/kernel selection,
   framework copies, overlap, and profiler perturbation.
3. From the existing source and evidence, identify which physical operations
   differ between Resident and FSDP2. Cite concrete files/functions.
4. Explain why 29.502 s/GPU of FSDP2 NCCL-not-overlapped becomes only a 6.686 s
   profiled wall gap and why neither value predicts the 10.894 s unprofiled
   single-run gap directly.
5. Analyze the identical 25.3166 GiB/rank D2H. What does it rule out, what does
   it not rule out, and what source attribution is still missing?
6. Diagnose the numerical mismatch. Rank likely causes: reduction order,
   checkpoint/recompute order, DTensor layout/materialization, optimizer state
   layout, and nondeterminism. Do not recommend relaxing thresholds merely to
   make the comparison pass.
7. Propose the smallest next evidence collection, not an optimization:
   - existing-SQLite per-phase kernel-signature count/duration delta;
   - operation markers for model math, FSDP materialization, gradient
     pack/scale, clip, AdamW and EMA;
   - same-layout repeat noise contract;
   - a correctness-equivalent unprofiled paired A/B only after parity is fixed.
8. Produce a table with PROVEN, STRONG INFERENCE and UNRESOLVED conclusions,
   followed by a prioritized next-step list.

Do not propose changing batch, trace, update counts, model calls, precision,
loss, optimizer semantics, or placement contract. Do not call the Resident arm
an exact theoretical upper bound until the numerical and physical-work
boundaries are resolved.
