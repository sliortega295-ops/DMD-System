# Wan1.3B Resident optimization ladder (Iterations 340-349)

## Status and claim boundary

This document publishes the code and evidence chain used to reduce the
Wan1.3B Resident No-GAN Full5 diagnostic wall from `36.825722 s` to
`26.227823 s` on 16 A100-40GB GPUs.

The final candidate passes the frozen semantic/capacity gates on all 16 ranks,
but the timing observations below are bounded, single-pair diagnostics. They
are **not** a three-pair formal speedup result. The independent-run strict
numerical comparator remains at the previously observed noise floor, so the
result is explicitly labelled diagnostic-only.

## Frozen workload

- 2 nodes x 8 A100-SXM4-40GB, world size 16.
- No-GAN Full5, B1/rank, global batch 16, Score packing disabled.
- One Generator, five Critic and five EMA updates; final `G3/C15/EMA15`.
- BF16 frozen backbone and FP32 LoRA arithmetic.
- Identical checkpoint/bootstrap, rank-qualified operation trace, RNG, input,
  model calls, loss, update order and timer boundary.
- Generator and Critic residuals remain resident on GPU.
- No parameter/activation offload, block streaming or model swapping.

Final-run identities:

- bootstrap SHA256: `6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d`
- operation-trace bundle SHA256: `654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`
- Helios Transformer SHA256: `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`
- model identity SHA256: `d52cf9e2d04451c018ac50c4ee1043c4ceba4014e51549bfda2ea5a123c65f54`

## Optimization ladder

| Iteration | Candidate / comparison | Full5 seconds | Result |
|---|---|---:|---|
| 340 | `resident-clean` | 36.825722 | PASS diagnostic |
| 340 | `resident-async128` | 35.416233 | `-3.83%`; promote |
| 342 | async128 control | 34.785844 | PASS diagnostic |
| 342 | async256 | 34.788327 | no gain; reject |
| 343 | async128 AUTO | 34.957049 | control |
| 343 | async128 Tree | 34.749601 | `-0.59%`; promote |
| 344 | Tree + chunk512 | 34.473506 | control |
| 344 | Tree + chunk2048 | 27.233468 | `-21.00%`; promote |
| 345 | same-HEAD eager | 27.181843 | control |
| 345 | no-grad block `torch.compile` | 33.730888 | `+24.09%`; reject |
| 348 | cast-cache OFF | 27.050073 | control |
| 348 | exact one-entry input-cast cache | 27.166219 | `+0.43%`; reject |
| 349 | fused-epilogue OFF | 27.052164 | control |
| 349 | exact no-grad LoRA fused epilogue | **26.227823** | `-3.05%`; winner |

The first and final observations differ by `10.597899 s` (`-28.78%`,
`1.404x`). Because they were not collected as one formal paired experiment,
that number describes the optimization chain rather than a formal speedup.

## What changed

### 1. Clean Resident runtime

`5c04ddd` removes forced synchronization and repeated housekeeping from the
timed path, makes adapter/train-mode transitions idempotent, reuses the five
Critic entries, and defers finite checks to the correctness path.

Primary code:

- `src/distillschedule/runtime/wan13b_fair.py`
- `scripts/distillschedule/probe_wan13b_upper_bound_world16.py`
- `runtime/gst_pool_helios.py`
- `runtime/critic_pool_helios.py`

### 2. Gradient-ready asynchronous LoRA reduction

`13bbd3d` adds deterministic 128 MiB FP32 LoRA gradient buckets and launches
SUM all-reduce on a dedicated communication stream as gradients become ready.
The final divide-by-16 convention remains unchanged. Iteration 342 proved that
increasing the cap to 256 MiB did not help.

Primary code:

- `src/distillschedule/runtime/wan13b_fair.py`
- `configs/distillschedule/baselines/wan13b_nogan_world16_fair.json`

### 3. NCCL Tree diagnostic arm

`acb66f4` adds a controlled `NCCL_ALGO=Tree` arm. A representative GPU0
diagnostic reduced NCCL union from `6.43268 s` to `2.73705 s` and NCCL exposed
versus non-NCCL kernels from `3.22120 s` to `1.37264 s`. Profiled wall time was
not used as the speed denominator.

### 4. Larger exact modulation tile

`08ca5fe` and `1fe2f0b` add the 2048-token modulation/residual chunk arm and
its fail-closed receipts. This was the dominant gain: `34.473506 s` to
`27.233468 s`, with essentially unchanged memory and no workload change.

### 5. Kernel attribution and exact no-grad LoRA epilogue

`c64d43a` establishes the PEFT-callsite -> ATen/runtime -> CUDA-kernel
attribution chain. Across the 28 Generator no-grad calls on rank 0 it observed:

| Operation family | Kernel sum |
|---|---:|
| BF16 -> FP32 LoRA input casts | 0.507424 s |
| FP32 -> BF16 output casts | 0.390234 s |
| LoRA scale/add pointwise | 1.270473 s |
| LoRA A/B GEMMs | 1.166739 s |
| Frozen BF16 base GEMMs | 1.661211 s |

`c9b7c0c` implements a no-grad-only Triton epilogue which performs FP32 scale
and add followed by the BF16 rounded store without materializing the eager
scale/add/output-cast chain. `67b2880` keeps the 112 FP32-base time-embedding
calls on the reference path. The final run fused 8,540 real LoRA invocations
per rank and recorded zero unsupported fallbacks.

Primary code:

- `src/distillschedule/runtime/wan13b_lora_cast.py`
- `src/distillschedule/runtime/wan13b_lora_epilogue.py`
- `scripts/distillschedule/summarize_wan13b_fair_world16.py`

## Kernel evidence

The Iteration 346 NCU audit sampled real workload kernels:

| Kernel family | SM SOL | Tensor active | DRAM SOL | Interpretation |
|---|---:|---:|---:|---|
| dominant BF16 GEMM | 91.60% | 91.60% | 22.32% | Tensor path near its sampled ceiling |
| FlashAttention forward | 64.27% | 64.27% | 2.82% | moderate kernel headroom |
| BF16/FP32 direct copy | 45.04% | 0% | 69.16% | bandwidth-bound; eliminate/fuse |

These are per-kernel NCU measurements, not end-to-end MFU. Full5 MFU remains
`NOT_AVAILABLE_NO_VALIDATED_FLOP_INVENTORY`.

## Final candidate evidence

- Run: `iter349_wan13b_fused_epilogue_candidate_timing_r5_20260824`.
- Source before this publication-only commit:
  `67b2880901abece947732ba5de45a5c2685ab274`.
- 16/16 semantic receipts PASS; 80 rank-step rollout identities.
- Final state `G3/C15/EMA15`; no CUDA/cgroup OOM.
- Full5 world maximum: `26.227822833 s`.
- Peak allocated: `23,508,250,624 B` (`21.89 GiB`).
- Peak reserved: `24,022,876,160 B`.
- Minimum driver free: `17,421,434,880 B`.
- Strict independent-run parity remains outside the preregistered threshold
  (worst rollout normalized L2 `0.0157532`, minimum cosine `0.999818`), hence
  `DIAGNOSTIC_ONLY` and no formal three-pair claim.

## Rejected and already-completed paths

- Async256: no measurable benefit; do not continue bucket-size search without
  new evidence.
- Blanket no-grad block `torch.compile`: slower and outside strict parity.
- Python input-cast cache: removed real copies but its bookkeeping cost was
  larger than the saved work.
- Schedule reorder is already active: Generator commit, precompute five B1
  no-grad rollouts, then five ordered Critic commits. Re-implementing it would
  duplicate completed work.

## Reproduction and code review

The accepted and rejected placements are registered in
`configs/distillschedule/baselines/wan13b_nogan_world16_fair.json`. The public
entry point is `scripts/distillschedule/run_wan13b_fair_world16.py`; its
launcher and summarizer enforce the placement-specific runtime receipts.

The machine-readable result is
`artifacts/manifests/iteration_340_349_wan13b_resident_optimization_20260824.json`.

