# Iteration 327: matched Resident Nsight diagnostic

## Status

- Run ID: `iter327_wan13b_resident_nsys_matched_20260822`
- Status: **PASS_DIAGNOSTIC_ONLY**
- Instrumentation/analyzer HEAD: `18a26bcd046adba49b30630a12e6060c8d193494`
- Matched FSDP2 profile: Iteration 326, workload HEAD `56e3d2b4522298d569a278e656a5ea602cc70785`, analyzer HEAD `c47135aef0d76ebf7eeb70ed874e2466393b8f92`
- Nsight Systems: `2024.6.2.225-246235244400v0`, isolated under `/home/lyy/Helios/tools/`
- Numerical cross-placement parity remains waived only for this diagnostic. No correctness or speedup claim is made.

## Frozen workload and validity

Both captures use real Wan2.1-T2V-1.3B, world16, B1/rank/global B16, Score packing off, G1/C5/EMA5, final G3/C15/EMA15, GPU residuals, GPU EMA, and no parameter/activation/optimizer offload, streaming, or model swap. The only workload variable is placement: `world16-fsdp` versus `resident`; Resident retains world16 mean LoRA-gradient all-reduce.

Resident profile validity: 16/16 workload receipts, 16/16 exact-range receipts, 80 unique rank-step rollout identities, two non-empty reports/SQLite files, 8/8 devices on each node, unresolved-kernel duration 0%, and zero CUDA/cgroup OOM delta. No GPU process or lock holder remained after teardown.

## Whole-window matched comparison

Denominator below is the mean of 16 per-rank exact-range walls in each **profiled** run. It is not a formal native latency denominator.

| Metric | FSDP2 | Resident | FSDP2 - Resident |
|---|---:|---:|---:|
| Profile exact-range wall / GPU | 77.336 s | 70.650 s | **6.686 s** |
| Non-NCCL compute union | 21.852 s | 25.171 s | -3.319 s |
| NCCL union | 41.441 s | 6.336 s | **35.105 s** |
| NCCL hidden by same-GPU compute | 11.939 s | 0.000 s | 11.939 s |
| NCCL not overlapped with same-GPU compute | 29.502 s | 6.336 s | **23.166 s** |
| Memcpy union | 9.831 s | 9.526 s | 0.305 s |
| No traced GPU activity | 20.369 s | 29.617 s | **-9.247 s** |

The matched profiled wall ratio is 1.0946x (FSDP2 9.46% slower; Resident reduces the profiled window by 8.64%). This is diagnostic only. The older unprofiled clue was 66.762 s versus 55.869 s, or FSDP2 19.50% slower / Resident 16.32% shorter; neither is a formal paired result because the numerical gate failed.

## Exact exclusive reconciliation

Per GPU, changing FSDP2 to Resident removes **35.105 s** of NCCL union, but it does not remove 35.105 s of wall:

- FSDP2 has 18.947 s more `NCCL-only`, 11.939 s more `compute+NCCL`, and 4.219 s more `NCCL+copy`.
- Resident has 15.258 s more `compute-only`, 3.914 s more `copy-only`, and 9.247 s more `no-traced activity`.
- `35.105 - (15.258 + 3.914 + 9.247) = 6.686 s`, exactly the matched profiled wall difference up to rounding.

Therefore the former 29.502 s/GPU “exposed NCCL” label was not causal critical-path time. Its correct meaning is only `NCCL not overlapped with non-NCCL kernels on the same GPU`.

## Collective and copy findings

| Collective not overlapped with compute | FSDP2 | Resident |
|---|---:|---:|
| All-gather | 18.192 s/GPU | 0 |
| Reduce-scatter | 10.965 s/GPU | 0 |
| All-reduce | 0.345 s/GPU | 6.336 s/GPU |

Resident removes FSDP parameter all-gather/reduce-scatter but pays an explicit world16 LoRA-gradient all-reduce. It does not become communication-free.

D2H is **identical** at 25.3166 GiB/rank in both captures. H2D is 12.8332 GiB/rank for FSDP2 and 12.8193 GiB/rank for Resident. Thus the large D2H is not placement-specific FSDP parameter movement and remains `UNATTRIBUTED_FRAMEWORK_D2H`.

## Stage clues

| Stage | FSDP2 wall | Resident wall | FSDP2 - Resident |
|---|---:|---:|---:|
| G update | 22.352 s | 18.181 s | +4.171 s |
| G rollouts | 16.344 s | 14.309 s | +2.035 s |
| C step10 | 10.214 s | 6.500 s | +3.714 s |
| C step11 | 6.934 s | 6.583 s | +0.351 s |
| C step12 | 6.907 s | 6.635 s | +0.271 s |
| C step13 | 7.213 s | 6.911 s | +0.302 s |
| C step14 | 6.548 s | 6.656 s | -0.108 s |

EMA subranges are slower in the Resident profile by about 4.12 s in aggregate and partially offset the G/C gains. These stage ranges are node-controller time windows applied to all local devices; without per-rank PID/TID-correlated NVTX they are clues, not exact per-rank causal ownership.

## Conclusion

The matched profile resolves the apparent contradiction. FSDP2 creates a large amount of NCCL activity, but only a fraction becomes net world wall because Resident exposes more compute-only, copy-only, and especially no-traced time, and still performs LoRA gradient all-reduce. Nsight profiling also perturbs the placements differently: profiled Resident/FSDP2 windows cannot replace the unprofiled A/B denominator.

The next smallest diagnostic question is source-attributing Resident's additional 9.247 s/GPU no-traced region and the placement-invariant 25.3166 GiB/rank D2H. No optimization is implemented here.
