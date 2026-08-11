# Iteration 294: Bounded pinned transition result

**Status:** Measured bounded negative for the joint candidate; correctness PASS.

## Frozen comparison

Both runs use strict-v4, official 384x640, Step10-A DP8 ZeRO-3 MB1/GAS2/global16, target global 4G+12C DP12 ZeRO-1, identical trace `654d4901...`, versions G2/C10/EMA10 -> G3/C15/EMA15, and unchanged graph/loss/optimizer/EMA/placement.

| Metric | Instrumented baseline | pinned64x2 + G3 bucket128 | Change |
|---|---:|---:|---:|
| Step10-A | 124.911580 s | 120.921975 s | -3.19% (diagnostic only) |
| transition | 49.839245 s | 46.086928 s | -7.53% |
| target H2D max | 33.168865 s | 29.745484 s | -10.32% |
| G3 broadcast max | 10.868068 s | 15.400286 s | +41.71% |
| Critic phase | 165.794245 s | 167.261416 s | +0.89% |
| Full5 | 341.819356 s | 335.253502 s | -1.92% |

The materializer used a bounded 256 MiB pinned arena per switching rank. Across roles, materialization wall time ranged from 16.216932 s to 29.720941 s. The maximum observed device-copy interval was 29.719029 s. Node cgroup peaks were 357,551,136,768 and 171,137,269,760 bytes; no new OOM occurred.

## Correctness

The real-14B run `rb16_f5_iter294_pinned64x2_g3b128_correctness_20260810_001` passed 16/16 rank records, exact frozen identities, Step10-A 16 samples, 60/60 target payloads, G3/C15/EMA15, queue flush, and launcher/cgroup OOM gates.

## Interpretation

Pinned staging is a valid primitive, but the joint preregistered candidate missed the >=10% transition gate. It relieved individual Generator materialization substantially, while eight-rank concurrent node-0 materialization limited the maximum improvement. The 128-MiB G3 bucketing was counterproductive: it reduced collective count but increased the critical-path broadcast interval. Therefore the next bounded test keeps pinned64x2 and restores the already validated legacy fixed-order per-tensor broadcast. This is not formal repeated timing and not a final speedup claim.
