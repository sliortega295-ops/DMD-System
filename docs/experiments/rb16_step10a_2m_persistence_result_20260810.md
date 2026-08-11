# Iteration 296: strict-v4 Step10-A 2M persistence

**Status:** Measured bounded negative for the 10% gate; boundary identity PASS.

One low-overhead Step10-only run used official 384x640, DP8 ZeRO-3, MB1/GAS2/global16, two strict-v4 selected graphs and two backwards, with the sole policy change `stage3_param_persistence_threshold=2,000,000`.

| Metric | strict-v4 baseline | 2M persistence | Change |
|---|---:|---:|---:|
| Generator-rank Step10-A median | 123.115506 s | 116.829453 s | -5.11% |
| world/node max Step10-A | 129.003622 s | 118.859191 s | -7.86% |
| wave0 rollout median | 20.675323 s | 19.705991 s | -4.69% |
| wave0 backward median | 38.368997 s | 36.111826 s | -5.88% |
| wave1 rollout median | 21.036667 s | 19.686933 s | -6.42% |
| wave1 backward median | 39.909053 s | 39.858838 s | -0.13% |

DeepSpeed reported 1,432 persistent parameter objects / 967,692,352 elements. The runtime resource monitor observed node0 GPU peaks of 31,762 MiB each (baseline 31,986 MiB) and node cgroup peak 356,686,761,984 bytes. No OOM occurred.

For all eight Generator ranks and both waves, trace identity, x0 SHA, injected gradient SHA and surrogate loss matched the frozen strict-v4 correctness run exactly. G2->G3 occurred once; C10/EMA10 were unchanged because the stop-after-Step10 contract intentionally did not execute the target phase. Low-overhead mode omits full post-update tensor/Adam export, so exact combined Full5 candidate correctness remains Not Run.

The candidate gives a real but sub-threshold improvement and is not the requested 85--105 s result. The next bounded candidate retains 2M persistence and changes only the ZeRO-3 prefetch/live window so it can prefetch one 351,394,304-element frozen block while retaining the current and next block.
