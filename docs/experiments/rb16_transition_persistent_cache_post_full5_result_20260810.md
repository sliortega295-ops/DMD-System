# Iteration 304: persistent pinned transition cache through Full5

**Status:** Measured, exact correctness PASS; formal repeated timing Not Run.

## Contract

The workload remains strict-v4: official 384x640, Step10-A DP8 ZeRO-3
MB1/GAS2/global16 with two selected graphs and one G2->G3 update, followed by
the fixed 4G+12C DP12 ZeRO-1 Critic phase with C10->C15 and EMA10->EMA15.
Only physical target-cache lifetime changed.

Attempt 01 completed the measured Full5 work but failed when a background
thread trimmed the pinned allocator after the Step15-A boundary. The cgroup OOM
counters did not change. Because that thread was immediately joined, attempt 02
performed the identical trim synchronously on the CUDA-owning main thread,
outside the frozen Full5 timer.

## Exact attempt 02

Run: `rb16_f5_iter304_persistent_cache_post_full5_correctness_20260810_002`

- all 16 ranks: PASS
- trace: `654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`
- versions: G3 / C15 / EMA15
- Step10-A samples: 16
- Critic payloads: 60 produced and 60 consumed
- OOM/OOM-kill delta: 0
- summary SHA-256:
  `9c79353fec00519045c7fc70e4ea35b0b0233ba5896209adaeee3a667a0da366`

| Boundary | World-max seconds |
|---|---:|
| Step10-A | 124.811 |
| Transition | **22.142** |
| Critic phase | 156.063 |
| Full5 to Step15-A boundary | **306.015** |
| Post-boundary allocator trim | 12.605 |
| Full5 + serial trim upper bound | 318.620 |

Node cgroup peaks were 716,569,948,160 and 257,567,371,264 bytes. GPU peaks
were 39,250 MiB and 37,002 MiB.

## Interpretation

The transition target (10--25 s) is met: 49.839 s to 22.142 s is a 55.57%
time reduction (2.25x throughput for that boundary). Relative to the Iteration
293 exact Full5 anchor (341.819 s), the frozen Full5 timer is 10.47% shorter;
even conservatively serializing the separately measured trim gives 318.620 s,
6.79% lower time.

This is not yet a formal repeated performance result. It is one exact
correctness/profile run. Step10-A remains 124.811 s and therefore misses the
85--105 s engineering target; the next iteration treats Step10 independently.
