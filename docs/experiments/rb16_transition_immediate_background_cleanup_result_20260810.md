# Iteration 300: immediate background pinned-cache cleanup

**Status:** Measured bounded negative; exact strict-v4 Full5 PASS; formal repeats Not Run.

The only policy change from Iteration 299 was to launch pinned allocator cleanup immediately after each target's direct H2D completed, then join it before Full5 success.  Cleanup was not removed and all workload semantics stayed frozen.

| Metric | Iter299 synchronous cleanup | Iter300 immediate background cleanup | Change |
|---|---:|---:|---:|
| Step10-A world max | 132.899935 s | 124.916147 s | diagnostic variance/recovery |
| Transition | 30.887268 s | 40.980166 s | **+32.68%** |
| Critic phase | 160.708690 s | 163.099019 s | +1.49% |
| Full5 | 328.480532 s | 333.102958 s | +1.41% |
| Direct H2D max | 2.769706 s | 2.946535 s | comparable |
| G3 broadcast max | 8.814932 s | 19.244619 s | contention regression |
| Student initialize max | 20.536355 s | 27.207748 s | contention regression |

All 16 ranks passed exact trace/input/version/queue checks and actual OOM deltas remained zero.  Node cgroup peaks were 716.754 GB and 171.579 GB.  The summary SHA-256 is `76b7bf788ed88d703a3fbe3ad7dc89ffcef8c48830dab0695800e7b6c886c2ef`.

The cleanup threads themselves took 6.47--26.18 s under node contention and completed before the final join (join wait below 0.2 ms).  Starting cleanup immediately after per-rank H2D is therefore the wrong dependency point: while logically asynchronous, it overlaps and slows G3 redistribution, DP12 initialization and the final target-ready fence.  This candidate is rejected.  One materially different bounded schedule remains: launch cleanup only after the global target layout is ready and count any interference or exposed join in the Critic/Full5 interval.
