# Iteration 299: persistent-pinned transition target

**Status:** Measured partial positive; exact strict-v4 Full5 PASS; formal repeats Not Run.

The ten cold-preconstructed switch targets were converted from pageable CPU state to private persistent pinned state during excluded cold preparation.  The timed transition then copied each registered tensor directly on a dedicated H2D stream and synchronously released the pinned allocator cache before declaring the target ready.  Step10-A policy, legacy G3 redistribution, graph, trace, versions, backend, placement, optimizer and EMA were unchanged.

| Metric | Iter293 pageable baseline | Iter299 persistent pinned | Change |
|---|---:|---:|---:|
| Step10-A world max | 124.911580 s | 132.899935 s | +6.39% |
| Transition world max | 49.839245 s | 30.887268 s | **-38.03%** |
| Critic phase | 165.794245 s | 160.708690 s | -3.07% |
| Full5 | 341.819356 s | 328.480532 s | -3.90% |
| Target direct-H2D CUDA interval max | Not isolated | 2.769706 s | diagnostic |
| Synchronous pinned-cache trim max | Not applicable | 11.277463 s | new critical-path cost |

Correctness passed on all 16 ranks: exact frozen trace/input identities, 16 Step10-A samples, 60 generated and consumed target payloads, G3/C15/EMA15, queue flush and no new OOM.  The summary SHA-256 is `9940fb0a0610391306ea9486ab528c96a3b25b433ccfce7ce762372439ecec01`.

Cold pin preparation was excluded from execution timing but fully reported: 10 switch targets, 31.186--33.668 GB each, 264.383 GB runtime-owned pinned state on node 0 and 62.372 GB on node 1.  Node cgroup peaks were 739.236 GB and 178.213 GB; actual OOM deltas were zero.  Cold pin preparation took 31.025--42.415 s per switch rank.

The candidate crosses the preregistered 10% transition gate but misses the 10--25 s target.  Causal data show why: direct DMA itself is only 2.36--2.77 s, while synchronous `host_emptyCache`/allocator trim costs 6.13--11.28 s on the target-ready critical path.  Since copied pinned backing has no remaining semantic consumer, the next bounded question is whether cleanup can execute concurrently with the 160.7-second Critic phase and be joined before Full5 completion.  This is a residency/scheduling change, not removal of cleanup or a workload change.
