# Iteration 301: late-start pinned-cache cleanup

**Status:** Transition target achieved, but bounded end-to-end negative; exact strict-v4 Full5 PASS; formal repeats Not Run.

Cleanup was held until the global target layout was ready and launched at Critic-phase entry.  It was joined before Full5 success.  This removed host deregistration from G3/DP12 setup without deleting it.

| Metric | Iter293 pageable | Iter299 immediate cleanup | Iter301 late cleanup |
|---|---:|---:|---:|
| Step10-A world max | 124.911580 s | 132.899935 s | 141.102216 s |
| Transition | 49.839245 s | 30.887268 s | **23.837731 s** |
| Critic phase | 165.794245 s | 160.708690 s | 184.964888 s |
| Full5 | 341.819356 s | 328.480532 s | 350.064454 s |

The transition is 52.17% below the pageable baseline and 22.82% below Iter299, satisfying the 10--25 s target.  G3 broadcast max fell to 6.578637 s and Student initialization max to 9.787566 s, confirming that cleanup no longer harms target setup.

However, cleanup then contended with Critic execution: cleanup took 6.40--20.59 s and Critic phase increased 15.09% versus Iter299.  Full5 became 6.57% slower than Iter299 and exceeded the preregistered 5% guardrail.  Therefore late cleanup is a mechanism-positive but end-to-end-negative Pareto point, not the final plan.

Correctness passed on all 16 ranks with exact frozen identities, G3/C15/EMA15, 60/60 payloads and flushed queues.  Node cgroup peaks were 714.702 GB and 173.751 GB; actual OOM deltas were zero.  Summary SHA-256: `74199aec519b2bde82e77d7d5b300dd11af571515c28e2875c3ef76f882cee6c`.

Transition cleanup scheduling is now closed: synchronous cleanup (Iter299) has the best measured Full5, while late cleanup is the only measured <=25 s transition but loses end-to-end.  No further thread/stagger/bucket tuning is justified without a multi-cycle cache-reuse contract.
