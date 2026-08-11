# Iteration 293: strict-v4 live-transition causal profile

## Result

**Measured real-14B correctness/profile PASS; formal timing Not Run.** The run preserved the strict-v4 384x640 contract and completed all 16 ranks with `G2/C10/EMA10 -> G3/C15/EMA15`, 60/60 payloads, and an empty queue. No policy changed relative to the validated continuous executor.

| phase | seconds (max relevant rank) |
|---|---:|
| Step10-A | 124.911580 |
| live transition | 49.839245 |
| Critic phase | 165.794245 |
| Full5 | 341.819356 |
| Phase-B world init | 1.616782 |
| Phase-B group creation | 0.666031 |
| switching Critic target H2D | 33.168625 |
| target Generator H2D | 28.799422 |
| G3 redistribution | 10.868301 |
| switching-rank Student DS init | 5.780526 |
| optimizer logical restore | 2.111337 |

## Causal correction

The measurement **refutes** the source-only hypothesis that 30 subgroup creations and 31 host barriers dominate the transition. Group creation took at most 0.666 s, under 2% of the 49.839 s transition. The groups are lazily materialized; source call count was not a time attribution.

The dominant removable interval is whole-module CPU-to-GPU materialization from pageable preconstructed state: switching Critic ranks spend up to 33.169 s and target Generator ranks up to 28.799 s. Retained Score ranks report 37--38 s inside `deepspeed.initialize`, but this includes DP12 waiting for the switching ranks; switching ranks require only 3.6--5.8 s for local DS initialization after H2D. G3's 818 broadcasts expose another 10.868 s maximum.

## Decision

GO for one bounded transition candidate: 64-MiB double-buffered pinned staging for target model H2D plus 128-MiB fixed-order G3 buckets. The helper must preserve registered Tensor/Parameter objects, values, dtype, shape and stride; use bounded arenas; and return only after the copy stream is ready.

NO-GO for deploying communicator collapse in this iteration because its measured transition ceiling is below 2%. A small gate may still be retained as future evidence, but it cannot be credited toward the 10--25 s target.

Step10-A persistence remains a separate experiment; it is not combined with the transition candidate.

## Evidence

- `artifacts/runs/rb16_f5_iter292_transition_instrumentation_correctness_20260810_001`
- `artifacts/manifests/iteration293_transition_causal_profile_result.json`
