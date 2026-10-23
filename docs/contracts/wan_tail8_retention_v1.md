# Experimental Resident Critic tail8 state retention

Explicit `retention_policy=tail8.v1`, exact blocks22..29 only. The original
retain4.v1 default, dictionary bytes, IDs and rejection of >4 blocks remain.
Same audited normal-autograd restoration wrapper as retain4; no new kernels,
math, checkpoint/data/dtype/global-batch/optimizer/RNG changes. Application
calls remain unchanged; only eight Critic blocks avoid checkpoint recompute.
No automatic search admission or historical cost reuse is enabled.

The earlier actual retain4 pair20.324208→20.002455s and allocated23.409→26.692GiB
motivate one bounded tail8 pair, after covering all six workloads. Capacity is
only a prediction (roughly30GiB allocated), not a fit claim. Original2GiB
allocated headroom and64GiB startup host reserve remain. Existing2 W1 launches
stay charged; at most2 additional same-source control/child native C2 launches
within W1cap4 and campaigncap24. Full-state comparison retains REVIEW and the
original thresholds; no strict/long-trajectory equivalence claim.
