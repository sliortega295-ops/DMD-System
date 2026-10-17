# H3 raw backward: first Fake graph, block 10

This is a new diagnostic source, not a replacement numerical gate or a speed arm.
Its parent is `3aaf818504a6e66e7e510b4a4240ac08c394a6b5`, whose block-49
evidence remains immutable. Only the opt-in raw-repeat contract changes from
block 49 to block 10, using a distinct schema. Native default execution and the
original training gradients are unchanged; no timing admission is possible.

Motivation: old cross-run fixed-coordinate probes first observed differences at
block 10 on most ranks. The block-49 full-gradient same-input repeat showed only
small DQ variability (nL2 about 3e-6 to 1e-5). It does not establish that the
entire backward or the 6-11% loss variation is harmless. Testing block 10 can
distinguish local raw-backward variability from differences already present in
upstream inputs at that depth. It cannot by itself prove the latter mechanism.

The original d2A contract, world16 workload, shape, seed, sigma distribution,
six gradient graphs and two continuous cycles remain unchanged. Rank 0 alone
repeats one actual first-Fake backward three times with deterministic=False
and twice with True, while the original gradients continue training. Full
DQ/DK/DV metrics and host reads use the existing implementation and finish fence.
No numeric threshold, math, cache policy, model or optimizer is changed.

One additional world16 diagnostic is allowed within the current W5 two-start
cap, after the active Wan owner exits. No automatic retry, speed comparison or
numerical PASS will be emitted. Failure/partial/extra calls remain fail-closed.
