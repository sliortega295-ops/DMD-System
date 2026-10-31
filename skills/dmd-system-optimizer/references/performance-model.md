# Performance reasoning: enough to choose the next useful experiment

A few sentences, relevant measurements and a rough cost estimate are usually
enough. Build a detailed model only when uncertainty could change the decision.
This is guidance, not an intake checklist or an admission gate.

## Objective and comparison

For original baseline T0, current control Tc and desired speedup S, target time is
T0/S and remaining saving is max(0, Tc - T0/S). A 1.5x speedup means 33.33% less
time at fixed work/resources, not 50% less time. Keep the original baseline as the
control improves. Do not mix batch sizes, update windows, precision or resources.

Record reproducible source/config/input and commands. Separate cold setup, steady
state, cleanup and full-job costs. A memory win is not necessarily a speed win.
Preserve stronger project requirements without inventing extra paperwork.

## Start broad, then focus

Consider layout, state lifetime, scheduling and kernels to avoid polishing a tiny
kernel while model swaps or redundant work dominate. Check existing fast paths and
backend/configuration issues first. Use a valid existing profile until the workload
or bottleneck changes. Explain what work changes, its frequency, critical-path
position and additional costs. Unknown magnitude can justify a cheap probe.

Summed kernel/API duration may exceed wall time. Activity union measures coverage;
activity outside compute measures exposure, not necessarily removable latency.
A long collective may wait for a late rank. Off-path work can cause contention,
while exposed work may have slack. Profiled time is for diagnosis, not a final win.

For a genuinely critical serial portion t, local speedup s and extra exposed cost h:

    saved_time = t * (1 - 1 / s) - h

This is conditional on the other costs and dependencies remaining comparable.
With competing paths, model only enough of the DAG to see the new critical path.
Do not sum overlapping gains or count an enabler's downstream benefit twice.
`dmd_perf.py dag` calculates a supplied graph, not unobserved contention or a
physical optimum. A weak prototype or practical forecast does not prove the whole
research goal impossible. State assumptions and what remains unknown.

## MFU when useful

Count useful work over the actual logical update window: no-grad rollout, selected
differentiable calls, Real/Fake Score, teacher/GAN, backward and EMA. LoRA trainable
parameters are not total FLOPs; frozen backbones may still participate in backward.
Do not blindly apply 6NT to all roles.

    MFU = useful_model_FLOPs / (window_seconds * sum_reserved_GPU_matched_peak)

Declare precision, dense/sparse convention, coverage and recomputation convention.
Report executed FLOPs separately when reuse changes work. Mixed precision may need
separate breakdowns. Occupancy, busy GPU and temporal coverage are not MFU. Missing
counters remain unknown. Expensive profiling should answer a decision, not recur
ritually in every timed run.

## Feasible next steps

Compare likely end-to-end value, confidence, effort, capacity and available resources.
Rough ranges are sufficient. An unavailable large idea must not indefinitely veto
a feasible small change; park it or use a cheap informative probe. Useful capacity
improvements need not invent a downstream speed forecast to be retained optionally.

After unsuccessful rounds, ask whether the bottleneck moved, the fast path failed
to execute or extra overhead consumed the saving. Change the hypothesis, not only
the candidate name. Keep reproducible small wins, but measure their combination
rather than adding percentages. The remaining [helpers](helpers.md) provide target,
inspection, descriptive comparison and explicit DAG arithmetic only. They do not
approve experiments, allocate budgets or replace the project's checker.
