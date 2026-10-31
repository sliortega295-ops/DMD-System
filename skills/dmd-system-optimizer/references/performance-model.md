# Performance reasoning: enough to choose the next useful experiment

Use these ideas when they help a decision; they are not an intake checklist.
A few sentences, relevant measurements and a rough cost estimate are usually
enough. Build a detailed model only when uncertainty could change the choice.

## Objective and fair comparison

For original baseline T0, current control Tc and desired speedup S, the target
is T0/S and the remaining saving is max(0, Tc - T0/S). A 1.5x speedup is 33.33%
less time at fixed work/resources, not 50% less time. Keep the original baseline
available as the working control improves. Do not mix different batch sizes,
update windows, model precisions or resource counts into one speedup.

Record enough source/config/input identity, warmup policy and commands to repeat
the comparison. A commit plus the relevant patch is sufficient for a lightweight
note; stronger project requirements still apply. Separate cold setup, steady
state, final cleanup and full-job costs. A memory win and a time win are distinct.

## Start broad, then focus

Consider device/parallel layout (M), state placement and lifetime (L), scheduling
(Q) and kernels (K). Do this to avoid polishing a tiny kernel while model swaps
or redundant recomputation dominate, not to demand four reports every round.
First check missing existing fast paths and configuration/backend problems.
Use the previous valid profile until the workload or bottleneck has changed.

Explain the current hypothesis: what work is removed/accelerated, how often it
occurs, why it is on the critical path, and what extra work/memory is introduced.
Unknown magnitude may justify a cheap probe. Do not fill unknown values with
invented zeros or delay useful work until every measurement is available.

## Avoid false ceilings and double counting

Summed kernel/API duration can exceed wall time. Activity union is timeline
coverage; activity outside compute is exposure, not automatically removable
latency. A long collective can include waiting for a late rank. Shortening an
off-path operation can help contention, while shortening exposed work may have
no effect when it has slack. Profiled time is for diagnosis, not the final win.

For a genuinely critical serial portion t, local speedup s and added exposed
cost h, a useful conditional estimate is:

    saved_time = t * (1 - 1 / s) - h

It is valid only while the other costs and dependencies remain comparable.
For competing paths, model just enough of the DAG to see the new critical path.
Do not sum overlapping gains or count an enabler's downstream benefit twice.
The optional `dmd_perf.py dag` computes a supplied fixed-order graph; it does
not discover dependencies, model all contention, or certify a physical optimum.

A shape-based compute/traffic bound can rule out one route, but a weak prototype
or estimated efficiency does not prove the overall goal impossible. State the
assumptions and what remains unknown. Measure whether reuse actually executes.

## MFU when it is useful, not as an admission form

Count useful work over the actual logical update window: no-grad rollout,
selected differentiable calls, Real/Fake Score, teacher/GAN, backward and EMA.
LoRA trainable-parameter count is not total FLOPs; a frozen backbone can still
participate in backward. Do not blindly apply 6NT to all roles.

    MFU = useful_model_FLOPs / (window_seconds * sum_reserved_GPU_matched_peak)

State the precision/dense-sparse convention, covered operations and whether
recomputation is excluded. Report executed FLOPs separately if caching changes
the work. Heterogeneous precision may need a breakdown rather than one MFU.
Busy GPU, occupancy and kernel temporal coverage are not MFU. Missing counters
remain unknown; no invented percentage. Expensive measurement should answer a
specific decision, not be repeated as a ritual in every timing run.

## Choose feasible experiments, then reassess

Compare likely end-to-end benefit, confidence, implementation/validation effort,
capacity and available resources. Rough ranges are enough; exact GPU-hours,
probabilities or a complete JSON opportunity portfolio are not required by the
default workflow. A cheap test can resolve a high-value unknown.

Consider only actually actionable alternatives when deferring a smaller win.
An unsupported or unaffordable large idea must not veto a feasible 1% change
forever. If its probe is itself too costly or adds no information, park that
route and choose another. A useful capacity enabler may be retained without
inventing a downstream speed forecast; validate the composition before claiming
that speed benefit.

After a few unsuccessful attempts, compare prediction with observation. Has the
bottleneck moved, the path failed to activate, or overhead consumed the saving?
Change the hypothesis, not only the candidate name. Small wins can accumulate,
but measure the final combination rather than adding their percentages.

`target` and `compare` are optional arithmetic helpers. `rank/gate` implement a
more prescriptive, schema-driven study and are advanced opt-in; their incomplete
metadata statuses are not universal prohibitions on authorized lightweight work.
An already adopted formal protocol cannot be bypassed to relabel failed evidence.
