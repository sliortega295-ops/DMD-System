# Practical experiments in the existing project

Use its runner, checker, source tree and WORK_LOG. Do not create a new trainer,
ledger or approval process. Keep the original baseline, best verified version and
experimental patch separately. Record enough source/config, input, command and
result information to reproduce the claim; stronger project rules still apply.

## One useful question per experiment

Explain the bottleneck and proposed mechanism briefly, then choose the cheapest
adequate check. Static/CPU tests catch configuration and lifecycle mistakes;
small primitive or trace probes resolve local uncertainty; representative updates
expose real gradients/state; matched unprofiled timing measures the actual effect.
Reuse unchanged evidence. A textual edit does not require restarting a full cluster.

Prefer existing fast paths and minimal adaptations. Measure whether the changed
path actually executes, including fallback counts when relevant. Do not alter
canonical workload profiles or source receipts to conceal an experiment. Preserve
logical samples, calls, updates and the user's intended precision/algorithm.

## DMD-specific risks worth checking

- Which G/Score/EMA version does each consumer use? Does reuse cross a commit?
- Shared frozen weights do not imply shared mutable LoRA, buffers or graph state.
  An adapter-mode switch must not race with an in-flight call.
- No-grad/detach or a cached value must not sever a required gradient path.
- Cache keys and invalidation must cover the inputs, state and randomness that
  affect the value. Test continuation when lifetimes cross updates.
- Batching can change loss weighting, RNG order, reductions, clipping or EMA
  timing; do not assume unchanged shapes imply unchanged training.
- Async copies require correct producer/consumer ownership. Include transient
  buffers and allocator behavior when checking target capacity.

Check the accelerated path, not only its original fallback. Final acceptance covers
the real combination against the original baseline, not merely a chain of local
comparisons. Longer trajectories are chosen by risk, not by a blanket bitwise rule.
See [validation](validation.md).

## Timing and effort

Use matched work, hardware and timer boundaries. Distinguish setup, steady state,
fill/drain and final cleanup. Warmup must not silently advance only one arm's state.
Profiler/checker overhead is diagnostic, not unprofiled performance evidence.
Single pairs are descriptive. Repeat/interleave when it can resolve an important
uncertainty, retain valid slow samples, and do not repeat until a desired result
appears. Respect existing formal protocols once adopted.

Track approximate effort and failed attempts. After unproductive rounds, reassess
the mechanism rather than rename it. Prefer feasible alternatives; no fixed
attempt ban or exact-budget form is imposed by this Skill. Explicit user limits,
allocated devices, project requirements and stop requests remain binding. Leave
room for final checks and use the runner's timeout/cancel path for stalls. Stop
only task-owned processes.

At a stop, failure or exhausted budget, save the best version, candidate, actual
outputs and missing checks. No PASS or completed sample quota is needed to hand
off honestly. Preserve already spent resources and past failures; removing an old
tool does not reset them. Optional indexing errors go in WORK_LOG and do not justify
more GPU runs. See [small helpers](helpers.md) only when useful.
