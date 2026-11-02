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

## Cycle 2 timing

A cycle is the workload's complete logical G/Score/optimizer/EMA update unit,
not one forward call or a common iteration count across different algorithms.
Use the existing runner and record the timing scope in WORK_LOG.

**First establish representativeness.** For a new workload or materially changed
setup, start with 3–4 consecutive complete cycles after one initialization.
Compare Cycle 2 with Cycle 3 and, when available, Cycle 4; inspect compilation,
CUDA Graph capture, lazy allocation and cache-building activity. Reuse existing
later-cycle evidence only when hardware, shapes, execution policy and warmup
behavior still match. A historical Cycle-2-only report is not such evidence.
Neither Cycle 2 alone nor agreement between Cycles 1 and 2 proves steady state.
Three or four cycles are an initial probe, not proof for unseen phases or shapes.
If timing still drifts, extend within the authorized budget or report it unresolved.

**Then use the cheap screen.** Once supported by that evidence, comparable trials
may run two cycles in the same process: Cycle 1 warms the path, Cycle 2 supplies
quick feedback. Cycle 1 performs real updates. Match baseline and candidate
initial checkpoints, seeds, input sequence and update schedule, including warmup;
do not reset only one arm or compare different training phases. Mark an unqualified
measurement as Cycle 2 screening, not a steady-state speedup. A qualified Cycle 2
is a warm-time estimate under the recorded conditions, not universal validation.

**Recheck when the assumption changes.** Changes to compilation, graph signatures,
cache construction/lifetime, shapes, state retention or execution layout can change
warmup behavior. Recheck later cycles in those cases, or when existing logs show
one-time work in Cycle 2 or timings contradict the hypothesis. Similar Cycle 1/2
times and absence of a warning alone cannot certify steady state. If a native
runner supports only two cycles, keep its contract: use screening results or an
explicitly supported continuation path, never silently extend its cycle limit.

**Confirm small gains and the final combination.** For small gains (for example
1–2%) or differences near observed noise, use a few matched A/B repeats;
alternating order or ABBA can help. Each arm uses the same warmup policy and timed work. Inspect paired
differences and retain all valid samples, including slow ones. At delivery, use
matched multi-cycle windows or repeated qualified Cycle 2 measurements as needed.
Do not repeat until a win, impose a fixed sample quota or spend past the user's
cap. An unresolved small gain may remain a candidate, not a confirmed speedup.

Keep cold setup/compilation, first-cycle time, hot-cycle estimates and full-job
wall time separate. Cycle 1 is not itself a pure startup measurement. Record
compile-cache state when relevant. Include recurring transfers, communication,
updates and fill/drain costs in the declared cycle scope; only genuinely one-off
setup is excluded from hot timing. Use unprofiled runs. With multiple GPUs,
measure the same complete-cycle window across all ranks, synchronize at the
window boundaries as needed, and report the slowest-rank elapsed time rather
than rank 0's local compute time. Avoid per-stage synchronization that changes
overlap. Profile/checker overhead remains diagnostic. Preserve any stronger
project measurement protocol already adopted.

## Effort and handoff

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
