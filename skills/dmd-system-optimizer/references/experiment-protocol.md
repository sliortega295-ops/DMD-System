# Practical experiment loop

This is guidance for ordinary optimization, not mandatory paperwork. Use the
project's runner, checker, log and existing resource rules. A WORK_LOG with code,
commands and actual output paths is sufficient; no separate ledger is required.

## Start once, then iterate

Identify the workload, original baseline, best verified version and relevant
uncommitted changes. Preserve logical samples, precision, G/Score/EMA clocks,
loss and gradient semantics. Reuse valid prior runs. Missing setup should block
only the work that depends on it, not all useful inspection or CPU tests.
Do not edit frozen plans/import digests to conceal a changed workload.

For each round:

1. Read the current bottleneck and relevant implementation/experience.
2. Explain the mechanism, likely benefit, principal risk and cheapest useful test.
3. Make a reversible change, or a coherent group behind separable switches.
4. Run affected quick checks, then short representative timing. Revert clearly
   wrong/worse defaults; keep promising candidates separate from verified ones.
5. Append actual results and approximate effort to WORK_LOG. Index reusable
   experience when convenient, including failures and unresolved observations.

No full M/L/Q/K report, new profile, complete cluster cycle or formal statistical
confirmation is required after every edit. Add detail when it changes the next
choice. A documentation edit does not invalidate an unchanged numerical run.
A source/config change affecting behavior does require appropriate new checks.

## Cheap checks first; final stack later

Use source/CPU checks for interfaces and launch logic; a representative primitive
for kernel behavior; a target run for state, update or distributed interactions.
Forward-only evidence does not qualify a backward change. Exercise the actual
accelerated branch, not just the old branch with checking turned on.

Low-risk local checks may admit an experimental stack for further screening.
At a useful milestone or final delivery, check the combination against the
original baseline and exercise the required complete logical updates. Test
continuation, checkpoints and version invalidation when those paths changed.
High-risk numerical changes need appropriate trajectory/quality evidence earlier.
Non-bitwise alone is not a requirement for a 100-step run.

Never replace actual training work with mocks or cached answers for a performance
claim. Tiny stand-ins and synthetic timings qualify their workflow only.
See [validation](validation.md) for quick versus final evidence.

## DMD questions to ask when relevant

- Which G/Score/EMA version does a consumer use? Does overlap cross an update?
- Does sharing preserve independent trainable adapters, buffers and autograd?
- Does a cache depend on prompt/sample, shape, timestep, precision, randomness,
  train/eval mode or weight version? Is it invalidated at the right boundary?
- Does batching alter sample weighting, RNG consumption, reduction, update or EMA
  order? Logical work must remain equivalent even when physical launches change.
- Do source buffers remain alive until async copies/consumers complete? Does the
  full phase fit memory, including temporary activations and collectives?

These are risk-directed questions, not fields that every candidate must fill.

## Timing without overtesting

Use matched work, allocated hardware, input/initial-state policy and timer scope.
Separate cold setup from steady state and count fill/drain and cleanup when they
belong to the delivered run. Warmup must not silently advance only one arm's state.
Profiler/validation overhead is not unprofiled performance evidence.

Short timing screens ideas; single-pair differences remain descriptive. For a
shortlist or final result, repeat/interleave with the baseline as needed for
observed noise and the intended claim. Preserve valid slow samples and failures;
do not keep rerunning until the desired number appears. No fixed number of ABA
blocks is required by this default workflow. If a formal protocol was already
adopted, its requirements still apply and cannot be loosened after failure.

## Effort, stalled runs and stopping

Record approximate elapsed time and attempts. Do not require exact GPU-hours or
a predeclared full-run count when the user did not request them. After a few
unproductive rounds, re-evaluate the mechanism and available alternatives; this
is a reminder to think, not a permanent ban after two attempts. A changed label
is not new evidence. Prefer a feasible small gain over an unaffordable large idea.

Explicit caps, allocated devices, project rules and user stops are binding.
Absence of numeric fields does not authorize new resources. Leave room for final
checks; do not start a run likely to exceed an explicit remaining limit. Use the
existing runner's timeout/cancel path for a stalled job, stopping only owned
processes. Save partial results and the best verified control at exhaustion or
stop. No final PASS is required to finish an honest handoff.

Optional indexing/accounting failures must not erase results or force more GPU
work. Record the tool error, keep native outputs and use a plain handoff. Existing
frozen studies are not silently reset: preserve their costs, failed evidence and
unmet requirements. See [advanced helpers](campaign-tools.md) only if that study
actually needs them.
