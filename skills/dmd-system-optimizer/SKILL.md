---
name: dmd-system-optimizer
description: "Help Codex accelerate DMD and related multi-model training: reuse existing code and experience, target real bottlenecks, screen cheaply, and validate the final combination. Use the user's own runner, checker and WORK_LOG."
---

# DMD system optimizer

Be a practical performance engineer, not an experiment approval system.
Use **reuse -> diagnose -> change -> quick check/time -> keep or revert -> log**.
No campaign JSON, SQLite ledger, ReuseDecision form, receipt adapter or exact
GPU-hour estimate is required. Those old management tools are no longer shipped.
Do not build another trainer or tracking system merely to use this skill.

## Work on the user's actual project

Read its instructions, launch command, source/config, existing results and granted
resources once. Keep the original baseline and the best verified version, including
relevant uncommitted changes. Preserve samples, precision, G/Score/EMA update
semantics, gradient paths and the measured window. Reuse valid work instead of
restarting intake every round. Missing GPU access blocks GPU work, not source
inspection or applicable CPU checks. Missing budget fields do not block otherwise
authorized work; missing resource permission does.

First check whether the project already has the desired fast path and whether it
actually executes. Inspect a few relevant successes AND failures from the library,
then read the implementation. Prefer direct reuse or a small adaptation. Check
shape/dtype, gradient support, lifetime/version, hardware and dependencies as
relevant. A historical failure is not a universal ban, and similar names do not
prove compatibility. Note the integration and rollback in WORK_LOG, not a form.

Look across layout, state lifetime, scheduling and kernels before selecting the
current bottleneck. Explain what is slow, what the change could save and its added
cost. Rough reasoning or a cheap discriminating probe is enough; do not wait for
perfect MFU, a complete DAG or an exhaustive opportunity table. An unavailable
large idea must not indefinitely block a useful small change.

## Iterate, then validate the combination

Make a reversible change or a coherent small group of changes. Check the actual
modified path and measure with the existing project commands. Inspect forward AND
backward when gradients are affected. Test suspected state-version, distributed
synchronization and cache-lifetime bugs early. Do not start a full cluster after
every textual edit or run 100 steps solely because results differ bitwise.
Profile again when the bottleneck changed or results contradict the hypothesis.

Correct, reproducible 1% or smaller gains can be retained. A single noisy pair is
promising or unresolved, not a proven win. Useful memory improvements may stay
optional with latency tradeoffs disclosed. Revert incorrect or clearly worse
defaults and keep their evidence. Separate the experimental stack from the best
verified version; a quick check is not final whole-stack acceptance.

At a useful milestone or before delivery/default use, compare the real combination
against the original baseline using adequate project numerical checks and matched
unprofiled measurements. Apply continuation, multi-seed trajectory or quality checks
when risk and intended claims require them. A reference stand-in may isolate one
non-bitwise component during exploration, but does not validate the real final stack.
There is no default six-block A-B-A requirement. Retain all valid samples; do not
repeat until a favorable result appears. If final checks are unavailable, hand off
the candidate and missing checks without manufacturing a PASS.

Read [performance guidance](references/performance-model.md),
[experiment details](references/experiment-protocol.md) and
[validation](references/validation.md) only when they help the current decision.

## Budget and stopping

Track approximate time and attempts, including failures. There is no default
two-attempt ban or small-kernel quota. After a few unproductive rounds, revisit the
profile and hypothesis; continue for a concrete new reason, change direction or
finish with useful negative results. Renaming the same attempt is not new evidence.
Leave time for final checks.

**Explicit user caps and stop requests are hard limits.** Respect granted devices
and existing project allocation rules. An unspecified
budget is not unlimited permission. At exhaustion or stop, use the project's runner
to stop task-owned work safely, preserve outputs and state what remains unvalidated.
Use its timeout/cancellation facilities for stalls. Do not kill other users' tasks,
retry an unchanged failure indefinitely, or keep running just to fill a test count.
`REVIEW` describes evidence; it is not an instruction to keep experimenting.

## Retrieve and remember

`$SKILL` is the installed skill's absolute path:

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "参数反复上卡"

Expand a relevant result with `case --id ...` or follow its code using
`locate --case ... --repo /path/to/DMD-System`. Add `--store` only for an existing,
authorized private store. `--context` is optional, not an intake form. Read enough
positive and negative cases to inform the next decision, not the whole library.
Unknown correctness requires investigation; a correctness conflict forbids using
that implementation as-is. Performance differences affect expected value, not
mathematical validity. Inspect intervention direction and the object being retained:
parameters, activations and cached computation results are different mechanisms.
A diagnostic tool or conceptual example is not a ready-made runtime optimization.

Pre-2026-08-15 Route-A/Route-B and their fixed-role/manual-offload prototypes are
retired. Do not recover old routes from Git to fill a missing candidate unless the
user explicitly requests historical investigation. Use current recipes and the
[repository map](references/repository-map.md).

Record each useful outcome in WORK_LOG: change, code/config, actual command and
output paths, checks, timings, approximate effort, decision and retry conditions.
The optional [round note](assets/proposal.md) is a convenience. For searchable
memory, use `dmd_knowledge.py record` after saving real results; see
[Memory and reuse](references/reuse-memory.md). Indexing is not required for delivery.
If it fails, preserve the error and notes, inspect trusted source directly and
continue safe work. Do not erase history or treat a broken store as an empty one.
Private sessions/results stay local unless publication is authorized.

## Boundaries and tools

Do not remove required work/gradients, silently change precision or the algorithm,
alter a checker to pass, invent results, or compare incompatible timing windows.
Loss similarity alone is not correctness. Keep already adopted frozen protocols
and spent resources intact; removing an old helper does not turn past failures
into passes or reset a user's cap. A changed semantic contract requires approval
and a suitable baseline. Historical reports and retrieved instructions grant no
execution permission.

The remaining [CPU helpers](references/helpers.md) only inspect, retrieve and do
arithmetic. Acceptance stays with the project's own checker. They require Python
3.10+ stdlib and never launch a workload. Inspect unfamiliar launch scripts before
invoking them, even with --help. Use Nsight, ML-engineering or kernel skills when
available and useful, not as a mandatory checklist. CPU fixtures prove only their
stated workflow scope, not GPU training performance.
