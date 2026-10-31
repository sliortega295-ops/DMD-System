# Quick checks while exploring; validate the final combination

Be strict about training semantics and honest claims, flexible about process.
Use the existing adequate project checker and its actual output. Do not build a
new acceptance system just because it emits native JSON/text instead of JUnit.
Existing frozen project requirements take precedence. Never lower tolerances
because a candidate failed them.

## During exploration

Run the cheapest check that addresses the change's risk. Source tests can catch
bad launch/config edits; primitive forward/backward checks can screen a fusion;
a short real update can expose dead gradients or wrong optimizer state. Check
the actual modified path and its fallback. Do not claim training qualification
from no-grad-only tests or a scalar CPU stand-in.

For a small local change, a relevant quick check plus short timing can be enough
to continue exploring with that candidate. It is not final acceptance. There is
no requirement to rerun a full distributed cycle after every edit or to run
100 steps just because tensors are not bitwise identical.

Stateful caches, offload, overlap, distribution and batching can fail only across
boundaries. Test the affected update/version, consumer lifetime, all-rank or
continuation behavior before relying on them. A suspected semantic/gradient
failure needs diagnosis now, not postponement until final validation.

When one component is inherently non-bitwise, a reference stand-in can help
isolate other changes. Check the real final combination too: stand-in results
do not validate interactions unique to the replacement.

## At a milestone or delivery

Compare the combined candidate with the **original baseline**, not only the
previous iteration. Cover required G/Score/optimizer/EMA updates and relevant
resume/export behavior. Longer multi-seed trajectories or generated-quality
checks are for the risks and claims that need them, especially authorized
precision/approximation changes or accumulating unexplained error. Loss curves
alone do not establish gradient/state correctness or unchanged final quality.

If final validation is unavailable, preserve the best verified version and label
the candidate and missing checks clearly. Do not certify it by changing the
checker, inventing evidence, or exhausting resources to force a PASS.

## Small gains and capacity value

Separate retaining code from claiming a speedup or changing defaults:

| Evidence | Practical decision |
| --- | --- |
| Relevant checks pass; repeatable small gain, even 1% | Keep within the tested scope; final stack checks still apply. |
| Quick checks pass; only one small/noisy timing difference | Retain as promising/unresolved, default off when not adequately checked. |
| Correct and useful memory/capacity improvement | Keep optional; disclose measured or unknown latency cost. |
| Wrong gradient/state, incompatible semantics or target-capacity failure | Revert unsafe defaults and preserve failure evidence. |
| Interrupted, stopped or out of resources | Save partial work; do not require more runs to record it. |

A change can be worth keeping without being worth more tuning. The best verified
version and an experimental stack may coexist without a database promotion step.
Do not sum individual gains; measure the combination.

## Timing effort follows the claim

Use matched unprofiled comparisons with consistent warmup and window boundaries.
A short probe screens a hypothesis; repeated/interleaved runs help resolve small
wins. Report every valid sample and its variability. More repetitions are useful
only when they can change a decision at acceptable cost. No automatic six-block
ABA rule or 2% minimum applies to the lightweight workflow. Do not claim formal
significance from visual similarity or one favorable measurement.

`dmd_perf.py compare` gives descriptive medians and paired differences, not a
numerical PASS or statistical certification. The original baseline stays available
while the current verified control advances.

## Advanced, opt-in fixed-sample confirmation

Only use `dmd_campaign.py` when a formal study or project explicitly chooses this
protocol. Its conservative test is unchanged: predeclare independent unprofiled
control/candidate/control blocks, compare each candidate to the faster control,
check control drift, and run a one-sided fixed-sample sign test above the declared
effect floor with multiplicity adjustment. Six all-positive blocks for one
candidate yield p=1/64; this is arithmetic, not a required default experiment.
It is not a confidence interval for mean speedup or proof of training quality.

Once adopted, keep that protocol, evidence and resource accounting intact. A
partial or failed formal check does not become PASS by switching to lightweight
mode. You can always hand off its incomplete result without new experiments.
Native suitable acceptance processes need not be converted into this schema.
See [campaign tools](campaign-tools.md) for commands and assurance limits.
