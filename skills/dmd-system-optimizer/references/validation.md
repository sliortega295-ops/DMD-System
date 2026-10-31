# Quick checks while exploring; validate the final combination

Use the project's adequate checker and its real output, whether JSON, JUnit or
text. Do not build a new acceptance system around it. Existing project requirements
still apply; never lower tolerances merely because a candidate failed.

## During exploration

Choose the cheapest check that addresses the changed path. Source tests can catch
launcher mistakes; forward/backward primitive checks screen a fusion; a short real
update can expose dead gradients or wrong state. A no-grad or scalar test does not
qualify distributed training. Check fallback when it matters.

A quick check plus short representative timing may justify continuing with a
candidate, not declaring the whole stack accepted. Do not run a full cluster after
every edit or 100 steps solely because results are not bitwise identical.
Stateful caches, offload, overlap, distribution and batching need checks at their
affected update/version, consumer-lifetime, all-rank or continuation boundaries.
Investigate suspected semantic or gradient bugs immediately.

A reference stand-in can isolate a non-bitwise component while checking other
changes. It cannot validate interactions unique to the real replacement.

## At a milestone or delivery

Compare the real combination with the original baseline, not only the previous
candidate. Cover required G/Score/optimizer/EMA updates and relevant resume/export
behavior. Longer multi-seed trajectories and quality checks are for the risks and
claims that require them, especially precision changes or accumulating unexplained
error. Similar loss curves alone do not establish correct gradients or state.

When final checks cannot run, preserve the best verified version, candidate patch,
completed evidence and missing checks. This is an incomplete delivery, not a reason
to invent PASS or continue past a stop or resource limit.

## Small gains, capacity and timing

Correct, reproducible 1% or smaller gains may be retained. A noisy single pair can
remain a promising candidate without becoming a certified win. A useful memory
reduction may remain optional with measured or unknown latency costs disclosed.
Target-capacity failure or wrong state/gradients must not become a safe default.

Use matched unprofiled comparisons with consistent work, hardware, warmup and
window boundaries. Keep every valid sample. Repeat/interleave when uncertainty can
change the decision at acceptable cost; do not repeat until a favorable result
appears. There is no packaged fixed-sample confirmation protocol or default 2%
minimum. Existing formal project protocols remain binding once selected.

`dmd_perf.py compare` only reports descriptive medians and paired differences;
it does not certify numerical acceptance or statistical significance. Individual
improvements are not additive: measure the combination. See [helpers](helpers.md).
