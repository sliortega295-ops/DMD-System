# Risk-based validation and result decisions

Freeze validators, input/checkpoint identity, tolerances and the intended claim
before changing a candidate. Validation depth follows the affected mechanism.
Existing stronger project requirements still apply.

| Change | Usually needed | Failure the checks must expose |
| --- | --- | --- |
| Equivalent pointwise/fusion path | V0, V1 | rounding/dtype, forward and backward, fallback, one complete update cycle |
| Residency/offload/prefetch/cache | V0, V1, V2 | transient memory, races, stale versions, continuation, checkpoint and composition |
| Distribution/batching/reduction | V0, V1, V2; V3 if risk remains | all-rank work, logical operation identity, reduction/RNG/update order |
| Authorized approximation or altered numerical regime | V0, V1, V2, risk-specific V3 | paired trajectory/quality under declared tolerances and population |

These are starting points, not automatic PASS rules. Removing a copy can affect
alias lifetime; a kernel that passed no-grad is not qualified for training.
Validate source guards against the actual call. No-grad frozen-base LoRA fusion
cannot silently remove trainable LoRA gradients. Compose changes against the
current accepted control; do not add gains from incompatible counterfactuals.

Non-bitwise results can be acceptable under a predeclared numerical contract.
A changed tensor's tolerance, gradient relative error, actual parameter/optimizer
updates, update counters and a complete cycle may resolve a local arithmetic
change. State changes require boundary and continuation checks. Long validation
must answer a particular residual question; there is no blanket 100-step rule.

Loss rising in one optimized run is neither proof of an error nor proof of
equivalence. Compare the same baseline trajectory, gradient/parameter/state
deviations and natural variation. An unavailable baseline leaves REVIEW.
Successful training does not establish resume, export, generation or output
quality; test the lifecycle endpoints relevant to the claimed delivery.

## Confirmation

The bundled campaign tool provides one conservative **optional statistical
confirmation protocol**, suitable for stable full-cycle measurements:

- Predeclare independent unprofiled process blocks, each control-before,
  candidate, control-after; retain all blocks and include all resource cost.
- For each block compute saving against the faster control. Reject excess
  within-block control drift. Process identifiers are invocation ids, not reusable
  OS PIDs. Independence and stable resource conditions still require runner checks.
- Test the number of savings above a declared effect floor with a one-sided,
  fixed-sample sign test. Divide alpha by the maximum number of confirmations.
  The init check rejects a repeat budget that could never reach this threshold.
- A successful test supports positive typical savings above that floor under
  independent, appropriately matched blocks. It is not a confidence interval for
  mean speedup, protection against every systematic bias, or long-run quality proof.

For example, six all-positive blocks with one predeclared candidate give
p=1/64 at alpha=.05. An actual 1% gain can pass; 1% is neither discarded nor
guaranteed significant. Noise, overhead and resource use still matter. This
example is arithmetic, not an experiment result.

Use an existing suitable repository protocol instead when present. Freeze it and
retain its result; do not squeeze it into this helper by inventing process blocks.
The simpler `dmd_perf.py compare` is descriptive only. Diagnostic trials and
cheap probes do not require the full confirmation protocol.

Result meanings: wrong -> DISCARD; unvalidated -> REVIEW; correct but uncertain
timing -> REVIEW; correct reproducible benefit -> KEEP; evidenced capacity value
without proven speed gain -> KEEP_OPTIONAL. Default promotion additionally
requires applicability and fallback PASS and a non-diagnostic claim. Approximate
results keep their approximation label. The current best control changes only
after an explicit successful promotion; the original baseline is immutable.
