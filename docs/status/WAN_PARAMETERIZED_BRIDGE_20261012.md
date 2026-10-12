# Wan native strategy integration: checkpoint 1

Starting source: `088b8ad53f16051577aebb5d75eb6e2f5370fc13`.
Recovered the previously unpushed native-action bridge into the existing branch.

## Implemented
The legacy Full5 bridge and explicitly contracted parameterized cycles share the
same native Generator, rollout, Critic and EMA handlers in `wan_cycle_plan.py`.
Result collection uses the declared Critic count. No alternative Euler loss,
sampler, model or launcher replaces the original workload. The old Full5 API and
canonical profiles retain their existing constraints.

## Validation
87 CPU handler/parameterized-cycle regressions passed again in the recovered
source tree. Its unmodified base tree was reconstructed with the pinned gitlinks
and matches the upstream tree `ffd27bfa2fba50f81a44b1d9c1e39d1f2a99e233`.
These are CPU dispatch tests, not original-model or distributed GPU evidence.
Full integration and downstream native adaptation remain to be validated.

## Next checkpoint
Connect a parameterized original-trace contract and the existing worker controls
to real strategy candidates BEFORE engine construction; then calibration/search
and selected-plan execution. Do not claim success from metadata alone. Preserve
original loss/trace/version semantics and the naive-FSDP2 performance baseline.

Each following checkpoint must contain code/tests and be saved on GitHub before
another substantial change. Main is not modified. No GPU run or speedup claimed.
