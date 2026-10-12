# Checkpoint 3: existing-strategy selection through hierarchical planning

OriginalWanTask now constructs a two-cycle ExecutionGraph/PlanningProblem and
uses the existing CostModel, hierarchical search and simulator. It jointly
selects admitted placement/chunk/epilogue/window choices without a winner lookup.
All sixteen resources are reserved by each action; internal async gradient
reduction remains inside the observed operation cost. No fake inter-role overlap
or artificial retained-window time discount is introduced.

17 new CPU tests passed: selection of multiple existing implementations when
costs favor them, exhaustive small-space comparison, capacity-driven switch to
FSDP, changed-task rejection and rehashed native-dispatch rejection. Costs in
these tests are synthetic, not historical/native GPU observations. Previously
published task/trace/bridge tests also remain available.

The selector accepts source-bound measured costs but the parameterized original
worker and collector connection is the next checkpoint. A successful model-only
selection is not a native performance result. Existing full-scale GPU budgets
are not reset; main and canonical profiles remain unchanged.
