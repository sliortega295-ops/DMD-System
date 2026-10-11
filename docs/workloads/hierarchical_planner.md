# Hierarchical planning: first integrated algorithm slice

This adds an opt-in search algorithm, a fixed-binding event-frontier solver,
and a parameterized G1/Rn/Cn compiler. It does **not** claim a parameterized
GPU training runtime for arbitrary resolutions/models, a new measured speedup,
or a replacement for the five canonical profiles. No native/GPU run is part
of this change. Existing Metropolis and registered-search defaults remain.

## 1. Current native decision spaces: hierarchical selector

```bash
python scripts/distillctl.py workloads search \
  --strategy model-based --search-algorithm hierarchical \
  --workload wan13b_nogan_resident_best --dry-run
```

With an adapter-produced problem/layout and source-bound cost model, the same
entrypoint searches real admitted choices and uses the **existing native
lowerers and validation controller**. For example, replace the dry-run with
`--planning-layout layout.json --cost-model costs.json --output search.json`.
`--emit-native-plan` and `--execute-native` keep their existing source,
calibration, correctness, workload and total timing-budget requirements.
No GPU job is launched without the existing explicit native-execution flag.

`search/model_search.py::search_plans(..., algorithm="hierarchical")` expands
structural variables first but keeps alternatives in the frontier. It never
commits to a greedy placement before considering state/kernel/schedule choices.
The structural names can be supplied with `--structural-decisions`; remaining
adapter decisions stay in the same search. This does not infer new capabilities
from arbitrary workload names or add missing options to a worker.

The inner decision DP memoizes equal **residual binding/constraint domains**,
including operation contracts, physical allocation identities and conditional
dependencies. Equal bytes or equal elapsed time alone do not imply equivalence.
Complete assignments are evaluated by the unchanged `simulator.py`; their
predictions remain exactly revalidatable by current native adapters.

Pruning uses explicit forbidden capabilities, unavoidable static/workspace
allocation, future-domain equivalence and a conservative time lower bound.
Cycle2 is `completion2 - completion1`. Critical-path/resource-work lower bounds
are applied only when a dependency fence proves all cycle2 work starts after
cycle1 completion; otherwise the time lower bound is **zero**. Summing kernel
unions or subtracting independent makespan estimates is not a safe bound.
Missing costs remain calibration requests, not infinity or guessed throughput.

`--beam-width 0` removes heuristic beam truncation. State/time budgets still
apply. `--no-search-pruning` disables equivalence/time-bound pruning for CPU
ablations; semantic and unavoidable-capacity rejection remain. Nonzero beam
width is heuristic, and any truncation reports `BEST_FOUND_ONLY`. A complete
search can certify only its **declared discrete simulator objective**, not
GPU-optimality or unseen/missing calibration. With missing costs, the label is
`CALIBRATED_SUBSET_OPTIMAL`, never a full search-space optimum. Seeds are optional
for CPU selection; native validation still requires its existing matched anchors.

## 2. Fine action scheduling: event-frontier DP / beam

`search/frontier_schedule.py::schedule_frontier` accepts the same ExecutionGraph,
explicitly resolved BindingRules, physical StateAllocations, Lifetimes, CostModel
and per-rank memory limits. It minimizes one fixed-binding DAG's makespan.

The event state contains completed operations, running operation identities and
remaining durations. Those determine ready calls, versions, adapter/resource
leases, and which exact saved tensors are still live. Labels retain elapsed
time and per-rank peaks; only equal future states admit Pareto dominance.
A transition either dispatches a legal ready call or waits for the next actual
completion. Waiting remains available even when work is ready: otherwise a
long low-priority call can incorrectly block a later critical call.

The solver counts shared storage once, retains outputs until the final consumer
**completes**, includes producer workspace, and honors adapter mutual exclusion.
Declared dependency/resource lower bounds are used without adding overlapping
resource totals. It assumes fixed nonpreemptive task costs, no hidden external
release times and no unresolved data-dependent branch. Those assumptions define
the discrete event-dispatch family, not a claim about every PyTorch program.
Starts are restricted to zero or a preceding completion. With producer-exit
allocations, an unrestricted off-event delayed start can be better; a regression
makes this scope explicit. Node/time budgets and beam losses remove the
within-family optimality claim as well.

This solver does not silently replace a native worker's fixed schedule.
Cross-pool transport, input-gradient ownership, CPU storage accounting and
arbitrary generated-order lowering still require explicit backend support.

## 3. Parameterized cycle and public CPU example

```bash
python scripts/distillctl.py workloads search --strategy cycle-frontier \
  --problem configs/distillschedule/planner_examples/cycle_g1f3_cpu.json \
  --output /tmp/new-cycle-plan.json
```

The numbers in this example are **synthetic nanoseconds/bytes**, not GPU data.
The CLI rejects native evidence claims, unknown fields and missing action costs.
It exports a generated graph and selected order, not a model launch command.

`runtime/parameterized_cycle.py` compiles 1..64 C updates and arbitrary declared
EMA positions for the explicit **post-G-commit independent-rollout** protocol.
Source/sample preparation order and parameter version constraints are retained.
Generated single-pool orders use the existing `StaticRoleRunner.prepare()`;
CPU handler tests verify dispatch, updates and retained-input release for
G1/F1, F2, F3, F5 and F8. The historical `cycle_plan.py` Full5 graph/contract is
unchanged; its five-update guards are not deleted to fake native support.

This compiler does not infer LightX2V vs DiffSynth semantics, change a sampler
from four to eight steps, convert LiveWan's pre-commit step0 data into post-commit
data, or grant arbitrary shape/LoRA/kernel support. Per-configuration native
adapters and shape/dtype capability matching are the next prerequisite.

## Validation and scope

New tests cover independent small-DAG enumeration, random-cost decision oracles,
intentional waiting, resource/adapter contention, saved-tensor lifetime, shared
storage, missing calibration, beam/budget reporting, native-plan tampering and
old/new CLI compatibility. Source-bound native-lowering tests use explicitly
synthetic receipts and are **not GPU correctness or performance measurements**.

Existing five-profile definitions, training workers, model math, sampler settings,
precision policies and native trial budgets are not modified. CI runs on CPU.
Known remaining work is parameterized native protocol/model adapters, composable
shape/dtype capability binding, CPU-memory/transport effects and generated
cross-pool lowering. These are not implied complete by an algorithm test pass.
