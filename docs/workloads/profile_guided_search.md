# Profile-guided registered-plan search (v1)

This is an opt-in extension of `distillctl workloads`, not a parallel training
runtime. The five canonical profiles and `registry.lock.json` are unchanged.
Historical winner times are not measurements of this branch.

## Scope and architecture

```
native receipt + per-device SQLite + nsys-ai doctor
    -> normalized profile observations
    -> frozen workload contract + registered applicable recipes
    -> bounded, resumable selection (beam width 2)
    -> existing profile compiler / dispatcher / workload worker
    -> native correctness gate, unprofiled two-cycle measurement
    -> measured ranking + independent winner confirmation
    -> source-bound execution plan (not a new canonical default)
```

The objective is cycle-2 latency. Cycle 1 is warm-up in the same process and
initialization; it is neither averaged with cycle 2 nor replaced with a profiled
time. The native workload owns timing boundaries, trace/sample/version/call
validation and numerical thresholds. Metadata compilation does not import Torch
or start a GPU worker.

| Canonical workload | Registered search space in v1 | GPU execution adapter |
|---|---|---|
| Wan1.3B No-GAN | FSDP reference; Resident recipes; local-state/bucket/kernel variants | Four recipes: FSDP, Resident async Tree, chunk2048, LoRA epilogue |
| LiveWan cross-scale | FSDP reference; existing C24--C34 typed pipeline plans | Native trajectory and two-cycle adapters |
| Helios No-GAN | Canonical reference | CPU registration only; no new GPU search |
| Helios GAN | Locked C25 reference | CPU registration only; no new GPU search |
| MiniMax-H3 | Grid1000 canonical reference | CPU registration only; approximate sigma semantics remain explicit |

`executable` means an adapter exists, **not** that this integration HEAD has
passed a GPU experiment. Catalog `NOT_RUN` fields are conservative registration
labels, not a mutable experiment database. Actual validation belongs to a
source/assets-bound search result. Unintegrated recipes fail before launch.

The [first world16 pilot report](evidence/profile_guided_search_20260906.md)
records measured Wan and LiveWan searches, their numerical REVIEW boundary and
independent confirmations. It does not change a canonical profile or claim to
reproduce historical winner times.

Wan's FSDP -> Resident Tree edge changes a registered bundle (placement and
gradient reduction implementation); it is **not a pure placement ablation**.
The subsequent chunk and LoRA-epilogue edges change one main factor. LiveWan's
initial FSDP -> C24 edge is likewise a complete pipeline recipe. C26/C27 are
siblings of C28, not erroneously inherited into every later recipe.

## What guides selection

The importer uses exact per-rank PID/device/NVTX windows, named SQLite columns,
read-only connections and a 300-second/node default query deadline (max 900).
Each SQLite is SHA256-bound to a hashed `nsys-ai doctor` report and must cover
all eight devices on its node. Missing ranks, parser errors, unresolved-kernel
coverage >=1%, invalid NVTX identity, overlapping cost phases, or reconciliation
failure reject the input. Duplicate phase names are retained as occurrences.

Features are compute/non-NCCL, NCCL not covered by compute, H2D/D2H, pointwise
name-family hints and no-traced activity. They are **GPU-second ranking hints**,
not a predictive latency model or removable time. Non-NCCL coverage is not SM
utilization/MFU; NCCL non-overlap is not proven critical-path wait. True dependency
wait remains `UNRESOLVED_REQUIRES_DEPENDENCY_EVIDENCE`.

Selection prioritizes measured-parent/beam-neighbor recipes, then measured
same-mechanism deltas where available, then profile-target hints. It records
every selection reason. It does not synthesize kernels, choose arbitrary TP/PP,
change batch size, relax numerical thresholds, or claim a global optimum. New
kernel mechanisms require separate evidence-driven kernel-design work before
registration. These are deliberate v1 limits, not a claim that universal
automatic scheduling has been solved.

Old source/assets/contract/cycle-1 profiles require `--historical-profile-prior`
and the same registered workload and hardware. For example, a historical
single-cycle profile may not have recorded the current two-cycle RNG seed. Do
not fabricate a matching semantic fingerprint: retain its observed contract,
including unknown fields. All identity differences are persisted and only
influence ordering. This flag never relaxes the native measurement gate: search
comparisons still require a common current source, assets, hardware, seed and
semantic fingerprint. A historical profile is not a matched latency estimate.

Some historical captures have kernels on all devices but phase NVTX only on
rank 0/8. Do not fabricate the other ranks' windows. The importer additionally
accepts an **explicit** `scope: REPRESENTATIVE_PRIOR_ONLY` plus `sampled_ranks`.
It validates exact windows only for those ranks, emits
`PASS_REPRESENTATIVE_PRIOR` / `world16_phase_attribution: NOT_ESTABLISHED`, and
requires the historical-prior search opt-in. It does not scale two GPUs' union
to sixteen GPUs or claim world16 utilization. All native measurement gates
still require 16/16; default profile import still requires all sixteen ranks.

## Commands

Run from a clean source checkout. Local controller evidence belongs under a
named project subtree; large traces and training artifacts stay remote.

```bash
# No GPU, SSH or model loading; all five profiles can be inspected.
python scripts/distillctl.py workloads search \
  --workload wan13b_nogan_resident_best --dry-run --max-trials 8

# Execute where the SQLite files reside. No capture is started by this command.
python scripts/distillctl.py workloads profile \
  --input /persistent/new-run/profile-input.json \
  --output /persistent/new-run/profile.json

# Run Wan first; LiveWan only after Wan finishes and releases its locks.
python scripts/distillctl.py workloads search \
  --workload wan13b_nogan_resident_best \
  --profile /local/project/evidence/wan-profile.json \
  --assets-config /local/project/configs/wan-assets.json \
  --output /local/project/evidence/wan-search \
  --run-id wan-search-r1 --max-trials 8 --historical-profile-prior

# Same command/identities/budget plus --resume, only after a clean completed
# trial boundary. An interrupted trial is preserved and not silently retried.
```

Asset JSON extends the existing best-known schema only with optional per-node
`ssh_port` and `address_family` (`ipv4`, `ipv6`, `auto`). It must bind the same
16-GPU allocation, environment and immutable identity files on both nodes.
No shell/environment performance overrides are accepted. CPU metadata RPC
resolves beside remote assets; no large files are pulled to the controller.

`profile-input.json` has schema `distillschedule.search.profile_input.v1`,
`cycle`, `identity`, `hardware`, and exactly 16 `ranks`. Every rank supplies:

- `node`, `rank`, `device_id`, `global_pid`, `sqlite`, `sqlite_sha256`;
- `parser_status: PASS`, `parser_errors: []`, a `nsys_ai_doctor` path/SHA binding;
- an exact `capture` and disjoint `phases`, each containing `start_ns`, `end_ns`,
  `global_tid`, `nvtx_text`, `coverage: exact`; phases also have name/role/occurrence.

PID/TID and window values must come from the actual native range receipts and
SQLite schema, not guessed rank numbers or global time envelopes. The CPU test
fixtures demonstrate the schema but are explicitly **not GPU evidence**.

After a successful search, replay its registered winner with a fresh run ID:

```bash
python scripts/distillctl.py workloads run \
  --workload wan13b_nogan_resident_best \
  --search-plan /local/project/evidence/wan-search/trials/<confirmation>/execution-plan.json \
  --run-id wan-winner-replay-r1 \
  --coordinator-output /local/project/evidence/wan-winner-replay-r1
```

The source, assets, numerical gate and recipe remain bound; only the output/run
identity changes. Old artifacts are never overwritten. Moving to a new code
HEAD requires a new search and correctness gate, not reusing the old receipt.

## Gates, cost and failure handling

At most eight **candidate timing attempts**, plus the baseline and one winner
confirmation, are admitted. Native correctness preparations run serially and
are cached per candidate. These extra preparations are included in recorded
search cost; the budget is not falsely described as eight total GPU launches.
No C50, new capture or extra workload starts automatically.

All native GPU launches retain the shared node lock across audit, construction,
execution and teardown. Search-only resource sampling records driver free HBM
and node cgroup memory/events every two seconds. Its sampled minimum/peak is
not a continuous exact extremum; allocator peaks come from native rank receipts.
Both arms use identical instrumentation. Timings from earlier unmonitored runs
are not substituted for the newly matched denominator.

Every accepted measurement requires all 16 native status/semantic/version/
census/finite/dtype/memory gates, unchanged cgroup OOM counters, owned cleanup,
and disabled profiler. A controller/SSH exception is preserved with logs and
stops execution; it is not inferred to be OOM. Only explicit native CUDA or host
cgroup capacity evidence can be a capacity failure. Ambiguous/interrupted runs
require reconciliation and are not automatically retried under the same ID.

The trajectory comparator's existing coarse PASS/REVIEW policy is retained.
It does not establish strict P0 numerical parity. Normalized native observations
therefore remain `numerical_status=REVIEW` with the raw comparator status and
metrics preserved. `diagnostic_winner` and `strict_winner` are separate; strict
winner remains null without a true strict gate. No threshold is relaxed merely
to select a faster recipe.

Outputs include immutable `search.json`, profile/identity, fsynced selection
journal, preparation receipts, RPC logs, per-trial plans/native measurements,
resource observations, all candidate failures, confirmation and `result.json`.
The result reports **best tested within budget**, not universal best or a
publication-ready speed claim when numerical eligibility is REVIEW.

## Source closure and current validation boundary

The integrated main checkout lacked a dynamic dependency:
`probe_static_fsdp2_full5_world16.py`, and its imported
`StaticFSDP2Full5NsightRange`. Both were restored from measured C25 source
`d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d`. No profiling is enabled by the restore.

Wan's genuine two-cycle/trajectory workers and their cache/deferred-finite
helper APIs were ported from `7fa5b866c544594094b8d3c421959bbc64bf9206`, preserving
newer integrated Helios helpers. Search arm selection is a default-off opt-in;
canonical winner flags and profile lock hashes stay fixed. The import map keeps
both original provenance and adaptation hashes rather than attributing new code
to historical measured commits. CPU acceptance now also checks dynamic file
imports; a fresh no-weight remote import gate remains necessary.

CPU tests cover contract/tamper rejection, all five registrations, resume,
budget, REVIEW/strict separation, per-device SQLite unions, native adapter
routing and preparation serialization. GPU results must be read from an exact
run's receipts; compilation, unit fixtures, or import success are not GPU PASS.

<a id="hierarchical_planner"></a>
## Hierarchical planning: first integrated algorithm slice

This adds an opt-in search algorithm, a fixed-binding event-frontier solver,
and a parameterized G1/Rn/Cn compiler. It does **not** claim a parameterized
GPU training runtime for arbitrary resolutions/models, a new measured speedup,
or a replacement for the five canonical profiles. No native/GPU run is part
of this change. Existing Metropolis and registered-search defaults remain.

### 1. Current native decision spaces: hierarchical selector

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

### 2. Fine action scheduling: event-frontier DP / beam

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

### 3. Parameterized cycle and public CPU example

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

### Validation and scope

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
