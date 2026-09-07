# Joint DMD planning: implementation and acceptance ledger

## Scope

The registered-recipe search remains available and its published evidence is
unchanged. The new, opt-in `--strategy model-based` implements a separate search
algorithm over the **same** `core.graph.ExecutionGraph`, not a second training
loop. Its proposed objective is native **cycle 2** wall time after one in-process
warm-up. Initialization, correctness probes, calibration and profiling are not
part of that denominator.

This feature branch is **not yet an end-to-end GPU-validated planner**. CPU
simulation, native backend support and GPU validation are different statuses.

## Method being implemented

1. An adapter freezes the workload, versions, sample/RNG identities and precision
   and emits operations plus supported physical implementations.
2. Sharing candidates are generated from equal frozen checkpoint content,
   architecture, dtype and placement, disjoint trainable storage, and explicit
   adapter/recompute scopes. Maximal sharing is **not** assumed fastest: it can
   prevent concurrent execution and incur handoff costs.
3. Existing kernels and their supported tuning knobs form an implementation
   library. Source inventory alone is not a correctness or composability proof.
4. Source/environment/hardware/assets/**precision**-bound native operation or
   microbenchmark receipts provide latency and workspace costs. Nsight unions
   are diagnostic features, never operation wall-time substitutes. Missing costs
   stay `UNCALIBRATED`; there are no made-up throughput estimates.
5. A dependency/resource simulator evaluates candidates, counts shared physical
   storage once and retains output tensors through their last consumer. It does
   not assume two operations can overlap simply because their DAG edges permit
   it. GPU/link reservations and shared-adapter mutual exclusion also apply.
6. Bounded Metropolis search mutates decisions; baseline and incumbent seeds are
   evaluated before spending the proposal budget. A bounded exhaustive CPU
   oracle checks the search on small fixtures.
7. A supported native lowerer must consume the selected decision vector exactly.
   It may not silently run the nearest old recipe. Only new native receipts can
   establish correctness, feasibility, latency or a winner.

The profile/cost-model/simulator/search decomposition is inspired by
[RealHF](https://arxiv.org/abs/2406.14088). The DMD-specific search dimensions are
frozen-base aliasing with independent LoRA/replay scopes, legal G/C reordering,
bounded cross-cycle lookahead, placement, and existing kernel implementations.
MCMC itself is not presented as a new research contribution. No RealHF code was
copied into this implementation.

## Implemented CPU core

| Component | Location | Current evidence |
|---|---|---|
| Versioned physical choices over existing graph | `search/plan_ir.py` | RAW/WAR/recompute, rank/state/resource and lifetime rejection tests |
| Frozen-backbone sharing generator | `search/sharing.py` | Disjoint adapter storage, immutable base identity; mutable-base sharing rejected |
| Operation costs | `search/cost_model.py` | Receipt hash/sample/identity reconciliation; profiled cost rejection |
| Resource and memory simulator | `search/simulator.py` | Hand-calculated serial, overlap, adapter mutex, memory tradeoff and link contention fixtures |
| Bounded search | `search/model_search.py` | New non-seed combination found; small-space exhaustive oracle; deterministic planner RNG |
| Existing implementation inventory | `search/implementation_library.py` | AST/source hashes without Torch imports; no generated kernels |
| Precision identity | `search/precision.py` | Explicit TF32 policy and runtime assertion; verification does not mutate Torch |
| Unified CLI | `workloads search --strategy model-based` | CPU-only path; legacy registered search unchanged |

Simulator times in unit tests are **synthetic nanoseconds**, not real model or
GPU measurements. A resource envelope includes its internal overlap; adding its
NCCL/kernel union again would double count it. `critical_predecessors` are model
predictions, not proof of a measured CUDA critical path. The simulator reports
cycle-2 activity that begins before cycle-1 completion so that finite-pipeline
fill/drain effects remain visible; two cycles alone do not prove asymptotic
steady state.

## Commands (CPU only)

```bash
python scripts/distillctl.py workloads search \
  --strategy model-based --workload livewan_crossscale_pipeline_best --dry-run

# problem.json must be adapter-produced. Real costs require source receipts.
python scripts/distillctl.py workloads search \
  --strategy model-based --workload wan13b_nogan_resident_best \
  --problem problem.json --cost-model costs.json \
  --max-proposals 10000 --search-timeout 600 --search-seed 0 \
  --output new-search-result.json
```

The second command runs the simulator/search, **not GPU workers**. A CPU fixture
cannot be promoted to a native plan. Existing outputs are never overwritten.
The existing registered native search remains the default strategy.

## Acceptance remaining before a planner speed claim

- Workload adapters emitting source-bound planning graphs from actual workload
  contracts, not user-written synthetic graphs.
- Exact generated-plan lowering and runtime receipts; no unsupported TP/PP,
  dynamic repartition or fabricated pool mappings.
- Native operation calibration and memory lifetimes for each admitted binding.
- Wan, then LiveWan, then H3 Grid1000 pilots under owned world16 locks. H3 keeps
  approximate Grid1000 semantics and FA3 CPU replay storage explicit; boundary
  offload `none` does not mean there are no CPU transfers.
- Per workload: at most eight timing runs (baseline, incumbent, up to five new
  plans, independent confirmation). Each uses two cycles, one initialization,
  unprofiled cycle 2 only. Numerical REVIEW is not strict equivalence.
- Canonical winner changes require measured evidence and separate review; this
  branch neither changes canonical profile hashes nor auto-merges main.

## Initial CPU acceptance, 2026-09-07

Inherited source: `d310597e774b18fea98118e00a58a4b912e89f19`, based on fetched main
`e69f4bf8b8bb64f54542fea74c452b53e090b526`. Branch:
`agent/dmd-joint-planner-20260907`.

- Full `tests/test_refactor_*.py`: **498 passed, 17 skipped**; skipped tests are
  not claimed as passing GPU checks.
- Existing five-workload integration/source-hash/compile/shell checks: **PASS**.
- Native generated-plan GPU runs/calibration: **NOT_RUN**.
- No historical timing has been relabeled as a result of this branch.

Small local evidence:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/`.
Raw model/profile assets remain remote. Two-node read-only preflight found 16
idle A100s and the prior Torch 2.10/CUDA 12.8 environment, but this does not grant
a future GPU lock; ownership must be checked again at launch.
