# H3 measured action costs → common search → native performance closure

**Completed, without repeating the old correctness/depth/self-repeat matrix.**
This is a native two-cycle **diagnostic** performance measurement, not a new
strict numerical-equivalence result or a demonstrated speedup over a matched baseline.
Grid1000 remains an approximate-sigma workload, not exact continuous DMD.

## Result

| Item | Seconds | Scope |
|---|---:|---|
| Calibration cycle1 | 775.162219220 | Action fences/barriers; **not native timing** |
| Calibration cycle2 | 771.453950295 | Component calibration only |
| Model prediction, selected cycle2 | 771.399059042 | Shared simulator prediction |
| Independent native cycle1 | 772.776955216 | Warm-up/cold diagnostic; excluded |
| **Independent native cycle2** | **769.785204601** | **Unprofiled, same process and initialization** |

Prediction minus measured: **1.613854441 s / 0.209650%**.
The denominator is the **maximum rank-local CUDA-fenced host cycle duration**,
not a cross-node min-start/max-stop wall. Initialization, plan construction,
calibration hooks, final-state probes, correctness JSON/gathers and teardown
are outside this native cycle timer. The existing two start/stop CUDA events
per cycle remain native observer instrumentation; no Nsight was launched.

This reproduces roughly the historical 772 s level. It does **not** establish
additional search-induced acceleration. The old 774.4016 s action correctness
run includes in-cycle full stochastic hashes; neither that run nor historical
772.332 s is used as a matched native denominator here.

## What was actually searched and executed

- Runtime source, both nodes clean: `bf59e39b8a6fe158ad15ebae07be7df71369f5fd`.
- Calibration: `jointp_h3_actions_perf_20260907_r1-calibration`, **window3**.
- Native confirmation: `jointp_h3_actions_perf_20260907_r1-confirmation`, **window5**.
- 22 measured action components, each bound to the observed physical layouts of
  all 16 ranks, source/environment/assets/precision and cold/warm context.
- 352 rank-action observations: 16 ranks × 2 cycles × 11 actions.
- Common `PlanningProblem` / resource-and-lifetime simulator / search / typed
  native lowering, not a workload-name lookup into a historical timing table.
- 20 CPU assignments evaluated; **5 feasible**, **15 UNCALIBRATED**. Only depth2
  and LoRA epilogue N128 have current action costs and prior native action proof.
  Other depth/tile costs were not guessed or imported from another source.
- All five windows have **equal predicted latency**; window5 wins the existing
  deterministic tie-break. This is not evidence window5 is faster than window1–4.
- Window controls how many separate B1 groups remain live. It does not batch
  rollouts, change GAS or create concurrent shared-backbone execution.

```text
Calibration:  g r0 r1 r2 c0 r3 c1 r4 c2 c3 c4
Confirmation: g r0 r1 r2 r3 r4 c0 c1 c2 c3 c4
```

All these actions reserve the same world16 pool; the model therefore adds their
measured envelopes, with no invented free G/C overlap. In fenced cycle2
calibration: Student update **142.624 s**, five rollout producers **326.202 s**,
five Fake updates **302.573 s**. These are calibration components, **not** an
Nsight compute/NCCL decomposition or independently measured native phase walls.

## Hard gates and numerical boundary

Both runs: **16/16**, **2/2 cycles**, final **G2/C10/EMA0**; per-rank per-cycle
Student/Fake/Teacher forwards **24/6/1**, backward **1/5**; 96 unique rank-qualified
samples per world cycle; Grid **300 wraps / 600 executions / 600 hits / 0 misses**.
All logged losses finite. No observed CUDA OOM; cgroup `oom` and `oom_kill`
deltas zero. Two-node GPU/owned-process/port/lock readback PASS.

32 native training/core files remain byte-identical to prior full-draw action
proof `878aff490ce28c661ac1d927f08f8846992baf45`. Calibration and native performance
use **metadata-only action evidence**; original correctness defaults retain full
value hashing. No replacement randomness is drawn. Compared across these runs,
192 stochastic actions match RNG boundaries/metadata, and 320 Fake consumers
remain RNG-neutral. This does not re-prove equality of current tensor values.

Numerics remain **REVIEW**, not strict PASS. For 192 rank/cycle loss observations,
symmetric relative error (denominator=max(abs(left),abs(right),1e-12)) has
median **0.4017%**, p95 **9.5434%**, max **65.6615%**. The maximum is cycle2 Student
rank10: **0.00441465 vs 0.01285625**, absolute difference 0.00844160. Maximum
absolute error is Fake cycle2 rank8/update1: **0.33506083 vs 0.38781497**.
These outliers are retained; matching RNG/census does not prove strict gradient
or training-trajectory equivalence. No threshold was relaxed to manufacture PASS.

## Memory and transfers

Native max allocated: **33.394808 GiB**;
reserved: **38.130859 GiB**.
Sampled node cgroup peaks: **451.533844 /
451.310093 GiB**. Kernel `memory.peak`
is unavailable; these are sampled maxima. Rank RSS high-water observations top
out at 29.753139 GiB.
Sampled minimum driver free: **498 /
636 MiB**. Completion does
not certify a robust 2 GiB *driver-free* margin. Simulator memory models tensor
allocation plus retained storage, not worst-case allocator fragmentation.

Boundary activation offload remains **none**, segment1; FA3 output/LSE storage
remains **cpu_staged**. Rank0-only logs show per cycle 300 D2H entries, 300 H2D
entries, 300 prefetches, 0 misses and 288 D2H backpressure waits. Each direction:
164,794,448,000 B in cycle1; 164,796,632,000 B in cycle2. These counters are not
all-rank totals or measured exposed transfer overhead. No transfer-elimination
claim is made.

## Reproduction, failures and source fix

Persistent remote root:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/h3_performance_r1/`

Exact deployed runtime:
`/share/project/shuchenweng/yongyan_liu/DMD-System_deploy/joint-h3-performance-bf59e39b8a6f-20260907`

Same validated MiniMax environment reused (Torch2.10+cu128, Diffusers0.40.dev,
Transformers4.57.6, PEFT0.18.1); no environment installation, driver change or
foreign process cleanup. Model/cache/Grid/FA3/source hashes and node commands
are in the small manifests. Raw rank receipts/logs/cost sources stay remote.

- Source preflight d730a89 was superseded by provenance hardening before GPU use.
- First local controller preflight missed its RPC log parent: no GPU launched.
- First confirmation preflight correctly rejected a plan JSON round-trip bug:
  `memory_limits` string keys sorted `0,1,10,...,2`, changing the order of the
  output memory-row array, not any memory/time value. CPU evidence verifies all
  fields equal after sorting only that array. Failed artifacts remain unchanged.
- CPU search was repeated **from the already saved PlanningProblem/cost model**,
  using the common parser/search/lowerer. No costs/timings were hand-edited;
  both native nodes independently revalidated the model/receipts before launch.
- After GPU completion, local commit `eb0cea1` canonicalizes simulator memory rows
  by numeric rank and adds the reproduced world16 JSON regression. It does not
  change native training. Runtime numbers above still belong to **bf59e39**, not
  the later fix/publication commits. Old emitted plans remain source-pinned;
  do not relabel them as measurements of a newer HEAD.
- Pre-run exact bf59e39 focused tests: **328 PASS**. Post-run serializer regression
  was RED on bf59e39 and GREEN with the scoped fix (**30 focused PASS**).
  Final post-fix source/planner/H3 regression suite: **352 PASS**; exact command
  and fix HEAD are in `manifests/tests-publication.json`.

## Stop boundary

Exactly one new two-cycle calibration and one two-cycle native confirmation
completed. No new Wan/LiveWan matrix, C50, Helios, Nsight, kernel or topology run.
No push, PR mutation, main merge or new strict winner in this unit. The measured
cost→search→native feedback unit is closed; broader depth/tile search and a new
matched speedup comparison remain separate, **NOT_RUN** tasks.
