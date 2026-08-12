# GAN Role-Phase World16 Execution (2026-08-12)

## Iteration preflight

- **Gate:** real-14B GAN role-phase Full5 correctness before formal timing.
- **Authorized hypothesis:** a bounded `8G+8Score/B16 -> 4G+12Critic/B12`
  role transition can complete logical steps 1000--1004 in 300--400 seconds
  without changing either phase-local workload.
- **GO / NO-GO:** **GO for implementation and no-model topology validation.**
  Real-model execution remains gated on tests, exact input receipts, empty GPUs,
  node locks, and a passing two-node topology lifecycle probe.
- **Frozen source base:** branch
  `codex/gan-rolephase-world16-exec-20260812`, base HEAD
  `afd64eb4552b28e12d502452929e8d58cad3236a` plus the explicitly recorded
  dirty implementation diff until it is committed and staged.
- **Frozen workload:** official 384x640 / BF16 / logical steps 1000--1004;
  systems-only GAN bootstrap; Phase A DP8/MB1/GAS2/B16, one Generator update;
  Phase B DP12/MB1/GAS1/B12, five Critic updates; EMA800->801 once after the
  Step1000 Critic commit.
- **Frozen logical/physical distinction:** the schedule labels are
  G200/C1000/EMA800 -> G201/C1005/EMA801, while Adam/scheduler counters remain
  physical G2/C10 -> G3/C15.
- **Frozen trace:** one immutable rank16 operation-keyed bundle; Phase A uses
  all 16 slots and Phase B projects slots 0--11. Trace repair is correctness,
  never a speedup.
- **Allowed changes:** executor/transport wiring, process-group lifecycle,
  fail-closed launcher, receipts/summarizer, bounded resident/tail-4 capacity
  fallback, and bugs that do not change model math.
- **Rejected/superseded paths:** ZeRO-2 (known double-reduction failure),
  global-world16 ZeRO-3 subgroup execution (known stall), 12+4 uneven Critic
  mapping, batched traces, changed loss scaling, new baseline reruns, and any
  matched-batch speedup claim.
- **Matching prior evidence:** the validated recovery uses node-local default
  world8 ZeRO-3 for Phase A and DP12 ZeRO-1 CPU optimizer for Phase B. The old
  pre-created single-world process-group prototype is not the production
  lifecycle and must not be invoked by the real worker.
- **New falsifiable question:** after exact state/trace restoration and bounded
  A->B->A handoffs, do all 16 ranks pass versions, update counts, Score-call
  inventory, P2P identities, queue flush, EMA recurrence, and allocator/cgroup
  admission; and, only then, is the three-run median Full5 wall time <=400 s?

## Live substrate evidence

At `2026-08-12T17:23:29Z` / `17:23:30Z`:

- node0: `job-ae0cd37b-2a89-489c-9a15-09932329c44a-master-0`;
- node1: `job-a248b179-3c88-49c3-ad58-74597d6f9133-master-0`;
- both nodes exposed 8 GPUs with no compute processes;
- both cgroups reported `oom=0`, `oom_kill=0`;
- environment: PyTorch `2.10.0+cu128`, DeepSpeed `0.18.4`, 8 visible GPUs;
- logical package, trace bundle, Generator/Score assets and both precast trees
  existed on both nodes;
- `/share/project` was 97% used (487 TiB reported free), so artifacts must stay
  bounded and no redundant model copies may be created.

## Claim boundary

This experiment is a systems-only phase-local B16/B12 resource/performance
Pareto point. It is not an official checkpoint-1000 continuation, not a
same-global-batch Sequential-vs-Pipeline speedup, and not training convergence
or quality evidence. Failed or unexecuted gates remain `NOT_RUN`.
