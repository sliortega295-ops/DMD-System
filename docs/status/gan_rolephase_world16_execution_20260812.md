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

## Validation attempt 01

- **Run:** `gan-rp16-validate-20260812-01`
- **Result:** **FAIL before model construction; preserved as non-capacity
  evidence.**
- **Observed:** both node launchers returned 1, all GPUs were released, and
  cgroup `oom`/`oom_kill` remained zero.
- **Failure signature:** validation called
  `get_process_group_ranks(GroupMember.NON_GROUP_MEMBER)` for the inactive
  Phase-B role.  PyTorch's shared non-member sentinel was then misreported as
  an unrelated singleton subgroup such as `(15,)`.
- **Root cause:** validation-only introspection bug; process-group creation and
  the frozen role topology were not changed.
- **Bounded recovery:** subgroup members inspect the real handle; non-members
  must prove they hold `NON_GROUP_MEMBER` and use the world-ordered static rank
  contract.  No model math, batch, optimizer, trace, or schedule changes.

## Validation attempt 02

- **Run:** `gan-rp16-validate-20260812-02`
- **Result:** **FAIL at the reverse local-world rendezvous; preserved as
  non-capacity evidence.**
- **Observed:** the global Phase-B topology passed the corrected check, but
  local rank 0 on each node failed recreating world8 with `EADDRINUSE`; GPUs
  were released and cgroup `oom`/`oom_kill` stayed zero.
- **Root cause:** the worker reused torchrun `--standalone`'s original
  `MASTER_PORT`, whose elastic-agent TCPStore remains alive after the first
  process group is destroyed.
- **Bounded recovery:** assign a distinct, explicit node-local reverse port to
  each node.  This is rendezvous plumbing only and does not change model math,
  phase topology, batch, trace, or timed-workload semantics.

## Validation attempt 03

- **Run:** `gan-rp16-validate-20260812-03`
- **Worker result:** **PASS on all 16 ranks.** The lifecycle completed local
  world8 -> global world16 4G/12C -> local world8, all launchers returned zero,
  and cgroup `oom`/`oom_kill` remained zero.
- **Initial summarizer result:** FAIL because `/proc/meminfo` byte arithmetic
  was emitted by awk in scientific notation and the offline parser required an
  integer literal.  This did not affect the worker or GPU schedule.
- **Bounded recovery:** accept valid numeric scientific notation for preserved
  monitor rows and emit integer-formatted host bytes in future runs.  Re-run
  only the offline summarizer; do not repeat the passing GPU lifecycle gate.
