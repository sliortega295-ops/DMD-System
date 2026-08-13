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

## Real-14B correctness attempt 01

- **Run:** `gan-rp16-correctness-resident-20260812-01`
- **Result:** **FAIL during Score construction; preserved as a non-capacity,
  non-mathematical launcher-asset bug.**
- **Observed:** all eight node1 Score ranks rejected the pre-cast state because
  130 official GAN-head tensors were missing.  The controller then cancelled
  node0.  No rank entered model compute, all GPUs were released, and both
  cgroups retained `oom=0`, `oom_kill=0`.
- **Failure signature:** the new launcher defaulted to the ordinary Score
  pre-cast `helios-score-transformer-mixed-bf16-v1`, while GAN-enabled model
  construction requires the already-built supplement
  `helios-score-transformer-mixed-bf16-gan-systems-v2` bound to bootstrap
  artifact `c528dd62...`.
- **Prior evidence/recovery:** the formal systems-only GAN path already pins
  the v2 GAN Score pre-cast.  Reuse that validated asset; do not weaken strict
  state-dict loading or synthesize heads independently per rank.
- **Bounded fix:** select the v2 GAN pre-cast by default and fail closed in the
  cheap input validator unless its marker is `PASS`, classified
  `systems_only_gan_precast`, and names the exact logical bootstrap artifact.
  This changes only initialization asset selection, not model math, batch,
  optimizer, trace, phase topology, or timed workload.

## Real-14B correctness attempt 02

- **Run:** `gan-rp16-correctness-resident-20260812-02`
- **Result:** **measured node0 cgroup-capacity failure during Phase-A selected
  execution; resident Phase-B GPU capacity remains undecided.**
- **Observed:** all 16 ranks constructed the corrected GAN models and wrote the
  `models_and_targets_ready` receipt.  Node0 then rose from about 633.18 GB at
  that boundary to a sampled 816,039,264,256 bytes, and `oom_kill` increased
  from 0 to 1.  Local rank6 was killed by SIGKILL; the controller cancelled
  node1.  There was no CUDA OOM.
- **Causal lifetime:** flex ranks 4--7 each retained both a pinned forward
  Critic target and a pinned reverse Generator target throughout Phase A.  The
  reverse target has no first use until after all five Phase-B Critic commits
  and after that rank's obsolete Critic can be released.  Four premature
  full-Generator backings consumed roughly another four model copies while
  Phase-A saved tensors were being offloaded.
- **Rejected recovery:** tail-4 Score block streaming changes Phase-B GPU
  residency and cannot resolve this pre-Phase-B host-lifetime failure.  A
  construction wave would only serialize creation while leaving the same
  long-lived backings resident.
- **Bounded recovery:** keep concurrent pre-cast construction, but do not
  construct the reverse Generator on flex ranks before Phase A.  After each
  flex rank releases its C1005 ZeRO-1 Critic, lazily construct and pin the
  reverse G201 target, then complete the existing B->A redistribution.  This
  preserves all model math and versions; its construction and pinning time is
  deliberately included in the formal handoff-to-A timing window.

## Real-14B correctness attempt 03

- **Run:** `gan-rp16-correctness-resident-20260812-03`
- **Result:** **the first lifetime fix materially lowered the Phase-A boundary
  but did not yet cross host admission.**
- **Measured change:** `models_and_targets_ready` fell from about 633.18 GB to
  431.64 GB.  The selected Generator activation-offload interval then added
  about 384 GB and reached a sampled 816,039,571,456 bytes; node0 `oom_kill`
  increased from 1 to 2.  All failures remained before Phase B, with no CUDA
  OOM; resident Score capacity therefore remains undecided.
- **New evidence:** after removing the four reverse targets, the remaining
  overlap is now the seven non-owner forward targets plus Phase-A saved
  tensors.  Only rank0's future Generator is needed before Phase A, because it
  is the destination for G201 extraction and the EMA owner.  Ranks1--3 do not
  need their Generator until G201 broadcast; ranks4--7 do not need their
  Critic until the A->B handoff.
- **Final bounded host-lifetime recovery:** preconstruct only rank0's G201
  destination.  Concurrently construct and pin the other seven forward targets
  after Phase-A activation handles are drained and the obsolete ZeRO-3 source
  is released.  Include this work in `handoff_to_b`; keep reverse Generators
  lazy after C1005 release.  This is not a construction wave and does not alter
  the workload.  If the same host-capacity failure persists after this layout,
  stop this recovery path rather than escalating further.

## Real-14B correctness attempt 04

- **Run:** `gan-rp16-correctness-resident-20260812-04`
- **Source:** immutable staged v7, HEAD
  `1ab00f986ce505dd996721fad61b090ba317bb1c`, source identity
  `d07ccf7d88274a81db71dd50fb78d877c3e17fcc2e5c06ecd69edd0956559fb0`.
- **Result:** **the bounded host-lifetime recovery passed, but resident Phase-B
  Critic failed actual CUDA admission.**
- **Progress before failure:** Phase A completed on all ranks, including the
  two-wave batch-16 Generator workload, one accumulation boundary and
  G200 -> G201. The cross-rank official-control receipt passed with zero local
  exceptions. All ranks entered the DP12/MB1/GAS1 Phase-B handoff.
- **Measured failure:** nine Critic ranks emitted explicit CUDA OOM in the first
  `_critic_loss` backward before any Critic commit. A representative request
  was 292 MiB. Ranks 4--7 had only 64--98 MiB driver free with about
  35.23--35.55 GiB PyTorch allocated; observed node1 ranks had about 266--274
  MiB free with 36.47 GiB allocated. No cgroup OOM occurred.
- **Resource evidence:** sampled cgroup peaks were 558,179,131,392 bytes on
  node0 and 124,305,186,816 bytes on node1. The prior empty
  `host_used_bytes` field prevents a claim about physical-host peak for this
  run.
- **Decision:** resident Score is a measured capacity failure. This is the
  preregistered trigger for exactly one tail-4 fallback; it is not formal
  timing evidence.
- **Evidence inventory:**
  `fad40d25c32391520048a8197cb40b5d795c42e4fc6585997dc16c98cd64ddbb`.

## Real-14B correctness attempt 05

- **Run:** `gan-rp16-correctness-tail4-20260812-05`
- **Result:** **non-capacity interface failure; no tail-4 conclusion.**
- **Failure signature:** eleven Critic ranks raised
  `ImmutableFrozenBlockStreamer.__init__() missing ... logical_component`
  before executing the streaming policy. Peer Gloo disconnects were
  downstream cancellation, not a second mechanism. No cgroup OOM occurred.
- **Bounded fix:** bind the existing streamer to
  `gan_rolephase_phase_b_critic`. Separately, make the low-overhead resource
  monitor emit a non-empty physical-host-memory value. Neither change affects
  model math, trace, batch, topology, update count or memory-policy choice.
- **Verification:** 47 focused role-phase tests pass on the staged server
  environment; shell syntax and Python compile checks pass.
- **Evidence inventory:**
  `157ba7344fc8b370bead0f92264437b3210ed66cebbb6f43a3d7502ff2519d7c`.

## Real-14B correctness attempt 06

- **Run:** `gan-rp16-correctness-tail4-20260812-06`
- **Source:** immutable staged v8, HEAD
  `a43b47b71a65bb2686e6178961b4a686541efe43`, source identity
  `5743bb3b5fe0d49dc331e43201ca75958b85a878031a27e1c36c9fb8aece2c44`.
- **Result:** **measured tail-4 CUDA-capacity failure; the current bounded
  action set is exhausted.**
- **Passed gates before failure:** all rank-qualified control receipts use the
  frozen trace and report cross-rank physical/logical call-count parity. Phase
  A again completed two samples per Generator/Score rank, eight official Score
  forwards per Score rank, one Generator accumulation boundary and
  G200 -> G201. All 16 ranks entered the DP12/MB1/GAS1 Phase-B topology. The
  tail-4 streamer was genuinely installed; surviving Score ranks recorded
  block 36--39 materialize/release events during checkpoint backward.
- **Causal failure:** before any C1000 -> C1001 commit, ranks 4--7 failed a
  292-MiB allocation in the first Critic `second_loss` backward. Driver-free
  memory was 124--268 MiB, PyTorch allocated 34.22--34.82 GiB, and
  3.20--3.45 GiB was reserved but unallocated. Rank10 later reached a
  362-MiB allocation with 352 MiB driver free; this and NCCL timeouts on peers
  are downstream of the first rank failures, not independent evidence.
- **Resource evidence:** node0 sampled cgroup/physical-host peaks were
  568,033,857,536 / 590,454,000,000 bytes; node1 peaks were
  189,531,119,616 / 212,715,000,000 bytes. Per-GPU sampled peaks were
  `[37458,37256,37530,37136,40152,40058,40030,40186]` MiB on node0 and
  `[40132,40140,40132,40132,40124,36622,36622,36620]` MiB on node1. Neither
  node changed its cgroup OOM counters.
- **Repeated-failure audit:** the 292-MiB first Critic checkpoint-recompute
  failure matches Iteration 277's bounded S4 negative result in mechanism and
  magnitude. Prior bounded work also established that static S5/S6/S8/S10
  escalation merely moved the failure deeper, and that this platform warns
  `expandable_segments` is unsupported. Therefore no allocator-flag retry or
  unregistered S5/S10 escalation is launched here.
- **Correctness:** **NOT REACHED.** No Critic commit, EMA update, reverse B -> A
  handoff or final summary can be claimed.
- **Performance:** **NOT RUN.** The requested three formal trials and the
  300--400-second target remain unmeasured; failed capacity runs are not timing
  denominators.
- **Gate:** **NO-GO for formal timing under the frozen Phase-B candidates
  `{resident, tail-4}`.** Expanding the action library (for example, explicitly
  preregistering a previously validated joint Score-streaming plus selective
  Critic-activation policy for the new DP12 topology) requires a new authorized
  iteration rather than silent patch escalation.
- **Evidence inventory:**
  `5808e07b650e271d1c53c69c1d5a1ef8dc8f2916e8daceb1731c8bc2acfac666`.


## Iteration 289 preregistration: DP12 tail10 + existing A128

- **Correction to Attempt 06:** Phase-B A128 pinned saved-tensor offload is
  installed unconditionally after the resident/streaming branch. Attempt 06
  therefore tested **tail4+A128**, not weight-only tail4.
- **Only authorized candidate:** DP12/MB1/GAS1/B12, frozen Score blocks 30--39
  streamed synchronously with window 1 and `HOST_SHARED_IMMUTABLE`, while the
  existing synchronous A128 activation policy remains unchanged. This adds six
  streamed blocks and removes 4,216,731,648 bytes (3.927 GiB) of static GPU
  storage per Critic rank relative to tail4 according to the audited block
  inventory.
- **Fail-closed receipts:** each Critic rank must report ten activation scopes,
  nonzero CPU copies, zero activation live bytes, the exact tail10 index set,
  exact streamer materialize/release reconciliation, zero resident bytes, and
  an idle streamer at the Phase-B boundary. The existing trace, updates,
  versions, queues, EMA and reverse-handoff gates remain unchanged.
- **Bounded stop rule:** one capacity/correctness run only. If confirmed
  tail10+A128 reproduces the first-backward CUDA OOM, do not try S11/S12 or an
  allocator workaround.
- **Current execution status:** **NOT RUN.** Local compile/shell/dry-run, 31
  focused role-phase tests and 184 CI refactor tests pass. The full maintained
  suite has 367 passed, 10 skipped and two pre-existing failures caused by
  archived Iteration-281 manifests absent from this worktree. Both BAAI SSH
  endpoints authenticate and then time out before any
  command output. Live GPUs, cgroup OOM counters, assets, remote tests, source
  staging and the real world16 run therefore remain unverified.
- **Performance status:** **NOT RUN.** No diagnostic or failed-capacity time may
  be used as a formal result.
