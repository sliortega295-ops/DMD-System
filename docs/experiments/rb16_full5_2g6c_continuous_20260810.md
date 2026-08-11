# RB16 continuous Full5: 8G4S4T to per-node 2G6C

## Scope

Experiment `RB16-F5-CONT-8G4S4T-TO-2G6C` starts from the immutable
`G2/C10/EMA10` logical state. Step10-A uses eight Generator, four Student,
and four Teacher ranks with DP8/MB1/GAS2/global batch 16. The live process
then changes ownership to two Generator and six Critic ranks per node
(global 4G+12C, DP12/MB1/GAS1/global batch 12), executes Step10-B and steps
11--14, performs five Critic and five early-step EMA updates, and stops before
Step15-A at `G3/C15/EMA15`.

The experiment is a continuous Full5 resource-throughput point. Its Critic
batch differs from the matched-batch Candidate-A result and therefore cannot
be used to restate Candidate-A's causal speedup.

## Preflight decision

**GO, Iteration 280.** Reuse the old ACK-free `2G+6C` queue law, but not its
isolated checkpoint start. Reuse Candidate-A Step10-A math, but restore its
original two-selected-graphs-first order. The transition must be in memory:
retained roles keep their physical model, switching ranks preconstruct only
their future CPU model outside the timed range, G3 LoRA is redistributed after
Step10-A, and DP12 C10 Adam state is restored from the layout-independent
logical package.

Closed paths are ZeRO-2, ZeRO-3/FSDP, direct DeepSpeed process-group creation,
unordered subgroup creation, per-route optimizer ACK, and whole-model
transition checkpoint save/reload.

## Status

- Source implementation: **Implemented** at DMD HEAD
  `0fb0363d2772947df1446ce6e742ec551684dcbf` and Runtime HEAD
  `ec1462bfb84ca444b3beae216c66632744e0d369`
- CPU/source tests: **PASS, 219 tests, 1 skipped** at the final source HEAD,
  including the post-run regression gate
- Continuous Pipeline internal correctness: **PASS**, 16/16 ranks, run
  `rb16_f5_stricttrace_v4_z3resume_correctness_20260810_003`
- Continuous Sequential correctness: **Not Run**
- Frozen same-layout Sequential/Pipeline numerical comparison: **Not Run**
- Formal low-overhead timing: **Not Run**

The successful run establishes one continuous real-14B functional execution
from `G2/C10/EMA10` to `G3/C15/EMA15` under the frozen Pipeline contract. It
does not establish a causal Pipeline speedup: the run used correctness-mode
SHA and detailed instrumentation, and a same-layout Sequential pair has not
run.

The source gate verifies native torch initialization before DeepSpeed attach,
globally serialized subgroup creation, exact source/target rank sets,
DP8/MB1/GAS2 and DP12/MB1/GAS1 ZeRO-1 contracts, the two-selected-graphs-first
Step10-A order, absence of ACK groups/messages, a maximum two-batch
produced-minus-committed distance, five Critic/EMA iterations, and absence of a
whole transition checkpoint API in the continuous executor.

## Preserved transition attempts

The first four correctness launches are diagnostic failures and are not
performance results:

1. Attempt001 reached Step10-A and then OOMed because switching ranks retained
   the old 26.475-GiB Generator while loading the future resident Critic.
2. Attempt002 proved that `to_empty(meta)` did not release the old allocation:
   DeepSpeed retained aliases to the original Parameters.
3. Attempt003 rejected `torch.utils.swap_tensors` because the live Parameters
   had use count 3. The meta/swap path is closed rather than repeatedly patched.
4. Attempt004 used the standard alias-safe `module.to(cpu)` path. It released
   24.111 GiB, but left 2.3635 GiB allocated. The target resident Critic became
   ready at 31.4375 GiB instead of the isolated-path 29.0740 GiB and OOMed on a
   292-MiB allocation in the first backward. The residual matches the obsolete
   Step10-A ZeRO-1 BF16 flat owner. The bounded next repair destroys the obsolete
   engine hooks and drops partition views before `bit16_groups_flat`; it does
   not change trace, batch, loss, residency, precision, or distributed policy.

Attempt004 measured Step10-A at 93.703 s and the rank-6 transition at 44.169 s.
These remain diagnostic values until a complete same-layout correctness pair
passes. Evidence is preserved in
`artifacts/manifests/iteration280_full5_2g6c_continuous_attempt004_backward_residual_oom.json`.

## Iteration 281 correction: Step10-A backend parity

The Iteration-280 implementation selected DP8 ZeRO-1 for engineering
convenience during the first live role-transfer proof. That was not the frozen
Step10-A backend used by the validated 8G+4S+4T result. Attempt005 completed the
transition and all five Critic/EMA commits, but its Step10-A took 93.563083 s,
versus the prior DP8 ZeRO-3 anchor of 48.551620 s (1.927x slower). Attempt005 is
therefore preserved only as transition-feasibility evidence and is explicitly
superseded for performance.

Iteration 281 restores the measured contract:

- Step10-A Generator: DP8 ZeRO-3, MB1/GAS2/global batch 16;
- CPU optimizer offload, no parameter offload;
- block-local non-reentrant `zero3_metadata_tolerant` checkpointing;
- no frozen-block streaming and no activation offload in Step10-A;
- canonical G2 master/moments/scheduler restored into per-parameter ZeRO-3
  shards from the layout-independent logical package.

The post-update representation change is deliberately not an in-place engine
conversion. All eight source Generator ranks gather G3 trainable tensors in a
fixed collective order; rank 0 writes the BF16 target representation into a
preconstructed CPU model; the obsolete DP8 ZeRO-3 engine is destroyed; ranks
0/1/8/9 materialize full resident no-grad Generator models; and the exact G3
representation is broadcast inside the target Generator group. Critic ranks
continue to use resident Score plus DP12 ZeRO-1 CPU optimizer.

The corrected real-14B run and timing are **Not Run** at preregistration. No
formal Full5 claim is allowed until the corrected backend, G2-to-G3 update,
G3 redistribution, Critic/EMA versions, trace identities, and queue flush all
pass. See `iteration281_full5_2g6c_zero3_step10a_prereg.json`.

### Iteration 281 subgroup-initialization correction

The first corrected real-world16 launch did not enter Step10-A.  It stalled for
at least 40 minutes after DeepSpeed printed `Before creating fp16 partitions`:
ranks 0--7 were inside ZeRO-3 initialization while ranks 8--15 had already
reached the outer Gloo control barrier.  This is not evidence that ZeRO-3
training is slower.  DeepSpeed 0.18.4 calls a bare `deepspeed.comm.barrier()`
at `stage3.py:773`; in the combined world16 process graph that selects the
default world16 NCCL group even though only the DP8 Generator subgroup is
constructing an engine.  The earlier 48.551620-second Generator run used a
separate default world8 and therefore did not encounter this condition.

The bounded recovery does not edit DeepSpeed or change the workload.  A scoped
wrapper routes only bare barriers issued during `deepspeed.initialize` to the
engine DP8 group and restores the original callable immediately afterward.
Explicitly grouped barriers are untouched.  A real world4/DP2 ZeRO-3 gate,
with ranks 2--3 waiting outside the engine subgroup, completed with exactly
three redirected barriers on each engine rank, full world4 flush, return code
0, and no OOM.  One corrected real16 correctness retry is authorized; timing
remains blocked.


### Iteration 282 control and process-world correction

Exact rank-qualified Step10 exits require a common physical ZeRO-3 schedule. The deferred-common-selected primitive passed CPU parity and a default-world ZeRO-3 forward/backward/update gate. The same engine stalled at its first forward when hosted as a subgroup of a larger default world, despite exact DP2 parameter/optimizer group inventory. Therefore the single-world16 subgroup architecture is closed. The retained backend contract is still Step10-A `8G+4S+4T` with DP8 ZeRO-3, but the Generator must own a default world8 during Step10-A. A new phase-reinitialization gate is required before the real run: gather logical G3, destroy the obsolete engine/group, rejoin a global world16 target phase, then broadcast only G3 trainables to the four resident no-grad Generator ranks. Real-14B correctness and timing remain Not Run.


### Iteration 283 phase-reinitialization gate

A two-node real NCCL/DeepSpeed toy gate passed: independent node-local Phase-A worlds, a default-world ZeRO-3 padded update, logical state gather, obsolete engine/group destruction, explicit global Phase-B world creation, exact resident Generator broadcast, a target ZeRO-1 Critic update, and global flush. The production recovery is therefore executable in principle. It does not yet validate 14B construction, G2-to-G3 state, target admission, EMA, or Full5 timing.

### Iteration 284 production process-world refactor

The production recovery is preregistered to preserve Step10-A as node0's independent default-world8 DP8 ZeRO-3 engine. Node1 owns an independent default-world8 for four Student and four Teacher ranks. After exact padded Step10-A completes, all eight Generator ranks gather logical G3 in fixed order, the obsolete engine and both Phase-A worlds are destroyed, and the same live processes explicitly form global world16 for the per-node `2G+6C` Critic phase. Only G3 logical trainables are broadcast into separately preconstructed resident Generator targets; no ZeRO shard is reinterpreted in place. The 93.563083-second ZeRO-1 Step10-A remains superseded, while exact-current-trace real-14B timing is **Not Run** until correctness passes.

The Iteration-284 production code gate passed (`19` focused, `215` DMD, `322` Runtime tests). The actual worker now launches per node with `torchrun --standalone`, preserves Step10-A as node0 default-world8 ZeRO-3, and explicitly reinitializes global world16 only after G3 gather and old-world destruction. Real-14B correctness and timing remain **Not Run**.

### Iteration 285 strict synchronized trace

The active immutable v4 trace has bundle hash
`654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`.
It excludes the previously amplified local-exception branch from this measured
normal-path contract and requires one synchronized exit schedule per logical
operation. Step10-A uses exit `[1,0,0]`: every Generator rank executes four
logical and physical DiT calls per GAS wave, eight calls total, with zero
dummy padding. The target Critic rollout is also rank-synchronized *within
each logical step*, but different steps retain their official trace exits:
steps 10--14 execute `6/6/5/4/6` calls with stage counts
`[2,2,2]/[2,2,2]/[2,2,1]/[2,1,1]/[2,2,2]`. This is a correctness repair and
is not counted as a speedup.

### Iteration 286 source-layout release gate

The first two real-14B retries proved the strict DP8 ZeRO-3 Step10-A,
`G2 -> G3` AdamW commit and exact G3 extraction, then failed closed while
checking whether the obsolete source representation had released CUDA
storage. A bounded DeepSpeed referrer audit found Engine forward-hook closures,
duplicate client-module ownership and ZeRO parameter cycles retaining obsolete
CUDA partition shards. After exact G3 extraction, the final repair removes the
obsolete Engine owners and replaces only the no-future-use `ds_tensor` source
partitions with empty CPU tensors. A real DP2 restore/step/release gate passed
before another 14B launch.

### Final correctness result

Run `rb16_f5_stricttrace_v4_z3resume_correctness_20260810_003` passed the
fail-closed summarizer with no errors:

- all 16 rank records report `PASS`;
- Step10-A covers 16 exact source identities and performs one `G2 -> G3`
  update under DP8 ZeRO-3, MB1/GAS2/global batch 16;
- the transition drains 1,919 obsolete source partition tensors and
  3,890,967,568 bytes per source rank, reducing allocated CUDA storage by
  3.6245 GiB/rank before target materialization;
- G3 is redistributed exactly to target Generator ranks `0,1,8,9` and remains
  SHA-exact and gradient-free through the Critic phase;
- the target phase generates and consumes 60/60 payloads with 60 unique trace
  identities, performs five DP12 Critic commits and five EMA updates, and
  finishes at `G3/C15/EMA15`;
- EMA uses the official early bootstrap `decay=0.0` and matches G3 exactly on
  each update;
- producer and consumer queues are empty, outstanding transport is zero, and
  per-route optimizer ACK remains disabled at the final boundary;
- node cgroup peaks are 330.794 GiB and 158.991 GiB, with zero new cgroup OOM
  or OOM-kill events;
- target Generator peak allocation/reservation is 33.758/34.801 GiB and target
  Critic peak is 34.411/35.049 GiB.

Correctness-mode diagnostic phase maxima are 124.320 s for Step10-A, 50.656 s
for the layout transition, 168.497 s for the Critic phase, and 344.846 s for
the continuous Full5 window. These values include correctness instrumentation
and are **not** formal timing or a speedup result. The authoritative summary is
`artifacts/runs/rb16_f5_stricttrace_v4_z3resume_correctness_20260810_003/summary/correctness_summary.json`;
formal Sequential/Pipeline comparison remains **Not Run**.

### Iteration 287 official backward-semantics audit

The upstream normal path samples one pyramid exit schedule on rank zero and
broadcasts it for the logical operation. The frozen world16 Step10 trace uses
[1,0,0]. Mapping its 16 samples to DP8/MB1/GAS2 does not create two new
logical operations: both GAS rounds must replay [1,0,0]. Each round therefore
has selected calls [1,2,3], and all three remain connected to the final DMD
backward because stages 1 and 2 have no preliminary no-grad call.

The historical 48.551620-second Step15-A artifact used connected calls
[1,2,3] in wave 0 and [3] in wave 1. It is valid only for that different
control trace and is no longer an exact official-baseline timing denominator.
This correction changes no runtime code and supplies no performance result.
