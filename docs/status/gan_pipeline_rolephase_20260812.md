# GAN pipeline role/lifecycle handoff (2026-08-12)

## Status

GPU experimentation is stopped at the user's request.  Both 8xA100 nodes were
rechecked after the last launch: no GAN/Helios/torchrun process remained and
both cgroups reported `oom=0`, `oom_kill=0`.

This document preserves the current implementation and measurements for a
pipeline redesign.  It does **not** claim that the requested 300--400 s target
was reached.

## Frozen workload

- 2 nodes x 8 A100-40GB, world size 16.
- Persistent ranks 0--7 Generator and ranks 8--15 Score/Critic.
- Official 384x640, BF16, logical steps 1000--1004.
- Global batch 16 = two waves x eight lanes, DP8/MB1/GAS2.
- Exactly one Generator update, five Critic updates, and one EMA update.
- Operation-keyed trace bundle:
  `d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15`.
- Workload contract digest:
  `e7bb6638e2d2d7356b93c90c0ac73a577f59cdeacf4f70a0c45b2cd29fac95a0`.

No loss, trace, batch, update, topology, or precision change is counted as an
optimization.

## Candidate implementation

The latest admitted policy was:

| Component | Policy |
|---|---|
| Generator train/update path | stream all 40 frozen blocks |
| Generator Critic-only rollouts | materialize the selected frozen blocks once for the bounded no-grad phase, then release |
| Score/Critic | keep blocks 0--29 resident; synchronously stream tail blocks 30--39 |
| Score Critic saved tensors | synchronous pinned-CPU offload for tensors >=128 MiB |
| Streaming window | 1 |
| Host ownership | `HOST_SHARED_IMMUTABLE` |
| Prefetch | disabled |

The actual server worker/runtime source is preserved under
`artifacts/source_snapshots/gan_pipeline_rolephase_20260812/`.  The orchestration
and fail-closed policy promotion are in
`scripts/distillschedule/run_gan_systems_formal_trial.py`.

## Capacity and correctness gate

The detailed-event `G40/S10/A128` run passed:

- 16/16 ranks `PASS`;
- Generator ranks: G=1, EMA=1;
- Score ranks: Critic=5;
- fairness, compute inventory, DP8/MB1/GAS2/B16, queue, version, and lifecycle
  receipts passed;
- no CUDA OOM and no cgroup OOM/OOM-kill;
- maximum allocated: 39,111,174,144 bytes;
- maximum reserved: 40,290,484,224 bytes;
- all block streamers and activation handles ended at zero live bytes.

Per rank:

| Role | Materializations | Materialized bytes | Other evidence |
|---|---:|---:|---|
| Generator | 600 | 421,673,164,800 | one phase-resident enter/exit |
| Score | 500 | 351,394,304,000 | tail-10 streaming |

Representative Score rank 8 opened 20 Critic scopes, issued 413 unique CPU
copies, offloaded 58,591,641,600 logical bytes, peaked at 5,603,328,000 live
CPU activation bytes, and ended at zero.

The observed global Full5 window was **868.488686 s**, but this was a
diagnostic capacity run with detailed shared-filesystem event logging and is
not formal-speedup eligible.

Curated evidence:
`artifacts/formal/gan-pipeline-rolephase-20260812/capacity_gate/`.

## Low-overhead timing attempt

The next launch removed only detailed per-pack/unpack/materialization JSON
events and kept exact rank timing receipts and summary counters.

### What completed

- all 16 model workers returned `PASS`;
- all 16 unprofiled exact timing receipts returned `PASS`;
- update counts remained G=1 / Critic=5 / EMA=1;
- no CUDA or cgroup OOM occurred;
- exact global window from earliest rank start to latest rank stop:
  **824.685006 s**;
- 80 logical samples / Full5 gives **0.097007 samples/s**, or
  **0.006063 samples/s/GPU**.

### Why it is not a formal result

The postprocessor still required detailed per-block `materialize_block` rows,
while formal low-overhead mode intentionally emitted only aggregate
install/remove counters.  It therefore failed with:

```text
eager block evidence inventory drift: .../rank_000_generator.jsonl
```

This is a receipt/postprocessing failure after successful model execution, not
a capacity failure.  The launch remains non-claimable because:

1. the aggregate receipt gate did not close;
2. only one low-overhead launch was run;
3. the preregistered three paired trials were not run.

Raw receipts and the preserved failure result are under
`artifacts/formal/gan-pipeline-rolephase-20260812/trial_0/`.

## Performance context

| Variant | Full5 seconds | Evidence status |
|---|---:|---|
| Sequential fixed-policy | 656.125 / 596.234 / 658.338; median 656.125 | formal, 3 paired trials |
| Original Pipeline fixed-policy | 1139.194 / 1043.215 / 1170.197; median 1139.194 | formal, 3 paired trials |
| G40/S10/A128 detailed | 868.489 | diagnostic only |
| G40/S10/A128 low-overhead | 824.685 | one measured exact window; not claimable |

The one low-overhead window is 27.61% shorter than the old Pipeline median
(1.381x throughput), but remains 25.69% slower than the Sequential median
(`Sequential / candidate = 0.796x`).  It is far outside the requested
300--400 s range.

These ratios are context for implementation review, not a new formal speedup
claim.

## Known implementation concerns for correction

1. Generator and Score phases are still serialized more than the intended
   wavefront; the memory-policy improvement is not equivalent to a corrected
   overlap schedule.
2. Generator and Score timing starts differ because each rank marks its first
   role-local model compute; global timing correctly uses the earliest start
   and latest stop, but the gap exposes role startup/idle time.
3. Score still performs 500 tail-block materializations per rank and large
   synchronous activation D2H/H2D traffic.
4. Pinned activation offload is synchronous; no prefetch/overlap claim is made.
5. The formal summary path needs an aggregate-counter receipt mode bound to the
   validated capacity manifest.  It must not infer or fabricate missing
   detailed events.
6. The 39.11 GB Score allocated peak leaves little capacity margin.

## Handoff decision

`STOPPED_FOR_PIPELINE_IMPLEMENTATION_CORRECTION`.

Focused review checks on the pinned server environment passed:

- DMD launcher/receipt/process-safety suite: 43 passed;
- executed runtime lifecycle suite: 34 passed, one existing PyTorch backward-hook warning;
- Python compile checks and shell syntax checks: passed.

No further GPU run, Nsight retry, offload search, or three-trial timing should
start from this handoff.  A corrected implementation should first restate its
legal overlap/version state machine, preserve the frozen workload contract,
and pass one real capacity/correctness gate before timing.
