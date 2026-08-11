# Iteration 291: strict-v4 Step10-A and representation-transition optimization audit

## Status and scope

**Measured diagnostic + source audit; no new GPU launch.** This iteration asks whether the
current continuous Full5 candidate has removable overhead in (1) strict-v4 Step10-A and
(2) the live `8G+4S+4T -> per-node 2G+6C` representation transition. It does not change
trace, batch, loss, update order, placement, precision, or numerical contract, and it does
not report a new speedup.

Live preflight on 2026-08-10 found both nodes reachable, repository HEAD
`02c059d73e98d13532d15a10e6c59c63c5923e9c` on
`distillschedule/rb16-full5-2g6c-continuous`, a clean worktree, no GPU processes, no current
cgroup OOM increment, and the historical `oom_kill=1` counter on both nodes. CPU/source
tests passed: **222 passed, 1 skipped**.

### Frozen contract

- official 384x640 and strict-v4 trace hash
  `654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3`;
- Step10-A: DP8 ZeRO-3, MB1/GAS2/global batch 16, two selected graphs followed by two
  backwards, one `G2 -> G3` commit;
- target phase: global 4G+12C / per-node 2G+6C, DP12 ZeRO-1 CPU optimizer;
- continuous versions `G2/C10/EMA10 -> G3/C15/EMA15` and exact queue flush.

The historical 48.551620-second Step15-A and the 93.563083-second ZeRO-1 transition probe
are not denominators: their graph/backend contracts differ. ZeRO-2, in-place ZeRO shard
reinterpretation, meta/swap repair, and default-world16 ZeRO-3 subgroup initialization
remain closed paths.

## 1. Step10-A: what the existing evidence says

The low-overhead strict-v4 run measured **123.115506 s** median Generator wall time:

| subphase | median seconds |
|---|---:|
| wave 0 rollout | 20.675323 |
| wave 0 backward | 38.368997 |
| wave 1 rollout | 21.036667 |
| wave 1 backward | 39.909053 |
| final engine step | 1.150879 |
| total gradient-file wait | 0.009794 |

Thus the detached-gradient exchange is not the Step10-A bottleneck. Required rollout and
checkpoint backward dominate.

The preserved Nsight diagnostic has profiler overhead (summed median phase wall
133.149125 s versus 123.115506 s low-overhead), but its causal decomposition is useful.
Summing the per-phase medians only for orientation gives:

| interval class | summed median seconds |
|---|---:|
| model compute kernels | 28.239920 |
| NCCL kernel union | 49.235247 |
| copy-engine union | 27.814135 |
| GPU-kernel-or-copy union | 103.741282 |
| no kernel/copy activity | 29.408873 |

These classes overlap; they must not be added into a synthetic speedup. Rank 0 issued
35,187 NCCL kernels across the two rollout/backward waves. The dominant NCCL kernel is
`ncclDevKernel_AllGather_RING_LL`; ReduceScatter is only about 0.1--0.15 s per backward.
This identifies repeated ZeRO-3 parameter materialization, not gradient reduction, as the
primary communication target.

Rank-0 CPU/GPU copy attribution across the capture is:

| copy path | GiB |
|---|---:|
| Device -> pageable host | 57.747357 |
| Pageable host -> device | 29.185779 |
| Device -> pinned host | approximately 0 |
| Device -> device | 221.310328 |

The largest D2H sizes are 113,049,600, 46,694,400, and 30,105,600 bytes, each repeated 80
times in a phase. They exactly match BF16 `[1,11040,5120]`, `[1,4560,5120]`, and
`[1,2940,5120]` storage sizes and do not occur as same-sized tensors in the frozen
checkpoint inventory. This strongly indicates block/checkpoint activation-boundary
traffic. However, the Nsight call-chain table is empty, while the runtime contract says
explicit activation offload is disabled. The producer is therefore **not yet proven**;
pinned conversion must follow a one-run ownership census rather than be applied blindly.

### Step10-A optimization opportunities, ranked

1. **Persist the small trainable materialization set in ZeRO-3.** The current config uses
   `stage3_param_persistence_threshold=0`, `stage3_max_live_parameters=256M`, a 64-MiB
   prefetch bucket, and `overlap_comm=false`. The logical Generator contains 818 trainable
   tensors / 1,251,442,688 elements. A 2,000,000-element threshold covers 737 tensors and
   1.789 GiB BF16; a 4,000,000 threshold covers 817 tensors and 2.316 GiB BF16. The frozen
   checkpoint contributes only 0.019 GiB below the 2M threshold. Existing peak allocation
   is about 31.24 GiB/rank, so a bounded 2M-persistence candidate is statically plausible,
   but still requires real allocator admission and exact-output validation.
2. **Tune prefetch/live/reuse policy, not the mathematical call count.** Measure a small
   preregistered set: baseline; 2M persistence; and 2M persistence plus a larger bounded
   prefetch/live window. `stage3_module_granularity_threshold` is available in the pinned
   DeepSpeed version, but should be tested only after persistence because it changes fetch
   grouping and can raise peak memory.
3. **Attribute then pin the pageable boundary traffic.** First install a no-policy-change
   saved-tensor/ZeRO materialization census that records storage IDs, shapes, generation,
   and owning call without retaining tensors. If the large copies are checkpoint boundary
   inputs, compare the same strict-v4 run with a bounded pinned arena and synchronous
   semantics before attempting async prefetch.
4. **Do not optimize file exchange first.** Gradient wait is 9.8 ms total. Replacing the
   Step10-A file handoff may improve cleanliness, but cannot explain the current 123 s.
5. **Do not remove selected calls/backwards.** strict-v4 requires selected stages 0/1/2 in
   both GAS waves. The old reduced-backward anchor is not a legal target.

## 2. Live representation transition: what is long

The continuous correctness run measured a **50.655504 s** transition on node0 and about
**47.825 s** on every node1 rank. Per-rank evidence includes:

- G3 extraction: 3.464 s on rank 0 and about 1.67 s on the other source Generator ranks;
- G3 redistribution: 4.030 s (rank 0), 2.581 s (rank 1), 6.463 s (rank 8), and 7.786 s
  (rank 9);
- obsolete source release: about 3.6245 GiB/rank and roughly 0.8--1.2 s in prior snapshots.

Retained Score ranks 10--15 perform neither G3 extraction nor Generator materialization,
yet still end at about 47.825 s. Therefore the critical path is mostly common process-group
and target-runtime setup, not the 818-tensor G3 copy alone.

The source currently creates one Gloo control group, then serially creates one Generator
NCCL group, one Critic group, 12 pair groups, and 16 singleton groups. It inserts a global
Gloo barrier before creation and after each of the 30 NCCL groups: approximately **31
serialized global barriers plus 30 NCCL group creations** inside the timed transition.
This serialization was a correctness repair for a prior group-creation race, but it is now
the first transition optimization target.

The second clear target is `_broadcast_generator_g3`: it performs **818 individual
broadcast collectives** for 1,251,442,688 BF16 elements (about 2.331 GiB). A bounded
64--128 MiB bucketed broadcast would reduce this to roughly 19--38 collectives without
changing the logical G3 tensor inventory.

The existing result lacks separate timestamps for global-world reinitialization, group
creation, target-model H2D, Critic DeepSpeed initialization, logical Adam restore, and the
final barrier. Their individual costs are therefore **Not Measured**.

### Transition optimization opportunities, ranked

1. **Add low-overhead subphase timestamps first.** Measure `G3 gather`, source release,
   old-world destroy, global-world init, group creation, target H2D, G3 broadcast, DP12
   engine initialization, logical optimizer restore, and final barrier. This is the next
   falsifiable experiment and changes no schedule.
2. **Collapse the communicator set.** Keep the Gloo lifecycle group and DP12 Critic group;
   gate whether pair P2P can safely use the global communicator with deterministic peers,
   and whether the DeepSpeed MPU adapter can represent model-parallel size 1 without 16
   singleton NCCL groups. Validate this first with the existing small distributed gate.
   Do not merely delete the barriers while retaining the old race.
3. **Bucket G3 redistribution.** Broadcast fixed-order contiguous chunks, then scatter
   views into the 818 named trainables. Preserve per-tensor inventory/SHA and use bounded
   staging; this addresses the measured 2.6--7.8 s portion.
4. **Pre-stage only CPU logical optimizer buffers.** If subphase timing shows DP12 engine
   restore is material, load/validate the immutable C10 moment slices before the timed
   transition, but attach them only after the legal DP12 group exists. Do not create a
   hidden checkpoint round trip.
5. **Treat placement continuity as a separate candidate.** An asymmetric target placement
   could reduce physical role switches, but it changes node placement and transport and
   must not be credited as an optimization of the frozen per-node 2G+6C candidate.

## 3. Quantitative ceiling and bounded next experiment

The evidence rules out a return to the old 48.55-second Step10-A anchor: strict-v4 performs
more legal backward work. It also rules out a single “pinned memory fixes everything”
story. The profiled Step10-A exposes both large ZeRO-3 all-gather time and pageable copy
time, with overlap between them.

A defensible engineering target is to test whether strict-v4 Step10-A can move from about
123 s into the **85--105 s** range and whether the transition can move from about 50.7 s
into the **10--25 s** range. These are hypotheses, not results. A matched Full5 performance
claim remains blocked until exact correctness and repeated low-overhead timing pass.

Preregistered order:

1. transition timing-only instrumentation and small process-group gate;
2. transition communicator collapse, then bucketed G3 broadcast;
3. Step10-A 2M persistence candidate;
4. Step10-A ownership census for the large pageable copies;
5. only then a bounded pinned synchronous candidate and, if legal lead exists, async
   prefetch;
6. exact strict-v4 correctness before any repeated Full5 timing.

## Evidence

- `artifacts/manifests/iteration286_real14b_attempt003_full5_correctness_pass.json`
- `artifacts/manifests/iteration287_step10a_low_overhead_attempt001_result.json`
- `artifacts/runs/rb16_step10a_strictv4_nsys_20260810_001/summary/step10a_nsys_phase_summary.json`
- `artifacts/runs/rb16_step10a_strictv4_nsys_20260810_001/summary/step10a_copy_attribution_rank0.json`
- `artifacts/runs/rb16_step10a_strictv4_nsys_20260810_001/summary/step10a_copy_memkind_rank0.json`
- `scripts/distillschedule/probe_routeb_full5_2g6c_continuous.py`
- `runtime/deepspeed_backend.py`
