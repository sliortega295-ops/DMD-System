# Eight-H100 / logical-B16 adaptation

The five-workload A100 worktrees and archived evidence are not modified. The
explicit `DISTILLSCHEDULE_HARDWARE_PROFILE=h100-8-b16` maps physical rank `r`
to original independent B1 rows `r,r+8`. Each loss is divided by two, with one
Generator commit and five Critic/EMA commits per No-GAN cycle. Default A100
behavior remains world16/B1. HSDP on this node is replicate2/shard4, not 2x8.

## Measured gates (not a model benchmark)

- Exact source integration at `70332d3883449fe2bb1ad8276528483c33d88ab0`.
- Eight-H100 tiny accumulation canary: 8/8 PASS, both real FSDP2 and Resident
  async accumulated gradients/updates match global-B16 reference. BF16 frozen
  compute and FP32 LoRA observed. No new cgroup OOM; GPU processes cleaned.
- Exact Wan source assets transferred directly between remote servers via an
  anonymous SSH pipe. Every model/LoRA payload checksum and every tensor
  descriptor in all sixteen rank-qualified traces verified; integrity PASS.
- Isolated Python overlay uses Torch 2.10.0+cu128 and exact Diffusers
  0.38.0.dev0. Existing training environment is not changed.

## Wan execution

`launch_wan13b_hardware_two_cycle_node.sh` holds project and node locks from
resource audit through teardown. It runs the existing two-cycle worker with
the explicit H100 accumulation adapter. Cycle1 is warmup/diagnostic; only
cycle2 is the steady-state performance denominator, matching the A100 study.

Run `correctness` independently for `world16-fsdp` and
`resident-async128-tree-chunk2048`; compare their fresh summaries with
`compare_wan13b_hardware_two_cycle.py`. Then run three alternating matched
unprofiled pairs, binding every timing run to the same-head correctness and
cross-arm comparison hashes. Never reuse A100 admission receipts.

The comparison preserves DEFAULT_P0_PARITY thresholds and reports all 160
rollouts plus topology-independent sampled final state. Sampled state is not
full-state proof. Numerical REVIEW permits only controlled diagnostic timing,
as in the original coarse-trajectory experiment, not strict-equivalence claims.

## Completed Wan1.3B result (2026-09-30)

The exact measured source is `11dd0b155cc72572c907a79abc1a66cbdca25392`;
subsequent LiveWan/Helios compatibility commits are not retroactively used as
this measurement identity. Both real model arms and all six unprofiled timing
runs have 8/8 physical-worker PASS covering the original 16 logical slots.

| Arm | Pair0 | Pair1 | Pair2 | Median | CV |
|---|---:|---:|---:|---:|---:|
| FSDP8 baseline | 56.797855115 | 56.788417506 | 58.960118638 | 56.797855115 | 2.175% |
| Resident async128/tree/chunk2048 | 37.582740035 | 37.509206853 | 37.511148368 | 37.511148368 | 0.112% |

Ratio of medians: **1.514159x**, controlled diagnostic only. Strict rollout
parity fails 160/160 unchanged gates; numeric status REVIEW, not proven benign
noise. Five sampled final-state components PASS; full-state parity NOT_RUN.
No new CUDA/cgroup OOM, and all run teardown receipts show no GPU process.
Profiling is NOT_RUN. The measured denominator is cycle2 after same-process
cycle1 warmup, not startup and not Nsight.

[Full report and raw summaries](../evidence/h100_five_workload_20260930/wan1p3b/report.md),
[JSON](../evidence/h100_five_workload_20260930/wan1p3b/result.json),
[raw timings](../evidence/h100_five_workload_20260930/wan1p3b/trials.csv).

## Remaining matrix

Helios No-GAN/GAN and LiveWan retain independent asset/runtime gates and are
NOT_RUN as model benchmarks. CPU reconstruction of both Helios base precast
packages is complete: all 12 full-file SHA hashes match A100, no GPU used.
LiveWan's missing checkpoint/cache/source payload transfer is in progress.
H3 is NO-GO until exact original canonical16-slot initial LoRA/RNG evidence is
available; seed equality or partial state sketches cannot substitute for it.
No speedup is inferred for the other four workloads.
