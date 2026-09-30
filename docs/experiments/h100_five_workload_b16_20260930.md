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

Model workload capacity, correctness comparison, timing, and profiling remain
NOT_RUN until their actual receipts exist. Other workloads have independent
asset/runtime gates; no speedup is inferred from the tiny accumulation canary.
