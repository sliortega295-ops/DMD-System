# LiveWan cross-scale persistent-pool optimization draft

## Task contract

- Objective: retain the exact LiveWan Wan2.1-14B Teacher to Wan2.1-1.3B
  Student SF-DMD workload while making a persistent mixed GPU pool at least
  1.30x faster than the single world16 FSDP2 sequential denominator.
- Frozen workload: source/bootstrap/trace identity, world16, B1/global-B16,
  368x640 and latent `B1x16x21x46x80`, four rollout evaluations, continuous
  shifted sigma, Student full training, Critic LoRA32, Teacher14B, G1/C5/EMA0,
  optimizer order, application call inventory, and output contract.
- Allowed changes: persistent role placement, legal operation ordering and
  overlap, queueing, transport, exact batching that preserves all 16 logical
  samples, communication timing, and mathematically exact kernels.
- Forbidden changes: model/loss/trace/RNG/update/batch/precision changes,
  dummy samples or model calls, reduced calls, parameter/activation offload,
  and using the published 8-H200/G1C1 result as the denominator.
- Correctness command: paired current-head correctness runs followed by
  `compare_livewan_crossscale_correctness.py`.
- Evaluation command: one unprofiled baseline and candidate timing run unlocked
  by a PASS paired correctness receipt.
- Promotion: 16/16 PASS, G1/C5/EMA0, exact call inventory, zero CUDA/cgroup OOM,
  admissible numerical parity, and `baseline_seconds / candidate_seconds >= 1.30`.

## Current evidence at HEAD 850ca299

- Sequential correctness: PASS, 16/16, 80.920269 s diagnostic path, peak
  allocated 21.904601 GiB/rank.
- Persistent 8G+8Score correctness: PASS, 16/16, 99.975609 s diagnostic path,
  peak allocated 33.858184 GiB/rank.
- Logical inventory and source/bootstrap/trace identity match exactly.
- Paired comparator: FAIL. Step-0 rollout tensors before either update are
  bitwise equal, but 63/64 post-update rollout tensors fail the strict P0 gate;
  worst normalized L2 is 0.200330 and minimum cosine is 0.979907.
- The comparator also has a proven placement-accounting defect: its FSDP
  Generator fingerprint is reduced 16 times after the full parameters are
  retained, so it reports exactly 16x the true numel and aggregate sums. After
  dividing those duplicated aggregates by 16, baseline and candidate final
  Generator sums agree within 6.7e-9 relative, but this aggregate evidence is
  too weak to waive rollout parity.
- The first-Adam-step path is numerically sensitive: baseline reduces one
  sample/rank through world16 FSDP reduce-scatter, while the candidate locally
  accumulates two samples/rank then uses an 8-rank all-reduce. The global mean
  is algebraically identical, but the floating-point reduction tree differs.
  This is a diagnosis target, not permission to relax the gate.

## Ranked next candidates

1. **C0 correctness evidence repair**: make final parameter fingerprints
   placement invariant and add correctness-only pre-clip gradient/global-norm
   receipts. This changes no timed operation and determines whether drift begins
   in backward reduction, clipping, or Adam.
2. **C1 score-priority ordering**: in step 0, compute and return both DMD
   gradients before Critic backward while keeping Critic loss0 then loss1 and a
   single Critic commit. This exposes both Generator backwards to the Score
   pool's Critic work without changing model versions or loss order.
3. **C2 measured pool rebalance**: only after a valid phase profile, test a
   score-heavy pool. A non-divisible pool must use heterogeneous real B1/B2
   local batches in the same FSDP call count; dummy samples/calls are forbidden.
   Capacity and exact loss normalization must pass before timing.
4. **C3 exact hot kernels/transport**: profile the promoted schedule with
   node-outer Nsight, then use kernel work only for a demonstrated hotspot.

## Evidence required

- Per role and step: compute, NCCL, P2P, H2D/D2H, host-gap, and exposed overlap.
- Per correctness run: gradient fingerprint before clipping, returned global
  norm, final parameter fingerprint, optimizer step identity, and rollout
  parity by logical step.
- Every rejected candidate remains append-only in the research log and
  candidate ledger; profiled wall time is diagnostic only.
