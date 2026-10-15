# Asymmetric component-cost planning on PR #34

## What changed

The existing `role-auto --cost-mode components` path can now search unequal
Generator/Score rank counts, not just a fixed balanced split. It still uses the
same RoleTask/RoleFunctions, shared CostModel/frontier solver and native FIFO
executor. No model-specific launch path or new optimization kernel was added.

Every chosen physical pool has its OWN window-one reference and instrumented
anchor, including its own per-direction P2P link costs. Component samples are
reused only for other FIFO windows in that exact pool, input, dtype, source and
hardware identity. We do not scale a 1G+2Score measurement to infer 2G+1Score.

The graph explicitly orders each Score receiver's inputs across DIFFERENT
producers. The return-before-next-receive edge is a conservative half-duplex
model, not a claim that isend blocks or that NCCL overlap is measured. Per-rank
producer fill/drain order remains the order executed by the existing backend.
The solver compares these executable FIFO families, not arbitrary schedules
which the runtime would ignore.

## Calibration budget

Anchor exploration honors the full native-run budget before model construction:
(samples + one unprofiled reference) per physical pool, plus one reserved
selected-plan confirmation. It samples diverse pool sizes/localities without
assuming balanced pools are fastest, then allocates predicted window targets.
Budgets omitted are explicit. Windows which induce the identical producer
execution order are not counted as new held-out schedules. This avoids filling
all candidate slots with separately measured window-one configurations.

## Evidence and boundaries

Pure tests cover receiver dependencies, acyclic native orders for unequal pools,
link endpoints, inactive-rank rejection, deterministic exploration, reserved
confirmation budget and execution-equivalent window deduplication. A synthetic
16-rank topology checks enumeration only; it is NOT a 16-GPU execution result.

Actual 3- and 4-process CPU/Gloo tests calibrate two physical pools, predict four
candidates from their window-one anchors, search using the shared service, and
run the selected candidate. Generator and Score parameters, Adam steps and both
moment tensors are compared with an independent single-process GLOBAL objective.
They additionally execute an explicitly held-out asymmetric window as a native
lowering test; that EXTRA test run is not mislabeled as an automatic winner or
included in the calibration-budget count. CPU timing is not a speedup result.

Only active replicated/resident pools are admitted to this cost projection.
FSDP/HSDP/offload remain separate measured-cycle capabilities, not extrapolated
from replicated anchors. No cross-hardware cost reuse, new-shape cost transfer,
shared-NIC contention model or arbitrary event dispatch is claimed. Full native
video DMD, A100/H100 NCCL and independent timing-prediction accuracy are Pending
cluster validation. Existing loss, batch, update count and parameter precision
are unchanged.
