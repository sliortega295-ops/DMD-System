# Component-derived producer dispatch on the existing role runtime

The components cost mode no longer has to lower every selected layout to the
same fixed-window fill/flush loop. For each admitted physical pool and maximum
live window, the existing frontier solver may interleave the monotonically
ordered Generator forwards and FIFO backwards. Releasing saved activations
before the nominal window is full can make an otherwise infeasible schedule
fit. The graph contains producer slot-release edges, Score receiver order,
explicit activation/VJP transfer resources, and measured saved-state lifetimes.

The selected graph order is projected into `producer_dispatch`, one explicit
forward/backward sequence per G rank. The EXISTING RolePoolExecutor executes
that sequence and the selected-run validator compares its actual events with
it. It does not silently replace the order with `generator_order(window)`.
All ranks agree on the dispatch identity before model work; mutation, missing
work, wrong owners, reordered forwards/backwards or early commits are rejected.
The public entry remains `workloads native program role-auto --cost-mode
components`; workload factories supply mathematics and data, never this order.

## Evidence

A synthetic capacity test shows a fixed-window graph is infeasible while the
frontier-chosen earlier-backward order is feasible. These costs are explicitly
fixtures, not measured speedups. A two-process CPU/Gloo test executes an order
which is different from EVERY fixed FIFO window, then compares both models and
Adam moments against an independently computed global-batch reference. Actual
three-/four-process calibration and selected execution also assert that the
runtime receives and follows the selected producer dispatch, with no extra
profiling enabled for the confirmation run. The final local combined component,
asymmetric, dispatch and legacy transport set passed 44 tests with no skips.

## Boundaries

This lowering admits replicated, resident pools only. It does not derive a new
FSDP collective schedule or an offload/prefetch schedule from replica costs.
The existing measured-cycle FSDP/offload capabilities remain unchanged. Forward
order, backward accumulation order, sample count, update boundaries, dtype and
source/hardware identities are preserved. Score uses its existing FIFO loop;
point-to-point messages are not remotely dispatched by a centralized executor.

Only the per-rank producer order is executed, not the predicted timestamps.
Isolated link costs and synchronized compute anchors do not model arbitrary
NCCL/NIC contention. Timing error is reported separately from state correctness;
this is not a guarantee that GPU timing matches the model. A100/H100 CUDA/NCCL,
full original video models, shared-state/sharded composition, production
checkpoint/restart and independent end-to-end speedups remain NOT_RUN or outside
this slice. Passing CPU tests does not close the complete MLSys evidence gap.
