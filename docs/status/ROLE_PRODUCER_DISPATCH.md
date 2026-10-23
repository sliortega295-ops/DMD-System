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

## Dispatch safety and a non-preset scheduling example

A further synthetic test assigns different activation sizes to logical calls.
The flexible graph finds a feasible hybrid forward/backward order whose modeled
makespan is below every feasible fixed-window preset; this proves a decision
beyond selecting a preset, not a real-GPU acceleration claim. The independent
native non-preset test checks that such orders are actually lowerable.

Real two-process negative tests provide different individually valid orders on
the two ranks, mutate one rank's order after admission, and provide an unsafe
order on both ranks. Every case rejects before the model constructor is called,
and both ranks reach a subsequent test barrier. Receipt validation additionally
rejects a missing rank and a worker that omitted the selected dispatch. The
expanded dispatch suite passed 13 tests locally; it overlaps earlier counts.

A prediction may choose the same producer sequence as the window-one anchor,
especially under tight capacity. Merely admitting a larger window is not proof
that the executed sequence is novel. Read the recorded producer_dispatch and
native events, not only the legacy was_full_cycle_measured layout flag, when
classifying a plan as a held-out execution order. No claim requires every
selected order to differ from an anchor or from an existing manual optimum.
