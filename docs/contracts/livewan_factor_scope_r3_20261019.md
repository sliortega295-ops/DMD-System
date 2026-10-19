# Self-Forcing factorial R3: reselected A and complete original-layout B

Opt-in only. Historical baseline, full, OOM and compatible-subset plans/receipts
are not rewritten. No default, canonical Cxx plan or main profile changes.

Work: 368×640×81, continuous sigma, actual global16, original trace/bootstrap,
seed20260830, G1/C5/EMA0, original precision/optimizer mathematics. One model
initialization, four complete cycles, C1 warm-up/observed qualification, native
C2 primary, C2–C4 mean supplemental; n=1, no profiler. Strict numerical/long
trajectory equivalence is not established by finite/census checks.

## A only: `only_A_g8_fsdp8_v1`

G ranks0–7 original FSDP8 units, true reshard; Score ranks8–15 resident Critic
and Teacher FSDP8. Keep the complete original pipeline, trace-balanced two
waves, legal depth-one lookahead, split-DMD shadow and communication schedule.
All 18 C51 B mechanisms OFF, original operators/checkpoint recomputation/RNG,
GPU foreach AdamW, no optimizer/parameter/activation offload or weight retention.

Each G owner handles two samples; selected-block surrogate has weight0.5/3.
FSDP SUM/8 on each backward gives the global16 sample mean. No second manual
gradient all-reduce; DTensor global clipping and optimizer references preserved.
Variable-length real rollouts need existing max-noop collective padding on
each wave, including exactly-once lookahead. Dummy calls use no KV, no gradient,
fork/restore RNG, discarded outputs; their cost is included, counted separately
from unchanged application work. This implementation is a bounded feasible A
candidate, not proof of a global optimum over every possible layout.

## B only: `only_B_full_fsdp16_v1`

Keep original world16 FSDP groups/units, sequential order, B1 sample per rank.
Carry all 18 actual active C51 mechanisms, not the earlier four-operator subset.
Use explicit Student/Critic/Teacher role dispatch and capture that role inside
the function truly saved by checkpoint. Keep original RNG; only Critic uses the
unchanged FA3 selective-checkpoint policy. Count actual original/replay entries.

G text K/V is built lazily after the owning CausalBlock FSDP pre-hook has
gathered its parameters. Own detached results, never parameter views. For one
no-grad `prepare_rollout`, change only the lifetime of the same gathered units
and autocast cache; first original forward gathers lazily. Clear K/V, clear
autocast, reshard all35 units and restore true reshard before any G commit.
Retain prompt/parameter/logical-version checks and original grad path. No
merged weights, quantization, stale reuse or full-model shadow is permitted.

Original source-bound G/Score operators/backends are composed under roles.
Source guards stay intact; installer artifacts and actual kernel counters are
recorded. Fused G AdamW operates on the original registered DTensor parameters.
Receipts must show all requested mechanisms actually executed. CPU stand-ins
prove plumbing/lifecycle, not CUDA/FSDP validity or numerical equivalence.

The four-cycle native runtime/collector checks all ranks, versions, calls,
finite state, resource/OOM deltas, exact source/trace/plan identities and the
new mechanism receipts. Failures remain failures; no fallback silently drops B.
