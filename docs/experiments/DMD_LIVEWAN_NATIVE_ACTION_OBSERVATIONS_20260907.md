# LiveWan native action and transport evidence

## Scope: completed missing observation unit, not a new speed result

Run: `jointp_livewan_actions_20260907_r1-native-observation`.
Executed source: `78e501bc1727013d94e356f697051b4d11cba4be`.
The source contains the default-off observer and its typed `action-observation`
entry point. This report's publication commit is not the execution commit.

The completed Wan component search/held-out run and LiveWan macro search were
**not repeated**. This unit supplies the previously missing native helper and
transport boundaries; it does not split old coupled-commit times into invented
independent G/Score/link costs. No new baseline, native timing, Nsight, H3 run,
environment installation, or canonical profile change occurred in this unit.

## Native verification

- Two owned nodes, 8 A100-SXM4-40GB each, driver 535.161.08.
- Reused Python 3.10.20 / Torch 2.10.0+cu128 / CUDA 12.8 / NCCL 2.27.5;
  exact source, overlay and asset paths remain in the saved execution plan.
- C34 persistent G0–7 / Score8–15, depth-1 lookahead, logical B16,
  existing physical Critic B2, two continuous cycles, seed 20260830.
- Existing trajectory probes and all native contracts remain enabled.
  Trajectory wall time is **not** a native cycle-2 performance measurement.
- **16/16 ranks PASS**, 32 rank-cycle rows, final **G2/C10/EMA0**.
- **160 producer/consumer logical sample identities** reconcile with their
  actual native operation keys, seeds and trace-balanced lane mapping.
  The 32 Generator backward samples match their corresponding rollout rows.
- **336 core helper observations**: 160 rollouts, 32 G-side Fake velocity calls,
  32 Teacher velocity calls, 32 Generator sample backwards and 80 Critic B2
  backwards. These helpers are not individual DiT-call counts; the native
  operation/call census remains authoritative.
- No extra observer synchronization, collective or tensor-value read. CUDA
  copy events run on the actual issuing stream; Gloo observations run on the
  actual sender thread. No cross-device clock alignment is assumed.

The initial two-node, two-GPU no-model transport canary also passed at
`0b1b62d8825484d3153ce758bf33d6126856fee8`. It exercised the native 17-message
key protocol and C5 shadow transfer with exactly checked small tensor values.
It is not model capacity, parity, throughput, or tensor-SHA evidence.

## What was newly separated

Teacher helper compute is now observed **inside**, separately from its enclosing
send/submit range. Critic backward includes the native asynchronous gradient
reducer; its subsequent commit joins any remainder. Adding a separate full
reduction duration would double-count work.

For the host runahead path, every G rank staged **17 × 4,945,920 B = 84,080,640 B**.
Across the eight paired lanes: **136 D2H copies, 672,645,120 B**, and the same
**136 H2D copies, 672,645,120 B**. Byte/event residuals are zero. This is the
intentional latent transport only, not a census of all framework memcpy.
Wait/send/shadow rows repeat payload metadata; summing their `bytes` would
incorrectly count the same data several times.

| Observed operation | Events, all owning ranks / 2 cycles | Median per-rank sum (s) | Clock / interpretation |
|---|---:|---:|---|
| Latent D2H | 136 | 0.005839 | Actual dedicated-copy-stream envelope |
| Latent H2D | 136 | 0.008639 | Actual consumer-stream envelope |
| Sender copy-ready wait | 136 | 0.978906 | Sender-thread host wait, including production dependency |
| Gloo send completion | 136 | 0.955311 | Host/runtime/network completion, not pure isolated link service |
| Gloo receive wait | 136 | 3.611894 | Consumer wait; must not be treated as network service cost |
| Streamer reuse copy wait | 104 | 0.000237 | Producer-thread wait for safe buffer reuse |
| Shadow source pack | 8 | 0.003432 | CUDA contextual envelope |
| Shadow receive/load wait | 8 | 0.009408 | CUDA contextual envelope including native dependency |
| Shadow target load | 8 | 0.001389 | CUDA contextual envelope |

One completed C5 shadow publication per paired lane was verified against its
native layout, dtype, element count and version receipt. This new real run does
not independently hash the shadow tensor values; the no-model canary checked
values and the unchanged native trajectory retains its existing state probes.

### Helper duration observations, cycle 2 only

These are median **per-owner-rank sums of contextual CUDA helper envelopes**,
not exclusive wall-time partitions and not formal performance denominators.
G and Score execute concurrently. Do not add the rows into a cycle latency.

| Family | Median owning-rank sum (s) |
|---|---:|
| G rollouts | 26.638758 |
| G-side frozen Fake velocity | 1.841219 |
| G sample backwards | 3.556738 |
| Score-pool Teacher velocity | 17.274697 |
| Score-pool Critic B2 backwards | 27.700785 |

## Memory and cleanup

Maximum allocated: **37,743,737,856 B (35.15160 GiB)**.
Maximum reserved: **40,883,978,240 B (38.07617 GiB)**.
Sampled node cgroup peaks: **80,381,321,216 B** and **479,524,003,840 B**.
Minimum sampled driver free: **461,373,440 B** and **8,388,608 B**.
These samples demonstrate **no new CUDA/cgroup OOM**, not robust headroom or
per-action workspace. In particular, no larger retained-output schedule is
admitted from this evidence alone.

Both native launchers exited zero after their native GPU-process check.
Independent post-run readback verified no GPU applications/owned workers,
free node locks, no listener on port 29938, and zero OOM/oom-kill deltas.

An analysis-only failure is preserved as `analysis-r1-failure.json`: the first
analyzer incorrectly required timing-only `teardown_status.txt` from a
trajectory run. The native trajectory intentionally does not emit that file.
Analysis r2 verifies the actual source-bound launcher exits, before/after
cgroup receipts, and independent cleanup readbacks. **No GPU rerun or synthetic
teardown receipt was used to fix the analyzer.**

## Planner boundary and remaining work

Status: **PASS_NATIVE_ACTION_OBSERVATIONS**.
Independent service-cost admission: **NOT_READY**.
New generated distributed action DAG execution: **NOT_RUN**.
New performance/speedup: **NOT_RUN**. Numerical comparison remains previous
**REVIEW**, not reasserted strict equivalence and not a new matched comparison.

The remaining LiveWan units are actual workspace/output lifetimes, scoped
service-cost admission, and native lowering plus held-out feedback for generated
pool actions. Gloo receive waits are not interchangeable with transfer latency;
teacher envelopes with sends and coupled C-commit macros remain inadmissible as
independent costs. The old, completed macro search remains valid within its
original restricted scope.

H3 operation-owned RNG/action binding and explicit diagnostic admission, and
the two Helios generated adapters, also remain unfinished. This is not a
five-workload completion claim.

## Evidence

Small source/summary bundle:
`artifacts/experiments/joint_planner_livewan_actions_20260907/`.
CPU focused gate: **187 passed**. Regression-red and analyzer-failure logs are
preserved. Reproduce bundle integrity with its `verify_bundle.py`; it does not
start GPU work. Archived deployment commands contain immutable historical IDs
and are records, **not commands to rerun unchanged**.

Persistent detailed controller evidence:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/livewan_components_r1/`.

Raw native rank/trajectory/observation receipts and derived analysis:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/livewan_components_r1/`.
Raw tensors, checkpoints and environments remain remote. The repo bundle
contains original source receipt paths and hashes, not their large payloads.
