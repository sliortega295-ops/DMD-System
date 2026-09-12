# LiveWan native observations → source-bound pool DAG

**Completed unit: CPU structure and native-evidence reconciliation.** No new
LiveWan GPU run, calibration, numerical comparison, timing or speedup is claimed.
The completed C34 observations were reused rather than repeating the matrix.

- Observed native source: `78e501bc1727013d94e356f697051b4d11cba4be`.
- Observed run: `jointp_livewan_actions_20260907_r1-native-observation`.
- Analysis source SHA256: `52b4a0e284e505ee87370dc0183769007d11aef1339682a2771cc5bfdcf484da`.
- Analysis run: `joint_planner_20260907/livewan_dag_r1`.
- Small evidence: `artifacts/experiments/joint_planner_livewan_pool_dag_20260907/`.

## What this adds beyond the already completed macro search

`search/livewan_pool_contract.py` converts the **actual 16-rank observations**
into the existing `ExecutionGraph`: 180 nodes, 397 dependencies, and all 160
rank-qualified sample/seed bindings over two cycles. A pool node represents
eight native rank invocations; it does not turn eight B1 calls into physical B8.
Generator remains B1; the existing Critic path remains physical B2/global B16.

The native order is not the Wan/H3 prepared-cycle template:

1. Step0 Critic consumes latents produced **before** that cycle's G commit.
   Steps1–4 consume post-commit G latents. A detached latent's provenance is not
   a live read of the current Generator parameters.
2. Fake Score in the G pool reads its separate frozen Critic shadow. The one
   inter-cycle transfer publishes **C5**, not five separate shadow updates.
3. Depth-1 lookahead permits next-cycle wave0 before C5 finishes, but not wave1
   before the shadow load. Neither step0 G commit nor C commit is artificially
   serialized behind the other.
4. Latent transfers, Teacher-velocity returns, shadow transfer, stream reuse,
   sender FIFO, async send completion, and final drain are explicit. The graph
   distinguishes semantic dependencies, native transport safety, the frozen
   source order, and the existing depth-1 policy.

Endpoint completion order is **not inferred** from a shared message identity.
Logical delivery is a predecessor of the two endpoint return/completion events;
neither host return is falsely declared to precede the other. These graph tokens
are metadata, not newly inserted CUDA/global barriers.

## Transport reconciliation

Counts below are **per paired lane over both cycles**, with eight lanes. They
are payload metadata, not aggregate memcpy throughput or exposed wall time.

| Existing transport | Messages/lane | Payload bytes/lane |
|---|---:|---:|
| NCCL rollout latent | 3 | 14,837,760 |
| Pinned-host/Gloo rollout latent | 17 | 84,080,640 |
| NCCL Teacher velocity | 4 | 19,783,680 |
| Paired Critic LoRA shadow | 1 | 127,795,200 |

The prior observed copy census remains authoritative: **136 D2H and 136 H2D**
latent copies, **672,645,120 B in each direction**, residual zero. Teacher
velocity is not mislabeled as an already combined DMD gradient.

## Deliberately unresolved memory and execution

There are 78 storage descriptors. Only the two pool-specific persistent pinned
buffer inventories have measured payload allocation (17 buffers per endpoint,
84,080,640 B per owning rank). **76 remain unmeasured**, not zero.

Prepared objects also retain context/replay/KV aliases, not just `x0`. The outer
`precomputed_first` handle survives through the next role call and its endpoint/
anchor bookkeeping; async send buffers survive their completion. A named Python
reference's last use is not proof that the aliased CUDA storage was released.
B2 assembly/workspace, model/optimizer state and reducer storage are unresolved.

Consequently this module **does not build a `PlanningProblem`**, admit service
costs, prove memory feasibility, or lower an alternative distributed schedule.
Missing edges are not permission to reorder the old worker. Those are the next
unfinished units; contextual helper durations must not become independent costs
by subtracting overlapping waits or treating unknown memory as zero.

## Verification and failure preservation

- Focused integration: **96 passed** (`tests-integration-r8.txt`).
- Original 16-rank observations independently revalidated remotely, no Torch
  import or CUDA context: `PASS_REUSED_NATIVE_RECEIPTS`.
- Serialized JSON roundtrip, exact extracted-evidence reconciliation, sample/
  source/transport/version tampering, async lifetime and order rejection tested.
- Bounded read-only review completed; final extracted-evidence tampering finding
  fixed and covered by regression tests.
- Two small-artifact transfer checks failed closed: CSV newline normalization,
  then a reformatted local summary's nonidentical bytes. No native run or remote
  derivation was repeated. Exact remote bytes were subsequently verified in a
  separate directory; previous copies and failure manifests remain preserved.

Raw receipts stay under
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/livewan_components_r1/`.
New remote CPU output is in the sibling `livewan_dag_r1/analysis-r1/`.
Local evidence is under
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/livewan_dag_r1/`.

**Remaining:** storage/alias lifetime observations, context-bound service costs,
explicit async pool-action lowering and held-out native feedback. No canonical
winner or main branch change.
