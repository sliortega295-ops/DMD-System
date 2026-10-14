# Repeated updates and trainable Score pool sharding

This continuation extends the existing PR #34 role backend. It does not add a
model-specific launcher, change an original loss, or declare GPU validation.

## Training inputs versus execution choices

`RoleFunctions.score_update_input(ScoreUpdateContext)` describes the original
algorithm's input for one Score update and logical microbatch. The context
contains cycle, update index, logical microbatch, retained latent, conditions,
and device. It returns a detached `(latent, conditions)` on the assigned device.
This callback is required for more than one Score update; one-update legacy
bindings still consume exactly the old input. Reusing or perturbing a cohort is
only the callback's explicit algorithm, not an invented fresh Generator rollout.
An algorithm needing new post-commit Generator samples still needs its actual
native data/rollout binding. Those samples are NOT generated automatically here.

Score parameters commit after each full globally normalized update; all G VJPs
have completed before this phase. Model instances, optimizers, and versions
persist across cycles. The source identity includes the new callback and helper.

## Automatically matched trainable Score sharding

The existing `role-auto`/calibration path includes `fsdp2_train` when the task has
Score updates and its original binding supplies `score_input_gradient` under
no-grad. The scalar score-loss/autograd.grad alternative is not converted to
that interface and is not admitted for trainable Score FSDP2 in this slice.
Frozen Score remains a separate, previously implemented capability.

Each native Score subgroup uses FSDP2, retaining the supplied optimizer and
redistributing existing moments through the shared migration helper. BF16
frozen base plus FP32 trainable parameters stay in their original dtypes when
parameter ownership can be represented by separate homogeneous leaves. No
silent casting, optimizer reset, or post-FSDP duplicate all-reduce is used.
Score loss normalization cancels FSDP2's mean exactly once, preserving the
global-microbatch objective. Native admission is checked collectively.

Collective-safe FIFO restrictions remain: equal G/Score pool sizes, at least
two Score shards, and equal per-shard calls. G FSDP2 additionally requires all
local forwards to fit its resident in-flight window. Invalid layouts are
excluded, not relabeled as replication. Pool sharding still uses conservative
dense-model capacity admission until real large-model peaks/initialization are
validated. The implementation does not promise that every model fits on A100.

## Evidence

Local Torch 2.10 CPU/Gloo tests: 5 new trainable-Score tests passed; 27 related
role/shard/continuous tests passed. The tests use 2/3/4 actual processes and
independent global-batch references, not injected collective mocks. Repeated
update fixtures cover 1/3/5 commits. Four-rank tests cover Score-only and dual
G+Score FSDP2, existing nonzero Adam moments, mixed BF16/FP32 owners, and two
persistent cycles. A separate calibration -> common search -> selected native
execution test verifies candidate admission and continued updates without
asserting a predetermined CPU winner. Exact remote CI must be checked on the
published commit; these counts are not A100/H100 validation.

The original five canonical training protocols, generated-video quality,
CUDA/NCCL correctness/performance, production checkpoint/restart, and general
shared-backbone plus arbitrary layout combinations are not established here.
The role planner still measures whole-cycle FIFO candidates rather than
predicting arbitrary unmeasured per-call schedules; this work does not claim
that the fine-grained ReaL-style scheduling gap has been closed.
