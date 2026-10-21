# Role-pool audit closure: implementation is not a GPU result

This continues PR #34 and audit comment 6168691313. No original workload loss,
precision, sample count or update contract is changed. Naive FSDP2 remains the
paper baseline. Existing role-pool cost search still uses measured whole-cycle
FIFO envelopes; it is not a new fine-grained planner or a predicted speedup.

## Addressed correctness gaps

- The offload copy stream waits for the producer before D2H. record_stream
  separately protects allocator lifetime. Host staging leases also wait for
  on-demand H2D completion on backward streams before release. CPU prefetch
  no longer makes an extra host copy outside its budget.
- Known transport bounds depend on each rank's actual assigned cohort, Score
  updates and G window. Offloading saved activations never removes the boundary
  latent/gradient buffers. Observed CUDA allocator peaks supplement known
  storage estimates in measured candidate admission. These are not guaranteed
  bounds for arbitrary unseen model activations or externally competing jobs.
- Guidance-only Score may have every parameter requires_grad=False and needs
  no optimizer. Input VJP still propagates to the ORIGINAL Generator graph.
- Candidate calibration releases the previous executor, its role groups and
  model/optimizer references before constructing the next trial.

## Original-algorithm gradient interface

RoleFunctions accepts exactly one of:

1. score_loss(model, latent_leaf, condition): scalar objective, differentiated
   by autograd.grad with respect to the transported input.
2. score_input_gradient(model, latent, condition): an original-algorithm helper
   called under no_grad, returning a detached per-microbatch latent gradient.

The second path permits a DMD adapter to compute Real/Fake scores and construct
its original target gradient WITHOUT replacing its loss with the scalar-objective
fixture. Runtime applies the common equal-microbatch normalization ONCE. Neither
interface automatically translates arbitrary DMD/GAN/EMA protocols. The current
native update family is G1 followed by 0..64 Score updates. More than one needs
an explicit `score_update_input` cycle/update/sample provider, including any
intentional reuse or transformation of the detached source cohort. Other update
orderings/protocols still need explicit semantic bindings.

## Continuing a selected plan

A live RolePoolExecutor now retains the same models, Adam moments and groups.
Each completed call to run() advances Program versions and the logical cycle.
To run another cycle, the factory must provide sample_at(cycle, index, role,
device). No implicit resampling or silent reuse of the first cohort is allowed.
Failed partial cycles are poisoned rather than retried as if no update happened.
The layout and task contract cannot be mutated on a live executor.

The public existing role-run command accepts --cycles N (default 1):

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/distillctl.py workloads native program role-run \
  --factory examples/open_workloads/unseen_score_pipeline.py:build \
  --parameters examples/open_workloads/unseen_score_pipeline.json \
  --distributed gloo --memory-limit-bytes 10000000 \
  --host-memory-limit-bytes 10000000 --plan /tmp/role-auto/plan.json \
  --cycles 3 --output /tmp/role-continuing --execute
```

First produce the matching plan with the documented role-auto command on this
same revision, hardware and parameters. This public example is a small random
CPU Conv3D test, not original Wan/H3 training. Every cycle emits an immutable
receipt; execution.json describes the final in-process state. This is NOT a
restart checkpoint. A fresh role-run starts from the factory's initial state;
it does not silently load state from the previous diagnostic confirmation.

## Evidence and unresolved scope

Tests compare model AND Adam tensors through three cycles with independent
single-process global-microbatch references, including explicit constructed
gradients, Score updates and saved-state offload. Additional tests go through
calibration, shared selection, selected-plan continuation and the actual public
torchrun CLI. Fake CUDA stream tests check host API ordering only; two real CUDA
regressions are skipped on CPU and must run on A100/H100 before GPU claims.

Implemented CPU/Gloo scope now includes bounded contiguous and balanced
per-node cuts, restricted Generator/frozen-Score/trainable-Score FSDP2 and
initialized Adam moment preservation. Trainable Score sharding requires the
constructed no-grad gradient interface, balanced paired pools and explicit
repeated-update input semantics; mixed dtypes require homogeneous leaf ownership.
See `test_open_workload_role_updates.py`, `test_open_workload_role_trainable_score.py`
and `test_open_workload_role_composition.py`. This is not new A100/H100 evidence:
those generic GPU paths remain NOT_RUN.

General pool-internal HSDP/shared-state layouts, arbitrary mixed parameter
ownership, cross-layout numerical
admission for arbitrary factories, fine-grained cost-predicted role schedules,
production checkpoint/restart, full original DMD adapters and real GPU quality /
performance evidence remain separate unsatisfied scope gates.
