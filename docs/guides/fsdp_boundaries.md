# Model-owned FSDP parameter groups

An adapter with a registered outer model method can explicitly name disjoint
child modules as native FSDP groups:

```python
ReplicaSemantics(
    module_bindings={"student_forward": "student"},
    backward_calls=("student_backward",),
    local_batch_size=1,
    forward_interface="registered_methods",
    forward_methods={"student_forward": "dmd_forward"},
    shard_boundaries={"student": ("blocks.0", "blocks.1", "blocks.2")},
)
```

The existing uniform FSDP2/HSDP installation applies `fully_shard` to these
children in the declared order, then to the remaining outer root. The root
registers its single outer method. Child groups use their ordinary `forward`
entries, so the model must call each child as a module rather than directly
reading that child's sharded parameters from an outer private method.

The declaration adds no model-specific training loop. Optimizer objects, groups
and initialized moments migrate to the replacement parameters once. Explicit
all-parameter shadow maps follow the final layout by full parameter name.
Declared immutable frozen roots can use the same grouping without an optimizer
or parameter gradients. A frozen forward can still preserve an input VJP when
its original Program declares a differentiable call.

Paths must resolve to distinct, nonempty child modules with real `forward`
implementations. Ancestor/descendant cuts and cross-group parameter/storage
aliases are rejected. An empty mapping retains existing whole-root identities.
Path order is part of the workload identity; changing a declared path, child
object or parameter after installation fails before the next cycle. Mixed
per-root layouts and role-pool sharding do not accept this declaration.

With `fsdp2_reshard`, native groups can free full parameters between child calls.
With `fsdp2_keep_forward`, visited groups can all stay materialized until their
backward or the complete-cycle boundary. Finalization explicitly reshards each
child and outer root because native `reshard()` is not recursive.

Validation uses small real Torch modules on CPU/Gloo, covering losses,
gradients, Adam moments, all-name EMA, immutable Teacher state, local input VJPs,
and fresh-process checkpoint continuation. Physical parameter-storage samples
are CPU observations. CUDA/HSDP topology, large-model construction peaks,
generation entries and export staging still need their own qualification;
this generic capability does not open an unqualified public model factory.
