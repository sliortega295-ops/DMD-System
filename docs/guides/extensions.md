# Add a model, algorithm or execution capability

Start with [the runnable external model template](../../examples/framework/newmodel.py)
and [its tutorial](../../examples/framework/README.md). It is a new architecture
outside W1-W6 and runs without a new solver branch or workload-name condition.

## Model and media adapter

A configured model factory receives a parameter mapping and returns
`distillschedule.training.ModelComponents`. Its named `state` contains actual
role modules, optimizers and any explicit scheduler/EMA/history state. A frozen
Teacher is constructed and frozen by this factory. The framework never guesses
which parameters to train or converts LoRA/base compute dtype implicitly.

`metadata` identifies the model/checkpoint/precision and any fixture/quality
boundary. `generate`, `decode` and `export` callbacks remain model-owned. Keep
immutable submodules explicitly declared; aliasing is an admitted execution
capability, not a factory side effect.

An offline encoder takes decoded media plus its parameter mapping and returns
the native tensor feeds expected by the model. Reuse that model's local
VAE/text/audio encoders and original conditioning conventions. Keep previous
and current chunk identities, masks, dimensions and sample rates explicit.
Feature import preserves shape/dtype; changing resolution means preparing new
valid model inputs, not resizing latent tensors. See [data formats](data.md).

## Algorithm adapter

The function signature is:

```python
def build_algorithm(components, feeds, parameters, cycle, seed):
    # Return an existing open_workload.torch_backend.Workload for ONE cycle.
    ...
```

Use `DMDHooks`/`build_dmd_workload` for its explicit Student1/FakeN semantics,
or `Program`/`Call` for different update/GAN/history graphs. The factory uses
the exact owned objects from `components.state`; hidden captured models,
optimizers, tensors or external autograd graphs reject. The trainer advances
cycle versions; the existing graph compiler checks all real state dependencies.

Declare `reuse_student` versus `fresh_student`, loss, RNG consumption, gradient
scaling/clipping, optimizer updates and EMA in the algorithm rather than
implicitly changing them with placement. Mutable history or EMA state must be
declared and checkpoint-serializable. Mark only pure, valid grad forwards as
replay-safe. The tiny example reuses the repository's DMD boundary functions,
but a new model need not adopt that example's algorithm.

## Physical execution capability

Extend the existing capability inventory and native execution backend, then
calibration/lowering. A capability must have a real runtime branch, explicit
model/precision/state/version/layout conditions, actual observed effects and a
reference comparison. The shared solver consumes measured candidates; adding
only a configuration flag or a predicted cost does not add backend support.

Use the existing [open Program interface](../workloads/open_workload_program.md),
[role pools](../workloads/open_role_pools.md) and
[native planning](../runbooks/native_planner_workflow.md). These paths share the
planning service but admit different physical capabilities. A new local model
does not automatically gain FSDP2, cross-pool overlap or a CUDA Graph backend.

## Minimal extension acceptance

Entrypoint files, installed framework code, and the actually resolved shipped
`runtime` compatibility package are source-bound. This is not recursive hashing
of arbitrary globals imported by a trusted external plugin. Pin and verify its
external helper/encoder/checkpoint dependencies in the adapter and return their
content identities in `ModelComponents.metadata`; those identities are bound
to training checkpoints. A plugin is trusted Python code, not a sandbox.

1. Prepare actual model-compatible inputs and record encoder/checkpoint identity.
2. Run real reference forward/backward/update; check intended trainables and
   frozen roles, optimizer ownership, loss, finite state and update versions.
3. Save/restart at a complete cycle and compare uninterrupted continuation,
   including RNG/data position and optimizer/EMA state.
4. Validate every proposed optimized capability against that reference, then
   collect costs and execute its saved plan.

CPU tiny tests prove interface/lifecycle behavior. Model quality, real large
model execution and hardware-specific correctness require their own evidence.
