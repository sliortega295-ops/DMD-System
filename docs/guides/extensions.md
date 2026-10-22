# Add a model, algorithm or execution capability

Start with [the runnable external model template](../../examples/framework/newmodel.py)
and [its tutorial](../../examples/framework/README.md). It is a new architecture
outside W1-W6 and runs without a new solver branch or workload-name condition.

Local `adapter.py:callable` plugins compile the source bytes bound by their
receipt, rather than reading or writing Python bytecode caches. A failed local
import removes its own partial module and preserves any prior module binding.
Package `module:callable` references retain normal Python import semantics.
Adapters are trusted Python code, not sandboxed input; pin their dependencies
and do not load an untrusted encoder, model or workflow plugin.

## Model and media adapter

A configured model factory receives a parameter mapping and returns
`distillschedule.training.ModelComponents`. Its named `state` contains actual
role modules, optimizers and any explicit scheduler/EMA/history state. A frozen
Teacher is constructed and frozen by this factory. The framework never guesses
which parameters to train or converts LoRA/base compute dtype implicitly.

`metadata` identifies the model/checkpoint/precision and any fixture/quality
boundary. `generate`, `decode`, `export`, `load_export` and optional `validate_batch`
callbacks remain model-owned. Keep
immutable submodules explicitly declared; aliasing is an admitted execution
capability, not a factory side effect.

An atomic native operation may return a structured receipt rather than a scalar
Tensor. Its optional `collect_metrics(outputs)` callback exposes the original
losses to the common logger without changing that operation's result or update.
Return a dictionary of nonempty metric names to finite **Python floats**; do not
recompute losses, mutate outputs/state, consume RNG or read GPU tensor values.
Tensor coercion, nonfinite values and collisions with ordinary scalar metrics
are rejected. Without the callback, existing scalar-Tensor logging is unchanged.

Inference does not construct an algorithm or restore optimizer/RNG state. The
original factory receives the effective non-device parameters recorded at
export, plus the current declared device. Its source, metadata and named module
class/state layouts must match the export. Put required non-module inference
state in deterministic factory parameters/metadata, or use a paired custom
serializer/loader; do not hide a dependency on optimizer history or the original
training dataset. Factory code/dependency assets are not bundled with weights.
An explicitly relocated local `.py` factory must keep identical source bytes
and the same callable symbol. Package/module references stay exact, and full
training-checkpoint identities are never relaxed for relocation.

By default, full named module state dictionaries use the safe built-in loader.
An optional `export(state, output_directory) -> JSON_metadata` writes only
relative artifact files beneath the supplied **staging** directory and must
have a paired `load_export(state, artifact_directory, JSON_metadata)` callback.
The latter restores that exact format after file hashes/path bounds and model
identity/layout are admitted. Never record the staging directory's absolute
path in portable metadata. `export.json` and `_SUCCESS` are framework-owned
reserved names. Missing explicit custom loaders reject; custom distributed
export callbacks are currently unsupported. The built-in distributed exporter
gathers full tensors on all owning ranks and writes shared files on primary.

`generate(state, feeds, parameters)` receives the new prepared sample under
no-grad with every declared module in evaluation mode. Its actual
`decode(value, parameters)` for portable video generation returns nonempty THWC
uint8 RGB/RGBA `video` (or generate may return that explicit decoded contract
itself, as a mapping containing `video`). Legacy checkpoint `evaluate` also
supports non-video safe tensor outputs without manufacturing frames; its receipt
then reports `video.status=NOT_AVAILABLE`. If a `video` key is supplied, its
shape/dtype are mandatory. The framework saves diagnostic pixels without latent
resizing or inferred decoding, and keeps quality validation **NOT_RUN**.
Callback parameters are finite canonical JSON; distributed evaluation admits
their common digest before any forward. Receipts preserve this actual parameter
map and the runtime `trainer.seed`, reseeded immediately before generation,
separately from any model-factory initialization seed.

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

### A different stateful workflow, not just a different architecture

[`newworkflow.py`](../../examples/framework/newworkflow.py) is an external plugin
using the same tiny genuine Student/Fake losses but a different explicit
workflow. Its three forwards read a previous-cycle feature tensor from declared
`history`. After all Student1/FakeN updates, a no-grad Program `Call` commits
Student parameter EMA, next-cycle history and both exponential schedulers.
The schedulers' learning-rate changes also declare optimizer writes. All EMA,
history and scheduler objects live in `components.state`; no model or optimizer
is captured in a closure. Persistent owner and version checks are the existing
Program checks, with no generic model/workflow-name branch or second solver.
The template explicitly admits CPU/single-process execution only; rank-local
history is not asserted to have replicated semantics.

Its fixture selects disjoint records through existing raw manifests/preparation.
Its held-out `student_teacher_flow_mse` measures deterministic t=0 flow agreement
with the **frozen random Teacher**, not pretrained media quality or a pass gate.
`validate_batch(state, feeds, parameters)` must be an inspectable function with
three positional arguments, return named finite scalar metrics, and never mutate
declared persistent state. It must use the supplied state/feeds, not captured
models/tensors/optimizers. The framework restores modes and RNG on both success
and failure; state mutation rejects without rolling the model/optimizer back.
Metrics and checkpoint identity are independent of generation callbacks. See
[periodic validation](training.md#optional-periodic-held-out-validation) for the
explicit config, stream, artifact and current local-only boundaries.

Portable inference deliberately uses the current Student with **zero history**,
not the training EMA or restored previous-cycle history. That deployment choice
matches the held-out metric convention and is recorded in model metadata. The
default exporter saves named modules only, not arbitrary EMA/history trees. If
your deployment uses those trees, put needed inference state in a frozen named
module or provide paired explicit export/load callbacks; never silently generate
from freshly initialized, unexported EMA/history. Full training checkpoints do
retain all of the template's declared state and exact continuation semantics.

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
