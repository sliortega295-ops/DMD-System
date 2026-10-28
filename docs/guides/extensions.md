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
no-grad with declared modules in evaluation mode. Its `decode(value,
parameters)` returns explicit real media: `image` HWC uint8, `video` THWC uint8,
`audio` with finite TC samples/native sample rate, or both AV. The requested
portable `--output-kind` must be actually present. Explicit named native
latents need their exact `latent_schema`; they are never relabeled as media.
See [output contracts](training.md#generate-and-export) for serialization,
legacy checkpoint evaluation and finite JSON `--parameters` behavior. Quality
validation remains NOT_RUN; the framework does not synthesize a renderer.

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

### Explicit model-owned forward methods

Ordinary `ReplicaSemantics` still means calling `module(...)`. Arbitrary
`forward_interface="private_methods"` bindings remain post-backward-only;
they cannot acquire FSDP hooks by adding a method name to metadata.

For a **direct, public model-owned method**, use the existing Context ABI with
an explicit binding:

```python
from distillschedule.open_workload.replica_contract import (
    ReplicaSemantics, module_method_binding,
)

bindings["predict"] = module_method_binding("student", "rollout")
semantics = ReplicaSemantics(
    module_bindings={"predict": "student"},
    backward_calls=("student_backward",),
    local_batch_size=2,
    forward_interface="registered_methods",
    forward_methods={"predict": "rollout"},
)
```

For the corresponding Program `Call`, `inputs` become positional method
arguments **in the declared input order**, and `attributes` become keyword
arguments. The binding resolves `ctx.state["student"].rollout` on every
invocation; it does not capture the original module or replace the algorithm.
Declare all trainable roots and separate backward/update calls as usual.
Root/name mismatches, missing or noncallable methods, private names, inherited
module-management APIs and arbitrary wrapper functions reject.

Only **one outer entry method per root** is admitted. Several bindings may use
that same method; FSDP registers it once. If `rollout` calls `denoise` internally,
only `rollout` is registered, so the inner method cannot reshard the root early.
Do not re-enter the same root through `module(...)` inside that registered
method. Calls still obey the existing forward/backward completion constraints.
Persistent cycles retain the installed entry method and optimizer ownership.

The uniform FSDP2/HSDP2 installer uses the official Torch 2.10
[`register_fsdp_forward_method`](https://docs.pytorch.org/docs/2.10/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.register_fsdp_forward_method)
after `fully_shard`. This does **not** make the method a DDP `module.__call__`:
DDP-overlap and per-module mixed layouts remain excluded for this interface.
Existing post-backward reference execution remains available. CPU2/Gloo tests
exercise real method forward/backward, Adam migration, persistent updates and
fresh-process local-shard checkpoint restore; CUDA/HSDP qualification is
separate. This backend prerequisite does not admit the existing flow-model
factories, LTX-22B, full-size construction, or a new precision/algorithm policy.

### Explicit parameter-shadow layout ownership

An EMA or other full named-parameter shadow must opt in **before** its state is
shared with a workload or session. Keep the algorithm's existing values and
math; declare only which registered root owns their layout:

```python
from dataclasses import replace
from distillschedule.training.tensor_state import ParameterShadows

state["ema"] = ParameterShadows("student", existing_ema)
semantics = replace(semantics, parameter_shadows={"ema": "student"})
```

The map must contain every named parameter, including frozen parameters, with
matching shape, dtype, device and stride; values must be finite, detached and
storage-disjoint from parameters, optimizer slots, feeds and other shadows.
Plain dictionaries are not implicitly converted. Native whole-root FSDP
installation redistributes the declared map **in place** to each parameter's
mesh/placements, preserving the map shared by the component, workload and
persistent session. Undeclared maps do not acquire a layout by name. Partial
subroot migration and parameter aliases are rejected. No decay, update clock,
clipping, optimizer, scheduler or frozen-parameter coverage is changed.
Native cycle admission also retains the installed map and every owner Parameter
identity in both session and workload views, including frozen parameters and
mixed layouts. Same-layout object replacements reject before execution; copied
trials and in-place checkpoint copies remain supported.

The existing checkpoint publisher and local-shard codec store the explicit
`distillschedule.parameter_shadows.v1` owner envelope. Same-topology restore
validates owner, state kind, full keys, shape/dtype, placement and finite local
shards before any rank mutates live state. Default dense EMA/ResidentEMA
classes and serialization remain unchanged; an empty declaration is omitted
from replica identities. This is a bounded backend prerequisite, **not** public
flow/LTX FSDP admission: frozen Teacher placement, full-size construction and
CUDA/HSDP validation remain separate.

### Explicit frozen forward-root ownership

Add `frozen_module_bindings={"teacher_forward": "teacher"}` to registered-method
semantics and include that binding in `forward_methods`. The Teacher must be a
nonempty all-frozen root declared in `immutable_modules`; its calls use the same
direct `module_method_binding` as trainable roles. No direct call may write the
root or bypass its declared entry. The existing trainable binding map and exact
optimizer ownership stay trainable-only. No Teacher optimizer or EMA is created.

Uniform FSDP2/HSDP2 installation includes this explicit root in forward context
ordering, parameter bytes and the final reshard boundary. Each gradient-bearing
entry needs a reachable declared backward completion, even when only the input
has a gradient. Input VJPs retain their local scaling. A no-grad Teacher call
stays no-grad; freezing parameters never selects the call's autograd mode.
Undeclared frozen roots stay resident, including post-backward reference
execution. DDP and mixed per-root layout admission are unchanged.

The CPU/Gloo2 fixture covers separate frozen roots, input-only VJPs, common
G1/F5 no-grad Teacher calls and same-topology local-shard checkpoint continuation.
Full model construction, block-level materialization, bounded integrity/export,
and native generation through a registered outer entry still require their own
integration before public flow-model FSDP admission. CUDA and HSDP topology
validation remain separate from this CPU proof.

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

## Projection of an owned training model for inference

An atomic native training owner need not expose optimizers or Teacher/Critic
models to inference. Optional `ModelComponents.inference_projection` receives
`(restored_components, effective_training_parameters)` and returns a validated
`InferenceProjection(components, factory, parameters)`. It is an export-only
view of current owned modules, not another model initialization or update.
Do not mutate the training owner, merge/round LoRA weights, replace current G
with EMA or capture a mutable owner in a callback. The common exporter retains
its safe named-module schema and strict source, metadata, layout, finite-value
and file-hash binding; training identity is separate provenance.

The inference-only factory constructs the declared original architecture and
strictly restores the **entire** exported state before forward. Missing keys
must not fall back to initialized/base/Hub weights. Do not bind metadata to live
random initial parameter values or old payload locations. Native nonpersistent
buffers must still be initialized by the model's actual constructor.

An optional read-only `encode_input` capability permits the
[common export-only preparer](export_inputs.md#adapter-extension-contract) to
encode new prompts/raw media from those restored codecs. The model owns its
record, shape, conditioning and source contracts. Missing capability refuses;
no training-bundle fallback or generic family-name branch is introduced.

`generate --output-kind latents` is an explicit opt-in when no real decoder
exists. Return named nonempty finite floating tensors and `latent_schema`
with each tensor's exact `layout`, `shape`, and Torch `dtype`. The default
`video` path remains decoded THWC uint8 media and refuses a fake decoder.
A latent diagnostic is not a generated video or a quality metric. See the
[actual original current-G example](../../recipes/helios_gan/PORTABLE_INFERENCE_CPU.md).


## Installed model and workflow plugins

A trusted independently installed package can register factories without
modifying the framework or general solver:

```toml
[project.entry-points."distillschedule.models"]
my-model = "my_package.adapter:build_model"
[project.entry-points."distillschedule.algorithms"]
my-workflow = "my_package.workflow:build_algorithm"
```

Use `model.name: my-model` / `algorithm.name: my-workflow` in the same training
configuration, or keep an explicit `factory`. Discovery reads distribution
metadata only; it never calls `EntryPoint.load()` or imports the advertised
module. Builtin names, duplicate names and non-plain installed references are
rejected. Effective prediction/capabilities are admitted from the constructed
`ModelComponents` and bound to checkpoint identity, not trusted from catalog
advertising. Plugins remain trusted Python, not sandboxed code; pin their
helper/runtime/assets in model metadata.

To reuse the explicit generic flow-DMD algorithm rather than write another
loss, implement the role method contract in
[physical capabilities](optimization_capabilities.md#model-adaptation-not-optimization-script-copying).
LTX demonstrates an actual different AV architecture using this same algorithm,
while a rank-two non-LTX model tests the absence of a fixed layout assumption.
Official DMD2 and Self-Forcing preserve their own mathematics instead.
