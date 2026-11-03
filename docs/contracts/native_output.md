# Native output and publication contracts

Output restores the existing native owner, then publishes current **unmerged**
Student weights and the required codec. It is not training resume, EMA selection,
a new trainer or permission to allocate GPUs. Source/CPU protocol evidence does
not establish full-size pretrained GPU capacity, quality or performance. Use the
[output commands](../guides/training.md#generate-and-export) and each recipe.

<a id="WAN_NATIVE_OUTPUT_V1"></a>
## Native W1 current-Generator output ABI v1

This is an output-only extension to the existing native W1 owner, not another
trainer, a checkpoint migration, or permission to launch GPU work. Existing
reference training, fixed-trace diagnostics, CPU-reference latent ABI v1 and
incomplete-stream resume retain their contracts.

### Public boundary

- `distillctl export` accepts W1 `original_wan_cohort_stream.v1` training
  configurations supplemented with the explicit `trainer.export` policy below.
  Enter on **every original world16 rank**, using the original topology/assets,
  selected strategy or plan, source, bootstrap and stream. It does not launch
  ranks, SSH, download assets, or select a different strategy.
- The existing worker reads the complete checkpoint header, validates the exact
  policy/identity/runtime/topology, constructs its original G/C/EMA owners and
  restores their native trainables/Adam/scheduler/EMA/RNG state. A terminal
  `cycle == cursor == configured horizon` is valid for **export only**. The
  ordinary resume path still requires an incomplete horizon.
- Output branches before cohort binding, graph preparation/capture, losses,
  optimizer/scheduler/EMA updates, final-state evidence writers, or timers.
  No input cursor or update clock advances. Current unmerged G is selected;
  bootstrap and EMA are never substituted or merged.

### Portable named modules

`distillschedule.wan_native_generator_inference.v2` uses the unchanged shared
`distillschedule.inference_export.v1` publisher. Every owner participates in
full-tensor materialization; only rank0 publishes `weights.pt`, `export.json`
and `_SUCCESS`. There is no custom distributed exporter or new weights format.

Named modules are `generator` (a pure `CurrentGenerator` view, prefix `model.`)
and `codec` (actual `diffusers.AutoencoderKLWan`). The stable Generator container
separates serialization layout from native/FSDP runtime class namespaces. It
does not evaluate, freeze, copy, merge, switch adapters, or otherwise mutate
the training G. BF16 bases, original FP32 exceptions and **FP32 unmerged LoRA**
are preserved exactly. C, Adam and EMA are restored by the checkpoint owner,
but are not copied into the inference artifact.

The companion `<output>.inference.json` is an ordinary standalone **Program**
configuration. Generation remains single-process and strictly loads both
named modules. It has no old training config, base checkpoint, codec directory,
optimizer, Teacher/Critic or data dependency. Local codec paths appear only in
the export request, never as required portable parameters. Provenance is
descriptive, not an instruction to re-open an old asset.

### Native forward and real decoding

- NEW offline preencoded features contain exactly BF16 `prompt_embeds` of the
  declared `[1, sequence, 4096]` shape. This path does not run T5 on a training
  owner or reuse a captured training prompt/history/trace.
- The actual original full Generator class, full LoRA/pyramid targets
  (`patch_short/mid/long`, excluding `down/up`), original scheduler and original
  `run_generator` three-stage/flow rollout are used. An explicit independent
  seed and the versioned fresh operation-input protocol below isolate noise.
  It is not the older zero-noise CPU diagnostic. No training RNG is consumed.
- Wan2.1 decoding is non-residual RGB3, z16, spatial8, temporal4. Full local
  config and safe FP32 shards are admitted before native G construction.
  Production architecture is base_dim96, `[1,2,4,4]`, two residual blocks;
  scaled random native-class fixtures are explicitly marked non-pretrained.
  Wan2.2 residual/patch2/in12 configurations are refused.
- The actual VAE decodes `normalized_latents * config.latents_std +
  config.latents_mean`, exactly once. **No training latent clamp** is applied.
  Native decoded RGB output is converted to THWC uint8. Resolution, frames,
  fps, codec and source/runtime/helper byte identities are explicit. The shared
  writer emits PNG frames plus `generation.pt/json` and a completion marker.
  This is decoded video/frame output, not an MP4 encoder or quality claim.

#### Fresh inference noise protocol v1 (factory ABI v2)

`distillschedule.wan_fresh_inference_noise.v1` is explicit in exported model
parameters and strict factory admission. It replaces the failed unreleased
native-output factory v1; it does **not** change legacy CPU-reference v1,
training defaults, or any frozen helper. Factory v1 parameters are refused.

The unchanged native `sample_block_noise` builds covariance
`(4/3) I4 - (1/3) ones(4,4)` at gamma=1/3. Its eigenvalues are
`[0, 4/3, 4/3, 4/3]`: valid singular PSD, not Cholesky-positive-definite.
Target Torch2.10 rejected it; local Torch2.14 can round the zero eigenvalue
slightly positive. This is not repaired by changing gamma, jittering the
diagonal, converting to iid noise, or silently enabling a global environment
compatibility flag.

For each NEW generation, the existing `wan_capture.input_stream` derives four
independent labelled CPU streams: SHA256 of compact JSON
`[seed, noise_schema_version, tensor_name]`, first eight bytes big-endian,
modulo `2**63`. Draws occur in this declared order:

1. `entry_noise`: full BCTHW `torch.randn`, BF16.
2. `rollout_noise`: a separate full BCTHW BF16 draw.
3. `block_noise/1`: existing `runtime.official_baseline.stable_block_noise`,
   FP32 four-vector iid draws minus their per-block mean, times `sqrt(4/3)`.
4. `block_noise/2`: the same exact PSD law on a separate labelled stream.

Blocks use the unchanged native 2x2 view/permute/interleave. The native
consumer alone casts FP32 block noise to its current latent dtype/device.
CPU sampling is independent of requested inference device and preserves the
caller's CPU/Python RNG; the enclosing native call also preserves its current
device RNG. This is a **new RNG protocol**, not old Cholesky draw ordering or
bitwise identity, nor resumed training RNG. The Gaussian covariance law is
preserved; generator/scheduler/flow math and shapes are unchanged.

Ephemeral fresh inputs implement only the original `operation_trace.tensor`
and `operation` seam; they contain no old trace file, history, training prompt
or cursor. The native call reads exactly those four tensors and one original
three-stage control sequence. Each stage exits at its configured last step
(`horizon - 1`), matching `last_step_only=True`; no exit draw or collective is
needed. Unexpected, repeated or unconsumed input/control access fails closed.
No original provider function/binding is monkeypatched.

### Admission, capacity, and failure

Export policy schema is `distillschedule.wan_native_output_request.v1`. It
requires seed, prompt_shape, fps, evidence kind, pinned local VAE config/shards,
truthful origin/license and explicit capacity budgets. Meta inventory bounds
the original architecture/adapters/codec before physical inference construction.
Budgets include FP32 construction, strict-load copies, nonpersistent buffers,
dense-attention/decoder activation reserves, full-tensor CPU staging **per rank
and summed over all owners**. This is a conservative capacity admission, not
a benchmark or free-memory reservation; OOM must fail, never change geometry.

Missing/incomplete/corrupt checkpoints, mismatched source/runtime/topology,
codec file hashes, missing/full state layouts, precision, adapter targets,
nonfinite weights, budgets and existing output paths fail closed. Shared
distributed publication failures preserve only explicitly unpublished staging
for audit; no `_SUCCESS` or companion config is fabricated. A published export
remains a valid weights artifact if later companion-config publication fails;
the command returns failure rather than claiming the full workflow completed.

Existing checkpoint source hashes are not rewritten to make an older run
match new code. Cross-source or cross-topology training resume is not added.
Any legitimately reviewed source overlay remains a separate integration gate.

The inference factory binds its owned Generator/VAE modules into the existing
Program `(generated, generation_parameters)` decoder callback. Strict restore
updates those modules in place; the generic Program API is unchanged.

<a id="HELIOS_NATIVE_OUTPUT_V1"></a>
## Helios No-GAN native Student output

The `original_helios_cohort_stream.v1` reference and SafeBundle owners support
`distillctl export` from their own complete same-source, same-topology native
checkpoint. The output supplement is
`distillschedule.helios_native_output_request.v1`; the portable model ABI is
`distillschedule.helios_native_generator_inference.v1`.

### Restore and publication

The original worker owns initialization, source/assets, policy, topology and
complete G/C/optimizer/scheduler/EMA/RNG restoration. Output branches after that
restore and before any cohort load, preparation, capture or update. A terminal
complete checkpoint is accepted only for export. Ordinary training restart
still requires an incomplete-stream boundary. The original G/C/EMA clocks and
live parameter identities, trainability and module modes are not changed.

The selected Student is **current full unmerged G**, not EMA and not a rebuilt
base. The export also includes the actual pinned FP32 Wan2.1 VAE. No merge,
latent clamp, synthetic conditioning or discarded training call is introduced.
Missing/changed sources, incomplete checkpoints, incompatible precision and
merged adapters fail explicitly. The frozen two-cycle benchmark path and GAN
variant are not widened by this interface.

The existing named-module publisher uses explicit primary-only CPU staging:
every owner participates in its full-tensor collectives, but only the primary
retains full CPU weights. Node-local codec construction, the primary's full
state, per-owner transient materialization and host/device reserve are bounded.
Physical/cgroup headroom is checked before codec allocation and again before
export; these observations do not reserve memory against unrelated processes.
The caller still owns the original native launcher and resources.

Publication retains the existing weights/manifest/completion-marker format.
The companion `.inference.json` config is published without overwriting another
file. If that later publication fails, the command fails and no config is
fabricated; the already committed weights artifact remains valid.

### Independent generation

The emitted Program config constructs only G and the codec, strictly reloads
all exported state, and consumes **new** BF16 `prompt_embeds` of the declared
shape. It requires neither the training checkpoint, old trace, optimizer nor
original base/codec files. Training resume and EMA remain separate APIs.

The actual original three-stage sampler uses the explicit labelled exact-PSD
noise law already shared with native Wan output. This is fresh inference RNG,
not old Cholesky draw-order replay or training RNG continuation. Original
forward/scheduler helpers are not patched. Generator dtype remains BF16 base
with FP32 unmerged LoRA. The real VAE applies latent std/mean exactly once and
decodes without latent clamping. Output is a decoded PNG frame sequence plus
the normal generation tensors/manifest, not an MP4 or a quality score.

The W2 request stays `helios14b`, No-GAN, 384×640, its declared 17/33 frames and
supported 1/2-step pyramid horizons. A standalone CPU or single CUDA device
needs its own full-model/activation/attention capacity; it is not the sharded
training layout. No resolution, frame, dtype or precision fallback is implicit.

Source/CPU acceptance covers the real owner/checkpoint protocol and a tiny
original G/native sampler/actual VAE in a fresh process. Pretrained native
world16 output and device capacity are separate GPU acceptance items.

<a id="HELIOS_GAN_NATIVE_OUTPUT_V1"></a>
## Helios GAN native Student output

The full fresh-stream W3 owner supports `distillctl export` from its own
complete, same-source, same-topology checkpoint. The request is
`distillschedule.helios_gan_native_output_request.v1`; the portable model is
`distillschedule.helios_gan_native_generator_inference.v1`. W2 and W1 portable
identities are not accepted as W3.

### Restore, then export

The actual original GAN worker restores G, Critic including all GAN heads,
Adam, scheduler, EMA, RNG and native version state before publication. At cycle
100 its native versions are G102/C510/EMA900; curriculum versions are
G300/C1500/EMA900. These namespaces are not interchangeable. A terminal complete
checkpoint is valid for export only; ordinary training still requires an
incomplete-stream boundary.

Only the current full **unmerged G** is selected, not EMA or a base-only
reconstruction. N70, merged shadows and inference-weight merging remain off.
The export adds the actual locally pinned FP32 Wan2.1 VAE. Live training
parameter identities, modes and gradient flags are preserved. No cohort is
consumed, capture started or optimizer/EMA updated. The original installed
runtime verifies this zero-cycle path and restores its hooks and policy state;
it does not manufacture a completed training cycle to satisfy a finish gate.

The implementation reuses the W2 codec, primary-only CPU staging, capacity
admission and atomic artifact publication. Every owner participates in native
full-tensor collectives. Only the primary retains complete CPU G state. Codec
construction, per-owner transient materialization, primary state and explicit
host/device reserves are admitted before allocation/publication. Physical and
cgroup checks observe current headroom, not a reservation. The existing native
launcher and distributed ownership remain the caller's responsibility.

### Independent new-prompt generation

The companion inference config constructs only the current G and codec. Strict
weight loading needs no old checkpoint, bootstrap, original base, old VAE path,
optimizer or GAN worker. Supply new BF16 `prompt_embeds` with the declared shape.
Generation uses the original three-stage sampler and labelled exact-PSD fresh
inference noise; original helpers are not patched and training RNG is not
continued. G retains BF16 base/FP32 LoRA, without merging.

The W3 task stays GAN, 384×640×33, with its original 2/2/2 rollout horizons.
A full unsharded G, codec and working set must fit the chosen standalone CPU or
single CUDA device; sharded training fit is not standalone inference fit. The
real VAE applies latent std/mean once, does not clamp latents, and produces PNG
frames plus normal generation tensors/metadata. Container encoding and quality
assessment are separate consumer steps.

Partial checkpoints, changed source/runtime/geometry, mixed model identities,
merged G, missing codec state and unsupported capacity fail explicitly. A later
companion-config publication error leaves any already committed valid weights
artifact intact but returns an error, with no fabricated config. This path has
source/CPU coverage; pretrained native world16 and GPU output remain separate
acceptance items.

<a id="PRIMARY_EXPORT_STAGING_V1"></a>
## Primary-only inference export staging

`training.inference.publish_export(..., primary_only=True)` is an internal,
explicit output-transport option for a native owner. Existing callers default
to the original all-owner CPU staging. It does not change training, the
`distillschedule.inference_export.v1` format, or the inference loader.

All ranks agree on the option and module layouts before materialization. Every
owner calls the same `DTensor.full_tensor` collectives, visiting named modules
and tensor keys in deterministic order. Only rank zero converts and retains
the dense tensors as CPU export state. Other owners release each transient
dense result without making a CPU state copy. A primary CPU-copy failure is
exchanged before any peer starts the next tensor collective. Only rank zero
writes weights and publishes the existing manifest and completion marker.

The primary still needs enough host memory for the complete exported state.
Every owner still needs capacity for its transient full tensor; no collective
is skipped and no bound on model-sized CUDA peak is implied. The calling owner
must separately admit host/cgroup and device capacity. This option is not a
capacity-policy override and does not enable any new model/backend combination.
Custom adapter exporters own their own staging and cannot request this option.

Tensor values, dtypes, layouts, nested safe extra state, live parameter identity,
trainability, module mode and RNG are preserved. Unsupported objects, divergent
rank options/layouts, corrupt exports and publication failures still fail.
Restore remains inference-only, not a training checkpoint resume.

The source/CPU proof uses small synthetic modules with actual two-process
Gloo/DTensor collectives. Pretrained 14B export, CUDA capacity and full native
world16 output require their separate integration acceptance.

<a id="W1_W2_NATIVE_STREAM_HORIZON_V1"></a>
## W1/W2 bounded native fresh-stream horizon v1

This source/CPU contract extends the existing finite fresh-cohort interfaces
from 64 to **100 complete cycles**, superseding only the former 64-cohort
ceiling (including `helios_nogan_safe_stream_v1.md`), not the immutable fixed-trace experiment.
It does not establish 100-cycle pretrained GPU execution, numerical parity,
convergence, decoded output or quality. Existing research-only bundle status
and hardware/dependency admission remain unchanged.

- W1 reference/manual/selected fresh stream and explicit incumbent stream v2:
  2..100 cohorts, indexed 0..99. Every cohort retains its task/trace/source
  identity and per-rank feature contract. A single seed is valid with 100
  distinct cohort identities; repeating a fixed trace is not training.
- W2 original reference and SafeBundle fresh streams: 2..100 cohorts, retaining
  the original warm G2/C10/EMA10 bootstrap, No-GAN33/222/world16/B1 and G1/C5/EMA5.
  All 80 features per cohort remain globally path/hash distinct. CPU1 reference
  owners additionally enforce their existing `max_train_steps` capacity; a
  requested 100-cycle stream needs capacity at least 100 (source clock 510).
  Current-G CPU latent-only export accepts those completed boundaries without
  widening its precision, budget, topology, model or output scope.
- W1 source steps are `first_step + 5*i .. first_step + 5*i + 4` for the
  incumbent; reference tasks retain their actual fake-update count. W2 source
  steps at indices 63/64/99 are 325..329 / 330..334 / 505..509. Original
  constant LR is not a finite scheduler horizon. Native beta curriculum with
  `dynamic_step=500` clamps progress at one; neither source steps nor EMA
  clocks restart at that point. Resolution-dependent native shift and the
  original per-stage budgets are unchanged.
- W1 complete versions are bootstrap G/C/EMA plus actual updates and absolute
  EMA cadence. From cold G0/C0/EMA0, incumbent cycle100 is G100/C500/EMA500.
  W2 cycle100 is G102/C510/EMA510. A checkpoint at completed cycle50 resumes
  inputs/cycles 51..100, not 1..50 again. Exact source/assets/stream/topology
  identities and persistent-state/RNG checks remain mandatory.
- Graphs remain transient. First-process-cycle capture is bounded to the
  original 12 rollout / 3 critic signature capacities. A later uncaptured
  signature still fails; 100 admitted cohorts do not promise that arbitrary
  fresh inputs fit that capture inventory. No shape, stage budget, fake-update
  count, RNG draw or safety gate may be changed to force graph admission.
- **Public manual fixed trace and calibration remain exactly two cycles**.
  Existing internal/selected-plan repeated-fixed-trace diagnostics retain
  their old 2..64 bound and explicit diagnostic classification. Neither can
  gain a 65..100 fixed-trace repetition bypass from this extension. Integers
  above100, booleans and fractional fresh-stream horizons are rejected.

The stream schema names stay unchanged: source/content fingerprints bind this
explicit contract extension to the current implementation. Frozen source and
helper pins, source-import receipts, old benchmark/calibration contracts and
training math are not rewritten. Changed code inventories require new inputs
and normal source-bound admission; old source receipts are historical evidence,
not silently current training identities.
