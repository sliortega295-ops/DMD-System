# Native W1 current-Generator output ABI v1

This is an output-only extension to the existing native W1 owner, not another
trainer, a checkpoint migration, or permission to launch GPU work. Existing
reference training, fixed-trace diagnostics, CPU-reference latent ABI v1 and
incomplete-stream resume retain their contracts.

## Public boundary

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

## Portable named modules

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

## Native forward and real decoding

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

### Fresh inference noise protocol v1 (factory ABI v2)

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

## Admission, capacity, and failure

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

## Evidence boundary

Focused tests cover actual installed small Wan2.1 VAE normalization/decoding,
local safe asset/refusal/RNG cases, native output-only terminal restore using
explicitly synthetic scalar/CUDA/world16 transport, and actual CPU Gloo/DTensor
all-owner/rank0 publication. A separate optional-stack test restores a tiny
actual original G at marker cycle100 through the public CLI/native owner,
then removes training assets and independently generates NEW-input video with
the original sampler and real VAE. It asserts unchanged helper bindings,
actual original forward counts, stable PSD draws and zero default Cholesky
sampler calls. It does not rerun a training cycle. Additional causal cases
characterize the singular covariance, exact substreams/layout/last-step
controls, RNG purity, protocol refusals and the operation-input routing seam.

CPU/source proofs, including tiny random original classes, **are not** full
1.3B pretrained GPU output, world16 qualification, training quality or speedup.
Those remain `NOT_RUN` until independently admitted on the target topology.

The inference factory binds its actual owned state into the decoder callback,
which follows the existing Program `(generated, generation_parameters)` API.
Strict restoration updates the bound G/VAE modules in place; generic Program
interfaces are unchanged. Focused coverage enters the real `_generate_values`
callback path with an actual small VAE, and independently reconstructs the
big-endian SHA256 substream seeds/draws rather than comparing a helper to itself.
