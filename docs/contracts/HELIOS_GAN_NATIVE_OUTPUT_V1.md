# Helios GAN native Student output

The full fresh-stream W3 owner supports `distillctl export` from its own
complete, same-source, same-topology checkpoint. The request is
`distillschedule.helios_gan_native_output_request.v1`; the portable model is
`distillschedule.helios_gan_native_generator_inference.v1`. W2 and W1 portable
identities are not accepted as W3.

## Restore, then export

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

## Independent new-prompt generation

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
