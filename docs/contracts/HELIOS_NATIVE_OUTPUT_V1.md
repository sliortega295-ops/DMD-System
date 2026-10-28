# Helios No-GAN native Student output

The `original_helios_cohort_stream.v1` reference and SafeBundle owners support
`distillctl export` from their own complete same-source, same-topology native
checkpoint. The output supplement is
`distillschedule.helios_native_output_request.v1`; the portable model ABI is
`distillschedule.helios_native_generator_inference.v1`.

## Restore and publication

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

## Independent generation

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
