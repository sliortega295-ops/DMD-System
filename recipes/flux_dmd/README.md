# FLUX.2 klein Base 4B: native single-device flow-DMD slice

Fixed source candidate:
[black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d),
revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`, source-audited Apache-2.0.
No source/weight grant is inferred for a different checkpoint or component.

The adapter executes actual Diffusers 0.37.1 `Flux2Transformer2DModel`,
`AutoencoderKLFlux2`, `Flux2KleinPipeline.encode_prompt`, and Transformers5
`Qwen3ForCausalLM` with `Qwen2TokenizerFast`. It retains hidden-state layers
**9/18/27 concatenated**, even in the tiny fixture's 28-layer narrow encoder.
Klein Base has `guidance_embeds=False`; no guidance tensor is passed. Native
forward accepts normalized sigma and internally multiplies it by 1000.

Native VAE latent 2x2 patchification precedes the VAE’s actual stored running
mean/variance BN normalization, then BNC packing. Decode performs exact BN
inverse, native unpatchification and the actual VAE. Image IDs are native
`(0,h,w,0)` and text IDs `(0,0,0,l)`; uniform batch geometry is validated because
the native Transformer uses the first batch's IDs. No FLUX.1, distilled klein,
reference-image/edit, embedded-guidance, causal/KV, full-parameter, AMP or
distributed substitution is admitted. Explicit single-device FP32 CUDA
reference training is qualified on actual tiny classes; CUDA trial planning
and optimization policies remain rejected pending bounded snapshot support.

Follow the [shared executable recipe](../flow_image/README.md) with family
`flux2-klein-base`; no alternate trainer is involved. Source and actual tiny
CPU and A100 reference G1/F5/lifecycle/codec behavior are separate from full pretrained V3–V5,
which remain **NOT_RUN**. The decoded random-fixture images establish no quality.
