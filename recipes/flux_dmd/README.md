# FLUX.2 klein Base 4B: flow DMD

Model source:
[black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d),
revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`, Apache-2.0.

The adapter executes actual Diffusers 0.37.1 `Flux2Transformer2DModel`,
`AutoencoderKLFlux2`, `Flux2KleinPipeline.encode_prompt`, and Transformers5
`Qwen3ForCausalLM` with `Qwen2TokenizerFast`. It retains hidden-state layers
**9/18/27 concatenated**.
Klein Base has `guidance_embeds=False`; no guidance tensor is passed. Native
forward accepts normalized sigma and internally multiplies it by 1000.

Native VAE latent 2x2 patchification precedes the VAE’s actual stored running
mean/variance BN normalization, then BNC packing. Decode performs exact BN
inverse, native unpatchification and the actual VAE. Image IDs are native
`(0,h,w,0)` and text IDs `(0,0,0,l)`; uniform batch geometry is validated because
the native Transformer uses the first batch's IDs. Use
`compute_precision: "bf16-mixed"` for native autocast with FP32 master/update
state; the default remains FP32. The direct
[immutable-sharing policy](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing)
uses the same persistent owner.
The shared recipe also exposes explicit equal-B1 post-backward replication on
CPU/Gloo or CUDA/NCCL, optionally with the same exact frozen-state sharing.
This is not DDP-forward wrapping or memory sharding; the
[replica contract](../flow_image/README.md#explicit-replicated-training-not-model-sharding)
defines rank-specific sampling, RNG and same-topology resume.

Follow the [shared executable recipe](../flow_image/README.md) with family
`flux2-klein-base`.
