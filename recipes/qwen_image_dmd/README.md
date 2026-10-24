# Original Qwen-Image T2I: native single-device flow-DMD slice

Fixed source candidate:
[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6),
revision `75e0b4be04f60ec59a75f475837eced720f823b6`, source-audited Apache-2.0.
This does not admit Qwen-Image Edit, later image generations or their conditions.

The adapter executes actual Diffusers 0.37.1 `QwenImageTransformer2DModel`,
`AutoencoderKLQwenImage`, `QwenImagePipeline.encode_prompt`, Transformers5
`Qwen2_5_VLForConditionalGeneration`, and `Qwen2TokenizerFast`. The actual native
prompt template and **crop34** are retained, not replaced by a short synthetic
embedding. The tiny fixture includes a real narrow VL class and explicit test
vocabulary. Actual empty-prompt features and masks are encoded and hash-bound.

The first recipe disables `zero_cond_t`, `use_additional_t_cond`,
`use_layer3d_rope` and `guidance_embeds`. Image metadata is natively nested as
`[[(1,h/2,w/2)]]` per batch item. Sigma is normalized; native output is the same
`v=epsilon-x0` convention used by common flow DMD. The pipeline's true CFG
includes norm rescaling, **unlike** common linear CFG, so **teacher_cfg=fake_cfg=1
is mandatory**; it is not claimed that the common loss reproduces native CFG.

Raw VAE input is BC1HW; posterior mode is normalized by the actual per-channel
mean/std, then natively packed in 2x2 patches. Decode performs the inverse and
uses the actual VAE. Variable native prompt lengths are fixed-right-padded to
the explicit cohort width with the native mask. No native tokens are silently
truncated; tests compare padded vs original native forward and execute three
different prompt lengths through persistent common training.

Transformers5's original-format text checkpoint key changes use its **actual
WeightRenaming/PrefixChange operations**, not guessed aliases; every selected
safe weight key/shape/dtype remains exact. Unknown converters/config fields,
reference-image/edit, causal/KV, full-parameter, AMP and distributed requests
fail closed. Explicit single-device FP32 CUDA reference training is qualified
on actual tiny classes, including complex RoPE ownership. The exact direct
[immutable-sharing policy](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing)
uses the same persistent owner; CUDA trial planning and other optimization
policies still reject unbounded role cloning.

Follow the [shared executable recipe](../flow_image/README.md) with family
`qwen-image`. Source/actual tiny CPU and A100 reference behavior is not full pretrained V3–V5
qualification; those gates remain **NOT_RUN**, including quality/convergence.
