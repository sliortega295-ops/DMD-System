# Original Qwen-Image T2I: flow DMD

Model source:
[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6),
revision `75e0b4be04f60ec59a75f475837eced720f823b6`, Apache-2.0.

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
safe weight key/shape/dtype remains exact.
`compute_precision: "bf16-mixed"` selects native autocast while
preserving FP32 master/update state and complex rotary buffers; FP32 is the default.
The direct
[immutable-sharing policy](../../docs/guides/optimization_capabilities.md#direct-single-cuda-immutable-sharing)
uses the same persistent owner.
Explicit equal-B1 post-backward replication on CPU/Gloo or CUDA/NCCL reuses
the shared lifecycle and can retain exact frozen-state sharing. It does not
wrap private role methods in DDP or shard model memory. Follow the
[replica contract](../flow_image/README.md#explicit-replicated-training-not-model-sharding)
for rank-specific data/noise and complete same-topology checkpoint recovery.

Follow the [shared executable recipe](../flow_image/README.md) with family
`qwen-image`.

## Opt-in frozen-Teacher shard-first construction

`model.parameters.construction: "cpu-fsdp2-frozen-teacher-v1"` reuses the
existing FLUX construction provider and local safetensor-slice loader. It
requires CPU/Gloo, rank-strided local-B1 data and explicit manual
`torch-callable-replicated-v1` / `fsdp2_reshard` execution: `bucket_cap_mb: 0`, local
`torch-callable-resident-v1`, `share_immutable: false`, and empty
`checkpoint_bindings`. Student/Fake dense initialization and LoRA draw order
remain unchanged. Omitting `construction` retains the default dense path.

Only the native frozen Teacher parameters are constructed on meta and loaded
directly into local DTensor shards. Qwen's plain complex `pos_freqs` and
`neg_freqs` are rebuilt by the actual CPU `QwenEmbedRope` constructor, with
source/code/content hashes in the construction receipt. Unknown or class-owned
constants, changed owners and unsupported layouts still reject. The common
G1/F5 loss, masks, RNG, Adam, scheduler and full-parameter EMA are unchanged.

Native two-rank CPU tests cover loss, pre-clip gradients, updates, exact
fresh-process recovery and current-Student PNG output from a new prompt.
Training assets are not required by the portable export.
