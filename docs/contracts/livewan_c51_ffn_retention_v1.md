# Versioned C51 frozen FFN-down retention composition

`c52-c51-critic-ffn-down-retention-v1` authenticates the complete original C51
plan and adds only the existing C44 GPU-local frozen BF16 FFN-down result reuse.
It is neither a pure-layout candidate nor the old FA3-recompute ablation. The
canonical C51/C49/C44/recompute IDs, configs and rejection cases are preserved.
The C49 validation view is available only after whole-parent authentication;
it does not change the actual plan, runtime Generator or effective plan hash.

Original368x640x81/globalB16/continuous/G1-C5-EMA0/precision/trace/RNG/update
semantics and all kernels remain. Attention still has300 original FA3 calls
and300 replay reads per Score rank over two continuous cycles. Additional
frozen FFN-down originals and replay reads must both be300 per Score rank,
with zero failures, no CPU offload and per-checkpoint lifetime. Generator
reports NOT_APPLICABLE. Source-verified C44 policy performs the real save/read,
not a numerical metadata claim. Only full pipeline native C2 can establish a
performance effect; a local Critic benefit may be hidden by Generator work.

A same-source conditional trajectory pair and actual metadata producer/consumer
bindings are required before native timing. Existing numerical and capacity
policies are unchanged. CPU synthetic chains are not GPU qualifications.
