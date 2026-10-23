# Wan2.1 / LiveWan CPU reference: origins and changes

This is a maintained source-only numerical closure, not either full application,
not pristine attention source, not pretrained model weights, and not a trainer.

- Wan2.1: https://github.com/Wan-Video/Wan2.1 at
  9737cba9c1c3c4d04b33fcad41c111989865d315. Complete wan/modules/model.py is
  byte-for-byte original, including constructors, normalization, RoPE, head and
  forward. Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
- LiveWan: https://github.com/JonathanColetti/LiveWan at
  1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5. Whole wanstreamer numerical modules
  are retained; only blockcausal.py/core.py absolute wan imports are statically
  relocated to this private package. Their numerical definition ASTs are intact.
- Attention is a PROMINENTLY DECLARED STATIC SELECTED-DEFINITION DERIVATIVE of
  LiveWan wan21_patches/modules/attention.py: original _needs_mask and
  sdpa_attention definitions are verbatim. Imports/backend context are maintained
  CPU bridges, not original optional CUDA/FlashAttention selection. No runtime
  AST extraction, native namespace aliases or CUDA facades are used.
- New package initializers, source-only loader, source/license inventory, and
  attention.py bridge enforce explicit backend=torch-sdpa-cpu, precision=float32.
  The bridge preserves original padding, bottom-right causal, GQA and scale math
  with explicit dtype=float32 and CPU MATH SDPA. Dropout, window/version overrides,
  outer CPU autocast, non-FP32/non-CPU inputs and over-budget attention fail.

Both original Apache-2.0 licenses are retained separately. Neither source tree
had an upstream NOTICE file in the audited inventory; this origin/change record
is a project modification notice, NOT an invented upstream NOTICE. The manifest
records original and shipped hashes and original selected/full-definition ASTs.
Packaged license copies are identical to the top-level retained license files.
No model-weight or dataset license is granted by these source licenses.

Raw original constructors/methods retain native defaults and trusted-pickle
TeacherLatents loading. Consumers must explicitly use CPU FP32, T2V, channels16,
patch(1,2,2), text4096/text_len512, compatible RoPE/head geometry, cuda_graphs=False
and generate_block, never the CUDA-synchronized stream convenience. Trusted data
validation must precede any later dataset read. No model/download is done by load.

Qualification is tiny deterministic non-pretrained CPU fixture forward/backward
only, not native368x640/81-frame or pretrained1.3B/14B, BF16/FlashAttention/GPU,
world16, Program/optimizer/checkpoint restart, C51, VAE decode or quality evidence.
