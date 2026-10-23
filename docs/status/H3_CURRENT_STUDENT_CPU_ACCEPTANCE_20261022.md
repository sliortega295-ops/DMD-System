# Original H3 current-Student output lifecycle

This source-scoped update adds the output leg of the W6 **cached original CPU
reference**. It does not replace native benchmark contracts, historical results
or the [earlier framework acceptance](FRAMEWORK_ORIGINAL_CPU_ACCEPTANCE_20261022.md).

The common `export` interface now projects the actual restored full frozen base
and **unmerged current Student LoRA64**, without Fake Score, discriminator,
optimizer, EMA, history or training-input state. A separate inference-only
factory strictly restores that export and calls the original parser, schedule
and eight-step rollout on genuinely new cached prompt/history conditions.
Output is explicitly named video BCTHW and stereo audio SCT **latents**, not
encoded media. Default video generation still refuses without an actual decoder.

## Observed verification

Implementation commit: `9094e6e`. The
[machine-readable acceptance](../../artifacts/framework/20261022/h3_current_student_reviewed_cpu.json)
binds exact changed file hashes, source closure, runtime, JUnit and review.

- Actual original-source, update, data, export and common lifecycle: **232 passed,
  zero skips** on Python3.12.3 / Torch2.10.0+cpu.
- Dedicated inference and CI checks, including rejection-path RNG nonmutation:
  **72 passed, zero skips**. Required refactor regression: **2153 passed and20
  subtests**, with85 pre-existing dependency/hardware skips.
- The actual built wheel was installed offline. Three separate processes
  trained, resumed through the labeled GAN-start2 fixture, exported and ran
  eight actual original DiT forwards. Continued state/RNG/data order exactly
  matched uninterrupted training; native new-input latents matched the same-seed
  reference. Old training assets, data and checkpoints were deleted before the
  final inference; no optimizer/discriminator was constructed in that process.
- Nine independent local review lenses completed with no actionable findings.
  The external-provider attempt returned no usable review; this is not
  cross-model corroboration or a hosted-CI result.

The maintained original numerical closure is unchanged. Projection does not
change live Parameter identities, role selection, module modes, RNG or the next
actual training update. Shape/dtype/source/runtime/topology/capacity and
checkpoint-weight corruption reject before mutation or real model allocation.

## Subsequent hosted CPU portability check

At test-only compatibility commit `40f5f6b`, the actual hosted original-H3
workflow completed **148 tests without skips on both Python3.10 and3.12**, with
minimum Torch2.10 CPU. Its required-case consumer also passed. The
[separate receipt](../../artifacts/framework/20261022/h3_current_student_hosted_cpu.json)
binds the run/jobs and raw-log hashes. The original manifest and all numerical
source bytes are unchanged: only the test canonicalizes empty PEP695 AST fields
that older Python parsers omit. This H3 result does not resolve the separate
distributed Gloo abort or establish that every CI workflow is green.

## Limits

This is tiny real-original-model **CPU** execution, not pretrained quality,
CUDA, distributed execution, capacity or performance validation. Canonical
GAN-start500 is unchanged; GAN-start2 is only an explicitly labeled transition
fixture. Original current AV values specify geometry; the native rollout draws
new noise and uses the supplied genuine prompt/previous AV conditions. No VAE,
text/audio encoder or media decoder is fabricated.

See the [runnable guide](../../recipes/h3_ar/PORTABLE_INFERENCE_CPU.md) and
[inference request](../../recipes/h3_ar/inference.cpu-latents.template.json).
The earlier report's W6 output-unavailable statement is superseded only for
this CPU cached-reference export/latent path. Other native output gaps and C51
physical pipeline recovery are not resolved by this change.
