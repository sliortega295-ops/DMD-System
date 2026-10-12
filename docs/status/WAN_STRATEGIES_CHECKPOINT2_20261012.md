# Checkpoint 2: original task, trace and strategy capabilities

The new original-Wan task is distinct from PR30's Euler fixture protocol. It
registers the existing WanImplementation variants (FSDP reference or Resident
async128/Tree, chunks 512/1024/2048 and optional no-grad LoRA epilogue), with
windows determined by the actual Fake count, not a hard-coded five.

A trace adapter reuses StaticWorld16Trace's payload/entry loaders while validating
new shapes, stage budgets, update/branch identities and exact rollout inventory.
Legacy trace admission is explicit and only for the exact original task. No
truncation, resize, seed substitution or fake calibration is performed.

95 targeted CPU contract, original-chunk binding and native-dispatch tests pass.
The old chunk setter was called on 30-block fixtures for all admitted variants.
These tests are not model/GPU tests. The original engines remain world16/B1,
rank256; other models and arbitrary layouts are not silently admitted.

Important: the three-stage sampler overrides the stage1 denoising list. Therefore
that list remains fixed; stage2_steps controls the real per-stage budgets. Do not
claim changing the unused list changes the number of native DiT calls.

Next: publish native rollout argument binding, measured costs, engine creation
and selected-plan execution behind the existing native command. At this
checkpoint the new capabilities are not yet the complete executable pipeline.
