# Native GAN TP4/DP2/PP2 transport component

Source parent: measured group2 adaptation
`d34279e0beb4e608ede64a52ae01dffb6b6eb9a5`.
Native Helios Transformer source required by the factory remains
`5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`.

The GAN-only PP view shares exactly the already partitioned model parameters.
Stage0 sends the original five conditioning/activation tensors followed by
its ordered native intermediate GAN taps. Stage1 appends its own taps and
executes the unmodified native output, Conv3d/GroupNorm/SILU heads and logits
assembly. Native per-block checkpoint dispatch is retained. Hook and final
heads must be owned only by stage1; do not duplicate physical parameters.

No taps may be omitted/detached. Their gradients contribute through the same
PP boundary as the ordinary hidden-state gradient. A structured native result
contains both sample and logits, not a sample tensor with a mock GAN receipt.
The GAN and ordinary Score views may share the same pruned parameter objects,
but scopes must not overlap or change before a pipeline invocation has joined.

This file does **not** assert a finished GAN worker, GPU correctness, fit or
C2 timing. Still required: real GAN bootstrap/head loading, structured output
transport in the cooperative proxy, GAN-G input-gradient routing, the existing
packed fake/real/perturbed-real R1 computation, replicated-head norm accounting,
and original G1/C5/EMA1 optimizer/version census. The source contract uses the
existing approximate R1 finite-perturbation loss, not a new higher-order Hessian.

Component tests execute real CPU Torch forward/backward and non-reentrant
checkpointing on small tensors. They are not the production workload and can
never provide its timing or a16-rank acceptance receipt.
