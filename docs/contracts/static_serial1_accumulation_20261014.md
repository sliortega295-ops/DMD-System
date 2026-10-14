# Static TP4×DP2×PP2 serial-window capacity fallback

The primary GPipe8 and bounded GPipe2 native Helios No-GAN controls have
preserved CUDA OOM evidence. Torch2.10 rejects GPipe1. The opt-in
`--pipeline-memory-policy serial1-accumulate8` uses a separate synchronous PP2
autograd transport, **not a patched Torch schedule**. Default remains GPipe8.

One original B1 trace executes at a time. Each DP replica accumulates all
eight independently identified samples with loss scaling 1/8; only then does
the unchanged DP SUM/2, model-global clip, AdamW and original EMA commit run.
TP/PP do not duplicate logical samples. Global batch remains16 and native
G1/C5/EMA5 order, original source prefix/blocks/suffix, precision, RNG/trace,
checkpointing, parameter versions, optimizer moments and call counts remain
fixed. No merge, replay cache, activation offload or optimized kernel is added.

This deliberately synchronous capacity fallback may be slower than GPipe8.
Report it separately, never relabel its time as an efficient GPipe8 result.
The application census excludes native checkpoint replay. Receipts require
eight samples before the model-average view is valid and zero pending backward
transports after each cycle.

CPU Gloo PP2 tests compare real checkpointed forward, parameter and upstream
pyramid gradients to unsplit source fixtures. CPU PASS is not targetGPU PASS.
Target GPU construction/capacity/C2 remain NOT_RUN until separate receipts exist.
