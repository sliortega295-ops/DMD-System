# W1 R2: FSDP2 + independent B subset (opt-in)

Authorization `om_x100b626e7bc1f0a4b4bf4a6d70a7430`; parent `a337d154`.
This is **not complete B**. The original `only_B` refusal is retained.

- New cell `only_B_fsdp_compatible`, exact source/config binding in R2 v1.
- A remains OFF: world16 FULL_SHARD, reshard-after-forward, original FP32
  LoRA/BF16 base, FP32 SUM/post-divide, chunk512, NCCL AUTO, original calls/order.
- B subset: original runtime constants/clean path/root mode idempotency,
  no-grad Generator LoRA epilogue, grad LoRA epilogue, width1536 Norm fusion.
  The original norm kernels accept arbitrary positive chunk sizes; registration
  and warm-up explicitly use 512. No kernel formula is changed.
- G/C Graph and capture cast cache remain OFF (A×B). Merged shadow, compile,
  new optimizer, parameter retention and offload remain OFF.
- Per-PEFT pre-hooks verify actual current unsharded parameter dtype and grad
  metadata without retaining parameter references or consuming RNG. A DTensor
  reaching the arithmetic scope is rejected, not converted via all-gather.
- Norm uses the current call's tensors; no parameter-derived values survive
  calls. Schedule cache is per allocation/device and contains only immutable
  schedule constants. LoRA epilogue's ContextVar closes on return/exception.
- Same four-cycle timer and C1 qualification; no profiler. One independent
  run; C2/C3/C4 and their mean, never n=3. Numeric status stays NOT_RUN unless
  supported by independent cross-arm numeric evidence.
- This source is new; existing baseline45fc and A/full a337 are provenance,
  not falsely labeled the same HEAD. No canonical default is changed.
