# Original LiveWan resident state ownership

This integration adds serialization views over the existing `ModelBundle`,
not another engine, optimizer or DMD loop. Generator full FP32 parameters and
Critic original FP32 LoRA parameters retain their actual owner and ordering.
The frozen BF16 Critic base is reconstructed by the native asset owner; its
admitted identity is bound to the saved state.

The views reject changed source/base/LoRA policy, parameter ownership, fixed
Adam options, raw moment layout, malformed steps and unconsumed gradients
before mutating live owners. The shared native optimizer adapter now checks
raw known pointwise slots before PyTorch can silently cast malformed moments;
it still permits other native adapters to restore scheduled learning rates
and their own scalar-step representation.

## Verified locally

- Actual pinned Apache-2.0 `LoRALinear`, real CPU AdamW and the production
  `ModelBundle` dataclass on explicitly tiny modules.
- Interrupted/uninterrupted updates through the existing atomic checkpoint
  writer/reader: weights, moments, options and Python/Torch/NumPy RNG agree.
- Existing Parameter objects, zero-gradient storage views, post-accumulate
  hooks, trainability and mode ownership are preserved.
- Parent regression: 123 tests pass, including original Helios continuation
  and existing fresh input admission. After behavior-preserving simplification,
  51 scoped tests pass without the optional model dependency overlay.
- Actual-source CI additionally requires these tests to execute without skips.

See the [small machine-readable receipt](../../artifacts/framework/20261021/original_livewan_state_cpu.json).
Hosted CI and original-model CUDA execution are not inferred from local tests.

## Not provided by these views

No full Wan model, native G1/C5 cycle, CUDA reducer, fused Adam, pipeline
transport/drain, physical two-slot input acknowledgement, shadow recreation or
fresh public training/restart is qualified here. The existing public W4 guard
remains unchanged. Reference FSDP/DTensor owners are deliberately not accepted
as resident split-pool owners. These lifecycle seams must be supplied and
validated separately before enabling native pipeline recovery.
