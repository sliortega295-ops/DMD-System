# Original W4 reference lifecycle — CPU/source acceptance

The [reference recipe](../../recipes/self_forcing/REFERENCE_TRAINING.md) adds a
separate opt-in finite training lifecycle. It reuses the actual original
baseline constructor, streamers, score context and one-cycle update helper;
it does not replace the DMD math, inherit a historical bootstrap or widen C51.

| Boundary | Observed evidence |
| --- | --- |
| Actual input owner | Pinned original TeacherLatents through the real InputLoader; five successful loads per rank/cycle and finite cursor |
| Actual state owners | CPU DTensor Generator and original LoRA/Adam Critic; exact in-place interrupted continuation and RNG; no replacement Parameter or optimizer |
| Restart rejection | Hash-valid wrong Adam step, cursor/rank/version/layout/dtype/source mismatch reject before mutation |
| Shape admission | Genuine original Wan 1.3B/14B constructors on meta tensors; no pretrained weights or forwards |
| Metadata collective failures | Real two-rank Gloo failures do not strand a peer or enter model construction |
| Public outer lifecycle | Actual unchanged baseline helper dispatch with explicitly facaded unavailable Wan/CUDA construction/math/resource surfaces; not native numerical proof |

Worker proof-first RED contained four intended missing-route/typed-owner/state
failures; a separate setup-path error is preserved, not counted as behavioral
RED. Final worker scope passed 226 tests without skips. Parent regression on
the integrated source passed 266 tests without skips, including full W2
reference ownership regression and unchanged C51 timing/profile/request gates.
The [structured receipt](../../artifacts/framework/20261021/original_livewan_reference_cpu.json)
pins exact source bytes, the JUnit and runtime scope. It is not a GPU receipt.

Three independent simplify lenses completed. Source validation is intentionally
separate at metadata, payload, live-owner and checkpoint boundaries; it was
not removed for brevity. One redundant live snapshot was eliminated while
preserving mode/source/layout validation. Parent lint also removed unused
bindings and made the immediate failure-receipt exception string explicit.
No training math or historical source digest was changed.

Original source pins remain LiveWan `1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5`
and Wan2.1 `9737cba9c1c3c4d04b33fcad41c111989865d315`.
The reference uses world16 FSDP2 Generator/Teacher **and replicated Critic**;
it is not all-role FULL_SHARD. Initial versions are genuine G0/C0/EMA0.
Checkpointing requires complete Gk/C5k/EMA0, cursor 5k and true zero-gradient
owner boundaries. Fixed horizon, source, assets, seed, precision, optimizer
and topology must match on restart.

Pretrained CUDA/NCCL/FSDP2 forward/backward, native world16 restart, physical
capacity, quality, numerical equivalence and performance are **NOT_RUN**.
C51 physical two-slot transport drain/shadow recovery and unified native VAE
generation/export remain unsupported. Existing numerical REVIEW/FAIL evidence
is unchanged. Minimum Torch2.10 CI is defined separately; local Torch2.14 CPU
results do not assert an exact-commit CI outcome.

No GPU, SSH, model download, performance experiment, main modification or
stopped-queue resume occurred.
