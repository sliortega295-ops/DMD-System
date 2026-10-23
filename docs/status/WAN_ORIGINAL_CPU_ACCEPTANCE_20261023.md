# Original Wan Full5 public CPU lifecycle

This is an additive acceptance of behavior commit `1b00120d5070e434de00c0343ac3ec362648f240`.
It does not replace the native world16 recipe or revalidate its historical
performance. The [recipe](../../recipes/wan_ar/CPU_REFERENCE.md) and
[structured receipt](../../artifacts/framework/20261022/wan_original_public_cpu.json)
describe the independently versioned reference.

## What actually ran

The complete original tiny Helios-class model used by W1 ran with its original
mixed BF16 base/FP32-only modules and FP32 LoRA. The original P0 actions, not a
substitute DMD loop, performed G1/C5/EMA5 from genuine G0/C0/EMA0 initialization.
All three Generator pyramid-patch adapters participated. Original captured
controls for every source rank were admitted; execution was CPU1/source-rank0/B1
per-sample, not world16 gradient reduction.

The common Program lifecycle completed four uninterrupted cycles and an exact
three-cycle plus fresh-process continuation. Compared state included parameters,
named Adam, LR schedules, EMA, clocks, consumed sources, data cursor and both
native/outer RNG. Actual partial updates poisoned the owner. Restoring a prior
complete boundary recovered it; malformed integer trainability flags were
rejected before any owner or RNG mutation.

Export projected the current full **unmerged** Generator, not EMA, Critic or
optimizer state. Projection preserved the next actual update. A fresh process
then generated with the original native rollout on new prompt tensors after
old training inputs, weights and checkpoints were removed. The exact native
zero-history inference branch returns FP32 latents; no output cast was added.
There is no VAE decoder or decoded-video/quality claim.

## Verification boundaries

| Check | Actual result |
| --- | --- |
| Final source and finalized layout lock: W1 model/update/restart/export/new-input inference | 9 passed, zero skips |
| Final-source W3/W6 original fresh-process recovery and output regressions | 7 passed, zero skips |
| Native Wan0.35.2 strategy/adapter regression | 173 passed, zero skips |
| Required refactor regression after device-predicate compatibility fix | 2153 passed, 20 subtests; 85 existing skips |
| General0.34 framework/data/Program regression | 971 passed; 279 dependency-lane/optional skips |
| Full original0.37 combined characterization before the narrow follow-up fixes | 196 passed, zero skips; not mislabeled final-byte qualification |
| Final lightweight wheel outside checkout | Six-workload discovery without Torch/model import |

Model proofs used Python3.12.3 and Torch2.10.0+cpu with the separate original
dependency lanes. Source-layout entries now bind all three telemetry-only
overlays to the actual behavior commit; the original import map is unchanged.
Default CUDA math was not changed, but no CUDA qualification is inferred.

Nine actual local review lenses and one independent finding-validation batch
completed. Both confirmed findings were fixed and tested. The external review
route stopped before provider egress; no independent-provider corroboration is
claimed. Known native Gloo shutdown failures on earlier hosted revisions remain
unresolved, not hidden by this local acceptance.

Pretrained quality, media decoding, distributed numerical equivalence and GPU
performance remain **NOT_RUN**. W2/W4/W5 native-output gaps and C51 physical
pipeline recovery are separate remaining work; this is not whole-framework
production-ready certification.
