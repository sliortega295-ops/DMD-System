# Original Helios No-GAN CPU reference acceptance

This is the separately versioned **W2 original warm CPU Program**, not a new
pretrained/GPU result or W1/W3 with a different label. Behavior commit
`8ce0cb00` closes its input, original update, complete-cycle recovery and current
Generator output paths. The existing native world16 and selected-plan recipes
are unchanged.

## Observed acceptance

| Check | Actual result |
| --- | --- |
| Original warm bootstrap, public update, data and restart | 91 passed, zero skips |
| Independent current-G export and new-prompt inference | 37 passed, zero skips |
| Actual two-report CI required-case consumer | Passed against both reports above |
| CI consumer and owned-fixture cleanup regressions | 17 passed |
| Required refactor regression | 2153 passed plus20 subtests;85 existing skips |

The original model tests ran on Python3.12.3, Torch2.10.0+cpu,
Diffusers0.37.1, PEFT0.17.1 and Transformers4.51.3, offline with one CPU thread
and CUDA uninitialized. Dependency deprecation warnings remain visible in the
raw logs. Hosted execution of the new W2 CI workflow is **NOT_RUN**, not inferred
from local checks. The machine-readable
[receipt](../../artifacts/framework/20261023/original_helios_nogan_cpu.json)
binds the actual source, reports and review outcomes.

## Mathematics and state boundaries

- The complete original tiny Helios class executes at its original384×640,
  33-frame geometry with `[2,2,2]` pyramid budgets and LoRA256. Original
  mixed-BF16/FP32 precision, clipping, Adam/LR and SparseEMA policy are retained.
- The warm owner is genuinely G2/C10/EMA10, including nonempty named optimizer
  moments. The test fixture performs two actual original cold-to-warm cycles;
  it does not claim historical Step10 training or pretrained quality. Public
  training rejects the separate synthetic-gradient state-reader fixture.
- Each subsequent cycle uses original No-GAN P0 calls and advances G1/C5/EMA5.
  Full3 versus1+new-process2 matches actual state, optimizer, LR/EMA, data and
  inner/outer RNG. Physical execution remains source-rank0 B1 on CPU, **not**
  global-B16 training or world16 gradient reduction.
- Receipt v2 binds every executed frozen base shard and configuration. A
  same-layout alternate base or stale receipt fails before construction or
  restore. Partial native failures poison all owners; recovery needs a valid
  complete boundary, not an in-place update retry.

## Output boundary

The shared export path reads the restored current full unmerged Generator,
not its bootstrap, Critic, optimizer or an EMA substitution. Export preserves
the next update and RNG. A separate fresh process removes the old training
payloads, reconstructs only the original Generator and executes original
`run_generator` on new prompt tensors. FP32 BCTHW latents are explicit diagnostic
output, not decoded media. The existing original zero-history inference branch
is not relabeled as training-history parity.

See the [recipe](../../recipes/helios_nogan/CPU_REFERENCE.md) for explicit asset,
input, training and inference templates. Raw encoders, decoder, pretrained
quality, CUDA/FSDP capacity and numerical qualification require their own
admission. This acceptance does not complete the remaining W4/W5 output or
C51 physical-pipeline recovery work.
