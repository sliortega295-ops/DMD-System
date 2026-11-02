# Performance and validation

## Current result

**The project maintainer confirmed on 2026-11-02 that the reported optimization
configurations meet the project's performance and correctness requirements.
Subsequent longer-cycle measurements showed comparable acceleration to the
Cycle 2 measurements used during optimization.**

Cycle 2 is the fast feedback window used to compare candidates, not the limit
of the training validation. The current conclusion above records the maintainer's
update; the linked experiment records retain their original dates and scopes.

## Reported measurements

The official workflow comparisons below use 16 A100-40GB GPUs, global batch 16
and unprofiled complete update cycles. DMD2 uses G1/C10; official Self-Forcing
uses G1/C5. The numbers retain their original Cycle 2 timing definition.

| Workflow | FSDP2 baseline, Cycle 2 (s) | Optimized, Cycle 2 (s) | Speedup |
| --- | ---: | ---: | ---: |
| Official SD1.5 DMD2 GAN, 512 × 512 | 7.514 | 3.475 | 2.162× |
| Official Self-Forcing Wan, 81 frames, 368 × 640 | 50.739 | 30.959 | 1.639× |

Both official optimized workflows also completed 100 cycles, a cycle-50 save
and fresh-process continuation to cycle 100, followed by Student export and
new-input generation. See [launch and configuration](official_optimized_training.md)
and the [run records](../status/official_optimized_framework.md).
The longer-cycle conclusion does not change the table into a 100-cycle mean.

## Reading current and historical results

Use the current recipe and this summary for the supported optimization
configurations. An old report's numerical label describes that experiment,
not the validation status of every later implementation in the same model family.
Original logs, failed candidates, source versions and measurement scopes remain
available in the research records. A specific failed candidate remains distinct
from the adopted optimization combinations, and an actual failed check still
requires investigation.

Acceptance follows the workload's correctness requirements; bitwise equality is
one possible check, not a universal definition of correct training. Hardware,
precision, data and algorithm settings remain part of each recipe.

## Measuring a change

Reuse matching long-cycle validation for an unchanged execution path. During
iteration, run Cycle 1 for warmup and compare Cycle 2. Recheck later cycles when
compilation, caches, shapes or scheduling change, and use paired measurements
for differences near the observed noise. Keep cold-start and complete-job time
separate from hot-cycle time. The [Skill timing policy](https://github.com/sliortega295-ops/DMD-System/blob/main/skills/dmd-system-optimizer/references/experiment-protocol.md#cycle-2-timing)
gives the procedure for new workloads and changed warmup behavior.
