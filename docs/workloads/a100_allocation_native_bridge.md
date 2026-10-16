# Bounded A100 allocation bridge

`distillschedule.search.a100_allocation` adds an explicit, opt-in callable
adapter for the original Wan pyramid workload on 8 or 16 physical A100 GPUs.
It preserves sixteen independent logical B1 samples: physical rank `r` uses
`[r]` on sixteen GPUs and `[r, r+8]` on eight. On eight GPUs, each loss is
scaled by 1/2 before the native rank-mean reduction; optimizer and EMA commits
occur only after both contributions. This is not B2 batching and does not
reduce global batch.

This module uses the common cost model, constrained planner, saved-prediction
validator, and native command lowering. Only actually observed Resident and
FSDP whole-cycle costs are admitted, separately bound to each allocation.
Unmeasured choices remain uncalibrated. Predicted timing is diagnostic, not
formal speedup evidence. Construction capacity is not predicted.

The adapter pins an isolated compatibility snapshot (HEAD
`8148e84de048820df32f0a12b5657f1d9f530323`, tree
`83fbc856430dc7b7ab34268ef69bf4754296e482`). Its parent snapshot has exactly the
published native source tree at `7a668bcb12ad37d831dced3aca1be4ec7be0a2fe`,
but reconstructed Git metadata; it is not claimed to have that commit's Git
ancestry. The compatibility delta only permits the explicit A1008 profile and
adds post-timer allocation receipts. Training functions and engines are not
replaced with the different generic Euler workload. See the campaign lineage
manifest for original-tree verification.

Each admitted calibration must carry actual rank receipts, content hashes,
source cleanliness, observed runtime/device identities, exact launch argv/env,
owned cleanup and unchanged cgroup OOM counters. Saved plans revalidate these
references and reject editing or cross-allocation reuse. Numerical comparison
is separate; slot/census PASS alone is not numerical parity.

The public `workloads hardware` CLI remains H100-v1-only. This adapter does
**not** widen its hardware admission, borrow its H100 launcher, or claim that
all workloads accept arbitrary device counts. A caller must use the explicit
new deployment schema and an allocation-aware owned launcher. GPU evidence and
remaining API scope are reported separately in the parameter-generalization
campaign; CPU tests are not GPU capacity or full-model numerical proof.
