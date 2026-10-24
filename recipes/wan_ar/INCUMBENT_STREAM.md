# W1 incumbent finite-stream lifecycle v2 — research only

`training.incumbent-stream.json` explicitly selects
`wan_ar.incumbent21a_stream.v2`. It uses the **same original Wan session,
engines, loss/update handlers, stream loader and native checkpoint owner** as
the original adapter, with the entire coupled incumbent bundle enabled:
Resident, chunk2048, no-grad and grad LoRA epilogues, modulated norm, rollout
CUDA Graph + exact adjacent-input cast reuse, and Critic grad CUDA Graph.
Nothing falls back to an unoptimized trainer. This is not a saved historical
plan and is not added to the default planner inventory.

## Admission and supported boundary

- Explicit finite `original_wan_cohort_stream.v1`, **2..64 distinct cohorts**.
  Requested cycles must equal its length. Use the existing `prepare-stream`
  workflow and its real per-rank input/trace validation; do not repeat a fixed
  trace or hand-edit source hashes. The assets' initial trace must be cohort 0.
- Only original Wan2.1 T2V 1.3B, world16/two nodes of eight, B1, 384x640x33,
  LoRA256, original three-stage `(2,2,2)` schedule and G1/C5/EMA5. Geometry,
  precision, original model file inventory and negative conditioning remain
  pinned. No retain-block/shared-pool/merged-LoRA experiment is admitted.
- Fresh hash-verified native bootstrap payloads are admitted only at cold
  **G0/C0/EMA0**, with the task starting at source step 0, original scheduler
  clocks and EMA policy. The exact old G2/C10/EMA10 synthetic initializer is
  also usable with source step 10. Neither is trained-state recovery: the
  original builder creates zero Adam moments. For any trained continuation,
  supply a complete native checkpoint rather than a renamed weights file.
- Actual graph inputs are copied into static tensors each call; returned
  rollout outputs are fresh clones. Parameter/buffer addresses and layouts,
  committed versions, source, trace, rank-qualified sampler census, and cast
  lifetime are checked. Cold capture must not consume CPU/CUDA/Python/NumPy RNG.

## Deliberately limited capture lifetime

Each process may cold-capture at most **12 rollout signatures and 3 Critic
signatures**, only during its first executed cycle. This includes the first
cycle after restart: captures bind the restored *next* version, not G3.
No old captures or cast tensors are loaded from a checkpoint.

**A later novel signature is rejected**, even when a legitimate new random
exit first selects a pyramid shape not seen during the process's cold cycle.
Thus not every otherwise valid fresh stream is executable by this opt-in.
There is no silent fallback, eager substitution, preconsumption of future
samples, extra update, or unbounded graph cache. Broader lazy capture requires
its own CUDA/reducer/RNG validation. Capture calls are counted separately and
remain included in actual compute/full input-and-execution wall measurements;
no new steady-state performance claim is made.

## Run and restart

Copy the template into your project directory and replace the explicit local
task/assets/stream/output placeholders. Use the copied configuration with the
public commands below in the already-qualified native world16 environment
(replace `/path/to/wan-stream.json` with its actual path):

```bash
distillctl config validate /path/to/wan-stream.json
distillctl doctor --config /path/to/wan-stream.json
distillctl train /path/to/wan-stream.json --dry-run
distillctl train /path/to/wan-stream.json
```

Configuration/dependency inspection and a dry run do not establish GPU
readiness or successful execution. This recipe does not install dependencies,
fetch weights or start machines. An explicit reference uses the original FSDP
strategy and the **same** prepared stream; it is never chosen as a hidden
fallback.

`checkpoint_every` publishes only complete, drained cohort boundaries: named
G/C weights and Adam ownership/moments, schedulers, EMA, native clocks, per-rank
RNG, source/assets/strategy/topology and stream cursor. Pending backward,
nonzero gradients, gradient communication, active graph/cast scope or retained
helper inputs prevent publication. Static Graph buffers are transient bounded
process caches, not checkpoint state.

To restart, keep source, task, complete stream, assets and topology identical,
and use a fresh output directory. Pass `--resume` with a completed
`checkpoints/cycle-NNNNNN` directory and `--output` with the fresh destination
(replace both example paths):

```bash
distillctl train /path/to/wan-stream.json --resume /path/to/run/checkpoints/cycle-000001 --output /path/to/resumed-run
```

Alternatively, set `trainer.resume_from` and `trainer.output_dir` in the copied
configuration and use `distillctl train` with that file. The last checkpoint at
the end of the entire finite stream cannot be used to invent additional
cohorts. Native checkpoint `_SUCCESS`, all-rank shard integrity and policy
validation apply.

## Evidence status

Source/CPU lifecycle tests exercise the original session/dispatcher, real CPU
autograd/Adam/scheduler/EMA/RNG and complete checkpoint restoration, with
explicit scalar model/helpers and fake CUDA capture drivers. They verify
orchestration/state and rejection boundaries, **not** real Wan math, Triton
kernels, CUDA Graph capture, world16 communication, HBM capacity, pretrained
quality, speedup or 100-cycle stability. All of those remain **NOT_RUN** for
this v2 integration; numerical status remains **REVIEW**. W1/U3/U7 and the
six-workload maturity plan are still partial.

Target-environment follow-up (2026-10-24 UTC) confirmed a real admission gap:
native history dropout can produce the unsupported all-zero-history amplified
branch during fresh cohort capture. That capture failed before GPU execution;
no seed selection, dropout change or fixed-trace substitution was used. The
separate v1 fixed-trace preset completed 16-rank functional execution, but this
does not qualify v2 stream checkpoint/restart. See
[actual GPU and stream boundary](../../docs/status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).

The original `training.json`/`wan_ar.incumbent21a.v1` two-cycle fixed-trace
preset, immutable asset pins, G3/G4 rules and fixed call census are unchanged.
