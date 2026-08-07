# Validated Runtime Compatibility Layer

`runtime/` contains the mechanisms that were exercised by the real Helios/A100 research path before the typed `src/distillschedule/` refactor was introduced.

They are kept here deliberately during the strangler migration. New workload semantics and execution plans belong under `src/distillschedule/`; a validated mechanism moves into the typed package only after operation-level parity has been checked.

The important mechanism families are:

- `activation_offload.py`: audited saved-tensor CPU offload with parameter/buffer and zero-stride exclusions.
- `dmd_boundary.py`: detached DMD gradient and Critic denoising objectives.
- `ema.py`: trainable-state-only EMA representation.
- `deepspeed_backend.py`, `zero3_checkpoint.py`, `zero_partition.py`: distributed backend contracts.
- `host_admission.py`: node-level capacity admission and reserve accounting.
- `operation_trace.py`, `rollout_trace.py`, `trace.py`: operation-keyed replay and low-intrusion/NVTX tracing.
- `critic_pool*.py`: Critic-only producer/consumer topology, transport, and version contracts.
- `generator_update_gate.py`: isolated Generator-update correctness gate.
- `residency.py`: physical-state residency ownership and version safety.

Additional large mechanisms such as frozen-block streaming, immutable model backing and pre-cast checkpoints are migrated in the same way; raw experiment outputs are not source code and stay outside Git.

Do not treat this directory as a second public API. The stable API is `distillschedule.*` under `src/`.
