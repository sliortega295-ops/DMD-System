# Frozen contract: RA16-F5-GLP-OFFLOAD

`RA16-F5-GLP-OFFLOAD` is a separately named, one-variable functional recovery
candidate after the bounded-negative `RA16-F5-PR12`. It does not reopen PR12,
automatically continue to g16, or authorize formal timing.

## Frozen workload

The model, official 384x640 trace, checkpoint-10, BF16 precision, two-node
world16 layout, ZeRO-1 CPU optimizer, MB1/GAS1/global batch 16, steps 10--14,
G/C/EMA update order, A32 activation policy, g12 Generator block set, Score
residency, and expected `G2/C10/EMA10 -> G3/C15/EMA15` transition are exactly
PR12.

## Only new mechanism

At the safe boundary after the Critic no-grad Generator rollout and before the
Score/Critic phase, the runtime may offload **only** the inactive Generator
DeepSpeed ZeRO-1 replicated low-precision `bit16_groups_flat` state to pageable
CPU memory. The state must be reloaded and the model Parameter views rebound
before EMA or any later Generator read. Gradient, optimizer moment, Score,
activation, trace, batch, loss, parameter version, and update semantics may not
change.

The runtime must fail closed unless it observes ZeRO-1, CPU optimizer offload,
no parameter offload, no live Generator gradients, a typed LP inventory, a
balanced DeepSpeed offload marker, and a balanced final lifecycle.

## Bounded gates

1. CPU unit and source/config tests;
2. one small CUDA ZeRO-1 offload/reload equality gate;
3. one world16 Full5 functional attempt under an atomic two-node GPU lock.

The world16 candidate is refuted if the typed state is not about 2.33 GiB/rank,
if state/value/version identity changes, if Critic entry allocation falls by
less than 2.0 GiB/rank, or if the same Critic backward OOM persists. Formal
repeated timing remains **Not Run** regardless of outcome.
