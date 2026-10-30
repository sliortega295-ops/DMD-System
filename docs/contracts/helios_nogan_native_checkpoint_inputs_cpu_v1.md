# W2 native-history full SafeBundle checkpoint-hidden CPU preservation v1

Public opt-in: `helios_nogan.native_history_safe_ed8_checkpoint_cpu.v1`.
Training identity: `distillschedule.helios_checkpoint_hidden_cpu_identity.v1`.
Checkpoint policy: `distillschedule.helios_checkpoint_hidden_cpu_checkpoint.v1`.
Executed boundary receipt:
`distillschedule.helios_g_checkpoint_hidden_cpu_execution.v1`.

This separately versioned capacity candidate composes the unchanged
[native-history SafeBundle contract](helios_nogan_native_history_safe_v1.md).
It does not widen or replace old presets, original history geometry or original
checkpoint callable admission. The mathematical algorithm, all true gradient
chains, RNG and call/update inventory are unchanged. No parameter/optimizer
offload or kernel tuning is authorized by this contract.

The only eligible saves are the exact first hidden tensor of each of the pinned
40 original whole Generator checkpoint blocks, once for each of the three
differentiable B1 pyramid stages. Shapes, BF16 dtype and capacity are exact;
extra inputs/stages and altered ownership/callables fail closed. Eight disjoint
rank partitions bound pinned or pageable node storage before allocation.

Registration is admitted while the original sharded FP32 DTensor owners are
visible, before FSDP materialization substitutes compute Parameters. The
already-admitted wrapper is installed only inside the complete-block phase.
All 120 updating-exit boundaries must copy and release their inputs, but not
every exit remains connected to the sole final-x0 loss: later original no-grad
prefixes sever earlier stages. The original validated executed stage schedule,
not configured exits or a restore-count range, determines the required suffix.
Each stage/block/shape record must show exactly one required restore and release,
or an actual unused release for a dead prefix. No extra backward is introduced.

Current policy/source, full SafeBundle, asset/input/mesh/version identities and
the declared capacity are admission requirements before original construction.
The unchanged startup/running host reserves are 512/64 GiB, and restore preserves
the unchanged 512 MiB GPU reserve. Fresh observations are not resource leases
against unrelated applications. Complete native checkpoint publication and
same-topology restore retain their existing all-rank/RNG/data/state gates.
Temporary input leases are drained and rebuilt, never saved as training state.

See the [public recipe and exact byte accounting](../../recipes/helios_nogan/CHECKPOINT_INPUT_CPU.md)
for the actual runtime boundary, host files, executed counters and refusal rules.
CPU synthetic/component proof cannot promote native HSDP capacity, numerical
parity, timing or quality. Target Torch 2.10 and target GPU evidence remain
**NOT_RUN** unless independently executed; `native_GO` remains false.
