# H100 full-parameter-resident reference

This is a separate, registered reference for the H100 experiment, not an
unmodified official `train_helios.py` claim or a frozen Route-A replacement.
The original Helios losses and trace-replay helpers are reused. Both different
14B backbones and all LoRA parameters remain complete on every GPU throughout
the timer. No FSDP, parameter streaming, model swap, or CPU offload is used.

The previously validated ZeRO-1 logical-checkpoint restore shards GPU FP32
master/Adam state. This avoids the documented ZeRO-2 double-reduction path; we
do not retest that mechanism or silently change to ZeRO-2/3 on failure. EMA
shadows remain on GPU and use the original dtype/recurrence and update predicate.

Frozen comparison: one owned eight-H100 host, global batch16, two independent
original B1 trace rows per physical rank (`r,r+8`). The existing outer commit
adapter divides each loss by two exactly once and commits once after both rows.
DeepSpeed uses GAS1 and an outer logical microbatch of two; the DiT calls stay
physical B1. FP32 optimizer masters/reduction and BF16 model compute match the
existing FSDP2 mixed-precision policy. Initial model/master/Adam/scheduler/EMA,
noise, timestep, crop, exits, score requests, and update order are unchanged.

No-GAN: G1/C5/EMA5, final logical G3/C15/EMA15.
GAN: G1/C5/EMA1, final logical G201/C1005/EMA801; systems-only initialization is
not an official checkpoint1000. No Generator/Critic backbone weight sharing.

Baseline disables all registered performance options. The best arm uses its
existing fully explicit registry policy. GPU residual storage is shared by both.
Source/assets/topology validate before model construction; actual gradient and
GPU Adam-state audits, 8 physical/16 logical receipts, inventory, versions,
finite values, OOM deltas and owned teardown remain required. Numeric parity
that has not run is `NOT_RUN`, not strict correctness. One matched unprofiled
pair is authorized; startup and cold construction are not training time.

The old FSDP2 reference remains available under `baseline`; this new reference
is explicitly `resident`. A failed resident capacity run is retained, never
relabeled as FSDP or repaired by adding offload.
