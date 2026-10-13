# Helios No-GAN static TP4/DP2/PP2 binding

The 14B path is explicit opt-in via `--workload helios-nogan` in the existing
native static worker. It keeps 40 native blocks (20/20), original G1/C5/EMA5
order, 16 independent rank-qualified B1 samples, and no DMD-specific
optimization. Native CPU tests do not imply a GPU capacity result.

## Source-specific state/precision

- Restore the original logical FP32 masters on CPU, then partition TP/PP.
- Restore **real** logical Adam first/second moments and the physical
  scheduler steps; do not use Wan's fresh-zero-moment bootstrap.
- Restore sparse EMA with shared read-only mmap pages. Only uniquely owned
  TP slices move to GPU; source shadow dtype and original recurrence remain.
- The reference 14B FSDP2 policy requests BF16 compute with FP32 masters.
  Explicit per-projection BF16 compute views preserve FP32 LoRA masters,
  autograd and PEFT adapter control. Root Linear and Conv3d use their native
  equations. This is precision compatibility, not a speed optimization.
- Wan default remains FP32 LoRA compute. No canonical profiles change.

The new CPU checkpoint fixtures observed expected-red missing-binding
failures, then exercise real nonzero Adam slices, sparse EMA dtype/recurrence,
and BF16 Linear/Conv3d forward/backward with FP32 master gradients.
GPU capacity/timing: **NOT_RUN at this commit**, pending both-node CPU gates
and a clean-source, under-lock owned world16 launch.
Numerical trajectory revalidation: **DEFERRED**, not a numerical PASS.
