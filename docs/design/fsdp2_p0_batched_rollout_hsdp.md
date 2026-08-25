# Static FSDP2 P0: batched rollout, Score packing, and node-local HSDP

## Status

This is an opt-in optimization stacked on the validated static world16 FSDP2
baseline. The parent baseline remains the performance denominator. This source
has not yet completed a real 2x8 A100 correctness or timing run after the P0
hardening changes.

## Frozen mathematical contract

Both No-GAN and GAN-on keep world16/global batch16 and the same immutable
rank-qualified operation trace. The implementation does not change losses,
random inputs, DiT exits, optimizer order, update counts, or final versions.
There is no model swap, role transition, CPU parameter streaming, parameter
offload, or activation offload.

The only legal changes are physical execution changes:

1. after the single Generator commit, precompute all five Critic inputs before
   any Critic update;
2. combine trace-equivalent no-grad Generator rollouts into local B1/B2/B5
   physical calls;
3. combine adapter-disabled Real-Score conditional and unconditional requests
   into one B2 forward;
4. optionally use a two-dimensional `replicate=2, shard=8` FSDP2 mesh so
   frozen-parameter all-gathers remain node-local.

## Conservative default

The default policy is deliberately the least aggressive candidate:

```text
world16-fsdp + rollout B1 + Score packing off
```

B2, B5, packing, and HSDP must be selected explicitly. B5 is not eligible for a
formal timing run until B2 has passed the same correctness/capacity protocol.

## Why the rollout reorder is legal

After the Full5 Generator update, every Critic rollout consumes the same new
Generator version (`G3` for No-GAN or `G201` for GAN-on). A Critic rollout does
not read the current Critic or EMA version. Therefore the five rollouts can be
moved ahead of the five ordered Critic commits while Critic versions remain
strictly sequential.

The implementation stores five full-resolution BF16 `x0` tensors. Critic trace
inputs are staged to GPU once during rollout precompute, retained until that
step's Critic update consumes them, and then evicted from the runtime cache.
GAN Stage-3 source tensors are likewise staged once per rank and reused. This
removes duplicate input H2D at the cost of additional temporary HBM residency,
which is part of the capacity gate rather than assumed free.

## True trace-batched rollout

A B5 candidate is not five B1 calls placed next to each other. The worker first
proves that all five logical steps have identical physical pyramid-stage
schedules and stack-compatible tensor inventories. A synthetic operation-trace
adapter then stacks each step-qualified noise/history tensor while preserving
its logical identity. Helios executes one physical model call per denoising
call index with batch dimension five.

For a trace with `K` physical DiT calls per rollout, the ideal number of
Generator FSDP materialization groups changes from `5K` to:

```text
B1: 5K
B2: 3K
B5: 1K
```

This is an analytical count only. Correctness receipts and operation-attributed
Nsight/NCCL evidence must confirm whether the framework realizes the reduction.
One NCCL CUDA kernel is not labelled as one FSDP materialization group.

`global_step` differs by logical step and is therefore not required to be equal
in the static batching signature. It is not simply assumed irrelevant: every
B2/B5 row in a correctness run is compared against that row's original-step B1
execution. A real dependency on the first member's `global_step` therefore
causes the integration parity gate to fail before timing can be pinned.

## Numerical parity gate

Mean-relative error is no longer sufficient on its own. B1-vs-packed rollout
outputs and Score-packing references must simultaneously satisfy:

```text
finite / no NaN/Inf
same shape and dtype
mean-relative <= 5e-3
normalized L2 <= 5e-3
cosine >= 0.9999
max-absolute <= 5e-2 + 1e-2 * reference_peak_abs
```

Correctness-only instrumentation also records local trainable gradient shards
immediately before every optimizer step: one Generator gradient state and five
Critic gradient states. After Full5 it records local trainable parameters and
Adam `exp_avg`/`exp_avg_sq` shards. Each tensor has shape/dtype/statistics and a
SHA-256 checksum.

`scripts/distillschedule/compare_static_fsdp2_p0_correctness.py` compares a
candidate against the required world16/B1/packing-off control. Only exact local
parameter, optimizer, and per-step gradient checksum equality is called exact
training-state equivalence; loss/stat differences are also reported for
localizing non-bitwise drift.

## Score request packing

The No-GAN Generator boundary logically requires Fake conditional, Real
conditional, and Real unconditional Score requests. Fake uses the Critic LoRA;
the two Real requests both disable the adapter and differ only in text
condition, so they can share one B2 model forward.

For GAN-on, a scoped patch replaces only the ordinary non-decoupled
`compute_kl_grad` Real pair. The official outer DMD/GAN helper remains the owner
of timestep/noise sampling, DMD loss, GAN-G, and latent-gradient extraction.
The physical Generator-side Score inventory becomes:

```text
Fake Score B1        x1
Real Score packed B2 x1
GAN-G B1             x1
```

instead of four complete Score forwards. Correctness mode also executes the two
original Real B1 requests and compares Real predictions and the normalized DMD
gradient before allowing the update.

## HSDP candidate

`node-hsdp` creates a 2x8 CUDA DeviceMesh with dimensions `(replicate, shard)`.
Each node owns one replicated model copy, and every 14B block is sharded across
the eight local A100s. This trades roughly one additional 1/16 shard per model
per GPU for avoiding cross-node frozen-weight all-gathers. Trainable LoRA/GAN
state is still synchronized across the replica dimension by FSDP2.

The HSDP candidate must pass an HBM/correctness canary before timing. It is not
assumed to outperform world16 FSDP.

## NCCL attribution

The P0 profile analyzer now separates matched NVTX windows for Generator
training, batched rollout, Score DMD calls, and Critic work. NCCL kernel names
are classified as all-gather, reduce-scatter, all-reduce, broadcast, send/recv,
or unknown. It reports event counts, clipped summed kernel time, and exact
interval unions per phase and per rollout batch.

Nsight's portable CUDA-kernel export does not provide a reliable collective
payload-byte field, so payload bytes remain `null` instead of being fabricated.
The `5K -> K` hypothesis is tested by comparing matched B1/B2/B5 observed
all-gather kernel counts/unions together with the analytical materialization
inventory; it is not inferred from total NCCL percentage alone.

## CI contract

Torch-free `refactor-ci` continues to test the generic runtime without pulling
in a large PyTorch wheel. A dedicated `p0-unit` job installs CPU PyTorch and
runs exactly:

```text
pytest -q tests/test_refactor_fsdp2_p0.py
```

The job parses JUnit output and fails unless all eight P0 tensor tests execute
with `8 tests / 0 skipped / 0 failures / 0 errors`. Therefore module-level
skipping can no longer produce a green P0 gate.

## External rollout-engine boundary

Helios has inference support in vLLM-Omni and related diffusion-serving stacks,
so a future runtime should adapt an existing serving engine rather than build
another DiT server from scratch. They are not a drop-in training replacement:
DistillSchedule needs latent-only output before VAE decode, exact operation
trace/noise/exit replay, hot synchronization of the newly committed Generator
LoRA, and a zero-copy or collective tensor return path.

This PR therefore exposes `rollout_backend` as a typed policy but enables only
`native-fsdp2-batch`. Unknown external names fail closed until that adapter is
implemented and numerically validated.

## Required experiment order

1. validate all policy contracts without 14B construction;
2. world16 FSDP B1 / packing-off correctness control;
3. world16 FSDP B1 / packing-on;
4. world16 FSDP B2 / packing-on;
5. B5 only after B2 passes fit, multi-metric parity, and state auditing;
6. node-HSDP B1 capacity/correctness;
7. only feasible candidates receive three unprofiled Full5 trials;
8. one separate representative Nsight run reports Compute, operation-attributed
   NCCL, H2D/D2H, active union, and no-traced activity.

Profiled latency is diagnostic and never enters a speedup claim.
