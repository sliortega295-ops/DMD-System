# Iteration 376: C21 Generator no-grad persistent unshard

## Contract and mechanism

C21 changes only the physical FSDP2 residency of the Generator during the five
already-existing B1 no-grad Critic-input rollouts. All 40 blocks plus the root
FSDP unit use `reshard_after_forward=False` inside that bounded phase. The
first rollout materializes the full parameters, the next four reuse them, and
all 41 units are explicitly resharded before Critic work. GAN steps
1000--1004, global B16, trace/RNG, local FA2, model calls, losses, update order,
G1/C5/EMA1 and final G201/C1005/EMA801 remain frozen.

## Runs

| Gate | Run | Result |
|---|---|---|
| validate-only | `iter376_gan_cuda_c21_persistent_unshard_validate_r1_20260830` | 16/16 PASS |
| correctness attempt 1 | `iter376_gan_cuda_c21_persistent_unshard_correctness_r1_20260830` | integration failure; no workload verdict |
| correctness | `iter376_gan_cuda_c21_persistent_unshard_correctness_r2_20260830` | 16/16 PASS |
| unprofiled timing | `iter376_gan_cuda_c21_persistent_unshard_timing_r1_20260830` | **125.814004444 s** |

Attempt 1 is retained. The hardened wrapper still unpacked the historical
two-value rollout helper return after C21 added a third diagnostic receipt.
The bounded fix only propagated that receipt. It changed no tensor, model
operation, collective, RNG, optimizer or schedule. CUDA and cgroup OOM deltas
were zero.

## Correctness and capacity evidence

- Exact code HEAD: `9b528b356b9351fe18e1e1247c2bd786bdc4a5ef`.
- Correctness summary SHA256:
  `51ac69e3b217f939bd000cd5a3f9d05aab9b59f4f4804e520df0ace61b83e0db`.
- Timing summary SHA256:
  `938d0fb7db10f9eb9e4ddfe2a466efc0edf2d6490d057bdf68ca8a64f13a45a1`.
- Every rank reports 41 FSDP groups fully unsharded before phase exit and zero
  unsharded groups afterward.
- Every rank reports G201/C1005/EMA801 and the existing semantic gates PASS.
- Timing peak across rank rollout receipts is about `37.006 GiB` allocated and
  `38.594 GiB` reserved. This is capacity-qualified but has little A100-40GB
  margin; C21 must remain fail-closed and topology-specific.
- Both nodes report zero new CUDA/cgroup OOM and no residual worker.

## Performance

| Denominator | Full5 seconds | Change |
|---|---:|---:|
| C0 | 173.014483004 | -- |
| C18 | 129.441379467 | -25.1847% vs C0 |
| C21 | **125.814004444** | -3.627375023 s / -2.8023% vs C18 |
| 30%-reduction target | 121.110138103 | C21 remains +4.703866341 s |

C21 is the new fastest correctness-qualified result: `1.37516x` over C0, or a
`27.2812%` wall-time reduction. It does not yet satisfy the registered 30%
reduction target.

## Next diagnostic boundary

The C20 profile predates C21 and is no longer representative of Generator
rollout communication or allocator behavior. The next permitted action is
one exact C21 node-outer Nsight diagnostic, keeping the `125.814004444 s`
unprofiled denominator frozen. The capture will decide whether the remaining
4.70 seconds are dominated by Score/Critic FSDP communication, allocator/host
gaps, or source-owned compute. It does not authorize a new optimization by
itself.
