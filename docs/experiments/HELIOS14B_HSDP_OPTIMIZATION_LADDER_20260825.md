# Helios14B node-HSDP optimization ladder (Iterations 351-367)

## Status and claim boundary

This document publishes the exact optimization chain that reduced the Helios
14B No-GAN Full5 wall from the historical world16 FSDP2 median
`118.546757 s` to `96.529020 s` on 16 A100-40GB GPUs.

The topology was changed once, in Iteration 351, from world16 FSDP2 to
node-local HSDP (`replicate=2`, `shard=8`). All later candidates use that HSDP
control. Consequently:

- world16 FSDP2 -> node-HSDP is a placement ablation;
- Iterations 353-367 are compute/runtime optimizations on the frozen HSDP
  substrate;
- the final `96.529020 s` observation is one unprofiled candidate trial, not a
  final three-trial median;
- the Iteration 359 profile is diagnostic-only and predates the latest fused
  pointwise path.

## Frozen workload

- 2 nodes x 8 A100-SXM4-40GB, 384x640 BF16.
- No-GAN Full5, B1/rank, global batch 16, Score packing disabled.
- One Generator, five Critic and five CUDA EMA updates.
- Initial `G2/C10/EMA10`; final `G3/C15/EMA15`.
- Generator and Critic residual storage: `cuda_out_of_place`.
- No parameter offload, parameter streaming, model swap or activation offload.
- Same checkpoint, operation trace, RNG, inputs, model calls, loss, optimizer,
  update order and Full5 timer boundary.

## Results

| Iteration | Candidate | Full5 seconds | Evidence |
|---|---|---:|---|
| historical | world16 FSDP2 | 118.546757 median | 3 trials |
| 351 | node-HSDP placement control | 117.304613 median | 2 trials |
| 353 | Generator BF16 scale-one LoRA epilogue | 116.498820 median | 2 trials |
| 354 | all-role exact scale-one LoRA add | 115.731202 median | 2 trials |
| 355 | schedule-compatible B2 | NOT_RUN | strict parity failed |
| 356 | modulation chunk2048 | 116.008085 | slower; rejected |
| 357 | local-shard clip candidate | NOT_RUN | <=0.94 ms/call ceiling |
| 358 | state-preserving optimizer dispatch warmup | 108.438765 median | 2 trials |
| 361 | Generator no-grad residual fusion | 106.699092 | 1 trial |
| 364 | trainable-timestep residual backward | 104.866740 | 1 trial |
| 365 | fused phase-local timestep gradient staging | 103.952151 | 1 trial |
| 366 | exact modulated-norm epilogue | 101.212336 | 1 trial |
| 367 | exact modulation materialization | **96.529020** | 1 trial |

Final comparisons:

- versus world16 FSDP2 median: `-22.017737 s`, `-18.5730%`, `1.22809x`;
- versus node-HSDP control median: `-20.775594 s`, `-17.7108%`, `1.21523x`;
- versus Iteration 366: `-4.683317 s`, `-4.6272%`, `1.04852x`.

The aspirational 85-second target is not reached. It requires another
`11.529020 s` (`11.9436%`) reduction from the current candidate.

## Optimization process and code

### 1. Placement: world16 FSDP2 -> node-HSDP

Iteration 351 changes only parameter placement/communication topology. The
measured median changed from `118.546757 s` to `117.304613 s` (`1.01059x`).
The small gain establishes that topology alone is not the dominant solution.

The policy and validation live in:

- `src/distillschedule/runtime/fsdp2_p0.py`
- `src/distillschedule/runtime/fsdp2_static.py`
- `scripts/distillschedule/probe_static_fsdp2_p0_world16.py`
- `scripts/distillschedule/summarize_static_fsdp2_p0_world16.py`

### 2. Exact LoRA epilogues

Commits `d70e535` and `2790c3d` specialize the observed BF16, scale-one LoRA
add and extend it from Generator no-grad calls to all eligible Generator and
Critic roles. The runtime fails closed on dtype/scale/adapter drift and records
fallback counts.

Primary code:

- `src/distillschedule/runtime/helios_lora_epilogue.py`
- `src/distillschedule/runtime/helios_lora_epilogue_kernel.py`

### 3. State-preserving optimizer dispatch warmup

`b134c6f` addresses a measured cold-first-update signature: the old profile
showed a `4.526 s` Generator optimizer window and `3.663 s` first Critic
optimizer window, while later Critic updates were about `0.066 s`.

Before the formal timer, the candidate performs a zero-gradient AdamW/DTensor
dispatch with `lr=0` and `weight_decay=0`, then exactly restores parameters,
Adam state and step counters and clears gradients. Receipts show zero changed
elements after restoration and explicitly exclude warmup from Full5 timing.
This reduced the median from `115.731202 s` to `108.438765 s`.

Primary code:

- `src/distillschedule/runtime/fsdp2_static.py`
- `tests/test_refactor_fsdp2_optimizer_warmup.py`

### 4. Exact residual and timestep-gradient fusion

Iterations 361, 364 and 365 progressively remove high-count pointwise and
staging work:

- Generator no-grad gated residual fusion (`5468e19`, `bcdb05a`);
- exact trainable `phase_local` timestep residual backward (`409cdc1`);
- fused phase-local timestep gradient staging (`4116dbe`, `8dc0fdb`).

They retain the original BF16-before-reduction order, trainable timestep
gradients, schedule and HSDP communication. Runtime receipts require exercised
calls and zero fallback.

Primary code:

- `src/distillschedule/runtime/helios_residual_fusion.py`
- `src/distillschedule/runtime/helios_residual_kernel.py`
- `tests/test_refactor_helios_residual_fusion.py`

### 5. Exact modulated-norm epilogue and materialization

`4b16e18` fuses the modulated-norm epilogue. `6d45b4e` additionally fuses the
scale/shift/table component materialization, and `6005c03` admits the live BF16
FSDP-unsharded modulation table while preserving FP32 support.

The first Iteration 367 correctness attempt at `6d45b4e` failed closed because
the host gate only accepted FP32 tables while the live table was BF16. No
workload-contract change was made. The bounded fix accepts only BF16/FP32 and
prewarms the exact BF16 specialization.

The A100 gate is bitwise for all 32/32 parity cases. The representative BF16
table microbenchmark changed from `0.209920 ms` to `0.094208 ms` (`2.2283x`).

Primary code:

- `src/distillschedule/runtime/helios_modulated_norm_fusion.py`
- `src/distillschedule/runtime/helios_modulated_norm_kernel.py`
- `tests/test_refactor_helios_modulated_norm_fusion.py`

## Iteration 359 Nsight evidence

The complete 16-rank post-warmup profile, collected at the `108.438765 s`
substrate, reconciled all phase partitions:

- rank0 critical path: `89.48%` non-NCCL compute-on-path;
- NCCL-only: `1.84%`, about `2.03 s/rank`;
- rank0 no-kernel complement: `8.68%`;
- strict 16-rank no-traced complement: `7.883%`;
- the remaining `138.184 GPU-s` is `UNKNOWN`, not labelled CPU idle;
- largest stages were Generator no-grad rollout (`36.012 s`) and Critic
  backward (`33.003 s`).

Most NCCL activity was already hidden by compute, so the experiment rejected
blind all-gather/prefetch work and moved to compute/launch/materialization
fusion. Hardware SM/Tensor/DRAM metrics were not captured; MFU remains
`NOT_AVAILABLE`. A fresh profile is required on the `96.529020 s` substrate.

## Final Iteration 367 evidence

- Run: `iter367_helios14b_hsdp_modnorm_materialization_timing_r1_20260825`.
- Tested source HEAD: `6005c036a8fc0df1d35938e845fbecb919d2cd3b`.
- Strict correctness: 16/16 PASS, G1/C5/EMA5, final `G3/C15/EMA15`.
- Full5 world maximum: `96.529019750 s`.
- Per-rank min/median/max: `96.507737 / 96.513712 / 96.528992 s`.
- Exact epilogue/materialization calls: 3,760/rank.
- Exact epilogue/materialization chunks: 54,640/rank.
- Reference fallback: zero.
- Peak allocated: `19.7455 GiB/rank`.
- Maximum reserved: `31.6641 GiB/rank`.
- Minimum driver free: `4.6117 GiB/rank`.
- CUDA/cgroup OOM delta: zero.

Raw correctness and timing summary SHA256 values are preserved in the
machine-readable manifest.

## Rejected and superseded paths

- Schedule-compatible B2 (`[[10,11],[12],[13],[14]]`) failed strict parity;
  timing was correctly left `NOT_RUN`.
- Chunk2048 regressed `0.2392%` on Helios14B; Wan1.3B's chunk result must not be
  generalized across models.
- Local-shard clipping had an optimistic CPU-probe ceiling below 1 ms/call and
  was stopped before a GPU run.
- The frozen-timestep residual hypothesis did not match the live trainable
  `phase_local` execution and was superseded by the exact trainable path.
- TP/PP/DP and a new legal schedule reorder have not been measured. Only the
  world16-FSDP2/node-HSDP placement ablation is complete.

## Remaining work

1. Run three independent unprofiled repetitions of the final candidate.
2. Collect one fresh diagnostic Nsight profile after the materialization
   fusion; do not reuse Iteration 359 as a current bottleneck decomposition.
3. Only then preregister one fair schedule or TP/PP/DP ablation, keeping the
   trace, model calls, batch, versions and update order frozen.

The machine-readable result is
`artifacts/manifests/iteration_351_367_helios14b_hsdp_optimization_20260825.json`.

