# Wan1.3B merged-LoRA shadow: 30-cycle self-noise experiment

Study: `wan13b_shadow_noise30_20261012_r1`
HEAD: `c69876197aeaccbab19b515b54d1022f07634422`
Preregistration SHA256: `7e1d6464abf31b28e29c4e4563f7f1316b25be5a83c867c5c4c50e6550dcd5f3`

**B and C fail the preregistered trajectory gate. No native paired C2 timing or speedup claim.**

All four arms completed 30/30 cycles, 16/16 ranks; G32/C160/EMA160; finite losses, frozen sample/RNG identities and call census PASS; no new CUDA/cgroup OOM. Each arm used one model initialization and no profiler.

## Final parameter movement and curves
Parameter deviation below is `||theta_X - theta_A|| / ||theta_A - theta_bootstrap||`, not deviation relative to full initial parameter magnitude. Loss difference is 30-cycle rank-mean curve normalized RMSE.

| Arm vs A | G parameter deviation | Critic parameter deviation | G loss NRMSE | Mean Critic loss NRMSE | Original G probe C30 nL2 |
|---|---:|---:|---:|---:|---:|
| A_repeat | 0.1650% | 0.0748% | 0.03869% | 0.00184% | 1.4724% |
| B_round_nearest | 1.5252% | 27.6171% | 0.44731% | 4.24430% | 1.9933% |
| C_stochastic | 1.4041% | 24.7643% | 0.39137% | 4.08293% | 1.8816% |

The original G probe can lie within 2x self-noise while the parameter trajectory fails by orders of magnitude. This is why a single rollout nL2 threshold is insufficient.

## Critic trajectory at all saved anchors
| Cycle | A-repeat noise | B deviation | C deviation |
|---|---:|---:|---:|
| 5 | 0.1112% | 0.6822% | 7.9520% |
| 10 | 0.0752% | 2.5780% | 12.9441% |
| 20 | 0.0729% | 11.6464% | 20.3634% |
| 30 | 0.0748% | 27.6171% | 24.7643% |

## Every-refresh rounding evidence (rank0, outside timers)
| Arm | First10→last10 mean lost_ratio | Last10 median bias_cos | C1→C30 fraction below half BF16 ulp |
|---|---:|---:|---:|
| B round-nearest | 0.992011→0.969348 | 0.93906982 | 99.813%→96.080% |
| C stochastic | 11.743508→4.980284 | -0.00000809 | 99.813%→96.164% |

**B:** rounding error is almost the negative of the LoRA update. All last10 refreshes satisfy the preregistered near-one lost/bias test; Critic parameter excess grows monotonically well beyond 2x noise.
**C:** private stochastic rounding removes directional weight-rounding cancellation, but error L2 remains about 5x delta in the tail. Critic trajectory excess still grows from 7.95% to 24.76%. This is not a successful training repair. Unbiased weight rounding does not imply identical nonlinear rollout/training; attributing all remaining drift to variance alone would require another experiment, not done here.

## Resources and timing boundary
| Arm | Peak allocated GiB | Peak reserved GiB | Status |
|---|---:|---:|---|
| wan_shadow_noise30_A_20261012_r1 | 21.878 | 22.383 | 30 cycles,16 ranks PASS |
| wan_shadow_noise30_Arepeat_20261012_r1 | 21.878 | 22.383 | 30 cycles,16 ranks PASS |
| wan_shadow_noise30_B_20261012_r1 | 24.584 | 25.422 | 30 cycles,16 ranks PASS |
| wan_shadow_noise30_Cstochastic_20261012_r1 | 24.584 | 25.422 | 30 cycles,16 ranks PASS |

Peaks include diagnostic audit/probe work. CUDA/cgroup OOM deltas are zero. Final audits: GPUs0MiB, no compute processes; both node flocks obtainable and port29873 free.

Four controller envelopes sum to 2983 s (13.258 GPU-hours for16cards), below32GPU-hour budget; one additional tiny no-model CUDA canary verified reproducibility/unbiasedness and unchanged global RNG.

Trajectory times in compact/raw receipts are diagnostic only: finite/grad-norm recording remains enabled. Probes, full snapshots, anchors, rounding audits and serialization are outside the recorded training window. Native C2 timing remains **NOT_RUN** for both merged arms.

## Engineering / reproducibility
- Default trajectory behavior and every other numerical threshold unchanged; explicit study opt-in only. Same norm fusion/epilogues ON and CUDA Graph OFF in every arm.
-95 focused CPU tests PASS on each node; report rendering succeeded before GPUs using existing base matplotlib3.7.3 import path, no installation.
-A pre-GPU shared-pyc compile collision and brief shared-filesystem visibility failure were preserved; compile caches moved to node-local scratch. No workload was lost to report failure.
-Full initial FP32 trainable tensor contents match across arms; snapshots at0/5/10/20/30 stay remote. Full optimizer/EMA numeric sketches and loss/grad curves reuse the existing trajectory comparator.
-Local source commit only; no push, no main changes, no extra optimization experiments.

Persistent remote report root: `/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/wan13b_shadow_noise30_20261012_r1`

Raw rank receipts and full LoRA snapshots: `/share/project/shuchenweng/yongyan_liu/DMD-System_worktrees/wan13b-shadow-noise30-20261012/artifacts/runs/<run_id>/execution/shadow_study`
