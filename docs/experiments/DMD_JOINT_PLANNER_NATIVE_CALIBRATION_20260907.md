# Native Wan stage calibration pilot — 2026-09-07

## Identity and claim boundary

- Run: `jointp_wan_stage_c1024_20260907_r1-calibration`
- Executed source: `4e3ffb59c21429f3e9e714f14006f5ae6afbd002`. Later evidence publication commits are not the executed source.
- Implementation: `joint-wan1-resident-c1024-e1` (non-seed Resident/chunk1024/epilogue-on).
- Same frozen Wan B1/global-B16, BF16 base/FP32 LoRA/reduction, matched TF32-on, G1/C5/EMA5 per cycle; two cycles, one construction.
- Existing native call sites and math only; no duplicated training loop. No canonical profile/main edits.
- Two owned nodes × 8 A100-SXM4-40GB; reused Python 3.10.20 / Torch 2.10.0+cu128 / CUDA 12.8 / NCCL 2.27.5 / driver 535.161.08. No installs.

## Real cluster outcome

- Both current-source correctness runs: 16/16 × 2 cycles PASS; coarse trajectory MATCHED. The legacy coarse status name contains `50C`; actual execution is **2**, not 50 cycles. Strict numerical equivalence remains unestablished.
- Calibration: **16/16 PASS; 224 ordered rank-stage observations; 14 assembled operation costs**. Final G4/C20/EMA20.
- All cost content hashes and operation signatures reconciled through `CostModel.verify_receipts()` on the remote substrate.
- Post-setup static 17.7618 GiB; max allocated 21.8996 GiB; max reserved 22.3281 GiB; sampled minimum driver free 16.2695 GiB.
- Sampled node cgroup peaks: 53.468 / 53.784 GiB. OOM/oom_kill increments 0/0 on both nodes; no CUDA OOM in completed workers.
- Native teardown and a subsequent independent audit: both GPU process lists empty, both node locks available.

## Cost samples, not formal cycle timing

Entry barrier/sync and trailing sync perturb cross-stage overlap. Duration is **max of sixteen synchronized rank-local stage envelopes**, not kernel union, GPU-seconds, or a counterfactual removable time. Cycle envelopes are diagnostic only: **28.927613 / 25.867769 s**. They must not be compared with old native times as speedup. The sum of stage maxima is not the exact global cycle wall time either.

| Native macro-stage | Cycle 1 max rank (s) | Cycle 2 max rank (s) |
|---|---:|---:|
| generator_update | 9.188945 | 5.856460 |
| critic_rollouts | 8.097875 | 8.083177 |
| critic0_update_ema | 2.346753 | 2.624152 |
| critic1_update_ema | 2.315580 | 2.317475 |
| critic2_update_ema | 2.319960 | 2.319782 |
| critic3_update_ema | 2.320170 | 2.323730 |
| critic4_update_ema | 2.319389 | 2.320804 |

Workspace is maximum live allocation above post-setup static. Retained outputs are included once; stage peaks survive internal peak resets. Capacity subtracts observed non-allocator residency and 2 GiB safety. Driver/NCCL transient residency is sampled, not continuously measured.

## Repairs and verification

1. Added an explicit generated-only `two-cycle-calibration` path. Normal timing rejects diagnostic receipts, even when their old schema is forged.
2. Executing the actual launcher Python heredoc exposed a wrong fall-through into its old one-cycle gate; fixed and regression-tested before GPU launch.
3. FSDP helpers reset allocator peaks inside rollout. The diagnostic context now samples those peaks before invoking the original reset and restores its wrapper on failure. The CPU regression test preserves a simulated peak of 1,200 B rather than incorrectly reporting 300 B. This canary was Resident; GPU validation of FSDP nested-reset capture is still NOT_RUN.
4. 49 focused tests, 558 full-refactor PASS / 17 skipped; five-profile integration PASS; runtime-head GitHub CI 6/6 SUCCESS. Bounded independent review completed; full CE review NOT_RUN.

## What is still missing

- This is **one measured layout fragment**, not the complete seven-implementation database. No historical times or guessed costs fill missing entries.
- A generated-plan native timing and confirmation still need execution. No new automatically discovered winner or speedup is claimed.
- Two earlier native timing attempts remain spent; **six remaining**, not a reset budget. This turn ran two required current-source correctness trajectories and one diagnostic calibration; zero formal timing attempts.
- LiveWan/H3 Grid1000 generated lowerers still require implementation.
- To reuse these costs, keep execution/controller source at the frozen runtime HEAD above and the same deployed import paths. Do not relabel them as a later evidence-publication HEAD.

## Artifacts

- Persistent controller evidence: `/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/wan_stage_calibration_r1`
- Raw native run: `/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/wan_native/jointp_wan_stage_c1024_20260907_r1-calibration`
- Small machine-readable publication: `artifacts/manifests/dmd_joint_planner_native_calibration_20260907.json`. It contains per-operation cost keys/hashes, all provenance links, runtime binding and budget.
- Old anchors, raw run IDs, environments and main were preserved.
