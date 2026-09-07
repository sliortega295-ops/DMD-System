# Joint-planner Wan native bridge — 2026-09-07

## Scope and source

**Incumbent revalidation, not automatic model-search discovery.** No main merge
or canonical winner change. Exact measured source:
`be877cef1b3599266a4d8b75f304006653cae5b9`.

New branch: `agent/dmd-joint-planner-20260907`; Draft PR #28.
Both nodes used the same clean detached source worktree:
`/share/project/shuchenweng/yongyan_liu/DMD-System_deploy/joint-planner-be877cef1b35-20260907`.
Existing deployments and environments were preserved.

Environment: 2 × 8 A100-SXM4-40GB, driver 535.161.08,
Python 3.10.20 at `/tmp/distillschedule-helios-official-torch210/bin/python`,
Torch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5. No package installation.
Helios Transformer source SHA256:
`5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`.

Matched contract: No-GAN, B1/global-B16, same trace/bootstrap, G1/C5/EMA5 per
cycle, BF16 base / FP32 LoRA and gradient reduction. The generated family
explicitly sets **both matmul and cuDNN TF32 ON** and verifies the runtime
observations. Both arms use GPU residual storage and reference GPU EMA;
parameter/activation/optimizer offload, streaming and model swap are off.
The Resident incumbent also uses its existing Tree reduction/chunk2048/LoRA
epilogue; this is **not** an isolated placement-only comparison.

## Completed gates

Correctness IDs:

- `jointp_wan_bridge_20260907_r1-gate-joint-wan1-fsdp-c512-e0`
- `jointp_wan_bridge_20260907_r1-gate-joint-wan1-resident-c2048-e1`

Each completed 2/2 cycles and 16/16 ranks. Native workload census, trace/sample,
finite, version, runtime precision and boundary checks passed. Final state on
every rank: **G4/C20/EMA20**. Coarse trajectory comparison returned matched;
its historical status label contains `50C`, but the validated cycle count is
**2**. Strict P0 numerical equivalence is **NOT_ESTABLISHED / REVIEW**.

## Raw unprofiled timing

| Arm | Cycle 1 warm-up (s) | Cycle 2 (s) | Construction max (s) | Peak allocated (GiB) | Peak reserved (GiB) | Min sampled driver free (GiB) | Max sampled node cgroup (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSDP control | 45.774857227 | 36.443518619 | 42.126 | 4.2983 | 6.5215 | 32.0098 | 52.3487 |
| Resident incumbent | 27.660357748 | 24.628109044 | 43.455 | 21.8989 | 22.3730 | 16.2246 | 54.1180 |

Construction is outside timing. Cycle 1 is not averaged into cycle 2. Each row
is **one independent process initialization, followed by two continuous cycles**.
The diagnostic ratio is `36.443518619053066 / 24.628109043929726 = 1.4797530153`.
It is neither a strict-equivalence speed claim nor a model-search improvement
over the existing incumbent. No Nsight time is used.

Timing IDs:

- `jointp_wan_anchors_20260907_r1-00-baseline`
- `jointp_wan_anchors_20260907_r1-01-incumbent`

Both have 16 passing native rank receipts, G4/C20/EMA20, no OOM increment and
passing owned teardown. Subsequent read-only SSH audit found no GPU compute
process on either node, both node locks free, cgroup `oom=0`, `oom_kill=0`.
Native allocator peaks and 2-second sampled driver/cgroup telemetry are separate
metrics; sampled minima/peaks are not continuous extrema.

## Remaining, explicitly NOT_RUN

- Native operation/layout calibration collection.
- New non-seed plan selected by calibrated model and executed on GPUs.
- LiveWan/H3 joint-planning native adapters and pilots.
- Independent confirmation of any newly found winner; strict parity.

Two of the authorized eight timing attempts have been spent. Source changes do
not erase that expenditure. Calibration is a separately reported diagnostic
cost, not formal training performance. The new validation controller is CPU
tested but cannot run until real, matching calibration receipts exist.

## Persistent evidence

Small machine-readable evidence is committed at:
`artifacts/manifests/dmd_joint_planner_wan_bridge_20260907.json`.
It includes content hashes and persistent references to native receipts.

Raw remote run root:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/wan_native/`.

Local small evidence and controller commands:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/`.
No model weights, checkpoints, SQLite or Nsight payloads were downloaded.
