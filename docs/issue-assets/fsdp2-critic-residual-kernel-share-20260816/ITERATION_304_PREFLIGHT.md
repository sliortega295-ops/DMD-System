# Iteration 304 preflight

- Date: 2026-08-16 America/New_York
- Scope: read-only profiling/telemetry of the already validated No-GAN static World16 FSDP2 path; no model, trace, optimizer, precision, schedule, or source change.
- DMD-System root: `/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_gpu_residual_20260816`
- DMD-System HEAD: `bd837c27e555c8868d3469e96a516e4f6c88a9bf`, clean on both nodes.
- Helios runtime: `/share/project/shuchenweng/yongyan_liu/DistillSchedule_fsdp2_gpu_residual_20260816`
- Transformer SHA-256: `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886` on both nodes.
- Correctness gate: `iter303_nogan_gpu_residual_correctness_20260816`, 16/16 PASS; summary SHA-256 `2073e03aed3c0079b163fd6b2788ffb585505bfdcb3bf72d6a767965dcf780f5`.
- Frozen policy: No-GAN, world16 FSDP2, rollout B1, Score packing off, Critic residual `cuda_out_of_place`, GPU EMA.
- Frozen native reference: three matched runs `139.846688622 / 142.677073989 / 141.546299770 s`, median `141.546299770 s`.
- Resource audit: 16/16 A100 40GB idle; no compute processes; no DistillSchedule GPU lock held.
- Prior evidence: the old full trace used the superseded Critic CPU-residual path. Its 48.89% compute-kernel union and phase communication tails may rank candidates but are not current-path utilization.
- Authorized falsifiable questions:
  1. What is low-overhead GPU activity during the current native Full5?
  2. What fraction of the current profiled Full5 rank-wall is occupied by non-NCCL compute kernels?
  3. After the Critic residual fix, which FSDP2 all-gather/reduce-scatter intervals remain exposed by phase?
- GO: exact dry-run command rendered successfully with a fresh run ID and pinned correctness/runtime hashes.

Telemetry and Nsight results remain diagnostic unless separately tied to a matched unprofiled timing result.
