# Official optimized workflows: source and run records

[Training commands](../guides/native_workloads.md#official_optimized_training) ·
[Performance overview](../guides/performance_validation.md)

## Performance sources

16 A100-40GB GPUs, global batch 16; complete native Cycle 2.

| Workflow | Profile | FSDP2 baseline (s) | Optimized (s) | Optimization run | Upstream revision |
| --- | --- | ---: | ---: | --- | --- |
| SD1.5 DMD2 GAN | `dmd2-sd15-resident-tree-v1` | 7.513676069 | 3.475052982 | `official_dmd2_min16_tree_r10` | `8d8fa55633d47cfb81bbc7a892e7248f9518763f` |
| Official Self-Forcing | `self-forcing-wan-fsdp2-r6-v1` | 50.739418775 | 30.959485489 | `official_sf_teacher_pair_r6` | `33593df3e81fa3ec10239271dd2c100facac6de1` |

Baseline runs: `official_dmd2_fsdp2_reference_r2` and
`official_sf_fsdp2_final_source_reference_r2`. DMD2's separate DDP reference is
6.113556639 s. Its benchmark uses full-size deterministic random initialization
and prepared synthetic prompts/latents. Self-Forcing uses Wan base weights.

## Public training and recovery

Both optimized workflows completed 100 cycles and 1600 rank-cycle records,
including cycle-50 checkpoint and fresh-process continuation to cycle 100.

| Workflow | Run | Per-rank updates | Framework source | Checkpoint manifest SHA256: C50 / C100 |
| --- | --- | --- | --- | --- |
| DMD2 | `framework-public-dmd2-sd15-optimized-100-r2` | G100 / C1000 | `7b84ebb43067fd7fcfca6b1621fb53c5a7430c38` | `d57e3581823798a59c7971c61d9160145455758af491a115586a44ef87dd593d` / `a9b82bda1493b0d563bb39019f100a37372420a9b1b8ed4fb4f26c06f12fdb82` |
| Self-Forcing | `framework-public-official-self-forcing-optimized-100-r4` | G100 / C500 | `b68c3c3a8bfec4e7adeb28c8b2eb598d1fa5be08` | `e14d703d6044feabf4e45ef741c7717727e8ba3231142574d334f7bf36a0b77f` / `a2c8e153f93eb2629c79d4290715bec6aa2e860238fe71aa9ae356b1d59c3183` |

Original samplers matched 32000 DMD2 stream IDs and 1600 Generator / 8000 Critic
Self-Forcing IDs, including checkpoint cursors. Both node owners cleaned up with
zero host cgroup OOM increments. Runtime report SHA256:

- DMD2: `cc08f986b1ffe4aae6754893f279c7fff4af20385dcea6c93a15788e2f8d0d99`
- Self-Forcing: `e1dbd2c9b00f023cb0ec3b92dbf3c5ac3a6def8721f8a17fe4bae3d02e0206f1`

## Current-Student export and output

Each output run restores the corresponding cycle-100 Student without retraining.

| Workflow | Output run | New inputs and output | Accepted receipt SHA256 |
| --- | --- | --- | --- |
| DMD2 | `framework-public-dmd2-export-generate-r2` | New CLIP-tokenized conditions; 512 × 512 RGB PNG | `f28efcd672aba4c20eb7c0b4d0f72540c1292efd13fc060c6496d41e8be61f5e` |
| Self-Forcing | `framework-public-sf-export-generate-r2` | New T5-encoded prompts; 81 RGB frames, 368 × 640 | `ade45ba8d93aa282852d9bf68991da64c0073fd11c588b0e83019079f16fbac2` |

Outputs were finite, PNG bytes matched decoded tensors, and both owners cleaned
up without OOM increments. Original receipts are in the framework maturation
artifact project, under `r115_official_native_output_controller` and
`r125_sf_native_output_binding` respectively.

## Implementation owners

- `training/official_profiles.py`: versioned profiles and source/run contracts.
- `training/official_optimizations/source_lock.json`: imported helper hashes.
- `training/official_optimizations/dmd2.py`: resident/DDP execution, fused
  optimizer, retained activations, classifier and cast scopes.
- `training/self_forcing_fsdp.py`: FSDP2 roles and local-shard checkpoint ABI.
- `training/official_optimizations/self_forcing.py`: R6 execution, version-bound
  replay, phase drains and transient cache release.

The public lifecycle owns complete-cycle saving, recovery and Student export.
Model-native code retains rollout, loss, precision and update clocks.
