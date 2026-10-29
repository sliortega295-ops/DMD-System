# Official optimized implementation: provenance and validation

The user-facing entry points and performance are documented in
[Official optimized training](../guides/official_optimized_training.md).

## Source mapping

| Workflow | Profile | Optimization run | Upstream revision |
|---|---|---|---|
| DMD2 SD1.5 GAN | `dmd2-sd15-resident-tree-v1` | `official_dmd2_min16_tree_r10` | `8d8fa55633d47cfb81bbc7a892e7248f9518763f` |
| Official Self-Forcing | `self-forcing-wan-fsdp2-r6-v1` | `official_sf_teacher_pair_r6` | `33593df3e81fa3ec10239271dd2c100facac6de1` |

DMD2: primary FSDP2 baseline `official_dmd2_fsdp2_reference_r2`,
7.5136760687455535s; selected 3.4750529816374183s (2.162×).
The fixed DDP reference, 6.113556639291346s, is an auxiliary comparison only.
Self-Forcing: FSDP2 baseline `official_sf_fsdp2_final_source_reference_r2`,
50.739418774843216s; selected 30.959485488943756s.

The DMD2 benchmark uses full-size deterministic random initialization and fixed
synthetic prompts/latents. Self-Forcing uses Wan base weights without the
upstream ODE initialization checkpoint. Both historical numerical results are
REVIEW; these measurements do not establish generation quality or convergence.

## Runtime and lifecycle

- `training/official_profiles.py`: profile contracts, upstream/run identity and
  source hashes; lightweight discovery.
- `training/official_optimizations/source_lock.json`: original and packaged file
  hashes plus executable-definition identities for the 20 imported helpers.
- `training/official_optimizations/dmd2.py`: actual native-cycle DDP owner,
  per-update cast/classifier scopes and observed execution census.
- `training/self_forcing_fsdp.py`: FSDP2 role placement and owner; same native
  fwdbwd and common update loop; local-shard checkpoint ABI.
- `training/official_optimizations/self_forcing.py`: complete R6 runtime,
  cycle boundaries, version checks and transient-state cleanup.
- `training/official_optimizations/bound_source.py`: source-private helper imports.

Raw upstream bodies remain source-locked. Public integration adds lifecycle
adaptations: geometry-aware host cache positions, repeatable pipeline creation,
startup-only helper canaries after resume, complete-cycle phase drains and
exception cleanup. The native training clock is independent of canary clocks.

## Public native lifecycle validation (2026-10-29)

These executions use the public `distillctl train` lifecycle, not the historical
benchmark launcher. Both use 16 A100-40GB GPUs, B1 per rank and global batch 16.

| Optimized workflow | Actual complete cycles / ranks | Updates per rank | Checkpoint and new-process continuation | Tested framework source |
|---|---|---|---|---|
| DMD2 SD1.5, full native UNet/CLIP/VAE | 100 / 16 | G100 / C1000 | Cycle 50 published; fresh process completed cycles 51–100 | `7b84ebb43067fd7fcfca6b1621fb53c5a7430c38` |
| Official Self-Forcing, Wan14B Teacher and Wan1.3B G/Fake | 100 / 16 | G100 / C500 | Cycle 50 published; fresh process completed cycles 51–100 | `b68c3c3a8bfec4e7adeb28c8b2eb598d1fa5be08` |

The DMD2 run is `framework-public-dmd2-sd15-optimized-100-r2`: all 1600 rank-cycle
records completed, both node owners cleaned up, and host cgroup OOM deltas were
zero. The original sampler also matched all 32000 independent-stream sample IDs
and both checkpoint stream cursors. The checkpoint manifest SHA256 values are
`d57e3581823798a59c7971c61d9160145455758af491a115586a44ef87dd593d` (cycle 50) and
`a9b82bda1493b0d563bb39019f100a37372420a9b1b8ed4fb4f26c06f12fdb82` (cycle 100).
The complete small runtime report has SHA256
`cc08f986b1ffe4aae6754893f279c7fff4af20385dcea6c93a15788e2f8d0d99`.

Official Self-Forcing run `framework-public-official-self-forcing-optimized-100-r4`
also completed all 1600 rank-cycle records, with both node owners cleaned up
and zero host cgroup OOM increments. The original sampler matched all 1600
Generator and 8000 Critic stream IDs and both saved stream cursors. Its
checkpoint manifest SHA256 values are
`e14d703d6044feabf4e45ef741c7717727e8ba3231142574d334f7bf36a0b77f` (cycle 50)
and `a2c8e153f93eb2629c79d4290715bec6aa2e860238fe71aa9ae356b1d59c3183`
(cycle 100). The complete small runtime report has SHA256
`e1dbd2c9b00f023cb0ec3b92dbf3c5ac3a6def8721f8a17fe4bae3d02e0206f1`.

DMD2 uses full-size deterministic random initialization and synthetic prepared
records; Self-Forcing uses existing Wan base weights, not the ODE initializer.
This validates the stated runtime scope, not pretrained generation quality or
new speed measurements. Full-size distributed export and new-input decoded
outputs remain separate validation items. The source revisions above identify
the actual executions, not every later framework commit.

## Validation levels

1. Historical native GPU measurements are those in the table above.
2. Packaged helper bytes and original executable definitions are checked on
   import; Python3.10/3.12 AST metadata differences are normalized explicitly.
3. CPU checks cover profile rejection, actual upstream helper execution,
   state/cache lifetime, exceptions, original reference updates and restart.
4. Actual public world16 runtime results are recorded above. Both optimized
   profiles completed 100 cycles with a cycle 50 new-process restart. These do not
   qualify the unmeasured reference backends or other model variants.

The versioned optimized profiles require the measured A100/world16 geometry,
update and precision contracts. Existing FP32 reference configurations continue
supporting their own admitted scope. Automatic clone-based calibration of these
opaque distributed owners is not enabled; direct training uses their explicit
profile. SDXL and W4 LiveWan do not inherit an SD1.5 or official Self-Forcing
profile merely because they share a model family.

Complete-cycle, same-topology recovery is GPU-validated for both exact
optimized executions above. Their full-scale portable generation remains a
separate validation item. The CPU and GPU status of
each path is kept separately in the integration report, rather than inferred from
another workflow's result.
