# Official optimized implementation: provenance and validation

The user-facing entry points and performance are documented in
[Official optimized training](../guides/official_optimized_training.md).

## Source mapping

| Workflow | Profile | Optimization run | Upstream revision |
|---|---|---|---|
| DMD2 SD1.5 GAN | `dmd2-sd15-resident-tree-v1` | `official_dmd2_min16_tree_r10` | `8d8fa55633d47cfb81bbc7a892e7248f9518763f` |
| Official Self-Forcing | `self-forcing-wan-fsdp2-r6-v1` | `official_sf_teacher_pair_r6` | `33593df3e81fa3ec10239271dd2c100facac6de1` |

DMD2: FSDP2 baseline `official_dmd2_fsdp2_reference_r2`, 7.5136760687455535s;
fixed DDP baseline 6.113556639291346s; selected 3.4750529816374183s.
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

## Validation levels

1. Historical native GPU measurements are those in the table above.
2. Packaged helper bytes and original executable definitions are checked on
   import; Python3.10/3.12 AST metadata differences are normalized explicitly.
3. CPU checks cover profile rejection, actual upstream helper execution,
   state/cache lifetime, exceptions, original reference updates and restart.
4. This newly integrated world16 optimized public path has not been executed on
   GPUs in this integration task. No new GPU experiment was started.

The versioned optimized profiles require the measured A100/world16 geometry,
update and precision contracts. Existing FP32 reference configurations continue
supporting their own admitted scope. Automatic clone-based calibration of these
opaque distributed owners is not enabled; direct training uses their explicit
profile. SDXL and W4 LiveWan do not inherit an SD1.5 or official Self-Forcing
profile merely because they share a model family.

Complete-cycle, same-topology checkpoints are supported by the source path;
new optimized distributed recovery and portable generation remain GPU
validation items. The CPU and GPU status of each path is kept separately in the
integration report, rather than inferred from another workflow's result.
