# PR #26 Helios source recovery and web-agent handoff

The missing Helios source is now checked into this PR under
`third_party/helios_compat/`. A GitHub-only coding agent can read it directly;
no local files, SSH connection, private filesystem path or source archive is
needed to continue integration.

This is a source-recovery change. The five-workload registry/CLI migration remains
the scope of PR #26, and GPU/server validation at its final integration HEAD is
still `NOT_RUN_ON_INTEGRATION_BRANCH`. Historical performance is not remeasured.

## Recovered bytes

Recovered from clean compatibility commit
`8b15dede4449ea0363600ab82a0db719d054a732`, retaining its original Git blobs.
The original tree and its commit were in a server/local Git repository; publishing
the DMD-System helper branches had not published this separate Helios tree.

| File under `third_party/helios_compat/source/` | SHA-256 |
|---|---|
| `helios/modules/transformer_helios.py` | `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886` |
| `helios/utils/utils_helios_post.py` | `e3ff9e5f8bf025ee740e436bc0c209c972f1e56eb0236b968fe6bd8284107004` |
| `helios/utils/utils_helios_base.py` | `ebd2dcb3bff01d807ae2a27b141a0e3ba8ea929d2c63e6440faab6379e69ea4a` |
| `helios/modules/helios_kernels/attention_dispatch.py` | `090e0cf5894eb3d57d6bbb609b5346ba909ea3a9422be8e3abc49f89518f6daf` |
| `helios/scheduler/scheduling_helios.py` | `30f5dea16ec0f9b247ef935044ca60811473c479fa5302dae89fa2a3553e929d` |

The complete snapshot includes 83 files / 1,799,673 bytes, with a SHA-256 and
original Git blob ID for each. Source inventory digest:
`0d21cad0e2bb6bea7a5bf38cc467f3dcb08eb2986f4746765d4315ff00fe14d4`.

## Why this is the needed Transformer

The following GitHub consumers independently require the exact `5d3ffe...` bytes:

| Consumer source commit | Required source check |
|---|---|
| Wan `631bc082c819d0728059e96663cdf31cd66d5b0b` | `scripts/distillschedule/launch_wan13b_fair_world16_node.sh` |
| Helios No-GAN `9c53d8e75b0981a3a9299e4f3bec2819ea105013` | `src/distillschedule/runtime/helios_modulated_norm_fusion.py` |
| Helios GAN C25 source `a78827099c89795416f9ac6c77284dd5bacb0383` | `src/distillschedule/runtime/helios_history_current_fusion.py` |

The matching performance-code commits remain, respectively,
`67b2880901abece947732ba5de45a5c2685ab274`,
`ad9e16b83800463c069e87b562f2d07a41bbeab3`, and
`d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d`.
Content identity alone does not validate a model environment or reproduce timings.

## Dependency ownership

1. Use `third_party/helios_compat/source` for the `helios` package.
2. Keep the DMD-System repository root and `src` on the integration import path.
3. Core Helios imports `load_precast_checkpoint_into_meta_model` from the existing
   `runtime.precast_checkpoint`, and `swap_wave_ranges` / `stable_block_noise`
   from the existing `runtime.official_baseline`. These exports are checked by
   the CPU verifier; no new active `runtime` package has been introduced.
4. The original `train_helios.py` has more legacy imports (including
   `deepspeed_phase_state`, `chunked_lora`, `safe_tiled_ffn` and legacy baseline
   options). Its full old runtime is preserved under `reference/runtime/` for
   inspection. Do not put that directory on the integration worker's import path
   or launch the archived entrypoint in place of the typed workload worker.
5. Attention kernel binaries and the Python environment remain external assets.
   Reuse their measured identities instead of installing from the archived,
   partially mutable `reference/requirements.txt`.

## Verification

```bash
python scripts/distillschedule/verify_helios_compat_source.py
python -m unittest discover -s tests -p test_refactor_helios_compat_source.py -v
```

The dedicated `helios-compat-source` CI executes these on Python 3.10 and 3.11
without Torch or GPU dependencies. An optional `--output <new-path>` saves a
machine-readable receipt and refuses to overwrite one that already exists.

Local source recovery verified:

- original bytes, Git blob identities, source totals and Python syntax: PASS;
- static core and historical local import-module graphs: PASS;
- six normal/fail-closed CPU tests: PASS;
- actual model imports / GPU / Full5 / performance reproduction: NOT_RUN.

The source snapshot intentionally preserves one original trailing-whitespace line
(`source/helios/utils/utils_helios_base.py:295`). `git diff --check` reports that
line; changing it would invalidate the recovered helper hash. New integration
code and documentation have no whitespace errors.

This does not certify arbitrary third-party package versions or dynamic imports.
Only the Transformer's content pin was independently matched against the listed
consumer checks. Helpers are from the same recovered commit, but matching their
hashes to every performance run's complete runtime receipt remains unverified.

## Continue PR #26 from GitHub

Fetch the latest `agent/integrate-five-workload-winners-20260905` before applying
any unpushed web-workspace changes. Preserve those changes; do not reset them away.

1. Run the source verifier and read the manifest and ownership rules above.
2. Continue the five canonical profiles, common CLI and source-to-integration map
   requested in the original task. The Helios source-content blocker is now
   addressable entirely from this branch.
3. Wire the real source root into the Wan / Helios workers. Resolve asset paths
   separately; preserve kernel-installation SHA checks. Verify import resolution
   against this packaged source rather than whichever `helios` is globally installed.
4. Resolve the GAN C25 effective Transformer pin explicitly. Its historical branch
   launcher default is `776a26...`, but its C25 installer requires `5d3ffe...`.
   Never silently accept a different source or remove the check.
5. Continue to lock H3 `activation_policy=none` while retaining its separate FA3
   output/LSE `cpu_staged` replay cache. Preserve C25 and C24 feature combinations;
   this source recovery adds no candidate optimization.
6. Commit the actual profile/CLI/H3 migration code, run the final integration CPU
   tests and CI, and keep GPU revalidation `NOT_RUN` until it is actually performed.
   Keep PR #26 Draft and do not merge main.

The restored source and CPU integrity checks are ready for GitHub-based
integration work. They do not make the unfinished five-workload integration a
server-validated release.
