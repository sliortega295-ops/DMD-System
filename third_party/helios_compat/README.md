# Recovered Helios A100 compatibility source

This directory makes the previously server-local Helios source available to the
five-workload integration in PR #26. The Transformer bytes match the SHA required
by the Wan Resident, Helios No-GAN fusion, and Helios GAN C25 fusion consumers.
It is a modified Helios compatibility snapshot, not an unmodified official release.

## Source identity

- Original project: <https://github.com/PKU-YuanGroup/Helios>
- Recovered compatibility Git commit: `8b15dede4449ea0363600ab82a0db719d054a732`
- Transformer SHA-256: `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`
- Snapshot: 83 original files, 1,799,673 bytes (about 1.72 MiB).
- Every recovered file is byte-identical to its source Git blob; see
  [source_manifest.json](source_manifest.json).
- The compatibility Git commit came from a local/server repository, and was not
  previously fetchable from GitHub. The checked-in files now provide the source;
  consumers need not fetch that old repository or find a local worktree.
- The precise official upstream base is not established by this synthetic
  compatibility history. The manifest says `UNRESOLVED` instead of inventing a pin.

## Two deliberately separate import roots

`source/helios/` contains the model, scheduler, kernel dispatch, DMD helpers and
the rest of the original `helios` Python package. It is the source root to use
for the integration workers. Its imports of `runtime.official_baseline` and
`runtime.precast_checkpoint` resolve to DMD-System's existing root `runtime/`.

`reference/` preserves `train_helios.py`, its original `runtime/` package and the
historical requirements file. It is for source inspection and reconstruction,
not the canonical five-workload entrypoint. Do **not** add `reference/` to the
integration worker's `PYTHONPATH`: its old `runtime` would shadow the current
DMD-System compatibility layer. No default launcher selects it.

For the integration work, from the DMD-System checkout:

```bash
python scripts/distillschedule/verify_helios_compat_source.py

export DMD_SYSTEM_ROOT="$PWD"
export HELIOS_RUNTIME_ROOT="$DMD_SYSTEM_ROOT/third_party/helios_compat/source"
export PYTHONPATH="$DMD_SYSTEM_ROOT/src:$DMD_SYSTEM_ROOT:$HELIOS_RUNTIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Use the selected workload worker after resolving its profile and assets. The
commands above verify source integrity and set an import path; they do not install
dependencies, choose an attention backend, launch a model or establish performance.
Do not invoke the archived training entrypoint as a replacement for a winner worker.

## Verification and limits

The verifier uses only the Python standard library. It checks SHA-256, original
Git blob IDs, the inventory total, Python syntax, local module resolution and the
runtime exports used by `source/helios`. It rejects altered/missing source and
unregistered files in the source roots. Six CPU tests exercise those checks.

Real Torch/Diffusers/PEFT/FlashAttention imports and GPU execution are **NOT_RUN**
for this source-recovery change. Checkpoints, trace bundles, pre-cast weights,
Grid tables, compiled FlashAttention extensions and the validated Python
environment remain separately versioned assets. The historical requirements file
includes mutable requirements and is **not** a dependency lock or an instruction
to reinstall the environment.

Only the Transformer has been independently matched to the published winner's
explicit content pin in this recovery. The other files are recovered from the
same clean compatibility commit and have individual hashes; matching each helper
against every historical run's runtime receipt is still a separate audit.

The GAN source branch's launcher default SHA (`776a26...`) differs from the C25
history/current fusion's required SHA (`5d3ffe...`). Resolve the measured C25
effective setting explicitly; do not remove the source check, silently choose the
launcher default, or modify this recovered file to satisfy a different default.

See [the integration handoff](../../docs/workloads/HELIOS_SOURCE_RECOVERY_20260905.md)
for the GitHub consumer pointers and remaining work.

## License and changes

The original Apache-2.0 license is in [LICENSE.txt](LICENSE.txt). Original source
copyright notices are retained. [NOTICE](NOTICE) describes the compatibility
modifications and this byte-preserving relocation.
