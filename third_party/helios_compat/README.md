# Helios compatibility source

`source/helios/` contains the model, scheduler, kernel dispatch and DMD helpers
used by the native Wan and Helios recipes. Its `runtime.*` imports resolve to
DMD-System's active root `runtime/` package.

## Source and license

- Original project: <https://github.com/PKU-YuanGroup/Helios>
- Compatibility source commit: `8b15dede4449ea0363600ab82a0db719d054a732`
- Transformer SHA-256: `5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886`
- [source_manifest.json](source_manifest.json) binds the 44 retained files.
- [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE) retain attribution and changes.

`reference/train_helios.py` is retained for original update/state ownership.
The unused historical `reference/runtime/` copy is available at the manifest's
`historical_reference` commit and in the local archive. It is not a second
runtime to maintain or add to `PYTHONPATH`.

## Source check and launch environment

```sh
python scripts/distillschedule/verify_helios_compat_source.py
export HELIOS_RUNTIME_ROOT="$PWD/third_party/helios_compat/source"
export PYTHONPATH="$PWD/src:$PWD:$HELIOS_RUNTIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

The source check verifies file hashes, Git blob identities, Python syntax and
imports against the active compatibility runtime. Use the selected workload
recipe and its dependency environment for training. Model weights, prepared
inputs and compiled attention backends are supplied separately.
