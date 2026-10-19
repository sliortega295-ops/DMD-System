from pathlib import Path
import subprocess
R=ROOT
def write(p,s):
    target=R/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s.strip()+'\n')
write('docs/history/readme-v0.2.md',subprocess.check_output(['git','show','HEAD:README.md'],cwd=R,text=True))
write('README.md',r'''
# DMD-System

**DMD-System** provides execution planning and runtime support for compound diffusion-distillation training. Model calls, training dependencies, parameter versions and intermediate-state lifetimes are described separately from device placement and execution policy.

The Python package and CLI retain their stable names: **`distillschedule`** and **`distillctl`**.

> **Research status.** The five canonical recipes below are the retained optimized implementations, not five newly GPU-validated release results. Their original correctness, diagnostic-timing and `NOT_RUN` labels are preserved. General workload planning is implemented for bounded, explicitly admitted capabilities; it is not automatic translation of arbitrary training Python. See [current scope](docs/architecture/repository_layout.md#execution-boundaries) before running large models.

## Install and inspect

Use a source checkout for the canonical model recipes, which require repository configuration files and externally supplied assets. Lightweight inspection does not import model weights or allocate GPUs.

```bash
python -m pip install -e '.[test]'
distillctl --help
distillctl workloads list
distillctl workloads explain --workload helios14b_nogan_best
```

Native Torch execution additionally requires `pip install -e '.[native]'` and a suitable PyTorch build for the target machine. Workload-specific backend dependencies and checkpoints are **not** installed or downloaded implicitly.

## Canonical optimized workloads

Exactly one source-pinned profile is exposed for each workload under [`configs/workloads/`](configs/workloads/). Historic candidate names and benchmark variants are not additional release recipes.

| Workload | Stable profile ID | Retained implementation |
|---|---|---|
| Wan 1.3B DMD | `wan13b_nogan_resident_best` | Resident models; asynchronous LoRA reduction; fused epilogue |
| Helios 14B, No-GAN | `helios14b_nogan_best` | Node-local HSDP; local FA2; selective checkpoint replay |
| Helios 14B, GAN-on | `helios14b_gan_best` | World16 FSDP2; C25 implementation |
| MiniMax-H3 | `minimax_h3_grid1000_best` | Shared-backbone LoRA; staged attention replay; **Grid1000 approximation** |
| LiveWan 14B → 1.3B | `livewan_crossscale_pipeline_best` | C34 actual-B16 cross-scale pipeline; canonical plan is self-contained |

These names select existing recipes; they do not imply that a recipe is optimal for a new shape or device. The generic planner is a separate entry point, not a lookup of these five winners. H3's approximation and LiveWan's numerical `REVIEW` must not be silently changed into exact-equivalence claims.

Populate the matching [`assets_examples/`](configs/workloads/assets_examples/) file with asset locations and fingerprints. First inspect the intended launch:

```bash
distillctl workloads run --workload helios14b_nogan_best \
  --assets-config /path/to/helios-assets.json \
  --run-id helios-check --mode correctness --dry-run
```

Removing `--dry-run` requests actual execution and may launch multi-node workers. Keep the original training contract, required assets and topology checks. See [workload runbooks](docs/workloads/BEST_KNOWN_WORKLOADS.md) and [cluster revalidation](docs/workloads/SERVER_REVALIDATION_CHECKLIST.md). Diagnostic timing is not formal speedup evidence.

## New workloads and automatic planning

A new workload supplies a semantic factory, bindings, explicit state and inputs. It does not provide preselected GPU layouts or winning recipes. Use the shared profiler, cost model, frontier search and executor:

```bash
distillctl workloads native program --help
distillctl workloads native program hardware --output /tmp/dmd-hardware.json
```

The [small CPU example](docs/workloads/open_workload_program.md) runs real PyTorch operations but is not a pretrained video benchmark. [Distributed programs](docs/workloads/open_workload_distributed.md) and [role pools](docs/workloads/open_role_pools.md) specify supported sharding, transfer and parameter-version contracts. Model construction and training require explicit `--execute` on these native entry points.

Component-cost role planning currently admits replicated/resident pools and a bounded producer schedule with fixed Score receive order. Sharding/offload options use their admitted execution and measurement paths; all combinations are not interchangeable. See [component scope](docs/status/ROLE_COMPONENT_ASYMMETRY.md) and [dispatch scope](docs/status/ROLE_PRODUCER_DISPATCH.md). CPU/Gloo checks do not establish A100/H100 CUDA/NCCL performance or full five-model training quality.

## Repository organization

```text
src/distillschedule/       public package: IR, profiling, search, bindings, runtime
src/runtime/               private compatibility namespace for audited model code
configs/workloads/         five immutable optimized profiles + asset examples
configs/distillschedule/   shared hardware, protocol and backend plan fragments
scripts/                   CLI and canonical launch/verification dependencies
examples/                  small conforming workload examples
tools/                     asset preparation, profiling and validation utilities
benchmarks/                baselines, ablation configurations, historical fixtures
tests/                     unit, contract and distributed regression tests
docs/                      usage, architecture, runbooks and evidence boundaries
third_party/               pinned upstream sources and licenses
```

Public code does not become a second copy of the model implementation. `src/runtime/` preserves the existing `runtime.*` imports required by the source-audited Helios adapter; it is now packaged rather than depending on the checkout working directory. Alternatives required by the planner or correctness comparisons remain available, but old one-off entry points are not mixed into normal launch instructions.

[Directory guide and migration map](docs/architecture/repository_layout.md) · [Documentation index](docs/README.md) · [Contribution rules](CONTRIBUTING.md)

## Tests and provenance

```bash
python scripts/distillschedule/check_best_known_integration.py
python -m pytest -q tests/test_repository_layout.py tests/test_refactor_best_known*.py
python -m pytest -q tests/test_open_*.py
```

The first command checks source fingerprints, local import closure, upstream pins and canonical profiles without launching GPUs. Additional distributed tests require the native dependencies. Run the supported Python/Torch matrix in CI; byte/AST provenance checks are not permission to weaken numerical tolerances.

Archived attempts are retained under benchmarks/history or in Git history. They are neither inputs to a new performance claim nor evidence that an unfinished path works. Keep checkpoints, datasets, raw traces and credentials outside Git. Every reported result must bind source, assets, shape, hardware and validation status.
''')
write('configs/workloads/README.md',r'''
# Canonical optimized profiles

This directory contains exactly five `*_best.json` profiles and their byte-level `registry.lock.json`. `distillctl workloads list` is the public index. `assets_examples/` contains host/asset templates, not new training variants.

The cleanup changes file locations, not their contents, training semantics, source commits or validation labels. Historical C24–C33 LiveWan configurations live in `benchmarks/ablations/livewan/`; the active C34 typed plan in `configs/distillschedule/plans/` is flattened with the same resolved-plan digest.

Do not hand-edit a locked profile to hide a failing gate, change a model, or reclassify diagnostic timing. Follow the profile registry and evidence contract when adopting a newly validated replacement. Different hardware/shape requests belong to the automatic planner, not to manually renaming a recorded winner.

From the repository root:

```bash
distillctl workloads list
distillctl workloads explain --workload livewan_crossscale_pipeline_best
```
''')
write('tools/README.md',r'''
# Developer utilities

`assets/` prepares input traces, checkpoint conversions and asset manifests. `profiling/` contains trace analysis and plotting. `validation/` contains standalone receipt comparisons, capacity checks and summarizers.

These tools are invoked explicitly; they are not public training recipes. Run them from the repository root with their `--help` before use. They may require model-specific dependencies or private assets. The CLI and canonical launcher dependency closure remain under `scripts/` so source-pinned recipes keep their validated entry points.

All relocations, including old command paths, are recorded in `docs/architecture/repository_cleanup.json`. Do not duplicate a moved implementation back into `scripts/`.
''')
write('benchmarks/README.md',r'''
# Reproducibility benchmarks

This directory is separate from the canonical runtime launch surface:

- `baselines/`: reference executors, trial controllers and historical comparison entry points.
- `ablations/livewan/`: incremental C-series configurations, retained for controlled comparisons and plan-composition regression tests.
- `kernels/`: standalone kernel/capacity canaries.
- `fixtures/historical/`: immutable small historical receipts and numeric summaries, **not** fresh execution results.

Only `configs/workloads/*_best.json` defines the retained optimized recipes. A baseline or ablation is not a second production version. Historical records preserve failed gates and `REVIEW`/`NOT_RUN` labels; directory cleanup never makes a performance result claimable.

Some finite search/backward-compatibility paths still resolve registered ablation implementations from this directory. That is intentional: an automatic planner needs legal alternatives, and a controlled benchmark needs its reference. Do not delete these as unused files based on their old iteration names.

Run commands from the repository root. Vendor pins, declared training semantics and matched workloads must remain unchanged between reference and candidate arms. Full model/GPU reruns require the assets and hardware in the corresponding runbook.
''')
write('scripts/README.md',r'''
# Stable entry points

Use `distillctl` (or `python scripts/distillctl.py`) rather than selecting an experiment-numbered script. The files under `distillschedule/` form the dependency closure of the five canonical launchers and current public package/plan entry points; they also include required parity and evidence helpers.

Asset preparation, ad-hoc profiling and standalone validation now live under `tools/`. Historical comparison executors are under `benchmarks/baselines/`. The relocation manifest is `docs/architecture/repository_cleanup.json`.

The retained scripts may still have `probe_` or `world16` names because source-pinned workload receipts and canonical launchers depend on them. Names do not imply unreferenced code. Changing those entry points is a separate validated runtime migration, not an untested file deletion.
''')
write('docs/README.md',r'''
# Documentation

## Start and extend

- [Repository guide](../README.md): install, five canonical recipes, automatic planning.
- [Directory architecture](architecture/repository_layout.md): public/runtime/benchmark boundaries and old-path migration.
- [Canonical workloads](workloads/BEST_KNOWN_WORKLOADS.md): source-pinned execution and runbooks.
- [New workload ABI](workloads/open_workload_program.md): semantic factory and call/state contracts.
- [Distributed programs](workloads/open_workload_distributed.md) and [role pools](workloads/open_role_pools.md): capabilities and restrictions.

## Correctness and measured scope

- [Source import map](workloads/source_import_map.json): upstream identities and current integrated file hashes.
- [Source migration](workloads/BEST_KNOWN_SOURCE_MAP.md) and [server revalidation](workloads/SERVER_REVALIDATION_CHECKLIST.md).
- [Component-cost planning](status/ROLE_COMPONENT_ASYMMETRY.md) and [native planned dispatch](status/ROLE_PRODUCER_DISPATCH.md): what is implemented and what remains unvalidated.
- [Frozen Helios world16 contract](contracts/routea_full5_world16.md): unchanged logical training work.
- [Historical Route-A status](status/routea_world16.md) and [No-GAN suite](runbooks/baai_no_gan_suite.md): preserve earlier diagnostic/Not Run boundaries; these are not the default public quick start.

## History and benchmarks

The [prior README](history/readme-v0.2.md) and `history/experiments/` retain earlier migration notes. [Benchmarks](../benchmarks/README.md) separates baselines, ablations and historical receipts from current canonical launchers. Never use an old experiment note as an authoritative current command without checking the [relocation map](architecture/repository_cleanup.json).
''')
write('docs/architecture/repository_layout.md',r'''
# Repository layout and canonical workload boundary

DMD-System uses a single `src/` package tree. Following the separation used by ReaLHF (API/implementation/search/system) and verl (models/workers/trainer/utils), model algorithms, execution mechanisms, recipes and developer experiments have distinct locations. This is an organization change, not an adoption of either framework's training algorithm.

## Ownership

| Location | Owns | Must not contain |
|---|---|---|
| `src/distillschedule/core`, `workloads` | IR, training protocol, state/version contracts | Hardware-specific winning recipes in algorithm definitions |
| `src/distillschedule/open_workload`, `adapters`, `backends`, `native` | Model/call bindings, execution backends | Hidden loss changes or automatic model downloads |
| `src/distillschedule/profiling`, `search`, `strategies` | Measurements, legal candidates, shared planning | Unmeasured GPU speedup claims |
| `src/distillschedule/runtime`, `evidence` | Dispatch, state lifecycle, source-bound result checks | Experiment-number-dependent algorithm switches |
| `src/runtime` | Audited compatibility mechanisms used by vendor adapters | A second public API or eager optional-dependency imports |
| `configs/workloads` | Exactly five locked optimized profiles | Historical losing candidates |
| `scripts` | Public CLI and canonical launcher dependency closure | Standalone trial/plot/conversion clutter |
| `tools` | Explicit development utilities | Default training entry points |
| `benchmarks` | References, ablations, historical receipts | Reclassification of failed/diagnostic results |

## What was retained

The five canonical profile JSON files, asset examples and registry lock were relocated byte-for-byte. The LiveWan C34 active template is self-contained; its **resolved JSON and fingerprint are identical** to the original include chain. The original raw C34 file and its predecessors remain as ablation fixtures.

The dependency closure, not filename age, decides which launchers remain. A `probe_` helper referenced by a final launcher is still necessary until a separately validated API extraction replaces it. Sharding/residency alternatives needed by the automatic planner are not deleted just because one historical experiment preferred another alternative.

`runtime.*` remains the import name used by the pinned Helios sources. Moving it to `src/runtime/` and packaging it fixes checkout-directory-dependent imports without editing the vendor or changing model math. Canonical recipes still require a source checkout; this cleanup does not claim a fully self-contained model-serving wheel.

## Execution boundaries

General programs supply explicit data/state bindings and dependency contracts. Implemented backend combinations must pass capability checks before planning; the system does not infer arbitrary Python side effects. Replicated/resident component planning can choose bounded producer-forward/FIFO-backward interleavings, while Score receive order remains fixed. Sharding/offload paths keep their own valid measurement and lowering restrictions.

CPU and small multi-process tests are source/runtime checks, not A100/H100 validation of the five pretrained workloads. Preserve the source runbooks and current `NOT_RUN`, `REVIEW` and diagnostic-only records. Do not advertise unrestricted M/L/Q planning, production restart or new speedups from this cleanup.

## Migration and recovery

[`repository_cleanup.json`](repository_cleanup.json) records the source commit, every old/new path and original file hash, canonical-profile hashes and the unchanged LiveWan resolved digest. [`../workloads/source_import_map.json`](../workloads/source_import_map.json) preserves upstream source origins and records layout-adjusted target hashes; `layout_origin` retains the pre-cleanup integrated identity.

History is not rewritten. Recover an old file with `git show <base-commit>:<old-path>`, or use the recorded destination when reproducing a benchmark. No force push, branch deletion or main merge is part of the cleanup.

## Validation

Run `python scripts/distillschedule/check_best_known_integration.py`, the layout regression tests, the normal typed tests and relevant native/distributed suites. Build a wheel and import both `distillschedule` and `runtime` from outside the checkout to test packaging separately. Canonical profile validation must remain model/GPU-free.

Source/layout failures must not be fixed by removing assertions, changing numeric tolerance, or dropping comparison arms. A pre-existing failure must be demonstrated at the original commit and reported separately.

References: [ReaLHF source tree](https://github.com/openpsi-project/ReaLHF/tree/main/realhf), [verl source tree](https://github.com/verl-project/verl/tree/main/verl).
''')
write('.editorconfig','''
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true

[*.py]
indent_style = space
indent_size = 4

[*.{yml,yaml}]
indent_style = space
indent_size = 2
''')
p=R/'src/distillschedule/cli.py';s=p.read_text();s=s.replace('''        if (candidate / "train_helios.py").is_file() and (candidate / "runtime").is_dir():
            return candidate''','''        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "src" / "distillschedule").is_dir()):
            return candidate
        if (candidate / "train_helios.py").is_file() and (candidate / "runtime").is_dir():
            return candidate''')
s=s.replace('''    parser = argparse.ArgumentParser(prog="distillctl")''','''    parser = argparse.ArgumentParser(
        prog="distillctl",
        epilog="Canonical recipes: distillctl workloads --help. "
               "New workloads: distillctl workloads native program --help.",
    )''')
s=s.replace('''    validate = subparsers.add_parser("validate"''','''    subparsers.add_parser("workloads", help="canonical recipes and native workload planning")

    validate = subparsers.add_parser("validate"''')
p.write_text(s)
p=R/'src/distillschedule/workloads/best_known/cli.py';s=p.read_text().replace('''argparse.ArgumentParser(prog="distillctl workloads")''','''argparse.ArgumentParser(
        prog="distillctl workloads",
        epilog="For new semantic workloads: distillctl workloads native program --help. "
               "Registered comparisons: distillctl workloads search --help.",
    )''');p.write_text(s)
p=R/'.gitignore'; s=p.read_text();s+='\n# Python packaging output\n*.egg-info/\nbuild/\ndist/\n';p.write_text(s)
