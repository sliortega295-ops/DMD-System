# Unified bounded native planner workflow

All commands use the existing `scripts/distillctl.py workloads` entrypoint.
No new trainer or alternate CUDA implementation is introduced. Profiles,
calibration, search, serialized execution and evidence remain separate gates.
The default calibration command is **source-only**; only `--execute-native`
launches through the existing two-node runner, lock and native gates.

## Capability boundary

| Workload | Shared entrypoint bindings | Calibration/search family | Current integration GPU |
|---|---|---|---|
| Wan No-GAN | 35 bounded action implementations | Existing source-qualified components; placement/chunk/shared pointwise/retention | NOT_RUN |
| Helios No-GAN | 20 action implementations | Existing FSDP/HSDP, chunk and retention components | NOT_RUN |
| Helios GAN | 4 action implementations | World16 FSDP, chunk, bounded persistent scope; fixed five-rollout group | NOT_RUN |
| H3 Grid1000 | 6 macro implementations per explicit request | CPU-staged or CUDA cache, depth/tile; joined native seven-stage calibration | NOT_RUN |
| LiveWan | 5 native implementations | Existing coupled FSDP/pipeline commit macros, not independent G/Score overlap | NOT_RUN |

Counts describe legal bindings, **not** measured candidates or an unrestricted
M/L/Q/K cross-product. Existing legacy commands remain valid. The open-workload
Program/Role path is separate: this does not port five original algorithms to
generic factories or claim arbitrary shapes/topologies/batch/update ratios.

## H3 example (run beside the persistent receipt files)

Use a clean committed checkout on both target nodes and a freshly verified
assets config, including its matching small-resolution prompt cache, Grid1000
manifest, Diffusers source, FA3 kernel and pinned LightX2V. Keep large model
payloads and raw observations on remote persistent storage. `HEAD`, `OUT`, and
`ASSETS` below are explicit operator-supplied paths/identity, not inferred sites.

```bash
python scripts/distillctl.py workloads candidates \
  --workload minimax_h3_grid1000_best \
  --request configs/distillschedule/requests/minimax_h3_480x832_grid1000.json

# Optional Nsight diagnostic import, not a cost model and never GPU MFU:
python scripts/distillctl.py workloads profile --input "$PROFILE_INPUT" --output "$PROFILE_OUTPUT"

# Omit --execute-native first to inspect the immutable intent without launching.
python scripts/distillctl.py workloads calibrate \
  --workload minimax_h3_grid1000_best --expected-head "$HEAD" \
  --run-id h3-v3-cal --output "$OUT/collection" --assets-config "$ASSETS" \
  --candidate joint-h3v3-h480-w832-f124-cpu_staged_trim-d2-n128 \
              joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128 \
  --execute-native

python scripts/distillctl.py workloads assemble-calibration \
  --workload minimax_h3_grid1000_best --expected-head "$HEAD" \
  --result "$OUT/collection/calibration-0/result.json" "$OUT/collection/calibration-1/result.json" \
  --output "$OUT/library"

python scripts/distillctl.py workloads search --strategy model-based \
  --workload minimax_h3_grid1000_best --problem "$OUT/library/problem.json" \
  --cost-model "$OUT/library/cost-model.json" --search-algorithm hierarchical \
  --run-id h3-v3-selected --assets-config "$ASSETS" \
  --emit-native-plan "$OUT/selected-plan.json" --output "$OUT/search.json"

python scripts/distillctl.py workloads run --workload minimax_h3_grid1000_best \
  --search-plan "$OUT/selected-plan.json" --dry-run
python scripts/distillctl.py workloads run --workload minimax_h3_grid1000_best \
  --search-plan "$OUT/selected-plan.json" --run-id h3-v3-confirm \
  --coordinator-output "$OUT/confirmation"
```

An emitted plan uses **correctness** mode, not timing. The minimal wiring
acceptance above needs up to two fresh correctness runs, two instrumented
calibrations and one selected correctness replay, all serial on 16 GPUs.
Calibration ranges are not unprofiled timing. For a matched diagnostic
performance campaign use existing model-based `--execute-native`, a fresh
`--validation-dir`, and `--max-trials 3` (matched control, incumbent,
independent confirmation); additional current-source correctness gates also
consume resources. The owning runner must budget/authorize these separately.

Do not reuse the source-only intent directory when switching to execution.
Use a new output/run ID. Assembly must run where the returned native
`layout_fragment`, `operation_costs`, rank and resource files actually exist;
it does not download raw files or silently copy metadata from another source.

## Required receipts

- Exact source, workload request/semantic, assets, hardware and environment
  identities; both node preflight/launcher manifests.
- 16 unique ranks, both native cycles, G2/F10/EMA0, unchanged census/Grid1000,
  finite losses, same-rank sampled state and native RNG comparison.
- Actual FA3 registration equals selected storage/trim/depth; CUDA CPU-related
  counters and allocator trim stay zero. Environment echo alone is insufficient.
- Memory/resource samples include setup/lifetime high-water marks, no CUDA or
  cgroup OOM, clean owned teardown, no unrelated workloads/preemption.
- Calibration samples are rederived from hashed native observations; changed
  files, wrong shape, cost substitution and missing ranks reject before launch.
- Record prediction error and numerical status honestly. CPU test outputs or
  historical 480x832 cache results are not current-HEAD GPU validation.
# H3 child toolchain environment

The resolved H3 child environment explicitly exports `PATH=/usr/bin:/bin`.
It does not inherit the controller's PATH, prepend an environment/CUDA bin
directory, or install a compiler. A clean allowlisted child may otherwise find
GCC via Python's executable-search fallback but leave GCC unable to find `ld`.
Serialized plans with a missing or changed PATH fail the normal plan/local
validation. Canonical profile files and training semantics are unchanged.

Before a cluster retry, qualify on each node using that exact selected child
environment: locate GCC and `ld`, compile/link a minimal CPU-only C program and
execute it. Record tool paths/hashes and return codes. Imports alone do not
exercise this JIT-toolchain dependency; a CPU linker probe is not CUDA/Triton
or native workload validation. Preserve the original failed run and generate
a fresh source-bound plan, run ID, qualification and sealed manifest.

## Allocated-node metadata transport

Node entries in the explicit asset config may set
`"rpc_transport": "argv-json-tty-v1"` when an SSH gateway cannot forward
stdin/EOF reliably. The default remains `stdin`. This option changes only
CPU metadata transport: the compressed JSON request is carried in a bounded
argument, the worker still reads the same request from a private input stream,
and exactly one framed JSON result is required. Raw stdout/stderr and the
request remain in the RPC journal; missing/duplicate frames fail closed.

This is not a GPU retry switch. An interrupted R9 owner still requires both
existing owner streams to reach a known terminal state before any new SSH.
Do not amend a failed immutable run directory to switch transports; use a new
run ID and requalify its source-bound manifest.

## H100 native bridge

`distillctl workloads hardware` connects the pinned eight-H100 Wan runtime to
the same public problem/cost/search implementation. It keeps physical workers
and logical B1 samples separate rather than changing global16 to global8.
The initial domain and unsupported workloads are explicit in
[hardware-native planning](../workloads/hardware_native_planning.md).
Its CPU tests are not evidence of H100 GPU execution, and an existing GPU
reservation must not be preempted to validate a new plan.
