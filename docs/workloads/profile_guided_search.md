# Profile-guided registered-plan search (v1)

This is an opt-in extension of `distillctl workloads`, not a parallel training
runtime. The five canonical profiles and `registry.lock.json` are unchanged.
Historical winner times are not measurements of this branch.

## Scope and architecture

```
native receipt + per-device SQLite + nsys-ai doctor
    -> normalized profile observations
    -> frozen workload contract + registered applicable recipes
    -> bounded, resumable selection (beam width 2)
    -> existing profile compiler / dispatcher / workload worker
    -> native correctness gate, unprofiled two-cycle measurement
    -> measured ranking + independent winner confirmation
    -> source-bound execution plan (not a new canonical default)
```

The objective is cycle-2 latency. Cycle 1 is warm-up in the same process and
initialization; it is neither averaged with cycle 2 nor replaced with a profiled
time. The native workload owns timing boundaries, trace/sample/version/call
validation and numerical thresholds. Metadata compilation does not import Torch
or start a GPU worker.

| Canonical workload | Registered search space in v1 | GPU execution adapter |
|---|---|---|
| Wan1.3B No-GAN | FSDP reference; Resident recipes; local-state/bucket/kernel variants | Four recipes: FSDP, Resident async Tree, chunk2048, LoRA epilogue |
| LiveWan cross-scale | FSDP reference; existing C24--C34 typed pipeline plans | Native trajectory and two-cycle adapters |
| Helios No-GAN | Canonical reference | CPU registration only; no new GPU search |
| Helios GAN | Locked C25 reference | CPU registration only; no new GPU search |
| MiniMax-H3 | Grid1000 canonical reference | CPU registration only; approximate sigma semantics remain explicit |

`executable` means an adapter exists, **not** that this integration HEAD has
passed a GPU experiment. Each recipe records `NOT_RUN_ON_SEARCH_BRANCH` until
source-bound receipts prove otherwise. Unintegrated recipes fail before launch.

Wan's FSDP -> Resident Tree edge changes a registered bundle (placement and
gradient reduction implementation); it is **not a pure placement ablation**.
The subsequent chunk and LoRA-epilogue edges change one main factor. LiveWan's
initial FSDP -> C24 edge is likewise a complete pipeline recipe. C26/C27 are
siblings of C28, not erroneously inherited into every later recipe.

## What guides selection

The importer uses exact per-rank PID/device/NVTX windows, named SQLite columns,
read-only connections and a 300-second/node default query deadline (max 900).
Each SQLite is SHA256-bound to a hashed `nsys-ai doctor` report and must cover
all eight devices on its node. Missing ranks, parser errors, unresolved-kernel
coverage >=1%, invalid NVTX identity, overlapping cost phases, or reconciliation
failure reject the input. Duplicate phase names are retained as occurrences.

Features are compute/non-NCCL, NCCL not covered by compute, H2D/D2H, pointwise
name-family hints and no-traced activity. They are **GPU-second ranking hints**,
not a predictive latency model or removable time. Non-NCCL coverage is not SM
utilization/MFU; NCCL non-overlap is not proven critical-path wait. True dependency
wait remains `UNRESOLVED_REQUIRES_DEPENDENCY_EVIDENCE`.

Selection prioritizes measured-parent/beam-neighbor recipes, then measured
same-mechanism deltas where available, then profile-target hints. It records
every selection reason. It does not synthesize kernels, choose arbitrary TP/PP,
change batch size, relax numerical thresholds, or claim a global optimum. New
kernel mechanisms require separate evidence-driven kernel-design work before
registration. These are deliberate v1 limits, not a claim that universal
automatic scheduling has been solved.

Old source/assets/cycle-1 profiles require `--historical-profile-prior`. Their
identity differences are persisted, and their measurements can only influence
ordering. Native search comparisons still require a common current source,
assets, hardware and semantic fingerprint.

## Commands

Run from a clean source checkout. Local controller evidence belongs under a
named project subtree; large traces and training artifacts stay remote.

```bash
# No GPU, SSH or model loading; all five profiles can be inspected.
python scripts/distillctl.py workloads search \
  --workload wan13b_nogan_resident_best --dry-run --max-trials 8

# Execute where the SQLite files reside. No capture is started by this command.
python scripts/distillctl.py workloads profile \
  --input /persistent/new-run/profile-input.json \
  --output /persistent/new-run/profile.json

# Run Wan first; LiveWan only after Wan finishes and releases its locks.
python scripts/distillctl.py workloads search \
  --workload wan13b_nogan_resident_best \
  --profile /local/project/evidence/wan-profile.json \
  --assets-config /local/project/configs/wan-assets.json \
  --output /local/project/evidence/wan-search \
  --run-id wan-search-r1 --max-trials 8 --historical-profile-prior

# Same command/identities/budget plus --resume, only after a clean completed
# trial boundary. An interrupted trial is preserved and not silently retried.
```

Asset JSON extends the existing best-known schema only with optional per-node
`ssh_port` and `address_family` (`ipv4`, `ipv6`, `auto`). It must bind the same
16-GPU allocation, environment and immutable identity files on both nodes.
No shell/environment performance overrides are accepted. CPU metadata RPC
resolves beside remote assets; no large files are pulled to the controller.

`profile-input.json` has schema `distillschedule.search.profile_input.v1`,
`cycle`, `identity`, `hardware`, and exactly 16 `ranks`. Every rank supplies:

- `node`, `rank`, `device_id`, `global_pid`, `sqlite`, `sqlite_sha256`;
- `parser_status: PASS`, `parser_errors: []`, a `nsys_ai_doctor` path/SHA binding;
- an exact `capture` and disjoint `phases`, each containing `start_ns`, `end_ns`,
  `global_tid`, `nvtx_text`, `coverage: exact`; phases also have name/role/occurrence.

PID/TID and window values must come from the actual native range receipts and
SQLite schema, not guessed rank numbers or global time envelopes. The CPU test
fixtures demonstrate the schema but are explicitly **not GPU evidence**.

After a successful search, replay its registered winner with a fresh run ID:

```bash
python scripts/distillctl.py workloads run \
  --workload wan13b_nogan_resident_best \
  --search-plan /local/project/evidence/wan-search/trials/<confirmation>/execution-plan.json \
  --run-id wan-winner-replay-r1 \
  --coordinator-output /local/project/evidence/wan-winner-replay-r1
```

The source, assets, numerical gate and recipe remain bound; only the output/run
identity changes. Old artifacts are never overwritten. Moving to a new code
HEAD requires a new search and correctness gate, not reusing the old receipt.

## Gates, cost and failure handling

At most eight **candidate timing attempts**, plus the baseline and one winner
confirmation, are admitted. Native correctness preparations run serially and
are cached per candidate. These extra preparations are included in recorded
search cost; the budget is not falsely described as eight total GPU launches.
No C50, new capture or extra workload starts automatically.

All native GPU launches retain the shared node lock across audit, construction,
execution and teardown. Search-only resource sampling records driver free HBM
and node cgroup memory/events every two seconds. Its sampled minimum/peak is
not a continuous exact extremum; allocator peaks come from native rank receipts.
Both arms use identical instrumentation. Timings from earlier unmonitored runs
are not substituted for the newly matched denominator.

Every accepted measurement requires all 16 native status/semantic/version/
census/finite/dtype/memory gates, unchanged cgroup OOM counters, owned cleanup,
and disabled profiler. A controller/SSH exception is preserved with logs and
stops execution; it is not inferred to be OOM. Only explicit native CUDA or host
cgroup capacity evidence can be a capacity failure. Ambiguous/interrupted runs
require reconciliation and are not automatically retried under the same ID.

The trajectory comparator's existing coarse PASS/REVIEW policy is retained.
It does not establish strict P0 numerical parity. Normalized native observations
therefore remain `numerical_status=REVIEW` with the raw comparator status and
metrics preserved. `diagnostic_winner` and `strict_winner` are separate; strict
winner remains null without a true strict gate. No threshold is relaxed merely
to select a faster recipe.

Outputs include immutable `search.json`, profile/identity, fsynced selection
journal, preparation receipts, RPC logs, per-trial plans/native measurements,
resource observations, all candidate failures, confirmation and `result.json`.
The result reports **best tested within budget**, not universal best or a
publication-ready speed claim when numerical eligibility is REVIEW.

## Source closure and current validation boundary

The integrated main checkout lacked a dynamic dependency:
`probe_static_fsdp2_full5_world16.py`, and its imported
`StaticFSDP2Full5NsightRange`. Both were restored from measured C25 source
`d865cdc0e5d7ee99fbe6102a5965de08c8c81b9d`. No profiling is enabled by the restore.

Wan's genuine two-cycle/trajectory workers and their cache/deferred-finite
helper APIs were ported from `7fa5b866c544594094b8d3c421959bbc64bf9206`, preserving
newer integrated Helios helpers. Search arm selection is a default-off opt-in;
canonical winner flags and profile lock hashes stay fixed. The import map keeps
both original provenance and adaptation hashes rather than attributing new code
to historical measured commits. CPU acceptance now also checks dynamic file
imports; a fresh no-weight remote import gate remains necessary.

CPU tests cover contract/tamper rejection, all five registrations, resume,
budget, REVIEW/strict separation, per-device SQLite unions, native adapter
routing and preparation serialization. GPU results must be read from an exact
run's receipts; compilation, unit fixtures, or import success are not GPU PASS.
