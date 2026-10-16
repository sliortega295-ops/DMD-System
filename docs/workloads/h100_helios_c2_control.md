# H100 Helios C2 source / launcher / admission unit

**CPU/source integration is implemented; H100 execution and real C2 calibration
remain NOT_RUN. The stopped campaign is not resumed.** The old 94.311432633 s
result is one Full5 and is neither a C2 sample nor an estimate in this adapter.
This document supersedes only the *remaining-control-plane-gap* statements in
the earlier frozen runtime-unit document; those eight files were not changed.

## Exact source delivery

The original native checkout remains clean at
`8fc70e8ce9f6e0243604812120119c9f4c078e55`. A separately authorized local source-only
provenance commit is `b524bf148a692c92384e54705df6ec403713a745`, directly on that
parent. Its complete binary/full-index diff and all new file hashes are recorded
in the source proof. The 25 KiB incremental bundle needs the existing exact 8fc
parent; it contains no models, checkpoints, environments or large asset copies.
The local derived checkout uses a shared object store, so deploy the incremental
bundle against an existing 8fc repository rather than copying its `.git` alone.

Only five native paths change: the exact documented default-off hook in the
original worker, the frozen C2 runtime module, the new stdlib control module,
a thin summary entry, and a new launcher. Original math helpers, hardened
worker, world16 guard, original Full5 launcher/auditor and policy files remain
byte-identical. The hook reuses one already-initialized engine pair and EMA;
it neither repeats `main` nor constructs another model.

The generated launcher retains the original project/node locks, idle-owner
check, eight-worker launch, deadline, source-health seal and cleanup limited
to its own process group. It requires both explicit `C2_OPT_IN=1` and the exact
compiled-plan file/hash. It only accepts No-GAN, correctness/timing and the
three source-pinned local-FA2 arms. The new prepare restores the C2 mode after
the inherited-variable scrub. The sole extra environment hardening pins
`LD_LIBRARY_PATH` to the selected Python environment's `lib` directory rather
than inheriting an unbound loader path; its H100 runtime compatibility is NOT_RUN.

## Public integration (two root-owned seams)

Apply the small registration patch in this directory to the integration source:

1. `distillschedule.workloads.best_known.cli.main`: route
   `hardware-helios-c2` to `distillschedule.search.hardware_helios_c2_cli.main`.
2. `CostModel.verify_receipts`: for family
   `helios_h100_native.coupled_cycle`, require measured scope and invoke
   `hardware_helios_c2.validate_component_receipt`. This is distinct from the
   Wan-only `hardware_native.*` dispatch and from the A100 `helios.*` family.

The adapter already calls its deep validator before common search/lowering.
The central registration is still necessary to protect arbitrary generic
`CostModel` consumers. The two seams were applied **temporarily in the isolated
source**, exercised through the actual public CLI and generic cost verifier,
then restored. Root alone integrates canonical source.

## Minimal CPU-only usage

After the two registrations, use the normal entry point:

```bash
python -m distillschedule.cli workloads hardware-helios-c2 --help

# Local source derivation, only with explicit authorization for this commit:
python -m distillschedule.cli workloads hardware-helios-c2 prepare-source \
  --parent-root /absolute/clean-8fc-source \
  --destination /absolute/new-source-only-checkout \
  --allow-local-provenance-commit --output /absolute/source-proof.json

# Rehash existing relocated assets in place, never copy/download them.
# This requires actual asset access; it has NOT_RUN status on this workstation.
python -m distillschedule.cli workloads hardware-helios-c2 rebind-assets \
  --source-root /absolute/new-source-only-checkout \
  --legacy-gate /absolute/original-8fc-full-hash-gate.json \
  --asset-root /absolute/existing-assets --arm h100-nograd-retain-nosync \
  --output /absolute/new-derived-head-asset-gate.json

# The legacy deployment contains the real selected Python, assets and eight-GPU
# inventory. Point its asset_gate path/hash at the freshly revalidated new gate.
python -m distillschedule.cli workloads hardware-helios-c2 bind-deployment \
  --legacy-deployment /absolute/legacy-deployment-with-new-asset-gate.json \
  --source-proof /absolute/source-proof.json \
  --repo-root /absolute/new-source-only-checkout --output /absolute/c2-deployment.json

python -m distillschedule.cli workloads hardware-helios-c2 compile \
  --deployment /absolute/c2-deployment.json --arm h100-nograd-retain-nosync \
  --mode correctness --run-id fresh-c2-correctness --enable-native-c2 \
  --output /absolute/c2-correctness-plan.json
python -m distillschedule.cli workloads hardware-helios-c2 dry-run \
  --plan /absolute/c2-correctness-plan.json
# Add --verify-assets for the real CPU-only native full-hash/trace/retention
# preflight. It renders the actual hardened worker argv; it never starts it.
```

A source-only dry-run reports assets **NOT_CHECKED**, not an execution admission.
With `--verify-assets`, absent/old-head assets or correctness fail closed. There
is intentionally no `run`, `ssh`, `resume`, `install` or `calibrate-launch` verb.
An independently authorized controller may later execute the returned exact
outer native argv/environment. That authorization does not exist in this task.

After real native correctness has run, `compile --mode timing --correctness
/absolute/fresh-same-arm-c2-summary.json` binds that exact reference. Native
prepare and admission reopen its raw proof, require the same deployment/arm
and a positive age of at most 24 hours; the old Full5 summary cannot qualify.
Use `admit --plan ... --summary ...` for either correctness or timing summaries.

`assemble-costs --deployment ... --observations ... --output-dir ...` accepts
only hash-referenced timing plans and summaries, each re-audited against all raw
files. The observations object is `{"observations":[{"plan":{"path":"...",
"sha256":"..."},"summary":{"path":"...","sha256":"..."}}]}`. The saved
`model.json` is consumed by `search --model ... --deployment ...
--correctness-map ... --run-id ... --enable-native-c2 --output ...`.
The correctness map maps each arm to its actual correctness path/hash ref.

## Admission and common model_plan semantics

- Exact clean derived source, complete allowed diff, frozen 8fc policy and
  original helper/dependency hashes; unchanged trace/checkpoint identities.
- All eight raw rank files with immutable SHA seals, exact 16-slot bijection,
  one host, observed allocation/Python/packages, no OOM or leftover contexts.
- Sealed NUL argv/environment/unset inputs must equal the native lowering;
  executed command and environment must agree. Raw cgroup/source/ownership
  logs are reread, not just a top-level PASS or claimed zero delta.
- Two cycles in the same initialized objects: G2/C10/EMA10 → G3/C15/EMA15 →
  G4/C20/EMA20. Per cycle: G backward2/optimizer1/scheduler1; C
  backward10/optimizer5/scheduler5; EMA5. Missing fields, early B1 commit,
  rebuilt objects, stale cache, absent retention restoration, changed shape,
  FA3 relabeling and timer/launch-interval contradictions are rejected.
- Original trace steps 10–14 are replayed per cycle, not new data/steps15–19.
  Both cycles' timings exclude initialization and added bookkeeping/audits;
  the frozen runtime's required counters remain in their declared timer scope.
- The historical runtime field `launcher_auditor_gate=NOT_IMPLEMENTED...` is
  not promoted or erased. It describes that frozen module's boundary. Only the
  separate new raw audit can produce `PASS_C2_PROTOCOL_ONLY`.
- Existing public `PlanningProblem`, `CostModel`, hierarchical search and
  `validate_saved_prediction` are used. A selected `model_plan` includes exact
  problem/cost/result source bindings and lowers to the actual native arm.
  Calibration is explicitly labeled `EXPLICIT_CALIBRATION`, never a planner win.
- Component samples are max-rank actual C1/C2 nanoseconds, separately. Memory
  reserves the whole physical GPU exclusively. This is a conservative capacity
  reservation, **not measured peak memory** or a fabricated memory estimate.
  No overlap discounts or free-standing G/C timing estimates are invented.

Numerical status remains **REVIEW**; strict full-state parity and claimable
speedup remain NOT_RUN/false. This is not proof of C2 performance, capacity,
manual-best proximity, or successful H100 execution.

## CPU evidence and remaining work

22/22 focused stdlib tests pass, plus two public integration checks. Tests call
the actual 8fc logical helper bodies with explicitly labeled CPU model fixtures;
asset access and raw health are test doubles. Fixtures cannot be relabeled as
native measured costs. The earlier 17 Full5 bridge and 13 C2-runtime tests are
retained. Source inspection, fresh-process policy inventory, shell syntax and
incremental-bundle verification are real local checks and import no Torch.

Remaining external work, all **NOT_RUN**: install the exact new source on the
receipt host only after new authorization, full-hash asset rebinding/preflight,
real correctness for each candidate, continuous C2 timing, full-state numerical
comparison, actual cost admission/search, and selected-plan confirmation. The
stopped H100 campaign must not auto-resume.
