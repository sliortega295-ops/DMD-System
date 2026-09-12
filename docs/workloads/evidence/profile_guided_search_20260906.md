# Profile-guided search v1: two world16 pilot closures

## What is implemented

`distillctl workloads profile -> search -> run --search-plan` reuses the existing profile compiler, dispatcher, node locks and native workload runners. The space is registered, finite and source-bound; this is not universal automatic kernel generation or a predictive latency model. Five workloads have CPU registration; only Wan and LiveWan have GPU search adapters in this phase.

Canonical defaults and registry lock remain unchanged. This search does **not** replace the already-published LiveWan C34 with C32. C27/C33/C34 were not timed within this eight-candidate budget, so C32 is only the best tested recipe, not a global or historical winner.

## Measurement boundary

Each native measurement initializes once and runs two continuous cycles. Cycle 1 is warmup; only cycle 2 ranks plans. One independent winner confirmation is reported, never the minimum of the confirmation and selection run. Native semantic/census/version/sample/finite gates pass, but the workload-owned coarse numerical result remains **REVIEW**. Strict numerical winners are therefore **not established**. No thresholds were changed.

| Pilot | Matched baseline cycle2 | Confirmed candidate cycle2 | Diagnostic time ratio | Confirmation allocated peak | Final state |
|---|---:|---:|---:|---:|---|
| wan13b_nogan_resident_best / resident-lora-epilogue | 43.527254 s | 36.786678 s | 1.1832x | 21.899 GiB | G4/C20/EMA20, 16/16 |
| livewan_crossscale_pipeline_best / c32 | 62.868594 s | 48.611781 s | 1.2933x | 35.153 GiB | G2/C10/EMA0, 16/16 |

These are new matched diagnostic measurements, **not reproduction or replacement of historical best times**. Baseline and candidate use a common source within each pilot; the two different workload pilots have different measured commits.

## Exact sources and process evidence

- Wan measured source: `0921deb12c64eba7f13fd0e12ac0ef8db19842a9`; run prefix `pgsearch-wan-0921deb-r4`.
- LiveWan measured source: `ff0c874a34fce9b34457767118d98cbfa8223a1e`; run prefix `pgsearch-livewan-ff0c874-r2`.
- Post-run metadata correction: `d2b47cb7c443b3408fbc740a10068a20c26b99e3`. Only descriptive reference role residency changed, with a CPU proof that runtime argv/environment/typed plan and semantic contract are unchanged. No old measurement or plan was relabeled.
- The actual LiveWan sequential reference has Student FSDP2 **keep-unsharded**, Critic **GPU resident replica**, and Teacher FSDP2 **reshard**. It is not three FULL_SHARD models.
- Node0/node1 source during LiveWan execution was identical and clean. All trial-owned teardown receipts passed; after search both nodes had no task GPU/worker processes, all eight GPUs free, node locks released, rendezvous ports 31371--31373 free, and cgroup oom/oom_kill counters still zero.
- Saved plans remain bound to the actual measured commits. Check out that commit to replay; do not bypass the new-HEAD rejection.

## Measured candidates

A chronological row is not necessarily a one-factor edge. Use the registered parent in the CSV: C26 and C28 are separate children of C25. FSDP -> Resident Tree and FSDP -> C24 are explicitly bundled endpoints, not pure placement ablations.

### wan13b_nogan_resident_best

| Candidate | Parent | Cycle1 s | Cycle2 s | Phase | Numerical |
|---|---|---:|---:|---|---|
| fsdp-reference | -- | 52.911065 | 43.527254 | baseline | REVIEW |
| resident-tree | fsdp-reference | 43.696587 | 41.060173 | candidate | REVIEW |
| resident-chunk2048 | resident-tree | 40.811556 | 37.553828 | candidate | REVIEW |
| resident-lora-epilogue | resident-chunk2048 | 39.847620 | 36.562443 | candidate | REVIEW |
| resident-lora-epilogue | resident-chunk2048 | 39.800907 | 36.786678 | confirmation | REVIEW |

Candidate timing attempts: 3; correctness preparations: 4; total native timing runs including reference and confirmation: 5.
Search cost including native preparation: 1452.884 s. This excludes earlier aborted searches, adaptation and profile import.

### livewan_crossscale_pipeline_best

| Candidate | Parent | Cycle1 s | Cycle2 s | Phase | Numerical |
|---|---|---:|---:|---|---|
| fsdp-reference | -- | 79.395188 | 62.868594 | baseline | REVIEW |
| c24 | fsdp-reference | 60.455793 | 54.848207 | candidate | REVIEW |
| c25 | c24 | 59.042859 | 53.222516 | candidate | REVIEW |
| c26 | c25 | 58.185063 | 52.621734 | candidate | REVIEW |
| c28 | c25 | 57.318273 | 51.941445 | candidate | REVIEW |
| c29 | c28 | 56.856754 | 50.428918 | candidate | REVIEW |
| c30 | c29 | 56.423922 | 49.693758 | candidate | REVIEW |
| c31 | c30 | 56.069164 | 49.010938 | candidate | REVIEW |
| c32 | c31 | 55.325664 | 48.643508 | candidate | REVIEW |
| c32 | c31 | 55.406844 | 48.611781 | confirmation | REVIEW |

Candidate timing attempts: 8; correctness preparations: 9; total native timing runs including reference and confirmation: 10.
Search cost including native preparation: 8542.852 s. This excludes earlier aborted searches, adaptation and profile import.

## Resource and claim caveats

- LiveWan confirmation allocated peak: 35.153 GiB; reserved peak: 38.076 GiB. Sampled minimum driver free reached 0.0 GiB. Completion without OOM does not prove spare production headroom; allocator cache/reservation and live allocations are different.
- LiveWan reference sampled node cgroup peaks were about 454.5/454.4 GiB; confirmation about 71.6/444.7 GiB. These include cold construction, not just the timed cycle. Current node limits were checked before launch; no environment or global cache was changed.
- Resource sampling is every 2 seconds, not a continuous exact minimum/peak. Native allocator peaks are separately recorded. No GPU-utilization/MFU claim is made.
- Historical SQLite data remained remote. Wan rank0/8 prior scope stays representative-only; LiveWan prior covers all sixteen ranks. Neither prior is used as a native latency denominator.
- Helios No-GAN, Helios GAN C25 and MiniMax-H3 Grid1000 search are CPU registration/dry-run only in this release. Their GPU search is **NOT_RUN**; Grid remains explicitly approximate.

## Preserved integration failures

Earlier immutable IDs remain available: Wan r1 loader API mismatch; Wan r2 state-sketch API mismatch at cycle1 anchor; Wan r3 CPU plotting dependency; LiveWan r1 rank-count/rank-cycle receipt schema mismatch. None was reclassified as OOM or promoted to a timing result. Fixes were scoped to source closure, optional compatibility and evidence readers. Shared environments were reused, never rebuilt or upgraded.

## Artifacts

Local code-HEAD CPU acceptance (`d2b47cb`): **468 passed, 17 skipped** in the full
refactor suite, plus the separate 18-test integration checker PASS (overlapping
coverage; do not add the counts). The skipped cases require unavailable local
Torch or matplotlib; remote tensor/import checks and real GPU receipts are
separate evidence. Compile, shell syntax, reviewed import hashes and all five
canonical profiles also pass the integration checker. GitHub CI on this
unpublished branch is NOT_RUN.

- [Machine-readable evidence and native SHA bindings](profile_guided_search_20260906.json)
- [Per-candidate timings and registered parent](profile_guided_search_20260906.csv)
- Persistent controller evidence: `/home/lyy/Helios/research_artifacts/DMD-System/profile_guided_search_20260906`.
- Large native artifacts remain under `/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/profile_guided_search_20260906/`.
- `wan_search_r4/result.json` and `livewan_search_r2/result.json` bind ranking, independent confirmation and replay plans.
- CPU test logs/manifests and append-only research log are in the controller evidence tree. Final local checks are recorded separately; no unpushed-head GitHub CI claim is made.

Search is stopped. No new kernel, C50, additional workload, Nsight capture, canonical promotion, push or main merge was performed.
