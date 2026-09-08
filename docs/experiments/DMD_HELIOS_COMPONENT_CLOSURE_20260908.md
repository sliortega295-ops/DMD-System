# Helios No-GAN / GAN: measured search → native execution

## Result and claim boundary

Both Helios workloads are connected to the existing common planner, not a new
workload-specific search engine. Real correctness and calibration preceded CPU
search; generated assignments were lowered and executed on 2 × 8 A100-40GB.
No Wan, LiveWan or H3 experiment was repeated in this unit.

| Variant / native trial | Implementation | Cycle1 warm-up (s) | Cycle2 native (s) |
|---|---|---:|---:|
| No-GAN baseline | world16 FSDP, chunk512, window5 | 96.616515 | 93.165184 |
| No-GAN incumbent | node HSDP 2×8, chunk512, window5 | 95.795071 | 91.767029 |
| No-GAN generated proposal | node HSDP 2×8, chunk512, window1 | 95.867011 | 91.736748 |
| No-GAN confirmation | same generated window1 | 95.919582 | **91.617123** |
| GAN baseline | world16 FSDP, chunk512, persistence off, window5 | 133.385999 | 126.901628 |
| GAN confirmation | same baseline; only modeled-feasible assignment | 133.576174 | **126.762420** |

Each row is a **fresh initialization followed by two cycles in the same
process**. Cycle2 is the maximum rank-local CUDA-fenced host wall, not an average
of the two cycles or cross-node min-start/max-stop wall. Setup, correctness
tensor probes, serialization, CPU cost validation and owned teardown are outside
the cycle timers. No Nsight/profiled time is in this table.

No-GAN baseline/confirmation diagnostic ratio: **1.016897×**. Relative to the
already optimized incumbent: **1.001636×** (~0.16%), not evidence of a material
new optimization. All five HSDP windows had equal predicted latency; window1
was a deterministic tie-break, not a prediction of faster runtime.
GAN's two rows are the **same candidate**; their ratio is repeat variation,
not speedup. These controls already have the canonical optimized CUDA kernels;
they are not the historical naive 118 s / 208 s baselines.

All actual trials: **16/16 ranks, two cycles, semantic gates PASS**, no observed
CUDA OOM or new cgroup `oom`/`oom_kill`, owned teardown PASS. Numerical evidence
is **REVIEW** (sampled state/loss checks), not full-state strict equivalence.
`strict_winner=null`; existing canonical profiles and historical winners are
unchanged. No new 20% acceleration or global search optimum is claimed.

## What the search and executor actually do

```text
source/asset/environment-bound native correctness
    → per-action CUDA-fenced component calibration
    → common PlanningProblem + cost/lifetime/resource simulation
    → bounded assignment search
    → typed lowering + common ExecutionGraph / StaticRoleRunner
    → native helpers in selected order
    → 16-rank evidence + cycle2 feedback
```

The existing P0 helpers still implement the models and losses. Their **dispatch
order** comes from the generated graph, not the old fixed one-cycle loop.
Assignments, actual choices, trace membership, application calls, version
dependencies and retained-latent lifetimes are checked against rank receipts.

| Variant | Real component records | Candidate space | Feasible | Missing costs | Predicted capacity rejects |
|---|---:|---:|---:|---:|---:|
| No-GAN | 40 | 20 | 10 | 10 | 0 |
| GAN | 22 | 4 | 1 | 2 | 1 |

No-GAN searches FSDP/HSDP × chunk512/2048 × retained window1..5. Only chunk512
has current measured component support. Its calibrated window5 and generated
window1 execute different real orders:

```text
window5: g r0 r1 r2 r3 r4 c0 e0 c1 e1 c2 e2 c3 e3 c4 e4
window1: g r0 c0 e0 r1 c1 e1 r2 c2 e2 r3 c3 e3 r4 c4 e4
```

For the window1 confirmation, **all 16 ranks in both cycles** report exactly
that executed order and peak live rollout count1. This is the concrete
no-fallback proof. Retained window changes separate B1 lifetimes, not batching.

GAN searches chunk512/2048 × Generator unshard persistence off/on, world16 FSDP
and window5 fixed. Its graph is `g,r_all,c0,e0,c1,c2,c3,c4`. `r_all` owns one
acquire → five separate B1 rollouts → release scope; no split persistence scope
or skipped Score call is admitted. Both native trials executed this graph on
all 16 ranks. Per cycle: one G commit, five C commits, one EMA; each GAN C update
keeps two native loss/backward helper calls and **one** optimizer commit.

Sharing is only the existing Score/Teacher adapter alias. There are no new
physical-backbone sharing layouts, TP/PP groups or arbitrary GPU pools here.
Unsupported/missing combinations are `UNCALIBRATED`, never filled with old wall
times or guessed NCCL discounts. Internal communication/compute overlap is
already inside each measured action envelope; the CPU model does not invent
additional inter-action overlap on the same world16 pool.

Predicted vs confirmation cycle2: No-GAN **91.877870 → 91.617123 s** (0.285%
relative prediction error); GAN **126.626025 → 126.762420 s** (-0.108%). These are
small, bounded validation sets, not general cost-model accuracy claims.

## GAN safety rejection is not an OOM

The C25 anchor (chunk2048, persistent G unshard) completed current-source
correctness **and** fenced calibration without OOM. Observed/model peak
allocated: **39,734,593,536 B = 37.005724 GiB/rank**. The per-rank model limit is
39,219,232,768–39,290,535,936 B after the unchanged 2 GiB margin and observed
non-allocator usage. It misses that modeled limit by roughly 0.41–0.48 GiB.

It remains `PREDICTED_CAPACITY_FAIL`, not `CUDA_OOM`. Fenced cycle2 calibration
was 124.029020 s, **not a native performance result**. No incumbent timing was
substituted from calibration or history. Native incumbent validation is
`NOT_RUN`, incumbent/improvement fields are null, and no extra proposal was
added to spend the skipped trial. The safe selected control was confirmed.

The old validator required both anchors to be feasible and aborted before any
native validation. The scoped fix preserves the rejection and skips only a
capacity-rejected incumbent, never a failed baseline. Admission also reaches
implicit correctness bootstrap, calibration and native execution: a rejected
peer cannot be launched indirectly. Fresh validation without an existing
source-bound preparation or another feasible bootstrap peer **fails closed**.
This run reused its already verified current-source correctness preparation.
Same-baseline confirmation jitter cannot produce a strict winner.

Native GAN control/confirmation max allocated: **24.944726 GiB**, max reserved:
**38.095703 GiB**; sampled minimum driver free **440 MiB**. Node cgroup sampled
peak across the two runs: **101.124119 GiB**. High allocator reservation explains
why the model's allocation margin is not a certificate of ≥2 GiB driver free.
No continuously sampled hardware capacity margin is claimed.

## Frozen workload and versions

- B1/rank, global16, 384×640, fixed immutable trace/cohort, BF16 native compute,
  FP32 reduction, local pinned FA2, Score packing off; no TP/PP/batching changes.
- Both residuals CUDA out-of-place, CUDA EMA, no parameter/activation offload,
  block streaming or model swap. Canonical loss/optimizer/EMA predicates remain.
- Two cycles repeat the same source cohort, explicitly
  `fixed_source_cohort_replay`, rather than inventing trace steps15..19/1005..1009.

| Variant | Source cohort | Initial logical G/C/EMA | Final logical G/C/EMA | Final physical G/C/EMA |
|---|---|---|---|---|
| No-GAN | 10..14 | 2/10/10 | **4/20/20** | 4/20/20 |
| GAN | 1000..1004 | 200/1000/800 | **202/1010/802** | **4/20/802** |

GAN uses an immutable **systems-only** bootstrap, not official checkpoint1000.
Its actual restored Adam/scheduler counters are G2/C10 while logical schedule
versions are G200/C1000. An initial adapter bug equated these namespaces and
correctly stopped before cycle1. The fix validates the immutable mapping and
applies offsets only to ledger/receipt labels; it does **not** rewrite Adam,
scheduler, EMA or the branch predicates. Action/cycle/final physical and logical
states are independently checked. Original failure evidence is retained.

## Source, environment and artifacts

Native No-GAN source: `85be7e4e36dea31bdc01bcfe8faf3d67efb58494`.
Native GAN source: `cc7e451bff7387640634a394f07047be8e0c089e`.
Final controller/collector source: `a7dbfc264acde942fc0fbf90ba8e65399a4372b4`.
Later report commits are not runtime HEADs.

GAN validation continued with a process-local, hashed **CPU controller** fix,
while native worker and calibration source both stayed cc7e451. Snapshot SHA256:
`950c1bb364f1553d9294c95682ce03f62d7c0e4bfe4a47ae627716e1ccc2e6e2`.
The final admission/label/jitter guards were subsequently tested and committed;
they were not relabelled as GPU execution of that final HEAD. Old-source costs
remain pinned and fail admission at a different HEAD; regenerate matching
calibration for new-source runs.

Both nodes reused `/tmp/distillschedule-helios-official-torch210`: Python3.10.20,
Torch2.10.0+cu128, CUDA12.8, NCCL2.27.5, Diffusers0.38.dev, Transformers5.3.0,
PEFT0.18.1, A100-SXM4-40GB, driver535.161.08. Selected binary, FA2, model,
bootstrap, trace and source hashes are in the manifests; no environment was
installed or altered. A transient shared-filesystem import read failed on node1;
one bounded read-only check showed exact committed bytes/zero NULs and fresh
preflight passed. Failed preflight artifacts were not overwritten.

Small auditable bundle:
`artifacts/experiments/joint_planner_helios_components_20260908/`.

Persistent controller/evidence root:
`/home/lyy/Helios/research_artifacts/DMD-System/joint_planner_20260907/artifacts/helios_native_r1/`.

Raw persistent node root:
`/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/helios_native_r1/`.

Key run prefixes:
`jointp_hv_20260908_85be7e4_nogan` and
`jointp_hv_20260908_cc7e451_gan`. The JSON/CSV contain exact suffixes and original
receipt SHA256s. Models, traces, full raw rank tensors and large artifacts remain
remote; only derived evidence and reproducibility metadata are included here.

## Validation and stop boundary

Final on-tree CPU regression: **1003 passed, 17 skipped**; **108 focused passed**;
350 Python files compiled; launch shell syntax, diff checks and both public
Helios model-based dry-runs passed. An initial final-test command used a wrong
test filename and ran zero tests; that command error is retained separately and
not counted as a code regression. Follow-up review caught and tested the
implicit-rejected-bootstrap and same-baseline-jitter cases. The first review
had independent coverage; the later admission review had agent-thread limits,
so follow-up owner verification/tests are not advertised as a full independent
review of the final patch.

Final two-node readback: 16 GPUs idle, no owned worker/profiler, locks available,
cgroup `oom`/`oom_kill` unchanged at0. No processes belonging to others were
cleaned. No new profiler, kernel optimization, C50, canonical winner replacement,
main merge or GitHub push in this unit. The branch also contains earlier
unpublished work, so all scoped commits remain local pending explicit publication.

### Operational validation

For a future new-source calibration/validation, first verify clean identical
node HEADs and exclusive locks. Healthy signals are all16 ranks, two cycles,
matching selected/executed order, physical/logical versions, source/asset
fingerprints, finite checks, zero OOM deltas and owned teardown. Stop on missing
rank/cost, unexpected order/version, stale correctness source, rejected-candidate
bootstrap, GPU query failure or nonzero owned teardown. Never repair admission
by dropping a margin, importing historical timing or falling back to the fixed
one-cycle loop. Owner: experiment operator; check each stage before proceeding.
