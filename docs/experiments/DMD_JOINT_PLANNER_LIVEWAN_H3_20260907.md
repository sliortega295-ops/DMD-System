# Native planner continuation: LiveWan complete, H3 numerical gate blocked

**Status:** LiveWan completed measured-cost → CPU search → native two-cycle
validation. H3's generated implementations run successfully, but cross-arm and
unchanged-arm numerical checks fail; **H3 calibration/search/timing did not run**.
No threshold was relaxed, no canonical winner changed, and main was not merged.

Execution sources (not the later publication commit):

- LiveWan: `06d5e67aab7e4aef0f110e8b315ac7f02c80ff4b`.
- H3: `b1706f073d09a5de03926c6e216273898af754f3`.
- Wan's earlier closed loop is preserved at its own execution source; it was not rerun.

## Environment and denominator

Two owned nodes, eight A100-SXM4-40GB each, driver 535.161.08. Python 3.10.20,
PyTorch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5; prior validated environments were
reused, not rebuilt. Exact assets, selected binaries, runtime fingerprints,
launches and rank receipts are hash-bound in the evidence. This is not a hash of
every installed package. Raw artifacts stay on the shared persistent filesystem.

Every run executes **two cycles in one process, after one initialization**.
Cycle 1 is warm-up/diagnostic; comparisons use cycle 2. Neither calibration nor
correctness-mode observations are substituted for independent native timing.
There is no Nsight run in this epoch. Host initialization, teardown and probes
outside the cycle boundary are not training time.

- LiveWan uses its native world timing receipt. The normalized native receipts
  record the exact `native_timer_denominator` and separate `construction` data.
- H3 uses **max(rank-local CUDA-fenced host duration)**, not synchronized global
  start/stop timestamps. Four device fences bound two cycles, no new mid-cycle
  global barrier. Loss/state/RNG host reads are outside those windows.
- The H3 times below came from **correctness runs** and are not admitted search
  measurements or a new speedup. Cross-node clock alignment is not established.

## LiveWan: completed loop

Same current source/assets/precision/hardware per comparison. Baseline is the
registered sequential world16-FSDP control; pipeline is the existing persistent
8G+8Score native path, depth-1 lookahead. Existing physical Critic B2 is retained;
logical global B16 and sample-weighted call/update census pass. This does **not**
claim a newly validated B1-only pipeline.

| Native trial | Implementation | Cycle 1 (s) | Cycle 2 (s) | Model prediction (s) |
|---|---|---:|---:|---:|
| baseline | FSDP, modulation/gate/row-major off | 83.708773 | 62.637898 | 62.416109 |
| incumbent | pipeline C34, all three on | 53.750258 | 47.066531 | 47.097270 |
| adjacent candidate | pipeline C33, row-major off | 55.414426 | 48.441613 | 48.404250 |
| independent confirmation | pipeline C34 | 53.614074 | 47.138629 | 47.097270 |

Confirmed **diagnostic** ratio: **62.637898 / 47.138629 = 1.328802×**.
All four native runs: 16/16 runtime/semantic/resource PASS; G2/C10/EMA0;
zero new CUDA/cgroup OOM; owned teardown PASS. Numerical status remains
**REVIEW**, `strict_winner=null`. It is not strict numerical equivalence.
The confirmation rediscovered the incumbent; **no faster new candidate** was found.

Three compositions were actually calibrated: 30 operation-cost receipts,
320 owner-rank-stage observations. CPU search agrees with the bounded exhaustive
oracle. Absolute native prediction errors were 0.354%, 0.065%, 0.077% and 0.088%.
These are **context-specific coupled commit macros**, already containing native
internal overlap. They are not independently predicted G/Score service costs,
nor a demonstration of arbitrary GPU-pool/DAG search or generalization to new
workloads. Two other legal compositions remain explicitly unmeasured.

Ledger: three correctness + three calibration + four native runs on this source.
Timing attempts = 4 (including confirmation), not four additional candidates.
The prior failed epoch is separately preserved and none of its costs were reused.

### Memory caveat

Peak allocated: baseline 21.91 GiB; confirmed pipeline 35.15 GiB. Some sampled
minimum driver-free values were only 8–12 MiB. Completed no-OOM runs therefore
**do not establish robust headroom**. Driver/allocator samples and the model's
memory limit are not continuously observed capacity guarantees.

## H3: native integration works, numerical admission fails

Frozen configuration: Grid1000 **approximate** sigma semantics, shared backbone,
world16 FSDP2, 768×1344×124, B1/rank, segment 1, boundary activation policy
`none`, all-50-block FA3 output/LSE **CPU-staged cache** retained. N128 LoRA
kernel tile fixed. Only pinned-staging inflight depth 1 versus 2 was compared.
`activation_policy=none` does not mean there are no CPU copies.

| Correctness diagnostic | Cycle 1 (s) | Cycle 2 (s) | Max allocated (GiB) | Max reserved (GiB) |
|---|---:|---:|---:|---:|
| depth1 | 773.264660 | 769.160942 | 33.395 | 37.098 |
| depth2 incumbent | 776.619708 | 772.570723 | 33.395 | 38.562 |
| unchanged depth1 self-repeat | 775.856636 | 780.012096 | 33.395 | 37.625 |

All three runs completed **2/2 cycles, 16/16 ranks**. Every rank/cycle has:

- Student/Fake/Teacher application forwards **24/6/1**;
- Student/Fake backward graphs **1/5**, fixed exit index 3;
- 96 rank-qualified sample identities per world cycle;
- Grid wraps/scoped executions/hits/misses **300/600/600/0**;
- final **G2/C10/EMA0**, finite logged losses, no new CUDA/cgroup OOM.

Sampled node cgroup peaks were approximately 443–451 GiB. Sampled minimum driver
free fell to 8 MiB on one depth2 node; this is not a claim of 2 GiB safety margin.
The complete per-node values and observation counts are in `h3/outcome.json`.

### Actual blocker and repeat experiment

**Both cross-depth comparison and unchanged-depth self-repeat fail the original
DEFAULT_P0_PARITY thresholds.** Sample identity and observed default Torch CPU/
rank-CUDA RNG boundaries match on all ranks. Other RNG generators are not
observed; matching those boundaries is not proof of every consumed random value.

| Sampled component | Cross depth1/depth2 failed / 512 | Depth1 self-repeat failed / 512 |
|---|---:|---:|
| Student parameters | 256 | 256 |
| Student Adam state | 80 | 80 |
| Fake parameters | 275 | 272 |
| Fake Adam state | 279 | 272 |

These are 32 tensors × 16 ranks with at most 16 local coordinates each, **not full
state comparisons**. Some failing tensors/moments have tiny absolute magnitude;
relative error and cosine can be ill-conditioned near zero. That observation is
not used to excuse failures or change the thresholds.

**Proven:** first Student loss and first Fake loss are bitwise equal on 16/16
ranks in both comparisons. By the second Fake loss, all 16 ranks differ, even in
the unchanged-depth repeat. Thus changing staging depth is **not necessary** to
produce the observed drift, and the result cannot be attributed uniquely to
that search knob.

**Strong inference:** the first Fake backward/update chain is the highest-value
place to localize the initial divergence. The existing compact FA3 backward is
called with nondeterministic mode (`fa3_replay_cache.py`, `_CompactFA3Autograd.backward`),
but these observations do not prove that FA3 dominates the end-to-end drift.
FSDP reductions, tiny-gradient/Adam amplification, or a shared-runtime bug remain
possible. No backend flag or training arithmetic was changed to force a pass.

**Not run:** H3 measured calibration, model search, independent native timing,
strict full-state equivalence, Exact continuous-sigma comparison, extra kernels,
HSDP/CP, C50 or either Helios workload. No H3 speedup/winner is published.
The controller stopped before calibration, preserving the comparison failure.
A single unchanged-arm correctness repeat was diagnostic attribution, not an
extra timing candidate or a workaround for the gate.

## Integration bugs fixed and preserved failures

1. Native model validation still routed through Wan's lowerer and replay chose
   Wan anchors for emitted H3 plans. The typed adapter and calibrated seed pair
   are now used; malformed/incompatible plans still fail closed.
2. The LiveWan macro assembler expected a raw `run_id` absent from native rank
   receipt schema. Source/path-bound native records now provide provenance;
   no identifier is invented. R1 failed evidence and raw calibration remain.
3. Runtime identity inspection had an LD override not used by actual clean
   LiveWan workers. The probe now follows the same clean environment.
4. H3's pinned LightX2V loader adds `config_path` and reads JSON scientific
   notation as strings. Generated config binding initially rejected construction.
   It now verifies that path against `H3_CONFIG`, and normalizes only Student/
   Fake LR/epsilon exactly as the pinned optimizer's existing `float()` calls.
   Wrong values, unknown fields and non-finite inputs still fail. No model math,
   optimizer value, sampling, activation policy or FA3 option changed.

The original H3 failure was a config-identity integration error, **not OOM**.
The fix had a reproduced red test, passing regression tests, and actual pinned
loader + CPU trainer-construction probes on both nodes before GPU execution.
One initial CPU probe unexpectedly initialized a CUDA generator; it terminated,
was preserved, then a new explicitly CPU-masked probe passed. Training GPU
visibility was never masked.

Current CPU suite: **768 passed, 17 skipped**; skipped GPU-specific tests are not
reported as passes. Compile/shell/diff checks passed. The completed CE review of
the main adapter patch produced three actionable findings; all were resolved
with regression tests. Independent external review exceeded its deadline and
provided no usable artifact, so external coverage is **not claimed**. The later
H3 loader fix additionally has explicit red/green and live integration evidence.

## Evidence, replay and stopping point

Small bundle: `artifacts/experiments/joint_planner_native_macros_20260907/`.
Manifest: `artifacts/manifests/dmd_joint_planner_native_macros_20260907.json`.
It contains original native normalized receipts, model costs and all 30 small
calibration source receipts, oracle/search outputs, H3 per-rank loss/state
summary, source hashes, failed-run records and cleanup proof. Large raw data,
models/environments and Grid binary are not copied into Git.

CPU-only integrity/search replay from the repository:

```bash
PYTHONPATH=src python artifacts/experiments/joint_planner_native_macros_20260907/verify_bundle.py
```

This verifies relocated copies by their original hashes and repeats the bounded
LiveWan CPU oracle over the serialized historical IR. Fresh native lowering on
a different HEAD correctly rejects the old costs. Replay does **not** reexecute
native trials or make current-source GPU claims. Exact historical remote commands remain in launch manifests listed
by each native receipt; do not reuse their immutable run IDs.

Final cleanup: both nodes have zero GPU applications/owned worker-profiler
processes, available node locks, and cgroup oom/oom_kill increments zero.

Next minimal task is a bounded comparison at the **first Fake backward/update**:
same inputs, per-operation gradients and pre/post reduction/clip/update samples,
with a same-arm repeat. If a deterministic diagnostic is needed, it must be a
separately labeled contract, not silently substituted into current results.
No further GPU work was automatically launched after the repeat.
