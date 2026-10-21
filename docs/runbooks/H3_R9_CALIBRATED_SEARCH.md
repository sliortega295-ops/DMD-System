# Opt-in H3 R9 calibrated search transport

Scope: only the existing 480×832×124 Grid1000 H3 request, CPU staged cache
with trim versus CUDA cache without trim, depth 2 / LoRA tile 128. Neither the
five canonical profiles nor any trainer, numerical threshold or FA3 split
setting changes. This is a transport/evidence adapter, not a new scheduler.

`workloads calibrate --execute-native --r9-transport DEPLOYMENT.json` reuses
`NativeTrialRunner`: two fresh native correctness runs and their existing
sampled state/loss/sample/RNG comparison precede two calibration runs. Run
CPU assembly and model search beside the persistent raw receipts. Use
`workloads run --search-plan PLAN.json --r9-transport DEPLOYMENT.json` for a
fresh actual execution of the saved selection. The default transport stays
unchanged; a transport-marked plan cannot silently fall back to it.

## Controller and receipt host

The deployment has exactly `schema_version=distillschedule.r9_transport.v1`
and three absolute `{path, sha256}` references: `owner`, `template`, and
`dependency_provenance`. The external owner must match the exact reviewed
SHA in `search/r9_contract.py`. The template supplies assignment, safety pins
and qualified runtime versions, never old numerical or cost authority. The
site packet supplies this deployment; absolute site paths are not embedded
in the portable library. Only its two explicit Torch/Transformers overlays
may enter the worker imports.

The controller validates current source/profile/candidate and reconstructs
all node commands/config/environment from the same compiler. For a saved
model plan this is explicitly a **portable envelope**, not local validation
of unavailable remote costs. CPU RPC `r9_rebind` and `r9_prepare` perform full
`validated_profile` (including every calibrated source hash) beside the raw
receipts. Every actual worker again calls full `validated_profile` and
`validate_local` before model construction. The saved `model_plan` survives
rebinding unchanged. JSON files use the common sorted, indented, finite-only
writer; the controller checks exact returned sealed-manifest bytes.

Each fresh run has prepare → two fresh hidden-CUDA dependency/import/linker
qualifications → seal → both owner admissions → both owner executions →
world collection. Only the two R9 SSH sessions exist during execution; CPU
RPCs resume only after both EOFs. The owner retains lock, UUID assignment,
FD/PID-birth, permit, peer, host reserve, OOM, deadline, lease, signal/reap and
terminal cleanup checks. The worker calls the unchanged integrated native
entrypoint. A resource sampler runs under one tracked worker per node; its
cluster compatibility is pending until the new source is actually run.

R9 receipts positively bind exact unprofiled argv/env, native rank evidence,
actual cache registration, selected plan, owner terminal state and resources.
They are **not** fabricated legacy `launch_manifest.json` or teardown text.
Missing R9 proof cannot fall back to legacy evidence. Failed admission logs
are immutable numbered attempts; `workloads r9-owner --plan
--admission-attempt N` permits explicit read-only retry only after previous
sessions reached EOF. One exclusive execution-start record forbids GPU
retry. Unknown owner terminal state forbids new sessions, including queries.

## Minimum acceptance and stopping rules

Five fresh world16 two-cycle runs on one new clean source are required:
1. CPU-cache correctness;
2. CUDA-cache correctness, then matched comparison;
3. CPU-cache calibration;
4. CUDA-cache calibration, then CPU assembly/search/serialization;
5. actual confirmation of whichever plan the measured search selects.

The comparison must be `REVIEW_SAMPLED_PARITY_ONLY` with **all** sampled
state/loss/sample/RNG fields `PASS`. Overall REVIEW limits the evidence scope;
it does not permit a numeric FAIL. If it fails, stop after run 2. If costs,
source, runtime, resources or the saved selection fail validation, stop before
run 5. Do not force CUDA to win. Unmeasured legal variants remain rejected.
Old HEAD timings (including the passing 1437839 operator-selected smoke)
are not calibration inputs to this source.

Calibration measures contextual CUDA-stream stage envelopes, aggregated by
max rank per stage, not cross-rank wall-clock unions. Five runs close the
bounded measured-selection execution path, not the separate full
`ModelValidationSession` journal (which requires at least three diagnostic
timing trials), robust speedup, continuous-exact semantics or long training
parity. That larger journal is not exposed through this narrow R9 CLI.
The deployment retains explicit per-run work/lease bounds; no automatic
failure retries, profiler or package/environment installation are allowed.

## Current evidence

CPU tests exercise synthetic receipt parsing, negative gates and the actual
correctness → calibration assembly → search → JSON roundtrip → selected-plan
collection path. Synthetic durations and receipts are explicitly fixtures,
not GPU evidence. New-source cluster validation remains **NOT_RUN** until the
site operator completes the above sequence and preserves actual receipts.
