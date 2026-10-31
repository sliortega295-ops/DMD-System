# Bind the existing checker to decisions

Use the project's frozen checker and runner. This thin adapter reads their actual
outputs and constructs a receipt; `settle` and `assess` reparse those bytes. It
does not execute arbitrary commands from retrieved material, install a trainer,
or claim an independent security sandbox.

The runner must own the invocation, unchanged checker and raw output. File hashes
detect drift; they do not prove that a process ran, that reported identities are
truthful, that test selection covers the model, or that an optimizer cannot edit
the output. Isolate checker ownership externally where available. This explicit
trust boundary remains even after parsing PASS/FAIL correctly.

## Freeze before running

In the schema 2 campaign contract, `receipt_policy.checkers` maps formats to
`{path, sha256, locator, version}` of the actual checker/runner. Remove unused
formats. `receipt_policy.checks` maps V0/V1 and each required risk level plus
optional compatibility/composition/fallback/capacity to
`{tests: [exact_unique_test_name], scope: "what this actually verifies"}`.
Freeze the complete selection. Names must uniquely identify cases in the JUnit
file; use stable names/parameter IDs in the existing checker.

Each reservation binds candidate/source/config/input identities, comparator id
and source, and `validation_scope`. `run_id` identifies the actual attempt.
Timing must compare current control; correctness can compare it or the frozen
original baseline. Default/control promotion additionally requires all risk-route
checks for the same candidate against the original baseline. A chain of adjacent
comparisons is insufficient. Retain actual checker scopes in the decision.

## Adapt and settle

    python3 "$SKILL/scripts/dmd_receipts.py" --input "$RUN/adapter-request.json" > "$RUN/receipt.json"

Start from `assets/checker-adapter.template.json`. Copy the frozen checker and
selection, the reserved binding and the raw file's hash. Put the resulting JSON
object in the settlement's `receipts` array, then settle normally. Reuse the same
input file and hashes when retrying. Raw and checker paths in the normalized
receipt are absolute and must remain available for reassessment.

Supported formats:

| Format | Actual input | Derived result |
| --- | --- | --- |
| `junit` | Project checker XML, such as pytest JUnit | Required test names present once, no failure/error, selected cases not skipped -> PASS; missing/skipped -> NOT_RUN; failures/errors -> FAIL |
| `aba-json` | Runner JSON described below | Timing values, process ids, measurement kind and scope |
| `attempt-json` | Runner failure JSON described below | Predeclared environmental invalidation; never candidate failure |

Named JUnit failures map to their frozen check labels. Unmapped failures or
inconsistent suite counts fail all labels conservatively. Capacity failure does
not invent numerical failure; failed compatibility/composition/fallback prevents
default enabling without erasing separately established correctness/benefit.
A hash-correct FAIL body cannot be overridden with PASS. Extra
checks, failed/skipped/duplicate tests and scope/version mismatches remain visible.
Compatibility, fallback and composition summaries likewise need parsed checks.
The existing `scripts/check_cpu_receipt.py` also consumes JUnit; the external CPU
integration test runs both it and this adapter on the same real checker output.
Other project formats need another explicit parser; they are not accepted as
handwritten PASS summaries.

An `aba-json` body has `status: PASS`, the exact `binding`, `unprofiled: true`,
`repeat_unit: independent_processes`, `measurement_kind: observed` (or synthetic
for a workflow fixture), and `timing` containing:

```json
{"run_id": "C1-block1", "process_ids": ["launch-a", "launch-b", "launch-a2"],
 "control_before": 100, "candidate": 99, "control_after": 100}
```

These numbers are syntax only. The real runner writes its own values, identifiers
and binding, and is frozen as the `aba-json` checker. Result JSON copies the parsed
timings; a mismatch is rejected. A/B/A invocations must be independent unprofiled
processes with the same logical work. Parsing does not attest process independence.

An `attempt-json` body has `status: error` or `interrupted`, the same binding,
and `sample` containing `status: invalid`, `valid_window: false`,
`failure_origin: environment`, and one allowed reason. That reason must already
be in `timing.replaceable_failures`. Copy sample/outcome into the settlement.
The runner must classify failure before inspecting speed. A candidate OOM or
numerical failure is not an environmental interruption. No legitimate slow
sample can be relabeled invalid; receipts and settlements are immutable.

## Reuse and the CPU integration boundary

`assets/reuse-decision.template.json` records existing equivalent behavior, mode,
dependencies/conflicts, integration, expected actual path and rollback. Fill it
before editing; attach path/numerical/rollback evidence before VERIFIED. Validate
with `dmd_reuse.py --input ...`; include the object in finish.experience when reuse
was involved. A retained reuse must be VERIFIED and match the candidate identity.

`tests/test_external_reuse.py` creates an independent scalar project, retrieves
and hash-checks a parameter-phase implementation, minimally integrates it, checks
actual scalar outputs and lifecycle/fallback, then keeps or rolls back an injected
error and retrieves the new private experience in a fresh process. The original
phase helper runs against a **scalar protocol stand-in**, not real FSDP. Its
timings are synthetic. This tests the workflow and negative control; real GPU,
autograd, multi-rank and long-trajectory qualification remain NOT_RUN.
