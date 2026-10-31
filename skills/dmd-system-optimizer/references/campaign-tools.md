# Optional CPU helpers

Use existing project runners and ledgers when they already satisfy the task.
These helpers require only Python 3.10+ stdlib and never launch GPU work.
`SKILL`, `REPO` and `RUN` below are explicit absolute paths. Keep run state in
the project's artifact directory, out of Git. Evidence paths resolve relative to
their input JSON file and include path, sha256 and an exact locator.

## Retrieve and reuse

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --query "参数保留 residency" --context "$RUN/context.json"
    python3 "$SKILL/scripts/dmd_knowledge.py" recall --query "attention FA3" --repo "$REPO"
    python3 "$SKILL/scripts/dmd_knowledge.py" inventory --repo "$REPO" --store "$RUN/knowledge"
    python3 "$SKILL/scripts/dmd_knowledge.py" search --store "$RUN/knowledge" --query "LoRA" --context "$RUN/context.json"

Read [reuse and memory](reuse-memory.md). Public priors and private records remain
separate. Inventory requires the DMD registry; external projects can create their
own implementation records with the experience template.

## Forecast and prioritize

    python3 "$SKILL/scripts/dmd_perf.py" inspect --repo "$REPO"
    python3 "$SKILL/scripts/dmd_perf.py" target --baseline-seconds 100 --speedup 1.5
    python3 "$SKILL/scripts/dmd_perf.py" rank --campaign "$RUN/campaign.json" --opportunities "$RUN/opportunities.json"
    python3 "$SKILL/scripts/dmd_perf.py" gate --campaign "$RUN/campaign.json" --opportunities "$RUN/opportunities.json" --candidate C1 --ledger "$RUN/ledger.sqlite"

`assets/campaign.template.json` describes the forecast; opportunities use their
own template. Null means unknown. A forecast is not an observed result.
Gate exit codes: 0 admitted bounded experiment, 2 invalid, 3 probe-only, 4 defer.
Without `--ledger`, budget checks use declared JSON only. With it, identities must
match and used **plus reserved** cost comes from the persistent ledger.
Gate does not reserve resources; reserve separately before a launch. A concurrent
reservation can exhaust budget after an otherwise valid gate.

The materiality threshold prioritizes exploration. It never decides whether to
keep an already measured 1% gain. `compare --input pair.json` reports descriptive
medians only; read the script's help and [performance model](performance-model.md).

## Persist reservations and actual costs

    python3 "$SKILL/scripts/dmd_campaign.py" init --state "$RUN/ledger.sqlite" --input "$RUN/contract.json"
    python3 "$SKILL/scripts/dmd_campaign.py" reserve --state "$RUN/ledger.sqlite" --run-id C1-probe1 --input "$RUN/reservation.json"
    python3 "$SKILL/scripts/dmd_campaign.py" settle --state "$RUN/ledger.sqlite" --run-id C1-probe1 --input "$RUN/settlement.json"
    python3 "$SKILL/scripts/dmd_campaign.py" status --state "$RUN/ledger.sqlite"

Start from `ledger-contract.template.json`, `run-reservation.template.json` and
`run-settlement.template.json`. Freeze a real baseline, validator hashes,
workload/semantics/hardware/timer/claim identities, budget and risk routes.
Claim classes: strict (original semantic contract), approximate (explicitly
authorized altered contract), diagnostic_review. Strict is not a synonym for
bitwise. A corrected baseline starts a separately documented comparison; it
cannot reset spent resources or earn speedup credit.

Reservation purpose is probe, validation or timing. GPU-hours include every
allocated GPU and authorized occupied time, including failed work. A confirmation
timing block has three independent processes: count all their costs and starts.
Define what a full run means once for this campaign; do not count cycles as runs.
Candidate identity/source/control/risk cannot change under the same candidate id.

SQLite transactions prevent concurrent over-reservation by these helpers.
Identical requests are idempotent. Unsettled reservations survive restarts.
Record actual overruns honestly: they remain in the ledger and block new
reservations. Settle errors/interrupted runs with their available receipts; do not
delete them. Helper accounting cannot stop an independently launched process or
prevent a user from editing/replacing the database.

## Optional frozen confirmation and control promotion

Use this only when the [fixed-sample protocol](validation.md) suits the task.
Exploration probes and descriptive measurements do not require this protocol.
An existing suitable repository acceptance process takes precedence.

    python3 "$SKILL/scripts/dmd_campaign.py" assess --state "$RUN/ledger.sqlite" --input "$RUN/result.json"
    python3 "$SKILL/scripts/dmd_campaign.py" promote --state "$RUN/ledger.sqlite" --input "$RUN/result.json"

Populate `result.template.json` with real runner receipts. All candidate runs must
be settled. Every frozen validation level must be PASS; any explicit FAIL prevents
retention. Each timing row has:

    {"run_id": "C1-block1", "process_ids": ["invocation-a", "invocation-b", "invocation-a2"],
     "control_before": 100, "candidate": 99, "control_after": 100}

Those numbers illustrate syntax only. Use exactly the predeclared repeat count,
all registered timing blocks once, independent unprofiled invocations and matched
work. Confirmation counts apply to candidates with timing reservations; cheap
probes consume resource budget without consuming a confirmation slot.
Example feasible arithmetic: repeat_count=6, max_candidates=1, alpha=.05; with
four confirmations use at least seven blocks to permit significance at all.
Choose actual counts from measurement needs and granted budget.

Assess persists an immutable result for that candidate. Changing its data after
seeing the decision is rejected. Incorrect -> DISCARD, incomplete -> REVIEW,
established benefit -> KEEP. Applicability/fallback PASS and a non-diagnostic
claim are additionally needed for default_enable. A capacity receipt can support
KEEP_OPTIONAL when speed is uncertain. Assess before promote. Promotion advances
the control while preserving the original baseline; it changes no model code.

The helper verifies hashes and declarations, not their scientific truth. It
does not independently run a validator, prove process independence, isolate an
optimizer or certify long-term training quality. Preserve those limits in reports.
