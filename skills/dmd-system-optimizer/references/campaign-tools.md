# Optional CPU helpers

Use existing project runners and ledgers when they already satisfy the task.
These helpers require only Python 3.10+ stdlib and never launch GPU work.
`SKILL`, `REPO` and `RUN` below are explicit absolute paths. Keep run state in
the project's artifact directory, out of Git. Evidence paths resolve relative to
their input JSON file and include path, sha256 and an exact locator.

## Retrieve and reuse

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --query "参数保留 residency" --context "$RUN/context.json" --store "$RUN/knowledge" --brief
    python3 "$SKILL/scripts/dmd_knowledge.py" recall --query "attention FA3" --repo "$REPO"
    python3 "$SKILL/scripts/dmd_knowledge.py" inventory --repo "$REPO" --store "$RUN/knowledge"
    python3 "$SKILL/scripts/dmd_knowledge.py" search --store "$RUN/knowledge" --query "LoRA" --context "$RUN/context.json"

Read [reuse and memory](reuse-memory.md). Private directories stay physically
separate; explicit `--store` options include them in the same retrieval. Inventory requires the DMD registry; external projects can create their
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
`candidate_id` fixes source/config/input/control/risk; `run_id` is the attempt id.
Validation attempts also name their comparator source and validation scope.
Use schema 2 contracts with frozen checker selection and version; see
[parsed receipts](checker-receipts.md). Schema 1 ledgers remain readable and their
costs remain charged, but raw declarations cannot earn new KEEP/default decisions.
Pre-upgrade judgments without saved input JSON remain historical and cannot be
revised safely; preserve their ledger, then explicitly create a linked campaign
carrying spent costs and current baseline evidence. Never silently reset budgets.

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

For correctness/capacity-only work, set `timing: null` and allow
`baseline.seconds: null`. Keep `measurement_kind: observed` for real CPU checks;
no artificial timing baseline is needed. Timing reservations and speed claims
are disabled in that frozen campaign. If later opening a separate confirmation
campaign, explicitly carry the previous evidence and remaining authorized budget;
do not reset spent costs or treat validation-only work as a speed baseline.

    python3 "$SKILL/scripts/dmd_campaign.py" assess --state "$RUN/ledger.sqlite" --input "$RUN/result.json"
    python3 "$SKILL/scripts/dmd_campaign.py" promote --state "$RUN/ledger.sqlite" --input "$RUN/result.json"

Populate `result.template.json` with real runner receipts. All candidate runs must
be settled. Every frozen correctness level must be PASS; a numerical failure
prevents retention. Auxiliary capacity/default-enable checks remain separate.
Each timing row has:

    {"run_id": "C1-block1", "process_ids": ["invocation-a", "invocation-b", "invocation-a2"],
     "control_before": 100, "candidate": 99, "control_after": 100}

Those numbers illustrate syntax only. Use exactly the predeclared repeat count,
all valid registered timing blocks once, independent unprofiled invocations and matched
work. Confirmation counts apply to candidates with timing reservations; cheap
probes consume resource budget without consuming a confirmation slot.
Example feasible arithmetic: repeat_count=6, max_candidates=1, alpha=.05; with
four confirmations use at least seven blocks to permit significance at all.
Choose actual counts from measurement needs and granted budget.
Six A–B–A blocks are 18 process starts: reserve them for a shortlist, not every edit.

`repeat_count` is the fixed number of valid samples; `max_attempts` separately
caps all timing attempts for one candidate. Freeze `replaceable_failures` before
the first run: supported reasons are `launch_failure`, `environment_interruption`
and `invalid_timer_window`. Empty means no replacement. An excluded settlement
must contain an `attempt-json` receipt and matching `sample` with `status: invalid`,
`valid_window: false`, `failure_origin: environment` and the exact reason. It must
end `error` or `interrupted`. The adapter reads these facts from runner output.
Valid slow observations still count. Candidate errors/OOM/numerical failure cannot
use this route; all attempts and overruns remain charged to the resource ledger.

Assess appends an immutable `decision_revision`; status exposes the entire history.
Identical evidence/attempt sets replay the same revision. REVIEW stays open: reserve
a validation attempt, settle its parsed receipt, and assess again with unchanged
timings and accumulated evidence. Complete timing inputs are frozen on first
assessment, including when V2 is missing. Recorded failures cannot be erased;
new failure evidence can invalidate a previous KEEP. Incorrect -> DISCARD,
incomplete -> REVIEW, established benefit -> KEEP. `composition`, `compatibility`,
`fallback`, and the risk-route checks against the original baseline must pass
before default_enable. Current control remains the timing comparator.

With correct risk-route checks, a parsed `capacity` check, `capacity_evidence`
and explicit `latency_tradeoff`, empty `timings` can yield KEEP_OPTIONAL, default
off and performance NOT_EVALUATED. This can later add a fixed speed confirmation.
Missing speed evidence is not a speed claim. Assess before promote; only the
latest decision may advance control. Promotion changes no model code.

## Settle, decide, and write private Memory

    python3 "$SKILL/scripts/dmd_campaign.py" finish --state "$RUN/ledger.sqlite" --input "$RUN/finish.json" --store "$RUN/knowledge"
    python3 "$SKILL/scripts/dmd_campaign.py" flush-memory --state "$RUN/ledger.sqlite"

Use `assets/finish.template.json`. `settlements` optionally contains pending
`{run_id, settlement}` rows; `result` is the full assess input. `experience`
describes the mechanism (`pattern`), conditions, correctness/performance conditions,
retry and failure scope, optionally a complete ReuseDecision/source reference.
Outcome, decision revision and evidence are derived from the ledger. Keep the
summary descriptive; generated outcome/qualification govern its interpretation.

The transaction commits settlements, decision and an immutable outbox record.
Publication is idempotent. If writing the private store fails, the committed
decision and pending record survive; fix the path/evidence and run `flush-memory`.
A retry cannot create another record or replace an earlier observation. Use a new
decision/record for new evidence. Status reports `memory_pending`.

The helper reparses receipts, but does not independently run a validator, prove
process independence, isolate an optimizer or certify long-term training quality.
The contract's `measurement_kind` is observed or synthetic; synthetic receipts
cannot enter an observed campaign. Synthetic decisions qualify workflow only.
