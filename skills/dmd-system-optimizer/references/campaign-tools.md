# Helpers: use only what the task needs

**The default workflow is the project's runner/checker plus WORK_LOG.** Nothing
on this page is a prerequisite for ordinary optimization. Helpers need Python
3.10+ stdlib and do not launch GPU work. `$SKILL`, `$RUN` and `$REPO` below are
explicit paths. Do not invent input fields just to make a helper accept a task.

## Useful standalone commands

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "参数保留"
    python3 "$SKILL/scripts/dmd_knowledge.py" locate --case helios-crosskv-gan-positive --repo "$REPO"
    python3 "$SKILL/scripts/dmd_perf.py" inspect --repo "$REPO"
    python3 "$SKILL/scripts/dmd_perf.py" target --baseline-seconds 100 --speedup 1.5
    python3 "$SKILL/scripts/dmd_perf.py" compare --input "$RUN/pair.json"

The numbers above illustrate syntax. `compare` is descriptive only, not a
numerical checker or statistical promotion. Keep raw native output alongside
any summary. Context JSON and private stores are optional; inventory requires
the DMD registry. See [reuse and memory](reuse-memory.md) for private recording.

## Advanced opt-in: forecasts and persistent campaigns

Use these only for an explicitly selected formal study or an existing project
requirement. Do not initialize them automatically when this Skill starts.
Do not silently abandon an already frozen protocol after seeing a failed result.

    python3 "$SKILL/scripts/dmd_perf.py" rank --campaign "$RUN/campaign.json" --opportunities "$RUN/opportunities.json"
    python3 "$SKILL/scripts/dmd_perf.py" gate --campaign "$RUN/campaign.json" --opportunities "$RUN/opportunities.json" --candidate C1 --ledger "$RUN/ledger.sqlite"
    python3 "$SKILL/scripts/dmd_campaign.py" init --state "$RUN/ledger.sqlite" --input "$RUN/contract.json"
    python3 "$SKILL/scripts/dmd_campaign.py" reserve --state "$RUN/ledger.sqlite" --run-id C1-probe1 --input "$RUN/reservation.json"
    python3 "$SKILL/scripts/dmd_campaign.py" settle --state "$RUN/ledger.sqlite" --run-id C1-probe1 --input "$RUN/settlement.json"
    python3 "$SKILL/scripts/dmd_campaign.py" status --state "$RUN/ledger.sqlite"

Forecasts use `assets/campaign.template.json` and `opportunities.template.json`.
A persistent campaign instead uses `ledger-contract.template.json` and the
reservation/settlement templates. These are different optional interfaces, not
forms every round must fill. Null means unknown, not fabricated zero.

Gate exit codes are 0 admitted, 2 invalid, 3 probe-only, 4 deferred. Without
`--ledger`, budgets are declarations; with it, identity and consumed plus
reserved costs bind to persisted state. Gate does not reserve or grant resources.
In lightweight work, choose feasible opportunities with reasoned judgment rather
than treating this schema-driven gate as a universal permission service.

Existing schema-2 campaigns freeze workload/source/config/input, checker versions,
tolerances, timing scope and budget. Claim classes are strict (original semantics,
not necessarily bitwise), approximate (explicitly authorized), diagnostic_review.
Use actual runner outputs through [receipt adapters](checker-receipts.md), not
handwritten PASS. Unsupported native formats do not require fake JUnit.

Reservations count all allocated resource costs and launches, including failures.
SQLite prevents concurrent over-reservation by these helpers; it does not police
other processes. Record actual overruns and reconcile unsettled work. Changing
chat, branch, candidate name or tracking tool does not reset an explicit cap.
Schema-1 judgments stay historical; new trusted judgments need the documented
schema-2 evidence. A linked replacement campaign must carry prior costs/evidence,
not silently restart them.

## Advanced opt-in: fixed-sample confirmation

    python3 "$SKILL/scripts/dmd_campaign.py" assess --state "$RUN/ledger.sqlite" --input "$RUN/result.json"
    python3 "$SKILL/scripts/dmd_campaign.py" promote --state "$RUN/ledger.sqlite" --input "$RUN/result.json"

Read [validation](validation.md) before choosing this protocol. Six A-B-A blocks
mean 18 independent process starts; they are an example, not a default budget.
Freeze the valid-sample count, maximum attempts/candidates, effect floor and drift
policy before measuring. Include every valid sample; do not drop slow results.
Predeclared environmental invalid-window replacements need an `attempt-json`
receipt; candidate errors/OOM/numerical failure are not environmental retries.
Every attempt still costs resources.

For correctness/capacity-only campaigns, `timing: null` and
`baseline.seconds: null` avoid artificial measurements. Real CPU checks can be
observed with zero GPUs. A partial timing set remains incomplete, not early
statistical acceptance. Correct capacity value can remain optional with its
latency tradeoff disclosed. Default/control promotion requires the prescribed
numerical, compatibility, composition and fallback evidence, including required
checks against the original baseline. Existing frozen thresholds are unchanged.

A REVIEW or missing sample count is not an instruction to run indefinitely.
At a stop or exhausted allocation, keep the best verified control and partial
results. If a helper cannot represent a failure/transition, retain its error and
native evidence in a plain handoff rather than fabricating rows, deleting samples
or launching more work. The formal verdict stays unresolved until correctly
handled; this fallback does not certify a candidate or reset spent resources.

## Optional structured Memory publication

    python3 "$SKILL/scripts/dmd_knowledge.py" record --store "$RUN/knowledge" --input "$RUN/experience.json"
    python3 "$SKILL/scripts/dmd_campaign.py" finish --state "$RUN/ledger.sqlite" --input "$RUN/finish.json" --store "$RUN/knowledge"
    python3 "$SKILL/scripts/dmd_campaign.py" flush-memory --state "$RUN/ledger.sqlite"

Ordinary projects can use `record` after saving native outputs, or keep WORK_LOG
only. `finish` is for existing campaigns: it combines settlements/assessment with
an immutable outbox record. Use `flush-memory` for a recoverable publication error;
do not retry unchanged malformed input forever. Preserve the pending record,
error and native results if repair is outside scope. Optional indexing cannot
require another GPU experiment or prevent an honest handoff.

These helpers check supplied content, not execution ownership, process independence,
scientific adequacy or long-run quality. Synthetic results qualify workflow only.
The lightweight default does not change advanced scripts or certify their remaining
edge cases; it avoids making them a dependency for every optimization task.
