# Small CPU helpers

Use the project's runner, numerical checker and WORK_LOG. These Python 3.10+
standard-library commands are conveniences, not admission or approval services.
They never launch GPU work, reserve resources, certify correctness or promote code.

## Find experience and source

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "参数反复上卡"
    python3 "$SKILL/scripts/dmd_knowledge.py" case --id helios-crosskv-gan-positive
    python3 "$SKILL/scripts/dmd_knowledge.py" locate --case helios-crosskv-gan-positive --repo "$REPO" --show-code
    python3 "$SKILL/scripts/dmd_knowledge.py" sources --query "H100"

Optional context describes observed facts. An explicitly selected private store
can be added with --store. Inventory requires the DMD implementation registry;
external projects need not migrate their code. See [Memory and reuse](reuse-memory.md).

## Inspect or calculate

    python3 "$SKILL/scripts/dmd_perf.py" inspect --repo "$REPO"
    python3 "$SKILL/scripts/dmd_perf.py" target --baseline-seconds 100 --speedup 1.5
    python3 "$SKILL/scripts/dmd_perf.py" compare --input "$RUN/pair.json"
    python3 "$SKILL/scripts/dmd_perf.py" dag --input "$SKILL/examples/synthetic-dag.json"

`inspect` accepts a Git root or a source archive. Archives return unknown Git identity,
not a fictitious clean commit. It no longer loads retired best-known execution policies.

The numbers and DAG example are illustrative, not benchmark evidence.
`compare` accepts timer_scope, semantic_contract, hardware_id, unprofiled=true,
control_seconds and candidate_seconds arrays; paired=true adds paired differences.
It reports all supplied samples descriptively and never returns numerical acceptance
or statistical certification. Keep the actual project output beside any summary.
`dag` calculates an explicitly supplied dependency graph; it does not discover
contention, capacity limits or a physical optimum.

## Save useful experience

Write native outputs and decisions in WORK_LOG first. For optional indexing:

    python3 "$SKILL/scripts/dmd_knowledge.py" record --store "$RUN/knowledge" --input "$RUN/experience.json"

Use assets/experience.template.json only when indexing is useful. Distinguish
observations, hypotheses, failures and incomplete checks. Preserve source and result
paths and keep private stores out of public commits. An indexing error does not
require another experiment or prevent an honest handoff.

The previous campaign ledger, receipt/reuse-form adapters, rank/gate commands and
associated forms are removed, not hidden behind an advanced mode. Existing users'
run databases and evidence are not deleted or modified by this cleanup. Export a
handoff with the previously used version when needed; never fabricate success or
reset costs to migrate a record. Historical source is recoverable from Git, not
part of the current installer.
