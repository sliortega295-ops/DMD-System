# Curated result manifests

This directory stores small, reviewable summaries that connect a code/config identity to an experiment outcome. Raw runs remain under ignored storage such as `artifacts/runs/`, the research NAS or an immutable release bundle.

A filename ending in `_result.json` does **not** imply success. Every manifest must carry an explicit status and claim boundary.

## Required principles

A curated manifest should include:

- schema version and run ID;
- source/config/plan identities or pointers to immutable launch manifests;
- frozen workload contract and rank topology;
- explicit functional status;
- timing status and baseline identity, when evaluated;
- OOM and host-reserve outcomes;
- checkpoint/evidence inventory; and
- limitations that prevent broader claims.

For Route-A world16, use the status vocabulary defined in [`../../docs/status/routea_world16.md`](../../docs/status/routea_world16.md). Partial evidence, a launcher failure or an interrupted JSONL tail must remain visible; the summarizer may ignore only one unterminated final JSONL record and may never skip malformed complete records.

## What does not belong here

Do not commit:

- full launcher/rank logs;
- phase traces or tensor dumps;
- checkpoints and model weights;
- `.nsys-rep` or SQLite profiler output;
- generated videos/evaluations; or
- repeated copies of failed one-off launch scripts.

A small failure summary is acceptable when it establishes a reusable mechanism or regression. Hundreds of raw attempt manifests are not.
