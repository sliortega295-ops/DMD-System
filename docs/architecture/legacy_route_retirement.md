# Retirement of early Route-A and Route-B prototypes

The pre-2026-08-15 Route-A/Route-B offload and fixed-role pipeline experiments
are no longer part of the current delivery. Start from [current recipes](../../recipes/README.md)
and `distillctl workloads list`. This retirement does not deprecate FSDP2,
activation offload, or multi-role scheduling as mechanisms.

Removed paths include the old route launchers, their exclusive runtime modules,
manual frozen-block streaming and pinned-materialization helpers, old plans and
runbooks, and tests whose only subject was those retired runners. The old Helios
typed launch adapter, which forced the Route-A CPU-offload policy, is also removed
from the default registry. Current Helios reference helpers and public/native
training entrypoints remain.

The optimizer Skill removes the same lineage from its case pack, implementation
map and fallback source index: 25 cases, 6 implementation entries and 25 source
entries. It retains 109 scoped entries, including current framework/FSDP2-era
work, all 16 H100 entries and all 25 ResearchBoost entries. No embedding or
runtime exclusion heuristic is needed to hide the old recommendations: their
content is absent from the distributed indexes.

Age alone does not determine removal. Current training still uses older
checkpoint readers, negative-prompt and trace helpers, sparse EMA, official GAN
loss bindings, and bootstrap/precast preparation. These stay at their audited
paths. The immutable upstream source snapshot and source-import/publication
receipts retain their bytes. The adapter registry change has an explicit appended
source-layout hash-chain entry.

[The retirement manifest](legacy_route_retirement.json) records each removed
path and its previous SHA-256, retired experience/implementation IDs, and concrete
shared dependencies retained. It is an audit inventory, not a retrieval source.
Recover the previous files only for an explicit historical investigation from
[commit f14447c7](https://github.com/sliortega295-ops/DMD-System/tree/f14447c7d733a31fcb5ed4ce4de4ca0bfeeaad1e).
This is a source retirement, not a parity-certified replacement or a new GPU
performance result.
