# Append-only unit log

- 2026-09-07: Reused the completed 16-rank LiveWan action observations. Added a source-bound CPU pool graph, not another worker. 96 focused tests pass. Native graph execution, service-cost and storage-lifetime admission remain NOT_RUN/UNADMITTED/UNRESOLVED.
- Remote CPU derivation PASS; exact-byte collection r1/r2 failed closed on serialization differences; r3 verified original remote bytes without rerunning native or analysis.
