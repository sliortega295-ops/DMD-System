# Wan1.3B Resident vs world16 FSDP2 matched Nsight evidence

This directory publishes the compact, auditable evidence from
`iter327_wan13b_resident_nsys_matched_20260822`.

- Full report: [`report.md`](report.md)
- Matched comparison: [`analysis/resident_vs_fsdp2_matched.json`](analysis/resident_vs_fsdp2_matched.json)
- Comparison figure: [`analysis/resident_vs_fsdp2_matched.png`](analysis/resident_vs_fsdp2_matched.png)
- Resident timeline: [`analysis/resident/wan13b_nsys_timeline.png`](analysis/resident/wan13b_nsys_timeline.png)
- Run contract and status: [`manifest.json`](manifest.json)
- Raw report/SQLite identities: [`manifests/raw_artifact_identity.txt`](manifests/raw_artifact_identity.txt)

The raw `.nsys-rep` and SQLite files are intentionally not committed: together
they are about 3.0 GiB. Their paths, sizes, and SHA256 identities are preserved
in the manifest above and the raw data remain on the experiment server.

## Claim boundary

The result is `PASS_DIAGNOSTIC_ONLY`. The matched profiled per-GPU exact-range
wall was 77.336 s for world16 FSDP2 and 70.650 s for Resident, a 6.686 s gap.
Profiled wall time is not a formal speed denominator. Numerical cross-placement
parity was waived only for this diagnostic, so this bundle does not claim a
correctness-equivalent or formal performance win.

## Related unprofiled diagnostic

The earlier unprofiled Iterations 321--322 evidence, including the failed
cross-placement numerical gate, is published at
[`../wan13b-resident-vs-fsdp2-unprofiled-20260820/`](../wan13b-resident-vs-fsdp2-unprofiled-20260820/README.md).
