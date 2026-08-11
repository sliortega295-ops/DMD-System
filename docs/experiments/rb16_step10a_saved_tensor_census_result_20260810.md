# Iteration 302: Step10-A saved-tensor ownership census

**Status: Measured diagnostic; PASS. No speedup claim.**

The strict-v4 workload remained fixed at official 384x640, DP8 ZeRO-3, MB1/GAS2/global
batch 16, two selected graphs, two backwards and one `G2 -> G3` update. A pass-through
`saved_tensors_hooks` census returned every original CUDA tensor unchanged and retained
only scalar metadata.

Across all eight Generator ranks, each rank opened six grad-enabled top-level calls and
reported exactly 80 saves for each of:

| shape / dtype | bytes | expected | observed per rank |
|---|---:|---:|---:|
| `[1,11040,5120]` BF16 | 113,049,600 | 80 | 80 |
| `[1,4560,5120]` BF16 | 46,694,400 | 80 | 80 |
| `[1,2940,5120]` BF16 | 30,105,600 | 80 | 80 |

The count is explained by two GAS waves times forty blocks per selected stage. Each rank
reported 1,302 packs, 1,298 unpacks, 40 aggregate signatures and zero census-retained
Tensor references. The difference between pack and unpack counts is preserved as raw
evidence: some tensors saved by autograd are not consumed by backward; it is not hidden or
force-balanced.

All 16 rank records returned PASS. The launch was gated by the frozen Iter295 exact
correctness summary, had no cgroup OOM increment, and reached cgroup peaks of
379,505,057,792 bytes on node0 and 79,576,207,360 bytes on node1. Because detailed JSONL
events were enabled, its timing is diagnostic only.

This proves the shape/count ownership correlation needed to test a bounded pinned policy.
It does not by itself prove that every Nsight copy is produced by these handles, nor does it
justify async prefetch. Iteration 303 therefore changes only these three saved-tensor
classes to synchronous pinned pack/unpack and first requires full strict-v4 correctness.

## Evidence

- `artifacts/runs/rb16_step10a_iter302_saved_tensor_census_20260810_001/`
- `artifacts/runs/rb16_step10a_iter302_saved_tensor_census_20260810_001/summary/saved_tensor_census_summary.json`
- summary SHA-256 `a23dcce8d5cce67157c2e21101e76c95901e82eca401582b8c29f125ad652399`
- `artifacts/manifests/iteration302_step10a_saved_tensor_census_result.json`
