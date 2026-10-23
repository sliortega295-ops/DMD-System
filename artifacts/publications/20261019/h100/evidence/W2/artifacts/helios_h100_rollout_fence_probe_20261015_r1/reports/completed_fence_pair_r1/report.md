# W2: retained G, then only internal rollout-fence elision

Exact source: 8fc70e8ce9f6e0243604812120119c9f4c078e55. Native, one independent trial per arm; no profiler. Denominator: original world-max completed Full5 timer, G1/C5/EMA5.

| Arm | Full5 s | Rank/slot execution | Snapshot allocated GiB | Snapshot reserved GiB |
|---|---:|---|---:|---:|
| Retained-G control | 95.534631493 | 8 physical /16 independent slots PASS | 52.008 | 55.125 |
| Same retained-G + interior fence elision | 94.311432633 | 8 physical /16 independent slots PASS | 52.008 | 55.125 |

Single-pair observed saving 1.223199s; ratio 1.012970x. No median, CV or significance claim. Final G3/C15/EMA15; retained41units restore inboth logical B1 scopes; no remaining unsharded units; model swap/streaming/transition0. Candidate actually recorded0 interior rollout fences versus2G/10Critic-rollout syncs and2G empty_cache calls incontrol. Outer timer fences and retention pre/post allocator flush remain.

All actual validate/correctness/native receipts PASS; no new cgroup/CUDA OOM. Original own teardown had0 GPUcontexts. This does NOT assert the node is idle now; unrelated tasks appeared later.

Full-state cross-arm numerical parity NOT_RUN; formal speedup eligibility remains FALSE. The frozen official-role-swap reference138.371948s is from a different revision: its descriptive ratio 1.467181x is not a newly same-revision placement-only experiment. Memory is maximum across recorded snapshots; native reset policies prevent treating it as an independently continuous peak.
