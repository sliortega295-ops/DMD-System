
## 2026-09-07T12:06:37.843420+00:00 H3 recovery preflight
{
  "status": "GO_H3_ONLY_NEW_SOURCE_RECOVERY",
  "utc": "2026-09-07T12:06:37.843420+00:00",
  "source_head": "b1706f073d09a5de03926c6e216273898af754f3",
  "source_clean": true,
  "tests": {
    "passed": 768,
    "skipped": 17
  },
  "prior_r2_failure": "Pinned config loader representation mismatch, not math mismatch/OOM",
  "fix": "Path provenance tied to H3_CONFIG + LR/eps values normalized exactly as pinned float optimizer inputs; changed values/unknown keys remain rejected",
  "native_loader_and_cpu_trainer": "PASS_BOTH_NODES",
  "assets_unchanged": true,
  "gpu_ownership": "two owned A100 nodes; no other GPU apps at fresh probe completion",
  "atomic_lock": "/tmp/distillschedule_gpu_node.lock through audit construction run teardown",
  "implementations": [
    "joint-h3v1-d1-n128",
    "joint-h3v1-d2-n128"
  ],
  "cycles": 2,
  "timing_cycle": 2,
  "profiler": "NOT_RUN",
  "native_timing_budget": 3,
  "prior_timing_attempts": 0,
  "livewan": "COMPLETE at 06d5e67; not rerun or rebound to H3 source",
  "raw_evidence_preserved": true
}

## 2026-09-07T13:13:52.089363+00:00 H3 numerical gate stopped

Both depth1/depth2 completed 2 cycles and 16/16 native runtime/census/replay/resource gates. Sample identities and default Torch RNG boundaries matched; sampled state/loss comparison FAIL. No calibration or search/timing validation started. Thresholds unchanged. First Fake loss is bitwise equal in rank0; drift appears after the first Fake backward/update. One unchanged depth1 two-cycle correctness repeat is authorized within this debugging scope to measure self-repeat drift; it is not a new optimization, timing candidate, or gate override.

## 2026-09-07T13:56:25.413309+00:00 outcome and stop

LiveWan native validation completed: matched cycle2 62.6378984375 / 47.13862890625 = 1.3288018742 diagnostic; incumbent rediscovered, no strict winner. H3 depth1/depth2 and unchanged depth1 repeat each finished 16/16, G2/C10/EMA0, 31F/6B per cycle and Grid 300/600/600/0. Both cross and self comparisons FAIL while default RNG/sample gates PASS. First Fake loss bitwise equal on all ranks, second Fake loss differs on all ranks, including self-repeat. The depth knob is not necessary for drift. Root operation remains unresolved; no thresholds changed; H3 calibration/search/native timing NOT_RUN. Both nodes cleaned, locks free, OOM delta zero. Stop GPU work; preserve evidence and publish scoped source/docs in existing Draft PR28.
