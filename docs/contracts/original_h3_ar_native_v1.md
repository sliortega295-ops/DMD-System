# W6 original DiffSynth H3 AR native adapter v1

This **opt-in task adapter** uses the common `distillctl workloads native --task`
front door, `search.service.select`, source-bound `CostModel`, and
`PreparedRolePlan`. It is not a sixth canonical frozen profile. The existing
five-profile registry, defaults, historical receipts and reported times are
unchanged. `workloads native list-adapters` discovers task schemas without
loading models.

## Frozen workload, not a protocol replacement

- Exact DiffSynth KS3 object snapshot: source-receipt SHA256
  `9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989`.
  This is not a verified Git commit. Admission reconciles the **complete Python
  source inventory** against that immutable receipt, not just four entry files.
- Original previous/current AV chunks: current 56 frames at 24 fps (~2.33 s),
  clean previous AV reference, native 768x1344 cached geometry. Actual target
  video `[1,24,17,48,84]`, audio `[2,32,93]`, previous video rows `[17136,96]`,
  previous audio rows `[186,32]`. Text length remains sample-qualified.
- Eight-step original DMD2; per update Student9/Fake2/Teacher1 forwards,
  Student1/Fake1 backwards and real Adam updates; G1/Fake1/EMA0; LoRA64;
  original BF16 autocast with `cache_enabled=False`, original FP32 master Adam.
  The dormant GAN branch remains configured with its original start step500;
  it must be inactive in both fresh cycles. No GAN/EMA/source-step substitution.
- Original `h3_dmd2.DMD2Trainer.update`, `H3Adapter`, `prepare_batch`,
  `dmd2_zero3.configure/initialize/step`, checkpoint helper and RNG progression.
  `H3ARSession` binds those functions; it does not copy/rewrite training math.
- One world16 group, two nodes x eight A100-SXM4-40GB; real validated
  Torch2.10.0+cu128. No world1/world8/Grid1000/fullseq/Diffusers fallback.
  The pinned native startup metadata helper intentionally rejects other GPUs.

## Admitted physical choices

Only two existing native mechanisms can be calibrated:

| ID | Actual native implementation |
|---|---|
| `zero3-native` | Original ZeRO3 communication/checkpoint policy, no added attention-result cache/fusions |
| `zero3-ap` | Accepted AP bundle: native async comm, block leaf/replay release, unit prefetch, pointwise/SwiGLU/RMS/LoRA implementations, CPU-staged selected-attention replay cache with bounded H2D prefetch and version-qualified epoch invalidation, host-register4 pinned allocator |

Both retain no parameter/optimizer offload and no **checkpoint-boundary**
offload. AP is **not copy-free or all-GPU-resident**: its attention replay
cache performs real D2H/H2D transfers and retains pinned host memory. Those
costs remain inside the opaque native update and common host capacity policy.
The saved plan binds every bundle choice, exact helper/library SHA256, source,
checkpoint admission, rank cache manifest, topology, runtime and capacity.

Historical native first-update control (~316.095 s) and accepted AP Cycle2
255.6706443373114 s are provenance only. Neither is embedded as a cost or an
automatic winner. They cannot establish timing on this new integration source.

## Honest planner scope

The single graph operation is **one complete original native update**. Its
version contract after warmup is Student1/Fake1/EMA0 -> Student2/Fake2/EMA0.
EMA0 is read/preserved, never falsely reported as an EMA write.

Each implementation needs fresh unprofiled two-cycle, all16-rank native
calibration receipts. Cost is max16 actual fenced Cycle2 duration; capacity is
post-setup static allocation plus a conservative all-rank observed additional
allocation envelope. No fake G/F components, overlap discounts, historical
scalar costs, or inferred new layouts enter the common planner. A candidate
that is unmeasured or outside common capacity is rejected. Search is limited
to these admitted observed implementations, not a global optimum claim.

`run` revalidates and independently recomputes saved cost/graph/order/timeline/
memory lowering before model construction. It takes its actual implementation
from the saved plan; no recipe override or old shell/worker filename is accepted.
The same persistent native session executes both updates through the common
prepared handler. Original native census/Adam/version/cache epoch and backend
execution witnesses must pass. C1 is warmup, C2 is the real second native
update with no RNG/optimizer reset; native CUDA fences/barriers remain in the
timer. State/RNG hashing and epoch bookkeeping remain outside timers.

## Source/assets and safety

Large checkpoint files are hashed **once in owner CPU/I/O admission before GPU
reservation**, with unchanged before/after stat. The hash-bound audit records
per-file SHA256 plus size/mtime/ctime and origin-host dev/inode. Workers strongly
hash the small audit/config/index/source, check complete payload inventory/stat,
and record a local stat witness; they do not re-read all weights to hash them.
Cross-host stat checks use size/mtime/ctime, never compare mount-local dev/inode.
This is explicit stat-bound reuse of the same payload identity, not a fresh
cryptographic full-payload read on every worker. Any stat/inventory change fails
closed and needs owner re-admission. Same-host dev/inode are also checked.

All source/helper/library pins, cache identity, actual sample geometry and
current capacity are validated before or during the original native path.
No SSH, GPU allocation, cleanup of foreign processes, download, installation,
or new launch scheduler exists in the adapter. An external authorized owner
must coordinate the two-node torchrun and resource sampling. Internal cgroup
`memory.peak` is conservative lifetime highwater, not an isolated workload host
peak; sampled driver-free checks do not prove an unsampled reserve minimum.

## Validation status

Pending cluster validation. CPU tests exercise the common planner, rehashed
plan tamper rejection, full source-inventory rejection, topology, versioned
native prepared handler, real two-cycle dispatcher, capacity policy and
single-hash/stat-bound checkpoint admission. They do not load a real H3 model
or establish native numerical/quality parity. New numerical status remains
`REVIEW_REQUIRES_MATCHED_REFERENCE`; speedup/optimality are not claimed.

## Capacity provenance and new policy relation

The historical C2 closeout reported peak allocated **33.728214263916016 GiB**
and peak reserved **38.416015625 GiB**. Node0/node1 owner lifecycle receipts
reported sampled host peaks **463624593408 / 464590323712 bytes**, sampled
minimum driver-free **2097152 / 60817408 bytes**. Sampling was about 5 seconds;
these minima do not certify an unsampled reserve minimum.

The original two-cycle owner explicitly required **64 GiB cgroup headroom at
entry** and zero OOM/oom_kill increments throughout. No explicit sustained GPU
reserve or sustained host reserve floor was found in that owner/manifest;
those historical policy fields remain **UNKNOWN**, not inferred to be 1 GiB.

The new worker has **no default capacity policy**: all four policy byte fields
must be supplied identically for calibration/manual/selected runs. It retains
the original 64 GiB **entry-only** gate independently. `host_reserve_bytes`
means a separately declared maintained check, not that historical entry gate.
Do not populate it with 64 GiB merely because the original owner had an entry
gate, or import the unrelated GAN 1 GiB GPU reserve. Preserve the authorized
original policy; any additional reserve is a new explicit constraint and may
reject a previously accepted implementation. The new receipt records the
actual startup current/maximum and does not reset native GPU peak counters
between setup, C1 and C2.

Sources: `h3_diffsynth_dmd2_world16_20261010/zero3_200s_r2/steady_two_cycle_20261012_r1/reports/remote_closeout_r1/{world16_two_cycle_result_r2,node_0_lifecycle,node_1_lifecycle}.json`
and `scripts/native_zero3_node_owner_two_cycle_r1.py:89-90` in that same run.
