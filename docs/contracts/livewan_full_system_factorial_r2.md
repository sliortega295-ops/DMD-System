# LiveWan full-system factorial R2 (CPU-reviewed experimental entry)

Opt-in run kind: `system-factorial-timing`. It is not a change to the canonical
C51, two-cycle timing, or trajectory contracts. Parent: cc2cec8c278b3a3d3ca77fc1c5cf322efaf1639a.
The canonical 20 JSON include files are SHA-pinned by `livewan_system_factorial_r2`.

## Supported cells

- `baseline`: original all-role world16 FULL_SHARD, physical B1, GPU optimizer,
  no teacher offload, immutable max-noop Student collective padding. Extra
  discarded physical calls remain separately counted; no effective sample is removed.
- `only_A`: full 8G+8Score / trace-balanced slot pairing / actual global B16 /
  Critic physical B2 / original async32 gradient strategy /
  host runahead / split-DMD shadow / C5,C10,C15 shadow hot-sync / depth-1 lookahead.
  B metadata is removed, explicit original 512-length context restored, foreach
  AdamW restored. Native checkpoint is delegated unchanged; body-entry audit
  observes recomputation, without substituting attention, RNG or gradients.
- `full`: exact full C51 optimizer, operators, text-K/V, cast, FA3 checkpoint output
  reuse, splitKV and fused RMS options retained. No stripped C4/C24 arm is used.
- `only_B`: BLOCKED. Full B includes unwrapped-resident/role/B2-dependent paths.
  This entry rejects only_B at controller, node header and plan lowering. It does
  not implement a new FSDP graph/cast backend or redefine B as a compatible subset.

Frozen workload: 368x640, 81 decoded frames (21 latent frames), continuous sigma,
seed 20260830, actual global B16, G1/C5/EMA0 per cycle, original source/assets,
FP32 trainable storage and original BF16/autocast policy. Source prepare aliases
must satisfy the existing byte-identity receipt; no source or capacity gate is removed.

## Four consecutive cycles, one initialization

The same initialization executes four complete native cycles. C1 is warmup;
C2, C3 and C4 are steady-state observations, NOT three independent repetitions.
The collector reports raw C2 and arithmetic mean(C2,C3,C4). No profiler is allowed.

The original continuous runner's endpoint remains immediately after each role's
terminal optimizer work and before next-cycle lookahead/hot-sync. A CUDA event
is enqueued without a midpoint barrier or host read. Thus the next endpoint delta
includes the intervening lookahead/hot-sync; no new global synchronization is
inserted that would destroy the pipeline. The same original final CUDA sync and
world barrier close the four-cycle envelope. Event extraction, state metrics,
serialization and cleanup occur after that boundary. Pipeline completion is the
max Score endpoint per cycle; sequential completion uses max world endpoint,
matching the parent two-cycle collector's convention.

C1 gradient-norm finite flags are produced on-device and read after C4, not inside
steady-state timing. Per-cycle commits/calls are observed at the real endpoint.
This is C1 observed-semantic / deferred finite checking, not a claim of strict
full-state numerical parity. Final state finiteness, source/trace/bootstrap,
versions, census, original cache receipts, node cleanup and OOM counters remain
mandatory. Numerical status remains REVIEW; missing/corrupt evidence fails closed.

The original standalone model-free GPU probes are not launched for R2: topology
is checked in the sole training allocation before model construction (target
Torch2.10+cu128, world16, eight ranks/device indices per distinct node). Original
in-worker transport/group setup still runs. No GPU probe/qualification precedes
or follows a cell as a second allocation.

## Producer / consumer

1. Existing controller generates the original canonical parent and exports arm,
   seed, source and resource inputs. `lower_parent` demands an exact parent,
   then produces the versioned, fully bound effective plan.
2. Real worker `_write_effective_plan_for_run` atomically writes the lowered plan
   only from local_rank0 on each node, before models/timer. Both node SHA values
   and their complete plan contents must agree at collection.
3. Original native loops and `_record_two_cycle_endpoint` observe commits and
   physical/logical invocation counters. Four event endpoints produce R2 receipts.
4. Real rank writer persists all 16 receipts. New collector uses the original
   trace-derived census formulas and original full-C51 mechanism validators;
   only_A requires actual native checkpoint body re-entry and zero SAC reads.
5. Missing ranks/cycles, wrong source/arm/hash, changed updates/census, OOM,
   incomplete full-A transport or enabled B in only_A are rejected.

CPU tests explicitly mock CUDA events, model math and transport, and mark all such
receipts SYNTHETIC_TEST_ONLY. Their summary has native timing fields null and is
not eligible for timing. Real CPU Torch checkpoint tests cover original replay,
outputs, gradients, RNG and Adam, but neither CPU class validates CUDA kernels,
real distributed progress or target Torch2.10 behavior. GPU execution remains
NOT_RUN until supervisor review, target/asset/resource checks and single-cell C1.

## Future invocation (not executed during CPU integration)

Use the existing `run_livewan_crossscale_world16.py`, with all original required
source, asset and node arguments plus `--run-kind system-factorial-timing
--system-factorial-arm <baseline|only_A|full> --system-factorial-seed 20260830`.
Baseline also requires `--execution-mode baseline --pipeline-variant b1
--baseline-placement world16-fsdp --student-rollout-padding max-noop`.
only_A/full require `--execution-mode pipeline
--pipeline-variant c51-continuous-b16-generator-fused-rms`, legacy-hybrid baseline
selector and padding off. Supply immutable four-cycle trace/bootstrap; the existing
trace/bootstrap validator requires cycles=4 and final G4/C20/EMA0. Do not adapt an
old receipt after the fact, rename a two-cycle trace, or mix source HEADs.
