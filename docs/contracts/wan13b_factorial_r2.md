# W1 R2 four-cycle factorial diagnostic contract

Schema: `distillschedule.wan13b_factorial_r2.v1`. This is an opt-in extension
of audited source `45c4940d1ddba89bc65fd6080483296f511916cc`, not a replacement
for the canonical two-cycle timer or the 1/2/5/50-cycle trajectory contract.

## Immutable workload and cells

Use the original world16, B1/global16, 384x640x33, LoRA256 workload, seed
20260903, five-step trace cohort and original bootstrap/model/negative-prompt
identities. Preserve BF16 frozen base, FP32 trainable LoRA and reduction,
matched TF32 override 1, default AdamW and reference GPU EMA. No offload,
streaming, swaps, score packing, new traces, extra optimizer updates or model
substitution is admitted. Initial G2/C10/EMA10 advances by G1/C5/EMA5 per
cycle, ending at G6/C30/EMA30. Exact source/asset hashes live in the versioned
runtime policy, not permissive command-line defaults.

- `baseline`: original world16 FULL_SHARD FSDP2, chunk512; B entirely OFF.
- `only_A`: full resident copies, original async128 MiB all-reduce, Tree,
  chunk2048; B entirely OFF, including the clean-runtime path, runtime
  constants cache and optional instance-local clean model-state optimization.
- `full`: the same complete A plus the exact audited six B switches: Generator
  no-grad LoRA fused epilogue, grad LoRA epilogue, normalization fusion,
  Generator rollout graph, Critic forward/backward graph, capture-scoped exact
  input-cast reuse. Score graph, fused-input LoRA-A, merged shadow, retention
  and torch.compile stay OFF. No later search winner is admitted.
- `only_B`: fail closed. A=OFF cannot supply the resident fixed-address graph
  representation required by the complete B package. A compatible subset must
  not silently replace this cell; it needs an explicit user definition and a
  separately reviewed implementation.

Every cell retains the *existing common* execution order: one Generator commit,
precompute the five B1 rollouts, then five ordered Critic/EMA commits. This is
not a newly introduced schedule, and does not attribute that common order to A.

## Source and launch boundary

The production controller's separate mode is `four-cycle-factorial-r2` with
`--factorial-config` and its exact `--factorial-config-sha256`. Configs are
external run artifacts created after source review, all cells bound to one new
clean descendant HEAD and the same executable file hash inventory. The
historical parent HEAD itself is rejected. The controller lowers every flag;
the node launcher retains executable, source, external Helios runtime, asset,
atomic ownership lock and capacity/process gates. It validates the versioned
config before acquiring the allocation lock. The native worker revalidates the
config, source files, effective flags and actual loaded assets before models.
The shared model builder consumes the same bound policy; legacy callers keep
their placement-derived defaults unchanged.

## C1 qualification in the same allocation

Construct models, optimizers, EMA and runtime caches once. Run one complete
original C1, including lazy graph cold work. After its timer closes and before
C2 opens, check the actual six rollout inventories against the frozen trace,
consumed payload identities, all loss finite flags, versions and mutable
installer/replay/reducer receipts. Gather a source/config/input-bound ordered
world16 observation set with 96 distinct rank-qualified payloads. A missing or
invalid rank stops the run before C2. This gate does not reset weights,
optimizers, EMA, counters, RNG or graph state. It does not add a separate
qualification allocation. Subsequent new graph captures are forbidden.

This is an observed semantic/mechanism qualification, **not strict cross-arm
numerical parity**. Numerical status remains `NOT_RUN` under this contract and
formal timing eligibility remains false. Existing numerical REVIEW evidence
must not be upgraded by this structural gate.

## C2-C4 and collection

Each of C2-C4 is another complete original G1/C5/EMA5 cycle over the identical
ordered rank-slot cohort. Timers retain the original synchronized wall boundary
including end-of-cycle synchronization/barrier. No host evidence extraction or
serialization is allowed inside C2-C4. Post-C4 collection checks finite flags,
all per-cycle census/version inventories, unchanged C1 graph cold counters,
configuration/runtime/source identity, two-host x eight-rank topology, memory,
cgroup OOM deltas and owned teardown receipts.

Report raw C2 rank-maximum seconds and the arithmetic mean of the C2/C3/C4
rank maxima. Exclude C1, never select the fastest cycle, and retain independent
run count n=1. No speedup can be computed from missing/failed rank evidence.
CPU integration results use `SYNTHETIC_TEST_ONLY`; the native collector refuses
that evidence kind and cannot turn it into native or numerical PASS.
