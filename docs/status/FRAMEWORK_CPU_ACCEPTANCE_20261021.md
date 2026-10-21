# Public framework: implementation and CPU acceptance

DMD-System provides a common data, training and inference lifecycle on its
existing Program, execution graph and measured planner. This report separates
that exercised framework from each original pretrained-model adapter. It
contains no new GPU performance or generation-quality result.

## Executed public lifecycle

| Requirement | Implementation and verification |
| --- | --- |
| Installation and discovery | A lightweight wheel installs outside the checkout; doctor and six-workload discovery import no Torch or model code. The functional wheel reuses existing CPU dependencies without downloads. |
| Raw and preencoded input | Real RGB frames/video, PCM WAV and text; explicit trusted encoder; hash-bound native feature trees with shape/dtype/sample identities. Missing inputs never become random features. |
| Continuous updates | Persistent models, optimizer ownership, scheduler/EMA/history, logical versions and an ordered sample cursor; actual DMD autograd and optimizer updates, not independent initialized trials. |
| Allocated distributed execution | One device per process in an explicit torchrun group. CPU/Gloo tests exercise reference, DDP, FSDP2 and four-rank HSDP, including keep-forward variants. Existing runtime owns layouts and communication. |
| Recovery | Atomic all-rank local-shard checkpoints at complete zero-gradient cycle boundaries. Fresh-process continuation matches uninterrupted module/optimizer/scheduler/EMA/history/RNG/cursor/version/policy state on the same source, input and topology. |
| Held-out diagnostics | Optional local-only model callback, fixed explicit records, exact RNG/mode restoration and declared-state mutation rejection. Fixture metrics are not pretrained-quality measurements. |
| Export and generation | Inference weights are separate from restart checkpoints. All owners participate in sharded export/evaluation. Local generation uses genuinely new prepared inputs without the original training artifacts; PNGs preserve actual decoder pixels. |
| Optional planning | Explicit reference/manual and measured auto/saved modes execute their admitted layout/order. Restart retains the saved policy instead of recalibrating or selecting a historical winner. |
| Independent extensions | An external model and a different stateful history/EMA/scheduler workflow use the existing Program ABI without generic solver model-name branches. |
| Failure handling | Data/source/shape/dtype drift, invalid cache/version, unsupported policies, incomplete or malformed shards, rank-control disagreement and ownership mismatches reject. Rank-local admission and metadata I/O failures propagate to peers without destroying caller-owned groups. |

The tiny Teacher is frozen random initialization. Its explicit media encoder
and decoded outputs prove interfaces and real updates, not pretrained quality.

## CPU verification and exact source scope

The local environment is Python 3.12 / Torch 2.14.0+cpu. A disposable
configuration dependency overlay was reused; no existing environment was
replaced. The public train extra requires Torch >=2.10. Native extras keep
their separately qualified floors. Minimum-version CI and its exact tested
head are reported separately by the PR checks.

At `e1e971f5307b86b88f6013abf52e8abf66481fc0`:

| Check | Observed result |
| --- | --- |
| Required refactor suite | **2153 passed, 85 skipped, 20 subtests passed** |
| Original Wan strategy/backend suite | **157 passed, 7 real-GPU skips** |
| Framework, data, open-workload and organization | **511 passed, 2 real-CUDA skips** |
| Original source/CLI/import audit | **PASS, 18 tests** |
| Functional installed wheel | **PASS_INSTALLED_FRAMEWORK_CPU**: real media, updates, fresh-process restart, measured/saved execution and independent new-input generation |
| Lightweight installed wheel | **PASS_ISOLATED_WHEEL**: six workloads, no Torch/model import |
| Offline source-distribution round trip | **PASS**: sdist → wheel → isolated install → doctor/discovery; all six original license/notice files remain byte-identical |

The only later runtime edit at
`9c182f195d241e5d13e1582b710f96b9157d4402` removes an unused local binding.
Its affected norm tests pass (**12 passed, 36 CUDA skips**), source admission
passes again, and F/E9 lint passes on all 40 continuation Python files. The
installed lifecycle is also rerun after that edit. The tutorial now selects a
record absent from its training fixture; architecture documentation identifies
the actual execution owner rather than an unused abstract interface.

The current-delta eight-lens review completed; all four confirmed P2 findings
were fixed and host-tested. They cover native factored/flattened optimizer
restore, collective continuation admission, bounded DDP telemetry and collective
metadata-failure handling. Different-provider review did not authenticate; its
failure and completed local adversarial fallback are retained. No cross-model
corroboration is claimed.

The earlier Triton-import and fresh-run source-census failures were fixed, not
skipped or assigned new goldens. One lightweight-build invocation selected an
environment without pip; the same source passed with the existing build Python.
Historical failures and the [older frozen receipt](../../artifacts/framework/20261021/cpu_acceptance.json)
remain unchanged. The [continuation receipt](../../artifacts/framework/20261021/cpu_acceptance_continuation.json)
records source hashes, suite scopes and output hashes. Existing selective
checkpoint/scheduler warnings remain visible; they do not establish future
Torch-version behavior.

## Six native integrations: implemented boundaries, not six complete trainers

All six original protocols retain separate source, asset, input, precision,
update and execution contracts. Selected-plan parameters reach original workers.
W5 approximate Grid1000 is never relabeled as W6 continuous-sigma AR DMD2.

W1 additionally exposes original-engine fresh-cohort continuation and opt-in
complete-cycle checkpoints over trainables/Adam/scheduler/EMA/clocks/RNG/cursor.
Its source dispatcher and original scalar engine are CPU tested;
pretrained-model/CUDA restart is **NOT_RUN**. Literal historical winner bundles
retain their two-cycle trace restrictions.

**W2-W6 still expose their bounded original worker APIs.** Arbitrary native long
training, complete native restart, native decoding and portable native export
are **not exposed** by this binding. Several upstream trainers have save hooks;
they require real data/source/topology adapters before becoming public training
capabilities. W6's archived snapshot is not a complete licensed upstream source
distribution. These are remaining implementation/source gaps, not merely
unexecuted GPU tests. This integration must not be described as full delivery
of six production-native training lifecycles.

The generic Program lifecycle does not support mid-cycle/cross-topology
recovery, hidden multi-device ownership, distributed periodic validation or
custom distributed export. Dense distributed export currently materializes
full CPU state on participating ranks; production host-capacity validation is
not claimed. Native optimizer loaders own their non-pointwise state schemas;
resident Adafactor and LBFGS continuation are tested, while FSDP optimizer
admission retains its narrower original restrictions.

See the [capability matrix](../guides/support_matrix.md),
[training guide](../guides/training.md) and
[native guide](../guides/native_workloads.md). Historical A100/H100 and numerical
REVIEW/FAIL records retain their original source and scope.

New GPU runs, SSH connections, model downloads, performance trials, main
changes and old-queue resumes: **zero**. The integration remains a reviewable
branch, not a production-readiness claim.
