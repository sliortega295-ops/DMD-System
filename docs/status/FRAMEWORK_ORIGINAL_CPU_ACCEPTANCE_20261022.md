# Public framework and original-model CPU integration

This report supersedes earlier **capability descriptions**, not their historical
measurements. DMD-System now has a common installation, input, training,
checkpoint, inference and extension path, with six distinct original protocols.
The common lifecycle reuses Program, the execution graph and measured planner.
It does not introduce another DMD solver or rewrite the original losses.

This is a CPU/source acceptance report. No GPU, pretrained-quality, capacity,
performance or large-model convergence result is included.

## What an external user can run

| Stage | Public interface and verified behavior |
| --- | --- |
| Install and inspect | Lightweight wheel and offline sdist→wheel installation; `doctor`, configuration and W1–W6 discovery do not initialize CUDA, import Torch/models or download assets. |
| Prepare data | Genuine RGB media, PCM audio and text through an explicitly supplied encoder; safe preencoded tensor import with shape/dtype/sample/content identities. No latent resize or missing-input synthesis. |
| Train | Persistent role models, trainables, optimizers, scheduler/EMA/history and versioned calls through the existing Program. Explicit finite or epoch sampling preserves restart/data order. |
| Select execution | Reference/manual works without search. Measured auto and saved plans bind real runtime implementations and operation order, rather than a fixed historical winner. |
| Recover | Atomic all-rank complete-cycle checkpoints bind local shards, source, layout, data, versions, optimizer and RNG. Fresh-process continuation matches uninterrupted training on the same topology. |
| Generate/export | Portable full inference weights are separate from restart state. The installed tiny example generates actual decoded pixels on genuinely new inputs after old training data/checkpoints are removed. Original Helios current-G export has a separate explicit native **latent diagnostic**, not decoded video. |
| Extend | A new architecture and a different stateful workflow use ordinary model/data/algorithm factories and the existing runtime; neither requires a model-name branch in the solver. |
| Reject and contain | Missing/changed input, wrong shape or dtype, source/cache drift, unsupported layout, incomplete shards and version mismatch fail explicitly. Distributed metric errors propagate collectively; native partial commits poison the owner instead of publishing a false checkpoint. |

The installed demonstration trains actual randomly initialized tiny models. It
is not a pretrained-quality measurement. See the [quick start](../../README.md),
[data guide](../guides/data.md), [training/recovery](../guides/training.md) and
[extension contract](../guides/extensions.md).

## Six protocols, not six renamed benchmark scripts

| Workload | Original-model integration | Current boundary |
| --- | --- | --- |
| W1 Wan | Fresh original-engine cohorts and common restart; separate full original mixed-precision CPU Program/current-G output | CPU1/source-rank0 per-sample is not world16 numerical proof. Historical winner bundles retain their separate contract. |
| W2 Helios No-GAN | Fresh direct reference input/update/state and common restart | Requires an explicit original Step10 warm checkpoint; arbitrary cold native startup is not claimed. |
| W3 Helios GAN | Actual full original tiny transformer/PEFT, history19 producer, original G1/C5/GAN/EMA, both backward policies, common training/restart and current unmerged-G export | CPU1 FP32/dense-SDPA reference; native latent output is zero-history diagnostic only; no VAE/decoded media or pretrained quality. |
| W4 Self-Forcing | Actual original input/LoRA/Adam/reference ownership, fresh cohorts and common same-topology restart | Optimized C51 transport/shadow/two-slot restart is not implemented by this reference route. |
| W5 full-sequence H3 | Original finite fresh-input G1/F5 loop, two-LoRA Adam/scheduler/state and common restart | Grid1000 is explicitly approximate; historical optimized numerical FAIL remains unchanged. |
| W6 autoregressive H3 | Actual cached previous/current AV parser, MiniMaxH3DiT, independent LoRA64 adapters and original 8-step/G1F1/continuous-sigma update through common checkpoints; separate current-Student output companion | Maintained cached-only licensed derivative, not a complete DiffSynth/SFT package. No AV encoder/decoder or new CUDA backend. |

W1/W3/W6 now have separately versioned **public CPU reference lifecycles**; earlier
reports saying those references were entirely missing are no longer current.
The frozen selected-plan benchmark workers are not widened or relabeled as
arbitrary-length production trainers. Native reference limits, dependency lanes
and exact operation support remain in the [capability matrix](../guides/support_matrix.md).

## Original code and state integrity

The Helios full transformer stays byte-identical to its admitted original source.
The explicit dense-SDPA model provider and original helper closure compile
pinned source, not possibly stale `.pyc` bytes. An already loaded unadmitted
helper namespace is rejected rather than overwritten; use a fresh process for
this source-qualified reference. Normal legacy helper imports remain unchanged.
Original rotary-buffer contents and all incoming model/optimizer/RNG state are
staged and checked before mutation.

W6 reuses the locally recovered original numerical code and retained Apache-2.0
support. Its manifest pins 16 output files, five whole original numerical files
and 91 selected original definitions. The limited support bridges are explicitly
maintained, not fabricated full-loader/training-module stubs. Original default
GAN start500 is preserved; the separate GAN-start2 fixture exists only to test
real transition/restart behavior. See [source/license provenance](../../licenses/diffsynth-h3-cached/ORIGIN_AND_CHANGES.md)
and [CPU reference use](../../recipes/h3_ar/CPU_REFERENCE.md).

## Later source-scoped additions

The foundation counts below describe their named commits, not every later
source byte. The [W6 current-Student acceptance](H3_CURRENT_STUDENT_CPU_ACCEPTANCE_20261022.md)
adds installed-wheel export/new-input original AV rollout and hosted
Python3.10/3.12 CPU qualification. The [W1 Full5 acceptance](WAN_ORIGINAL_CPU_ACCEPTANCE_20261023.md)
adds actual mixed-precision original-model training, exact fresh-process
continuation and current unmerged G output. Each has its own source and receipt;
their tests are not summed into a new all-source claim.

## Foundation verification

Source acceptance is split honestly: the general/refactor/Wan/Self-Forcing
regressions ran on `2a341e6a`; the complete original Helios lane ran on
`0c700087` after a four-line **test-only** helper-fixture admission fix. Packaged
`src/` and `runtime/` bytes have not changed since `4fca1ac3`. The installed-wheel
and source-distribution proofs ran on those same production bytes.

| Check | Observed CPU result |
| --- | --- |
| Required refactor | 2153 passed, 20 subtests passed; 85 existing skips |
| General framework/data/Program/organization, helper0.34 lane | 991 passed; 88 CUDA/other-dependency-lane skips |
| Original Wan, native0.35.2 lane | 173 passed; zero skips |
| Original Self-Forcing reference, native0.35.2 lane | 115 passed; zero skips |
| Full original Helios model/owner/data/training/inference,0.37.1 | **187 passed; zero skips** |
| Actual cached-original H3/inputs/public training | **79 passed; zero skips** |
| Original-source CLI/import integration | PASS, 18 tests; five frozen legacy profiles and six project workloads |
| Functional installed wheel | PASS: raw media, real gradients, exact new-process recovery, measured/saved execution, new-input decoded pixels after old training payload removal |
| Lightweight wheel and offline sdist round trip | PASS: isolated install/doctor/six-workload discovery without Torch/model import; eight license/notice files preserved exactly |
| Documentation/package/catalog follow-up | 62 passed; zero skips; 67 local document links resolve |
| CI producer→consumer checks | Both dedicated no-skip/required-case scripts passed against the actual local JUnit files; not a hosted-CI result |
| Static checks | Compileall and 17 scripts' shell syntax passed; no new F/E9 findings (two pre-existing F401 remain); no configured typechecker |

All original regression lanes above used actual **Torch2.10.0+cpu, Python3.12.3**.
The installed functional demo used **Torch2.14.0+cpu**; the lightweight build
Python3.12.14 had no Torch. These are separate claims, not renamed environments.
The [machine-readable receipt](../../artifacts/framework/20261022/public_original_reviewed_cpu.json)
records exact source and test hashes, source-scoped counts, preserved failures
and remaining implementation limits.

Each suite runs in its compatible dependency lane. Helios helpers (Diffusers
0.34), Wan/reference Self-Forcing (0.35.2) and the full original Helios provider
(0.37.1) are **not** one interchangeable environment. Actual source-model proofs
have separate no-skip CI gates. Existing CUDA-only or unavailable optional-source
skips in the general suite are not counted as completed original-model proofs.
The full original Helios and cached H3 dedicated lanes require their actual
fresh-process and native update cases without skips.

The full-cluster review completed ten independent local lenses and one finding
validator. All five confirmed items were handled: source-only helper admission,
immutable rotary buffers, latent/media validator separation, actual W6
post-critic-commit failure containment, and collective host-metric rejection.
The external-provider route produced no usable review; local adversarial review
is not cross-model corroboration. Reports retain the failed attempts, including
the combined-test helper-import ordering failure; no production namespace gate,
assertion, numerical tolerance or canonical protocol was relaxed to pass it.

## Remaining qualification and unsupported operations

- The original W1/W2/W4/W5 owners have source/CPU proofs, not new pretrained
  distributed-CUDA or convergence results. W3/W6 exercise real tiny original
  models, not production-sized pretrained execution.
- W2/W4/W5 still have no unified native generate/export callback here; existing
  upstream tools are distinct. W1/W3/W6 now have separate CPU Program output
  companions, not new CUDA-native output backends. W6 lacks AV encoding/decoding.
  Original Helios export requires the admitted editable source checkout and
  runtime; the lightweight wheel is not that native source bundle.
- Complete-cycle, same-topology recovery is supported. Mid-cycle/cross-topology
  recovery, distributed periodic validation and custom distributed export reject.
  Dense distributed export needs full CPU materialization; production host
  capacity is not qualified by a tiny functional test.
- Earlier published revisions have a hosted native Gloo child shutdown SIGABRT.
  Exact unchanged local reproductions passed, but the cause is **unresolved**;
  this report does not call hosted CI universally green.
- Selected F/E9 lint has no newly introduced findings; two unchanged legacy F401
  diagnostics remain. The project has no configured static typechecking gate.
- User plugins/dependencies are trusted Python, not a sandbox. Restricted loading
  and source hashes do not eliminate all storage check/use races.

The integration remains on a reviewable branch/Draft PR. These limits must be
qualified before claiming all six large-model paths production-ready. Historical
A100/H100 results and numerical REVIEW/FAIL evidence are not changed by this work.
