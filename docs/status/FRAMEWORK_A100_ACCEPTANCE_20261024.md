# Bounded A100 framework acceptance — 2026-10-24 UTC

This is functional GPU qualification, not a new speedup, pretrained-quality
or mature-release claim. Existing failures and unsupported combinations remain
explicit. No H100 work, model downloads or production-default changes were made.

## Official reference lifecycles: actual CUDA PASS

Production source: `d526101e11316d4ed9926bacaa93b9a2e5a6b181`.
The three test-policy files listed in the machine-readable receipt are the
only code changes for this acceptance; no training mathematics, tolerance or
assertion was changed. Official source pins remain those of the adapters.

| Path | Actual model scope | GPU tests | What passed |
| --- | --- | ---: | --- |
| Official DMD2 SD1.5 | Random tiny original SDUniModel/SDGuidance with actual UNet/CLIP/VAE | 2 | Native-oracle update/state/RNG, fresh-process restart, current-G export and new-prompt image |
| Official DMD2 SDXL | Random tiny original dual-text-conditioned native path | 2 | Native-oracle update/state/RNG, fresh-process restart, current-G export and new-prompt image |
| Official Self-Forcing | Random tiny original CausalWan/Wan/DMD, 21 latent frames | 3 | Native oracle, checkpoint/EMA/RNG, fresh-process restart, current-G output/CPU inference relocation, causal/cache attention gradients |

Total: **7 passed, zero failures, zero skips**, 346.11 seconds of test-suite
wall time. Tests ran serially on one owned A100-SXM4-40GB, not as distributed
training. Runtime: Python 3.10.20, Torch 2.10.0+cu128, Diffusers 0.37.1,
Transformers 4.51.3, PEFT 0.17.1. Only existing source/dependency assets were
used; the original native training environment was not upgraded.

### Exact equality requires an explicit deterministic test policy

The first meaningful CUDA attempt failed an exact state assertion. A bounded
SD1.5 diagnostic compared the original trainer oracle, the public owner and an
independent same-owner repeat, from the same initial state and RNG:

| CUDA policy | Loss/RNG exact | Unequal model tensors (oracle vs owner) | Unequal model tensors (owner repeat) |
| --- | --- | ---: | ---: |
| Default nondeterministic | Yes | 54 | 119 |
| Deterministic algorithms + CUBLAS workspace | Yes | 0 | 0 |

The deterministic diagnostic also had zero unequal gradients or optimizer
state tensors. The tests now explicitly apply that same policy in every
fresh CLI process. This fixes an unstated test precondition; it does **not**
prove default CUDA kernels deterministic, relax numerical criteria or change
the framework's production defaults. The initial failure is preserved.

## W1 full native preset: actual world16 functional PASS

The unchanged clean `d526101e` source was executed through the public
`distillctl train` entry, not a hand-written substitute training command:
`wan_ar.incumbent21a.v1`, two nodes of eight A100-40GBs, B1/global16,
384x640x33, LoRA256, G1/C5/EMA5 per cycle. The original pretrained backbone,
hash-pinned systems-only bootstrap and **fixed synthetic feature trace** were
used. This is not real-data quality or fresh-sample training evidence.

- **16/16 ranks, 32/32 rank-cycles completed**; finite and source-derived
  model-call census checks passed on every rank.
- Native versions advanced G2/C10/EMA10 -> G4/C20/EMA20.
- Actual runtime counters verified Resident, asynchronous gradient reduction,
  chunk2048, LoRA/norm fusion, rollout Graph/cast reuse and Critic gradient
  Graph; no merged weights were enabled. Critic Graph recorded ten backward
  replays per rank across the two cycles.
- Reported peak allocated/reserved across ranks: **23.4094/30.0020 GiB**.
  Scope is the existing native counters since the last reset, not total
  process-lifetime allocation. Both node owners recorded no new cgroup OOM,
  no remaining GPU process and completed birth-bound cleanup.
- Node1's SSH transport timed out, but both independent node lifecycle
  receipts and all 16 final rank receipts were recovered from persistent
  storage. No second GPU launch was used to replace that transport result.

This proves the public preset installs and executes the complete existing
bundle. It does **not** prove a new numerical comparison, speedup, checkpoint
restart or generation: the fixed-trace v1 entry explicitly does not expose
those lifecycle operations. Its saved final state is evidence, not a complete
restart checkpoint. Numerical comparison remains
`NOT_RUN_REQUIRES_MATCHED_REFERENCE`.

### Initial fresh-stream refusal (preserved historical result)

A separate, single CPU admission attempted two fresh RNG cohorts using existing
source-shaped synthetic features and the unchanged native input configuration.
It failed before any GPU work with:

```text
wan.capture: all-zero history uses the unsupported rank-local amplified branch;
no silent resampling or disabling of dropout is allowed
```

Therefore the new optimized stream's world16 save/resume was **not run**.
The failed partial capture is preserved. Do not select a favorable seed,
turn off native history dropout, relabel the strict-global benchmark as normal
training or replay a fixed trace to claim this gap closed. Supporting that
valid branch needs explicit native sampler/collective alignment and Graph
signature-lifetime work before the continuation acceptance can proceed.

### Resident stream follow-up: actual world16 continuation

The separate `3d7bb03982890e1b5bd46f8510243009902a3d47` source implemented
the original rank-local amplified-history branch under the explicit
`source_native_amplified_v1` policy. It did not disable history dropout,
resample the seed, change the source-shaped features, substitute strict-global
sampling, enable merged weights or alter the original loss/update handlers.
The old refusal and all frozen fixed-trace contracts remain intact. Only the
Resident stream admits this branch; FSDP/HSDP collective alignment is not
inferred from it.

| Native run | Completed ranks | Rank-cycles | Complete-cycle checkpoint |
| --- | ---: | ---: | --- |
| Uninterrupted, two cohorts | 16 | 32 | Cycle 1 and 2 |
| New process, resume cycle 1 to 2 | 16 | 16 | Cycle 2 |
| Same-start resumed repeat, noise diagnostic | 16 | 16 | No additional restart checkpoint; full final state evidence |

Seed42 and the unchanged native input configuration yielded 62 and 58
rank-local amplified-history entries in the two prepared cohorts. All runs
completed G4/C20/EMA20. All-rank original call census, sample/cohort and
committed-version checks matched. The uninterrupted/resumed comparison
verified the complete 16-shard G/C/Adam/scheduler/EMA state plus RNG, cursor
and versions: RNG and non-tensor metadata matched exactly, but floating
tensor state did not satisfy the unchanged strict `rtol=1e-5, atol=1e-7`
comparison. Its status remains **REVIEW**.

To distinguish restoration defects from default CUDA execution noise, the
same cycle-1 checkpoint was resumed one more time without changing the
configuration. This repeat also received strict **REVIEW**. For complete
replicated rank0 state, the restart distance divided by the same-start repeat
distance was:

| State | Restart / repeat L2 distance |
| --- | ---: |
| Generator trainables | 0.984 |
| Critic trainables | 0.927 |
| Generator Adam moments | 0.990 |
| Critic Adam moments | 0.953 |
| Generator EMA | 0.984 |

These measurements support **noise-scale functional restart for this bounded
test**, not bitwise equivalence, a relaxed numerical gate or a long-trajectory
claim. The noise-distance scan covers full replicated rank0 tensors; the
uninterrupted/resumed checkpoint/RNG comparison covers all 16 ranks. Do not
conflate those scopes.

All six node-owner receipts recorded completed cleanup and zero new cgroup
OOM/OOM-kill events. Raw checkpoints remain in remote persistent storage.
The source features are still **synthetic**, reused in two distinct native RNG
cohorts. No real-data quality, 100-cycle stability, performance, native-stream
generation/export or arbitrary later Graph signature has been qualified.
See the [portable stream acceptance index](../../artifacts/framework/20261024/wan_stream_acceptance.json).

## What this does not establish

- Actual pretrained SD1.5/SDXL weights and official Self-Forcing `ode_init`
  initialization were not located in the inspected existing assets. There
  was no automatic download or substitution with LiveWan initialization.
- No production-size capacity, long-run quality or convergence validation.
- No AMP/BF16/FSDP qualification for these official single-device owners.
- CUDA automatic-plan trials still reject unsupported whole-owner snapshots.
- Modern Wan2.2/Hunyuan, FLUX/Qwen and LTX have separately scoped tiny native
  CUDA reference acceptance below, not pretrained capacity or quality
  qualification. Their CUDA planning/trial paths remain refused.
- Passing these tests does not qualify all six large native workloads or
  their optimized long-training checkpoint paths.

## Evidence

Small, portable index: [native CUDA receipt](../../artifacts/framework/20261023/native_cuda_acceptance.json).
World16 index: [W1 native functional receipt](../../artifacts/framework/20261023/wan_world16_public_acceptance.json).

Detailed run root:
`research_artifacts/DMD-System/framework_maturation_20261023/r6_a100_gpu_closure/`
under the existing local Helios project and remote persistent project.
Large model states and generated test assets remain on the remote filesystem.

- `reports/cuda-native-lifecycle-r3.xml`: actual seven-test CUDA outcome.
- `logs/cuda-native-lifecycle-r3.log`: original pytest output.
- `reports/attempt_ledger.jsonl`: all attempted batches, including failures.
- `manifests/cuda-test-policy-r3.json`: test source hashes, unchanged training source.
- `logs/gpu-processes-after-r3.csv`, GPU/cgroup receipts: cleanup evidence.
- `reports/wan-native-functional-summary.json`: all rank identities, native
  execution effects, version/census and resource receipts.
- `reports/wan-stream-admission-r1.json`: CPU-only fresh-stream refusal.

## LTX native AV reference: actual CUDA PASS

[Portable receipt](../../artifacts/framework/20261024/ltx_native_acceptance.json).
Tested source is `3d7bb039` plus the explicit `source-ltx-r4` patch
`fb5a8652f988902a261680f8ceee92d6abf09c169d6e8277c27f92734705d86d`;
its production/test changes are recorded in local integration `6618f63a`.
This is not an assertion that every later documentation/source change ran on
that exact snapshot.

- **4 PASS, zero failures/skips**, 503.32 s suite wall, one A100 on node1.
- Actual random-tiny pinned upstream AV Transformer, Gemma4/connectors,
  video/audio codecs; full-parameter and unmerged LoRA modes both exercised.
- Direct flow-DMD oracle and public Program agree exactly on losses, gradients,
  named Adam, scheduler, EMA, clocks and RNG under the explicit deterministic
  test policy. No production tolerance or DMD mathematics changed.
- Same-device new-process C1-to-C3 restart exactly matches uninterrupted C3.
  Current-G export decodes a new prepared prompt on CUDA and can relocate to
  CPU. Native raw-video/audio encoding also executes on the owned CUDA device.
- Corrupt checkpoint/EMA/optimizer/counter/input identities fail closed.
  Unbounded CUDA trial cloning and unsupported policies reject before owners
  are constructed. CPU planning remains independently qualified.
- Target CPU regression: **139 PASS, zero skips, CUDA uninitialized**. Shared
  target Torch2.10 regression: 2152 PASS, 86 explicitly optional skips.

The first GPU attempt failed in CPU fixture construction: native AUTOMATIC
attention inspected CUDA0. The repair binds the native CPU/Ampere provider in
the adapter's private source namespace; it does not hide CUDA or replace the
attention math. Its parent-red test and original failure remain archived.
Node0 was subsequently occupied by unidentified GPU processes; no job was
stopped. Node1 passed fresh environment/hash/ownership checks under an atomic
lock. After the suite, no compute processes remained there, the lock was
available, and cgroup OOM/OOM-kill increments were zero (the prior OOM-kill
counter of 3 is retained, not rewritten as zero).

This does **not** validate the pretrained 22B pipeline, generation quality,
multi-rank execution, CUDA planning, or performance. Export-only raw-input
preparation is a separate CPU follow-up, not included in these four tests.

That follow-up now passes **67/67 tests, zero skips**, in target Torch2.10 with
CUDA uninitialized (`source-ltx-r6`, patch `0eeb45be…7bcb31`). For both full and
LoRA Students, tests retire the original training assets/checkpoint and prepare
genuinely new PNG image/frame, stereo WAV and text-by-path inputs, as well as
new prompts, solely from the export. Both prepared branches generate decoded
AV through the current trained Student. Same-process RNG is preserved and
changed encoder-helper sources reject before any codec execution/publication.
This is actual tiny native CPU coverage, not pretrained quality or another GPU
result; its source and receipt hashes are separate in the portable index.

## Operational validation

Re-run these optional suites only in an owned, matching CUDA environment.
Any oracle/state/RNG mismatch is a failed gate, not a performance tradeoff.
Keep default-mode nondeterminism distinct from deterministic restart proof.
Before a pretrained deployment, admit original assets and capacity, then
validate that production configuration independently; do not reuse tiny
capacity or quality labels.


## FLUX / Qwen native image reference: actual CUDA PASS

[Portable receipt](../../artifacts/framework/20261024/modern_image_cuda_acceptance.json).
Exact source is `3d7bb039` plus the frozen `source-modern-r2` patch
`4d284f1776ee55a2b6a9303e867c67b721de9552e76f9318f7f0382ea70e71ae`.
Later CI/documentation additions are separately CPU checked, not relabeled as
that GPU snapshot.

- **8 PASS, zero failures/skips**, 850.71 s suite wall: two FLUX cases, two
  Qwen-Image cases and four LTX full/LoRA shared-ownership regression cases.
- Actual random-tiny upstream Transformers, text models and codecs on one
  A100, FP32 reference, unmerged image LoRA, unchanged four-step G1/F5 math.
- Native oracle and public Program agree exactly on losses, gradients, Adam,
  scheduler, EMA, clocks and RNG under the same explicit deterministic test
  policy. This is not a change to production defaults.
- Qwen complex RoPE retains its imaginary component. CUDA owns role/media
  tensors, Adam steps/moments and EMA; invalid state restoration fails before
  mutation. New-process C1-to-C3 equals uninterrupted C3 complete state.
- Exported current-Student weights generate a new-prompt PNG on CUDA, and
  relocate to CPU, with the original training assets absent.
- Target native CPU regression: **100 PASS, zero skips**, CUDA uninitialized.
  This suite covers the changed image/LTX owners, not native video-family GPU.
- One GPU suite launch. Fresh completion audit: no compute processes, node lock
  released, zero additional cgroup OOM/OOM-kill events. Sampled process-driver
  peak was 1,364 MiB; this is not per-workload allocated/reserved capacity.

The existing `distillctl` recipe now admits explicit `cuda:0` reference training,
resume and generation. CPU default and offline preparation remain unchanged.
CUDA auto/manual/saved-plan trials, AMP, distributed ownership and large
pretrained capacity/quality remain unqualified and fail closed where required.
Native Wan2.2/Hunyuan CUDA is not established by this image/LTX suite.

## Wan2.2 / HunyuanVideo native reference: actual CUDA PASS

[Portable receipt](../../artifacts/framework/20261024/native_video_cuda_acceptance.json).
Exact source is `85e774aeb094eb5073440dea5ec7a6f7869eb7bc` plus
`source-video-r2`, patch
`5f660bbd8b8665f2b3afb03798136f3d4459c5608ed7c7834ac38ecd3ba8a773`.
Subsequent CI/documentation changes are separately CPU checked, not relabeled as
that GPU snapshot.

- **10 PASS, zero failures/skips**, 684.04 s suite wall: three Wan2.2 TI2V-5B
  T2V cases, three HunyuanVideo-1.5 T2V cases, two native FLUX/Qwen update cases
  and two native LTX full/LoRA update regressions. This is one serial suite
  launch, not ten distributed runs or a performance measurement.
- Actual pinned tiny upstream Transformers, native text/tokenizer and VAE
  classes; FP32 reference, B1, four Euler steps, G1/F5, unmerged LoRA.
  Two devices were visible, but training used **only local CUDA1** to test
  non-default device ownership; caller current-device selection stayed intact.
- Independent native ABI/gradient/Euler/text/codec checks, followed by full
  DMD loss, gradient, Adam, scheduler, EMA and RNG comparison, passed under the
  existing explicit deterministic test policy. Production defaults are unchanged.
- New-process C1-to-C3 restart exactly matched uninterrupted C3. Exported
  current-Student weights generated new-prompt RGB frames with original assets
  and checkpoints retired. CPU relocation exactly matched a separately
  reconstructed CPU current-Student reference, not a cross-device tolerance.
- Native null conditions are moved from the validated CPU preparation rather
  than recomputed. VAE normalization and the exact inverse-input decoder are
  compared to native operations; raw pre-normalization tensors are not falsely
  assumed bitwise identical after an FP32 normalization round trip.
- Target CPU **198 PASS**, then **51 PASS** for the test-only follow-up; both
  zero skips, CUDA uninitialized. Production files were identical in the two
  frozen source snapshots. CI now requires the video device-contract tests.
- Fresh cleanup verified no remaining compute processes, node lock available,
  and zero cgroup OOM/OOM-kill increments. Peak sampled driver allocation on
  CUDA1 was 1,664 MiB, not a pretrained or per-workload capacity bound.

Both public recipes now support explicit single-CUDA reference training,
complete-cycle same-device restart and portable generation. CUDA planning,
manual strategies/trial cloning, AMP/distributed ownership, pretrained sizes,
production geometry and generation quality remain independently unqualified.
This does not complete U14/U15 or the whole mature-framework acceptance plan.


## Five common-flow families: direct manual CUDA immutable sharing

[Portable receipt](../../artifacts/framework/20261024/native_flow_cuda_immutable_sharing.json).
Actual CUDA source: `2dcc112862548895972aae6567068192a07f7bbf` plus
`source-sharing-r1`, patch
`964a8fd4f675bc7ace5e1220f252856281dc4af761907e843272de148b79ecc5`.
The integrated test-only follow-up (`source-sharing-r2`, patch
`6d9ce79fc3c5fbaec54ca85fc9a658bf7f81e2144e7dc2d843235921302d25b4`)
has identical production hashes and separate CPU proof; it was not rerun on GPU.

- **10 PASS, zero failures/skips**, 1535.18 s test-suite wall, one serial launch.
  LTX-2.5 LoRA, FLUX, Qwen-Image, Wan2.2 and HunyuanVideo each exercise an
  actual native reference/manual update comparison and fresh-process lifecycle.
- Single A100 CUDA1, FP32, actual random-tiny upstream classes and native
  codecs, four Euler steps, G1/F5. No model-weight merge or changed DMD math.
- All three cycles match reference state, losses, gradients, Adam, scheduler,
  EMA, sample identity, versions, RNG and outputs under the same deterministic
  test-only policy. Immutable parameters really alias; mutable LoRAs and
  optimizer ownership stay separate. Storage savings are measured from tracked
  tensors, not inferred from configuration.
- New-process C1-to-C3 matches uninterrupted C3; current-Student export works
  on genuinely new inputs after removing original assets/checkpoints. Video
  and image outputs additionally exercise independent CPU relocation.
- Target CPU: R1 **313 PASS**, R2 **315 PASS**, no skips or CUDA initialization.
  Reviewer #1 found an acceptance JSON field overwritten by a raw-input loop;
  two CPU tests reproduced the wrong label before the writer was fixed.
  R1 video JSONs stay unchanged. Their actual manual execution was already
  verified by policy/effects assertions and separate native sharing proofs.
- The first derived collector's explanatory text prematurely called two R2
  GPU cases qualified. That sentence is explicitly corrected in the portable
  receipt's erratum: R2 GPU is **NOT_RUN**, not a second test suite.
- End audit: no GPU processes, node lock released, zero new cgroup OOM or
  OOM-kill events. Maximum sampled driver allocation was 1634 MiB on CUDA1;
  this is neither full-model capacity nor a speedup measurement.

The complete required refactor regression also passed in the original target
Torch2.10 lane: **2152 PASS, 86 explicit SKIP, 20 subtests**, CUDA uninitialized.
The skips are not GPU qualification; prior environment collection failures
remain in the raw record.

Only the exact documented persistent manual policy is added. CUDA auto/saved
plans still need capacity-bounded trial snapshots; AMP and distribution remain
separate work. Sharing starts after constructing the roles and therefore does
not yet reduce construction peak. This does not complete U2/U9 or the mature
framework plan, and does not promote tiny fixture parity into user-workload
numerical acceptance or pretrained quality.

## FLUX / Qwen: native replicated training and restart

[Portable receipt](../../artifacts/framework/20261024/native_flow_image_replicated_acceptance.json).
Source is `8b4f90e0c98c2a90356f12e87979c62dad44b194` plus `source-r4`,
patch `4f90b9af2a4a6dd7c273a69c7ec02c60f23225d39590ce0fe1e07320038cb076`.

- **2 PASS, zero failures/skips**, actual two-A100 NCCL, 1072.908 s suite wall.
  This is functional acceptance, not a workload performance measurement.
- Actual random-tiny FLUX and Qwen-Image classes, FP32 LoRA, local B1 per rank,
  G1/F5. Three cycles compare against an independent sequential native oracle
  that averages distinct B1 gradients before clipping and Adam. Later Student
  gradients are nonzero; reference/manual state and gradients match exactly.
- The actual private native methods run without DDP forward wrappers. The
  existing post-backward reducer, immutable-sharing implementation, Program,
  data owner and checkpoint lifecycle execute the selected policy.
- Each policy completes fresh CLI partial/resumed/uninterrupted training and
  all-rank full-state comparison, current-Student export and new-prompt PNG
  generation. Missing checkpoint shards are rejected. Rank-specific samples
  and RNG are checked; single-process random streams remain unchanged.
- Target CPU R4: **204 PASS**, zero skips. Both prior actual Gloo2 native
  proofs are reused only because all production and native-helper bytes are
  identical. The earlier stale static-test failure and two test-harness errors
  remain preserved, not rewritten as successful runs. Original target refactor
  regression is **2152 PASS / 86 optional SKIP**, production bytes unchanged.
- First owner invocation failed before GPU construction solely on Git's
  abbreviated diff-header formatting. The isolated owner pins the abbreviation;
  all source hashes and numerical policies remain unchanged. Successful cleanup
  confirmed no compute processes and no new cgroup OOM events. Peak sampled
  driver allocation was 1236 MiB on each active GPU, not pretrained capacity.

The original receipt did not establish world16; its separate follow-up is
recorded below. Replication does not shard model memory; all roles still
materialize before immutable sharing. AMP, CUDA trial planning, pretrained
capacity/quality and performance remain NOT_RUN. This is a bounded U2/U3
delivery, not completion of the mature-framework plan.

### Two-node world16 follow-up — 2026-10-25 UTC

[Separate portable receipt](../../artifacts/framework/20261025/native_flow_image_world16_acceptance.json),
run `r12-native-flow-world16-r4`, source manifest
`3b678d32d318019cffcc48a3f84ced26e68934489fc35a4e1c4c0736a9fb190f`.

- Both actual tiny image families completed the native oracle on **16/16 ranks**,
  with three continuous cycles, independent B1 samples/RNG, mean gradients before
  clipping/Adam, and nonzero later Student gradients. Reference and manual
  immutable-sharing state match exactly on every rank.
- Both policies also completed public CLI partial training, fresh-process
  continuation, and uninterrupted control. All 16 checkpoint shards match in
  model/optimizer state, RNG, sample cursor and versions. Each uninterrupted run
  consumes 48 distinct records and performs G3/F15 per process.
- Current-Student exports decode a new prompt into an image through the native
  components. This final portable-output check runs on **CPU**, not CUDA.
- Both node owners completed cleanup; no remaining owned compute process or new
  cgroup OOM/OOM-kill was observed. The earlier shared-filesystem visibility and
  manifest-durability failures remain archived. The repair bounds metadata waits
  within the control-group deadline and flushes/fsyncs metadata before rename;
  it does not retry mismatching hashes or change training mathematics.

This is **SYNTHETIC_TEST_ONLY** actual-class FP32 replica lifecycle acceptance.
It is not pretrained capacity, convergence, image quality, AMP, FSDP, CUDA
planning, or a speedup claim. Those gates and the overall framework remain open.
