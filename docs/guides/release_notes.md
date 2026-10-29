# Bounded framework release notes

This is release hardening of the existing **0.3.0** framework, not a declaration
that the mature-framework plan or its GPU/pretrained milestones are complete.
No new throughput, speedup, convergence or quality result is claimed.

## Reachable capabilities and evidence tiers

* All six original direct manual research presets retain their exact two-cycle
  contracts and source/numerical statuses. W3 merged-LoRA/N70 stays OFF; W5 Grid1000's
  optimized numerical **FAIL** remains unresolved, never a stable default.
  W1's unchanged fixed-trace preset now has actual public-entry world16
  functional evidence: 16 ranks, two cycles, version/census/runtime-effect
  checks complete. Its synthetic feature trace does not establish quality or
  fresh-data continuation, and no new speedup or numerical-parity claim is made.
* W1's `wan_ar.incumbent21a_stream.v2` uses the complete original Graph/cast/
  fusion owner with 2..100 distinct fresh cohorts and complete-cycle
  checkpoint/restart. First-process-cycle capture coverage remains explicit;
  later novel signatures fail. Existing world16 two-cohort continuation and
  new-process restoration are retained, including its source-native amplified
  zero-history policy; that earlier integration gap is resolved for Resident
  streams, not silently admitted by other backends.
* W2's `helios_nogan.safe_stream_ed8.v1` binds the original node-HSDP SafeBundle
  and 2..100 fresh cohorts. W3's full GAN stream now preserves native
  G1/C5/EMA1, all GAN heads/Adam state, physical/curriculum clocks and N70 OFF.
  W4's full C51 stream includes actual two-wave drain and two-pool checkpoint
  lifecycle. W6's native AR stream saves/restores the actual ZeRO-3 engine and
  both LoRAs at complete G1/F1/EMA0 boundaries. These new continuity paths have
  source/CPU protocol proof; their native GPU/100-cycle acceptance is pending.
* W1/W2/W3 now export their actual restored current unmerged G with a real
  Wan2.1 VAE and emit independent new-prompt decoded video configs. Tests use
  actual tiny original G/sampler/VAE in a fresh process without old training or
  codec directories. They preserve live training state and do not merge LoRA.
  W4's [full C51 output](livewan_native_output.md) restores both pools and
  exports the current full Student for independent original-streamer video.
  W5's [original Grid output](h3_grid_inference.md) includes the four frozen
  rollout-table entries and original packed AV sampler. W6's
  [native AR output](h3_ar_native_inference.md) preserves original mixed
  tensor storage and the eight-step previous/current AV sampler. Original
  codecs supply video/audio decoding; [raw H3 AV preparation](h3_audio_video_encode.md)
  connects actual previous/current frame directories and stereo WAV to the
  existing cached-input trainer. These output/encoding chains have target CPU
  and fresh-process proofs; their pretrained native GPU acceptance is separate.
* Official DMD2 SD1.5/SDXL and official Self-Forcing retain actual pinned native
  owners and algorithms, now with explicit single-device CPU/CUDA FP32 source
  placement and safe host-state restore/export ownership. CPU original-model
  tests are mandatory. The actual random-tiny CUDA suites in `tests/gpu`
  passed **7/7 on A100 with Torch 2.10**, including native-oracle equality,
  fresh-process restart and current-G generation/export. Exact assertions use
  an explicit deterministic test policy in every process, not changed training
  defaults. This does not qualify pretrained weights, full-GAN training or
  production capacity. See the [GPU acceptance scope](../status/FRAMEWORK_A100_ACCEPTANCE_20261024.md).
  CPU-only Torch still reports these suites NOT_RUN; a skip is never GPU proof.
  The unchanged reference profiles still refuse distributed/AMP execution and
  CUDA clone-based auto/plan trials. A separate explicit DMD2 FP32 role-DDP
  profile now has actual SD1.5/SDXL CPU2/Gloo persistent-update and new-process
  full-state recovery evidence; its CUDA/NCCL path is not yet GPU-qualified.
  Independently bound Generator/Fake checkpoint policies and official
  Self-Forcing post-commit consumed-KV release have scoped native CPU evidence.
  These are opt-ins, not the full separately measured mixed-precision/fused
  native portfolios. See [native-owner policies](native_owner_backends.md).
* The separate full-native optimized SD1.5 DMD2 and official Self-Forcing public
  workflows completed 100 cycles on 16 A100s, including durable cycle-50
  checkpoints and fresh-process continuation. DMD2 additionally completed
  same-source native world16 current-Student export and independent new-prompt
  512 × 512 image generation, with tensor/PNG integrity and owner cleanup accepted.
  Official Self-Forcing full-native decoded-output acceptance remains pending.
  DMD2 random initialization/synthetic records and Self-Forcing Wan base
  initialization (not ODE) establish runtime stability, not pretrained quality.
  [Exact source and receipt scope](../status/official_optimized_framework.md#public-native-lifecycle-validation-2026-10-29)
  remains separate from historical timing and tiny reference suites.
* Modern FLUX.2 klein Base T2I, original Qwen-Image T2I, Wan2.2 TI2V-5B's T2V
  subpath and HunyuanVideo-1.5 T2V use actual tiny upstream native Transformers,
  text/tokenizers and VAEs. CPU1/FP32/B1 unmerged-LoRA G1/F5, raw media, complete
  fresh-process recovery, current-Student export and native RGB decode are
  **SYNTHETIC_TEST_ONLY**. Common export-only preparation encodes truly new
  prompts/raw media after all training assets and checkpoints are deleted.
* All four modern-flow families additionally execute real measured common
  auto/saved-plan lifecycles. Clone trials qualify only tiny synthetic CPU1/FP32
  state; `LOCAL_PRETRAINED_UNVERIFIED` refuses before cloning. The tracked
  selected-storage limit is **not** host RSS or paired-trial capacity admission.
* LTX retains its bounded AV recipe, native codecs and common flow-DMD lifecycle.
  Its actual tiny-model A100 single-device FP32 reference has 4/4 native-oracle,
  full/LoRA update, fresh-process recovery and output checks. Full monolithic
  22B capacity, distributed ownership and pretrained quality remain outside
  that acceptance.

The [support matrix](support_matrix.md) distinguishes each current executable
path from its measured GPU evidence. Reuse completed receipts; do not rerun
old model/lifecycle suites just to move documentation to a newer commit.
Missing long acceptance uses one seed and 100 complete cycles with one
midpoint save/new-process continuation. Pretrained quality, native CUDA
capacity and hosted CI are not inferred from source or tiny CPU tests.

## Packaging, migration and refusal behavior

The wheel/sdist now retain exact curated recipe/config/template/helper bytes
and original license/notice paths. `distillschedule.recipes.directory()` locates
the installed resources without Torch, cwd lookup or editable-source fallback.
Model payloads, prepared training data, checkpoints, traces, credentials and
dependency environments are not redistributed. Helpers do not fetch weights;
external source fetchers are explicit source-only operator actions.

Use separate TF4 official/Helios, TF5 modern-flow and LTX environments; never
repair a conflict by upgrading a validated environment or changing source
hashes. Modern qualification uses Python3.12/Torch2.10CPU/Diffusers0.37.1/
Transformers5.14.1/PEFT0.18.1 and mandatory ffmpeg+ffprobe for container-video
tests. Frame directories use Pillow. See [installed tutorial](installation.md).

Existing configs/checkpoints are not silently migrated. Pretrained asset
initialization is not `--resume`. A resumable checkpoint binds effective model,
algorithm, topology, input inventory/cursors, source/runtime, policy, optimizer,
EMA/schedulers and RNG. Missing/wrong source, changed cache/precision/geometry,
unowned tensors or unsupported snapshot/clone layouts refuse before update;
portable inference exports do not contain resumable training state.
Distributed checkpoint `evaluate`/`export` now reuse resume's collective file
admission before loading local shards. Six real Gloo/CPU FSDP2 output tests
passed at source `12ead25e`; this is CPU verification, not another GPU run or
permission to reuse an old checkpoint with a different training identity.

Reviewed raw native serialization defaults are admitted by the specific adapter
without rewriting the original source/config bytes or weakening inventory
hashes. Wan's historical `clip_output:false` is only an exact validated no-op;
`true`, unknown keys, invalid native token IDs and variant/geometry drift still
reject. Qwen optional native defaults and Hunyuan primary model `pad_token_id`
remain distinct from the required valid native tokenizer padding contract.
Every role/text/VAE has explicit parameter-budget admission before concrete
construction; an asset label is not pretrained authenticity or memory proof.
The modern native loader now copies and releases one safetensors shard at a
time after complete header admission. It preserves real tied parameters and
constructor-owned buffers without retaining all previous shard payloads.
Single-file weights still require their full payload plus model and conversion
space; this is not total host-memory admission.

The dedicated H3 codec CI lane requires real original encoder/decoder,
packing, raw-input and installed-package proofs without private model assets.
The modern-flow lane now also requires both FP32/BF16 shard-release proofs
and exact constructor RNG behavior; missing or skipped required proofs fail.

Framework Apache-2.0 does not replace external terms: official DMD2 source is
CC-BY-NC-SA-4.0, LTX source uses LTX-2.x Community terms, Hunyuan model assets
use Tencent Hunyuan terms; all model/text/tokenizer/VAE/data terms are the
operator's responsibility. Local hashes alone authenticate none of those rights.
See [retained third-party notices](../../THIRD_PARTY.md) and individual recipes.

Hosted CI is a required release gate at the final committed source. A local
installed-wheel receipt does not claim that hosted CI ran or passed, or that a
GPU release tag has been qualified.
