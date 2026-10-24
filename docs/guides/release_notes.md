# Bounded framework release notes

This is release hardening of the existing **0.3.0** framework, not a declaration
that the mature-framework plan or its GPU/pretrained milestones are complete.
No new throughput, speedup, convergence or quality result is claimed.

## Reachable capabilities and evidence tiers

* All six original direct manual research presets retain their exact two-cycle
  contracts and source/numerical statuses. W3 merged-LoRA/N70 stays OFF; W5's
  optimized numerical **FAIL** remains unresolved, never a stable default.
  W1's unchanged fixed-trace preset now has actual public-entry world16
  functional evidence: 16 ranks, two cycles, version/census/runtime-effect
  checks complete. Its synthetic feature trace does not establish quality or
  fresh-data continuation, and no new speedup or numerical-parity claim is made.
* W1's separate `wan_ar.incumbent21a_stream.v2` retains the complete coupled
  incumbent bundle on the original fresh-cohort session and checkpoint owner.
  Actual-dispatch CPU scalar state/restart tests use fake CUDA capture drivers;
  world16, Graph/Triton, HBM and pretrained numerical validation are NOT_RUN.
  Only bounded 2..64-cohort streams with first-process-cycle capture coverage
  are admitted; later novel signatures fail explicitly. See the
  [stream recipe](../../recipes/wan_ar/INCUMBENT_STREAM.md) for exact limits.
  A real target-environment CPU capture now confirms that the native all-zero
  history amplified branch blocks an otherwise valid stream; it was not
  bypassed. This is an implementation gap, not merely a missing GPU test.
* W2's separate `helios_nogan.safe_stream_ed8.v1` binds the actual SafeBundle
  and original fresh-cohort lifecycle (2..64); original-helper CPU and Gloo
  recovery/failure tests are not FA2/Triton/node-HSDP CUDA validation. Its original
  preset and reference are unchanged. W3/W4/W6 full optimized native long
  lifecycles are **not complete**; their separate references are not substitutes.
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
  Distributed/AMP and CUDA clone-based auto/plan trials remain refused.
* Modern FLUX.2 klein Base T2I, original Qwen-Image T2I, Wan2.2 TI2V-5B's T2V
  subpath and HunyuanVideo-1.5 T2V use actual tiny upstream native Transformers,
  text/tokenizers and VAEs. CPU1/FP32/B1 unmerged-LoRA G1/F5, raw media, complete
  fresh-process recovery, current-Student export and native RGB decode are
  **SYNTHETIC_TEST_ONLY**. Common export-only preparation encodes truly new
  prompts/raw media after all training assets and checkpoints are deleted.
* Three admitted modern families additionally execute real measured common
  auto/saved-plan lifecycles. Clone trials qualify only tiny synthetic CPU1/FP32
  state; `LOCAL_PRETRAINED_UNVERIFIED` refuses before cloning. The tracked
  selected-storage limit is **not** host RSS or paired-trial capacity admission.
* LTX retains its separate narrow AV/source/CPU recipe and native codecs, not a
  full monolithic 22B, arbitrary architecture or GPU backend qualification.

All full pretrained CPU/GPU execution, real-GPU capacity and numerical gates,
production geometry, sustained lifecycle, convergence and quality validation
remain **NOT_RUN** unless a specific independent receipt says otherwise.

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

Reviewed raw native serialization defaults are admitted by the specific adapter
without rewriting the original source/config bytes or weakening inventory
hashes. Wan's historical `clip_output:false` is only an exact validated no-op;
`true`, unknown keys, invalid native token IDs and variant/geometry drift still
reject. Qwen optional native defaults and Hunyuan primary model `pad_token_id`
remain distinct from the required valid native tokenizer padding contract.
Every role/text/VAE has explicit parameter-budget admission before concrete
construction; an asset label is not pretrained authenticity or memory proof.

Framework Apache-2.0 does not replace external terms: official DMD2 source is
CC-BY-NC-SA-4.0, LTX source uses LTX-2.x Community terms, Hunyuan model assets
use Tencent Hunyuan terms; all model/text/tokenizer/VAE/data terms are the
operator's responsibility. Local hashes alone authenticate none of those rights.
See [retained third-party notices](../../THIRD_PARTY.md) and individual recipes.

Hosted CI is a required release gate at the final committed source. A local
installed-wheel receipt does not claim that hosted CI ran or passed, or that a
GPU release tag has been qualified.
