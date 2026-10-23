# Model families and reference algorithms

A model name selects checkpoints, prediction conventions, encoders/decoders and
trainable state. An algorithm selects actual rollout/loss/update mathematics.
An execution policy chooses only admitted physical changes. The command remains
`distillctl` for every family; reference mode is not a second training CLI.

## Choose a compatible integration

| Model / workflow | Native mathematics | Data and output | Environment |
| --- | --- | --- | --- |
| SD1.5 / official DMD2 | Actual pinned `SDUniModel`/`SDGuidance`; single-step, optional original GAN | Prompt tokens; independent guidance/real/denoising streams when required; native VAE image output | `train,dmd2` |
| SDXL / official DMD2 | Actual dual CLIP, added conditions, backward simulation; four-step or explicitly initialized single-step | Both tokenizers; native VAE; no fabricated pooled or unconditional context | `train,dmd2` |
| Official Self-Forcing Wan | Actual causal random-exit block rollout, KV state, DMD loss, native G/C and EMA clocks | Native T5 prompt encoding or pinned embeddings; native full-step causal inference and VAE video | `train,self-forcing-official` |
| LTX-2.5 / generic No-GAN DMD | Actual AV LTX Transformer; explicit rectified-flow surrogate and Fake regression policy | Actual Gemma4/connectors, native video/audio VAEs, vocoder; AV or explicit latents | isolated `train,ltx25`, Python >=3.11 |
| FLUX.2 klein Base 4B / explicit flow DMD | Actual Flux2 Transformer, no guidance embeddings, normalized sigma; not official DMD2 | Native Qwen3 layers 9/18/27, patch/BN VAE, exact immutable sharing, restart and current-Student decoded images | isolated `train,modern-flow`, CPU1 FP32 only |
| Original Qwen-Image T2I / explicit flow DMD | Actual QwenImage Transformer, native text mask/img_shapes; CFG=1 only | Actual Qwen2.5-VL template/crop34, fixed masked right padding, native VAE, exact restart/current-Student images | isolated `train,modern-flow`, CPU1 FP32 only |
| Existing W1–W6 | Their distinct original contracts, not aliases of the above algorithms | See each recipe and six-workload support matrix | Existing separate qualified lanes |

Official Self-Forcing is **not W4 LiveWan**. W4 retains the cross-scale
14B Teacher / 1.3B Student implementation and its own sampling/update contract.
W5 full-sequence Grid1000 is approximate; W6 previous/current autoregressive
continuous-sigma H3 remains a different workflow.

## Explicit local assets

The optional providers use hash-locked upstream source, supplied locally, with
its original license. They never download model weights. The source-only fetch
helpers are optional, explicit operator actions, not training side effects.
DMD2 is CC-BY-NC-SA-4.0; LTX is LTX-2.x Community License; their sources are not
bundled or relicensed as framework Apache-2.0.

A model manifest binds actual architecture, every required component, all local
checkpoint bytes and relevant dependency versions. Missing or changed assets,
shape/precision disagreement, wrong prediction meaning and incomplete restore
state fail before training or output publication. A compatible shape alone is
not permission to substitute epsilon, velocity, flow or x0.

See the runnable source-specific guides:

- [SD1.5 DMD2](../../recipes/dmd2_sd15/README.md)
- [SDXL DMD2](../../recipes/dmd2_sdxl/README.md)
- [Official Self-Forcing](../../recipes/self_forcing_official/README.md)
- [LTX-2.5](../../recipes/ltx25/README.md)
- [FLUX.2 klein Base 4B](../../recipes/flux_dmd/README.md)
- [Original Qwen-Image](../../recipes/qwen_image_dmd/README.md)

## Native iterations and public cycles

The persistent lifecycle saves **both** the public complete-cycle index and the
algorithm's actual native step/update counters. DMD2's macrocycle contains one
Generator commit and its configured guidance updates. Its two LR schedulers
advance on the original native iteration clock, not an invented common clock.
Official Self-Forcing preserves independent G and C sample consumption and its
post-increment EMA initialization. LTX's explicit No-GAN recipe advances each
role's scheduler on that role's actual update. These are not interchangeable
losses or update schedules.

`train --resume` requires a complete checkpoint at a full cycle boundary and the
same topology/source/input contract. It restores model/optimizer/scheduler/EMA,
inner and outer RNG, every data-stream position and logical versions. Starting
from supplied pretrained weights is **initialization**, not continuation.
Transient KV/replay/cast caches are reconstructed, not portable training state.

## Inference is a separate state projection

`export` projects the **current Student/Generator**, plus actual owned inference
codecs/conditioners, without Teacher/Fake/Adam or training-data paths. It does
not merge trainable LoRA or silently switch to EMA. `generate --weights` uses a
new explicitly prepared input and the source-bound inference factory; neither
the old training data nor the training checkpoint is needed.

Use `--output-kind image|video|audio|audio_video|latents` only when the adapter
actually supplies that output. Native latent tensors are diagnostics with an
explicit named layout, not RGB pixels. Video+audio output is decoded frame
files and a native-rate WAV, not a promise of muxed pretrained-quality movies.

## Evidence boundary

The new providers are qualified with **actual upstream model classes at small
random CPU configurations**, real gradients, native-source oracle comparisons
where supplied, fresh-process continuation and export/generation after hiding
training artifacts. This is not pretrained quality, long-term convergence or
new CUDA/distributed/capacity/performance validation. Unsupported combinations
are rejected rather than presented as validated support. Existing native
A100/H100 recipes and historical results retain their own identity and limits.

## Separate installation lanes

In a clean, model-specific environment, install CPU Torch first for CPU
qualification; a native CUDA environment instead keeps its independently
admitted runtime. Nothing in normal model construction installs dependencies.

```sh
# Choose ONE lane in this environment, never all extras together.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,dmd2]'                 # SD1.5 or SDXL
# Alternatives in separate environments:
# python -m pip install -e '.[train,self-forcing-official]'
# python -m pip install -e '.[train,livewan-wan21-reference]'
```

LTX requires **Python >=3.11** (qualification/CI: 3.12) and a separate
Transformers5 environment. First install `torch==2.10.0`, `torchaudio==2.10.0`
and `torchvision==0.25.0` from the CPU index, then `.[train,ltx25]`.
Its exact optional dependencies include Transformers5.14.1, tokenizers0.22.2,
huggingface-hub1.7.1, scipy1.16.3 and colour-science0.4.7. An installed PEFT
from the SD environment is not a substitute for native LTX unmerged LoRA.
The core lightweight package remains Python>=3.10; LTX's transitive Python
requirements are not silently omitted on an older interpreter.

Original Helios helper `.34`, Wan/LiveWan `.35.2`, and full-original Helios
`.37.1` pins remain separate. These new extras do not upgrade an existing
experiment environment, and no combined `all` extra is advertised.

## Verification lanes

Each source-only fetch is an explicit operator action described in the recipe;
no model weights are fetched. Supply the resulting root, then run the provider
suite with `CUDA_VISIBLE_DEVICES=''`, offline Hub modes and one CPU thread:

| Separate environment | Supplied source / tests |
| --- | --- |
| `train,test,dmd2` | `DMD2_SOURCE_ROOT`; `tests/test_training_dmd2_*.py`, `tests/test_training_sd_binding.py`, `tests/test_owner_execution_capability.py` |
| `train,test,self-forcing-official` | `SELF_FORCING_SOURCE_ROOT`; `tests/test_training_self_forcing_*.py` |
| `train,test,ltx25` | `LTX_SOURCE_ROOT`; `tests/test_training_dmd_flow.py`, `tests/test_training_ltx_*.py` |
| `train,test,livewan-wan21-reference` | Optional exact Wan2.1 codec source through `DMD_WAN21_TEST_UPSTREAM`; the three public LiveWan input/training/inference test files |

The [dedicated CI workflow](../../.github/workflows/official-models-cpu.yml)
fetches the hash-locked source/license closures, supplies those roots and
requires actual named proofs with **zero skips**. A missing upstream root may
skip optional cases in a broad generic test collection; that is not provider
acceptance. Each provider lane must execute its original native tests.
The older generic `.34` helper job uses a tested explicit file selector before
starting pytest to exclude W4's three `.35.2` public full-class files; their
dedicated job actually runs them instead. Shell-expanded files are not
reliably excluded by pytest's recursive `--ignore` option.

`python scripts/check_framework_install.py` builds and installs a wheel in a
new isolated environment, reusing only the invoking numerical dependency roots
in their real import order. Its receipt binds the actual Torch version/origin,
so a Torch2.10 overlay cannot silently become a base Torch2.14 run. It does not
resolve/download dependencies or modify the invoking environment.

## Initial native image-family slice

FLUX.2 klein Base and original Qwen-Image are explicitly **source/actual tiny
CPU references**, not completed pretrained/GPU/quality integrations. Their
static compatibility metadata rejects known official DMD2/Self-Forcing,
CUDA/distributed and unsupported causal/cache model options before model
imports/weight loading, including matching explicit registered factory calls.
Arbitrary existing plugins without a declaration remain extensible.

The first admitted recipes are same-family **unmerged LoRA**, G1/F5 and four
explicit Euler steps through existing `training.dmd`, with teacher/fake CFG=1.
They do not reproduce official DMD2 or Self-Forcing. Native Qwen norm-rescaled
CFG is not silently replaced with common linear CFG. Real local safetensors,
complete shards/configs, actual text/VAE/tokenizer assets and hash-bound data
are required, with no random production fallback. Explicit BF16/F32 storage to
FP32 conversion is distinct from unqualified BF16 compute. Three different
prompt lengths, exact new-process state and actual native decoded output are
CPU-tested with `SYNTHETIC_TEST_ONLY` weights; full pretrained V3–V5 remain
**NOT_RUN**. See the [shared image recipe](../../recipes/flow_image/README.md).

## Initial HunyuanVideo-1.5 T2V slice

`hunyuan15-t2v` reuses the same common flow-DMD/lifecycle and actual immutable
sharing, not another trainer or a model-name searcher. The bounded reference is
CPU1 FP32/B1, same-family unmerged LoRA, four Euler steps and G1/F5/CFG1.
It executes actual native Hunyuan15 Transformer, Qwen2.5 **text-only** model
with native template/crop108/hidden-3, native T5/slow ByT5 glyph extraction
(including the valid no-glyph zero/allfalse sentinel), and temporal4 Hunyuan VAE.
Native raw videos/frames, prompt geometry and source-bound safe features share
one validator; silent truncation/resampling and I2V/meanflow/distilled/causal
options reject. Every role/codec has an explicit meta parameter-budget gate
before concrete training or portable reconstruction.

Fresh-process exact complete-state recovery, real shared/unshared updates and
current-Student + both native text/tokenizer/VAE portable decoded new-input
video are tested using actual tiny **SYNTHETIC_TEST_ONLY** classes. This is not
pretrained, GPU, production geometry/capacity, long-run or quality evidence.
The model assets use **Tencent Hunyuan terms, not Apache-2.0**; dependency
implementation licenses are separate. See the
[bounded recipe and source/license pin](../../recipes/hunyuan_video_dmd/README.md).
New-input preparation still explicitly uses original local assets; export-only
public preparation is not supplied or claimed here. All full pretrained/GPU/
capacity/quality gates remain **NOT_RUN**.
