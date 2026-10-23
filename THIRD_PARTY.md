# Third-party source and assets

The framework package metadata declares Apache-2.0; the root LICENSE contains
that license. Preserve upstream licenses and notices. This list records source
ownership, not permission to redistribute model weights or datasets.

| Component | Ownership and license location |
| --- | --- |
| Recovered Helios compatibility source | [Notices](third_party/helios_compat/NOTICE) and included upstream license; contains explicitly recorded compatibility changes |
| MiniMax H3 compatibility | Preserve the license files in the H3 component and corresponding upstream/model distribution |
| Maintained DiffSynth H3 cached-AV source derivative | [Retained Apache-2.0 license](licenses/diffsynth-h3-cached/LICENSE.Apache-2.0), [origins and changes](licenses/diffsynth-h3-cached/ORIGIN_AND_CHANGES.md); packaged symbol/source manifest in the explicit `distillschedule.reference.diffsynth_h3_cached` namespace; not a complete upstream distribution or model/data grant |
| Full original Wan2.1 / LiveWan CPU FP32 numerical closure | Retained [Wan2.1](licenses/livewan-wan21-reference/Wan2.1-LICENSE.txt) and [LiveWan](licenses/livewan-wan21-reference/LiveWan-LICENSE.txt) Apache-2.0 licenses, [origins and modifications](licenses/livewan-wan21-reference/ORIGIN_AND_CHANGES.md); private `distillschedule.reference.livewan_wan21` manifest distinguishes whole original source, static import relocations, selected-definition SDPA derivative and CPU bridges; not native BF16/FlashAttention qualification, a trainer, or a model/data license |
| LightX2V | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| LiveWan | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| Wan2.1 | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| Optional dependencies | PyTorch, Diffusers, DeepSpeed, NumPy, Pillow and others retain their original licenses |

Models and data are explicitly supplied by the operator; recipes do not upload
or download them. Source identity and original notices must accompany future
vendoring or relocation. The root license does not replace upstream licenses.

## Optional native model integrations

These sources are **not vendored in the framework wheel**. The adapters require
explicit local, file-hash-locked source closures; recipe fetch helpers are
optional source-only actions and never fetch weights.

| Integration | Pinned original source | License / limits |
| --- | --- | --- |
| Official DMD2 (SD1.5 / SDXL) | [tianweiy/DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f) | **CC-BY-NC-SA-4.0**, retained and checked in the external closure; not covered by the framework Apache-2.0 grant. Review upstream terms before use/redistribution. |
| Official Self-Forcing | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing/tree/33593df3e81fa3ec10239271dd2c100facac6de1) | Original Apache-2.0 license retained in its external source closure. Distinct from LiveWan W4. |
| LTX-2.5 / ltx-core 1.5 | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2/tree/e98f6d927d83c92ef2be6aec76c8440b19306e39/packages/ltx-core) | **LTX-2.x Community License**, retained and hash-checked externally; not Apache-2.0 or a model-weight grant. |
| Optional original Wan2.1 decoder | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315) | Separate Apache-2.0 `wan/modules/vae.py` + `LICENSE.txt` closure, independent of the frozen packaged W4 source manifest. |

The source importer uses private namespaces and explicitly recorded import or
CPU/device boundary adaptations. Actual numerical bodies execute from the
pinned source; there is no model-weight, dataset or checkpoint download in
training. Model/tokenizer/VAE/Gemma asset licenses are separate and must be
supplied/reviewed by the operator. Procedural/random original-class CPU
fixtures establish interface and mathematical/lifecycle behavior only—not
pretrained quality or permission to redistribute external assets.

## Native FLUX.2 klein Base and original Qwen-Image CPU adapters

These adapters import, rather than vendor, actual **Diffusers 0.37.1** and
**Transformers 5.14.1** classes. Their original Apache-2.0 notices and licenses
remain with those dependency distributions. PEFT0.18.1 belongs to the separate
qualified modern-flow import lane; trainable LoRA remains unmerged native
Linear arithmetic, not upstream inference-only weight fusion. Actual package
source hashes and exact local asset inventories are bound into prepared
features, checkpoints and portable exports.

| Fixed source candidate | Source-audited model repository/license | Explicit limits |
| --- | --- | --- |
| FLUX.2 klein Base 4B | [black-forest-labs/FLUX.2-klein-base-4B at a3b4f4](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d), Apache-2.0 | Only the named Base T2I version; not FLUX.1, distilled klein or other variants |
| Original Qwen-Image | [Qwen/Qwen-Image at 75e0b4](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6), Apache-2.0 | Original T2I only; not Edit or later generations |

Qwen3 and Qwen2.5-VL text models/tokenizers, VAE/model assets, and datasets
remain operator-supplied components with their own source and license
obligations. A repository label plus local hashes does not authenticate
pretrained provenance or grant redistribution rights. No model weights or
third-party source are downloaded/bundled by these recipes. Random actual-class
fixtures are labeled **SYNTHETIC_TEST_ONLY** and establish only CPU interface,
math/state/codec behavior, never pretrained/GPU/quality or performance proof.

## HunyuanVideo-1.5 T2V native CPU adapter

The fixed source-only model candidate is
[hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v at 286be7c](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/tree/286be7ce72277246578a3e3cc2487e95ddae5bcf).
**Model assets are subject to Tencent Hunyuan Community terms (repository tag
`other`), not Apache-2.0.** Operators must review the original model/text/
tokenizer/data terms; local hashes do not prove pretrained authenticity or
grant redistribution rights. No full model config, weights or third-party
implementation is vendored or downloaded by this adapter. The old upstream
example ID without `Diffusers` returned401 during the recorded source audit;
that failure is retained rather than retried or bypassed.

Actual Diffusers0.37.1 and Transformers5.14.1 code is imported from separately
licensed Apache-2.0 dependency distributions, whose notices remain with those
packages. It is not a replacement license for Tencent assets. The tiny native
Qwen2.5 text-only/T5/ByT5/Hunyuan Transformer/temporal4 VAE fixtures use random
weights and are explicitly **SYNTHETIC_TEST_ONLY**; native semantics, lifecycle,
sharing and portable RGB decode are not pretrained/GPU/capacity/quality or
performance validation. See [source pin](recipes/hunyuan_video_dmd/source.json)
and [bounded recipe](recipes/hunyuan_video_dmd/README.md).
