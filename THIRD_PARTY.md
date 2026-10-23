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
