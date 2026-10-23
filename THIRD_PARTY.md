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
