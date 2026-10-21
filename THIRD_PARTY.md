# Third-party source and assets

The framework package metadata declares Apache-2.0; the root LICENSE contains
that license. Preserve upstream licenses and notices. This list records source
ownership, not permission to redistribute model weights or datasets.

| Component | Ownership and license location |
| --- | --- |
| Recovered Helios compatibility source | [Notices](third_party/helios_compat/NOTICE) and included upstream license; contains explicitly recorded compatibility changes |
| MiniMax H3 compatibility | Preserve the license files in the H3 component and corresponding upstream/model distribution |
| LightX2V | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| LiveWan | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| Wan2.1 | Git submodule in [`.gitmodules`](.gitmodules); upstream license and pinned revision |
| Optional dependencies | PyTorch, Diffusers, DeepSpeed, NumPy, Pillow and others retain their original licenses |

Models and data are explicitly supplied by the operator; recipes do not upload
or download them. Source identity and original notices must accompany future
vendoring or relocation. The root license does not replace upstream licenses.
