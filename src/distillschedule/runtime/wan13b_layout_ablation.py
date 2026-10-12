"""Default-off measured layout/scheduling ablation; no optimized tensor kernels."""
from __future__ import annotations
import os
ENV="DISTILLSCHEDULE_WAN13B_LAYOUT_ONLY_ABLATION"
ARM="RESIDENT_ASYNC128_TREE_CHUNK2048_LAYOUT_ONLY"
def requested():
    value=os.environ.get(ENV,"0")
    if value not in {"0","1"}:raise ValueError("layout-only ablation must be0/1")
    return value=="1"
def check(placement, *, enabled, flags):
    if not enabled:return
    if placement!="resident-async128-tree-chunk2048":raise ValueError("layout-only ablation requires registered Resident placement")
    if any(flags.values()):raise ValueError("layout-only ablation forbids optimized kernels, graphs, merged weights and cast caches")
