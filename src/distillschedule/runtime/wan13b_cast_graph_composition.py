"""Admission for exact adjacent-input cast reuse inside rollout graph capture.

The original Python cast-cache arm lost time in bookkeeping. Reusing the
exact same Tensor.to result while capturing a CUDA graph removes duplicate
device copies without executing that bookkeeping on graph replay. The opt-in
is the existing cast-cache flag WITH the existing epilogue and graph flags.
Default and the previously rejected standalone trajectory arm are unchanged.
"""
from __future__ import annotations


def check(placement: str, *, cast_cache: bool, fused: bool, graph: bool,
          input_gemm: bool = False, shadow: bool = False) -> bool:
    if not cast_cache:
        return False
    if (placement != "resident-async128-tree-chunk2048" or not fused or not graph
            or input_gemm or shadow):
        raise ValueError("Trajectory cast reuse requires Resident epilogue + rollout graph, no input GEMM or merged shadow")
    return True


def validate_build(build: dict, *, enabled: bool) -> dict:
    g = build.get("generator", {}).get("metadata", {}).get("generator_nograd_lora_cast_cache", {})
    c = build.get("critic", {}).get("metadata", {}).get("generator_nograd_lora_cast_cache", {})
    if c.get("enabled"):
        raise AssertionError("Generator cast reuse entered Critic")
    if not enabled:
        if g.get("enabled"):
            raise AssertionError("Unrequested cast reuse executed")
        return dict(enabled=False)
    if (not g.get("enabled") or g.get("cache_capacity") != 1
            or g.get("cache_hit_count", 0) <= 0 or g.get("cache_miss_count", 0) <= 0
            or g.get("avoided_destination_write_bytes", 0) <= 0):
        raise AssertionError("Exact cast reuse has no executed cache hits")
    return dict(enabled=True, mechanism="EXACT_INPUT_CAST_REUSE_IN_GRAPH_CAPTURE",
                arithmetic_changed=False, Python_bookkeeping_on_graph_replay=False,
                generator=g, critic=c)
