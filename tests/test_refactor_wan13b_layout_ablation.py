"""CPU checks for a default-off measured layout/scheduling arm."""
import ast,os
from pathlib import Path
import pytest
from distillschedule.runtime.wan13b_layout_ablation import check,requested,ENV
ROOT=Path(__file__).parents[1]
def test_default_off(monkeypatch):
    monkeypatch.delenv(ENV,raising=False);assert not requested()
def test_only_registered_layout(monkeypatch):
    monkeypatch.setenv(ENV,"1");assert requested()
    check("resident-async128-tree-chunk2048",enabled=True,flags={})
    with pytest.raises(ValueError):check("world16-fsdp",enabled=True,flags={})
def test_reject_kernel_or_graph_factors():
    for flag in ("epilogue","norm","graph","shadow","input_gemm","cast_cache"):
        with pytest.raises(ValueError):check("resident-async128-tree-chunk2048",enabled=True,flags={flag:True})
def test_original_default_guard_retained():
    source=(ROOT/"scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py").read_text()
    assert "((not fused and not layout_only) or cast_cache)" in source
    assert "layout_only_ablation" in source
    for name in ("probe_wan13b_trajectory_world16.py","summarize_wan13b_trajectory_world16.py","summarize_wan13b_two_cycle_timing_world16.py"):
        ast.parse((ROOT/"scripts/distillschedule"/name).read_text())
        assert "LAYOUT_ONLY" in (ROOT/"scripts/distillschedule"/name).read_text()
