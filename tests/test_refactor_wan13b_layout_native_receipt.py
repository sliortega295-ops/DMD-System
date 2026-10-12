"""Execute the exact post-timer receipt function without importing CUDA models."""
import ast
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_layout_ablation import ENV

ROOT = Path(__file__).parents[1]
WINNER = "resident-async128-tree-chunk2048"


def validator():
    path = ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_validate_fused_receipt")
    ns = {"Any": object, "RESIDENT_WINNER": WINNER}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[node.name]


def disabled_build():
    return {r: {"metadata": {"generator_nograd_lora_fused_epilogue": {
        "requested": False, "enabled": False, "fused_invocation_count": 0,
    }}} for r in ("generator", "critic")}


def test_native_layout_disabled_helpers_are_audited(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    row = validator()(disabled_build(), placement=WINNER)
    assert row["generator"]["enabled"] is False


def test_default_winner_guard_is_not_waived(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(AssertionError, match="not enabled"):
        validator()(disabled_build(), placement=WINNER)


@pytest.mark.parametrize("field,value", [("requested", True), ("enabled", True),
                                          ("fused_invocation_count", 1)])
def test_layout_rejects_executed_or_requested_helpers(monkeypatch, field, value):
    monkeypatch.setenv(ENV, "1")
    row = disabled_build()
    row["generator"]["metadata"]["generator_nograd_lora_fused_epilogue"][field] = value
    with pytest.raises(AssertionError): validator()(row, placement=WINNER)


def test_layout_rejects_wrong_placement(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError): validator()(disabled_build(), placement="world16-fsdp")
