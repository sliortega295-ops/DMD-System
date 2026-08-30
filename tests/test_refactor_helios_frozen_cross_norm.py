from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/distillschedule/runtime/helios_frozen_cross_norm_kernel.py"
CANARY = ROOT / "scripts/distillschedule/canary_helios_frozen_cross_norm.py"


def test_c27_standalone_sources_parse_and_pin_actual_helios_owner() -> None:
    kernel_source = KERNEL.read_text()
    canary_source = CANARY.read_text()
    ast.parse(kernel_source)
    ast.parse(canary_source)
    assert "_ChunkedFrozenNorm.apply" in canary_source
    assert (
        "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
        in canary_source
    )
    assert "tensor_parity_metrics" in canary_source
    assert "1.15" in canary_source


def test_c27_is_not_integrated_before_canary_passes() -> None:
    probe = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    policy = (ROOT / "src/distillschedule/runtime/fsdp2_p0.py").read_text()
    launcher = (
        ROOT / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
    ).read_text()
    for source in (probe, policy, launcher):
        assert "frozen_cross_norm" not in source

