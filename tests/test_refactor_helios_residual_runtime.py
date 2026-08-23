from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from distillschedule.runtime.helios_residual_runtime import (
    validate_chunked_residual_runtime,
)


def _load_module(tmp_path: Path, *, storage_capable: bool):
    helper_extra = ", residual_storage='cpu_inplace'" if storage_capable else ""
    autograd_extra = ", residual_storage" if storage_capable else ""
    if storage_capable:
        body = """
        first = getattr(self, '_distillschedule_chunked_residual_storage', 'cpu_inplace')
        second = getattr(self, '_distillschedule_chunked_residual_storage', 'cpu_inplace')
        return first, second
"""
    else:
        body = """
        return 'historical_cpu_inplace'
"""
    source = f"""
def _apply_chunked_gated_residual(hidden_states, residual{helper_extra}):
    return hidden_states

class _TembAwareChunkedResidual:
    @staticmethod
    def forward(ctx, hidden_states, residual{autograd_extra}):
        return hidden_states

class HeliosTransformerBlock:
    def forward(self):
{body}
"""
    path = tmp_path / ("capable.py" if storage_capable else "historical.py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_residual_policy_rejects_historical_silent_cpu_runtime(
    tmp_path: Path,
) -> None:
    module = _load_module(tmp_path, storage_capable=False)
    with pytest.raises(RuntimeError, match="silent CPU-staging fallback"):
        validate_chunked_residual_runtime(
            module,
            requested_storage="cuda_out_of_place",
        )


def test_cuda_residual_policy_receipts_storage_capable_runtime(tmp_path: Path) -> None:
    module = _load_module(tmp_path, storage_capable=True)
    receipt = validate_chunked_residual_runtime(
        module,
        requested_storage="cuda_out_of_place",
    )
    assert receipt["storage_capable"] is True
    assert receipt["helper_has_storage_parameter"] is True
    assert receipt["autograd_has_storage_parameter"] is True
    assert receipt["block_storage_getter_count"] == 2
    assert len(receipt["transformer_source_sha256"]) == 64


def test_fair_launcher_defaults_to_validated_gpu_residual_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = (root / "scripts/distillschedule/run_wan13b_fair_world16.py").read_text()
    launcher = (
        root / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
    ).read_text()
    for source in (controller, launcher):
        assert "DistillSchedule_fsdp2_gpu_residual_20260816" in source
        assert "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886" in source
    assert "correctness receipt uses a different Helios residual runtime" in launcher

    summarizer = (
        root / "scripts/distillschedule/summarize_wan13b_fair_world16.py"
    ).read_text()
    assert 'residual_runtime.get("storage_capable") is not True' in summarizer
    assert '"helios_transformer_sha256"' in summarizer
