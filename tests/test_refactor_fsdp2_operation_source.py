from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from distillschedule.profiling.fsdp2_operation_source import (
    DEFAULT_TARGET_BYTES,
    OperationSourceCapture,
    OperationSourceTrace,
    checkpoint_context_fn,
)


ROOT = Path(__file__).resolve().parents[1]


def test_operation_source_is_bounded_and_does_not_install_saved_hooks(tmp_path) -> None:
    source = (
        ROOT / "src/distillschedule/profiling/fsdp2_operation_source.py"
    ).read_text()
    assert "torch.autograd.graph.saved_tensors_hooks(" not in source
    trace = OperationSourceTrace(rank=0, max_events=2)
    trace.append({"event": "one"})
    trace.append({"event": "two"})
    trace.append({"event": "three"})
    output = tmp_path / "trace.json"
    trace.dump(output)
    value = json.loads(output.read_text())
    assert value["matched_events"] == 3
    assert value["recorded_events"] == 2
    assert value["dropped_events"] == 1


def test_operation_source_defaults_to_known_critic_copy_shape() -> None:
    capture = OperationSourceCapture(enabled=False, rank=0)
    assert capture.trace.target_bytes == DEFAULT_TARGET_BYTES == 113_049_600
    assert capture.trace.events == []
    with pytest.raises(ValueError):
        OperationSourceCapture(enabled=False, rank=0, max_events=0)


def test_checkpoint_context_pair_is_marker_only() -> None:
    original, recompute = checkpoint_context_fn()
    assert hasattr(original, "__enter__")
    assert hasattr(recompute, "__enter__")


def test_p0_worker_and_launcher_keep_canary_fail_closed() -> None:
    worker = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    launcher = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    controller = (
        ROOT / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
    ).read_text()
    assert 'choices=(10,)' in worker
    assert '"--operation-source requires --critic-only-step"' in worker
    assert "rank == args.operation_source_rank" in worker
    assert "saved_tensors_hooks_installed_by_tracer\": False" in worker
    assert "DISTILLSCHEDULE_CRITIC_ONLY_STEP" in launcher
    assert "DISTILLSCHEDULE_OPERATION_SOURCE" in launcher
    assert "DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE" in launcher
    assert "DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE" in launcher
    assert "helios_transformer_hash_mismatch" in launcher
    assert 'choices=("cpu_inplace", "cuda_out_of_place")' in worker
    assert 'else args.generator_residual_storage' in worker
    assert 'default="cuda_out_of_place"' in worker
    assert 'DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE:-cuda_out_of_place' in launcher
    assert 'DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE:-cuda_out_of_place' in launcher
    assert '"--generator-residual-storage"' in controller
    assert '"--critic-residual-storage"' in controller
    assert 'default="cuda_out_of_place"' in controller
    assert 'args.variant != "no-gan" or args.mode != "profile"' in controller
    assert 'export LD_LIBRARY_PATH="${PYENV}/lib' in launcher
    assert 'pyenv = Path("/tmp/distillschedule-helios-official-torch210")' in controller
    assert '"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}; "' in controller
