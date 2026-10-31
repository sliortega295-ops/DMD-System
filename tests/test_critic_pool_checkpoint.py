import hashlib
import json
from pathlib import Path

import pytest
import torch

from runtime.critic_pool_checkpoint import (
    CandidateBTransitionPackage,
    LogicalCheckpointPackage,
    _exact_equal_on_source_device,
    load_candidate_b_generator_module_into_trainables,
    restore_zero1_optimizer_partition_from_logical,
    restore_zero3_optimizer_partition_from_logical,
)


def _write(root: Path, *, ema_step: int = 10) -> dict:
    root.mkdir()
    manifest = {
        "schema_version": "distillschedule.logical_checkpoint.v3",
        "global_step": 10,
        "input_trace_hash": "t" * 64,
        "components": {
            "generator": {
                "parameter_version": 2,
                "scheduler_state": {"last_epoch": 2},
            },
            "critic": {
                "parameter_version": 10,
                "scheduler_state": {"last_epoch": 10},
            },
        },
        "ema": {"optimization_step": ema_step},
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_full5_logical_package_requires_g2_c10_ema10(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "logical")
    package = LogicalCheckpointPackage.load(
        tmp_path / "logical", expected_trace_hash="t" * 64
    )
    assert package.manifest_hash == manifest["manifest_hash"]


def test_full5_logical_package_rejects_ema_skip_state(tmp_path: Path) -> None:
    _write(tmp_path / "logical", ema_step=11)
    with pytest.raises(ValueError, match="EMA10"):
        LogicalCheckpointPackage.load(tmp_path / "logical")


def test_exact_restore_verifier_accepts_equal_cpu_tensors() -> None:
    source = torch.arange(8, dtype=torch.float32)
    assert _exact_equal_on_source_device(source.clone(), source)
    changed = source.clone()
    changed[-1] += 1
    assert not _exact_equal_on_source_device(changed, source)


def test_candidate_b_generator_module_restore_is_order_exact(tmp_path: Path) -> None:
    root = tmp_path / "transition"
    root.mkdir()
    (root / "model_ema").mkdir()
    (root / "model_ema" / "sparse_ema_manifest.json").write_text("{}")
    (root / "rng").mkdir()
    for rank in range(16):
        path = root / "rng" / f"rank_{rank:02d}.pt"
        path.write_bytes(b"x")
    values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.bfloat16)
    payload = root / "generator.bf16"
    values.view(torch.uint16).numpy().tofile(payload)
    manifest = {
        "schema_version": "distillschedule.routeb_candidate_b_transition_logical.v1",
        "status": "PASS",
        "source_transition_manifest_sha256": "a" * 64,
        "critic_logical_manifest_hash": "b" * 64,
        "versions": {"generator": 3, "critic": 10, "ema_optimization_step": 10},
        "generator": {
            "logical_numel": 5,
            "parameter_names": ["a", "b"],
            "records": {
                "a": {"offset_elements": 0, "numel": 2, "shape": [2]},
                "b": {"offset_elements": 2, "numel": 3, "shape": [3]},
            },
            "payload": {
                "path": "generator.bf16",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            },
        },
        "rng": {
            str(rank): {"path": f"rng/rank_{rank:02d}.pt", "bytes": 1}
            for rank in range(16)
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    package = CandidateBTransitionPackage.load(root)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(2))
            self.b = torch.nn.Parameter(torch.zeros(3))

    model = Model()
    restored = load_candidate_b_generator_module_into_trainables(model, package)
    assert restored["logical_numel"] == 5
    assert model.a.dtype == model.b.dtype == torch.bfloat16
    assert torch.equal(model.a, values[:2])
    assert torch.equal(model.b, values[2:])


def test_zero1_logical_restore_supports_padded_dp12(tmp_path: Path) -> None:
    root = tmp_path / "logical"
    root.mkdir()
    values = {
        "master": torch.arange(10, dtype=torch.float32),
        "first_moment": torch.arange(10, dtype=torch.float32) + 10,
        "second_moment": torch.arange(10, dtype=torch.float32) + 20,
    }
    payloads = {}
    for name, value in values.items():
        path = root / f"{name}.f32"
        value.numpy().tofile(path)
        payloads[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "dtype": "torch.float32",
        }
    manifest = {
        "schema_version": "distillschedule.logical_checkpoint.v3",
        "global_step": 10,
        "input_trace_hash": "t" * 64,
        "components": {
            "generator": {"parameter_version": 2, "scheduler_state": {"last_epoch": 2}},
            "critic": {
                "parameter_version": 10,
                "optimizer_step": 10,
                "scheduler_state": {"last_epoch": 10},
                "model_state": {"logical_samples": 10},
                "logical_numel": 10,
                "parameter_names": ["weight"],
                "payloads": payloads,
            },
        },
        "ema": {"optimization_step": 10},
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    package = LogicalCheckpointPackage.load(root)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10))

    model = Model()
    master = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
    optimizer = torch.optim.AdamW([master], lr=4e-7)

    class Zero:
        partition_gradients = False
        single_partition_of_fp32_groups = [master]

        def __init__(self) -> None:
            self.optimizer = optimizer

        def _link_all_hp_params(self) -> None:
            self.linked = True

    class Scheduler:
        def __init__(self) -> None:
            self.value = None

        def load_state_dict(self, value) -> None:
            self.value = value

    class Engine:
        def __init__(self) -> None:
            self.module = model
            self.optimizer = Zero()
            self.lr_scheduler = Scheduler()
            self.global_steps = 0
            self.global_samples = 0

        @staticmethod
        def zero_optimization_stage() -> int:
            return 1

    engine = Engine()
    result = restore_zero1_optimizer_partition_from_logical(
        engine,
        package,
        component="critic",
        partition_rank=3,
        dp_world_size=4,
    )
    assert result["valid_numel"] == 1
    assert result["padding_numel"] == 2
    assert torch.equal(master, torch.tensor([9.0, 0.0, 0.0]))
    state = optimizer.state[master]
    assert torch.equal(state["exp_avg"], torch.tensor([19.0, 0.0, 0.0]))
    assert torch.equal(state["exp_avg_sq"], torch.tensor([29.0, 0.0, 0.0]))
    assert int(state["step"].item()) == 10
    assert engine.global_steps == 10 and engine.global_samples == 40
    assert engine.lr_scheduler.value == {"last_epoch": 10}


def test_zero3_logical_restore_uses_per_parameter_partitions(tmp_path: Path) -> None:
    root = tmp_path / "logical"
    root.mkdir()
    values = {
        "master": torch.arange(8, dtype=torch.float32),
        "first_moment": torch.arange(8, dtype=torch.float32) + 10,
        "second_moment": torch.arange(8, dtype=torch.float32) + 20,
    }
    payloads = {}
    for name, value in values.items():
        path = root / f"{name}.f32"
        value.numpy().tofile(path)
        payloads[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "dtype": "torch.float32",
        }
    manifest = {
        "schema_version": "distillschedule.logical_checkpoint.v3",
        "global_step": 10,
        "input_trace_hash": "t" * 64,
        "components": {
            "generator": {
                "parameter_version": 2,
                "optimizer_step": 2,
                "scheduler_state": {"last_epoch": 2},
                "model_state": {"logical_samples": 16},
                "logical_numel": 8,
                "parameter_names": ["a", "b"],
                "records": {
                    "a": {"offset_elements": 0, "numel": 5, "shape": [5]},
                    "b": {"offset_elements": 5, "numel": 3, "shape": [3]},
                },
                "payloads": payloads,
            },
            "critic": {
                "parameter_version": 10,
                "scheduler_state": {"last_epoch": 10},
            },
        },
        "ema": {"optimization_step": 10},
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    package = LogicalCheckpointPackage.load(root)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.empty(0))
            self.b = torch.nn.Parameter(torch.empty(0))
            self.a.ds_numel, self.a.ds_tensor = 5, torch.zeros(3)
            self.b.ds_numel, self.b.ds_tensor = 3, torch.zeros(2)

    model = Model()
    master = torch.nn.Parameter(torch.zeros(5), requires_grad=True)
    optimizer = torch.optim.AdamW([master], lr=2e-6)

    class Zero:
        fp32_partitioned_groups_flat = [master]

        def __init__(self) -> None:
            self.optimizer = optimizer

        @property
        def state(self):
            # Mirror DeepSpeedZeroOptimizer_Stage3: the outer wrapper exposes
            # the nested optimizer's mapping, which defeated shallow type
            # inference in the first production repair.
            return self.optimizer.state

        def set_local_hp_param(self, value, parameter, state_key=None) -> None:
            offset = 0 if parameter is model.a else 3
            destination = (
                master
                if state_key is None
                else optimizer.state[master][state_key]
            )
            destination.narrow(0, offset, value.numel()).copy_(value)

    class Scheduler:
        def __init__(self) -> None:
            self.value = None

        def load_state_dict(self, value) -> None:
            self.value = value

    class Engine:
        def __init__(self) -> None:
            self.module = model
            self.optimizer = Zero()
            self.lr_scheduler = Scheduler()
            self.global_steps = 0
            self.global_samples = 0

        @staticmethod
        def zero_optimization_stage() -> int:
            return 3

    engine = Engine()
    result = restore_zero3_optimizer_partition_from_logical(
        engine,
        package,
        component="generator",
        partition_rank=1,
        dp_world_size=2,
    )
    assert result["partitioning"] == "per_parameter_zero3"
    assert result["local_valid_numel"] == 3
    assert result["local_padding_numel"] == 2
    assert result["optimizer_wrapper_chain"] == [
        f"{Zero.__module__}.{Zero.__name__}",
        "torch.optim.adamw.AdamW",
    ]
    assert result["optimizer_step_representation"] == "torch_singleton_tensor"
    assert torch.equal(master, torch.tensor([3.0, 4.0, 0.0, 7.0, 0.0]))
    state = optimizer.state[master]
    assert isinstance(state["step"], torch.Tensor)
    assert state["step"].numel() == 1 and int(state["step"].item()) == 2
    assert torch.equal(state["exp_avg"], torch.tensor([13.0, 14.0, 0.0, 17.0, 0.0]))
    assert torch.equal(state["exp_avg_sq"], torch.tensor([23.0, 24.0, 0.0, 27.0, 0.0]))
    assert torch.equal(model.a.ds_tensor, torch.tensor([3.0, 4.0, 0.0]))
    assert torch.equal(model.b.ds_tensor, torch.tensor([7.0, 0.0]))
    assert engine.global_steps == 2 and engine.global_samples == 32
    assert engine.lr_scheduler.value == {"last_epoch": 2}

    master.grad = torch.ones_like(master)
    optimizer.step()
    assert int(state["step"].item()) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_restore_verifier_compares_cuda_target_on_cpu() -> None:
    source = torch.arange(8, dtype=torch.float32)
    target = source.cuda()
    assert _exact_equal_on_source_device(target, source)
