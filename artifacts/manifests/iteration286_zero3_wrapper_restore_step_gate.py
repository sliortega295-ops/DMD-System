#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import gc
import weakref
import inspect
import types
from pathlib import Path

import torch
import torch.distributed as dist

from runtime.critic_pool_checkpoint import (
    LogicalCheckpointPackage,
    restore_zero3_optimizer_partition_from_logical,
)
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)
from scripts.distillschedule.probe_routeb_full5_2g6c_continuous import (
    _destroy_obsolete_zero3_engine,
    _verify_obsolete_zero3_release,
)


class Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(5))
        self.b = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a.sum() + self.b.sum()


def _write_package(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    values = {
        "master": torch.arange(8, dtype=torch.float32),
        "first_moment": torch.arange(8, dtype=torch.float32) + 10,
        "second_moment": torch.arange(8, dtype=torch.float32) + 20,
    }
    payloads = {}
    for key, value in values.items():
        path = root / f"{key}.f32"
        value.numpy().tofile(path)
        payloads[key] = {
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
                "scheduler_state": {
                    "base_lrs": [2e-6],
                    "last_epoch": 2,
                    "verbose": False,
                    "_step_count": 3,
                    "_get_lr_called_within_step": False,
                    "_last_lr": [2e-6],
                    "lr_lambdas": [None],
                },
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


def _step_values(engine) -> list[int]:
    values = []
    for state in engine.optimizer.state.values():
        if "step" not in state:
            continue
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise TypeError(f"non-native Adam step state: {type(step)}")
        values.append(int(step.item()))
    return values


def _referrer_summary(reference: weakref.ReferenceType) -> list[dict[str, object]]:
    value = reference()
    if value is None:
        return []
    summary = []
    for owner in gc.get_referrers(value):
        record: dict[str, object] = {"type": type(owner).__qualname__}
        if isinstance(owner, dict):
            record["keys"] = [
                str(key) for key, item in owner.items() if item is value
            ][:16]
            record["dict_size"] = len(owner)
            record["dict_owner_types"] = sorted(
                {
                    type(candidate).__qualname__
                    for candidate in gc.get_referrers(owner)
                    if getattr(candidate, "__dict__", None) is owner
                }
            )[:16]
        elif isinstance(owner, types.CellType):
            functions = []
            for closure in gc.get_referrers(owner):
                if not isinstance(closure, tuple):
                    continue
                for candidate in gc.get_referrers(closure):
                    if inspect.isfunction(candidate) and candidate.__closure__ is closure:
                        functions.append(
                            f"{candidate.__module__}.{candidate.__qualname__}"
                        )
            record["closure_functions"] = sorted(set(functions))[:32]
        elif isinstance(owner, (list, tuple)):
            record["length"] = len(owner)
        elif hasattr(owner, "__self__"):
            record["self_type"] = type(owner.__self__).__qualname__
        summary.append(record)
    return summary[:32]


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError("gate requires world2")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    root = Path(os.environ["GATE_ROOT"])
    package_root = root / "logical"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        _write_package(package_root)
    dist.barrier()

    model = Toy()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-6, betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    zero = build_zero_optimization_config(
        stage=3,
        reduce_bucket_mib=1,
        allgather_bucket_mib=1,
        overlap_comm=False,
        contiguous_gradients=False,
        stage3_param_persistence_threshold=0,
        optimizer_offload_device="cpu",
        optimizer_offload_pin_memory=False,
    )
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 2,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    config.update(build_client_optimizer_offload_overrides("cpu"))
    import deepspeed

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        dist_init_required=False,
    )
    package = LogicalCheckpointPackage.load(package_root)
    record = restore_zero3_optimizer_partition_from_logical(
        engine,
        package,
        component="generator",
        partition_rank=rank,
        dp_world_size=world,
    )

    expected = {
        0: {"a": [0.0, 1.0, 2.0], "b": [5.0, 6.0]},
        1: {"a": [3.0, 4.0, 0.0], "b": [7.0, 0.0]},
    }[rank]
    valid_lengths = {0: {"a": 3, "b": 2}, 1: {"a": 2, "b": 1}}[rank]
    exact_before = {}
    masters_before = [
        value.detach().cpu().clone()
        for value in engine.optimizer.fp32_partitioned_groups_flat
    ]
    for name, parameter in engine.module.named_parameters():
        master = engine.optimizer.get_local_fp32_param(parameter).cpu()
        moment = engine.optimizer.get_local_fp32_param(parameter, "exp_avg").cpu()
        variance = engine.optimizer.get_local_fp32_param(parameter, "exp_avg_sq").cpu()
        expected_master = torch.tensor(expected[name])
        expected_moment = torch.zeros_like(expected_master)
        expected_variance = torch.zeros_like(expected_master)
        valid = valid_lengths[name]
        expected_moment[:valid] = expected_master[:valid] + 10
        expected_variance[:valid] = expected_master[:valid] + 20
        exact_before[name] = bool(
            torch.equal(master, expected_master)
            and torch.equal(moment, expected_moment)
            and torch.equal(variance, expected_variance)
            and torch.equal(parameter.ds_tensor.float().cpu(), expected_master)
        )

    steps_before = _step_values(engine)
    if not all(exact_before.values()) or set(steps_before) != {2}:
        raise AssertionError(
            f"restore mismatch rank={rank} exact={exact_before} steps={steps_before}"
        )

    input_value = torch.tensor(1.0, device=engine.device, dtype=torch.bfloat16)
    loss = engine(input_value)
    engine.backward(loss)
    engine.step()
    steps_after = _step_values(engine)
    masters_after = [value.detach().cpu().clone() for value in engine.optimizer.fp32_partitioned_groups_flat]
    changed = any(not torch.equal(a, b) for a, b in zip(masters_before, masters_after))
    changed_flag = torch.tensor(int(changed), device=engine.device, dtype=torch.int32)
    dist.all_reduce(changed_flag, op=dist.ReduceOp.MAX)
    global_master_changed = bool(changed_flag.item())
    state_finite = all(
        torch.isfinite(value).all().item()
        for state in engine.optimizer.state.values()
        for key, value in state.items()
        if key in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor)
    )
    finite_flag = torch.tensor(int(state_finite), device=engine.device, dtype=torch.int32)
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    global_state_finite = bool(finite_flag.item())
    scheduler_step = int(engine.lr_scheduler.state_dict()["last_epoch"])
    engine_global_steps_after = int(engine.global_steps)
    update_status = (
        "PASS"
        if set(steps_after) == {3}
        and engine_global_steps_after == 3
        and scheduler_step == 3
        and global_master_changed
        and global_state_finite
        else "FAIL"
    )
    engine_ref = weakref.ref(engine)
    module_ref = weakref.ref(engine.module)
    optimizer_ref = weakref.ref(engine.optimizer)
    parameter_ref = weakref.ref(next(engine.module.parameters()))
    release = _destroy_obsolete_zero3_engine(engine, device=engine.device)
    engine = None
    model = None
    optimizer = None
    scheduler = None
    loss = None
    input_value = None
    parameter = None
    master = moment = variance = None
    gc.collect()
    release = _verify_obsolete_zero3_release(
        release, device=torch.device("cuda", local_rank)
    )
    object_liveness = {
        "engine": engine_ref() is not None,
        "module": module_ref() is not None,
        "optimizer": optimizer_ref() is not None,
        "parameter": parameter_ref() is not None,
    }
    live_referrers = {
        "engine": _referrer_summary(engine_ref),
        "module": _referrer_summary(module_ref),
        "parameter": _referrer_summary(parameter_ref),
    }
    released = float(release["released_allocated_gib"])
    release_flag = torch.tensor(
        int(released > 0.0),
        device=torch.device("cuda", local_rank),
        dtype=torch.int32,
    )
    dist.all_reduce(release_flag, op=dist.ReduceOp.MIN)
    status = "PASS" if update_status == "PASS" and release_flag.item() == 1 else "FAIL"
    out = {
        "status": status,
        "rank": rank,
        "torch": torch.__version__,
        "deepspeed": deepspeed.__version__,
        "record": record,
        "exact_before": exact_before,
        "optimizer_steps_before": steps_before,
        "optimizer_steps_after": steps_after,
        "engine_global_steps_after": engine_global_steps_after,
        "scheduler_step_after": scheduler_step,
        "master_changed": changed,
        "global_master_changed": global_master_changed,
        "optimizer_state_finite": state_finite,
        "global_optimizer_state_finite": global_state_finite,
        "zero3_release": release,
        "object_liveness_after_release": object_liveness,
        "live_referrers_after_release": live_referrers,
    }
    (root / f"rank_{rank:02d}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()
    if status != "PASS":
        raise SystemExit(out)


if __name__ == "__main__":
    main()
