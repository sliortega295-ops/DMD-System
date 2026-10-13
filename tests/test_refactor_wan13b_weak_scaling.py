"""The world8 experiment is an explicit contract, not a global-constant patch."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_execution import WanDistributedExecution


def test_world16_default_is_unchanged_and_world8_requires_optin():
    ordinary = WanDistributedExecution()
    assert ordinary.world_size == ordinary.global_batch == 16
    assert ordinary.experiment == "canonical_world16"
    with pytest.raises(ValueError, match="explicit weak_scaling"):
        WanDistributedExecution(world_size=8)
    weak = WanDistributedExecution(world_size=8, experiment="weak_scaling")
    assert weak.global_batch == 8 and weak.batch_per_rank == 1
    assert weak.sample_count_per_cycle == 48
    assert weak.gradient_divisor == 8.0


@pytest.mark.parametrize("value", [True, 0, 2, 12, 32])
def test_unsupported_world_is_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        WanDistributedExecution(world_size=value, experiment="weak_scaling")


def test_world_contract_rejects_changed_batch_and_wrong_live_world():
    with pytest.raises(ValueError, match="B1"):
        WanDistributedExecution(world_size=8, batch_per_rank=2, experiment="weak_scaling")
    contract = WanDistributedExecution(world_size=8, experiment="weak_scaling")
    with pytest.raises(ValueError, match="live world"):
        contract.validate_live_world(16)
    contract.validate_live_world(8)


def test_existing_engine_defaults_remain_world16():
    import torch
    from distillschedule.runtime.wan13b_upper_bound import (
        GLOBAL_BATCH, WORLD_SIZE, ReplicatedTrainableAllReduceEngine,
    )
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter

    assert WORLD_SIZE == GLOBAL_BATCH == 16
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = ReplicatedTrainableAllReduceEngine(model, optimizer, scheduler, physical_step=2)
    assert engine.dp_world_size == engine.train_batch_size() == 16
    assert engine.global_samples == 32
    fsdp = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=2)
    assert fsdp.dp_world_size == fsdp.train_batch_size() == 16


def _mean_worker(rank, rendezvous, result_root):
    import torch
    import torch.distributed as dist
    from distillschedule.runtime.wan13b_upper_bound import ReplicatedTrainableAllReduceEngine

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8)
    try:
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        execution = WanDistributedExecution(world_size=8, experiment="weak_scaling")
        engine = ReplicatedTrainableAllReduceEngine(
            module, optimizer, scheduler, physical_step=2, gradient_clip=100.0,
            audit_gradients=True, execution=execution,
        )
        engine.backward(module.weight.sum() * (rank + 1))
        engine.step()
        row = {
            "rank": rank, "weight": module.weight.item(),
            "global_samples": engine.global_samples, "step": engine.global_steps,
            "dp_world_size": engine.dp_world_size, "batch": engine.train_batch_size(),
            "sync": engine.gradient_sync_receipts,
        }
        Path(result_root, f"rank{rank}.json").write_text(json.dumps(row))
    finally:
        dist.destroy_process_group()


def test_real_gloo_world8_mean_gradient_and_step_accounting(tmp_path):
    import torch.multiprocessing as mp

    rendezvous = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(_mean_worker, args=(rendezvous, str(tmp_path)), nprocs=8, join=True)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(8)]
    for row in rows:
        # Mean(1..8) is 4.5, not the old world16 division (2.25).
        assert row["weight"] == -3.5
        assert row["global_samples"] == 24 and row["step"] == 3
        assert row["dp_world_size"] == row["batch"] == 8
        assert row["sync"][0]["world_size"] == 8


def test_fsdp_policy_and_async_reducer_take_the_typed_execution():
    from distillschedule.runtime import wan13b_fair as fair
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter

    assert "execution" in inspect.signature(fair.apply_wan13b_fair_fsdp2_policy).parameters
    assert "execution" in inspect.signature(fair.build_fair_engine).parameters
    assert "dp_world_size" in inspect.signature(FSDP2EngineAdapter).parameters
    source = inspect.getsource(fair.GradReadyAsyncResidentEngine._finish_gradient_sync)
    assert "self.dp_world_size" in source
    source = inspect.getsource(fair.apply_wan13b_fair_fsdp2_policy)
    assert "execution.gradient_divisor" in source
    assert "execution.world_size" in source


def test_old_trace_defaults_and_explicit_rank8_admission():
    from runtime.static_world16_trace import StaticWorld16Trace
    assert "execution" in inspect.signature(StaticWorld16Trace).parameters
    source = inspect.getsource(StaticWorld16Trace)
    assert "self.world_size" in source
    assert "rank_slots" in source
    assert "len(bundle" in source


def test_worker_post_timer_validator_calls_match_real_api_signatures():
    import ast
    import importlib

    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/distillschedule/probe_wan13b_weakscale_two_cycle.py"
    tree = ast.parse(source.read_text())
    validators = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("distillschedule.runtime.wan13b_"):
            for alias in node.names:
                if alias.name.startswith("validate"):
                    module = importlib.import_module(node.module)
                    validators[alias.asname or alias.name] = inspect.signature(getattr(module, alias.name))
    assert len(validators) == 7
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in validators:
            validators[node.func.id].bind(*[object() for _ in node.args],
                                          **{kw.arg: object() for kw in node.keywords})
            count += 1
    assert count == 7

