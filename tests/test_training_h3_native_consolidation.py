"""Real installed DS consolidation body; CPU collectives are TEST-ONLY doubles.

This does not construct a ZeRO-3 CUDA engine, run H3 training or prove native
world16 execution. Original H3Adapter classes are reused from the R32 fixture.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def test_native_consolidation_discovery_does_not_import_models_or_torch():
    script = "import sys;from distillschedule.training.h3_ar_inference import consolidate_current_student;assert callable(consolidate_current_student);assert 'torch' not in sys.modules;assert 'deepspeed' not in sys.modules"
    result = subprocess.run([sys.executable, "-B", "-c", script], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def native_method_fixture(monkeypatch):
    from test_training_h3_ar_continuation import ProtocolEngine
    from distillschedule.strategies.h3_ar_session import H3ARSession
    from distillschedule.training.h3_ar_state import NativeCheckpoint
    from distillschedule.training import h3_ar_inference as output
    from distillschedule.training import checkpoint
    ds = importlib.import_module("deepspeed.runtime.engine")
    source = ProtocolEngine()
    engine = ds.DeepSpeedEngine.__new__(ds.DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    # DeepSpeed keeps module in __dict__, unlike nn.Module registration.
    # This is an unconstructed CPU serialization harness, not a real engine.
    object.__setattr__(engine, "module", source.module)
    object.__setattr__(engine, "destroy", lambda: None)
    engine.optimizer = source.optimizer
    engine.global_steps = 6
    engine.zero_optimization_partition_weights = lambda: True
    engine.zero_optimization_stage = lambda: 3
    engine._optimizer_has_ckpt_event_prologue = lambda: False
    engine._optimizer_has_ckpt_event_epilogue = lambda: False
    engine.is_gradient_accumulation_boundary = lambda: True
    for index, parameter in enumerate(engine.module.parameters()):
        parameter.ds_id = index
        parameter.ds_shape = parameter.shape
    source.module.adapter.select("fake_score", trainable=True)
    source.optimizer.active_role, source.optimizer.train_heads = "student", False
    # Nonzero, distinguishable current Student and Fake updates; no training is
    # claimed by this serialization fixture.
    with torch.no_grad():
        for p in source.module.adapter.role_parameters("student"): p.fill_(.375)
        for p in source.module.adapter.role_parameters("fake_score"): p.fill_(-.625)
    session = H3ARSession.__new__(H3ARSession)
    session.engine, session.adapter = engine, source.module.adapter
    session.optimizer, session.discriminator = engine.optimizer, source.module.discriminator
    session.native = source.native_api
    session.trainer = SimpleNamespace(generator_step=3, critic_step=3)
    session.rank, session.local_completed, session.device, session.pool = 0, 0, torch.device("cuda:0"), None
    session.input_provider = SimpleNamespace(completed=3, pending=None)
    native = NativeCheckpoint.__new__(NativeCheckpoint)
    native.session = session
    native.progress = SimpleNamespace(cursor=session.input_provider)
    events = []
    @contextmanager
    def cpu_collective(parameters, modifier_rank):
        events.append(("TEST_ONLY_GATHER", len(parameters), modifier_rank))
        yield
    monkeypatch.setattr(ds.deepspeed.zero, "GatheredParameters", cpu_collective)
    monkeypatch.setattr(ds.dist, "get_rank", lambda: session.rank)
    monkeypatch.setattr(output.distributed, "rank_world", lambda: (session.rank, 16))
    monkeypatch.setattr(output.distributed, "gather", lambda value: [deepcopy(value) for _ in range(16)])
    monkeypatch.setattr(output.distributed, "checked", lambda callback, label: callback())
    monkeypatch.setattr(output, "admit_host_memory", lambda needed, reserve: {
        "required_bytes": needed, "reserve_bytes": reserve, "available_bytes": 2**40})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2**40, 2**40))
    cpu_rng = checkpoint.rng_state
    monkeypatch.setattr(output, "rng_state", lambda **kw: cpu_rng(device="cpu"))
    budget = dict(max_parameters=40_000_000_000, max_consolidated_bytes=256*2**30,
                  host_reserve_bytes=2**20, device_reserve_bytes=2**20)
    return output, session, native, budget, events


def test_current_unmerged_student_uses_real_installed_method_and_keeps_owner(native_method_fixture):
    output, session, native, budget, events = native_method_fixture
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    before = output.owner_state(session)
    rng = rng_state(device="cpu")
    expected = {k: v.detach().clone() for k, v in session.adapter.state_dict().items()
                if ".adapters.fake_score." not in k}
    values, receipt = output.consolidate_current_student(session, native, budget=budget)
    assert values.keys() == expected.keys()
    assert all(torch.equal(values[k], v) for k, v in expected.items())
    assert not any("fake_score" in k or "discriminator" in k for k in values)
    assert any("adapters.student.A.weight" in k for k in values)
    assert all(t.device.type == "cpu" for t in values.values())
    assert receipt["cycle"] == 3 and receipt["versions"] == dict(student=3, fake=3, ema=0)
    assert receipt["merged_lora"] is False and receipt["native_global_steps"] == 6
    assert before == output.owner_state(session) and equal(rng, rng_state(device="cpu"))
    assert any(row[0] == "TEST_ONLY_GATHER" for row in events)
    assert all(values[k].data_ptr() != v.data_ptr() for k, v in session.adapter.state_dict().items() if k in values)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["pending", "clock", "budget", "host", "device", "method"])
def test_invalid_owner_or_capacity_fails_before_native_gather(native_method_fixture, monkeypatch, fault):
    output, session, native, budget, events = native_method_fixture
    if fault == "pending": session.input_provider.pending = 4
    elif fault == "clock": session.engine.global_steps = 5
    elif fault == "budget": budget["max_parameters"] = 1
    elif fault == "host":
        monkeypatch.setattr(output, "admit_host_memory", lambda *a: (_ for _ in ()).throw(ValueError("host capacity")))
    elif fault == "device": monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (0, 2**40))
    elif fault == "method": session.engine._zero3_consolidated_16bit_state_dict = lambda **kw: {}
    with pytest.raises(ValueError): output.consolidate_current_student(session, native, budget=budget)
    assert events == []


def test_nonprimary_returns_no_cpu_snapshot(native_method_fixture):
    output, session, native, budget, events = native_method_fixture
    session.rank = 7
    values, receipt = output.consolidate_current_student(session, native, budget=budget)
    assert values is None and receipt["retained_on_primary_only"] is True
    assert any(row[0] == "TEST_ONLY_GATHER" for row in events)


def test_backend_failure_restores_rng_and_preserves_error(native_method_fixture, monkeypatch):
    output, session, native, budget, events = native_method_fixture
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    ds = importlib.import_module("deepspeed.runtime.engine")
    @contextmanager
    def fail(parameters, modifier_rank):
        torch.rand(1)
        raise RuntimeError("injected native gather failure")
        yield
    monkeypatch.setattr(ds.deepspeed.zero, "GatheredParameters", fail)
    before, rng = output.owner_state(session), rng_state(device="cpu")
    with pytest.raises(RuntimeError, match="injected native gather failure"):
        output.consolidate_current_student(session, native, budget=budget)
    assert before == output.owner_state(session) and equal(rng, rng_state(device="cpu"))
