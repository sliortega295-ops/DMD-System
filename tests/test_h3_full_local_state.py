"""Real CPU producer -> consumer; SYNTHETIC_TEST_ONLY, never GPU evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.backends.training.minimax_h3 import local_state_probe as p


def fixture():
    torch.manual_seed(194)
    module = torch.nn.Module()
    module.block = torch.nn.Module()
    for kind in ("A", "B"):
        setattr(module.block, "lora_" + kind, torch.nn.ModuleDict({
            role: torch.nn.Linear(4, 2, bias=False) for role in ("student", "fake")}))
    params = {role: [v for n, v in module.named_parameters() if f".{role}." in n]
              for role in ("student", "fake")}
    return SimpleNamespace(shared_model=SimpleNamespace(denoiser_module=lambda: module),
                           trainable_params=params["student"], fake_trainable_params=params["fake"],
                           optimizer=torch.optim.AdamW(params["student"], lr=1e-3),
                           fake_optimizer=torch.optim.AdamW(params["fake"], lr=1e-3))


def rng():
    return {"CPU_SYNTHETIC_TEST_ONLY": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def probe(root, **kwargs):
    return p.FullLocalStateProbe(torch=torch, rank=0, source_head="a" * 40,
        run_root=root, run_id="SYNTHETIC_TEST_ONLY", arm="d2A", rng_snapshot=rng, **kwargs)


def update(trainer):
    for parameters, optimizer in ((trainer.trainable_params, trainer.optimizer),
                                  (trainer.fake_trainable_params, trainer.fake_optimizer)):
        optimizer.zero_grad()
        sum((value * torch.randn_like(value)).square().sum() for value in parameters).backward()
        optimizer.step()


def capture_three(tmp_path):
    trainer = fixture()
    observer = probe(tmp_path)
    observer.capture(trainer, "initial", outside_cycle=True)
    for tag in ("cycle1", "cycle2"):
        update(trainer)
        observer.capture(trainer, tag, outside_cycle=True)
    return trainer, observer, observer.receipt()


def test_real_cpu_capture_roundtrip_preserves_all_local_values_rng_and_updates(tmp_path):
    trainer, observer, receipt = capture_three(tmp_path)
    assert p.verify_receipt(receipt, root=tmp_path, rank=0, source_head="a" * 40,
                            run_id="SYNTHETIC_TEST_ONLY", arm="d2A")
    assert receipt["snapshots"][0]["tensor_count"] == 4
    assert receipt["snapshots"][-1]["tensor_count"] == 16
    saved = torch.load(Path(receipt["snapshots"][-1]["path"]).with_name("state.pt"), weights_only=True)
    rows, _ = p.inventory(trainer, torch)
    assert set(saved) == {row["key"] for row, value in rows if torch.is_tensor(value)}
    for row, value in rows:
        if torch.is_tensor(value):
            assert torch.equal(saved[row["key"]], value.detach())
            assert saved[row["key"]].dtype == value.dtype
    without = fixture()
    update(without)
    update(without)
    left, _ = p.inventory(trainer, torch)
    right, _ = p.inventory(without, torch)
    assert rng() == observer.rng_snapshot()
    for (a, av), (b, bv) in zip(left, right, strict=True):
        assert a == b and torch.equal(av.detach(), bv.detach())
    assert not torch.cuda.is_initialized()


def test_refuses_wrong_order_inside_cycle_repeated_snapshot_and_oversize(tmp_path, monkeypatch):
    trainer = fixture()
    observer = probe(tmp_path)
    with pytest.raises(ValueError): observer.capture(trainer, "cycle1", outside_cycle=True)
    with pytest.raises(ValueError): observer.capture(trainer, "initial", outside_cycle=False)
    monkeypatch.setattr(p, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="bound"): observer.capture(trainer, "initial", outside_cycle=True)
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("mutation", ("source", "rank", "arm", "missing", "hash", "cycles", "admission"))
def test_consumer_rejects_real_producer_files_with_missing_or_drifted_bindings(tmp_path, mutation):
    _, _, receipt = capture_three(tmp_path)
    kwargs = dict(root=tmp_path, rank=0, source_head="a" * 40, run_id="SYNTHETIC_TEST_ONLY", arm="d2A")
    if mutation == "source": kwargs["source_head"] = "b" * 40
    elif mutation == "rank": kwargs["rank"] = 1
    elif mutation == "arm": kwargs["arm"] = "d1A"
    elif mutation == "missing": Path(receipt["snapshots"][1]["path"]).with_name("state.pt").unlink()
    elif mutation == "hash": Path(receipt["snapshots"][1]["path"]).with_name("state.pt").write_bytes(b"bad")
    elif mutation == "cycles": receipt["snapshots"].pop()
    elif mutation == "admission": receipt["numerical_admission"] = True
    with pytest.raises(ValueError): p.verify_receipt(receipt, **kwargs)


def test_inventory_rejects_unknown_optimizer_state_and_nonfinite(tmp_path):
    trainer = fixture()
    update(trainer)
    trainer.optimizer.state[trainer.trainable_params[0]]["unknown"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected optimizer"): probe(tmp_path).capture(trainer, "initial", outside_cycle=True)
    trainer.optimizer.state[trainer.trainable_params[0]].pop("unknown")
    with torch.no_grad(): trainer.trainable_params[0].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"): probe(tmp_path).capture(trainer, "initial", outside_cycle=True)


def test_local_dtensor_path_never_requests_global_reconstruction():
    calls = []
    local = torch.ones(2, 3)

    class LocalOnly:
        def detach(self): calls.append("detach"); return self
        def to_local(self): calls.append("to_local"); return local
        def full_tensor(self): raise AssertionError("global gather forbidden")

    assert p._local(LocalOnly()) is local
    assert calls == ["detach", "to_local"]


def test_new_flag_default_unchanged_source_bound_exclusive_and_never_timing():
    from test_h3_cache_roundtrip import base_plan
    from distillschedule.backends.training.minimax_h3 import divergence_probe as d
    base = base_plan()
    control = d.bind_plan(base, "d1A")
    assert "full_local_state" not in control["h3_divergence"]
    child = d.bind_plan(base, "d1A", full_local_state=True)
    assert child["h3_divergence"]["full_local_state"] == p.source_contract()
    assert d.is_diagnostic_plan(child)
    for mutation in (
        lambda x: x["nodes"][1]["environment"].__setitem__("H3_PLANNER_FULL_LOCAL_STATE", "0"),
        lambda x: x["h3_divergence"]["full_local_state"].__setitem__("helper_sha256", "0" * 64),
        lambda x: x.__setitem__("mode", "timing"),
        lambda x: x["nodes"][0]["environment"].__setitem__("H3_PLANNER_RAW_BACKWARD_REPEAT", "1"),
    ):
        bad = deepcopy(child); mutation(bad)
        with pytest.raises(ValueError): d.is_diagnostic_plan(bad)
    with pytest.raises(ValueError): d.bind_plan(base, "d1A", full_local_state=1)
    with pytest.raises(ValueError): d.bind_plan(base, "d1A", full_local_state=True, cache_roundtrip=True)
    orphan = deepcopy(base); orphan["nodes"][0]["environment"]["H3_PLANNER_FULL_LOCAL_STATE"] = "1"
    with pytest.raises(ValueError): d.is_diagnostic_plan(orphan)
    assert not torch.cuda.is_initialized()


def test_real_divergence_producer_reaches_snapshot_writer_and_consumer(tmp_path, monkeypatch):
    from distillschedule.backends.training.minimax_h3 import divergence_probe as d
    module = torch.nn.Module()
    for name in d.PARAMETERS:
        current = module
        for field in name.split(".")[:-1]:
            if field not in current._modules:
                current.add_module(field, torch.nn.Module())
            current = current._modules[field]
        current.register_parameter("weight", torch.nn.Parameter(torch.arange(8.).view(2, 4) / 16))
    params = {role: [v for n, v in module.named_parameters() if f".{role}." in n]
              for role in ("student", "fake")}
    trainer = SimpleNamespace(shared_model=SimpleNamespace(denoiser_module=lambda: module, device="cpu"),
        trainable_params=params["student"], fake_trainable_params=params["fake"],
        optimizer=torch.optim.AdamW(params["student"]), fake_optimizer=torch.optim.AdamW(params["fake"]),
        fa3_replay_cache_registration=SimpleNamespace(receipt=lambda: {"SYNTHETIC_TEST_ONLY": True}))
    # CPU-only facade for the existing CUDA RNG-reader boundary, not GPU proof.
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    observed = d.FirstFakeProbe(torch=torch, rank=0, source_head="a" * 40,
        run_root=tmp_path, run_id="SYNTHETIC_TEST_ONLY", arm="d2A", candidate_id=d.ARMS["d2A"])
    observed.full_state_enabled = True
    try:
        observed.initial(trainer, rng())
        for cycle in (1, 2):
            observed.begin_cycle(cycle, rng())
            update(trainer)
            observed.fake_index = 5  # Fixture native-loop completion; not real model census evidence.
            observed.after_cycle(cycle, rng(), trainer=trainer)
        receipt = observed.full_state.receipt()
        assert p.verify_receipt(receipt, root=tmp_path, rank=0, source_head="a" * 40,
                                run_id="SYNTHETIC_TEST_ONLY", arm="d2A")
    finally:
        observed.close()
    assert d._ACTIVE is None and not torch.cuda.is_initialized()


def test_observer_passes_native_trainer_only_after_stop_boundary():
    import inspect
    from distillschedule.backends.training.minimax_h3.planner_cycle_observer import H3PlannerCycleObserver
    source = inspect.getsource(H3PlannerCycleObserver.after_cycle)
    assert source.index("if self.active is not None") < source.index("trainer=trainer")
    assert "full_state_enabled" in source
    # All model and optimizer invocation sites remain in the unmodified native loop.
    import ast
    tree = ast.parse(Path(p.__file__).read_text())
    banned = {"full_tensor", "all_gather", "all_reduce", "barrier", "backward", "step",
              "manual_seed", "rand", "randn", "set_rng_state"}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in banned for n in ast.walk(tree))
