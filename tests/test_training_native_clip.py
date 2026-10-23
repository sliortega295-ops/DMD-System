"""Optimizer-owned clipping on real Torch, original GAN helpers and CPU DTensor.

The original-helper fixture supplies source-qualified loss/name/reset code and
the exact head on tiny CPU construction, not pretrained Helios or public W3.
"""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
from test_training_end_to_end import equal
from test_training_helios_gan_state import (
    SOURCES, _assert_moments, _owner, _update, original as original,
)


def _scalar_engine(owned=("head", "base")):
    model = nn.Module()
    for name in ("base", "head", "unowned"):
        model.register_parameter(name, nn.Parameter(torch.tensor(0.)))
    optimizer = torch.optim.SGD([{"params": [getattr(model, n)]} for n in owned], lr=1.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
    engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=3, global_batch=2)
    engine._trajectory_capture_grad_norm = True
    return engine


def test_static_all_trainable_default_math_matches_real_torch():
    model = nn.ParameterList([nn.Parameter(torch.tensor(0.)), nn.Parameter(torch.tensor(0.))])
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
    engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0)
    reference = nn.ParameterList([nn.Parameter(p.detach().clone()) for p in model])
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1.)
    engine.backward(4 * model[0] + 3 * model[1])
    (4 * reference[0] + 3 * reference[1]).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    reference_optimizer.step()
    engine.step()
    assert all(torch.equal(p, q) for p, q in zip(model, reference))
    assert all(p.requires_grad and p.grad is None for p in model)
    assert engine.global_steps == engine.step_calls == scheduler.last_epoch == 1
    assert engine.global_samples == 16 and not hasattr(engine, "last_gradient_norm")


@pytest.mark.parametrize("disabled", [(), ("base",), ("base", "head")])
def test_owned_accumulated_gradients_are_clipped_even_after_flags_change(disabled):
    engine = _scalar_engine()
    engine.module.base.grad = torch.tensor(4.)
    engine.module.head.grad = torch.tensor(3.)
    for name in disabled:
        getattr(engine.module, name).requires_grad_(False)
    owned = [p for group in engine.optimizer.param_groups for p in group["params"]]
    expected = [nn.Parameter(p.detach().clone()) for p in owned]
    for p, reference in zip(owned, expected):
        reference.grad = p.grad.clone()
    reference_optimizer = torch.optim.SGD(expected, lr=1.)
    expected_norm = torch.nn.utils.clip_grad_norm_(expected, 1.)
    reference_optimizer.step()
    engine.step()
    for parameter, reference in zip(owned, expected):
        assert torch.equal(parameter, reference)
        assert parameter.grad is None
    assert engine.last_gradient_norm == float(expected_norm) == 5.
    assert engine.global_steps == 4 and engine.global_samples == 8 and engine.step_calls == 1
    assert engine.lr_scheduler.last_epoch == 1
    assert all(group["lr"] == .7 for group in engine.optimizer.param_groups)
    assert all(p.requires_grad is (n not in disabled) for n, p in engine.module.named_parameters())


def test_optimizer_group_order_and_missing_gradients_reach_the_real_clip(monkeypatch):
    engine = _scalar_engine()
    engine.module.head.grad = torch.tensor(3.)
    owned = [p for group in engine.optimizer.param_groups for p in group["params"]]
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def observe(parameters, max_norm):
        parameters = list(parameters)
        assert [id(p) for p in parameters] == [id(p) for p in owned]
        assert parameters[0].grad.item() == 3. and parameters[1].grad is None
        calls.append(max_norm)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    engine.step()
    assert calls == [1.] and engine.last_gradient_norm == 3.
    assert engine.module.base.item() == engine.module.unowned.item() == 0.


def test_unowned_high_gradients_do_not_affect_clipping_or_get_cleared(monkeypatch):
    engine = _scalar_engine(("head",))
    engine.module.base.grad = torch.tensor(1000.)
    engine.module.unowned.grad = torch.tensor(2000.)
    engine.module.head.grad = torch.tensor(3.)
    real_clip = torch.nn.utils.clip_grad_norm_
    reference = nn.Parameter(torch.tensor(0.))
    reference.grad = torch.tensor(3.)
    expected_norm = real_clip([reference], 1.)
    optimizer = torch.optim.SGD([reference], lr=1.)
    optimizer.step()
    calls = []

    def observe(parameters, max_norm):
        parameters = list(parameters)
        assert [id(p) for p in parameters] == [id(engine.module.head)]
        calls.append(max_norm)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    engine.step()
    assert calls == [1.] and engine.last_gradient_norm == float(expected_norm) == 3.
    assert torch.equal(engine.module.head, reference) and engine.module.head.grad is None
    assert engine.module.base.item() == engine.module.unowned.item() == 0.
    assert engine.module.base.grad.item() == 1000. and engine.module.unowned.grad.item() == 2000.


def test_empty_optimizer_ownership_refuses_before_clipping_or_clock_mutation(monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD([{"params": []}], lr=1.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=3, global_batch=2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    gradients = [p.grad.clone() for p in model.parameters()]
    calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def observe(parameters, max_norm):
        calls.append(True)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    with pytest.raises(RuntimeError, match="optimizer.*no parameters"):
        engine.step()
    assert not calls
    assert equal(before, (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), gradients))
    assert engine.global_steps == 3 and engine.global_samples == 6 and engine.step_calls == 0
    assert not hasattr(engine, "last_gradient_norm")


@pytest.mark.parametrize("separate", [True, False])
def test_original_helper_accumulated_gradients_match_independent_real_clip_and_adam(
    original, monkeypatch, tmp_path, separate,
):
    owner = _owner(original)
    engine = owner.engine
    # Explicitly force real clipping rather than merely witnessing its input.
    engine._gradient_clip = 1e-5
    engine._trajectory_capture_grad_norm = True
    named = dict(owner.model.named_parameters())
    names = {id(p): name for name, p in named.items()}
    owned = [p for group in engine.optimizer.param_groups for p in group["params"]]
    owned_names = [names[id(p)] for p in owned]
    assert set(owned_names) == owner.base | owner.extra
    reference = {n: nn.Parameter(named[n].detach().clone()) for n in owned_names}
    groups = [{**{k: deepcopy(v) for k, v in group.items() if k != "params"},
               "params": [reference[names[id(p)]] for p in group["params"]]}
              for group in engine.optimizer.param_groups]
    reference_optimizer = torch.optim.AdamW(groups)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(
        reference_optimizer, step_size=engine.lr_scheduler.step_size, gamma=engine.lr_scheduler.gamma)
    reference_optimizer.load_state_dict(deepcopy(engine.optimizer.state_dict()))
    reference_scheduler.load_state_dict(deepcopy(engine.lr_scheduler.state_dict()))
    real_clip = torch.nn.utils.clip_grad_norm_
    witnesses, backwards = [], []

    def after_backward(boundary):
        backwards.append((boundary, {n: p.grad.detach().clone() for n, p in named.items()
                                     if p.grad is not None}))

    def observe(parameters, max_norm):
        parameters = list(parameters)
        # These ownership/phase assertions precede the real optimizer mutation.
        assert [id(p) for p in parameters] == [id(p) for p in owned]
        assert max_norm == 1e-5
        assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in parameters)
        assert all(named[n].requires_grad is (not separate) for n in owner.base)
        assert all(named[n].requires_grad is True for n in owner.extra)
        assert all(named[n].requires_grad is False for n in owner.frozen)
        first_boundary, first = backwards[-2]
        final_boundary, final = backwards[-1]
        assert first_boundary is False and final_boundary is True
        assert set(first) == owner.base and set(final) == owner.base | owner.extra
        if separate:
            assert all(torch.equal(first[n], final[n]) for n in owner.base)
        for name in owned_names:
            reference[name].grad = final[name].clone()
        expected_norm = real_clip(list(reference.values()), max_norm)
        assert expected_norm.item() > max_norm
        observed_norm = real_clip(parameters, max_norm)
        assert torch.equal(observed_norm, expected_norm)
        assert all(torch.equal(named[n].grad, reference[n].grad) for n in owned_names)
        witnesses.append({"owned_names": owned_names, "max_norm": max_norm,
                          "unclipped_norm": observed_norm.item(),
                          "nonzero_gradients": {n: int(torch.count_nonzero(final[n])) for n in owned_names}})
        reference_optimizer.step()
        reference_scheduler.step()
        reference_optimizer.zero_grad(set_to_none=True)
        return observed_norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    helper_calls = []
    for update in (1, 2):
        _, _, calls = _update(original, owner, separate, after_backward)
        helper_calls.append(calls)
        assert len(witnesses) == update and len(backwards) == 2 * update
        assert engine.last_gradient_norm == witnesses[-1]["unclipped_norm"]
        _assert_moments(owner, update)
        for name in owned_names:
            assert torch.equal(named[name], reference[name]), name
            assert equal(engine.optimizer.state[named[name]], reference_optimizer.state[reference[name]]), name
        assert equal(engine.optimizer.state_dict(), reference_optimizer.state_dict())
        assert equal(engine.lr_scheduler.state_dict(), reference_scheduler.state_dict())
    (tmp_path / "original-helper-clip-proof.json").write_text(json.dumps({
        "scope": "EXACT_ORIGINAL_HELPER_HEAD_NAME_RESET_TINY_CPU_NOT_PRETRAINED_OR_PUBLIC_W3",
        "source_sha256": SOURCES, "torch": torch.__version__, "separate": separate,
        "helper_calls": helper_calls, "real_clip_witnesses": witnesses,
        "named_parameters_and_adam_moments_exact": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    }, indent=2) + "\n")


def _dtensor_clip_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = nn.Module()
        full_gradients = {"base": torch.tensor([4., 0.]), "head": torch.tensor([0., 3.]),
                          "unowned": torch.tensor([1000., 1000.])}
        for name, gradient in full_gradients.items():
            parameter = nn.Parameter(distribute_tensor(torch.zeros(2), mesh, [Shard(0)]))
            parameter.grad = distribute_tensor(gradient, mesh, [Shard(0)])
            model.register_parameter(name, parameter)
        model.base.requires_grad_(False)
        optimizer = torch.optim.SGD([{"params": [model.head]}, {"params": [model.base]}], lr=1.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
        engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0, global_batch=2)
        engine._trajectory_capture_grad_norm = True
        expected = {n: nn.Parameter(torch.zeros(2)) for n in ("head", "base")}
        for name, parameter in expected.items():
            parameter.grad = full_gradients[name].clone()
        expected_norm = torch.nn.utils.clip_grad_norm_(list(expected.values()), 1.)
        reference_optimizer = torch.optim.SGD(list(expected.values()), lr=1.)
        reference_optimizer.step()
        engine.step()
        assert engine.last_gradient_norm == expected_norm.item() == 5.
        for name, reference in expected.items():
            assert torch.equal(getattr(model, name).full_tensor(), reference), name
            assert getattr(model, name).grad is None
        assert torch.equal(model.unowned.grad.full_tensor(), full_gradients["unowned"])
        assert torch.count_nonzero(model.unowned.full_tensor()) == 0
        assert model.base.requires_grad is False and model.head.requires_grad is True
        assert engine.step_calls == engine.global_steps == scheduler.last_epoch == 1
        assert engine.global_samples == 2 and torch.cuda.is_initialized() is False
        (Path(output) / f"rank-{rank}.json").write_text(json.dumps({
            "rank": rank, "world_size": 2, "backend": "gloo", "placements": "Shard(0)",
            "torch": torch.__version__, "global_owned_gradient_norm": engine.last_gradient_norm,
            "owned_update_exact": True, "unowned_gradient_untouched": True,
            "cuda_initialized": torch.cuda.is_initialized(),
        }) + "\n")
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_dtensor_clip_uses_global_optimizer_owned_norm(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_dtensor_clip_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)),
             nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [r["rank"] for r in ranks] == [0, 1]
    assert all(r["global_owned_gradient_norm"] == 5. for r in ranks)
