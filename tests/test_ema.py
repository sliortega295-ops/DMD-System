import copy

import pytest
import torch

from runtime.ema import TrainableStateEMA


def test_trainable_state_ema_omits_frozen_state_and_matches_helios_decay():
    trainable = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
    frozen = torch.nn.Parameter(torch.tensor([7.0, 9.0]), requires_grad=False)
    ema = TrainableStateEMA(
        [("trainable", trainable), ("frozen", frozen)],
        decay=0.99,
        update_after_step=750,
    )

    assert ema.entry_count == 1
    assert ema.trainable_logical_numel == 2
    assert ema.frozen_logical_numel == 2
    assert ema.logical_state_bytes == 8

    trainable.data.add_(2.0)
    ema.optimization_step = 751
    decay = ema.step()
    assert decay == 2 / 11
    expected = torch.tensor([1.0, 3.0]) * decay + trainable.detach() * (1 - decay)
    torch.testing.assert_close(dict(ema.named_state_tensors())["trainable"], expected)
    torch.testing.assert_close(frozen, torch.tensor([7.0, 9.0]))


def test_trainable_state_ema_uses_current_partition_reference():
    parameter = torch.nn.Parameter(torch.empty(0))
    parameter.ds_numel = 4
    parameter.ds_tensor = torch.tensor([1.0, 2.0])
    ema = TrainableStateEMA([("partitioned", parameter)])

    parameter.ds_tensor = torch.tensor([5.0, 6.0])
    ema.step()
    torch.testing.assert_close(next(ema.state_tensors()), parameter.ds_tensor)


def test_trainable_state_ema_can_offload_shadow_to_cpu():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 4.0]))
    ema = TrainableStateEMA([("offloaded", parameter)], device="cpu")
    assert next(ema.state_tensors()).device.type == "cpu"
    parameter.data.add_(1.0)
    ema.step()
    torch.testing.assert_close(next(ema.state_tensors()), parameter.detach())


def test_trainable_state_ema_matches_upstream_for_mutable_parameters():
    upstream = pytest.importorskip("helios.utils.create_ema_zero3")
    EMAModel_Zero3 = upstream.EMAModel_Zero3
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    full = EMAModel_Zero3(copy.deepcopy(model), decay=0.99, update_after_step=750)
    sparse = TrainableStateEMA(
        model.named_parameters(), decay=0.99, update_after_step=750, device="cpu"
    )
    full.optimization_step = sparse.optimization_step = 751

    for parameter in model[1].parameters():
        parameter.data.add_(0.25)
    full.step(model.parameters())
    sparse.step()

    full_trainable = {
        name: parameter for name, parameter in full.model.named_parameters() if parameter.requires_grad
    }
    for name, shadow in sparse.named_state_tensors():
        torch.testing.assert_close(shadow, full_trainable[name])
    for source, shadow in zip(model[0].parameters(), full.model[0].parameters()):
        torch.testing.assert_close(source, shadow)
