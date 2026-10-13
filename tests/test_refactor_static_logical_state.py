"""Actual CPU checkpoint state and BF16-compute/FP32-master contract."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def test_root_linear_and_conv_keep_fp32_master_with_bf16_compute():
    from distillschedule.runtime.static_logical_state import MasterComputeProjection

    torch.manual_seed(2301)
    for original, value in (
        (nn.Linear(8, 4, bias=False), torch.randn(2, 8, dtype=torch.bfloat16)),
        (nn.Conv3d(2, 4, 1, bias=False), torch.randn(1, 2, 2, 3, 3, dtype=torch.bfloat16)),
    ):
        expected_weight = original.weight.detach().bfloat16().requires_grad_()
        expected = (F.linear(value, expected_weight) if isinstance(original, nn.Linear)
                    else F.conv3d(value, expected_weight))
        expected.float().square().sum().backward()
        wrapped = MasterComputeProjection(original)
        actual = wrapped(value.float())  # PEFT may cast from its master metadata.
        actual.float().square().sum().backward()
        assert wrapped.weight is original.weight
        assert wrapped.weight.dtype == wrapped.weight.grad.dtype == torch.float32
        assert list(dict(wrapped.named_parameters())) == ['weight']
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(wrapped.weight.grad, expected_weight.grad.float(), rtol=0, atol=0)
    with pytest.raises(ValueError, match='Linear/Conv3d'):
        MasterComputeProjection(nn.LayerNorm(8))


def _package(tmp_path):
    names = ['first.lora_B.default.weight', 'root.lora_A.default.weight']
    shapes = [(8, 3), (2, 3)]
    records = {}
    offset = 0
    for name, shape in zip(names, shapes):
        n = torch.tensor(shape).prod().item()
        records[name] = dict(offset_elements=offset, numel=n, shape=shape)
        offset += n
    payloads = {}
    for index, payload in enumerate(('first_moment', 'second_moment')):
        data = torch.arange(offset, dtype=torch.float32) + index + 1
        path = tmp_path / (payload + '.bin')
        path.write_bytes(data.numpy().tobytes())
        payloads[payload] = dict(path=path.name, bytes=offset*4, dtype='torch.float32')
    spec = dict(parameter_names=names, records=records, payloads=payloads,
                optimizer_step=2, scheduler_state=dict(last_epoch=2),
                optimizer=dict(algorithm='AdamW', param_groups=[dict(lr=2e-6,
                    betas=[0., .999], eps=1e-8, weight_decay=.001)]))
    return SimpleNamespace(root=tmp_path, component=lambda _role: spec), names


def test_logical_adam_restores_real_nonzero_tp_shards_not_fresh_zero_state(tmp_path):
    from distillschedule.runtime.static_logical_state import restore_logical_static_optimizer
    from distillschedule.runtime.static_3d_state import StaticParameterPlacement

    package, names = _package(tmp_path)
    items = [StaticParameterPlacement(names[0], nn.Parameter(torch.ones(2, 3)), 'output', 0),
             StaticParameterPlacement(names[1], nn.Parameter(torch.ones(2, 3)), 'replicated', None)]
    optimizer = torch.optim.AdamW([p.parameter for p in items], lr=2e-6,
                                 betas=(0., .999), eps=1e-8, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    # A real LambdaLR state, not an incomplete synthetic last_epoch fragment.
    package.component('generator')['scheduler_state'] = scheduler.state_dict()
    package.component('generator')['scheduler_state'].update(last_epoch=2, _step_count=3)
    receipt = restore_logical_static_optimizer(optimizer=optimizer, scheduler=scheduler,
        placements=items, package=package, component='generator', tp_rank=2)
    assert receipt['optimizer_step'] == 2
    assert receipt['moments'] == 'EXACT_LOGICAL_FP32_PAYLOAD_TO_DECLARED_TP_PLACEMENT'
    torch.testing.assert_close(optimizer.state[items[0].parameter]['exp_avg'],
                               torch.arange(13, 19, dtype=torch.float32).reshape(2, 3))
    torch.testing.assert_close(optimizer.state[items[1].parameter]['exp_avg_sq'],
                               torch.arange(26, 32, dtype=torch.float32).reshape(2, 3))
    assert scheduler.last_epoch == 2
    optimizer.param_groups[0]['lr'] = 1e-4
    with pytest.raises(ValueError, match='hyperparameter'):
        restore_logical_static_optimizer(optimizer=optimizer, scheduler=scheduler,
            placements=items, package=package, component='generator', tp_rank=2)


def test_logical_ema_preserves_source_dtype_recurrence_and_owned_subset(tmp_path):
    from distillschedule.runtime.static_logical_state import StaticLogicalEMA
    from distillschedule.runtime.static_3d_state import StaticParameterPlacement
    from distillschedule.runtime.fsdp2_static import ema_update_reference

    package, names = _package(tmp_path)
    package.manifest = dict(ema=dict(optimization_step=10))
    root = tmp_path / 'ema'
    root.mkdir()
    full = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    torch.save({'module.'+names[0]: full, 'module.'+names[1]: torch.ones(2, 3)}, root/'values.pt')
    (root/'sparse_ema_manifest.json').write_text(json.dumps(dict(schema_version='distillschedule.sparse_ema.v1',
        tracked_names=['module.'+n for n in names], shards=[dict(file='values.pt')],
        decay=.99, update_after_step=0, optimization_step=10, cur_decay_value=.5)))
    item = StaticParameterPlacement(names[0], nn.Parameter(torch.ones(2, 3)), 'output', 0)
    ema = StaticLogicalEMA(placements=(item,), package=package, tp_rank=2, device='cpu')
    expected = full[4:6].clone()
    assert ema._shadows[0].dtype == expected.dtype
    torch.testing.assert_close(ema._shadows[0], expected, rtol=0, atol=0)
    decay = ema.get_decay(11)
    ema_update_reference(expected, item.parameter.detach().to(expected.dtype), decay)
    ema.step()
    torch.testing.assert_close(ema._shadows[0], expected, rtol=0, atol=0)
    assert ema.optimization_step == 11
    assert ema.receipt()['initial_optimization_step'] == 10
    assert not torch.cuda.is_initialized()
