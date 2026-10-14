"""Focused CPU gates; these do not claim FSDP GPU/capacity success."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.pp_fsdp_checkpoint import _StagePackageView
from distillschedule.runtime.pp_fsdp_native import PPFSDPTrainableState
from scripts.distillschedule.probe_wan13b_pp_fsdp_world16 import compiled_main


def test_native_main_placement_binding_compiles_without_cuda():
    assert callable(compiled_main())
    assert not torch.cuda.is_initialized()


def test_stage_package_view_does_not_modify_payload_or_source():
    spec = dict(parameter_names=['prefix', 'blocks.0', 'blocks.1', 'postfix'],
                records={'prefix': {'offset_elements': 0}}, payloads={'first_moment': {'path': 'immutable'}})
    package = SimpleNamespace(root=Path('/unused'), component=lambda c: spec)
    view = _StagePackageView(package, 'generator', ['blocks.1', 'postfix'])
    result = view.component('generator')
    assert result['parameter_names'] == ['blocks.1', 'postfix']
    assert spec['parameter_names'] == ['prefix', 'blocks.0', 'blocks.1', 'postfix']
    assert result['payloads'] is spec['payloads']
    with pytest.raises(ValueError):
        _StagePackageView(package, 'generator', ['unknown'])
    with pytest.raises(ValueError):
        view.component('critic')


def test_one_mean2_loss_and_one_commit_ledger():
    state = PPFSDPTrainableState.__new__(PPFSDPTrainableState)
    state.accumulations = 0
    loss = torch.tensor(2.0, requires_grad=True)
    assert state.mean_microbatch_group_loss(loss, group_size=2) is loss
    assert state.accumulations == 2
    with pytest.raises(ValueError):
        state.mean_microbatch_group_loss(loss, group_size=2)
    state.accumulations = 0
    with pytest.raises(ValueError):
        state.mean_microbatch_group_loss(loss, group_size=8)


def test_placement_adapter_never_installs_tp_or_new_loss():
    from scripts.distillschedule import probe_wan13b_pp_fsdp_world16 as entry
    source = inspect.getsource(entry.build_role)
    assert 'bind_wan_helios_tp4' not in source
    assert 'shard_native_pp_stage' in source
    assert 'native.configure_fsdp2_checkpointing' in source
