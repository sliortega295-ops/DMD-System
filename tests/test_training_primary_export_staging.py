"""SYNTHETIC_TEST_ONLY export transport; actual CPU Gloo/DTensor collectives.

No model training, pretrained capacity, GPU qualification or speed evidence.
"""
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training.adapters import ModelComponents
from distillschedule.training import inference
from distillschedule.training.checkpoint import cpu_tree, rng_state
from test_training_end_to_end import equal


class ExtraStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.bfloat16).reshape(4, 4))
        self.register_buffer('counter', torch.tensor([7], dtype=torch.int64))
        self.extra = {'z': [torch.tensor([.125, .5], dtype=torch.float64), None],
                      'a': (torch.tensor([2., 3., 4., 5.]), 'state', True)}

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state


def publish(components, output, **options):
    source = dict(reference='SYNTHETIC_TEST_ONLY:build', sha256='1' * 64)
    model = dict(factory=source['reference'], source=source, parameters={},
                 metadata=dict(evidence_kind='SYNTHETIC_TEST_ONLY'))
    return inference.publish_export(components, output, model=model,
        identity=dict(model=source, model_metadata=model['metadata']), cycle=100, **options)


def test_primary_only_single_owner_preserves_format_values_and_live_state(tmp_path):
    model = ExtraStateModel()
    components = ModelComponents(state={'student': model})
    before, before_rng = cpu_tree(model.state_dict()), rng_state(device='cpu')
    flags = [(id(p), p.requires_grad) for p in model.parameters()]
    manifest = publish(components, tmp_path / 'portable', primary_only=True)
    root, admitted = inference.read_export(tmp_path / 'portable')
    weights = torch.load(root / 'weights.pt', weights_only=True)
    assert equal(weights['student'], before) and admitted == manifest
    assert manifest['schema_version'] == inference.SCHEMA
    assert set(manifest) == {'schema_version', 'status', 'kind', 'checkpoint_cycle', 'identity',
                            'model', 'world_size', 'export', 'files', 'quality_validation'}
    assert equal(model.state_dict(), before) and equal(before_rng, rng_state(device='cpu'))
    assert model.training and flags == [(id(p), p.requires_grad) for p in model.parameters()]
    with torch.no_grad():
        model.weight.zero_()
    inference.restore_modules(components, root, admitted)
    assert equal(model.state_dict(), before)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad', [None, 0, 1, 'primary'])
def test_primary_only_rejects_nonboolean_before_materialization(tmp_path, monkeypatch, bad):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid policy reached tensor materialization')
    monkeypatch.setattr(inference, 'cpu_tree', forbidden)
    with pytest.raises(ValueError, match='primary_only.*bool'):
        publish(ModelComponents(state={'student': ExtraStateModel()}), tmp_path / 'bad', primary_only=bad)
    assert not (tmp_path / 'bad').exists()


def test_primary_only_custom_exporter_refuses_without_callback(tmp_path):
    def forbidden(*args):
        pytest.fail('custom callback cannot promise publisher-owned staging')
    components = ModelComponents(state={}, export=forbidden, load_export=forbidden)
    with pytest.raises(ValueError, match='primary.only.*named.module'):
        publish(components, tmp_path / 'bad', primary_only=True)
    assert not (tmp_path / 'bad').exists()


def _owner(rank, rendezvous, directory, scenario):
    from datetime import timedelta
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    from distillschedule.open_workload.replicated import control_plane
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=45))
    root = Path(directory)
    full_calls, staged, writes = [], [], []
    original_full, original_cpu, original_save = DTensor.full_tensor, inference.cpu_tree, torch.save
    try:
        mesh = init_device_mesh('cpu', (2,))
        model = ExtraStateModel()
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Shard(0)]))
        model.extra['z'][0] = distribute_tensor(model.extra['z'][0], mesh, [Shard(0)])
        model.extra['a'] = (distribute_tensor(model.extra['a'][0], mesh, [Shard(0)]), 'state', True)
        components = ModelComponents(state={'student': model})
        # Capture before interception; all owners genuinely restore dense values.
        before = cpu_tree(model.state_dict())
        before_rng = rng_state(device='cpu')
        flags = [(id(p), p.requires_grad) for p in model.parameters()]

        def full(value, *args, **kwargs):
            full_calls.append(list(value.shape))
            return original_full(value, *args, **kwargs)

        def stage(value):
            if scenario != 'legacy':
                assert rank == 0, 'nonprimary copied or retained full CPU weights'
            staged.append(type(value).__name__)
            if scenario == 'copy_failure' and isinstance(value, torch.Tensor):
                raise MemoryError('injected primary CPU staging failure')
            return original_cpu(value)

        def save(*args, **kwargs):
            assert rank == 0, 'nonprimary wrote portable weights'
            writes.append('weights')
            if scenario == 'disk_failure':
                raise OSError('injected portable disk failure')
            return original_save(*args, **kwargs)

        DTensor.full_tensor, inference.cpu_tree, torch.save = full, stage, save
        options = {} if scenario == 'legacy' else dict(primary_only=True)
        if scenario == 'mode_mismatch':
            options['primary_only'] = rank == 0
        message = {'copy_failure': 'primary CPU staging failure',
                   'disk_failure': 'portable disk failure', 'mode_mismatch': 'staging.*polic'}.get(scenario)
        with control_plane(dist.group.WORLD):
            if message:
                with pytest.raises(ValueError, match=message):
                    publish(components, root / 'portable', **options)
            else:
                result = publish(components, root / 'portable', **options)
                assert result['world_size'] == 2
        if scenario == 'mode_mismatch':
            assert full_calls == staged == writes == []
        elif scenario == 'copy_failure':
            assert len(full_calls) == 1, 'copy failure must stop every owner before the next tensor collective'
            assert writes == []
        else:
            assert len(full_calls) == 3, 'every owner participates in every sharded tensor collective'
        if scenario != 'legacy' and rank != 0:
            assert staged == writes == []
        if scenario == 'legacy':
            assert staged, 'default legacy all-owner CPU staging changed'
        assert equal(before_rng, rng_state(device='cpu'))
        assert model.training and flags == [(id(p), p.requires_grad) for p in model.parameters()]
        # Stop instrumentation before the separate live-state comparison.
        DTensor.full_tensor, inference.cpu_tree, torch.save = original_full, original_cpu, original_save
        assert equal(before, cpu_tree(model.state_dict()))
        assert dist.is_initialized(), 'publisher destroyed the embedding owner'
        (root / f'rank-{rank}.json').write_text(json.dumps(dict(full_calls=full_calls,
            staged=staged, writes=writes, cuda_initialized=torch.cuda.is_initialized())))
        dist.barrier()
    finally:
        DTensor.full_tensor, inference.cpu_tree, torch.save = original_full, original_cpu, original_save
        dist.destroy_process_group()


@pytest.mark.parametrize('scenario', ['success', 'copy_failure', 'disk_failure', 'mode_mismatch', 'legacy'])
def test_primary_only_real_dtensor_owners(tmp_path, scenario):
    mp.spawn(_owner, args=(str(tmp_path / 'rendezvous'), str(tmp_path), scenario), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    assert all(not row['cuda_initialized'] for row in rows)
    if scenario in {'success', 'legacy'}:
        root, manifest = inference.read_export(tmp_path / 'portable')
        weights = torch.load(root / 'weights.pt', weights_only=True)
        expected = ExtraStateModel()
        assert equal(weights['student'], expected.state_dict())
        inference.restore_modules(ModelComponents(state={'student': expected}), root, manifest)
        assert rows[0]['writes'] == ['weights'] and rows[1]['writes'] == []
    else:
        assert not (tmp_path / 'portable/_SUCCESS').exists()
        assert not (tmp_path / 'portable/export.json').exists()
