"""Native FSDP2 source, CPU local-shard checkpoint and module-hook contracts.

The two-rank Linear graph is an explicitly small storage/ownership test, not a
substitute for source-qualified CUDA SD1.5 execution. Actual native SD classes
are checked separately when the optional pinned source/dependencies are present.
"""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_fsdp_reference_discovery_does_not_import_numerical_providers():
    source = str(Path(__file__).resolve().parents[1] / 'src')
    command = f'''
import json, sys
sys.path.insert(0, {source!r})
from distillschedule.training.strategy_catalog import resolve
value = resolve('capability', 'official-dmd2-fsdp2-reference')
assert value['implementation'] == 'distillschedule.training.dmd2_fsdp:DMD2FSDP2Owner'
assert value['automatic_enable'] is False
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'transformers', 'peft'))
print(json.dumps(value))
'''
    result = subprocess.run([sys.executable, '-I', '-c', command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_versioned_fsdp_reference_rejects_cpu_before_assets(tmp_path):
    from distillschedule.training.sd import _arguments
    with pytest.raises(ValueError, match='FSDP2.*CUDA world16'):
        _arguments(dict(source_root=str(tmp_path), model_id=str(tmp_path), device='cpu',
            native_role_backend='fsdp2-native-sd15-v1',
            settings=dict(generator_lr=1e-6, guidance_lr=1e-6)), sdxl=False)


def test_sharding_helper_is_source_locked_and_not_a_benchmark_loop():
    from distillschedule.training.dmd2_fsdp import source_identity
    value = source_identity()
    assert value['origin_file_sha256'] == 'd6f18fd90bc7d20cd1f723751976d2e73a922fdb8de83fe7a4cdb905e9debd1d'
    assert set(value['definitions']) == {'require', 'shard_tree', 'bind_classifier_call'}
    assert value['benchmark_loop_imported'] is False


def test_fsdp_owner_disallows_clone_and_auto_before_any_tensor_allocation():
    from distillschedule.training.dmd2_fsdp import DMD2FSDP2Owner
    owner = DMD2FSDP2Owner.__new__(DMD2FSDP2Owner)
    with pytest.raises(ValueError, match='cannot be cloned'):
        deepcopy(owner)
    for mode in ('auto', 'plan'):
        with pytest.raises(ValueError, match='separate qualification'):
            owner.admit_execution(mode, {})


class _Classifier(torch.nn.Module):
    """CPU direct-call ABI fixture; mirrors native single direct-forward site."""
    def __init__(self):
        super().__init__()
        self.fake_unet = torch.nn.Linear(4, 4)

    def compute_cls_logits(self, value):
        return self.fake_unet.forward(value)


def test_classifier_direct_forward_is_scoped_to_one_owner_and_undoes_hooks():
    from distillschedule.training.dmd2_fsdp import ClassifierCallScope
    first, second = _Classifier(), _Classifier()
    calls = []
    handle = first.fake_unet.register_forward_pre_hook(lambda *args: calls.append('first'))
    plain = first.compute_cls_logits(torch.ones(1, 4))
    with ClassifierCallScope(first):
        actual = first.compute_cls_logits(torch.ones(1, 4))
        assert torch.equal(actual, plain) and calls == ['first']
        second.compute_cls_logits(torch.ones(1, 4))
        assert calls == ['first']
        with pytest.raises(ValueError, match='already'):
            ClassifierCallScope(first)
    first.compute_cls_logits(torch.ones(1, 4))
    assert calls == ['first']
    handle.remove()


def _fsdp_cpu_worker(rank, rendezvous, directory):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from distillschedule.training.dmd2_fsdp import shard_tree, source_identity, local_finite
    from distillschedule.training.dmd2_owner import _hash_tensor, _tensor_identity
    from distillschedule.training.tensor_state import pack, decode, layout
    from distillschedule.training.checkpoint import _validate_optimizer_layout
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(91 + rank)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('fsdp',))
        receipt = shard_tree(model, mesh, role='CPU_STORAGE_FIXTURE')
        assert receipt['local_numel'] < receipt['global_numel']
        assert all(isinstance(p, DTensor) for p in model.parameters())
        # Optimizers are constructed AFTER real placement; no stale Parameter refs.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sample = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10 + rank * .1
        for _ in range(2):
            model(sample).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict())
        saved = pack(state)
        assert local_finite(state)
        assert isinstance(_hash_tensor(next(model.parameters())), str)
        info = _tensor_identity(next(model.parameters()))
        assert info['local_layout'] == layout(next(model.parameters()))
        assert saved['model']['0.weight']['_tensor_kind'] == 'distillschedule.dtensor_local.v1'
        assert saved['model']['0.weight']['local'].shape != model[0].weight.shape
        path = Path(directory) / f'rank-{rank}.pt'
        torch.save(saved, path)
        restored = decode(torch.load(path, weights_only=True), [mesh])
        _validate_optimizer_layout(optimizer, restored['optimizer'], 'fixture')
        assert _hash_tensor(restored['model']['0.weight']) == _hash_tensor(model[0].weight)
        bad = deepcopy(saved)
        bad['model']['0.weight']['layout']['mesh_dim_names'] = ['wrong']
        with pytest.raises(ValueError, match='topology'):
            decode(bad, [mesh])
        bad = deepcopy(restored['optimizer'])
        next(iter(bad['state'].values()))['exp_avg'] = next(model.parameters()).float().to_local()
        with pytest.raises(ValueError, match='shape/dtype|layout'):
            _validate_optimizer_layout(optimizer, bad, 'fixture')
        with pytest.raises(ValueError, match='ALREADY_SHARDED'):
            shard_tree(model, mesh, role='again')
        (Path(directory) / f'rank-{rank}.json').write_text(json.dumps(dict(
            backend='CPU_GLOO_STORAGE_FIXTURE', source=source_identity(), receipt=receipt,
            cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


def test_real_cpu_fsdp_local_shards_and_adam_checkpoint(tmp_path):
    mp.spawn(_fsdp_cpu_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    assert all(row['cuda_initialized'] is False for row in rows)
    assert rows[0]['source'] == rows[1]['source']
