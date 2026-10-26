"""Actual native UNets, role DDP and the public rank checkpoint lifecycle.

FP32 CPU/Gloo functionality, NOT the measured mixed-precision GPU portfolio.
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training.config import TrainingConfig, load_training_config
from test_training_dmd2_cpu import source_root, REPO
from test_training_end_to_end import equal


def _worker(rank, rendezvous, config_path, output, cycles, resume):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=180))
    try:
        from distillschedule.training.session import train
        config = load_training_config(config_path)
        raw = config.to_dict()
        raw['trainer'].update(cycles=cycles, output_dir=output,
                              distributed=dict(backend='gloo', timeout_seconds=180))
        if resume:
            raw['trainer']['resume_from'] = resume
        result = train(TrainingConfig.from_dict(raw, source=config.source))
        Path(output, f'worker-{rank}.json').write_text(json.dumps(result))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def _phase(root, name, config, cycles, resume=None):
    out = root / name
    mp.spawn(_worker, args=(str(root / (name + '.rdzv')), str(config), str(out), cycles, resume),
             nprocs=2, join=True)
    rows = [json.loads((out / f'worker-{rank}.json').read_text()) for rank in range(2)]
    assert all(row['completed_cycles'] == cycles and row['finite'] for row in rows)
    return out, rows[0]['checkpoint']


@pytest.mark.parametrize('variant', ['sd15', 'sdxl'])
def test_actual_distributed_continuation_is_exact_across_fresh_processes(tmp_path, source_root, variant):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'assets', source_root, variant=variant, cycles=3, ratio=2)
    path = Path(files['config'])
    value = json.loads(path.read_text())
    value['model']['parameters']['native_role_backend'] = 'ddp-fp32-v1'
    value['data']['sharding'] = 'rank_strided'
    path.write_text(json.dumps(value))
    _, first = _phase(tmp_path, 'first', path, 1)
    resumed_dir, resumed = _phase(tmp_path, 'resumed', path, 3, first)
    full_dir, full = _phase(tmp_path, 'full', path, 3)
    owners = []
    for rank in range(2):
        a = torch.load(Path(resumed) / f'rank-{rank:05d}.pt', weights_only=True)
        b = torch.load(Path(full) / f'rank-{rank:05d}.pt', weights_only=True)
        assert equal(a, b), 'all state, RNG, policy and input progress must survive restart'
        owner = b['state']['native_cycle']['value']
        assert owner['native_step'] == owner['guidance_updates'] == 6
        assert owner['generator_updates'] == 3
        assert all(row['last_epoch'] == 12 for row in owner['schedulers'].values())
        assert not any('.module.' in name for name in owner['model'])
        assert all(float(state['step']) == (3 if role == 'generator' else 6)
                   for role, opt in owner['optimizers'].items() for state in opt['state'].values())
        owners.append(owner)
        metrics = [json.loads(line) for line in (resumed_dir / f'ranks/rank-{rank:05d}/metrics.jsonl').read_text().splitlines()]
        assert [row['cycle'] for row in metrics] == [2, 3]
        for row in metrics:
            actual = row['execution']['effects']['native_collective_owner']
            assert actual['preparations'] == 1
            assert actual['runtime']['zero_gradients']
            assert actual['runtime']['roles']['generator']['forward_calls'] > 0
            assert actual['runtime']['roles']['guidance']['forward_calls'] > 0
    assert equal(owners[0]['model'], owners[1]['model']), 'real DDP must synchronize all trained parameters'
    assert not equal(owners[0]['rng'], owners[1]['rng']), 'rank-local native streams must remain independent'
    before = torch.load(Path(first) / 'rank-00000.pt', weights_only=True)['state']['native_cycle']['value']
    assert all(equal(tensor, owners[0]['model'][name]) for name, tensor in before['model'].items()
               if name.startswith(('guidance_model.real_unet.', 'text_encoder.', 'vae.')))


def test_role_backend_requires_live_group_before_loading_assets(tmp_path):
    from distillschedule.training.sd import _arguments
    with pytest.raises(ValueError, match='initialized.*group'):
        _arguments(dict(source_root=str(tmp_path), model_id=str(tmp_path),
            native_role_backend='ddp-fp32-v1', settings=dict(generator_lr=1e-4, guidance_lr=2e-4)), sdxl=False)
