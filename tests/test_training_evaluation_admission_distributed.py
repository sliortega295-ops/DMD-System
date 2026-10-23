"""Two actual CPU/Gloo owners reject divergent mounted evaluation bytes."""
import json
import os
from pathlib import Path
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, config_path, root):
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        from distillschedule.training.config import load_training_config
        from distillschedule.training import session
        import distillschedule.data as data
        config = load_training_config(config_path)
        trained = session.train(config)
        actual_dataset = data.PreparedDataset
        alias = Path(root) / 'mounted-evaluation-manifest.json'

        def mounted_dataset(path):
            # SYNTHETIC_TEST_ONLY mount namespaces; both branches use the real
            # prepared-manifest/file decoder and independent, actual bytes.
            if str(path) == str(alias):
                return actual_dataset(Path(root) / ('input-a' if rank == 0 else 'input-b')
                                      / 'prepared/manifest.json')
            return actual_dataset(path)

        def before_consensus(*args, **kwargs):
            raise AssertionError('generation entered before evaluation identity consensus')

        data.PreparedDataset = mounted_dataset
        original_generate = session._generate_values
        session._generate_values = before_consensus
        try:
            with pytest.raises(ValueError, match='ranks disagree on evaluation input identity'):
                session.evaluate(config, trained['checkpoint'], Path(root) / 'rejected',
                                 prepared=alias)
        finally:
            data.PreparedDataset = actual_dataset
            session._generate_values = original_generate
        assert not (Path(root) / 'rejected').exists()
        (Path(root) / ('rank-' + str(rank) + '.json')).write_text(json.dumps({
            'status': 'REJECTED_BEFORE_GENERATION', 'backend': 'gloo', 'device': 'cpu',
            'gpu_execution': 'NOT_RUN', 'mount_namespace': 'SYNTHETIC_TEST_ONLY'}))
    finally:
        dist.destroy_process_group()


def test_external_evaluation_inputs_agree_by_content_before_forward(tmp_path):
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.data import prepare
    for name, count in (('input-a', 4), ('input-b', 5)):
        build_fixture(tmp_path / name, samples=count)
        prepare(tmp_path / name / 'raw/manifest.json', tmp_path / name / 'prepared')
    raw = json.loads((tmp_path / 'input-a/train.json').read_text())
    raw['data'].update(prepared=str(tmp_path / 'input-a/prepared/manifest.json'), sharding='rank_strided')
    raw['trainer'].update(cycles=1, output_dir=str(tmp_path / 'train'),
                          distributed={'backend': 'gloo', 'timeout_seconds': 60})
    config = tmp_path / 'distributed.json'; config.write_text(json.dumps(raw))
    mp.spawn(_worker, args=(str(tmp_path / 'rendezvous'), str(config), str(tmp_path)),
             nprocs=2, join=True)
    assert len(list(tmp_path.glob('rank-*.json'))) == 2
