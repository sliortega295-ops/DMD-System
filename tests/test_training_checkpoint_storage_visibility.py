"""CPU fault-injection around the actual checkpoint producer/consumer.

Delayed metadata visibility is injected, never data or fabricated rank receipts.
"""
from datetime import timedelta
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, root, mode):
    from distillschedule.training import checkpoint as cp
    debug = mode.endswith('-debug')
    mode = mode.removesuffix('-debug')
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0', CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            world_size=2, rank=rank, timeout=timedelta(seconds=30))
    metadata = dist.new_group(backend='gloo', timeout=timedelta(seconds=2)) if mode == 'short' else None
    if debug:
        from torch.distributed.distributed_c10d import _ProcessGroupWrapper
        assert isinstance((metadata if metadata is not None else dist.group.WORLD)
                          ._get_backend(torch.device('cpu')), _ProcessGroupWrapper)
    output = Path(root)/'checkpoint'
    counts = {'staging': 0, 'shard': 0, 'published': 0}
    is_dir, is_file, digest = Path.is_dir, Path.is_file, cp._sha
    def visible_dir(path):
        if rank == 1 and path.name.startswith('.checkpoint.tmp-'):
            counts['staging'] += 1
            if mode in ('nonshared', 'short') or counts['staging'] <= 3:
                return False
        return is_dir(path)
    def visible_file(path):
        if rank == 0 and path.parent.name.startswith('.checkpoint.tmp-') and path.name == 'rank-00001.pt':
            counts['shard'] += 1
            if counts['shard'] <= 3:
                return False
        if path.parent == output and path.name == '_SUCCESS':
            counts['published'] += 1
            if counts['published'] <= 3:
                return False
        return is_file(path)
    def changed_digest(path):
        if mode == 'corrupted' and rank == 1 and Path(path).name == 'rank-00001.pt':
            return '0'*64
        return digest(path)
    try:
        state = {'value': torch.tensor([float(rank)])}
        expected = None if mode == 'delayed' else ('shared' if mode in ('nonshared', 'short') else 'hash|incomplete')
        with ExitStack() as stack:
            if metadata is not None:
                from distillschedule.open_workload.replicated import control_plane
                stack.enter_context(control_plane(metadata))
            stack.enter_context(patch.object(Path, 'is_dir', visible_dir))
            stack.enter_context(patch.object(Path, 'is_file', visible_file))
            stack.enter_context(patch.object(cp, '_sha', changed_digest))
            # The short control-plane test retains the real production 30s cap.
            if mode != 'short':
                stack.enter_context(patch.object(cp, '_STORAGE_VISIBILITY_TIMEOUT', .5, create=True))
            if expected:
                with pytest.raises(ValueError, match=expected):
                    cp.save_checkpoint(output, state, identity={'cpu_test': True}, cycle=1, cursor=2,
                                       versions={'value': 1})
            else:
                cp.save_checkpoint(output, state, identity={'cpu_test': True}, cycle=1, cursor=2,
                                   versions={'value': 1})
                assert counts['published'] >= 4, 'save returned before local publication visibility'
                payload = cp.read_checkpoint(output, identity={'cpu_test': True})
                assert torch.equal(payload['state']['value']['value'], state['value'])
                assert counts['staging'] >= 4 if rank == 1 else counts['shard'] >= 4
            if metadata is not None:
                # Failure must leave the short control plane usable, not timed out.
                from distillschedule.open_workload.replicated import gather
                assert gather(rank) == [0, 1]
        dist.barrier()
        if expected:
            assert not output.exists()
            assert not list(Path(root).glob('.checkpoint.tmp-*/_SUCCESS'))
        assert not torch.cuda.is_initialized()
        (Path(root)/f'rank{rank}.json').write_text(json.dumps(dict(mode=mode, counts=counts,
            evidence_kind='SYNTHETIC_TEST_ONLY_GLOO_CHECKPOINT', CUDA_initialized=False)))
    finally:
        if metadata is not None:
            dist.destroy_process_group(metadata)
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['delayed', 'nonshared', 'corrupted', 'short', 'delayed-debug', 'short-debug'])
def test_real_gloo_checkpoint_handles_visibility_without_weakening_rejections(tmp_path, mode, monkeypatch):
    if mode.endswith('-debug'):
        monkeypatch.setenv('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    mp.spawn(_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), mode), nprocs=2, join=True)


def test_manifest_and_success_are_flushed_before_atomic_publication(tmp_path, monkeypatch):
    from distillschedule.training import checkpoint as cp
    synced = set()
    fsync, rename = os.fsync, os.rename

    def track_fsync(fd):
        result = fsync(fd)
        stat = os.fstat(fd)
        synced.add((stat.st_dev, stat.st_ino))
        return result

    def publish(source, destination):
        for name in ('state.pt', 'manifest.json', '_SUCCESS'):
            stat = (Path(source)/name).stat()
            assert (stat.st_dev, stat.st_ino) in synced, name + ' not durable before publication'
        assert (Path(source)/'_SUCCESS').read_text().strip() == cp._sha(Path(source)/'manifest.json')
        return rename(source, destination)

    monkeypatch.setattr(os, 'fsync', track_fsync)
    monkeypatch.setattr(os, 'rename', publish)
    state = {'value': torch.tensor([1.0])}
    cp.save_checkpoint(tmp_path/'checkpoint', state, identity={'test': True}, cycle=1,
                       cursor=1, versions={'value': 1})
    assert torch.equal(cp.read_checkpoint(tmp_path/'checkpoint', identity={'test': True})
                       ['state']['value']['value'], state['value'])


@pytest.mark.parametrize('filename', ['manifest.json', '_SUCCESS'])
def test_metadata_sync_failure_never_publishes_checkpoint(tmp_path, monkeypatch, filename):
    from distillschedule.training import checkpoint as cp
    fsync = os.fsync

    def reject_metadata_sync(fd):
        stat = os.fstat(fd)
        for path in tmp_path.glob('.checkpoint.tmp-*/' + filename):
            target = path.stat()
            if (target.st_dev, target.st_ino) == (stat.st_dev, stat.st_ino):
                raise OSError('metadata persistence failed')
        return fsync(fd)

    monkeypatch.setattr(os, 'fsync', reject_metadata_sync)
    with pytest.raises(OSError, match='metadata persistence failed'):
        cp.save_checkpoint(tmp_path/'checkpoint', {'value': torch.tensor([1.0])},
                           identity={'test': True}, cycle=1, cursor=1, versions={'value': 1})
    assert not (tmp_path/'checkpoint').exists()
    assert not list(tmp_path.glob('.checkpoint.tmp-*'))
