"""CPU fault-injection around the actual checkpoint producer/consumer.

Delayed reads are injected after real writers attest their bytes. No fabricated
rank receipts or replacement checkpoint is ever accepted.
"""
from datetime import timedelta
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_training_end_to_end import equal


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


def _content_worker(rank, rendezvous, root, mode, filename):
    from distillschedule.training import checkpoint as cp
    from distillschedule.open_workload.replicated import control_plane, gather
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0', CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            world_size=2, rank=rank, timeout=timedelta(seconds=30))
    # Retain the production visibility cap: only this real group's 2s timeout
    # may bound the wait, including when DETAIL wraps its backend.
    metadata = dist.new_group(backend='gloo', timeout=timedelta(seconds=2))
    output = Path(root)/'checkpoint'
    original_open = Path.open
    observations = []
    fault_started = None
    source_shard_reads = 0

    class StreamingRead(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, 'publication loaded an entire shard in memory'
            return super().read(size)

    def observed_open(path, access='r', *args, **kwargs):
        nonlocal fault_started, source_shard_reads
        access = kwargs.pop('mode', access)
        source_fault = mode.startswith('source-')
        target = (rank == 0 and path.parent.name.startswith('.checkpoint.tmp-')
                  if source_fault or mode.startswith('peer-') else rank == 1 and path.parent == output)
        names = ('manifest.json', '_SUCCESS') if mode == 'coherent-wrong' else (filename,)
        if mode == 'normal' or not target or path.name not in names or 'r' not in access:
            return original_open(path, access, *args, **kwargs)
        if mode == 'source-shard':
            source_shard_reads += 1
            if source_shard_reads == 1:  # The writer's actual local attestation.
                return original_open(path, access, *args, **kwargs)
        with original_open(path, 'rb') as handle:
            contents = handle.read()
        assert len(contents) == path.stat().st_size
        if path.name == '_SUCCESS':
            assert len(contents) == 65
        if fault_started is None:
            fault_started = time.monotonic()
        if mode in ('delayed-missing', 'peer-delayed-missing') and len(observations) < 3:
            observations.append({'name': path.name, 'missing': True})
            raise FileNotFoundError(path)
        if 'delayed-' in mode and len(observations) >= 3:
            value = contents
        elif mode in ('delayed-zero', 'persistent-zero', 'source-zero'):
            value = b'\0' * len(contents)
        elif mode in ('partial-zero', 'delayed-partial', 'source-shard'):
            value = b'\0' + contents[1:]
        elif mode == 'delayed-truncated':
            value = contents[:len(contents)//2]
        elif mode.startswith('peer-'):
            # Stale bytes come from this rank's real previous-cycle producer,
            # not a fake accepted shard or an invented digest.
            with original_open(Path(root)/'previous'/filename, 'rb') as handle:
                value = handle.read()
            assert value != contents
        elif mode == 'coherent-wrong' and path.name == '_SUCCESS':
            with original_open(output/'manifest.json', 'rb') as handle:
                changed = handle.read().replace(b'"cycle": 1', b'"cycle": 9')
            value = (hashlib.sha256(changed).hexdigest() + '\n').encode()
        elif path.name == '_SUCCESS':
            value = b'0' * 64 + b'\n'  # A valid-looking, nonzero, wrong digest.
        else:
            value = contents.replace(b'"cycle": 1', b'"cycle": 9')
            assert value != contents
        observations.append({'name': path.name, 'stat_size': path.stat().st_size,
                             'returned_bytes': len(value), 'all_zero': not any(value),
                             'expected_sha256': hashlib.sha256(contents).hexdigest(),
                             'observed_sha256': hashlib.sha256(value).hexdigest()})
        if filename.endswith('.pt'):
            return StreamingRead(value)
        return io.BytesIO(value) if 'b' in access else io.StringIO(value.decode())

    try:
        torch.manual_seed(23 + rank)
        random.seed(23 + rank)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        state = {'model': model, 'optimizer': optimizer, 'value': torch.tensor([float(rank)])}
        error = None
        with control_plane(metadata):
            if mode.startswith('peer-'):
                cp.save_checkpoint(Path(root)/'previous', state, identity={'cpu_test': True},
                                   cycle=0, cursor=0, versions={'model': 0, 'value': 0})
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        versions = {'model': 1, 'value': 1}
        policy = {'selection': 'cpu-content-readiness'}
        progress = {'stream': {'cursor': 2}}
        expected = {'state': cp.pack_state(state), 'rng': cp.rng_state(), 'cycle': 1, 'cursor': 2,
                    'versions': versions, 'policy': policy, 'data_progress': progress}
        with control_plane(metadata), patch.object(Path, 'open', observed_open):
            try:
                cp.save_checkpoint(output, state, identity={'cpu_test': True}, cycle=1, cursor=2,
                                   versions=versions, policy=policy, data_progress=progress)
            except ValueError as exc:
                error = str(exc)
            fault_elapsed = None if fault_started is None else time.monotonic() - fault_started
            errors = gather(error)
            (Path(root)/f'content-rank{rank}.json').write_text(json.dumps({
                'rank': rank, 'mode': mode, 'filename': filename, 'errors': errors,
                'observations': observations, 'fault_elapsed': fault_elapsed,
                'CUDA_initialized': torch.cuda.is_initialized(),
                'evidence_kind': 'SYNTHETIC_TEST_ONLY_GLOO_CHECKPOINT'}))
            assert errors[0] == errors[1], 'ranks disagreed on publication outcome'
            fault_rank = 0 if mode.startswith(('source-', 'peer-')) else 1
            if mode == 'normal' or rank != fault_rank:
                assert not observations, 'read fault affected a non-target rank'
            if mode == 'normal' or 'delayed-' in mode:
                assert error is None, error
                if 'delayed-' in mode and rank == fault_rank:
                    assert len(observations) >= 4
                    assert all(row.get('missing') or row['observed_sha256'] != row['expected_sha256']
                               for row in observations[:3])
                    assert observations[-1]['observed_sha256'] == observations[-1]['expected_sha256']
                payload = cp.read_checkpoint(output, identity={'cpu_test': True})
                assert equal(payload, expected), 'saved state/Adam/RNG/progress differs from producer'
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(1)
                    state['value'].add_(1)
                    for slots in optimizer.state.values():
                        slots['exp_avg'].add_(1)
                torch.manual_seed(99)
                random.seed(99)
                restored = cp.restore_checkpoint(output, state, identity={'cpu_test': True})
                assert equal(restored, expected)
                assert equal(cp.pack_state(state), expected['state'])
                assert equal(cp.rng_state(), expected['rng'])
            else:
                assert error is not None, 'unattested content was accepted'
                if not mode.startswith('source-'):
                    # Active publication can wait for exact attested bytes,
                    # but permanently wrong content must still fail boundedly.
                    assert 'visibility timed out' in error
                    assert filename in error
                    if rank == fault_rank:
                        assert .4 <= fault_elapsed < 1.5
                        assert len(observations) >= 2
                        assert observations[-1]['expected_sha256'] in error
                        assert observations[-1]['observed_sha256'] in error
                else:
                    assert 'hash' in error
                    if rank == fault_rank:
                        assert len(observations) == 1, 'publisher source corruption was retried'
                        assert fault_elapsed < .4
                # Producer rejection must precede rename. A consumer rejection
                # leaves the committed directory intact as failure evidence.
                prepublication = mode.startswith(('source-', 'peer-'))
                assert output.exists() != prepublication
                if prepublication:
                    assert list(Path(root).glob('.checkpoint.tmp-*')), 'failed staging evidence was removed'
            assert gather(rank) == [0, 1], 'failure poisoned the short control plane'
            (Path(root)/f'content-rank{rank}.json').write_text(json.dumps({
                'rank': rank, 'mode': mode, 'filename': filename, 'errors': errors,
                'observations': observations, 'fault_elapsed': fault_elapsed,
                'full_state_and_rng_readback': error is None,
                'control_plane_usable': True, 'CUDA_initialized': torch.cuda.is_initialized(),
                'evidence_kind': 'SYNTHETIC_TEST_ONLY_GLOO_CHECKPOINT'}))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group(metadata)
        dist.destroy_process_group()


@pytest.mark.parametrize('mode,filename', [
    ('normal', '_SUCCESS'),
    ('delayed-zero', '_SUCCESS'), ('delayed-zero', 'manifest.json'),
    ('delayed-missing', '_SUCCESS'), ('delayed-missing', 'manifest.json'),
    ('delayed-partial', '_SUCCESS'), ('delayed-partial', 'manifest.json'),
    ('delayed-truncated', '_SUCCESS'), ('delayed-truncated', 'manifest.json'),
    ('peer-delayed-stale', 'rank-00001.pt'), ('peer-delayed-missing', 'rank-00001.pt'),
    ('peer-persistent-stale', 'rank-00001.pt'),
    ('persistent-zero', '_SUCCESS'), ('persistent-zero', 'manifest.json'),
    ('wrong-hash', '_SUCCESS'), ('changed-manifest', 'manifest.json'),
    ('partial-zero', '_SUCCESS'), ('coherent-wrong', 'manifest.json'),
    ('source-zero', '_SUCCESS'), ('source-zero', 'manifest.json'),
    ('source-changed', '_SUCCESS'), ('source-changed', 'manifest.json'),
    ('source-shard', 'rank-00000.pt'),
])
def test_real_gloo_publication_requires_publisher_metadata_content(tmp_path, mode, filename):
    mp.spawn(_content_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), mode, filename),
             nprocs=2, join=True)


@pytest.mark.parametrize('mode,filename', [
    ('delayed-zero', '_SUCCESS'), ('persistent-zero', '_SUCCESS'),
    ('delayed-partial', 'manifest.json'), ('coherent-wrong', 'manifest.json'),
    ('peer-delayed-stale', 'rank-00001.pt'), ('peer-persistent-stale', 'rank-00001.pt'),
])
def test_content_visibility_uses_short_metadata_timeout_with_detail(tmp_path, mode, filename, monkeypatch):
    monkeypatch.setenv('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    mp.spawn(_content_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), mode, filename),
             nprocs=2, join=True)


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


@pytest.mark.parametrize('filename', ['manifest.json', '_SUCCESS'])
@pytest.mark.parametrize('fault', ['zero', 'partial', 'truncated'])
def test_single_process_publication_never_retries_wrong_content(tmp_path, monkeypatch, filename, fault):
    from distillschedule.training import checkpoint as cp
    output = tmp_path/'checkpoint'
    original_open = Path.open

    def wrong_publication(path, access='r', *args, **kwargs):
        access = kwargs.pop('mode', access)
        if path != output/filename or 'r' not in access:
            return original_open(path, access, *args, **kwargs)
        with original_open(path, 'rb') as handle:
            contents = handle.read()
        value = (b'\0' * len(contents) if fault == 'zero' else b'\0' + contents[1:]
                 if fault == 'partial' else contents[:len(contents)//2])
        return io.BytesIO(value)

    monkeypatch.setattr(Path, 'open', wrong_publication)
    monkeypatch.setattr(cp.time, 'sleep', lambda _: pytest.fail('single-process publication retried'))
    with pytest.raises(ValueError, match='visibility timed out'):
        cp.save_checkpoint(output, {'value': torch.tensor([1.0])}, identity={'test': True},
                           cycle=1, cursor=1, versions={'value': 1})
    assert output.is_dir(), 'do not rewrite committed failure evidence'


@pytest.mark.parametrize('fault,match', [
    ('marker-zero', 'hash changed'), ('manifest-zero', 'hash changed'),
    ('marker-hash', 'hash changed'), ('source', 'identity changed'),
    ('topology', 'topology changed'), ('runtime', 'runtime version changed'),
    ('missing-marker', 'not complete'), ('missing-manifest', 'not complete'),
    ('missing-shard', 'shard missing'), ('corrupt-shard', 'shard missing'),
])
def test_restart_rejections_never_use_publication_visibility_wait(tmp_path, monkeypatch, fault, match):
    from distillschedule.training import checkpoint as cp
    output = tmp_path/'checkpoint'
    identity = {'source': 'original'}
    cp.save_checkpoint(output, {'value': torch.tensor([1.0])}, identity=identity,
                       cycle=1, cursor=1, versions={'value': 1})
    marker, manifest = output/'_SUCCESS', output/'manifest.json'
    if fault in ('marker-zero', 'manifest-zero'):
        target = marker if fault == 'marker-zero' else manifest
        target.write_bytes(b'\0' * target.stat().st_size)
    elif fault == 'marker-hash':
        marker.write_text('0' * 64 + '\n')
    elif fault == 'source':
        identity = {'source': 'changed'}
    elif fault in ('topology', 'runtime'):
        document = json.loads(manifest.read_text())
        if fault == 'topology':
            document['topology'] = {'world_size': 2}
        else:
            document['torch'] = 'changed-runtime'
        manifest.write_text(json.dumps(document))
        marker.write_text(cp._sha(manifest) + '\n')
    elif fault.startswith('missing-'):
        {'missing-marker': marker, 'missing-manifest': manifest,
         'missing-shard': output/'state.pt'}[fault].unlink()
    else:
        (output/'state.pt').write_bytes(b'corrupt shard')
    monkeypatch.setattr(cp.time, 'sleep', lambda _: pytest.fail('restart retried invalid content'))
    if fault != 'corrupt-shard':
        with pytest.raises(ValueError, match=match):
            cp.read_manifest(output, identity=identity)
    with pytest.raises(ValueError, match=match):
        cp.read_checkpoint(output, identity=identity)
    state = {'value': torch.tensor([9.0])}
    before_state, before_rng = cp.pack_state(state), cp.rng_state()
    with pytest.raises(ValueError, match=match):
        cp.restore_checkpoint(output, state, identity=identity)
    assert equal(cp.pack_state(state), before_state)
    assert equal(cp.rng_state(), before_rng)
