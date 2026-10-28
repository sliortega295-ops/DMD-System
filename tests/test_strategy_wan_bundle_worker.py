"""SYNTHETIC_TEST_ONLY: real worker restart wiring, not Wan math/world16 proof.

Scalar CPU Adam/autograd comes from native_cpu_bundle. CUDA transport/capture,
model assets, peer ranks and CUDA-only kernel metadata are synthetic. Worker,
session, dispatcher, cohort/cursor, control-plane admission, checkpoint IO,
native state adapters and receipt validators are the actual production owners.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist

from distillschedule.runtime.wan_bundle_lifecycle import WanBundleLifecycle
from distillschedule.search.wan_candidate_bundle import ASSETS
from distillschedule.strategies import wan_worker
from distillschedule.training.checkpoint import pack_state, rng_state
from distillschedule.training.native_state import engine_state
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from test_refactor_wan_incumbent_bundle import _executed_receipts
from test_strategy_wan_bundle_stream import native_cpu_bundle  # noqa: F401
from test_training_end_to_end import equal


def _state(parts):
    return pack_state(engine_state({'generator': parts.g, 'critic': parts.c}, ema=parts.ema))


def _owners(parts):
    return tuple(id(value) for engine in (parts.g, parts.c)
                 for value in (engine, engine.optimizer, engine.lr_scheduler, *engine.module.parameters())) + (
                     id(parts.ema), *(id(value) for value in parts.ema._shadows))


def _native_final_writers():
    # Execute the two unchanged original writer functions, without importing
    # the heavyweight model loader at module scope (the same AST seam used by
    # other native handler tests). Neither writer is replaced with a test fake.
    path = Path(__file__).resolve().parents[1] / 'scripts/distillschedule/probe_wan13b_upper_bound_world16.py'
    names = {'_save_trainable_evidence', '_save_optimizer_evidence'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    namespace = {'torch': torch, 'file_sha256': file_sha256}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def _write_stream(f, root):
    """Synthetic captured assets, read by the real load_stream validator."""
    from distillschedule.strategies.wan_inputs import WanInputSources
    from distillschedule.strategies.wan_stream import SCHEMA, WanCohortSources
    if (root / 'stream.json').exists():
        # A fresh interpreter reopens the exact same immutable fixture stream.
        f.stream = json.loads((root / 'stream.json').read_text())
        return root / 'stream.json'
    root.mkdir()
    sources = WanCohortSources(tuple(WanInputSources(
        tuple(str(root / f'rank-{rank}.pt') for rank in range(16)), cohort=index)
        for index in range(3)))
    identity = dict(f.stream['identity'], task=f.task.to_dict(), sources=sources.to_dict(),
                    inventory=[{'evidence_kind': 'SYNTHETIC_TEST_ONLY', 'cohort': i} for i in range(3)],
                    training_config_sha256='d'*64)
    entries = []
    for index, entry in enumerate(f.stream['cohorts']):
        cohort = root / f'cohort-{index:04d}'
        cohort.mkdir()
        capture = dict(task=entry['task'], expected_trace_hash=entry['trace_hash'],
                       input_inventory=identity['inventory'][index],
                       source_fingerprint=identity['source_fingerprint'],
                       training_config_sha256=identity['training_config_sha256'])
        (cohort / 'capture.json').write_text(json.dumps(capture))
        entries.append(dict(entry, index=index, trace_bundle=str(cohort),
                            capture_sha256=file_sha256(cohort / 'capture.json')))
    document = dict(schema_version=SCHEMA, task=f.task.to_dict(), identity=identity,
                    cohorts=entries, status='READY_FRESH_COHORT_STREAM',
                    evidence_kind='SYNTHETIC_TEST_ONLY', gpu_execution='NOT_RUN')
    document['stream_sha256'] = fingerprint(document)
    f.stream = document
    path = root / 'stream.json'
    path.write_text(json.dumps(document))
    return path


class _CpuWorld16Transport:
    """Fake only torch.distributed transport, never checkpoint/control owners.

    Nonzero peer files copy the actual writer's rank-0 payload. This exercises
    all-shard publication/read admission, NOT distributed state/RNG correctness.
    """
    def __init__(self):
        self.initialized = False
        self.group = NS()
        self.events = []
        self.output = None

    def init_process_group(self, backend, **kwargs):
        assert backend == 'nccl' and not self.initialized
        self.initialized = True
        self.events.append('init_nccl')

    def new_group(self, *, backend, **kwargs):
        assert backend == 'gloo' and self.initialized
        # Match the existing metadata-timeout reader's real transport seam.
        self.group._get_backend = lambda device: NS(options=NS(_timeout=kwargs['timeout']))
        self.events.append('new_gloo')
        return self.group

    def destroy_process_group(self, group=None):
        assert self.initialized
        if group is self.group:
            self.events.append('destroy_gloo')
        else:
            assert group is None
            self.events.append('destroy_nccl')
            self.initialized = False

    def all_gather_object(self, rows, value, *, group=None):
        assert self.initialized and group is self.group and len(rows) == 16
        self.events.append('gloo_gather')
        if isinstance(value, dict) and set(value) == {'rank', 'host', 'launcher'}:
            peers = [dict(rank=rank, host=f'SYNTHETIC_TEST_ONLY-node-{rank//8}',
                          launcher=dict(node=rank//8, local_rank=rank%8, local_size=8))
                     for rank in range(16)]
        elif isinstance(value, dict) and set(value) == {'rank-00000.pt'}:
            # The production writer has already flushed and hashed rank zero.
            shards = list((self.output / 'checkpoints').glob('.cycle-*.tmp-*/rank-00000.pt'))
            assert len(shards) == 1 and file_sha256(shards[0]) == value['rank-00000.pt']
            peers = [value]
            for rank in range(1, 16):
                peer = shards[0].with_name(f'rank-{rank:05d}.pt')
                shutil.copyfile(shards[0], peer)
                peers.append({peer.name: file_sha256(peer)})
        else:
            peers = [deepcopy(value) for _ in rows]
        rows[:] = peers

    def barrier(self):
        assert self.initialized
        self.events.append('barrier')


@pytest.fixture
def cpu_worker(native_cpu_bundle, monkeypatch, tmp_path):
    f = native_cpu_bundle
    stream = _write_stream(f, tmp_path / 'stream')
    (tmp_path / 'SYNTHETIC_TEST_ONLY.json').write_text(json.dumps({
        'scope': 'scalar CPU worker restart with fake CUDA/ranks/assets/kernel metadata',
        'gpu_execution': 'NOT_RUN', 'wan_math': 'NOT_RUN', 'world16_execution': 'NOT_RUN'}))
    for key, value in dict(WORLD_SIZE=16, LOCAL_WORLD_SIZE=8, RANK=0, LOCAL_RANK=0, GROUP_RANK=0).items():
        monkeypatch.setenv(key, str(value))
    transport = _CpuWorld16Transport()
    for name in ('init_process_group', 'new_group', 'destroy_process_group', 'all_gather_object', 'barrier'):
        monkeypatch.setattr(dist, name, getattr(transport, name))
    monkeypatch.setattr(dist, 'is_initialized', lambda: transport.initialized)
    monkeypatch.setattr(dist, 'get_rank', lambda group=None: 0)
    monkeypatch.setattr(dist, 'get_world_size', lambda group=None: 16)
    monkeypatch.setattr(dist, 'get_backend', lambda group=None: 'gloo' if group is transport.group else 'nccl')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    for name in ('set_device', 'manual_seed_all', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: NS(
        name='SYNTHETIC_TEST_ONLY_CPU', uuid='SYNTHETIC_TEST_ONLY', total_memory=1, major=0, minor=0))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1, 1))
    for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda device: 0)
    # The real precision setup mutates these flags; keep other tests isolated.
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', torch.backends.cudnn.allow_tf32)

    model_files = [dict(path='transformer/config.json', bytes=1, sha256='e'*64)]
    monkeypatch.setitem(ASSETS, 'model_identity_sha256', fingerprint(model_files))
    manifest = dict(lora_rank=256, model_files=model_files, model_identity_sha256=ASSETS['model_identity_sha256'],
        components={role: dict(optimizer_step=0, scheduler_step=0, seed=123) for role in ('generator', 'critic')},
        ema=dict(optimization_step=0, decay=.999, update_after_step=0))
    bootstrap = NS(manifest=manifest, manifest_hash='c'*64, component=lambda role: manifest['components'][role])
    identity = dict(ASSETS, bootstrap_hash=bootstrap.manifest_hash,
        trace_bundle_hash=f.stream['cohorts'][0]['trace_hash'], trace_binding_hash='b'*64,
        model_files={'config.json': 'e'*64})
    asset_dirs = [tmp_path / name for name in ('model', 'bootstrap', 'trace', 'negative')]
    for directory in asset_dirs:
        directory.mkdir(exist_ok=True)
    manifest['model_root'] = str(asset_dirs[0])
    assets = wan_worker.OriginalWanAssets(str(asset_dirs[0]), str(asset_dirs[1]), bootstrap.manifest_hash,
        str(asset_dirs[2]), identity['trace_bundle_hash'], str(asset_dirs[3]))
    # Physical bootstrap IO is synthetic in both the public lowerer and worker.
    # Source identity, stream checks, protocol lowering and execution stay real.
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    monkeypatch.setattr(Wan13BBootstrap, 'load', lambda *args, **kwargs: bootstrap)

    def load_assets(task, spec, *, rank):
        from distillschedule.strategies.wan_trace import load_original_trace
        assert task == f.task and spec == assets and rank == 0
        trace = load_original_trace(task, f.stream['cohorts'][0]['trace_bundle'],
                                    expected_hash=identity['trace_bundle_hash'])
        negative = NS(embedding=NS(to=lambda **kwargs: torch.zeros(1)))
        return bootstrap, trace, negative, identity

    parts = []
    def load_backend(root):
        # Importing the backend must not construct engines before the worker
        # admits the resume header. The actual session calls this builder.
        fair = NS(base=NS())
        def build_engines(**kwargs):
            current = f.build(backend_only=True)
            # Fixed CUDA-only fields are metadata fixtures, not evidence that
            # norm/epilogue/reducer kernels ran. Graph/cast/lifecycle counts
            # remain the real controllers' live dictionaries.
            metadata_fixture = _executed_receipts()
            for role in ('generator', 'critic'):
                actual = current.build[role]['metadata']
                current.build[role].update(deepcopy(metadata_fixture[role]))
                current.build[role]['metadata'].update(actual)
                current.build[role].update(
                    engine=dict(gradient_sync='WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE'),
                    placement=dict(param_compute_dtype_policy='PRESERVE_ORIGINAL_PER_PARAMETER',
                        trainable_parameter_dtypes=['torch.float32'], frozen_parameter_dtypes=['torch.bfloat16']))
                current.build[role]['metadata'].update(
                    evidence_kind='SYNTHETIC_TEST_ONLY', modulation_chunk_runtime=dict(effective_tokens=[2048]),
                    generator_nograd_lora_fused_epilogue=dict(requested=True, enabled=role == 'generator',
                                                            fused_invocation_count=1, fused_element_count=1))
            fair.base.p0 = current.fair.base.p0
            current.owners = _owners(current)
            parts.append(current)
            return current.g, current.c, current.ema, current.build
        fair.base._build_engines = build_engines
        for name, writer in _native_final_writers().items():
            setattr(fair.base, name, writer)
        return fair

    monkeypatch.setattr(wan_worker, 'load_assets', load_assets)
    monkeypatch.setattr(wan_worker, 'load_original_backend', load_backend)
    begins, boundaries = [], []
    begin = WanBundleLifecycle.begin
    boundary = WanBundleLifecycle.validate_checkpoint_boundary
    fault = NS(pending_communication=False)

    def observe_begin(self, **kwargs):
        begins.append(dict(index=kwargs['index'], binding=getattr(self.session, '_stream_binding', None),
            versions=self.session.versions(), state=_state(self.session), rng=rng_state(),
            cold=not self.rollout._cache and not self.critic.cache, owners=_owners(self.session)))
        return begin(self, **kwargs)

    def observe_boundary(self):
        boundaries.append(self.session.g.global_steps)
        if fault.pending_communication:
            self.session.c._buckets_by_index = [NS(work=NS(is_completed=lambda: False))]
        return boundary(self)

    monkeypatch.setattr(WanBundleLifecycle, 'begin', observe_begin)
    monkeypatch.setattr(WanBundleLifecycle, 'validate_checkpoint_boundary', observe_boundary)

    def configuration(name, *, every=1, resume=None, stop=None):
        from distillschedule.training.config import TrainingConfig
        (tmp_path / 'task.json').write_text(json.dumps(f.task.to_dict()))
        (tmp_path / 'assets.json').write_text(json.dumps(assets.__dict__))
        return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
            model=dict(workload='W1'), algorithm={},
            data=dict(input_kind='original_wan_cohort_stream.v1', prepared=str(stream)),
            execution=dict(adapter='native', mode='manual', preset='wan_ar.incumbent21a_stream.v2',
                           task=str(tmp_path / 'task.json'), assets=str(tmp_path / 'assets.json')),
            trainer=dict(cycles=3, output_dir=str(tmp_path / name), checkpoint_every=every,
                         **(dict(resume_from=str(resume)) if resume else {}),
                         **(dict(stop_after_cycle=stop) if stop is not None else {}))),
            source=tmp_path / 'training.json')

    def run(name, *, every=1, resume=None, stop=None, public=False):
        transport.output = tmp_path / name
        event_start = len(transport.events)
        try:
            if public:
                from distillschedule.training.native import run_native
                return run_native(configuration(name, every=every, resume=resume, stop=stop))
            return wan_worker.run_worker(f.task, assets, output=transport.output, mode='train',
                strategy=f.strategy, cycles=3, stream_path=stream, checkpoint_every=every, resume_from=resume,
                **(dict(stop_after_cycle=stop) if stop is not None else {}))
        finally:
            assert not transport.initialized
            if len(transport.events) > event_start:
                assert transport.events[event_start:][-2:] == ['destroy_gloo', 'destroy_nccl']

    return NS(run=run, f=f, parts=parts, begins=begins, boundaries=boundaries,
              transport=transport, fault=fault, root=tmp_path, configuration=configuration, assets=assets)


@pytest.mark.parametrize('checkpoint_cycle', [1, 2], ids=['two_remaining', 'one_remaining'])
def test_worker_checkpoint_resume_matches_uninterrupted(cpu_worker, checkpoint_cycle):
    h = cpu_worker
    full = h.run('uninterrupted', every=checkpoint_cycle)
    reference, = h.parts
    expected, expected_rng = _state(reference), rng_state()
    checkpoint = h.root / 'uninterrupted/checkpoints' / f'cycle-{checkpoint_cycle:06d}'
    payload = torch.load(checkpoint / 'rank-00000.pt', weights_only=True)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    assert (checkpoint / '_SUCCESS').read_text().strip() == file_sha256(checkpoint / 'manifest.json')
    assert manifest['cycle'] == manifest['cursor'] == checkpoint_cycle
    assert set(manifest['files']) == {f'rank-{rank:05d}.pt' for rank in range(16)}
    assert set(payload['state']) == {*(f'{role}.{kind}' for role in ('generator', 'critic')
                                     for kind in ('weights', 'optimizer', 'clocks', 'scheduler')), 'ema'}
    assert h.boundaries == ([1, 2, 3] if checkpoint_cycle == 1 else [2, 3])
    assert sorted(path.name for path in checkpoint.parent.iterdir()) == (
        ['cycle-000001', 'cycle-000002', 'cycle-000003'] if checkpoint_cycle == 1 else
        ['cycle-000002', 'cycle-000003'])
    assert reference.owners == _owners(reference)
    begin_offset, boundary_offset = len(h.begins), len(h.boundaries)

    resumed = h.run('resumed', every=checkpoint_cycle, resume=checkpoint)
    restarted = h.parts[-1]
    first = h.begins[begin_offset]
    assert first['index'] == checkpoint_cycle and first['cold']
    assert first['binding'] == (h.f.stream['stream_sha256'], checkpoint_cycle - 1)
    assert first['versions'] == payload['versions']
    assert equal(first['state'], payload['state']) and equal(first['rng'], payload['rng'])
    assert restarted.owners == _owners(restarted) == first['owners']
    assert equal(expected, _state(restarted)) and equal(expected_rng, rng_state())
    assert [step for role, step in restarted.seen if role == 'g'] == list(range(5*checkpoint_cycle, 15, 5))
    assert [step for role, step in restarted.seen if role == 'c'] == list(range(5*checkpoint_cycle, 15))
    assert h.boundaries[boundary_offset:] == list(range(checkpoint_cycle + 1, 4))
    assert resumed['initial_versions'] == payload['versions']
    assert resumed['final_versions'] == full['final_versions'] == dict(generator=3, critic=15, ema=15)
    assert [row['cycle'] for row in resumed['cycles']] == list(range(checkpoint_cycle + 1, 4))
    for actual, uninterrupted in zip(resumed['cycles'], full['cycles'][checkpoint_cycle:], strict=True):
        for key in ('cycle', 'versions', 'cohort', 'bundle_lifecycle', 'call_census', 'prepared', 'rollouts', 'generator', 'critic'):
            assert actual[key] == uninterrupted[key], key
        assert actual['finite'] and actual['checkpoint_outside_compute_ns'] >= 0
    restart = resumed['training_restart']
    assert restart['resume_cycle'] == checkpoint_cycle and restart['resume_from'] == str(checkpoint)
    for key in ('checkpoint_identity', 'checkpoint_policy'):
        assert restart[key] == full['training_restart'][key]
    final_checkpoint = Path(restart['latest_checkpoint'])
    final = torch.load(final_checkpoint / 'rank-00000.pt', weights_only=True)
    assert final['cycle'] == final['cursor'] == 3 and final['versions'] == full['final_versions']
    assert equal(final['state'], expected) and equal(final['rng'], expected_rng)
    for role, first_version in (('generator', checkpoint_cycle + 1), ('critic', checkpoint_cycle*5)):
        mechanism = 'generator_rollout_cuda_graph' if role == 'generator' else 'critic_grad_cuda_graph'
        assert resumed['build'][role]['metadata'][mechanism]['stream_lifecycle']['first_version'] == first_version
    from safetensors.torch import load_file
    for key, value in resumed['state_evidence'].items():
        original = full['state_evidence'][key]
        assert file_sha256(Path(value['path'])) == value['sha256']
        assert file_sha256(Path(original['path'])) == original['sha256']
        # Safetensors metadata map ordering need not give byte-identical files.
        assert equal(load_file(value['path']), load_file(original['path']))
    assert resumed['backend_effects']['candidate_bundle_effects']['actual_compute_observed']
    assert resumed['status'] == 'COMPLETED_ORIGINAL_WAN_WORK' and not resumed['formal_speedup_eligible']
    # Native CUDA status strings here only demonstrate the worker code path.
    assert resumed['hardware']['gpu_name'] == 'SYNTHETIC_TEST_ONLY_CPU'
    assert json.loads((h.root / 'resumed/rank_00.json').read_text()) == json.loads(json.dumps(resumed))


@pytest.mark.parametrize('fault, message', [
    ('identity', 'identity changed'), ('policy', 'policy/progress'), ('cursor', 'policy/progress'),
    ('completed', 'policy/progress'), ('incomplete', 'checkpoint is not complete'),
    ('shard', 'shard missing or hash changed')])
def test_worker_rejects_invalid_resume_before_updates(cpu_worker, fault, message):
    h = cpu_worker
    h.run('source')
    checkpoint = h.root / 'source/checkpoints/cycle-000001'
    if fault == 'completed':
        checkpoint = h.root / 'source/checkpoints/cycle-000003'
    elif fault == 'incomplete':
        (checkpoint / '_SUCCESS').unlink()
    elif fault == 'shard':
        with (checkpoint / 'rank-00000.pt').open('ab') as handle:
            handle.write(b'SYNTHETIC_TEST_ONLY_CORRUPTION')
    else:
        path = checkpoint / 'manifest.json'
        manifest = json.loads(path.read_text())
        if fault == 'identity':
            manifest['identity']['native']['source_head'] = 'changed'
        elif fault == 'policy':
            manifest['policy']['strategy_sha256'] = '0'*64
        else:
            manifest['cursor'] += 1
        path.write_text(json.dumps(manifest))
        (checkpoint / '_SUCCESS').write_text(file_sha256(path) + '\n')
    count = len(h.begins)
    with pytest.raises(ValueError, match=message):
        h.run('rejected', resume=checkpoint)
    assert len(h.begins) == count
    assert len(h.parts) == (2 if fault == 'shard' else 1)
    if fault == 'shard':
        assert not h.parts[-1].seen
        assert h.parts[-1].g.global_steps == h.parts[-1].c.global_steps == 0
    receipt = json.loads((h.root / 'rejected/rank_00.json').read_text())
    assert receipt['status'] == 'FAILED_ORIGINAL_WAN_WORK'
    assert 'cycles' not in receipt and not (h.root / 'rejected/checkpoints').exists()


def test_worker_quiescence_rejects_pending_communication_before_publication(cpu_worker):
    h = cpu_worker
    h.fault.pending_communication = True
    with pytest.raises(ValueError, match='checkpoint quiescence.*pending native gradient communication'):
        h.run('pending')
    assert h.boundaries == [1]
    assert not (h.root / 'pending/checkpoints').exists()
    receipt = json.loads((h.root / 'pending/rank_00.json').read_text())
    assert receipt['status'] == 'FAILED_ORIGINAL_WAN_WORK'
    assert receipt['final_versions'] == dict(generator=1, critic=5, ema=5)
    assert [row['cycle'] for row in receipt['cycles']] == [1]
    assert receipt['training_restart']['latest_checkpoint'] is None
