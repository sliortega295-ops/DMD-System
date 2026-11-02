"""Pinned actual LTX AV post-backward replicas; random native CPU fixtures only."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_training_flow_image_device import CUDA_MANUAL, REPLICATED_MANUAL
from test_training_ltx_cpu import source_root


def test_ltx_replication_catalog_is_exact_lazy_and_torch_free():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from distillschedule.training.catalog import resolve, validate_compatibility
entry = resolve('model', 'ltx25')
assert entry['execution_constraints']['replicated']['manual_implementation']['reduction'] == 'post_backward'
for device, backend in [('cpu', 'gloo'), ('cuda:3', 'nccl')]:
    validate_compatibility({'factory': entry['factory']},
        {'factory': 'distillschedule.training.dmd:build_algorithm'},
        {'device': device, 'distributed': {'backend': backend}}, {'mode': 'reference'})
assert not any(name in sys.modules for name in ('torch', 'transformers', 'ltx_core'))
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('device,backend', [('cpu', 'gloo'), ('cuda:1', 'nccl')])
def test_ltx_replica_policy_rejects_before_data_rng_and_model(tmp_path, monkeypatch, device, backend):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import session
    value = dict(schema_version='distillschedule.training.v1', model=dict(name='ltx25', parameters={}),
        algorithm=dict(name='dmd'), data=dict(prepared='MISSING', sharding='rank_strided'),
        execution=deepcopy(REPLICATED_MANUAL), trainer=dict(device=device,
            distributed=dict(backend=backend), cycles=1, output_dir=str(tmp_path/'out')))
    config = TrainingConfig.from_dict(value, source=tmp_path/'request.json')
    def forbidden(*args, **kwargs):
        pytest.fail('replica policy refusal must precede data, model and RNG')
    for name in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, name, forbidden)
    invalid = [dict(mode='auto'), dict(mode='plan', plan='MISSING'), CUDA_MANUAL,
        dict(REPLICATED_MANUAL, order=[]), dict(mode='reference', implementation={}),
        dict(mode='reference', adapter='native'), dict(mode='reference', memory_limit_bytes=1),
        *[dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'], reduction=kind))
          for kind in ('ddp_overlap', 'fsdp2_reshard', 'hsdp2_reshard', 'per_module')],
        dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'], bucket_cap_mb=False)),
        dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'],
            local=dict(CUDA_MANUAL['implementation'], share_immutable=1)))]
    for execution in invalid:
        with pytest.raises(ValueError, match='REPLICATED execution requires'):
            session.construct(replace(config, execution=execution))
    value['execution'] = dict(mode='reference')
    TrainingConfig.from_dict(value, source=tmp_path/'reference.json')
    value['trainer']['distributed']['backend'] = 'nccl' if backend == 'gloo' else 'gloo'
    with pytest.raises(ValueError, match='backend'):
        TrainingConfig.from_dict(value, source=tmp_path/'wrong-backend.json')
    assert not torch.cuda.is_initialized()


def _owned_group(monkeypatch):
    """Admission-only witness; native Gloo acceptance below never mocks ranks."""
    for key, value in dict(WORLD_SIZE=2, RANK=1, LOCAL_WORLD_SIZE=2, LOCAL_RANK=1, GROUP_RANK=0).items():
        monkeypatch.setenv(key, str(value))
    for name, value in dict(is_initialized=True, get_world_size=2, get_rank=1, get_backend='gloo').items():
        monkeypatch.setattr(torch.distributed, name, lambda *args, _value=value, **kwargs: _value)


def test_ltx_training_rank_ownership_and_single_device_inference_stay_distinct(monkeypatch):
    from test_training_ltx_device import _parameters
    import distillschedule.training.ltx as ltx
    _owned_group(monkeypatch)
    class ReachedSource(Exception):
        pass
    def source(*args):
        raise ReachedSource
    monkeypatch.setattr(ltx, 'original', source)
    with pytest.raises(ReachedSource):
        ltx.build_model(_parameters())
    for key, value in [('WORLD_SIZE', '3'), ('RANK', '0'), ('LOCAL_WORLD_SIZE', '3'),
                       ('LOCAL_RANK', '0'), ('GROUP_RANK', '1'), ('LOCAL_WORLD_SIZE', None)]:
        with monkeypatch.context() as patch:
            patch.delenv(key) if value is None else patch.setenv(key, value)
            with pytest.raises(ValueError, match='torchrun|ownership'):
                ltx.build_model(_parameters())
    for name, value in [('is_initialized', False), ('get_world_size', 1), ('get_backend', 'nccl')]:
        with monkeypatch.context() as patch:
            patch.setattr(torch.distributed, name, lambda *args, _value=value, **kwargs: _value)
            with pytest.raises(ValueError, match='torchrun|ownership|Gloo'):
                ltx.build_model(_parameters())
    parameters = dict(source_root='MISSING', architecture={}, media_architecture={}, binding='00'*32,
        trainable=dict(mode='full'), generation_sigmas=[1., 0.])
    with pytest.raises(ValueError, match='distributed'):
        ltx.build_inference_model(parameters)
    assert not torch.cuda.is_initialized()


def test_ltx_replica_cuda_metadata_is_global_but_local_index_ownership_is_strict(monkeypatch):
    """Source-only CUDA admission witness, never a CUDA execution claim."""
    from test_training_ltx_device import _parameters
    import distillschedule.training.ltx as ltx
    _owned_group(monkeypatch)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda: 'nccl')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    class ReachedSource(Exception):
        pass
    def source(*args):
        raise ReachedSource
    monkeypatch.setattr(ltx, 'original', source)
    with pytest.raises(ReachedSource):
        ltx.build_model(dict(_parameters(), device='cuda:1'))
    for device in ('cuda:0', 'cuda:2'):
        with pytest.raises(ValueError, match='device|LOCAL_RANK'):
            ltx.build_model(dict(_parameters(), device=device))
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'current_device', lambda: 0)
        with pytest.raises(ValueError, match='LOCAL_RANK'):
            ltx.build_model(dict(_parameters(), device='cuda:1'))
    assert ltx._training_placement(torch.device('cuda:0'), 2) == ltx._training_placement(torch.device('cuda:1'), 2)
    assert ltx._training_placement(torch.device('cuda:1'), 1) == dict(device='cuda:1', world_size=1, precision='FP32')
    assert not torch.cuda.is_initialized()


def test_ltx_replicated_request_binds_local_owner_data_precision_and_original_policy(monkeypatch):
    import distillschedule.training.ltx as ltx
    from distillschedule.data.ltx_inputs import SCHEMA
    _owned_group(monkeypatch)
    schedule = [1., .75, .5, .25, 0.]
    state = {name: SimpleNamespace(device=torch.device('cpu'), generation_sigmas=schedule,
        compute_precision='fp32') for name in ('student', 'fake', 'teacher')}
    request = SimpleNamespace(model=dict(parameters={}), trainer=dict(device='cpu', distributed=dict(backend='gloo')),
        data=dict(sharding='rank_strided'), execution=dict(mode='reference'),
        algorithm=dict(factory='distillschedule.training.dmd:build_algorithm', parameters=dict(rollout_sigmas=schedule)))
    manifest = dict(metadata=dict(schema_version=SCHEMA))
    for execution in (dict(mode='reference'), deepcopy(REPLICATED_MANUAL)):
        request.execution = execution
        ltx.admit_training(state=state, request=request, prepared_identity='test', prepared_manifest=manifest)
    for field, value in [('data', {}), ('data', dict(sharding='rank_strided', streams={})),
                         ('trainer', dict(device='cpu', distributed=dict(backend='nccl'))),
                         ('execution', CUDA_MANUAL), ('execution', dict(mode='auto'))]:
        bad = deepcopy(request); setattr(bad, field, value)
        with pytest.raises(ValueError, match='rank_strided|backend|private flow|record'):
            ltx.admit_training(state=state, request=bad, prepared_identity='test', prepared_manifest=manifest)
    state['fake'].device = torch.device('cuda:0')
    with pytest.raises(ValueError, match='ownership'):
        ltx.admit_training(state=state, request=request, prepared_identity='test', prepared_manifest=manifest)


@pytest.mark.parametrize('retire_at', [2, 3])
def test_ltx_replica_driver_restores_every_retired_input_after_partial_rename_failure(tmp_path, monkeypatch, retire_at):
    """Test-driver cleanup only: metadata stubs never execute a model or group."""
    import native_flow_replicated as driver
    fixture = tmp_path/'fixture'; fixture.mkdir()
    paths = [fixture/'assets', fixture/'prepared', fixture/'raw']
    for path in paths:
        path.mkdir(); (path/'sentinel').write_text(path.name)
    value = dict(model=dict(factory='distillschedule.training.ltx:build_model',
        parameters=dict(assets=str(paths[0]/'assets.json'))), data=dict(prepared=str(paths[1]/'manifest.json')))
    monkeypatch.setattr(driver, 'load_training_config', lambda _: SimpleNamespace(to_dict=lambda: deepcopy(value)))
    monkeypatch.setattr(driver, 'fresh_cli', lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, 'verify_restart', lambda root, mode, world: Path(root)/mode/'checkpoint')
    def forbidden(*args):
        pytest.fail('partial retirement failure must not reach portable generation')
    monkeypatch.setattr(driver, 'portable_output', forbidden)
    original, calls = Path.rename, []
    def rename(self, target):
        if self in paths:
            calls.append(self)
            if len(calls) == retire_at:
                raise OSError('retirement failure witness')
        return original(self, target)
    monkeypatch.setattr(Path, 'rename', rename)
    with pytest.raises(OSError, match='retirement failure witness'):
        driver.lifecycle(tmp_path/'request.json', tmp_path/'lifecycle')
    assert len(calls) == retire_at
    assert all((path/'sentinel').read_text() == path.name for path in paths)
    assert not any(path.with_name(path.name+'-retired').exists() for path in paths)


@pytest.fixture(scope='module')
def replica_assets(tmp_path_factory, source_root):
    from distillschedule.training.plugins import load_callable
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo/'recipes/ltx25/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path_factory.mktemp('ltx-replicas')/'fixture', source_root, cycles=6)


def test_ltx_independent_oracle_preserves_default_teacher_cfg2_and_cfg1_call_order(replica_assets, monkeypatch):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    from flow_replica_oracle import sequential_b1_oracle
    for scale in (1., 2.):
        config = load_training_config(replica_assets['lora_config'])
        assert config.algorithm['parameters']['teacher_cfg'] == 2.
        model, data, _, _ = construct(config)
        calls = []
        for role in ('teacher', 'fake'):
            original = model.state[role].denoise
            def denoise(*args, _role=role, _original=original, **kwargs):
                calls.append((_role, kwargs['conditional']))
                return _original(*args, **kwargs)
            monkeypatch.setattr(model.state[role], 'denoise', denoise)
        policy = {**config.algorithm['parameters'], 'teacher_cfg': scale}
        proof = sequential_b1_oracle(model.state, [data.sample(0), data.sample(1)], policy, 67)
        expected_teacher = [('teacher', True)] + ([('teacher', False)] if scale != 1 else [])
        assert [row for row in calls if row[0] == 'teacher'] == expected_teacher * 2
        assert [row for row in calls if row[0] == 'fake'] == [('fake', True)] * 12
        assert model.state['dmd_counters'].tolist() == [1, 1, 5]
        assert len(proof['gradients']) == 6
        assert all(bool(torch.isfinite(loss)) for row in proof['losses'] for loss in row.values())
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(os.environ.get('DMD_LTX_REPLICATED_CPU') != '1',
    reason='opt in with the pinned isolated LTX lane and DMD_LTX_REPLICATED_CPU=1')
@pytest.mark.parametrize('compute_precision', ['fp32', 'bf16-mixed'])
def test_actual_native_ltx_gloo2_default_cfg2_oracle_and_fresh_av_lifecycle(replica_assets, tmp_path, compute_precision):
    from native_flow_replicated import request, lifecycle
    repo = Path(__file__).resolve().parents[1]
    config = request(replica_assets['lora_config'], tmp_path/'replicated.json', 'cpu')
    value = json.loads(config.read_text())
    value['trainer']['cycles'] = 3
    if compute_precision != 'fp32':
        value['model']['parameters']['compute_precision'] = compute_precision
    config.write_text(json.dumps(value))
    assert value['algorithm']['parameters']['teacher_cfg'] == 2.
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
        str(repo/'tests/native_flow_replicated.py'), 'oracle', '--config', str(config), '--output', str(tmp_path/'oracle')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    (tmp_path/'oracle.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads((tmp_path/'oracle/proof.json').read_text())
    assert len(proof['ranks']) == 2 and all(row['average_before_clip_adam'] for row in proof['ranks'])
    assert all(row['oracle_cycles'] == 3 and row['later_student_gradient_nonzero'] for row in proof['ranks'])
    assert all(row['compute_precision'] == compute_precision for row in proof['ranks'])
    assert all(row['family'] == 'ltx25' for row in proof['ranks'])
    assert all(row['native_evidence_kind'] == 'RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED' for row in proof['ranks'])
    assert all(len(record['cycles']) == 3 for row in proof['ranks'] for record in row['records'])
    lifecycle(config, tmp_path/'lifecycle', ranks=2)
    assert not torch.cuda.is_initialized()
