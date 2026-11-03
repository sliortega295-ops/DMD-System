"""Public native LTX block ownership; tiny random upstream models, CPU/Gloo only."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_training_ltx_cpu import source_root


CONSTRUCTION = 'cpu-fsdp2-native-blocks-v1'
MANUAL = dict(mode='manual', implementation=dict(backend='torch-callable-replicated-v1',
    reduction='fsdp2_reshard', bucket_cap_mb=0, local=dict(backend='torch-callable-resident-v1',
        share_immutable=False, checkpoint_bindings=[])))


def _request(tmp_path):
    from test_training_ltx_device import _parameters
    return dict(schema_version='distillschedule.training.v1',
        model=dict(name='ltx25', parameters=dict(_parameters(), construction=CONSTRUCTION)),
        algorithm=dict(name='dmd'), data=dict(prepared='MISSING', sharding='rank_strided'),
        execution=deepcopy(MANUAL), trainer=dict(device='cpu', distributed=dict(backend='gloo'),
            cycles=1, output_dir=str(tmp_path/'out')))


def test_ltx_block_construction_catalog_is_exact_lazy_and_torch_free(tmp_path):
    result = subprocess.run([sys.executable, '-c', '''
import sys
from distillschedule.training.catalog import resolve, validate_compatibility
entry = resolve('model', 'ltx25')
constraint = entry['construction_constraint']
assert constraint['recipe'] == 'cpu-fsdp2-native-blocks-v1'
assert entry['execution_constraints']['replicated']['manual_implementation']['reduction'] == 'post_backward'
validate_compatibility({'factory': entry['factory'], 'parameters': {'construction': constraint['recipe']}},
    {'factory': 'distillschedule.training.dmd:build_algorithm'},
    {'device': 'cpu', 'distributed': {'backend': 'gloo'}},
    {'mode': 'manual', 'implementation': constraint['execution']['manual_implementation']})
assert not any(name in sys.modules for name in ('torch', 'transformers', 'ltx_core'))
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ltx_block_helper_source_receipt_is_exact_and_torch_free():
    result = subprocess.run([sys.executable, '-c', '''
import hashlib
from pathlib import Path
import sys
import distillschedule.training.ltx_sharding as provider
receipt = provider._provider_source()
assert receipt == {'module': provider.__name__, 'sha256': hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()}
assert not any(name in sys.modules for name in ('torch', 'transformers', 'ltx_core'))
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ltx_block_public_policy_rejects_before_data_model_and_rng(tmp_path, monkeypatch):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import session
    config = TrainingConfig.from_dict(_request(tmp_path), source=tmp_path/'request.json')
    def forbidden(*args, **kwargs):
        pytest.fail('policy rejection must precede data, RNG and model ownership')
    for name in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, name, forbidden)
    invalid = [dict(mode='reference'), dict(mode='auto'), dict(mode='plan', plan='MISSING'),
        dict(MANUAL, order=[]), dict(MANUAL, trials=1), dict(MANUAL, resident_budget={})]
    for reduction in ('post_backward', 'fsdp2_keep_forward', 'hsdp2_reshard', 'per_module'):
        invalid.append(dict(MANUAL, implementation=dict(MANUAL['implementation'], reduction=reduction)))
    invalid.append(dict(MANUAL, implementation=dict(MANUAL['implementation'],
        local=dict(MANUAL['implementation']['local'], share_immutable=True))))
    for execution in invalid:
        with pytest.raises(ValueError, match='construction'):
            session.construct(replace(config, execution=execution))
    for field, value in [('compute_precision', 'bf16-mixed'), ('device', 'cuda:0'),
                         ('construction', 'foreign')]:
        parameters = dict(config.model['parameters'], **{field: value})
        with pytest.raises(ValueError, match='construction'):
            session.construct(replace(config, model=dict(config.model, parameters=parameters)))
    with pytest.raises(ValueError, match='construction'):
        session.construct(replace(config, trainer=dict(config.trainer, distributed={'backend': 'nccl'})))
    assert not torch.cuda.is_initialized()


def test_ltx_block_constructor_requires_live_public_provider_before_assets(monkeypatch, tmp_path):
    import distillschedule.training.ltx as ltx
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import _construct_model
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted native block constructor reached source or assets')
    monkeypatch.setattr(ltx, 'original', forbidden)
    config = TrainingConfig.from_dict(_request(tmp_path), source=tmp_path/'request.json')
    with pytest.raises(ValueError, match='provider'):
        ltx.build_model(config.model['parameters'])
    with pytest.raises(ValueError, match='CPU/Gloo ranks'):
        _construct_model(config)
    with pytest.raises(ValueError, match='provider'):
        ltx._build_model(config.model['parameters'], construction=object())


def test_ltx_block_failed_constructor_closes_single_use_provider():
    from distillschedule.training.ltx_sharding import _NativeBlockConstruction
    factory = object()
    def failing(parameters, **kwargs):
        raise OSError('native owner construction failed')
    provider = _NativeBlockConstruction({'construction': CONSTRUCTION}, factory, failing)
    with pytest.raises(OSError, match='native owner construction failed'):
        provider.construct(factory, dict(provider.parameters))
    assert provider.started and provider.closed and not provider.attached
    with pytest.raises(ValueError, match='provider/factory/parameters'):
        provider.construct(factory, dict(provider.parameters))


def test_native_ltx_block_pytree_rejects_conflicting_registration(source_root):
    result = subprocess.run([sys.executable, '-c', '''
import sys
from torch.utils._pytree import register_pytree_node, tree_leaves
from distillschedule.training.ltx_source import original
from distillschedule.training.ltx_sharding import _register_native_arguments
source = original(sys.argv[1]).source
cls = source.import_module('ltx_core.model.transformer.transformer_args').TransformerArgs
register_pytree_node(cls, lambda value: (['foreign-owner'], None), lambda values, context: None)
node = object.__new__(cls)
assert tree_leaves(node) == ['foreign-owner']
try:
    _register_native_arguments(source)
except ValueError as error:
    assert 'already registered' in str(error)
else:
    raise AssertionError('conflicting native pytree ownership was silently overwritten')
assert tree_leaves(node) == ['foreign-owner']
''', str(source_root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope='module')
def block_assets(tmp_path_factory, source_root):
    from distillschedule.training.plugins import load_callable
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo/'recipes/ltx25/create_cpu_fixture.py') + ':build_fixture')
    original = builder.__globals__['tiny_architectures']
    def two_blocks():
        transformer, media = original()
        transformer['num_layers'] = 2
        return transformer, media
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(builder.__globals__, 'tiny_architectures', two_blocks)
        return builder(tmp_path_factory.mktemp('ltx-blocks')/'fixture', source_root, cycles=6)


def _block_request(assets, output, mode):
    from distillschedule.training.config import load_training_config
    value = load_training_config(assets[mode]).to_dict()
    value['model']['parameters']['construction'] = CONSTRUCTION
    value['algorithm']['parameters']['max_grad_norm'] = 1e-4
    value['data']['sharding'] = 'rank_strided'
    value['trainer'].update(cycles=3, distributed=dict(backend='gloo', timeout_seconds=180))
    value['execution'] = deepcopy(MANUAL)
    Path(output).write_text(json.dumps(value))
    return output


@pytest.mark.skipif(os.environ.get('DMD_LTX_SHARDED_CPU') != '1',
    reason='explicit pinned CPU LTX block-FSDP lane required')
@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_actual_native_ltx_two_block_gloo2_oracle_and_public_fresh_av_lifecycle(block_assets, tmp_path, config_key):
    from ltx_sharded_lifecycle import lifecycle
    config = _block_request(block_assets, tmp_path/'request.json', config_key)
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
        str(repo/'tests/ltx_sharded_lifecycle.py'), 'oracle', '--config', str(config), '--output', str(tmp_path/'oracle')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600,
        env={**os.environ, 'OMP_NUM_THREADS': '1'})
    (tmp_path/'oracle.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads((tmp_path/'oracle/proof.json').read_text())
    assert len(proof['ranks']) == 2
    assert all(row['average_before_clip_adam'] and row['oracle_cycles'] == 3 for row in proof['ranks'])
    assert all(row['native_group_installations'] == 9 and row['native_method_registrations'] == 3 for row in proof['ranks'])
    lifecycle(config, tmp_path/'lifecycle')
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(os.environ.get('DMD_LTX_SHARDED_CPU') != '1',
    reason='explicit pinned CPU LTX block-FSDP lane required')
def test_actual_native_ltx_two_block_gloo2_preserves_disabled_ema(block_assets, tmp_path):
    config = _block_request(block_assets, tmp_path/'no-ema.json', 'config')
    value = json.loads(config.read_text())
    value['model']['parameters']['ema_decay'] = None
    config.write_text(json.dumps(value))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
        str(repo/'tests/ltx_sharded_lifecycle.py'), 'oracle', '--config', str(config), '--output', str(tmp_path/'oracle')],
        capture_output=True, text=True, timeout=600, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    (tmp_path/'oracle.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads((tmp_path/'oracle/proof.json').read_text())
    assert all(row['records'][-1]['ownership']['ema'] == 'DISABLED_UNCHANGED' for row in proof['ranks'])
