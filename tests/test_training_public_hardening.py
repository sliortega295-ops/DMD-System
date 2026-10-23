"""Real public chains and fail-closed metadata; CPU, never a GPU receipt."""
import json
from pathlib import Path

import pytest
import torch


def test_cleared_model_contract_cannot_reuse_prevalidated_metadata():
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.prediction import PredictionSpec
    components = ModelComponents(state={'counter': 0}, prediction=PredictionSpec('flow'),
                                 capabilities=('video',)).validate()
    components.prediction = None
    components.capabilities = ()
    with pytest.raises(ValueError, match='prediction/capabilities'):
        components.validate()
    assert '_distillschedule_model_contract' not in ModelComponents(state={'counter': 0}).validate().metadata


def test_upstream_importer_implementation_is_part_of_source_identity(tmp_path):
    from distillschedule.training import upstream
    from distillschedule.workloads.best_known.registry import file_sha256
    source = tmp_path / 'native.py'; source.write_text('def native(): return 1\n')
    locked = upstream.PinnedSource(tmp_path, {'native.py': file_sha256(source)})
    assert locked.identity['importer_sha256'] == file_sha256(Path(upstream.__file__))


def _first(ctx):
    return ctx.inputs['x'] + 1


def _second(ctx):
    return ctx.inputs['first'] * 2


@pytest.mark.parametrize('measure,expected', [(False, 1), (True, 5)])
def test_session_only_synchronizes_component_boundaries_when_measuring(measure, expected):
    from distillschedule.open_workload.torch_backend import Session, Workload
    from distillschedule.workloads.program import Program, Call
    workload = Workload(Program((
        Call('first', 'first', ('x',), (), (), grad='no_grad'),
        Call('second', 'second', ('first',), (), (), grad='no_grad')),
        initial_versions={}, feeds=('x',), outputs=('second',)),
        {'first': _first, 'second': _second}, {}, {'x': torch.tensor([2.])})
    session = Session(workload, dict(backend='torch-callable-resident-v1',
                                    share_immutable=False, checkpoint_bindings=[]))
    calls = []
    real_sync = session.synchronize
    def observe():
        calls.append('sync')
        real_sync()
    session.synchronize = observe
    result = session.run(measure=measure)
    assert torch.equal(result['second'], torch.tensor([6.]))
    assert len(calls) == expected
    if not measure:
        assert session.profile['first']['timing_method'] == 'host_dispatch_wall_not_gpu_completion'


def test_replicated_selective_policy_requires_actual_local_effects():
    from distillschedule.open_workload.checkpoint_policy import native_effect_evidence
    implementation = dict(backend='torch-callable-replicated-v1', reduction='post_backward',
        bucket_cap_mb=0, local=dict(backend='torch-callable-resident-v1',
            share_immutable=False, checkpoint_bindings=['fwd'], checkpoint_policy='save_matmul'))
    absent = native_effect_evidence(implementation, {'replica_effects': [{}, {}]})
    assert absent['active'] is False
    actual = native_effect_evidence(implementation, {'replica_effects': [
        dict(selective_save_requests=2, selective_native_cache_bytes=16),
        dict(selective_save_requests=3, selective_native_cache_bytes=24)]})
    assert actual['active'] is True


def _stream_algorithm(components, feeds, parameters, cycle, seed):
    from distillschedule.examples.tiny_av import build_algorithm
    return build_algorithm(components, feeds['generator'][0], parameters, cycle, seed)


def test_independent_training_streams_allow_separate_heldout_validation(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import train
    from test_training_end_to_end import fixture
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    prepared = raw['data'].pop('prepared')
    raw['data']['streams'] = {'generator': {'prepared': prepared}}
    raw['data']['validation_prepared'] = prepared
    raw['model']['factory'] = str(Path(__file__).with_name('test_training_validation.py')) + ':with_metric'
    raw['algorithm']['factory'] = __file__ + ':_stream_algorithm'
    raw['trainer'].update(cycles=2, output_dir=str(root / 'with-validation'),
                          validation={'every_cycles': 1, 'samples': 1})
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    assert result['validation']['status'] == 'COMPLETED_ADAPTER_METRICS'
    assert result['completed_cycles'] == 2
    rows = [json.loads(line) for line in (root / 'with-validation/validation.jsonl').read_text().splitlines()]
    assert [row['cycle'] for row in rows] == [1, 2]


def _generate_with_offset(state, feeds, parameters):
    from distillschedule.examples.tiny_av import generate
    if set(parameters) != {'offset'} or type(parameters['offset']) not in (int, float):
        raise ValueError('explicit offset required')
    return generate(state, feeds, {}) + parameters['offset']


def _model_with_parameters(parameters):
    from distillschedule.examples.tiny_av import build_model
    components = build_model(parameters)
    components.generate = _generate_with_offset
    return components


def test_common_cli_forwards_finite_json_generation_parameters(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import train, export
    from test_training_end_to_end import fixture, run_cli
    root, config = fixture(tmp_path, samples=3)
    raw = config.to_dict()
    raw['model']['factory'] = __file__ + ':_model_with_parameters'
    raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    path = root / 'parameters-training.json'; path.write_text(json.dumps(raw))
    config = TrainingConfig.from_dict(raw, source=path)
    trained = train(config)
    export(config, trained['checkpoint'], root / 'weights')
    parameters = root / 'generation-parameters.json'
    parameters.write_text(json.dumps({'offset': .125}))
    result = run_cli('generate', path, '--weights', root / 'weights',
                     '--prepared', config.data['prepared'], '--parameters', parameters,
                     '--output', root / 'new-video')
    assert result['generation_parameters'] == {'offset': .125}
    import os, subprocess, sys
    for index, payload in enumerate(('[1]', '{"offset": NaN}', '{"offset": 1, "offset": 2}')):
        parameters.write_text(payload)
        rejected = subprocess.run([sys.executable, '-m', 'distillschedule', 'generate', str(path),
            '--weights', str(root / 'weights'), '--prepared', config.data['prepared'],
            '--parameters', str(parameters), '--output', str(root / ('bad-' + str(index)))],
            capture_output=True, text=True, env=os.environ.copy())
        assert rejected.returncode == 2
        assert not (root / ('bad-' + str(index))).exists()


def test_cached_upstream_source_rejects_importer_source_drift(tmp_path, monkeypatch):
    from distillschedule.training import upstream
    from distillschedule.workloads.best_known.registry import file_sha256
    native = tmp_path / 'native.py'; native.write_text('def native(): return 1\n')
    importer = tmp_path / 'importer.py'
    importer.write_bytes(Path(upstream.__file__).read_bytes())
    monkeypatch.setattr(upstream, '__file__', str(importer))
    locked = upstream.PinnedSource(tmp_path, {'native.py': file_sha256(native)})
    assert locked.import_module('native').native() == 1
    importer.write_text(importer.read_text() + '\n# changed source\n')
    with pytest.raises(ValueError, match='importer'):
        locked.import_module('native')
