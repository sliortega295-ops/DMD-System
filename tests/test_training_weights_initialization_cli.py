"""The unified weights initialization command dispatches explicit producer schemas."""
import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize('family', ['wan', 'livewan'])
def test_import_weights_dispatches_versioned_producer_without_training(tmp_path, monkeypatch, capsys, family):
    from distillschedule.cli import main
    from distillschedule.training import wan_generator_import, livewan_generator_import
    selected = {'wan': wan_generator_import, 'livewan': livewan_generator_import}[family]
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('wrong producer importer selected')
    for module in (wan_generator_import, livewan_generator_import):
        monkeypatch.setattr(module, 'import_generator_weights', forbidden)
    def initialize(request, output):
        calls.append((request, output))
        return dict(status='EXPORTED', initialization_only=True, training_cycles_executed=0)
    monkeypatch.setattr(selected, 'import_generator_weights', initialize)
    path = tmp_path/'initialization.json'; path.write_text(json.dumps(dict(schema_version=selected.SCHEMA)))
    output = tmp_path/'portable'
    assert main(['import-weights', str(path), '--output', str(output)]) == 0
    assert calls == [(path, output)]
    assert json.loads(capsys.readouterr().out)['training_cycles_executed'] == 0


def test_unknown_initialization_schema_rejects_before_importer(tmp_path, monkeypatch, capsys):
    from distillschedule.cli import main
    from distillschedule.training import wan_generator_import, livewan_generator_import
    def forbidden(*args, **kwargs):
        raise AssertionError('unknown schema entered a model importer')
    for module in (wan_generator_import, livewan_generator_import):
        monkeypatch.setattr(module, 'import_generator_weights', forbidden)
    path = tmp_path/'initialization.json'; path.write_text('{"schema_version":"UNKNOWN"}')
    assert main(['import-weights', str(path), '--output', str(tmp_path/'portable')]) == 2
    assert 'unsupported weights initialization schema' in capsys.readouterr().err
    assert not (tmp_path/'portable').exists()


def test_duplicate_initialization_schema_rejects_before_importer(tmp_path, monkeypatch, capsys):
    from distillschedule.cli import main
    from distillschedule.training import wan_generator_import, livewan_generator_import
    def forbidden(*args, **kwargs):
        raise AssertionError('duplicate schema entered a model importer')
    for module in (wan_generator_import, livewan_generator_import):
        monkeypatch.setattr(module, 'import_generator_weights', forbidden)
    path = tmp_path/'initialization.json'; path.write_text('{"schema_version":"x","schema_version":"y"}')
    assert main(['import-weights', str(path), '--output', str(tmp_path/'portable')]) == 2
    assert not (tmp_path/'portable').exists()


def test_actual_livewan_cli_publishes_original_tiny_student(tmp_path, capsys):
    missing = [n for n in ('torch','peft','diffusers','omegaconf') if importlib.util.find_spec(n) is None]
    if missing:
        if os.environ.get('R225_REQUIRE_UPSTREAM') == '1':
            pytest.fail('actual existing upstream dependencies required: '+str(missing))
        pytest.skip('actual existing upstream dependencies unavailable')
    from test_training_livewan_generator_import import fixture_request
    from distillschedule.cli import main
    case = fixture_request(tmp_path/'OLD-origin', qualified_runtime=True)
    output = tmp_path/'portable'
    capsys.readouterr()
    assert main(['import-weights', str(case['path']), '--output', str(output)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['training_cycles_executed'] == value['checkpoint_cycle'] == 0
    assert value['identity']['weights_initialization']['origin_cycle'] == 100
    assert value['identity']['weights_initialization']['origin_numerical_status'] == 'FAIL_EXTREME_NUMERICAL_DRIFT'
    assert (output/'_SUCCESS').is_file()
