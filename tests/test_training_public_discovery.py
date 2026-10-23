"""Public discovery must not import optional models or plugin code."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_installed_entry_points_are_lazy_and_do_not_shadow_builtins(monkeypatch):
    from distillschedule.training import catalog
    class Entry:
        name = 'outside-small-model'
        value = 'outside_package.adapter:build_model'
        dist = SimpleNamespace(metadata={'Name': 'outside-model-package'}, version='1.2.3')
        def load(self):
            raise AssertionError('discovery must never load a plugin')
    def entry_points(*, group):
        return [Entry()] if group == 'distillschedule.models' else []
    monkeypatch.setattr(catalog.importlib.metadata, 'entry_points', entry_points)
    value = catalog.resolve('model', Entry.name)
    assert value['factory'] == Entry.value
    assert value['provider'] == {'distribution':'outside-model-package', 'version':'1.2.3',
                                 'entry_point_group':'distillschedule.models'}
    assert catalog.describe()['models'][Entry.name] == value
    assert value['capability_validation'] == 'ON_FACTORY_ADMISSION'


@pytest.mark.parametrize('name,value', [('sd15','new.adapter:build'),
    ('bad name','new.adapter:build'), ('outside-small-model','new.adapter:build [extra]'),
    ('outside-small-model','/untrusted/file.py:build'),
    ('outside-small-model','new.adapter:missing:part')])
def test_invalid_or_colliding_entrypoint_rejected(monkeypatch, name, value):
    from distillschedule.training import catalog
    ep = SimpleNamespace(name=name, value=value,
        dist=SimpleNamespace(metadata={'Name':'outside-model-package'}, version='1.2.3'))
    monkeypatch.setattr(catalog.importlib.metadata, 'entry_points',
                        lambda *, group: [ep] if group == 'distillschedule.models' else [])
    with pytest.raises(ValueError):
        catalog.describe()


def test_strategy_and_capability_discovery_is_stdlib_only():
    script = '''
import json, sys
from distillschedule.training.strategy_catalog import describe, resolve
v = describe()
assert 'resident' in v['strategies'] and 'fsdp2' in v['strategies']
assert 'frozen-parameter-sharing' in v['capabilities']
assert resolve('strategy','resident') == v['strategies']['resident']
assert not any(x in sys.modules for x in ('torch','diffusers','transformers','ltx_core'))
print(json.dumps(v))
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    for entry in value['capabilities'].values():
        assert entry['implementation'] and entry['applicability'] and entry['evidence']
        assert entry['automatic_enable'] is False


@pytest.mark.parametrize('command', ['models','algorithms','strategies','capabilities'])
def test_all_public_discovery_commands_avoid_optional_imports(command):
    result = subprocess.run([sys.executable, '-c', '''
import sys
from distillschedule.cli import main
assert main([sys.argv[1],'list']) == 0
assert not any(x in sys.modules for x in ('torch','diffusers','transformers','ltx_core'))
''', command], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_installed_distribution_metadata_resolves_without_loading_plugin(tmp_path):
    site = tmp_path/'installed'; info = site/'external_model-1.2.3.dist-info'; info.mkdir(parents=True)
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: external-model\nVersion: 1.2.3\n')
    (info/'entry_points.txt').write_text('[distillschedule.models]\nexternal-installed = external_unavailable.adapter:build_model\n\n[distillschedule.algorithms]\nexternal-algorithm = external_unavailable.adapter:build_algorithm\n')
    result = subprocess.run([sys.executable, '-c', '''
import json, sys
sys.path.insert(0, sys.argv[1])
from distillschedule.training.catalog import resolve
from distillschedule.training.config import TrainingConfig
assert resolve('model','external-installed')['provider']['distribution'] == 'external-model'
config = TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
    model={'name':'external-installed'}, algorithm={'name':'external-algorithm'},
    data={'prepared':'unavailable/manifest.json'}, execution={'mode':'reference'},
    trainer={'cycles':1,'output_dir':'uncreated'}), source='request.json')
assert config.model['factory'] == 'external_unavailable.adapter:build_model'
assert config.algorithm['factory'] == 'external_unavailable.adapter:build_algorithm'
assert 'external_unavailable' not in sys.modules and 'torch' not in sys.modules
print(json.dumps(config.to_dict()))
''', str(site)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src')})
    assert result.returncode == 0, result.stdout + result.stderr
