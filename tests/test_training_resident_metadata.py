"""Configuration inspection must not require the optional training runtime."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import distillschedule


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_public_budget_and_config_inspection_without_training_imports(tmp_path, device):
    # A fresh interpreter also catches accidental imports when the parent
    # pytest process already has Torch or model classes loaded.
    program = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'numpy', 'diffusers', 'transformers',
                'safetensors', 'accelerate', 'peft'} or fullname in {
                'distillschedule.training.sd', 'distillschedule.training.self_forcing',
                'distillschedule.open_workload.resident_trial'}):
            raise AssertionError('configuration imported training: ' + fullname)
sys.meta_path.insert(0, NoTrainingImports())
from distillschedule.training.resident import ResidentTrainingBudget, SCHEMA
from distillschedule.training.config import TrainingConfig
trial = dict(host_limit_bytes=2**31, snapshot_limit_bytes=2**29,
             result_limit_bytes=2**29, scratch_bytes=2**27, reserve_bytes=2**27)
if sys.argv[2].startswith('cuda'):
    trial.update(device_limit_bytes=2**32, host_growth_bytes=2**27,
                 device_state_growth_bytes=2**27, device_gradient_bytes=2**26,
                 device_activation_bytes=2**29, device_workspace_bytes=2**27,
                 device_restore_bytes=2**26, device_reserve_bytes=2**27)
value = dict(schema_version=SCHEMA, construction_limit_bytes=2**27,
             host_staging_limit_bytes=2**27, trial=trial)
assert ResidentTrainingBudget.parse(value, device=sys.argv[2]).to_dict() == value
raw = dict(schema_version='distillschedule.training.v1', model=dict(name='sd15'),
           algorithm=dict(name='dmd2'), data=dict(prepared='features/manifest.json'),
           execution=dict(mode='auto', resident_budget=value),
           trainer=dict(cycles=3, seed=17, device=sys.argv[2], output_dir='run'))
config = TrainingConfig.from_dict(raw, source='train.json')
assert config.execution['resident_budget'] == value
assert TrainingConfig.from_dict(config.to_dict(), source=config.source).identity == config.identity
for name in trial:
    for invalid in (False, 0, -1, 1.5, '1', None):
        changed = dict(value, trial=dict(trial, **{name: invalid}))
        try: ResidentTrainingBudget.parse(changed, device=sys.argv[2])
        except ValueError: pass
        else: raise AssertionError((name, invalid))
assert 'torch' not in sys.modules
print(json.dumps(dict(device=sys.argv[2], valid_config=True, torch_imported=False)))
'''
    package_root = Path(distillschedule.__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, '-I', '-c', program, str(package_root), device],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(device=device, valid_config=True, torch_imported=False)


def test_budget_runtime_reexports_preserve_class_and_error_identity():
    # The real trial must still use these same classes, not a permissive
    # metadata look-alike whose checks diverge during model construction.
    from distillschedule.core.contracts import ContractViolation
    from distillschedule.open_workload import resident_budget, resident_trial
    for name in ('TrialBudget', 'CudaTrialBudget', '_require'):
        assert getattr(resident_trial, name) is getattr(resident_budget, name)
    with pytest.raises(ContractViolation) as error:
        resident_trial.TrialBudget(1, 1, 1, 1, False).validate()
    assert error.value.path == 'open.resident_trial'
