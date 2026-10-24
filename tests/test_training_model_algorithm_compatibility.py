"""Lazy registration is not a GPU, pretrained quality, or algorithm equivalence claim."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_new_flow_family_catalog_is_lazy_and_precise():
    code = '''import json, sys
from distillschedule.training.catalog import describe
value = describe()
assert 'torch' not in sys.modules
for name in ('flux2-klein-base', 'qwen-image', 'wan22-ti2v5b-t2v', 'hunyuan15-t2v'):
    row = value['models'][name]
    assert row['validation']['gpu'] == 'NOT_RUN'
    assert row['validation']['quality'] == 'NOT_RUN'
    assert 'dmd_flow_v1' in row['capabilities']
    assert 'causal' not in row['capabilities']
print(json.dumps(value))
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=os.environ)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['models']['qwen-image']['source']['revision']


def test_declared_flow_compatibility_rejects_before_torch_or_asset_loading():
    code = '''import sys
from distillschedule.training.config import TrainingConfig
for name in ('flux2-klein-base', 'qwen-image', 'wan22-ti2v5b-t2v', 'hunyuan15-t2v'):
    for algorithm in ('dmd2', 'self-forcing'):
        value = dict(schema_version='distillschedule.training.v1', model=dict(name=name, parameters={'assets':'MISSING'}),
            algorithm=dict(name=algorithm), data=dict(prepared='MISSING'), execution=dict(mode='reference'),
            trainer=dict(cycles=1, output_dir='MISSING', device='cpu'))
        try:
            TrainingConfig.from_dict(value, source='train.json')
        except ValueError as exc:
            assert 'incompatible' in str(exc), str(exc)
        else:
            raise AssertionError('known incompatible model/algorithm accepted')
assert not any(k in sys.modules for k in ('torch', 'diffusers', 'transformers'))
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, env=os.environ)
    assert result.returncode == 0, result.stdout + result.stderr


def test_static_device_cache_rejections_and_unknown_plugin_extension_are_lazy():
    code = '''import sys
from copy import deepcopy
from distillschedule.training.config import TrainingConfig
from distillschedule.training.catalog import resolve
for name in ('flux2-klein-base','qwen-image','wan22-ti2v5b-t2v','hunyuan15-t2v'):
    base=dict(schema_version='distillschedule.training.v1',model=dict(factory=resolve('model',name)['factory'],parameters={'assets':'MISSING'}),
        algorithm=dict(name='dmd'),data=dict(prepared='MISSING'),execution=dict(mode='reference'),
        trainer=dict(cycles=1,output_dir='MISSING',device='cpu'))
    # Single-device CUDA reference is now an explicit, metadata-only capability
    # for the four native flow families, never for trial cloning policies.
    value=deepcopy(base)
    value['trainer']['device']='cuda:0'
    TrainingConfig.from_dict(value,source='train.json')
    for mode in ('auto','manual','plan'):
        value['execution']['mode']=mode
        try:TrainingConfig.from_dict(value,source='train.json')
        except ValueError:pass
        else:raise AssertionError('unsupported CUDA policy accepted')
    for kind in ('distributed','causal','kv_cache'):
        value=deepcopy(base)
        if kind=='distributed':value['trainer']['distributed']={'backend':'gloo'}
        else:value['model']['parameters'][kind]=True
        if kind=='distributed' and name in ('flux2-klein-base','qwen-image'):
            # Only these two image owners now admit explicit reference Gloo
            # replication. Metadata discovery remains lazy; wrapper/trial
            # policies and the video owners' distribution remain refused.
            TrainingConfig.from_dict(value,source='train.json')
            value['execution']['mode']='auto'
            try:TrainingConfig.from_dict(value,source='train.json')
            except ValueError as exc:assert 'REPLICATED' in str(exc),str(exc)
            else:raise AssertionError('unsupported replicated trial accepted')
            continue
        try:TrainingConfig.from_dict(value,source='train.json')
        except ValueError as exc:assert 'incompatible' in str(exc),str(exc)
        else:raise AssertionError('unsupported native image capability accepted')
    base['model']={'factory':'external_uninstalled_package.adapter:build','parameters':{'causal':True}}
    base['algorithm']={'name':'dmd2'}
    TrainingConfig.from_dict(base,source='train.json')
assert not any(key in sys.modules for key in ('torch','diffusers','transformers'))
'''
    result = subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,env=os.environ)
    assert result.returncode == 0,result.stdout+result.stderr
