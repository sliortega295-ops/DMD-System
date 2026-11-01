"""Diagnose default-loader parity with actual original model and exact checks.

Runs in the dedicated full-model dependency lane, not the stdlib Skill suite.
"""
import json
from pathlib import Path
import os
import subprocess
import sys

import torch

from test_helios_reference_model_cpu import _tiny_model, _inputs


def test_default_loader_preserves_nonpersistent_buffers_and_exact_outputs(tmp_path):
    model = _tiny_model().eval()
    inputs = _inputs(3)
    with torch.no_grad():
        outputs = model(**inputs, gan_mode=True)
    model.save_pretrained(tmp_path/'model', safe_serialization=True)
    torch.save({'inputs':inputs,'outputs':outputs,'state':model.state_dict(),
                'buffers':{k:v.detach().clone() for k,v in model.named_buffers()}},tmp_path/'reference.pt')
    code = r'''
import json,sys,torch
from pathlib import Path
from distillschedule.adapters.helios.reference_model import load_reference_model_class
torch.set_num_threads(1)
root=Path(sys.argv[1]); expected=torch.load(root/'reference.pt',map_location='cpu',weights_only=True)
cls=load_reference_model_class(backend='torch-sdpa')
model=cls.from_pretrained(str(root/'model'),use_default_loader=True).eval()
state=model.state_dict(); buffers=dict(model.named_buffers())
report={'parameter_keys_equal':state.keys()==expected['state'].keys(),
 'bad_parameters':[k for k,v in expected['state'].items() if k not in state or not torch.equal(v,state[k])],
 'buffer_keys_equal':buffers.keys()==expected['buffers'].keys(),
 'buffers':{k:{'equal':k in buffers and torch.equal(v,buffers[k]),
              'device':str(buffers[k].device) if k in buffers else 'missing',
              'max_abs':float((v-buffers[k]).abs().max()) if k in buffers and not buffers[k].is_meta else None}
            for k,v in expected['buffers'].items()}}
with torch.no_grad(): actual=model(**expected['inputs'],gan_mode=True)
report['outputs']=[{'equal':torch.equal(a,b),'max_abs':float((a-b).abs().max()),
                    'finite':bool(torch.isfinite(a).all()),'shape':list(a.shape)}
                   for a,b in zip(actual,expected['outputs'])]
print('REFERENCE_LOADER_DIAGNOSTIC='+json.dumps(report,sort_keys=True),flush=True)
assert report['parameter_keys_equal'] and not report['bad_parameters'],report
assert report['buffer_keys_equal'] and all(b['equal'] for b in report['buffers'].values()),report
assert all(o['equal'] and o['finite'] for o in report['outputs']),report
assert not torch.cuda.is_initialized()
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,cwd=Path(__file__).resolve().parents[1],
                            capture_output=True,text=True,timeout=120)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'REFERENCE_LOADER_DIAGNOSTIC=' in result.stdout
