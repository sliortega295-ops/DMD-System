import inspect
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_attention_checkpoint import (
    AttentionCheckpointContexts, attention_metadata, discover_flash_forward)


def inputs():
    def tensor(shape):
        return SimpleNamespace(ndim=4, is_cuda=True, dtype="bf16",device="cuda:0",
            shape=shape, stride=lambda dim=None: 1 if dim == -1 else (shape[1]*1536,1536,128,1))
    return (tensor((2,19320,12,128)), tensor((2,512,12,128)), tensor((2,512,12,128)),
            0.0, 128**(-.5), False, -1, -1, 0.0, None, False)


def test_metadata_has_no_gpu_read():
    assert len(attention_metadata(SimpleNamespace(bfloat16="bf16"), inputs(), {})) == 3
    source = inspect.getsource(attention_metadata)
    assert all(token not in source for token in ('.cpu(', '.item(', '.tolist(', 'synchronize(', 'copy_('))


@pytest.mark.parametrize('index,value', [(3,.1),(4,.5),(5,True),(6,32),(7,32),(8,.2),(10,True)])
def test_math_or_dropout_change_rejected(index,value):
    args=list(inputs());args[index]=value
    with pytest.raises(RuntimeError):
        attention_metadata(SimpleNamespace(bfloat16="bf16"),tuple(args),{})


def test_device_precision_and_teacher_rejected():
    for attr,value in [('dtype','fp32'),('is_cuda',False),('shape',(2,19320,40,128))]:
        args=inputs();setattr(args[0],attr,value)
        with pytest.raises(RuntimeError):
            attention_metadata(SimpleNamespace(bfloat16="bf16"),args,{})


def test_contexts_are_per_checkpoint_and_mutation_protected():
    source=inspect.getsource(AttentionCheckpointContexts)
    assert 'allow_cache_entry_mutation=False' in source
    assert 'ledger: dict[str, Any]' in inspect.getsource(AttentionCheckpointContexts.__call__)
    assert 'CheckpointPolicy.MUST_SAVE' in source
    assert 'CheckpointPolicy.PREFER_RECOMPUTE' in source
    assert not any(s in source for s in ('.cpu(', 'manual_seed(', 'preserve_rng_state=False'))


def test_plan_only_has_no_cuda_import(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,str(root/'scripts/distillschedule/canary_livewan_attention_checkpoint.py'),
                      '--plan-only','--output',str(tmp_path/'canary.json')],check=True,capture_output=True,text=True)
    d=json.loads(p.stdout)
    assert d['world16']=='NOT_RUN' and not (tmp_path/'canary.json').exists()
