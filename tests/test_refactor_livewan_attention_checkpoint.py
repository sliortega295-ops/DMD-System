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


def test_c26_plan_is_one_factor_and_fail_closed(monkeypatch):
    import copy
    import json
    from pathlib import Path
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_attention_checkpoint import PLAN_CONTRACT
    from distillschedule.runtime.livewan_crossscale_worker import _critic_attention_checkpoint_enabled
    root=Path(__file__).resolve().parents[1]/'configs/distillschedule/plans'
    a=root/'livewan_wan13b_14b_sf_dmd_c25_continuous_b16_score_rmsnorm.json'
    b=root/'livewan_wan13b_14b_sf_dmd_c26_continuous_b16_critic_attention_checkpoint.json'
    da,db=json.loads(a.read_text()),json.loads(b.read_text())
    assert db['metadata'].pop('critic_attention_checkpoint')==PLAN_CONTRACT
    assert da==db
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    plan=load_plan(b).plan.to_dict()
    assert _critic_attention_checkpoint_enabled(plan,mode='pipeline')
    assert not _critic_attention_checkpoint_enabled(load_plan(a).plan.to_dict(),mode='pipeline')
    with pytest.raises(RuntimeError):
        _critic_attention_checkpoint_enabled(plan,mode='baseline')
    for key,value in [('preserve_rng_state',False),('cache_location','cpu'),('query_tokens',512)]:
        changed=copy.deepcopy(plan);changed['metadata']['critic_attention_checkpoint'][key]=value
        with pytest.raises(RuntimeError):
            _critic_attention_checkpoint_enabled(changed,mode='pipeline')


def test_runtime_sac_census_cannot_be_inferred_only_from_config():
    from distillschedule.runtime.livewan_attention_checkpoint import runtime_receipt_errors,PLAN_CONTRACT,INTERFACE_SHA256
    r={'name':PLAN_CONTRACT['name'],'source':{'interface_sha256':INTERFACE_SHA256},
       'execution_status':'PASS','production_installed':True,'self_attention_only':True,
       'contract_failures':0,'checkpoint_calls':300,'expected_checkpoint_calls':300,
       'original_fa_calls':300,'replay_fa_cache_reads':300,'cpu_offload':False,
       'allow_cache_entry_mutation':False}
    assert not runtime_receipt_errors(r,rank=8,expected_calls=300)
    for k,v in [('replay_fa_cache_reads',0),('checkpoint_calls',299),('cpu_offload',True),('contract_failures',1)]:
        assert runtime_receipt_errors(dict(r,**{k:v}),rank=8,expected_calls=300)
    assert runtime_receipt_errors(r,rank=0,expected_calls=300)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE','checkpoint_calls':0},rank=0,expected_calls=300)
    assert runtime_receipt_errors(r,rank=8,expected_calls=300,query_tokens=16560)
    assert not runtime_receipt_errors(dict(r,query_tokens=16560,key_tokens=16560),rank=8,
                                      expected_calls=300,query_tokens=16560)


def test_request_attention_policy_uses_actual_new_sequence_metadata(monkeypatch):
    torch=pytest.importorskip('torch')
    import torch.utils.checkpoint as checkpoint
    captured=[]
    monkeypatch.setattr(checkpoint,'create_selective_checkpoint_contexts',
                        lambda policy,**kw:captured.append(policy) or ('original','replay'))
    op=object();contexts=AttentionCheckpointContexts(SimpleNamespace(bfloat16='bf16'),op,{},
        self_attention_only=True,query_tokens=16560)
    assert contexts()==('original','replay')
    q,k,v,*tail=inputs()
    q.shape=(2,16560,12,128)
    policy=captured[0];ctx=SimpleNamespace(is_recompute=False)
    assert policy(ctx,op,q,k,v,*tail)==checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
    k.shape=v.shape=q.shape
    assert policy(ctx,op,q,k,v,*tail)==checkpoint.CheckpointPolicy.MUST_SAVE
    q.shape=(2,19320,12,128)
    with pytest.raises(RuntimeError,match='sequence shape'):
        policy(ctx,op,q,k,v,*tail)
