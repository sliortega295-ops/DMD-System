"""SYNTHETIC_TEST_ONLY native metadata barrier, default off and failure tests."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import pytest
import torch

R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manual_barrier',R/'scripts/native_metadata_barrier.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_disabled_wrapper_preserves_original_function_identity_and_rng():
    fn=lambda *args,**kwargs:None
    before=torch.get_rng_state().clone()
    c=m.NativeMetadataBarrier([0,34876,34880])
    assert c.wrap(fn)is fn
    for cycle in(1,2):
        row=c.finish_cycle(cycle)
        assert not row['enabled'] and row['effective_backend_host_value_reads']==0
    assert torch.equal(before,torch.get_rng_state())


def test_actual_tolist_before_original_call_and_two_cycle_read_census():
    events=[]
    class Cu:
        def tolist(self):
            events.append('native_tolist')
            return [0,34876,34880]
    cu=Cu();q,k,v=object(),object(),object()
    def fn(a,b,c,d,e,*args,**kwargs):
        assert(a,b,c,d,e)==(q,k,v,cu,cu)
        assert events[-1]=='native_tolist'
        events.append('original_FA3')
        return q
    c=m.NativeMetadataBarrier([0,34876,34880],enabled=True)
    wrapped=c.wrap(fn)
    for cycle in(1,2):
        for i in range(350):assert wrapped(q,k,v,cu,cu,34876,34876,causal=False)is q
        row=c.finish_cycle(cycle)
        assert row['actual_native_metadata_reads']==350 and not row['tensor_math_changed']
        assert row['original_FA3_backend_unchanged']
    assert events==['native_tolist','original_FA3']*700


@pytest.mark.parametrize('failure',['wrong_cu_identity','wrong_cu_value','too_few_reads'])
def test_changed_contract_or_incomplete_execution_fails(failure):
    cu=torch.tensor([0,34876,34880],dtype=torch.int32)
    c=m.NativeMetadataBarrier([0,34876,34880],enabled=True)
    wrapped=c.wrap(lambda *args,**kwargs:None)
    if failure=='wrong_cu_identity':
        with pytest.raises(ValueError,match='IDENTITY'):wrapped(None,None,None,cu,cu.clone())
    elif failure=='wrong_cu_value':
        cu[1]=34875
        with pytest.raises(ValueError,match='VALUE'):wrapped(None,None,None,cu,cu)
    else:
        wrapped(None,None,None,cu,cu)
    with pytest.raises(ValueError,match='CENSUS'):c.finish_cycle(1)


def test_real_session_composition_only_wraps_optin_backend():
    cu=torch.tensor([0,34876,34880],dtype=torch.int32)
    f=lambda *args,**kwargs:'unchanged'
    policy=SimpleNamespace(bounds=cu.tolist(),backend=SimpleNamespace(flash_attn_varlen_func=f))
    session=SimpleNamespace(manual_attention_policy=policy,manual_state_policy=None)
    c=m.install_on_session(session,enabled=False)
    assert policy.backend.flash_attn_varlen_func is f
    assert c.finish_cycle(1)['effective_backend_host_value_reads']==0
    with pytest.raises(ValueError):
        m.install_on_session(SimpleNamespace(manual_attention_policy=policy,manual_state_policy=object()),enabled=True)
