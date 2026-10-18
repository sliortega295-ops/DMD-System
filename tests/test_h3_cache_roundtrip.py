"""CPU/SYNTHETIC_TEST_ONLY production producer/consumer; not GPU evidence."""
from copy import deepcopy
from pathlib import Path
import ast
import pytest
import torch
from distillschedule.backends.training.minimax_h3 import cache_roundtrip_probe as p


def cpu_digest(value):
    return torch.tensor(p.cpu_fingerprint(torch,value),dtype=torch.int64)


def emit(probe, *, mutate=False):
    for graph in range(6):
        values={}
        for block in range(50):
            out=torch.full((1,3,2,4),graph+block/64,dtype=torch.bfloat16)
            lse=torch.full((1,2,3),graph-block/64,dtype=torch.float32)
            values[block]=(out,lse)
            probe.capture('original',block,values[block])
        for block in reversed(range(50)):
            copied=tuple(x.clone()for x in values[block])
            if mutate and graph==2 and block==10:copied[0][0,0,0,0]+=1
            probe.capture('replay',block,copied)
            probe.capture('backward',block,())


def test_real_producer_full_census_matches_and_never_admits_numerics(monkeypatch):
    audit=p.RoundTripProbe(torch,digest=cpu_digest);audit.selftest=dict(status='SYNTHETIC_TEST_ONLY_NO_GPU')
    state=torch.get_rng_state().clone()
    for cycle in (1,2):
        audit.begin(cycle)
        with monkeypatch.context()as mp:
            mp.setattr(torch.Tensor,'cpu',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unsafe host read')))
            emit(audit)
            with pytest.raises(ValueError):audit.finish_cycle(outside_cycle=False)
        row=audit.finish_cycle(outside_cycle=True)
        assert row['pairs']==600 and row['mismatches']==0
    value=audit.receipt()
    assert value['pairs']==1200 and value['status']=='MATCHED_FINGERPRINTS_NOT_NUMERICAL_ADMISSION'
    assert not value['formal_timing_eligible'] and not value['numerical_admission']
    assert torch.equal(state,torch.get_rng_state())and not torch.cuda.is_initialized()


def test_payload_change_not_hidden_by_shape_or_finite():
    audit=p.RoundTripProbe(torch,digest=cpu_digest);audit.begin(1);emit(audit,mutate=True)
    result=audit.finish_cycle(outside_cycle=True)
    assert result['mismatches']==1
    bad=next(x for x in result['rows']if not x['fingerprint_match'])
    assert (bad['graph'],bad['block'],bad['tensor'])==(2,10,'out')
    with pytest.raises(ValueError):audit.receipt()


def test_incomplete_reordered_or_extra_graph_rejected():
    audit=p.RoundTripProbe(torch,digest=cpu_digest);audit.begin(1)
    v=(torch.zeros(1,2,3,dtype=torch.bfloat16),torch.zeros(1,2,3))
    with pytest.raises(ValueError):audit.capture('replay',49,v)
    audit.capture('original',0,v)
    with pytest.raises(ValueError):audit.capture('original',0,v)
    with pytest.raises(ValueError):audit.capture('replay',49,v)
    with pytest.raises(ValueError):audit.finish_cycle(outside_cycle=True)
    audit=p.RoundTripProbe(torch,digest=cpu_digest);audit.begin(1);emit(audit)
    with pytest.raises(ValueError):audit.capture('original',0,v)


def test_integer_fingerprint_detects_order_bits_and_preserves_logical_layout():
    assert p.reference_words([0,1,2,65535])!=p.reference_words([1,0,2,65535])
    for dtype in (torch.bfloat16,torch.float32):
        x=torch.arange(24,dtype=torch.float32).to(dtype).view(1,2,3,4).transpose(1,2)
        assert p.cpu_fingerprint(torch,x)==p.cpu_fingerprint(torch,x.contiguous())
        assert p.cpu_fingerprint(torch,torch.tensor([0.],dtype=dtype))!=p.cpu_fingerprint(torch,torch.tensor([-0.],dtype=dtype))
    with pytest.raises(ValueError):p.reference_words([-1])


def base_plan():
    base=dict(integration_head='a'*40,profile_id='minimax_h3_grid1000_best',mode='correctness',native_mode='correctness',run_id='SYNTHETIC_TEST_ONLY',asset_config=dict(run_root='/unread'),search_candidate=dict(id='joint-h3v1-d1-n128'))
    env=dict(EXPECTED_INTEGRATION_HEAD='a'*40,DISTILLSCHEDULE_SEARCH_CANDIDATE_ID='joint-h3v1-d1-n128',DISTILLSCHEDULE_RUN_KIND='correctness',H3_PLANNER_TWO_CYCLE='1',H3_PLANNER_CLOCK='host_fenced',DISTILLSCHEDULE_RUN_ID=base['run_id'],DISTILLSCHEDULE_RUN_ROOT='/unread/'+base['run_id'])
    return dict(base,nodes=[dict(node_rank=i,environment=dict(env))for i in (0,1)])


def test_plan_binding_default_unchanged_and_all_orphan_or_drift_rejected():
    from distillschedule.backends.training.minimax_h3 import divergence_probe as d
    base=base_plan();control=d.bind_plan(base,'d1A')
    assert 'cache_roundtrip'not in control['h3_divergence']
    child=d.bind_plan(base,'d1A',cache_roundtrip=True)
    assert child['h3_divergence']['cache_roundtrip']==p.source_contract()and d.is_diagnostic_plan(child)
    assert all(n['environment']['H3_PLANNER_CACHE_ROUNDTRIP']=='1'for n in child['nodes'])
    for mutate in (lambda x:x['h3_divergence']['cache_roundtrip'].__setitem__('cycles',1),
        lambda x:x['nodes'][1]['environment'].__setitem__('H3_PLANNER_CACHE_ROUNDTRIP','0'),
        lambda x:x['h3_divergence']['cache_roundtrip']['source_sha256'].__setitem__('cache_roundtrip_kernels.py','0'*64),
        lambda x:x['h3_divergence'].__setitem__('raw_repeat',{})):
        bad=deepcopy(child);mutate(bad)
        with pytest.raises(ValueError):d.is_diagnostic_plan(bad)
    orphan=deepcopy(base);orphan['nodes'][0]['environment']['H3_PLANNER_CACHE_ROUNDTRIP']='1'
    with pytest.raises(ValueError):d.is_diagnostic_plan(orphan)
    with pytest.raises(ValueError):d.bind_plan(base,'d1A',raw_repeat=True,cache_roundtrip=True)
    with pytest.raises(ValueError):d.bind_plan(base,'d1A',cache_roundtrip=1)
    assert not torch.cuda.is_initialized()


def test_gpu_hash_wrapper_has_no_host_tensor_reads_or_rng_calls():
    path=Path(p.__file__).with_name('cache_roundtrip_kernels.py')
    tree=ast.parse(path.read_text())
    banned={'cpu','item','tolist','numpy','rand','randn','manual_seed','synchronize'}
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in banned for n in ast.walk(tree))
    assert set(p.source_contract()['source_sha256'])=={'cache_roundtrip_probe.py','cache_roundtrip_kernels.py'}
