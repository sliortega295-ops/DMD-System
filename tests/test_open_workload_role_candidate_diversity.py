"""A candidate budget must cover distinct G/Score splits, not just prefix cuts."""
import pytest
from distillschedule.open_workload.role_program import RoleTask, AllocatedWorld, enumerate_layouts


def test_world16_budget_explores_balanced_and_asymmetric_pools_and_windows():
    task=RoleTask(8,(1,2,4,8))
    hw=AllocatedWorld((2**28,)*16,tuple([0]*8+[1]*8))
    result=enumerate_layouts(task,hw,generator_bytes=2000,score_bytes=2000,max_candidates=4)
    plans=result['candidates']
    assert len(plans)==4
    assert any(len(p.generator)==8 for p in plans), [p.to_dict() for p in plans]
    assert len({len(p.generator) for p in plans})>=3, [p.to_dict() for p in plans]
    assert {p.window for p in plans}=={1,4}
    assert result['enumeration_complete'] is False
    assert any(x['reason']=='SEARCH_BUDGET' for x in result['rejected'])
    assert tuple(p.to_dict() for p in plans)==tuple(p.to_dict() for p in enumerate_layouts(
        task,hw,generator_bytes=2000,score_bytes=2000,max_candidates=4)['candidates'])


def test_offload_and_resident_compete_in_budget_on_rank_pool():
    task=RoleTask(4,(1,2,4))
    hw=AllocatedWorld((2**26,)*16,host_capacities=(2**26,)*16)
    result=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=1000,
                             allow_offload=True,max_candidates=4)
    assert len(result['candidates'])==4
    assert {p.activation_policy for p in result['candidates']}=={'resident','cpu_offload_prefetch'}
    assert len({len(p.generator) for p in result['candidates']})>=2
    assert any(p.window==4 for p in result['candidates'])


def test_infeasible_balanced_pool_admits_only_real_legal_layouts():
    task=RoleTask(2,(2,16))
    # Rank1 can hold Score, not Generator; balanced 2G+2S is impossible.
    # Rank0 can produce and the three remaining ranks can score.
    hw=AllocatedWorld((10000,950,10000,10000))
    result=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=500,max_candidates=2)
    assert all(len(p.generator)==1 for p in result['candidates'])
    assert all(x['reason'] in {'ROLE_CAPACITY','SEARCH_BUDGET'} for x in result['rejected'])


def test_node_local_split_enters_small_world16_budget_without_forcing_winner():
    task=RoleTask(8,(1,2,4,8))
    hw=AllocatedWorld((2**28,)*16,tuple([0]*8+[1]*8))
    r=enumerate_layouts(task,hw,generator_bytes=2000,score_bytes=2000,max_candidates=8)
    plans=r['candidates']
    assert any(p.generator==(0,1,2,3,8,9,10,11) and p.score==(4,5,6,7,12,13,14,15)
               for p in plans), [p.to_dict() for p in plans]
    assert any(p.generator==tuple(range(8)) for p in plans)
    assert len(plans)==8
    assert tuple(p.to_dict() for p in plans)==tuple(p.to_dict() for p in enumerate_layouts(
        task,hw,generator_bytes=2000,score_bytes=2000,max_candidates=8)['candidates'])
