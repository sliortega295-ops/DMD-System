"""Asymmetric native FIFO prediction: budgets/precedence, then real Gloo checks.

Synthetic observations test search mechanics, not A100/H100 performance.
"""
from dataclasses import replace
import pytest

from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts
from distillschedule.open_workload.role_prediction import (
    admission, graph_for, projection, generator_order, select_anchor_targets,
)
from distillschedule.search.plan_ir import ancestors
from distillschedule.workloads.best_known.registry import fingerprint


@pytest.mark.parametrize('world,cut,n',[(3,1,5),(3,2,5),(4,1,7),(4,3,7),(5,2,7)])
def test_asymmetric_graph_obeys_all_receivers_not_only_producers(world,cut,n):
    task=RoleTask(n,(2,4),score_updates=1)
    hw=AllocatedWorld((10**7,)*world)
    for window in (1,2,3,4):
        layout=PoolLayout(tuple(range(cut)),tuple(range(cut,world)),window)
        graph,specs=graph_for(task,layout,hw)
        before=ancestors(graph)
        assert len(graph.operations)==5*n+2
        for score in layout.score:
            indices=[i for i in range(n) if layout.owners(i)[1]==score]
            for left,right in zip(indices,indices[1:]):
                assert f'r{left}' in before[f'a{right}']
        for producer in layout.generator:
            order=generator_order([i for i in range(n) if layout.owners(i)[0]==producer],window)
            assert all(a in before[b] for a,b in zip(order,order[1:]))
        assert all(f'b{i}' in before['commit_g'] for i in range(n))
        assert 'commit_g' in before['updates_s']
        for i in range(n):
            g,s=layout.owners(i)
            assert specs[f'a{i}'][0]==f'link:{g}:{s}'
            assert specs[f'r{i}'][0]==f'link:{s}:{g}'


def test_receiver_dependency_crosses_different_producers():
    task=RoleTask(5,(2,4));hw=AllocatedWorld((10**7,)*3)
    graph,_=graph_for(task,PoolLayout((0,1),(2,),2),hw)
    before=ancestors(graph)
    assert 'r0' in before['a1']  # Score 2 must not consume G1's input before G0's.
    assert 'r1' in before['a2']


@pytest.mark.parametrize('max_candidates,native_runs,samples',[(4,3,1),(8,8,1),(8,10,2),(1,3,1)])
def test_bounded_anchors_cover_splits_and_leave_confirmation(max_candidates,native_runs,samples):
    task=RoleTask(24,(2,4));hw=AllocatedWorld((10**7,)*16,(0,)*8+(1,)*8)
    inventory=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=1000,max_candidates=1024)
    selected,omitted=select_anchor_targets(task,hw,inventory['candidates'],
        max_candidates=max_candidates,samples=samples,max_native_runs=native_runs)
    groups={fingerprint(projection(c)) for c in selected}
    assert len(selected)<=max_candidates
    assert len(groups)*(samples+1)+1<=native_runs
    assert omitted
    assert select_anchor_targets(task,hw,inventory['candidates'],max_candidates=max_candidates,
        samples=samples,max_native_runs=native_runs)[0]==selected
    if max_candidates>1:
        assert any(c.window>1 for c in selected)
    if native_runs>=8:
        assert len({len(c.generator) for c in selected})>1


def test_equivalent_windows_are_not_counted_as_new_unseen_schedules():
    task=RoleTask(4,(2,4));hw=AllocatedWorld((10**7,)*4)
    layouts=[PoolLayout((0,1),(2,3),w) for w in range(1,5)]
    selected,omitted=select_anchor_targets(task,hw,layouts,max_candidates=4,samples=1,max_native_runs=3)
    assert [c.window for c in selected]==[1,2]
    assert {r['reason'] for r in omitted}=={'EQUIVALENT_NATIVE_FIFO'}


def test_inactive_ranks_and_unfunded_anchor_fail_before_execution():
    task=RoleTask(2,(2,4));hw=AllocatedWorld((10**7,)*4)
    with pytest.raises(ValueError,match='active replicated resident'):
        admission(task,PoolLayout((0,),(1,2,3),1),hw)
    with pytest.raises(ValueError,match='confirmation exceed budget'):
        select_anchor_targets(RoleTask(4,(2,4)),hw,[PoolLayout((0,1),(2,3),1)],
                              max_candidates=4,samples=2,max_native_runs=3)
