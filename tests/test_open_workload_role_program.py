"""New role-gradient tasks use source-independent graphs and hardware-bound layouts."""
import pytest
from distillschedule.open_workload.role_program import RoleTask,AllocatedWorld,PoolLayout,enumerate_layouts


def test_graph_preserves_original_gradient_and_commit_versions():
    for n,k in ((1,0),(2,1),(3,3),(5,2)):
        task=RoleTask(n,(2,5,8),score_updates=k,parameters={'freq':n})
        graph=task.program().compile()
        assert len(graph.operations)==5*n+k+2
        assert graph.metadata['final_versions']['generator']==1
        assert graph.metadata['final_versions']['score']==k
        assert graph.metadata['final_versions']['gradient_g']==n+1
        deps={(e.source,e.target) for e in graph.dependencies}
        assert all((f'b{i}','commit_g') in deps for i in range(n))
        assert all((f'r{i}',f'b{i}') in deps for i in range(n))
        assert all((f'g{i}',f'f{i}') in deps for i in range(n))
        assert graph.topological_order()[-1]=='__open_finish__'


def test_graph_changes_as_data_or_frequency_changes():
    a=RoleTask(2,(1,8,8));b=RoleTask(3,(1,8,8));c=RoleTask(2,(1,8,9))
    assert len({a.digest,b.digest,c.digest})==3
    assert len({a.program().digest,b.program().digest,c.program().digest})==3
    with pytest.raises(ValueError):RoleTask(0,(1,8,8))
    with pytest.raises(ValueError):RoleTask(2,(1,0,8))


def test_hardware_enumerates_nonhardcoded_split_and_pipeline_window():
    task=RoleTask(4,(2,16));h=AllocatedWorld((2000,)*8,(0,0,0,0,1,1,1,1))
    r=enumerate_layouts(task,h,generator_bytes=500,score_bytes=700,max_candidates=100)
    assert {tuple(range(i)) for i in range(1,8)} <= {c.generator for c in r['candidates']}
    assert (0,1,4,5) in {c.generator for c in r['candidates']}
    assert (tuple(range(4)),tuple(range(4,8)),2) in [(c.generator,c.score,c.window) for c in r['candidates']]
    assert all(c.validate(task,h) for c in r['candidates'])
    assert PoolLayout((0,1),(2,3),2).owners(3)==(1,3)


def test_capacity_rejection_and_bounded_enumeration_are_truthful():
    t=RoleTask(5,(16,16))
    h=AllocatedWorld((5000,)*4)
    r=enumerate_layouts(t,h,generator_bytes=200,score_bytes=500,max_candidates=3)
    assert 0<len(r['candidates'])<=3
    assert r['enumeration_complete'] is False
    assert any(x['reason']=='ROLE_CAPACITY' for x in r['rejected'])
    assert any(x['reason']=='SEARCH_BUDGET' for x in r['rejected'])
    with pytest.raises(ValueError):enumerate_layouts(t,AllocatedWorld((128,128)),generator_bytes=400,score_bytes=400)


def test_refuse_overlap_unassigned_gpu_and_invalid_transport():
    t=RoleTask(2,(2,8));h=AllocatedWorld((5000,)*4)
    for x in [PoolLayout((0,1),(1,2,3),1),PoolLayout((0,),(1,2),1),
              PoolLayout((),(0,1,2,3),1),PoolLayout((0,1),(2,3),3)]:
        with pytest.raises(ValueError):x.validate(t,h)


def test_host_budget_gates_offload_but_never_forces_it():
    t=RoleTask(3,(2,4));h=AllocatedWorld((4000,4000),host_capacities=(4096,4096))
    report=enumerate_layouts(t,h,generator_bytes=600,score_bytes=500,
                             allow_offload=True,max_candidates=8)
    assert {c.activation_policy for c in report['candidates']}=={'resident','cpu_offload_prefetch'}
    for c in report['candidates']:
        assert c.validate(t,h)
        assert PoolLayout.from_dict(c.to_dict())==c
    no_host=AllocatedWorld((4000,4000))
    report=enumerate_layouts(t,no_host,generator_bytes=600,score_bytes=500,
                             allow_offload=True,max_candidates=8)
    assert all(c.activation_policy=='resident' for c in report['candidates'])
    assert any(r['reason']=='HOST_CAPACITY' for r in report['rejected'])
