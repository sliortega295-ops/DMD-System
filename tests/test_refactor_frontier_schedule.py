"""Independent small DAG enumeration and event-state scheduling regressions."""
from dataclasses import replace
import itertools
import random

import pytest

from distillschedule.core.graph import Operation,Dependency,ExecutionGraph
from distillschedule.core.enums import OperationKind as K,GradientMode as G
from distillschedule.search.cost_model import CostModel,OperationCost
from distillschedule.search.plan_ir import BindingRule,StateAllocation,Lifetime
from distillschedule.search.frontier_schedule import schedule_frontier
from test_refactor_model_planning import BINDING


def make(durations,pools,edges):
    ops=tuple(Operation(str(i),0,'r',K.GENERATOR_ROLLOUT,G.NO_GRAD)
              for i in range(len(durations)))
    graph=ExecutionGraph(ops,tuple(Dependency(str(a),str(b)) for a,b in edges),{'initial_versions':{}})
    bindings={str(i):BindingRule(str(i),{},str(i),(pool,),(f'gpu:{pool}',)) for i,pool in enumerate(pools)}
    costs=CostModel(BINDING,tuple(OperationCost(str(i),(v,),0,{'kind':'CPU_FIXTURE_ONLY'})
                                 for i,v in enumerate(durations)),scope='CPU_FIXTURE_ONLY')
    return graph,bindings,costs


def serial_order_oracle(durations,pools,edges):
    # Independently enumerate every topological task order and apply resource
    # append scheduling. For these small resource-only, nonpreemptive fixtures,
    # this includes the start order of an optimal event schedule.
    best=float('inf')
    for order in itertools.permutations(range(len(durations))):
        pos={k:i for i,k in enumerate(order)}
        if any(pos[a]>=pos[b] for a,b in edges):continue
        ends={};last={}
        for i in order:
            start=max([ends[a] for a,b in edges if b==i]+[last.get(pools[i],0)])
            ends[i]=start+durations[i];last[pools[i]]=ends[i]
        best=min(best,max(ends.values()))
    return best


def test_optimal_schedule_must_intentionally_wait_with_ready_work():
    g,b,c=make([1,10,1,10],[0,1,1,0],[(0,2),(2,3)])
    r=schedule_frontier(g,b,c,memory_limits={0:100,1:100})
    assert r['predicted_makespan_ns']==12  # greedy non-delay is 21
    by={v['operation']:v for v in r['timeline']}
    assert by['1']['start_ns']>=2
    assert r['search_diagnostics']['wait_transitions']>0
    assert r['search_diagnostics']['optimality']=='FIXED_BINDING_EVENT_MODEL_OPTIMAL'


@pytest.mark.parametrize('seed',range(20))
def test_random_small_dags_match_independent_order_oracle(seed):
    rng=random.Random(seed);n=5
    ds=[rng.randrange(1,8) for _ in range(n)];pools=[rng.randrange(2) for _ in range(n)]
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.25]
    g,b,c=make(ds,pools,edges)
    r=schedule_frontier(g,b,c,memory_limits={0:100,1:100})
    assert r['predicted_makespan_ns']==serial_order_oracle(ds,pools,edges)


def test_shared_adapter_mutex_removes_cross_gpu_concurrency_and_counts_storage_once():
    g,b,c=make([4,6],[0,1],[])
    plain=schedule_frontier(g,b,c,memory_limits={0:100,1:100})
    g=replace(g,operations=tuple(replace(o,adapter_mode='role') for o in g.operations))
    state=StateAllocation('base',(0,1),10,('r',),True,'same',True)
    b={k:replace(v,states=('base',)) for k,v in b.items()}
    shared=schedule_frontier(g,b,c,states=(state,),memory_limits={0:100,1:100})
    assert plain['predicted_makespan_ns']==6 and shared['predicted_makespan_ns']==10
    assert all(v['static_bytes']==10 for v in shared['memory'])


def test_retained_storage_lives_through_last_consumer_completion():
    # Producer0 -> consumer1. Independent task2 cannot overlap consumer1 when
    # 20 bytes of saved state + 20 bytes task2 workspace exceeds the limit.
    g,b,c=make([1,5,2],[0,0,1],[(0,1)])
    c=replace(c,records=(*c.records[:2],replace(c.records[2],workspace_bytes_per_rank=20)))
    life=Lifetime('saved','0',('1',),(1,),20,allocation_point='producer_exit')
    r=schedule_frontier(g,b,c,lifetimes=(life,),memory_limits={0:100,1:30})
    assert r['status']=='PASS_CPU_SCHEDULE'
    ts={x['operation']:x for x in r['timeline']}
    assert ts['2']['end_ns']<=ts['0']['end_ns'] or ts['2']['start_ns']>=ts['1']['end_ns']
    assert max(v['peak_bytes'] for v in r['memory'])<=30


def test_equal_size_different_producers_do_not_merge_lifetimes():
    g,b,c=make([1,1,4,4],[0,0,0,0],[(0,2),(1,3)])
    lives=(Lifetime('a','0',('2',),(0,),20,'producer_exit'),
           Lifetime('b','1',('3',),(0,),20,'producer_exit'))
    r=schedule_frontier(g,b,c,lifetimes=lives,memory_limits={0:25})
    assert r['predicted_makespan_ns']==10
    pos={k:i for i,k in enumerate(r['order'])}
    assert pos['2']<pos['1'] or pos['3']<pos['0']
    assert r['search_diagnostics']['capacity_pruned']>0


def test_cpu_beam_and_budget_never_imply_exactness():
    g,b,c=make([1,10,1,10],[0,1,1,0],[(0,2),(2,3)])
    for kw in ({'beam_width':1},{'max_states':1}):
        r=schedule_frontier(g,b,c,memory_limits={0:100,1:100},**kw)
        assert r['search_diagnostics']['optimality']=='BEST_FOUND_ONLY'
        assert r['gpu_execution']=='NOT_RUN'


def test_zero_duration_actions_and_same_boundary_release():
    g,b,c=make([0,2,0],[0,0,0],[(0,1),(1,2)])
    life=Lifetime('x','0',('1',),(0,),10,'producer_exit')
    r=schedule_frontier(g,b,c,lifetimes=(life,),memory_limits={0:10})
    assert r['predicted_makespan_ns']==2 and r['order']==['0','1','2']


def test_no_memory_solution_is_not_reported_as_cuda_oom():
    g,b,c=make([2],[0],[])
    c=replace(c,records=(replace(c.records[0],workspace_bytes_per_rank=11),))
    r=schedule_frontier(g,b,c,memory_limits={0:10})
    assert r['status']=='NO_FEASIBLE_SCHEDULE' and r['predicted_makespan_ns'] is None


def test_unproven_gradients_and_incomplete_bindings_rejected():
    g,b,c=make([1],[0],[])
    with pytest.raises(ValueError):schedule_frontier(g,{},c,memory_limits={0:10})
    with pytest.raises(ValueError):schedule_frontier(g,{'0':replace(b['0'],resources=('host',))},c,memory_limits={0:10})
    with pytest.raises(ValueError):schedule_frontier(g,{'0':replace(b['0'],when={'x':'y'})},c,memory_limits={0:10})
    with pytest.raises(ValueError):schedule_frontier(replace(g,operations=(replace(g.operations[0],condition='random'),)),b,c,memory_limits={0:10})


def test_optimality_is_event_family_not_unrestricted_delayed_start():
    # Event-family optimum is 16. An unrestricted schedule can start B at t=5,
    # finish it exactly when A releases its saved state at t=10, and end at11.
    # The solver must not advertise unrestricted RCPSP/global GPU optimality.
    g,b,c=make([0,10,5,1],[0,0,1,1],[(0,1),(2,3)])
    lives=(Lifetime('a','0',('1',),(1,),10,'producer_exit'),
           Lifetime('b','2',('3',),(1,),10,'producer_exit'))
    r=schedule_frontier(g,b,c,lifetimes=lives,memory_limits={0:100,1:10})
    assert r['predicted_makespan_ns']==16
    assert r['search_diagnostics']['dispatch_model']=='ZERO_OR_COMPLETION_EVENTS_ONLY'
    assert r['search_diagnostics']['native_optimality']=='NOT_CLAIMED'
