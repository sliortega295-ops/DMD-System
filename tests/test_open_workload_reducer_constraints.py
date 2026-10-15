"""Search must respect reducer lifetime as well as mathematical dependencies."""
from dataclasses import replace
import pytest
from distillschedule.workloads.program import Call, Program
from distillschedule.open_workload.replica_contract import ReplicaSemantics, execution_graph
from distillschedule.open_workload.replicated import ddp_order_admitted
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from test_open_workload_replica_contract import with_replica


def two_forwards_one_update():
    w=with_replica()
    # Two independent losses under a single parameter version, accumulated before
    # the update. They are mathematically reorderable, but not all DDP orders are.
    calls=[c for c in w.program.calls if c.id not in ('u0','u1')]
    old=next(c for c in w.program.calls if c.id=='u1')
    calls.append(replace(old,after=('b0',)))
    w.program=Program(tuple(calls),w.program.initial_versions,w.program.feeds,w.program.outputs)
    return w.validate()


def test_ddp_completion_edges_restrict_only_native_execution():
    w=two_forwards_one_update()
    logical=w.program.compile()
    graph=execution_graph(w.program,w.replica_semantics,'ddp_overlap')
    assert ('b0','p1') not in {(e.source,e.target) for e in logical.dependencies}
    assert ('b0','p1') in {(e.source,e.target) for e in graph.dependencies}
    assert logical.operations==graph.operations
    bad=('p0','p1','t0','t1','l0','l1','b0','b1','u1','__open_finish__')
    runner=StaticRoleRunner('open_runtime',{'call':lambda ctx:None,'barrier':lambda ctx:None},
                           versions=VersionLedger(w.program.initial_versions))
    runner.prepare(logical,order=bad)
    with pytest.raises(ValueError):runner.prepare(graph,order=bad)
    assert not ddp_order_admitted(w,bad)
    assert execution_graph(w.program,w.replica_semantics,'post_backward')==logical


def test_backward_cannot_commit_optimizer_before_reduction():
    w=with_replica()
    calls=[replace(c,reads=('grad','adam'),writes=('grad','adam')) if c.id=='b0' else c
           for c in w.program.calls]
    w.program=Program(tuple(calls),w.program.initial_versions,w.program.feeds,w.program.outputs)
    with pytest.raises(ValueError,match='separate calls'):w.validate()


def test_backend_constraints_are_used_in_shared_cost_solver(tmp_path):
    # Synthetic timing only tests search correctness, not actual speed.
    from distillschedule.search.cost_model import CostModel, OperationCost
    from distillschedule.search.plan_ir import BindingRule
    from distillschedule.search.service import BoundCandidate, select
    w=two_forwards_one_update();g=execution_graph(w.program,w.replica_semantics,'ddp_overlap')
    binding={'source_head':'1'*40,'hardware_fingerprint':'a'*64,'environment_fingerprint':'a'*64,
             'asset_fingerprint':'a'*64,'semantic_fingerprint':'a'*64,'precision_fingerprint':'a'*64}
    records=tuple(OperationCost(c.operation_id,(1,),0,{'fixture_only':True}) for c in g.operations)
    costs=CostModel(binding,records,scope='CPU_FIXTURE_ONLY')
    rules={c.operation_id:BindingRule(c.operation_id,{},c.operation_id,(0,),('gpu:0',),(),c.operation_id)
           for c in g.operations}
    result=select([BoundCandidate('ddp',{},g,rules,costs,(),(),{0:1000},'test')])
    order=result['ranked'][0]['order']
    assert order.index('b0')<order.index('p1')
