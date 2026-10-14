from dataclasses import replace
import pytest
import torch
from distillschedule.open_workload.replica_contract import ReplicaSemantics
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from test_open_workload_execution import build


def with_replica():
    w = build(n=2)
    w.replica_semantics = ReplicaSemantics({'forward':'learner'}, ('b0','b1'), 8)
    return w.validate()


def test_replica_is_semantics_not_candidate_selection():
    w=with_replica()
    assert w.identity['replica_semantics']['objective']=='equal_rank_mean'
    assert candidates(w) == candidates(build(n=2))
    assert 'backend' not in w.replica_semantics.to_dict()


@pytest.mark.parametrize('changes', [
    {'objective':'sum'}, {'local_batch_size':0}, {'local_batch_size':True},
    {'module_bindings':{'forward':'guide'}}, {'backward_calls':('u0',)},
    {'backward_calls':('b0','b0')}])
def test_bad_replica_contract_rejected(changes):
    w=with_replica(); w.replica_semantics=replace(w.replica_semantics,**changes)
    with pytest.raises(ValueError): w.validate()


def test_lifecycle_is_connected_to_real_calls_without_changing_reference():
    w=with_replica(); seen=[]
    class Hooks:
        def prepare(self,s): seen.append('prepare')
        def before_run(self,s,order): seen.append(('order',tuple(order)))
        def before_call(self,s,c): seen.append(('before',c.id))
        def after_call(self,s,c): seen.append(('after',c.id))
        def finish(self,s): seen.append('finish')
    impl=candidates(w)['candidates'][0]
    ref=Session(w,impl);a=ref.run()
    s=Session(w,impl,lifecycle=Hooks());b=s.run()
    assert compare(ref.result_state(a),s.result_state(b))['status']=='PASS'
    assert seen[0]=='prepare' and seen[-1]=='finish'
    assert len([x for x in seen if isinstance(x,tuple) and x[0]=='after'])==len(w.program.calls)
