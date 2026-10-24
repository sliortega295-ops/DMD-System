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


def test_private_methods_are_post_backward_only_and_defaults_preserve_identity():
    from distillschedule.open_workload.replica_contract import execution_graph
    from distillschedule.open_workload.replicated import inventory, ddp_order_admitted
    from distillschedule.open_workload.sharded import rejection
    from distillschedule.open_workload.mixed_layout import domains
    w = with_replica()
    legacy = w.replica_semantics.to_dict()
    assert set(legacy) == {'module_bindings', 'backward_calls', 'local_batch_size', 'objective'}
    w.replica_semantics = replace(w.replica_semantics, forward_interface='private_methods',
                                 random_streams='rank_independent_v1')
    w.validate()
    assert w.identity['replica_semantics']['random_streams'] == 'rank_independent_v1'
    assert not ddp_order_admitted(w, tuple(c.id for c in w.program.calls))
    assert rejection(w) == 'PRIVATE_METHODS_REQUIRE_POST_BACKWARD'
    assert set(domains(w)['learner']) == {'post_backward'}
    choices = inventory(w, limit=100)
    assert {o['reduction'] for o in choices['candidates']} == {'post_backward'}
    assert 'ddp_async_mean' not in choices['capabilities']
    assert execution_graph(w.program, w.replica_semantics, 'post_backward') == w.program.compile()
    with pytest.raises(ValueError, match='private.*post.backward'):
        execution_graph(w.program, w.replica_semantics, 'ddp_overlap')
    for field in ('forward_interface', 'random_streams'):
        wrong = replace(w.replica_semantics, **{field: 'guessed'})
        with pytest.raises(ValueError):
            replace(w, replica_semantics=wrong).validate()


def test_sharded_capability_without_replica_semantics_fails_closed():
    from distillschedule.open_workload.sharded import rejection
    assert rejection(build(n=2)) == 'REPLICA_SEMANTICS_REQUIRED'


def test_rank_random_namespace_is_reproducible_and_does_not_advance_outer_rng():
    import random
    from distillschedule.open_workload.torch_backend import random_scope
    from distillschedule.workloads.best_known.registry import fingerprint
    before = torch.get_rng_state().clone(), random.getstate()
    derived = int(fingerprint({'seed': 43, 'logical_call': 'call'})[:16], 16) % (2**63)
    expected = torch.randn(5, generator=torch.Generator().manual_seed(derived))
    with random_scope(43, 'call', torch.device('cpu')):
        assert torch.equal(expected, torch.randn(5)), 'legacy logical-call bytes changed'
    rows = []
    for rank in (0, 1, 0):
        namespace = dict(schema='rank_independent_v1', rank=rank, world_size=2)
        with random_scope(43, 'call', torch.device('cpu'), namespace=namespace):
            rows.append((torch.randn(5), random.random()))
    assert torch.equal(rows[0][0], rows[2][0]) and rows[0][1] == rows[2][1]
    assert not torch.equal(rows[0][0], rows[1][0]) and rows[0][1] != rows[1][1]
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
