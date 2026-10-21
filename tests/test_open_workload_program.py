"""Unregistered workload semantics; no model-specific planner or fake timing."""
import pytest
from distillschedule.workloads.program import Call, Program
from distillschedule.search.plan_ir import validate_versions


def make(n):
    calls = [Call('s', 'predict', ('x',), ('student',), grad='grad', replay_safe=True)]
    calls += [Call('t', 'teacher', ('x',), ('teacher',)),
              Call('loss', 'loss', ('s','t'), grad='grad'),
              Call('bw', 'backward', ('loss',), ('grad',), ('grad',), grad='grad', backward_of=('loss',)),
              Call('update', 'step', ('bw',), ('student','grad'), ('student','grad'))]
    for i in range(n):
        calls += [Call(f'read{i}', 'score', ('x',), ('student',))]
    return Program(tuple(calls), {'student':0,'teacher':0,'grad':0}, ('x',), (f'read{n-1}',), {'n':n})


@pytest.mark.parametrize('n',[1,2,5,8])
def test_arbitrary_frequency_and_roundtrip(n):
    p=make(n); g=p.compile(); validate_versions(g)
    assert Program.from_dict(p.to_dict()).digest==p.digest
    bw=next(o for o in g.operations if o.operation_id=='bw')
    assert bw.required_versions['student']==0
    assert {'s','t','loss'}<=set(bw.metadata['inputs'])
    assert g.metadata['final_versions']=={'student':1,'teacher':0,'grad':2}
    assert len(g.operations)==6+n


def test_late_backward_cannot_read_mutated_parameters():
    with pytest.raises(ValueError,match='version changed'):
        Program((Call('f','f',reads=('w',),grad='grad'),
                 Call('u','u',reads=('w',),writes=('w',)),
                 Call('b','b',inputs=('f',),grad='grad',backward_of=('f',))),{'w':0})


def test_independent_calls_are_not_forced_to_source_order():
    g=Program((Call('a','f',reads=('a',)),Call('b','f',reads=('b',))),{'a':0,'b':0}).compile()
    assert all(e.target=="__open_finish__" for e in g.dependencies)


def test_raw_and_war_added_automatically():
    g=Program((Call('r','f',reads=('w',)),Call('u','f',reads=('w',),writes=('w',)),
               Call('r2','f',reads=('w',))),{'w':0}).compile()
    assert {('r','u'),('u','r2')}<={(e.source,e.target) for e in g.dependencies}


@pytest.mark.parametrize('bad',[
    {'id':'a','binding':'f','replay_safe':True},
    {'id':'a','binding':'f','grad':'grad','replay_safe':True,'reads':('w',),'writes':('w',)},
    {'id':'a','binding':'f','backward_of':('unknown',)},
])
def test_illegal_replay_or_backward_rejected(bad):
    with pytest.raises(ValueError): Call(**bad)


def test_no_name_whitelist_or_fixed_five_contract():
    for name in ('unseen_model_2026','my_loss','totally_new_teacher'):
        p=Program((Call('work',name,('input',)),),{},('input',),('work',))
        assert p.compile().operations[0].metadata['binding']==name
