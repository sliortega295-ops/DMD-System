from dataclasses import replace
import pytest
from test_program_contracts import program
from distillschedule.programs.capabilities import CapabilityRegistry, Capability, ExecutableOption
from distillschedule.programs.contracts import CallContract
from distillschedule.search.plan_ir import StateAllocation


def option(key="a",**kw):
    return ExecutableOption(key,(0,),("gpu:0","executor:serial"),lambda ctx:None,lambda:None,{"source":"test"},**kw)


def registry():
    r=CapabilityRegistry()
    r.register(Capability("ref","user.function",{},("no_grad","grad"),lambda op,c:(option(),)))
    return r


def test_property_matching_not_workload_name():
    r=registry();r.register(Capability("fast","user.function",{"dtype":"bf16"},("no_grad",),lambda op,c:(option("b"),)))
    p=program();m=r.match(p)
    assert len(m.options['produce'])==1
    p=replace(p,calls={**p.calls,"produce":CallContract("user.function",{"dtype":"bf16"})})
    assert len(r.match(p).options['produce'])==2
    assert r.match(replace(p,name="unknown-model")).digest==r.match(p).digest
    assert all(len(v)==1 for k,v in r.match(p).options.items() if k!='produce')


def test_unknown_operator_no_implicit_fallback():
    p=program();p=replace(p,calls={**p.calls,"produce":CallContract("new.unsupported")})
    with pytest.raises(ValueError,match="no executable capability"):registry().match(p)


def test_setting_and_physical_alias_conflicts_pruned():
    from distillschedule.programs.capabilities import MatchedProgram
    p=program(1)
    a=option("cpu0",settings={"model-layout":"0"})
    b=option("cpu1",settings={"model-layout":"1"})
    m=MatchedProgram(p,{k:(a,b) for k in p.calls})
    assert len(list(m.combinations()))==2
    state=StateAllocation("same",(0,),8,("x",),True,"a")
    bad=replace(state,bytes_per_rank=16)
    m=MatchedProgram(p,{"produce":(option(states=(state,)),),"consume0":(option(states=(bad,)),)})
    assert not list(m.combinations())


def test_handler_required_before_any_search():
    with pytest.raises(ValueError,match="executable"):
        replace(option(),invoke=None)
