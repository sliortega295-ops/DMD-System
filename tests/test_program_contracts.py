from dataclasses import replace
import pytest
from distillschedule.core.graph import ExecutionGraph, Operation, Dependency
from distillschedule.core.enums import GradientMode as G, OperationKind as K
from distillschedule.programs.contracts import CallContract, WorkloadProgram


def program(n=2):
    ops=[Operation("produce",0,"arbitrary-producer",K.GENERATOR_ROLLOUT,G.NO_GRAD)]
    ops += [Operation(f"consume{i}",i,"previously-unknown-role",K.CRITIC_UPDATE,G.GRAD,
            required_versions={"score":i},produced_versions={"score":i+1},
            metadata={"inputs":["produce"]}) for i in range(n)]
    edges=[Dependency("produce",f"consume{i}") for i in range(n)]
    edges += [Dependency(f"consume{i}",f"consume{i+1}") for i in range(n-1)]
    graph=ExecutionGraph(tuple(ops),tuple(edges),{"initial_versions":{"score":0},"final_versions":{"score":n}})
    return WorkloadProgram(graph,{op.operation_id:CallContract("user.function",{"shape":[2,7]}) for op in ops},
                           {"updates":n},{"code":"fixture-v1","input":"fixture"})


@pytest.mark.parametrize("n",[1,2,3,5,8])
def test_unseen_program_frequency_roundtrip(n):
    p=program(n)
    assert len(p.graph.operations)==n+1
    assert WorkloadProgram.from_dict(p.to_dict()).digest==p.digest
    assert replace(p,name="neither-Wan-nor-known-family").digest==p.digest


def test_changed_shape_or_semantics_changes_identity():
    p=program()
    assert replace(p,parameters={"updates":3}).digest != p.digest
    assert replace(p,calls={**p.calls,"produce":CallContract("user.function",{"shape":[9,11]})}).digest!=p.digest


def test_missing_bindings_and_hidden_inputs_rejected():
    p=program()
    with pytest.raises(ValueError,match="each graph operation"):
        replace(p,calls={})
    first=replace(p.graph.operations[0],metadata={"inputs":["consume0"]})
    with pytest.raises(ValueError,match="dependency ancestors"):
        replace(p,graph=replace(p.graph,operations=(first,*p.graph.operations[1:])))


def test_versions_checked_before_model_loading():
    p=program()
    wrong=replace(p.graph.operations[1],required_versions={"score":8})
    with pytest.raises(ValueError):
        replace(p,graph=replace(p.graph,operations=(p.graph.operations[0],wrong,*p.graph.operations[2:])))


def test_unknown_json_fields_and_nan_rejected():
    p=program();raw=p.to_dict();raw["best_recipe"]="secret-winner"
    with pytest.raises(ValueError):WorkloadProgram.from_dict(raw)
    with pytest.raises(ValueError):CallContract("native",{"x":float("nan")})
