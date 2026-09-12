from types import SimpleNamespace as NS

import pytest

from test_refactor_wan_stage_calibration import FakeBackend, SOURCE, DEVICE
from distillschedule.search.wan_action_calibration import ActionCalibrator
from distillschedule.search.wan_action_costs import action_graph, component_contract
from distillschedule.search.wan_implementation import WanImplementation


class Tensor:
    is_cuda = True
    device = "cuda:0"
    dtype = "torch.bfloat16"
    shape = (1,16,9,48,80)
    def __init__(self, pointer, size):
        self.pointer,self.size = pointer,size
    def untyped_storage(self):
        return NS(data_ptr=lambda:self.pointer,nbytes=lambda:self.size)
    def cpu(self):
        raise AssertionError("tensor value reads are forbidden")


def fixture(window=3):
    backend = FakeBackend()
    backend.is_tensor = lambda x:isinstance(x,Tensor)
    impl = WanImplementation("resident",2048,True,rollout_window=window)
    work = {op.metadata["action"]:{"physical_batch":1,"logical_batch":16,"shape":[1,16,9,48,80]}
            for op in action_graph().operations}
    collector = ActionCalibrator(torch=backend,distributed=backend,device=DEVICE,rank=0,
        implementation=impl,source_head=SOURCE,work=work,clock=backend.clock,evidence_scope="CPU_FIXTURE_ONLY")
    return backend,collector,impl,work


def test_collect_actual_components_and_retained_lifetimes_without_value_reads():
    backend,c,impl,work = fixture()
    graph = action_graph(3)
    live = 0
    for n,op in enumerate(graph.operations):
        action = op.metadata["action"]
        with c.stage(op.metadata["cycle"],action) as sample:
            backend.body(allocated=100+live+20,peak_allocated=100+live+80)
        if action.startswith("r"):
            x=Tensor(100+n,20)
            sample.record_output((x,{"alias":x}))
            live += 20
        elif action.startswith("c"):
            live -= 20
    result=c.finish()
    assert result["status"] == "PASS"
    assert len(result["stages"]) == 32
    assert len(result["output_lifetimes"]) == 10
    assert result["peak_retained_input_bytes"] == 60
    assert result["retained_input_bytes_at_finish"] == 0
    assert all(row["workspace_bytes_per_rank"] == 80 for row in result["stages"])
    for op,row in zip(graph.operations,result["stages"]):
        contract=component_contract(op,impl,work[op.metadata["action"]])
        assert row["key"] == contract.digest
        assert row["component_contract"] == contract.to_dict()
    assert all(x["bytes"] == 20 for x in result["output_lifetimes"])


def test_missing_output_metadata_is_not_silently_treated_as_zero_memory():
    backend,c,_,_ = fixture(1)
    with c.stage(1,"g"):
        backend.body()
    with c.stage(1,"r0"):
        backend.body()
    with pytest.raises(ValueError,match="output"):
        with c.stage(1,"c0"):
            pass
    assert c.snapshot()["status"] == "FAIL"


def test_failed_body_keeps_original_exception_and_failure_evidence():
    _,c,_,_ = fixture()
    with pytest.raises(RuntimeError,match="original model error"):
        with c.stage(1,"g"):
            raise RuntimeError("original model error")
    row=c.snapshot()
    assert row["status"] == "FAIL"
    assert row["stages"][0]["operation_id"] == "cycle1.g"
    assert row["stages"][0]["status"] == "FAIL"


def test_lifetime_reconciliation_rejects_changed_bytes_or_missing_outputs():
    from copy import deepcopy
    from distillschedule.search.wan_action_calibration import validate_lifetimes
    backend,c,_,_ = fixture(3)
    for n,op in enumerate(action_graph(3).operations):
        with c.stage(op.metadata["cycle"],op.metadata["action"]) as sample:
            backend.body()
        if op.metadata["action"].startswith("r"):
            sample.record_output(Tensor(100+n,20))
    good=c.finish()
    validate_lifetimes(good)
    for corrupt in ("bytes","entry","missing"):
        bad=deepcopy(good)
        if corrupt=="bytes":bad["output_lifetimes"][0]["bytes"]+=1
        elif corrupt=="entry":bad["stages"][1]["entry_retained_input_bytes"]+=1
        else:bad["output_lifetimes"].pop()
        with pytest.raises(ValueError):validate_lifetimes(bad)
