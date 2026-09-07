"""Measure generated native actions using the existing synchronized observer.

Only explicit calibration calls this module. Tensor *storage metadata* is read
after each rollout's calibrated range, never tensor values. No copies/streams/
collectives are introduced beyond the documented diagnostic stage fences.
"""
from contextlib import contextmanager
from copy import deepcopy

from .contracts import require
from .plan_ir import nonnegative
from .wan_stage_calibration import StageCalibrator
from .wan_action_costs import (ADAPTER, MEMORY_DEFINITION, TIMING_DEFINITION,
                               action_graph, component_contract)


class _OutputSample:
    def __init__(self, collector, operation_id):
        self.collector, self.operation_id = collector, operation_id

    def record_output(self, value):
        self.collector.record_output(self.operation_id, value)


class ActionCalibrator(StageCalibrator):
    SCHEMA = "distillschedule.wan_action_rank_calibration.v1"
    ADAPTER = ADAPTER
    MEMORY_DEFINITION = MEMORY_DEFINITION
    TIMING_DEFINITION = TIMING_DEFINITION

    def __init__(self, *, work, **kwargs):
        self.work = deepcopy(work)
        self.torch = kwargs["torch"]
        self._live = {}
        self._lifetimes = []
        self._pending_output = None
        self._peak_retained = 0
        require(kwargs["implementation"].rollout_window is not None,
                "action calibration requires a generated window")
        super().__init__(**kwargs)
        for op in self._graph.operations:
            self.contract(op)  # Fail before model compute if a work signature is absent.

    def operation_graph(self):
        return action_graph(self._implementation.rollout_window)

    def contract(self, operation):
        return component_contract(operation,self._implementation,
                                  self.work[operation.metadata["action"]]).to_dict()

    def _retained_bytes(self):
        return sum(s["bytes"] for stores in self._live.values() for s in stores.values())

    @contextmanager
    def stage(self, cycle, stage_name):
        if self._pending_output is not None:
            self._reject("previous rollout output metadata was not recorded")
        before = self._retained_bytes()
        operation_id = f"cycle{cycle}.{stage_name}"
        sample = _OutputSample(self,operation_id)
        try:
            with super().stage(cycle,stage_name):
                yield sample
        finally:
            if self._stages and self._stages[-1]["operation_id"] == operation_id:
                row = self._stages[-1]
                row["component_contract"] = row["operation_contract"]
                row["entry_retained_input_bytes"] = before
                if "peak_allocated_bytes" in row["memory_after"]:
                    row["workspace_bytes_per_rank"] = max(0,
                        row["memory_after"]["peak_allocated_bytes"] - self._static_bytes - before)
        if stage_name.startswith("r"):
            self._pending_output = operation_id
        elif stage_name.startswith("c"):
            source = f"cycle{cycle}.r{stage_name[1:]}"
            if source not in self._live:
                self._reject("critic consumed an unobserved rollout output")
            stores = self._live.pop(source)
            self._lifetimes.append(dict(producer=source, consumer=operation_id,
                bytes=sum(x["bytes"] for x in stores.values()),
                allocation_point="producer_exit", storages=list(stores.values())))

    def record_output(self, operation_id, value):
        """Called outside the measured producer range, before its next consumer."""
        if self._pending_output != operation_id or self._active is not None:
            self._reject("unexpected/duplicate output metadata")
        stores = {}
        def visit(item):
            if self.torch.is_tensor(item):
                if item.is_cuda:
                    storage = item.untyped_storage()
                    key = (str(item.device),storage.data_ptr())
                    stores[key] = {"bytes":storage.nbytes(), "device":str(item.device)}
            elif isinstance(item,dict):
                for child in item.values(): visit(child)
            elif isinstance(item,(tuple,list)):
                for child in item: visit(child)
            else:
                require(item is None or isinstance(item,(str,int,float,bool)),
                        "unsupported retained output metadata object")
        try:
            visit(value)
            require(stores and sum(x["bytes"] for x in stores.values()) > 0,
                    "rollout retained output has no CUDA storage")
            other = {key for live in self._live.values() for key in live}
            require(not (other & stores.keys()), "cross-output aliases require explicit shared lifetime accounting")
        except BaseException as error:
            self._fail(self._stages[-1],"output_metadata",error)
            raise
        self._live[operation_id] = stores
        self._stages[-1]["produced_output_bytes"] = sum(x["bytes"] for x in stores.values())
        self._peak_retained = max(self._peak_retained,self._retained_bytes())
        self._pending_output = None

    def snapshot(self):
        return {**super().snapshot(),"work_signatures":deepcopy(self.work),
                "output_lifetimes":deepcopy(self._lifetimes),
                "peak_retained_input_bytes":self._peak_retained,
                "retained_input_bytes_at_finish":self._retained_bytes(),
                "pending_output":self._pending_output,
                "tensor_value_reads":0,"output_metadata_outside_calibrated_range":True,
                "memory_definition":MEMORY_DEFINITION}

    def finish(self):
        if self._pending_output is not None or self._live:
            self._reject("unreleased/missing rollout output metadata at finish")
        return super().finish()


def validate_lifetimes(value):
    """Reconcile retained bytes from producing and consuming native actions."""
    require(value.get("memory_definition") == MEMORY_DEFINITION, "action memory definition changed")
    require(value.get("pending_output") is None and value.get("retained_input_bytes_at_finish") == 0,
            "incomplete retained output observation")
    rows = value["output_lifetimes"]
    require(len(rows) == 10 and len({r["producer"] for r in rows}) == 10,
            "missing/duplicate output lifetimes")
    lives = {r["producer"]:r for r in rows}
    live,peak = {},0
    for row in value["stages"]:
        key = row["operation_id"]
        prefix,action = key.split(".")
        before = sum(live.values())
        require(row.get("entry_retained_input_bytes") == before,"entry retained bytes differ")
        if action.startswith("r"):
            life = lives[key]
            size = nonnegative(life["bytes"],"retained storage bytes")
            require(size > 0 and size == row.get("produced_output_bytes"),"produced output bytes differ")
            require(life["consumer"] == prefix+".c"+action[1:] and life["allocation_point"] == "producer_exit",
                    "output consumer/allocation point changed")
            require(size == sum(nonnegative(s["bytes"],"storage bytes") for s in life["storages"]),
                    "output storage reconciliation failed")
            live[key] = size
            peak = max(peak,sum(live.values()))
        elif action.startswith("c"):
            source = prefix+".r"+action[1:]
            require(source in live,"missing consumed rollout")
            del live[source]
        require(row["workspace_bytes_per_rank"] == max(0,
            row["memory_after"]["peak_allocated_bytes"]-value["post_setup_static_bytes"]-before),
            "action workspace reconciliation failed")
    require(not live and value["peak_retained_input_bytes"] == peak,"retained peak/teardown differs")
