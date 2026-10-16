"""Dispatch shared ORIGINAL helper geometry without changing model identity."""
from distillschedule.search.contracts import require
from distillschedule.search.helios_request import trace_request, TRACE_KEY as HELIOS_KEY
from distillschedule.strategies.wan_task import OriginalWanTask, TRACE_KEY as WAN_KEY


def rollout_geometry(operation_trace):
    contract = operation_trace.manifest.get("contract", {})
    require(not (HELIOS_KEY in contract and WAN_KEY in contract), "ambiguous Wan/Helios task identity")
    request = trace_request(operation_trace)
    if request is not None:
        return request
    raw = contract.get(WAN_KEY)
    return OriginalWanTask() if raw is None else OriginalWanTask.from_dict(raw)
