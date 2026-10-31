"""Dispatch shared ORIGINAL helper geometry without changing model identity."""
from distillschedule.search.contracts import require
from distillschedule.search.helios_request import trace_request, TRACE_KEY as HELIOS_KEY
from distillschedule.strategies.wan_task import OriginalWanTask, TRACE_KEY as WAN_KEY


def rollout_geometry(operation_trace):
    contract = operation_trace.manifest.get("contract", {})
    from distillschedule.training.helios_inputs import HeliosCohortTask, TRACE_KEY as TRAINING_KEY
    from distillschedule.training.helios_gan_inputs import GANCohortTask, TRACE_KEY as GAN_KEY
    require(sum(key in contract for key in (HELIOS_KEY, WAN_KEY, TRAINING_KEY, GAN_KEY)) <= 1,
            "ambiguous Wan/Helios task identity")
    if GAN_KEY in contract:
        return GANCohortTask.from_dict(contract[GAN_KEY]).request
    if TRAINING_KEY in contract:
        return HeliosCohortTask.from_dict(contract[TRAINING_KEY]).request
    request = trace_request(operation_trace)
    if request is not None:
        return request
    raw = contract.get(WAN_KEY)
    return OriginalWanTask() if raw is None else OriginalWanTask.from_dict(raw)
