"""Finite native adapter routing, never a shell command or winning-recipe map."""
from functools import partial
from types import SimpleNamespace
from .contracts import require
from .wan_implementation import WAN
from .livewan_implementation import LIVEWAN
from .h3_implementation import H3

SUPPORTED_WORKLOADS = frozenset((WAN, LIVEWAN, H3))


def adapter_for(workload):
    from .wan_implementation import WAN, WanImplementation
    if workload == WAN:
        from . import wan_model_adapter as m
        return SimpleNamespace(ADAPTER=m.ADAPTER, ADAPTERS=(m.ADAPTER,"wan_prepared_actions_v2"), implementation_type=WanImplementation,
            capability_report=m.capability_report, build_problem=m.build_problem,
            validate_native_problem=m.validate_native_problem,
            lower_prediction=m.lower_prediction, validate_lowered_plan=m.validate_lowered_plan)
    from . import native_macro_adapter as m
    require(workload in (m.LIVEWAN, m.H3), "native model adapter not implemented")
    return SimpleNamespace(ADAPTER=m.ADAPTER, implementation_type=m.implementation_type(workload),
        capability_report=partial(m.capability_report, workload), build_problem=m.build_problem,
        validate_native_problem=m.validate_native_problem,
        lower_prediction=m.lower_prediction, validate_lowered_plan=m.validate_lowered_plan)
