"""Reusable capability matching by call facts, never workload-name dispatch.

Model bindings expose reference functions and facts. Capabilities are separate
implementations/transforms; input programs do not enumerate or name best plans.
This API admits only options with an actual bound handler and completion fence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import itertools
from typing import Any, Callable, Iterable, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.graph import Operation
from distillschedule.runtime.role_runner import OperationContext
from distillschedule.search.cost_model import ComponentContract, CostModel, OperationCost
from distillschedule.search.plan_ir import BindingRule, Lifetime, StateAllocation
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import CallContract, WorkloadProgram, json_copy
from .solver import BoundCase


@dataclass(frozen=True)
class ExecutableOption:
    """One actual implementation of a logical call, with explicit physical facts.

    invoke MUST finish all its asynchronous work before returning, or synchronize
    must join it. Cross-device copies, cache effects and optimizer updates cannot
    be guessed/inserted by the planner. `settings` jointly bind shared choices:
    two calls cannot select different layouts for the same resident state.
    """
    key: str
    ranks: tuple[int, ...]
    resources: tuple[str, ...]
    invoke: Callable[[OperationContext], Any] = field(repr=False, compare=False)
    synchronize: Callable[[], None] = field(repr=False, compare=False)
    implementation_identity: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, str] = field(default_factory=dict)
    states: tuple[StateAllocation, ...] = ()
    lifetimes: tuple[Lifetime, ...] = ()
    workspace_upper_bound: int = 0
    is_reference: bool = False

    def __post_init__(self):
        require(self.key and self.ranks and self.resources,"incomplete executable option",path="program.option")
        require(callable(self.invoke) and callable(self.synchronize),"executable and completion fence required",path="program.option")
        require(bool(self.implementation_identity),"implementation source identity required",path="program.option")
        require(type(self.workspace_upper_bound) is int and self.workspace_upper_bound>=0,
                "workspace bound must be declared",path="program.option")
        require(len(set(self.ranks))==len(self.ranks) and all(type(r) is int and r>=0 for r in self.ranks),
                "invalid execution ranks",path="program.option")
        require(len(set(self.resources))==len(self.resources) and {f"gpu:{r}" for r in self.ranks}<=set(self.resources),
                "each device must be reserved (solver device labels use gpu:<rank>)",path="program.option")
        require(all(isinstance(k,str) and isinstance(v,str) for k,v in self.settings.items()),
                "settings must be string assignments",path="program.option")

    def to_dict(self):
        return {"key":self.key,"ranks":list(self.ranks),"resources":list(self.resources),
                "implementation_identity":json_copy(self.implementation_identity),
                "settings":dict(self.settings),"states":[asdict(s) for s in self.states],
                "lifetimes":[asdict(s) for s in self.lifetimes],
                "workspace_upper_bound":self.workspace_upper_bound,"is_reference":self.is_reference}


@dataclass(frozen=True)
class Capability:
    name: str
    operator: str
    required_properties: Mapping[str, Any]
    grad_modes: tuple[str, ...]
    bind: Callable[[Operation, CallContract], Iterable[ExecutableOption]] = field(repr=False,compare=False)

    def rejection(self,op,call):
        if call.operator!=self.operator:return "OPERATOR_INTERFACE"
        if op.grad_mode.value not in self.grad_modes:return "GRADIENT_MODE"
        missing=[k for k,v in self.required_properties.items() if call.properties.get(k)!=v]
        return "PROPERTY:"+",".join(missing) if missing else None


class CapabilityRegistry:
    def __init__(self):self._entries={}

    def register(self,capability: Capability):
        require(capability.name and capability.name not in self._entries,"duplicate capability",path="program.registry")
        self._entries[capability.name]=capability

    def match(self,program: WorkloadProgram):
        choices={};rejections=[]
        for op in program.graph.operations:
            found=[]
            for cap in self._entries.values():
                reason=cap.rejection(op,program.calls[op.operation_id])
                if reason:
                    rejections.append({"operation":op.operation_id,"capability":cap.name,"reason":reason});continue
                bound=tuple(cap.bind(op,program.calls[op.operation_id]))
                require(all(isinstance(v,ExecutableOption) for v in bound),"invalid executable binding",path="program.registry")
                found.extend(bound)
            require(found,f"no executable capability for {op.operation_id}/{program.calls[op.operation_id].operator}",path="program.registry")
            require(len({v.key for v in found})==len(found),"duplicate option key",path="program.registry")
            choices[op.operation_id]=tuple(found)
        return MatchedProgram(program,choices,tuple(rejections))


@dataclass(frozen=True)
class MatchedProgram:
    program: WorkloadProgram
    options: Mapping[str, tuple[ExecutableOption, ...]]
    rejections: tuple[Mapping[str, str], ...] = ()

    @property
    def digest(self):
        return fingerprint({"program":self.program.digest,
            "options":{k:[o.to_dict() for o in v] for k,v in self.options.items()}})

    def combinations(self,max_combinations=256):
        """Bounded DFS with shared-setting and physical-storage consistency pruning."""
        require(type(max_combinations) is int and 1<=max_combinations<=100000,"invalid binding budget",path="program.match")
        keys=self.program.graph.topological_order();produced=0
        def visit(index,selected,settings,states,leases):
            nonlocal produced
            if produced>=max_combinations:return
            if index==len(keys):
                produced+=1;yield dict(selected);return
            key=keys[index]
            for option in self.options[key]:
                if any(k in settings and settings[k]!=v for k,v in option.settings.items()):continue
                if any(s.id in states and states[s.id]!=s for s in option.states):continue
                if any(s.id in leases and leases[s.id]!=s for s in option.lifetimes):continue
                yield from visit(index+1,{**selected,key:option},{**settings,**option.settings},
                    {**states,**{s.id:s for s in option.states}},
                    {**leases,**{s.id:s for s in option.lifetimes}})
        yield from visit(0,{}, {},{}, {})


def call_cost_contract(matched,operation,option):
    op=next(o for o in matched.program.graph.operations if o.operation_id==operation)
    return ComponentContract(f"program.{matched.program.calls[operation].operator}",
        {"program_sha256":matched.program.digest,"call":matched.program.calls[operation].to_dict(),
         "logical_operation":op.operation_id,"required_versions":dict(op.required_versions)},
        {"declared":matched.program.calls[operation].properties.get("dtype","native")},
        {"ranks":list(option.ranks),"resources":list(option.resources)},
        option.to_dict(),{"native_identity":json_copy(matched.program.identity),"scope":"isolated_bound_call"})


def compile_cases(matched,models,memory_limits,*,max_combinations=256):
    """Use the same solver cases as original-model planners; no synthetic defaults.

    `models` maps (operation_id, option_key) to a measured/synthetic-for-tests
    CostModel with that exact contract. Missing measurements never become zero.
    """
    options=list(matched.combinations(max_combinations+1))
    require(len(options)<=max_combinations,"binding combinations exceed budget; increase budget explicitly",path="program.compile")
    require(options,"no compatible capability combination",path="program.compile")
    for selected in options:
        states={};leases={};rules={};records=[];binding=None;scope=None
        for key,option in selected.items():
            require((key,option.key) in models,f"UNCALIBRATED {key}/{option.key}",path="program.compile")
            model=models[key,option.key];model.verify_receipts()
            if binding is None:binding=dict(model.binding);scope=model.scope
            require(dict(model.binding)==binding and model.scope==scope,"mixed calibration identities",path="program.compile")
            contract=call_cost_contract(matched,key,option);cost=model.get_component(contract)
            records.append(cost)
            require(cost.workspace_bytes_per_rank==option.workspace_upper_bound,
                    "workspace model does not match bound implementation",path="program.compile")
            rules[key]=BindingRule(key,{},cost.key,option.ranks,option.resources,
                                   tuple(s.id for s in option.states),contract.digest)
            states.update({s.id:s for s in option.states});leases.update({s.id:s for s in option.lifetimes})
        selection={k:v.key for k,v in selected.items()}
        combined=CostModel(binding,tuple(records),scope=scope)
        yield BoundCase(fingerprint(selection),matched.program.graph,rules,combined,tuple(states.values()),
                        tuple(leases.values()),dict(memory_limits),selection)
