"""Typed physical execution plan, separate from DMD algorithm semantics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
from .contracts import ContractViolation, reject_unknown_keys, require_mapping
from .enums import HostStatePolicy, InstrumentationMode, RolloutEngineKind, TrainingBackendKind
from .specs import HardwareSpec, WorkloadSpec


def _m(v:Any,p:str)->Mapping[str,Any]: return require_mapping(v,path=p)
def _need(m:Mapping[str,Any],k:str,p:str):
    if k not in m: raise ContractViolation(f"missing required field {k!r}",path=p)
    return m[k]
def _s(v:Any,p:str)->str:
    if not isinstance(v,str) or not v.strip(): raise ContractViolation("expected a non-empty string",path=p)
    return v.strip()
def _i(v:Any,p:str,minv:int=0)->int:
    if isinstance(v,bool) or not isinstance(v,int) or v<minv: raise ContractViolation(f"expected integer >= {minv}",path=p)
    return v
def _strs(v:Any,p:str)->tuple[str,...]:
    if v is None:return ()
    if not isinstance(v,(list,tuple)):raise ContractViolation("expected a sequence",path=p)
    out=tuple(_s(x,f"{p}[]") for x in v)
    if len(set(out))!=len(out):raise ContractViolation("entries must be unique",path=p)
    return out
def _ints(v:Any,p:str)->tuple[int,...]:
    if not isinstance(v,(list,tuple)):raise ContractViolation("expected a sequence",path=p)
    out=tuple(_i(x,f"{p}[]") for x in v)
    if len(set(out))!=len(out):raise ContractViolation("entries must be unique",path=p)
    return out

def _meta(v:Any)->dict[str,Any]: return {} if v is None else dict(_m(v,"metadata"))

@dataclass(frozen=True)
class RankGroupSpec:
    name:str; ranks:tuple[int,...]; components:tuple[str,...]
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"RankGroupSpec":
        m=_m(v,path); reject_unknown_keys(m,{"name","ranks","components"},path=path)
        return cls(_s(_need(m,"name",path),f"{path}.name"),_ints(_need(m,"ranks",path),f"{path}.ranks"),_strs(_need(m,"components",path),f"{path}.components"))
    def to_dict(self):return {"name":self.name,"ranks":list(self.ranks),"components":list(self.components)}

@dataclass(frozen=True)
class TrainingBackendSpec:
    kind:TrainingBackendKind; zero_stage:int=0; parameter_offload:str="none"; optimizer_offload:str="none"; gradient_checkpointing:bool=True; config_path:str|None=None; options:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"TrainingBackendSpec":
        m=_m(v,path); reject_unknown_keys(m,{"kind","zero_stage","parameter_offload","optimizer_offload","gradient_checkpointing","config_path","options"},path=path)
        kind=TrainingBackendKind.parse(_need(m,"kind",path),path=f"{path}.kind"); z=_i(m.get("zero_stage",0),f"{path}.zero_stage")
        if z>3: raise ContractViolation("zero_stage must be between 0 and 3",path=f"{path}.zero_stage")
        if kind!=TrainingBackendKind.DEEPSPEED and z: raise ContractViolation("zero_stage is only valid for DeepSpeed",path=f"{path}.zero_stage")
        po=str(m.get("parameter_offload","none")).lower(); oo=str(m.get("optimizer_offload","none")).lower()
        if po not in {"none","cpu","nvme"} or oo not in {"none","cpu","nvme"}:raise ContractViolation("unsupported offload device",path=path)
        cp=m.get("config_path")
        return cls(kind,z,po,oo,bool(m.get("gradient_checkpointing",True)),_s(cp,f"{path}.config_path") if cp else None,dict(m.get("options",{})))
    def to_dict(self):return {"kind":self.kind.value,"zero_stage":self.zero_stage,"parameter_offload":self.parameter_offload,"optimizer_offload":self.optimizer_offload,"gradient_checkpointing":self.gradient_checkpointing,"config_path":self.config_path,"options":dict(self.options)}

@dataclass(frozen=True)
class RolloutBackendSpec:
    engine:RolloutEngineKind=RolloutEngineKind.NATIVE; endpoint:str|None=None; protocol:str="in_process"; exact_version:bool=True; max_version_staleness:int=0; options:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="rollout")->"RolloutBackendSpec":
        if v is None:return cls()
        m=_m(v,path); reject_unknown_keys(m,{"engine","endpoint","protocol","exact_version","max_version_staleness","options"},path=path)
        eng=RolloutEngineKind.parse(m.get("engine","native"),path=f"{path}.engine"); ep=m.get("endpoint"); stale=_i(m.get("max_version_staleness",0),f"{path}.max_version_staleness")
        if eng!=RolloutEngineKind.NATIVE and not ep and not bool(dict(m.get("options",{})).get("embedded",False)): raise ContractViolation("external rollout needs endpoint or options.embedded=true",path=f"{path}.endpoint")
        return cls(eng,_s(ep,f"{path}.endpoint") if ep else None,_s(m.get("protocol","in_process"),f"{path}.protocol"),bool(m.get("exact_version",True)),stale,dict(m.get("options",{})))
    def to_dict(self):return {"engine":self.engine.value,"endpoint":self.endpoint,"protocol":self.protocol,"exact_version":self.exact_version,"max_version_staleness":self.max_version_staleness,"options":dict(self.options)}

@dataclass(frozen=True)
class ActivationOffloadSpec:
    enabled:bool=False; threshold_mib:int=0; components:tuple[str,...]=()
    @property
    def threshold_bytes(self):return self.threshold_mib*1024*1024
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"ActivationOffloadSpec":
        if v is None:return cls()
        m=_m(v,path); reject_unknown_keys(m,{"enabled","threshold_mib","components"},path=path)
        return cls(bool(m.get("enabled",False)),_i(m.get("threshold_mib",0),f"{path}.threshold_mib"),_strs(m.get("components",[]),f"{path}.components"))
    def to_dict(self):return {"enabled":self.enabled,"threshold_mib":self.threshold_mib,"components":list(self.components)}

@dataclass(frozen=True)
class FrozenBlockStreamingSpec:
    enabled:bool=False; components:tuple[str,...]=(); block_indices:tuple[int,...]=(); block_count:int=0; window:int=1; host_policy:HostStatePolicy=HostStatePolicy.PRIVATE; mmap_cache:str|None=None; pin_host_state:bool=False
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"FrozenBlockStreamingSpec":
        if v is None:return cls()
        m=_m(v,path); reject_unknown_keys(m,{"enabled","components","block_indices","block_count","window","host_policy","mmap_cache","pin_host_state"},path=path)
        idx=tuple(_i(x,f"{path}.block_indices[]") for x in m.get("block_indices",[])); cache=m.get("mmap_cache")
        return cls(bool(m.get("enabled",False)),_strs(m.get("components",[]),f"{path}.components"),idx,_i(m.get("block_count",0),f"{path}.block_count"),_i(m.get("window",1),f"{path}.window",1),HostStatePolicy.parse(m.get("host_policy","private"),path=f"{path}.host_policy"),_s(cache,f"{path}.mmap_cache") if cache else None,bool(m.get("pin_host_state",False)))
    def to_dict(self):return {"enabled":self.enabled,"components":list(self.components),"block_indices":list(self.block_indices),"block_count":self.block_count,"window":self.window,"host_policy":self.host_policy.value,"mmap_cache":self.mmap_cache,"pin_host_state":self.pin_host_state}

@dataclass(frozen=True)
class LifecycleSpec:
    split_modulation:bool=False; activation_offload:ActivationOffloadSpec=field(default_factory=ActivationOffloadSpec); frozen_block_streaming:FrozenBlockStreamingSpec=field(default_factory=FrozenBlockStreamingSpec); construction_wave_size:int=0; swap_wave_size:int=0; prefetch_distance:int=0; trim_host_after_move:bool=False; immutable_model_backing_components:tuple[str,...]=(); metadata:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="lifecycle")->"LifecycleSpec":
        if v is None:return cls()
        m=_m(v,path); reject_unknown_keys(m,{"split_modulation","activation_offload","frozen_block_streaming","construction_wave_size","swap_wave_size","prefetch_distance","trim_host_after_move","immutable_model_backing_components","metadata"},path=path)
        return cls(bool(m.get("split_modulation",False)),ActivationOffloadSpec.from_mapping(m.get("activation_offload"),path=f"{path}.activation_offload"),FrozenBlockStreamingSpec.from_mapping(m.get("frozen_block_streaming"),path=f"{path}.frozen_block_streaming"),_i(m.get("construction_wave_size",0),f"{path}.construction_wave_size"),_i(m.get("swap_wave_size",0),f"{path}.swap_wave_size"),_i(m.get("prefetch_distance",0),f"{path}.prefetch_distance"),bool(m.get("trim_host_after_move",False)),_strs(m.get("immutable_model_backing_components",[]),f"{path}.immutable_model_backing_components"),_meta(m.get("metadata")))
    def to_dict(self):return {"split_modulation":self.split_modulation,"activation_offload":self.activation_offload.to_dict(),"frozen_block_streaming":self.frozen_block_streaming.to_dict(),"construction_wave_size":self.construction_wave_size,"swap_wave_size":self.swap_wave_size,"prefetch_distance":self.prefetch_distance,"trim_host_after_move":self.trim_host_after_move,"immutable_model_backing_components":list(self.immutable_model_backing_components),"metadata":dict(self.metadata)}

@dataclass(frozen=True)
class EvidenceSpec:
    run_id:str; output_dir:str; start_step:int=0; steps:int=1; instrumentation:InstrumentationMode=InstrumentationMode.NONE; semantic_assertions:bool=True; operation_trace:str|None=None; checkpoint:str|None=None; metadata:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="evidence")->"EvidenceSpec":
        m=_m(v,path); reject_unknown_keys(m,{"run_id","output_dir","start_step","steps","instrumentation","semantic_assertions","operation_trace","checkpoint","metadata"},path=path)
        op=m.get("operation_trace"); cp=m.get("checkpoint")
        return cls(_s(_need(m,"run_id",path),f"{path}.run_id"),_s(_need(m,"output_dir",path),f"{path}.output_dir"),_i(m.get("start_step",0),f"{path}.start_step"),_i(m.get("steps",1),f"{path}.steps",1),InstrumentationMode.parse(m.get("instrumentation","none"),path=f"{path}.instrumentation"),bool(m.get("semantic_assertions",True)),_s(op,f"{path}.operation_trace") if op else None,_s(cp,f"{path}.checkpoint") if cp else None,_meta(m.get("metadata")))
    def to_dict(self):return {"run_id":self.run_id,"output_dir":self.output_dir,"start_step":self.start_step,"steps":self.steps,"instrumentation":self.instrumentation.value,"semantic_assertions":self.semantic_assertions,"operation_trace":self.operation_trace,"checkpoint":self.checkpoint,"metadata":dict(self.metadata)}

@dataclass(frozen=True)
class LegacyEntrypointSpec:
    config_path:str; entrypoint:str; python_executable:str="python"; cwd:str|None=None; command:tuple[str,...]=(); extra_env:Mapping[str,str]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="legacy")->"LegacyEntrypointSpec":
        m=_m(v,path); reject_unknown_keys(m,{"config_path","entrypoint","python_executable","cwd","command","extra_env"},path=path)
        cwd=m.get("cwd")
        return cls(_s(_need(m,"config_path",path),f"{path}.config_path"),_s(_need(m,"entrypoint",path),f"{path}.entrypoint"),_s(m.get("python_executable","python"),f"{path}.python_executable"),_s(cwd,f"{path}.cwd") if cwd else None,tuple(str(x) for x in m.get("command",[])),{str(k):str(val) for k,val in dict(m.get("extra_env",{})).items()})
    def to_dict(self):return {"config_path":self.config_path,"entrypoint":self.entrypoint,"python_executable":self.python_executable,"cwd":self.cwd,"command":list(self.command),"extra_env":dict(self.extra_env)}

@dataclass(frozen=True)
class ExecutionPlan:
    schema_version:str; plan_id:str; workload:WorkloadSpec; hardware:HardwareSpec; rank_groups:tuple[RankGroupSpec,...]; training_backends:Mapping[str,TrainingBackendSpec]; rollout:RolloutBackendSpec; lifecycle:LifecycleSpec; evidence:EvidenceSpec; legacy:LegacyEntrypointSpec|None=None; metadata:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="plan")->"ExecutionPlan":
        m=_m(v,path); reject_unknown_keys(m,{"schema_version","id","workload","hardware","rank_groups","training_backends","rollout","lifecycle","evidence","legacy","metadata","profile_key"},path=path)
        groups=tuple(RankGroupSpec.from_mapping(x,path=f"rank_groups[{i}]") for i,x in enumerate(_need(m,"rank_groups",path)))
        backs={str(k):TrainingBackendSpec.from_mapping(val,path=f"training_backends.{k}") for k,val in _m(_need(m,"training_backends",path),"training_backends").items()}
        plan=cls(_s(m.get("schema_version","1"),"schema_version"),_s(_need(m,"id",path),"id"),WorkloadSpec.from_mapping(_need(m,"workload",path)),HardwareSpec.from_mapping(_need(m,"hardware",path)),groups,backs,RolloutBackendSpec.from_mapping(m.get("rollout")),LifecycleSpec.from_mapping(m.get("lifecycle")),EvidenceSpec.from_mapping(_need(m,"evidence",path)),LegacyEntrypointSpec.from_mapping(m["legacy"]) if m.get("legacy") else None,_meta(m.get("metadata")))
        ranks={r for g in groups for r in g.ranks}
        if any(r>=plan.hardware.world_size for r in ranks):raise ContractViolation("rank group contains rank outside world_size",path="rank_groups")
        return plan
    @property
    def profile_key(self)->str:return self.workload.profile_key
    def to_dict(self):return {"schema_version":self.schema_version,"id":self.plan_id,"workload":self.workload.to_dict(),"hardware":self.hardware.to_dict(),"rank_groups":[g.to_dict() for g in self.rank_groups],"training_backends":{k:v.to_dict() for k,v in self.training_backends.items()},"rollout":self.rollout.to_dict(),"lifecycle":self.lifecycle.to_dict(),"evidence":self.evidence.to_dict(),"legacy":self.legacy.to_dict() if self.legacy else None,"metadata":dict(self.metadata)}
