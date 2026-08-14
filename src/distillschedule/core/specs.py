"""Typed model, shape, workload and hardware specifications."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
from .contracts import ContractViolation, reject_unknown_keys, require_mapping
from .enums import ModelRole, Precision


def _map(v: Any, path: str) -> Mapping[str, Any]: return require_mapping(v, path=path)
def _meta(v: Any) -> dict[str, Any]: return {} if v is None else dict(_map(v, "metadata"))
def _need(m: Mapping[str, Any], k: str, p: str):
    if k not in m: raise ContractViolation(f"missing required field {k!r}", path=p)
    return m[k]
def _s(v: Any, p: str) -> str:
    if not isinstance(v, str) or not v.strip(): raise ContractViolation("expected a non-empty string", path=p)
    return v.strip()
def _i(v: Any, p: str, *, minv: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minv: raise ContractViolation(f"expected integer >= {minv}", path=p)
    return v
def _slug(v: str) -> str: return v.lower().replace("_", "-").replace(" ", "-")

@dataclass(frozen=True)
class AdapterSpec:
    kind: str = "none"
    rank: int = 0
    alpha: float | None = None
    trainable: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)
    @classmethod
    def from_mapping(cls, v: Any, *, path: str) -> "AdapterSpec":
        if v is None: return cls()
        m=_map(v,path); reject_unknown_keys(m,{"kind","rank","alpha","trainable","options"},path=path)
        rank=_i(m.get("rank",0),f"{path}.rank")
        alpha=m.get("alpha")
        return cls(_s(m.get("kind","none"),f"{path}.kind"),rank,float(alpha) if alpha is not None else None,bool(m.get("trainable",False)),dict(m.get("options",{})))
    def to_dict(self): return {"kind":self.kind,"rank":self.rank,"alpha":self.alpha,"trainable":self.trainable,"options":dict(self.options)}

@dataclass(frozen=True)
class ModelSpec:
    name: str; family: str; size: str; role: ModelRole; checkpoint: str
    subfolder: str|None=None; precision: Precision=Precision.BF16; adapter: AdapterSpec=field(default_factory=AdapterSpec); metadata: Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"ModelSpec":
        m=_map(v,path); reject_unknown_keys(m,{"name","family","size","role","checkpoint","subfolder","precision","adapter","metadata"},path=path)
        sub=m.get("subfolder")
        return cls(_s(_need(m,"name",path),f"{path}.name"),_s(_need(m,"family",path),f"{path}.family"),_s(_need(m,"size",path),f"{path}.size"),ModelRole.parse(_need(m,"role",path),path=f"{path}.role"),_s(_need(m,"checkpoint",path),f"{path}.checkpoint"),_s(sub,f"{path}.subfolder") if sub else None,Precision.parse(m.get("precision","bf16"),path=f"{path}.precision"),AdapterSpec.from_mapping(m.get("adapter"),path=f"{path}.adapter"),_meta(m.get("metadata")))
    def to_dict(self): return {"name":self.name,"family":self.family,"size":self.size,"role":self.role.value,"checkpoint":self.checkpoint,"subfolder":self.subfolder,"precision":self.precision.value,"adapter":self.adapter.to_dict(),"metadata":dict(self.metadata)}

@dataclass(frozen=True)
class SpatialSize:
    height:int; width:int
    @classmethod
    def from_value(cls,v:Any,*,path:str)->"SpatialSize":
        if isinstance(v,Mapping): h,w=v.get("height"),v.get("width")
        else:
            if not isinstance(v,(list,tuple)) or len(v)!=2: raise ContractViolation("expected [height, width]",path=path)
            h,w=v
        return cls(_i(h,f"{path}.height",minv=1),_i(w,f"{path}.width",minv=1))
    def to_list(self): return [self.height,self.width]

@dataclass(frozen=True)
class LatentShape:
    channels:int; frames:int; height:int; width:int
    @property
    def spatial(self)->SpatialSize: return SpatialSize(self.height,self.width)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str)->"LatentShape":
        m=_map(v,path); reject_unknown_keys(m,{"channels","frames","height","width"},path=path)
        return cls(*[_i(_need(m,k,path),f"{path}.{k}",minv=1) for k in ("channels","frames","height","width")])
    def to_dict(self): return {"channels":self.channels,"frames":self.frames,"height":self.height,"width":self.width}

@dataclass(frozen=True)
class ShapeSpec:
    batch_size:int; frames:int; resolution:SpatialSize; latent:LatentShape; pyramid:tuple[SpatialSize,...]=(); text_tokens:int=0
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="workload.shape")->"ShapeSpec":
        m=_map(v,path); reject_unknown_keys(m,{"batch_size","frames","resolution","latent","pyramid","text_tokens"},path=path)
        pyr=tuple(SpatialSize.from_value(x,path=f"{path}.pyramid") for x in m.get("pyramid",[]))
        return cls(_i(_need(m,"batch_size",path),f"{path}.batch_size",minv=1),_i(_need(m,"frames",path),f"{path}.frames",minv=1),SpatialSize.from_value(_need(m,"resolution",path),path=f"{path}.resolution"),LatentShape.from_mapping(_need(m,"latent",path),path=f"{path}.latent"),pyr,_i(m.get("text_tokens",0),f"{path}.text_tokens"))
    @property
    def profile_key(self)->str:
        p="_".join(f"{x.height}x{x.width}" for x in self.pyramid) or "none"
        return f"b{self.batch_size}-f{self.frames}-lc{self.latent.channels}-lf{self.latent.frames}-{self.resolution.height}x{self.resolution.width}-l{self.latent.height}x{self.latent.width}-t{self.text_tokens}-p{p}"
    def to_dict(self): return {"batch_size":self.batch_size,"frames":self.frames,"resolution":{"height":self.resolution.height,"width":self.resolution.width},"latent":self.latent.to_dict(),"pyramid":[x.to_list() for x in self.pyramid],"text_tokens":self.text_tokens}

@dataclass(frozen=True)
class DMDBranchesSpec:
    dmd:bool=True; gan:bool=False; reward:bool=False; vae_decode:bool=False; ema:bool=True; classifier_free_guidance:bool=True
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="workload.branches")->"DMDBranchesSpec":
        if v is None: return cls()
        m=_map(v,path); reject_unknown_keys(m,{"dmd","gan","reward","vae_decode","ema","classifier_free_guidance"},path=path)
        return cls(**{k:bool(m.get(k,getattr(cls(),k))) for k in ("dmd","gan","reward","vae_decode","ema","classifier_free_guidance")})
    def to_dict(self): return {k:getattr(self,k) for k in ("dmd","gan","reward","vae_decode","ema","classifier_free_guidance")}

@dataclass(frozen=True)
class DMDScheduleSpec:
    generator_update_every:int=5; critic_update_every:int=1; ema_update_every:int=1; ema_start_step:int=0; ema_bootstrap_before_start:bool=False; ema_generator_only_after_start:bool=False; ema_after_critic:bool=True
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="workload.schedule")->"DMDScheduleSpec":
        if v is None:return cls()
        m=_map(v,path); reject_unknown_keys(m,{"generator_update_every","critic_update_every","ema_update_every","ema_start_step","ema_bootstrap_before_start","ema_generator_only_after_start","ema_after_critic"},path=path)
        return cls(_i(m.get("generator_update_every",5),f"{path}.generator_update_every",minv=1),_i(m.get("critic_update_every",1),f"{path}.critic_update_every",minv=1),_i(m.get("ema_update_every",1),f"{path}.ema_update_every",minv=1),_i(m.get("ema_start_step",0),f"{path}.ema_start_step"),bool(m.get("ema_bootstrap_before_start",False)),bool(m.get("ema_generator_only_after_start",False)),bool(m.get("ema_after_critic",True)))
    def to_dict(self): return {"generator_update_every":self.generator_update_every,"critic_update_every":self.critic_update_every,"ema_update_every":self.ema_update_every,"ema_start_step":self.ema_start_step,"ema_bootstrap_before_start":self.ema_bootstrap_before_start,"ema_generator_only_after_start":self.ema_generator_only_after_start,"ema_after_critic":self.ema_after_critic}

@dataclass(frozen=True)
class WorkloadSpec:
    name:str; adapter:str; generator:ModelSpec; score:ModelSpec; shape:ShapeSpec; schedule:DMDScheduleSpec=field(default_factory=DMDScheduleSpec); branches:DMDBranchesSpec=field(default_factory=DMDBranchesSpec); metadata:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="workload")->"WorkloadSpec":
        m=_map(v,path); reject_unknown_keys(m,{"name","adapter","generator","score","shape","schedule","branches","metadata","profile_key"},path=path)
        obj=cls(_s(_need(m,"name",path),f"{path}.name"),_s(_need(m,"adapter",path),f"{path}.adapter"),ModelSpec.from_mapping(_need(m,"generator",path),path=f"{path}.generator"),ModelSpec.from_mapping(_need(m,"score",path),path=f"{path}.score"),ShapeSpec.from_mapping(_need(m,"shape",path),path=f"{path}.shape"),DMDScheduleSpec.from_mapping(m.get("schedule"),path=f"{path}.schedule"),DMDBranchesSpec.from_mapping(m.get("branches"),path=f"{path}.branches"),_meta(m.get("metadata")))
        if obj.generator.role != ModelRole.GENERATOR: raise ContractViolation("generator model role must be 'generator'",path=f"{path}.generator.role")
        return obj
    @property
    def profile_key(self)->str:
        g=f"g-{_slug(self.generator.family)}-{_slug(self.generator.size)}-{self.generator.precision.value}-r{self.generator.adapter.rank}"
        s=f"s-{_slug(self.score.family)}-{_slug(self.score.size)}-{self.score.precision.value}-r{self.score.adapter.rank}"
        return f"{g}__{s}__{self.shape.profile_key}"
    def to_dict(self): return {"name":self.name,"adapter":self.adapter,"generator":self.generator.to_dict(),"score":self.score.to_dict(),"shape":self.shape.to_dict(),"schedule":self.schedule.to_dict(),"branches":self.branches.to_dict(),"metadata":dict(self.metadata)}

@dataclass(frozen=True)
class HardwareSpec:
    nodes:int; local_world_size:int; world_size:int; gpu_memory_gib:float; node_memory_gib:float|None=None; reserve_gib:float=0; device_type:str="cuda"; metadata:Mapping[str,Any]=field(default_factory=dict)
    @classmethod
    def from_mapping(cls,v:Any,*,path:str="hardware")->"HardwareSpec":
        m=_map(v,path); reject_unknown_keys(m,{"nodes","local_world_size","world_size","gpu_memory_gib","node_memory_gib","reserve_gib","device_type","metadata"},path=path)
        nodes=_i(_need(m,"nodes",path),f"{path}.nodes",minv=1); local=_i(_need(m,"local_world_size",path),f"{path}.local_world_size",minv=1); world=_i(_need(m,"world_size",path),f"{path}.world_size",minv=1)
        if nodes*local != world: raise ContractViolation("nodes * local_world_size must equal world_size",path=path)
        return cls(nodes,local,world,float(_need(m,"gpu_memory_gib",path)),float(m["node_memory_gib"]) if m.get("node_memory_gib") is not None else None,float(m.get("reserve_gib",0)),str(m.get("device_type","cuda")),_meta(m.get("metadata")))
    def to_dict(self): return {"nodes":self.nodes,"local_world_size":self.local_world_size,"world_size":self.world_size,"gpu_memory_gib":self.gpu_memory_gib,"node_memory_gib":self.node_memory_gib,"reserve_gib":self.reserve_gib,"device_type":self.device_type,"metadata":dict(self.metadata)}
