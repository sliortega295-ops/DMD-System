"""Exact node-local Nsight capture range for GAN Full5 diagnostics."""
from __future__ import annotations
import json, os, socket, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION="distillschedule.gan_nsys_range.v1"
START_STEP=1000
STOP_STEP=1004

class ProfilerApi(Protocol):
    def start(self)->None: ...
    def stop(self)->None: ...

class TorchCudaProfilerApi:
    @staticmethod
    def _check(result:Any, operation:str)->None:
        status=result[0] if isinstance(result,tuple) and result else result
        if status is None:return
        try: code=int(status)
        except (TypeError,ValueError): code=int(getattr(status,"value",0))
        if code!=0: raise RuntimeError(f"{operation} failed with CUDA runtime status {code}")
    def start(self)->None:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable for Nsight capture")
        self._check(torch.cuda.cudart().cudaProfilerStart(),"cudaProfilerStart")
    def stop(self)->None:
        import torch
        self._check(torch.cuda.cudart().cudaProfilerStop(),"cudaProfilerStop")

@dataclass(frozen=True)
class NsightRangeReceipt:
    rank:int; local_rank:int; controls_profiler_api:bool; start_step:int; stop_step:int
    start_point:str; stop_point:str; start_unix_ns:int; stop_unix_ns:int; elapsed_ms:float; host:str
    def to_dict(self)->dict[str,Any]:
        return {"schema_version":SCHEMA_VERSION,"status":"PASS","profiled":True,"diagnostic_only":True,
                "rank":self.rank,"local_rank":self.local_rank,"controls_profiler_api":self.controls_profiler_api,
                "logical_window":[self.start_step,self.stop_step],"start_point":self.start_point,"stop_point":self.stop_point,
                "start_unix_ns":self.start_unix_ns,"stop_unix_ns":self.stop_unix_ns,"elapsed_ms":self.elapsed_ms,
                "host":self.host,"formal_speedup_eligible":False}

class GanFull5NsightRange:
    def __init__(self,*,api:ProfilerApi|None=None,receipt_path:str|Path|None=None,rank:int|None=None,
                 local_rank:int|None=None,enabled:bool=True,controls_profiler_api:bool|None=None)->None:
        self.api=api or TorchCudaProfilerApi(); self.receipt_path=Path(receipt_path) if receipt_path is not None else None
        self.rank=int(os.environ.get("RANK","-1")) if rank is None else int(rank)
        self.local_rank=int(os.environ.get("LOCAL_RANK","-1")) if local_rank is None else int(local_rank)
        self.enabled=bool(enabled); self.controls_profiler_api=(self.local_rank==0 if controls_profiler_api is None else bool(controls_profiler_api))
        self._started=False; self._stopped=False; self._start_unix_ns=None; self._start_perf_ns=None
    @classmethod
    def from_env(cls)->"GanFull5NsightRange":
        enabled=os.environ.get("DISTILLSCHEDULE_GAN_NSYS_EXACT_RANGE","0").strip().lower() in {"1","true","yes","on"}
        rank=int(os.environ.get("RANK","-1")); local_rank=int(os.environ.get("LOCAL_RANK","-1")); path=None
        if enabled:
            root=os.environ.get("DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR","").strip()
            if not root: raise ValueError("DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR is required for exact Nsight range")
            if rank<0 or local_rank<0: raise ValueError("RANK and LOCAL_RANK are required for exact Nsight range receipt")
            path=Path(root)/f"rank_{rank:03d}.json"
        return cls(receipt_path=path,rank=rank,local_rank=local_rank,enabled=enabled,controls_profiler_api=(local_rank==0))
    @property
    def active(self)->bool:return self._started and not self._stopped
    def before_model_compute(self,logical_step:int)->None:
        if not self.enabled or logical_step!=START_STEP:return
        if self._started:raise RuntimeError("GAN Full5 Nsight range started more than once")
        self._start_unix_ns=time.time_ns(); self._start_perf_ns=time.perf_counter_ns()
        if self.controls_profiler_api:self.api.start()
        self._started=True
    def after_critic_commit(self,logical_step:int)->NsightRangeReceipt|None:
        if not self.enabled or logical_step!=STOP_STEP:return None
        if not self._started:raise RuntimeError("GAN Full5 Nsight range stopped before it started")
        if self._stopped:raise RuntimeError("GAN Full5 Nsight range stopped more than once")
        if self.controls_profiler_api:self.api.stop()
        stop_perf=time.perf_counter_ns(); stop_unix=time.time_ns(); self._stopped=True
        assert self._start_perf_ns is not None and self._start_unix_ns is not None
        receipt=NsightRangeReceipt(self.rank,self.local_rank,self.controls_profiler_api,START_STEP,STOP_STEP,
            "before_first_model_compute","after_step1004_critic_commit",self._start_unix_ns,stop_unix,
            (stop_perf-self._start_perf_ns)/1_000_000.0,socket.gethostname())
        if self.receipt_path is not None:self._write_receipt(receipt)
        return receipt
    def assert_complete(self)->None:
        if self.enabled and not(self._started and self._stopped):raise AssertionError("exact GAN Nsight capture did not observe both Full5 boundaries")
    def _write_receipt(self,receipt:NsightRangeReceipt)->None:
        assert self.receipt_path is not None; path=self.receipt_path; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(f"refusing to overwrite Nsight range receipt: {path}")
        tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(receipt.to_dict(),indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def validate_nsys_receipts(root:str|Path,*,expected_world_size:int=16,expected_local_world_size:int=8)->dict[str,Any]:
    if expected_world_size<=0 or expected_local_world_size<=0 or expected_world_size%expected_local_world_size:raise ValueError("invalid expected world sizes")
    root=Path(root); files=sorted(root.glob("rank_*.json"))
    if len(files)!=expected_world_size:raise ValueError(f"Nsight receipt count={len(files)}, expected={expected_world_size}")
    controllers=[]
    for expected_rank,path in enumerate(files):
        value=json.loads(path.read_text()); rank=int(value.get("rank",-1)); local_rank=int(value.get("local_rank",-1))
        if value.get("schema_version")!=SCHEMA_VERSION or value.get("status")!="PASS":raise ValueError(f"invalid Nsight range receipt: {path}")
        if rank!=expected_rank or local_rank!=rank%expected_local_world_size:raise ValueError(f"Nsight rank mapping drift: {path}")
        expected_controller=local_rank==0
        if bool(value.get("controls_profiler_api"))!=expected_controller:raise ValueError(f"Nsight profiler-controller drift: {path}")
        if expected_controller:controllers.append(rank)
        if value.get("logical_window")!=[START_STEP,STOP_STEP]:raise ValueError(f"Nsight range window drift: {path}")
        if value.get("start_point")!="before_first_model_compute" or value.get("stop_point")!="after_step1004_critic_commit":raise ValueError(f"Nsight range boundary drift: {path}")
        if value.get("diagnostic_only") is not True or value.get("formal_speedup_eligible") is not False:raise ValueError(f"profiled run incorrectly marked formal: {path}")
    expected_controllers=list(range(0,expected_world_size,expected_local_world_size))
    if controllers!=expected_controllers:raise ValueError(f"Nsight controller ranks={controllers}, expected={expected_controllers}")
    return {"schema_version":"distillschedule.gan_nsys_receipt_bundle.v1","status":"PASS","world_size":expected_world_size,
            "local_world_size":expected_local_world_size,"controller_ranks":controllers,"logical_window":[START_STEP,STOP_STEP],
            "profiled":True,"diagnostic_only":True,"formal_speedup_eligible":False,"receipts":[str(path) for path in files]}

__all__=["GanFull5NsightRange","NsightRangeReceipt","ProfilerApi","SCHEMA_VERSION","START_STEP","STOP_STEP","TorchCudaProfilerApi","validate_nsys_receipts"]
