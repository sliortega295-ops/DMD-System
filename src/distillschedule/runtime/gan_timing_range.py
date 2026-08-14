"""Low-overhead exact wall-clock range for formal GAN Full5 timing.

Unlike Nsight diagnostics this recorder adds no barrier, CUDA event, CUDA
synchronize, or profiler API call. Each rank records only two host timestamps:
immediately before its first model compute at logical step1000 and immediately
after its step1004 Critic optimizer commit. The global five-step window is
``max(stop) - min(start)`` across world16, so stragglers are included while
post-step visualization/checkpoint/final-save work is excluded.
"""
from __future__ import annotations
import json,os,socket,time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
SCHEMA_VERSION="distillschedule.gan_full5_timing_rank.v1"; START_STEP=1000; STOP_STEP=1004
@dataclass(frozen=True)
class GanTimingReceipt:
    rank:int; start_unix_ns:int; stop_unix_ns:int; host:str
    def to_dict(self)->dict[str,Any]:return {"schema_version":SCHEMA_VERSION,"status":"PASS","profiled":False,"diagnostic_only":False,"rank":self.rank,"logical_window":[START_STEP,STOP_STEP],"start_point":"before_first_model_compute","stop_point":"after_step1004_critic_commit","start_unix_ns":self.start_unix_ns,"stop_unix_ns":self.stop_unix_ns,"local_elapsed_seconds":(self.stop_unix_ns-self.start_unix_ns)/1e9,"host":self.host}
class GanFull5TimingRange:
    def __init__(self,*,receipt_path:str|Path|None=None,rank:int|None=None,enabled:bool=True)->None:
        self.receipt_path=Path(receipt_path) if receipt_path is not None else None; self.rank=int(os.environ.get("RANK","-1")) if rank is None else int(rank); self.enabled=bool(enabled); self._start_ns=None; self._stop_ns=None
    @classmethod
    def from_env(cls)->"GanFull5TimingRange":
        enabled=os.environ.get("DISTILLSCHEDULE_GAN_EXACT_TIMING","1").strip().lower() in {"1","true","yes","on"}; rank=int(os.environ.get("RANK","-1")); path=None
        if enabled:
            root=os.environ.get("DISTILLSCHEDULE_GAN_TIMING_RECEIPT_DIR","").strip()
            if not root:
                output=os.environ.get("DISTILLSCHEDULE_OUTPUT_DIR","").strip()
                if output:root=str(Path(output)/"timing_receipts")
            if not root:raise ValueError("DISTILLSCHEDULE_GAN_TIMING_RECEIPT_DIR or DISTILLSCHEDULE_OUTPUT_DIR is required for exact GAN timing")
            if rank<0:raise ValueError("RANK is required for exact GAN timing")
            path=Path(root)/f"rank_{rank:03d}.json"
        return cls(receipt_path=path,rank=rank,enabled=enabled)
    def before_model_compute(self,logical_step:int)->None:
        if not self.enabled or logical_step!=START_STEP:return
        if self._start_ns is not None:raise RuntimeError("GAN Full5 timing started more than once")
        self._start_ns=time.time_ns()
    def after_critic_commit(self,logical_step:int)->GanTimingReceipt|None:
        if not self.enabled or logical_step!=STOP_STEP:return None
        if self._start_ns is None:raise RuntimeError("GAN Full5 timing stopped before start")
        if self._stop_ns is not None:raise RuntimeError("GAN Full5 timing stopped more than once")
        self._stop_ns=time.time_ns(); receipt=GanTimingReceipt(self.rank,self._start_ns,self._stop_ns,socket.gethostname())
        if self.receipt_path is not None:self._write(receipt)
        return receipt
    def assert_complete(self)->None:
        if self.enabled and (self._start_ns is None or self._stop_ns is None):raise AssertionError("exact GAN Full5 timing did not observe both boundaries")
    def _write(self,receipt:GanTimingReceipt)->None:
        assert self.receipt_path is not None; path=self.receipt_path; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(f"refusing to overwrite timing receipt: {path}")
        tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(receipt.to_dict(),indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def validate_timing_receipts(root:str|Path,*,expected_world_size:int=16)->dict[str,Any]:
    root=Path(root); files=sorted(root.glob("rank_*.json"))
    if len(files)!=expected_world_size:raise ValueError(f"timing receipt count={len(files)}, expected={expected_world_size}")
    starts=[]; stops=[]; local=[]
    for rank,path in enumerate(files):
        v=json.loads(path.read_text())
        if v.get("schema_version")!=SCHEMA_VERSION or v.get("status")!="PASS":raise ValueError(f"invalid timing receipt: {path}")
        if int(v.get("rank",-1))!=rank:raise ValueError(f"timing receipt rank drift: {path}")
        if v.get("logical_window")!=[START_STEP,STOP_STEP]:raise ValueError(f"timing window drift: {path}")
        if v.get("profiled") is not False or v.get("diagnostic_only") is not False:raise ValueError(f"formal timing receipt is profiled/diagnostic: {path}")
        start,stop=int(v["start_unix_ns"]),int(v["stop_unix_ns"])
        if stop<=start:raise ValueError(f"non-positive timing interval: {path}")
        starts.append(start); stops.append(stop); local.append((stop-start)/1e9)
    global_seconds=(max(stops)-min(starts))/1e9
    if global_seconds<=0:raise ValueError("global Full5 timing window is non-positive")
    return {"schema_version":"distillschedule.gan_full5_timing_bundle.v1","status":"PASS","claimable":True,"profiled":False,"diagnostic_only":False,"logical_steps":[1000,1001,1002,1003,1004],"rank_count":expected_world_size,"global_training_window_seconds":global_seconds,"min_rank_elapsed_seconds":min(local),"max_rank_elapsed_seconds":max(local),"receipts":[str(p) for p in files]}
__all__=["GanFull5TimingRange","GanTimingReceipt","SCHEMA_VERSION","START_STEP","STOP_STEP","validate_timing_receipts"]
