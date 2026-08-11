#!/usr/bin/env python3
"""Run pinned Helios GAN math with timing/profiling boundary hooks only."""
from __future__ import annotations
import os,runpy
from pathlib import Path
from typing import Any
from distillschedule.runtime.gan_timing_range import GanFull5TimingRange
from distillschedule.runtime.nsys_range import GanFull5NsightRange

def _step(kwargs:dict[str,Any],name:str)->int:
    value=kwargs.get("global_step")
    if isinstance(value,bool) or not isinstance(value,int):raise RuntimeError(f"{name} did not receive integer global_step")
    return value

def _barrier()->None:
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():raise RuntimeError("exact world16 Nsight profiling requires initialized torch.distributed")
    dist.barrier()

def _install_hooks()->tuple[GanFull5TimingRange,GanFull5NsightRange]:
    timing=GanFull5TimingRange.from_env(); nsys=GanFull5NsightRange.from_env()
    import helios.utils.utils_helios_post as post
    original_generator,original_critic=post._generator_loss,post._critic_loss
    def generator_loss(*args:Any,**kwargs:Any):
        logical_step=_step(kwargs,"_generator_loss")
        if logical_step==1000:
            # Formal wall-clock: no synchronization/profiler overhead.
            timing.before_model_compute(logical_step)
            # Nsight is a separate diagnostic mode and deliberately adds
            # world16 barriers so the two node-local captures have exact bounds.
            if nsys.enabled:
                _barrier(); nsys.before_model_compute(logical_step); _barrier()
        return original_generator(*args,**kwargs)
    def critic_loss(*args:Any,**kwargs:Any):
        logical_step=_step(kwargs,"_critic_loss"); result=original_critic(*args,**kwargs)
        if logical_step==1004:
            timing.after_critic_commit(logical_step)
            if nsys.enabled:
                _barrier(); nsys.after_critic_commit(logical_step)
        return result
    post._generator_loss=generator_loss; post._critic_loss=critic_loss
    return timing,nsys

def main()->None:
    root=Path(os.environ["HELIOS_RUNTIME_ROOT"]).expanduser().resolve(); upstream=root/"train_helios.py"
    if not upstream.is_file():raise FileNotFoundError(f"pinned Helios entrypoint missing: {upstream}")
    timing,nsys=_install_hooks()
    completed=False
    try:
        runpy.run_path(str(upstream),run_name="__main__")
        completed=True
    finally:
        # Preserve the upstream exception as the root cause.  Receipt
        # completeness is only meaningful after the workload itself returns.
        if completed:
            timing.assert_complete()
            if nsys.enabled:nsys.assert_complete()
if __name__=="__main__":main()
