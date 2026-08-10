#!/usr/bin/env python3
"""Run pinned Helios GAN math with diagnostics-only boundary hooks."""
from __future__ import annotations
import os,runpy
from pathlib import Path
from typing import Any
from distillschedule.runtime.nsys_range import GanFull5NsightRange

def _step(kwargs:dict[str,Any],name:str)->int:
    value=kwargs.get("global_step")
    if isinstance(value,bool) or not isinstance(value,int):raise RuntimeError(f"{name} did not receive integer global_step")
    return value

def _barrier()->None:
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():raise RuntimeError("exact world16 Nsight profiling requires initialized torch.distributed")
    dist.barrier()

def _install_nsys_hooks()->GanFull5NsightRange:
    gate=GanFull5NsightRange.from_env()
    if not gate.enabled:return gate
    import helios.utils.utils_helios_post as post
    original_generator,original_critic=post._generator_loss,post._critic_loss
    def generator_loss(*args:Any,**kwargs:Any):
        logical_step=_step(kwargs,"_generator_loss")
        if logical_step==1000:
            _barrier(); gate.before_model_compute(logical_step); _barrier()
        return original_generator(*args,**kwargs)
    def critic_loss(*args:Any,**kwargs:Any):
        logical_step=_step(kwargs,"_critic_loss"); result=original_critic(*args,**kwargs)
        if logical_step==1004:
            _barrier(); gate.after_critic_commit(logical_step)
        return result
    post._generator_loss=generator_loss; post._critic_loss=critic_loss; return gate

def main()->None:
    root=Path(os.environ["HELIOS_RUNTIME_ROOT"]).expanduser().resolve(); upstream=root/"train_helios.py"
    if not upstream.is_file():raise FileNotFoundError(f"pinned Helios entrypoint missing: {upstream}")
    gate=_install_nsys_hooks()
    try:runpy.run_path(str(upstream),run_name="__main__")
    finally:
        if gate.enabled:gate.assert_complete()
if __name__=="__main__":main()
