"""Reusable native Torch-function binding, including existing kernel capabilities.

A workload supplies math/input functions, not optimization variants. The library
adds reference/checkpoint implementations only when the declared pure/replay-safe
contract permits them. This does not rewrite an opaque training loop or migrate
arbitrary model meshes. GPU fences are explicit; no CPU result is GPU evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import inspect
from typing import Callable, Mapping

from distillschedule.core.contracts import require
from .capabilities import Capability, CapabilityRegistry, ExecutableOption


@dataclass(frozen=True)
class TensorFunction:
    operator: str
    function: Callable
    arguments: Callable
    implementation_identity: Mapping
    workspace_upper_bound: int
    ranks: tuple[int,...]=(0,)
    devices: tuple[str,...]=("cpu",)
    states: tuple=()
    lifetimes: tuple=()
    replay_safe: bool=False
    concurrent_safe: bool=False


def register_tensor_function(registry: CapabilityRegistry, port: TensorFunction):
    import torch
    require(len(port.ranks)==len(port.devices),"explicit rank/device mapping required",path="program.torch")
    require(all(d=="cpu" or d.startswith("cuda:") for d in port.devices),"unsupported device backend",path="program.torch")
    require(callable(port.function) and callable(port.arguments),"native math/input functions required",path="program.torch")
    require(len(set(port.devices))==len(port.devices) or port.devices==("cpu",),
            "do not alias multiple logical devices to the same physical device",path="program.torch")
    def synchronize():
        for device in port.devices:
            if device.startswith("cuda:"):torch.cuda.synchronize(torch.device(device))
    def bind(mode):
        def factory(op,call):
            def invoke(ctx):
                args=port.arguments(ctx)
                require(isinstance(args,tuple),"arguments binding must return a tuple",path="program.torch")
                grad=op.grad_mode.value!="no_grad"
                with torch.set_grad_enabled(grad):
                    if mode=="checkpoint":
                        from torch.utils.checkpoint import checkpoint
                        return checkpoint(port.function,*args,use_reentrant=False,preserve_rng_state=True)
                    return port.function(*args)
            resources=tuple(f"gpu:{r}" for r in port.ranks)
            if not port.concurrent_safe:resources+=("executor:serial",)
            resources+=tuple("physical:"+d for d in port.devices if d.startswith("cuda:"))
            return (ExecutableOption(f"{port.operator}.{mode}",port.ranks,resources,invoke,synchronize,
                {"native":dict(port.implementation_identity),"mode":mode,"torch":torch.__version__,
                 "devices":list(port.devices),"concurrent_safe":port.concurrent_safe},states=port.states,lifetimes=port.lifetimes,
                workspace_upper_bound=port.workspace_upper_bound,is_reference=mode=="reference"),)
        return factory
    registry.register(Capability(f"{port.operator}.reference",port.operator,{},
        ("no_grad","grad","external_grad"),bind("reference")))
    if port.replay_safe:
        registry.register(Capability(f"{port.operator}.checkpoint",port.operator,
            {"replay_safe":True},("grad",),bind("checkpoint")))


def register_lora_epilogue(registry, arguments, *, device, source_identity, workspace_upper_bound):
    """Reuse the EXISTING Wan fusion based on operation/dtype properties, not model name.

    This is the output scale/add/cast only, not LoRA projection or training-loop
    replacement. The fused implementation is deliberately no-grad and CUDA/BF16.
    """
    import torch
    def reference(base,delta,scale):return (base.float()+delta*scale).to(base.dtype)
    register_tensor_function(registry,TensorFunction("lora.epilogue",reference,arguments,
        source_identity,workspace_upper_bound,devices=(device,)))
    if not device.startswith("cuda:"):return
    try:import triton
    except ImportError:return
    from distillschedule.runtime.wan13b_lora_epilogue import fused_lora_epilogue
    def factory(op,call):
        def invoke(ctx):
            base,delta,scale=arguments(ctx)
            require(base.dtype==torch.bfloat16 and delta.dtype==torch.float32,
                    "fused epilogue requires BF16 base/FP32 delta",path="program.torch")
            require(base.device==delta.device==torch.device(device),"tensor/device mismatch",path="program.torch")
            with torch.no_grad():return fused_lora_epilogue(base,delta,scale)
        return (ExecutableOption("lora.epilogue.wan-fused",(0,),("gpu:0","executor:serial"),
            invoke,lambda:torch.cuda.synchronize(device),
            {"source":dict(source_identity),"native_function":"wan13b_lora_epilogue.fused_lora_epilogue",
             "triton":triton.__version__},workspace_upper_bound=workspace_upper_bound),)
    registry.register(Capability("lora.epilogue.wan-fused","lora.epilogue",
        {"dtype":"bfloat16","delta_dtype":"float32"},("no_grad",),factory))
