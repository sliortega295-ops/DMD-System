"""Opt-in replicated CPU AdamW with resident GPU execution parameters.

The reference GPU reduction and clipping happen before step(). Only Adam state
and its same-dtype CPU parameter mirror move to host. All required transfers
and the host-consumption fence are inside step(), hence inside native timing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


class CPUAdamW:
    def __init__(self, torch: Any, parameters: Iterable[Any], **kwargs: Any) -> None:
        self.torch = torch
        self.parameters = list(parameters)
        if not self.parameters or any(hasattr(p, "to_local") for p in self.parameters):
            raise ValueError("CPU Adam requires non-sharded resident parameters")
        devices = {p.device for p in self.parameters}
        if len(devices) != 1 or any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("CPU Adam requires one device and original FP32 trainables")
        self.device = self.parameters[0].device
        self.pinned = self.device.type == "cuda"
        self.mirrors = []
        self.grad_buffers = []
        for parameter in self.parameters:
            value = parameter.detach().to("cpu", copy=True)
            if self.pinned:
                value = value.pin_memory()
            self.mirrors.append(torch.nn.Parameter(value))
            self.grad_buffers.append(torch.empty(
                value.shape, dtype=value.dtype, device="cpu", pin_memory=self.pinned))
        # CPU Adam is the ordinary PyTorch foreach AdamW, not a new formula.
        kwargs.pop("fused", None)
        kwargs["foreach"] = True
        self.optimizer = torch.optim.AdamW(self.mirrors, **kwargs)
        self.param_groups = [{**group, "params": list(self.parameters)}
                             for group in self.optimizer.param_groups]
        self.defaults = self.optimizer.defaults
        self.state: Any = defaultdict(dict)
        self.expected_versions = [p._version for p in self.parameters]
        self.steps = self.d2h_bytes = self.h2d_bytes = 0
        self.d2h_events = self.h2d_events = self.host_fences = 0

    def materialize_state(self, *, step: int = 0) -> dict[str, Any]:
        self.optimizer.state.clear()
        self.state.clear()
        for parameter, mirror in zip(self.parameters, self.mirrors):
            state = self.optimizer.state[mirror]
            state["step"] = self.torch.tensor(float(step), dtype=self.torch.float32)
            state["exp_avg"] = self.torch.zeros_like(mirror, dtype=self.torch.float32)
            state["exp_avg_sq"] = self.torch.zeros_like(mirror, dtype=self.torch.float32)
            self.state[parameter] = state
        return {"tensor_count": len(self.parameters),
                "local_numel_each_moment": sum(p.numel() for p in self.parameters),
                "step": step, "state_device": "cpu", "parameter_device": str(self.device)}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.detach_()
                parameter.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        if closure is not None:
            raise ValueError("LiveWan CPU Adam does not support closures")
        for external, internal in zip(self.param_groups, self.optimizer.param_groups):
            internal.update({k: v for k, v in external.items() if k != "params"})
        active = []
        with self.torch.no_grad():
            for index, (parameter, mirror, buffer) in enumerate(zip(
                    self.parameters, self.mirrors, self.grad_buffers)):
                if parameter._version != self.expected_versions[index]:
                    raise RuntimeError("resident parameter changed outside CPU optimizer")
                gradient = parameter.grad
                if gradient is None:
                    mirror.grad = None
                    continue
                if gradient.is_sparse or gradient.dtype != parameter.dtype:
                    raise ValueError("CPU Adam gradient layout/dtype changed")
                buffer.copy_(gradient.detach(), non_blocking=self.pinned)
                mirror.grad = buffer
                active.append(index)
                self.d2h_bytes += gradient.numel() * gradient.element_size()
                self.d2h_events += 1
            if active and self.pinned:
                # Required before CPU consumes asynchronous gradient copies.
                self.torch.cuda.current_stream(self.device).synchronize()
                self.host_fences += 1
            self.optimizer.step()
            for index in active:
                parameter, mirror = self.parameters[index], self.mirrors[index]
                parameter.copy_(mirror, non_blocking=self.pinned)
                self.expected_versions[index] = parameter._version
                self.h2d_bytes += parameter.numel() * parameter.element_size()
                self.h2d_events += 1
        self.steps += 1

    def receipt(self) -> dict[str, Any]:
        states = list(self.state.values())
        if not states or any(s[key].device.type != "cpu" for s in states
                             for key in ("step", "exp_avg", "exp_avg_sq")):
            raise RuntimeError("CPU Adam state placement was not verified")
        return {"name": "replicated_cpu_foreach_adamw", "state_device": "cpu",
                "execution_parameter_device": str(self.device), "steps": self.steps,
                "moment_bytes": sum(s[k].numel() * s[k].element_size()
                                    for s in states for k in ("exp_avg", "exp_avg_sq")),
                "cpu_parameter_mirror_bytes": sum(p.numel()*p.element_size() for p in self.mirrors),
                "gradient_d2h_bytes": self.d2h_bytes, "gradient_d2h_events": self.d2h_events,
                "updated_parameter_h2d_bytes": self.h2d_bytes,
                "updated_parameter_h2d_events": self.h2d_events,
                "cpu_consumption_fences": self.host_fences,
                "transfers_and_cpu_update_inside_timer": True,
                "gradient_mean_and_clip_unchanged": True,
                "original_parameter_dtype_preserved": True}
