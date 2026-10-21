"""Restart adapters for existing engine-owned model/Adam/scheduler/EMA state.

These are serialization views, not replacement engines or a new training loop.
Frozen bases are reconstructed by the native asset loader; the caller must bind
those immutable checkpoint identities in the outer checkpoint manifest.
All loads preserve live Parameters, hooks and gradient-buffer views.
"""
from __future__ import annotations
from copy import copy, deepcopy

from .checkpoint import _class, _validate_layout
from .tensor_state import layout


def _named(engine):
    from distillschedule.open_workload.torch_backend import unwrap_module
    model = unwrap_module(engine.module)
    parameters = dict(model.named_parameters())
    owned = {id(p) for group in engine.optimizer.param_groups for p in group["params"]}
    names = {id(p): name for name, p in parameters.items()}
    if not owned <= names.keys():
        raise ValueError("native optimizer owns parameters outside its model")
    return model, {name: p for name, p in parameters.items() if id(p) in owned}, names


class TrainableWeights:
    def __init__(self, engine):
        self.engine = engine

    def state_dict(self):
        model, selected, _ = _named(self.engine)
        return {"tensors": {name: p.detach() for name, p in selected.items()},
                "buffers": dict(model.named_buffers()),
                "training": {name: module.training for name, module in model.named_modules()},
                "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()}}

    def validate_state_dict(self, value):
        current = self.state_dict()
        _validate_layout(current["tensors"], value["tensors"], "native trainables")
        _validate_layout(current["buffers"], value["buffers"], "native buffers")
        if current["requires_grad"] != value["requires_grad"]:
            raise ValueError("native trainability changed")
        if (current["training"].keys() != value["training"].keys()
                or any(type(mode) is not bool for mode in value["training"].values())):
            raise ValueError("native module modes changed")

    def load_state_dict(self, value):
        import torch
        self.validate_state_dict(value)
        model, selected, _ = _named(self.engine)
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, parameter in selected.items():
                parameter.copy_(value["tensors"][name].to(parameter.device))
            for name, buffer in buffers.items():
                buffer.copy_(value["buffers"][name].to(buffer.device))
        for name, module in model.named_modules():
            module.training = value["training"][name]


class OptimizerState:
    def __init__(self, engine):
        self.engine = engine

    def _names(self):
        _, _, names = _named(self.engine)
        return [[names[id(p)] for p in group["params"]] for group in self.engine.optimizer.param_groups]

    def state_dict(self):
        optimizer = self.engine.optimizer
        return {"class": _class(optimizer),
                "parameters": self._names(), "optimizer": optimizer.state_dict()}

    def _stage(self, value):
        import torch
        optimizer = self.engine.optimizer
        if (value["parameters"] != self._names()
                or value["class"] != _class(optimizer)):
            raise ValueError("native optimizer parameter ownership changed")
        staged = copy(optimizer)
        staged.param_groups = [dict(group, params=list(group["params"])) for group in optimizer.param_groups]
        staged.state = {}
        staged.load_state_dict(value["optimizer"])
        for parameter, states in staged.state.items():
            for tensor in states.values():
                if isinstance(tensor, torch.Tensor) and tensor.ndim:
                    _validate_layout(parameter, tensor, "native optimizer moment")
        return staged

    def validate_state_dict(self, value):
        self._stage(value)

    def load_state_dict(self, value):
        staged = self._stage(value)
        self.engine.optimizer.state = staged.state
        self.engine.optimizer.param_groups = staged.param_groups


class EngineClocks:
    _COUNTERS = ("global_steps", "global_samples", "_step_calls")

    def __init__(self, engine):
        self.engine = engine

    def state_dict(self):
        return {"counters": {key: getattr(self.engine, key) for key in self._COUNTERS},
                "policy": {"global_batch": self.engine.train_batch_size(),
                           "gradient_clip": self.engine._gradient_clip,
                           "dp_world_size": self.engine.dp_world_size},
                "accumulation_boundary": self.engine._gradient_accumulation_boundary}

    def validate_checkpoint_boundary(self):
        import torch
        from distillschedule.open_workload.torch_backend import physical_tensor
        if getattr(self.engine, "_active_backward", False) or not self.engine._gradient_accumulation_boundary:
            raise ValueError("native checkpoint has an outstanding backward/accumulation")
        # The Resident reducer owns persistent zero-gradient views. They must
        # remain attached; converting them to None would break the next step.
        if any(p.grad is not None and bool(torch.count_nonzero(physical_tensor(p.grad)))
               for p in self.engine.module.parameters()):
            raise ValueError("native checkpoint has unconsumed gradients")

    def validate_state_dict(self, value):
        if (value["policy"] != self.state_dict()["policy"]
                or set(value["counters"]) != set(self._COUNTERS)
                or any(type(item) is not int or item < 0 for item in value["counters"].values())
                or value["accumulation_boundary"] is not True):
            raise ValueError("native engine clock/policy is not a complete update boundary")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        for key, item in value["counters"].items():
            setattr(self.engine, key, item)
        self.engine._gradient_accumulation_boundary = True


class EMAState:
    _FIXED = ("decay", "min_decay", "update_after_step", "device", "implementation")

    def __init__(self, ema, *, named_parameters=None):
        self.ema = ema
        if named_parameters is None:
            if not hasattr(ema, "_names"):
                raise ValueError("native EMA requires semantic parameter names from its actual owner")
            self.names = list(ema._names)
        else:
            names = {id(p): name for name, p in named_parameters.items()}
            self.names = [names[id(p)] for p in ema._parameters]
            if hasattr(ema, "_names") and self.names != list(ema._names):
                raise ValueError("native EMA name order differs from its actual Generator")

    def state_dict(self):
        from torch.distributed.tensor import DTensor
        return {"names": self.names,
                "parameters": [layout(p) if isinstance(p, DTensor) else
                               {"shape": list(p.shape), "dtype": str(p.dtype)} for p in self.ema._parameters],
                "shadows": [shadow.detach() for shadow in self.ema._shadows],
                "policy": {key: getattr(self.ema, key) for key in self._FIXED if hasattr(self.ema, key)},
                "optimization_step": self.ema.optimization_step,
                "cur_decay_value": self.ema.cur_decay_value}

    def validate_state_dict(self, value):
        current = self.state_dict()
        if any(current[key] != value[key] for key in ("names", "parameters", "policy")):
            raise ValueError("native EMA ownership/layout/policy changed")
        _validate_layout(current["shadows"], value["shadows"], "native EMA shadows")
        if type(value["optimization_step"]) is not int or value["optimization_step"] < 0:
            raise ValueError("native EMA clock changed")

    def load_state_dict(self, value):
        import torch
        self.validate_state_dict(value)
        with torch.no_grad():
            for shadow, restored in zip(self.ema._shadows, value["shadows"], strict=True):
                shadow.copy_(restored.to(shadow.device))
        self.ema.optimization_step = value["optimization_step"]
        self.ema.cur_decay_value = deepcopy(value["cur_decay_value"])


def engine_state(engines, *, ema=None):
    """Named serialization views over actual already-built native engines."""
    state = {}
    for role, engine in engines.items():
        if not isinstance(role, str) or not role or "." in role:
            raise ValueError("native role names must be explicit non-dotted strings")
        state.update({role + ".weights": TrainableWeights(engine),
                      role + ".optimizer": OptimizerState(engine),
                      role + ".clocks": EngineClocks(engine)})
        if engine.lr_scheduler is not None:
            state[role + ".scheduler"] = engine.lr_scheduler
    if ema is not None:
        if "generator" not in engines:
            raise ValueError("native EMA requires an explicit Generator owner")
        state["ema"] = EMAState(ema, named_parameters=_named(engines["generator"])[1])
    return state
