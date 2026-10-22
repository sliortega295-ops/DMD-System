"""Serialization views over actual resident C51 ModelBundle G/C state owners.

These are not engines, clocks, a training loop, or physical checkpoint admission.
The local boundary check does NOT drain transport, acknowledge input consumption,
or reconstruct critic shadows. Public fresh W4 training remains unavailable.
Native asset loaders reconstruct frozen bases/teacher; their admitted receipts
are bound here, not their model-sized weights. Outer checkpoint identity must
add the native contract, operators, inputs, horizon/window and physical topology.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

from distillschedule.data.livewan_inputs import _source_identity
from distillschedule.runtime.livewan_lora_kernel import _source_sha256
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, SHA256_RE
from distillschedule.workloads.livewan_crossscale import LORA_RANK, PIPELINE_8G8S_LAYOUT
from .checkpoint import _class, _validate_optimizer_layout
from .native_state import TrainableWeights, OptimizerState

LORA_SHA256 = "71bbcdc932e1290734df55e15cf5ed998bda1f41a3cb607684f28da71fcf0824"


def _require(condition, message):
    if not condition:
        raise ValueError("LiveWan resident state: " + message)


def _same(a, b):
    try:
        return fingerprint(a) == fingerprint(b)
    except (TypeError, ValueError):
        return False


def _source(root):
    # Reuse pinned gitlink/clean-source/license admission; additionally qualify
    # the actual LoRA code, which is not part of the data owner's source list.
    value = _source_identity(root)
    _require(file_sha256(Path(value["root"]) / "wanstreamer/lora.py") == LORA_SHA256,
             "original LoRA source hash changed")
    return {**value, "files": {**value["files"], "wanstreamer/lora.py": LORA_SHA256}}


def _reference_lora(source):
    path = Path(source["root"]) / "wanstreamer/lora.py"
    spec = importlib.util.spec_from_file_location("_distillschedule_original_lora_" + LORA_SHA256[:16], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoRALinear


class _ResidentRole:
    """A named serialization-only module/optimizer view, never an engine."""

    def __init__(self, bundle, role, source, original_lora):
        self.bundle, self.role, self.source, self.original_lora = bundle, role, source, original_lora
        self.module = bundle.student if role == "generator" else bundle.critic
        self.optimizer = bundle.generator_optimizer if role == "generator" else bundle.critic_optimizer
        self.module_field = "student" if role == "generator" else "critic"
        self.binding_field = role + "_optimizer_parameter_names"
        self.binding = dict(getattr(bundle, self.binding_field))
        self.parameters = [(name, id(p)) for name, p in self.module.named_parameters(remove_duplicate=False)]
        self.groups = [[id(p) for p in group["params"]] for group in self.optimizer.param_groups]
        self.reducer = bundle.critic_gradient_reducer
        self.identity = self._identity()

    def _base(self):
        metadata = self.bundle.metadata.get(self.module_field)
        _require(isinstance(metadata, dict) and metadata.get("placement") == "gpu_resident_replica",
                 "only resident G/C placement is supported")
        if self.role == "generator":
            value = metadata.get("state")
            _require(isinstance(value, dict) and value.get("kind") in {"base_weights", "stage1_state"},
                     "missing admitted Student base/stage1 receipt")
        else:
            value = {"kind": "critic_base", "sha256": metadata.get("base_sha256")}
            _require(metadata.get("lora_rank") == LORA_RANK, "critic LoRA rank policy changed")
        _require(isinstance(value.get("sha256"), str) and SHA256_RE.fullmatch(value["sha256"]),
                 "missing admitted base identity")
        return deepcopy(value)

    def _options(self):
        import torch
        _require(type(self.optimizer) is torch.optim.AdamW and len(self.optimizer.param_groups) == 1,
                 "actual fixed single-group AdamW owner required")
        group = self.optimizer.param_groups[0]
        expected = {"lr": 2e-6 if self.role == "generator" else 4e-5, "betas": (0., .99),
                    "eps": 1e-8, "weight_decay": 0., "amsgrad": False, "maximize": False,
                    "capturable": False, "differentiable": False}
        implementation = (self.bundle.metadata["student"].get("optimizer_impl")
                          if self.role == "generator" else "foreach_adamw")
        _require(implementation in {"foreach_adamw", "fused_adamw"}, "optimizer implementation changed")
        expected.update(foreach=True if implementation == "foreach_adamw" else None,
                        fused=True if implementation == "fused_adamw" else None)
        # New Torch versions may expose AdamW's always-on decoupling explicitly.
        if "decoupled_weight_decay" in group:
            expected["decoupled_weight_decay"] = True
        options = {key: value for key, value in group.items() if key != "params"}
        _require(_same(options, expected), "fixed AdamW group options changed")
        return deepcopy(options)

    def _lora(self):
        import torch
        result, parameters = {}, []
        for name, module in self.module.named_modules():
            cls = type(module)
            if cls.__name__ != "LoRALinear":
                continue
            # A file-spec import and the worker's package import create distinct
            # class objects from the same admitted code. Compare their original
            # constructor/forward code, not Python class IDs or a class name alone.
            original_forward = getattr(cls, "_distillschedule_original_forward", cls.forward)
            path = Path(self.source["root"]) / "wanstreamer/lora.py"
            constructor_code = getattr(cls.__init__, "__code__", None)
            forward_code = getattr(original_forward, "__code__", None)
            _require(cls.__bases__ == (torch.nn.Module,)
                     and constructor_code == self.original_lora.__init__.__code__
                     and forward_code == self.original_lora.forward.__code__
                     and Path(constructor_code.co_filename).resolve() == path
                     and Path(forward_code.co_filename).resolve() == path,
                     "LoRA module is not the qualified original class")
            _require(type(module.rank) is int and module.rank == LORA_RANK
                     and type(module.scale) in (int, float) and module.scale == 1.
                     and module.enabled is True, "LoRA rank/scale/enabled policy changed")
            _require(isinstance(module.base, torch.nn.Linear)
                     and tuple(module.a.shape) == (LORA_RANK, module.base.in_features)
                     and tuple(module.b.shape) == (module.base.out_features, LORA_RANK),
                     "LoRA rank/base dimensions changed")
            _require(getattr(module.forward, "__func__", None) is cls.forward,
                     "instance LoRA forward ownership changed")
            # Existing plan-qualified specializations preserve the original
            # class/Parameters. Bind their installed implementation as state
            # identity too; installation/numerical admission remains their owner.
            result[name] = {"rank": module.rank, "scale": module.scale, "enabled": module.enabled,
                            "forward_sha256": _source_sha256(cls.forward),
                            "operator": getattr(cls, "_distillschedule_kernel_name", "original")}
            parameters.extend((module.a, module.b))
        if self.role == "critic":
            expected = self.bundle.critic_lora_params
            _require(result and [id(p) for p in parameters] == [id(p) for p in expected],
                     "actual critic LoRA parameter inventory changed")
            metadata = self.bundle.metadata.get("critic", {})
            _require(metadata.get("lora_modules") == len(result)
                     and metadata.get("lora_numel") == sum(p.numel() for p in parameters),
                     "critic LoRA construction receipt changed")
        else:
            _require(not result, "resident Generator must own full parameters, not critic LoRA")
        return result

    def _identity(self):
        import torch
        from torch.distributed.tensor import DTensor
        from distillschedule.runtime.livewan_crossscale_worker import _bind_trajectory_optimizer_parameter_names
        layout = PIPELINE_8G8S_LAYOUT
        metadata = self.bundle.metadata
        _require(metadata.get("generator_ranks") == list(layout.generator_ranks)
                 and metadata.get("score_ranks") == list(layout.score_ranks)
                 and metadata.get("critic_ranks") == list(layout.score_ranks)
                 and (self.bundle.generator_group_size, self.bundle.score_group_size,
                      self.bundle.critic_group_size) == (8, 8, 8), "resident 8G/8Score owner topology changed")
        _require(getattr(self.bundle, self.module_field) is self.module
                 and getattr(self.bundle, self.role + "_optimizer") is self.optimizer,
                 "live module/optimizer owner changed")
        other = "critic" if self.role == "generator" else "generator"
        _require(getattr(self.bundle, "critic" if other == "critic" else "student") is None
                 and getattr(self.bundle, other + "_optimizer") is None
                 and not getattr(self.bundle, other + "_optimizer_parameter_names"),
                 "split-pool rank acquired an additional training owner")
        named = list(self.module.named_parameters(remove_duplicate=False))
        _require([(n, id(p)) for n, p in named] == self.parameters
                 and len({id(p) for _, p in named}) == len(named), "live named parameter inventory changed")
        _require([[id(p) for p in group["params"]] for group in self.optimizer.param_groups] == self.groups,
                 "optimizer parameter order/ownership changed")
        tensors = [p for _, p in named] + list(self.module.buffers())
        _require(all(not isinstance(p, DTensor) and not getattr(p, "_is_flat_param", False)
                     and p.layout == torch.strided for p in tensors),
                 "DTensor/flattened placements are not admitted by this resident view")
        trainables = {id(p) for _, p in named if p.requires_grad}
        _require(trainables and all(p.dtype == torch.float32 for _, p in named if p.requires_grad),
                 "resident trainables must be FP32")
        try:
            names = _bind_trajectory_optimizer_parameter_names(
                named_parameters=named, optimizer=self.optimizer, expected_parameter_ids=trainables)
        except AssertionError as exc:
            raise ValueError("LiveWan resident state: optimizer/model ownership changed") from exc
        _require(names == self.binding == getattr(self.bundle, self.binding_field)
                 and all(names[id(p)] == name for name, p in named if id(p) in names),
                 "construction-time resident name binding changed")
        if self.role == "generator":
            _require(len(trainables) == len(named), "resident Generator must own all model parameters")
        base = self._base()
        lora = self._lora()
        if self.role == "critic":
            _require(trainables == {id(p) for p in self.bundle.critic_lora_params},
                     "critic trainability differs from its actual LoRA owner")
            _require(all(p.dtype == torch.bfloat16 for _, p in named if not p.requires_grad),
                     "resident critic frozen base must retain BF16 representation")
        return {"role": self.role, "source": _source(self.source["root"]), "base": base,
                "model_class": _class(self.module), "lora": lora, "optimizer_options": self._options(),
                "parameters": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype),
                                "requires_grad": p.requires_grad} for n, p in named],
                "buffers": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
                            for n, p in self.module.named_buffers()]}

    def validate(self):
        _require(_same(self.identity, self._identity()), "source/base/LoRA/layout identity changed")
        _require(self.bundle.critic_gradient_reducer is self.reducer, "critic reducer owner changed")

    def validate_checkpoint_boundary(self):
        import torch
        self.validate()
        if self.reducer is not None:
            from distillschedule.runtime.livewan_grad_reducer import GradReadyAsyncGroupReducer
            _require(type(self.reducer) is GradReadyAsyncGroupReducer
                     and not self.reducer._active_backward and not self.reducer._pending_sync,
                     "local reducer has outstanding backward/reduction")
            for parameter in self.bundle.critic_lora_params:
                _require(parameter.grad is self.reducer._parameter_view.get(id(parameter)),
                         "reducer gradient storage view changed")
        _require(not any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                         for p in self.module.parameters()), "local owner has unconsumed gradients")
        # Deliberately no reducer.finish/zero_grad, transport wait, shadow sync,
        # cursor movement or claim of physical complete-cycle quiescence here.


class _Weights(TrainableWeights):
    def state_dict(self):
        self.engine.validate()
        return {**super().state_dict(), "identity": deepcopy(self.engine.identity)}

    def validate_state_dict(self, value):
        self.engine.validate()
        _require(isinstance(value, dict) and set(value) == {"tensors", "buffers", "training", "requires_grad", "identity"}
                 and _same(value["identity"], self.engine.identity), "saved source/base/LoRA identity changed")
        super().validate_state_dict(value)

    def validate_checkpoint_boundary(self):
        self.engine.validate_checkpoint_boundary()


class _Optimizer(OptimizerState):
    def _admit(self, value):
        import torch
        self.engine.validate()
        _require(isinstance(value, dict) and set(value) == {"class", "parameters", "optimizer", "identity"}
                 and _same(value["identity"], self.engine.identity), "saved optimizer owner identity changed")
        saved = value["optimizer"]
        _require(isinstance(saved, dict) and set(saved) == {"state", "param_groups"}
                 and isinstance(saved["state"], dict) and isinstance(saved["param_groups"], list)
                 and len(saved["param_groups"]) == len(self.engine.groups), "AdamW materialized inventory changed")
        ids = []
        for group, live in zip(saved["param_groups"], self.engine.optimizer.param_groups):
            _require(isinstance(group, dict) and isinstance(group.get("params"), list)
                     and len(group["params"]) == len(live["params"])
                     and _same({k: v for k, v in group.items() if k != "params"}, self.engine.identity["optimizer_options"]),
                     "saved fixed AdamW group policy changed")
            ids.extend(group["params"])
        _require(all(type(i) is int for i in ids) and all(type(i) is int for i in saved["state"])
                 and len(set(ids)) == len(ids)
                 and set(saved["state"]) == set(ids), "AdamW state parameter inventory changed")
        for slots in saved["state"].values():
            _require(isinstance(slots, dict) and set(slots) == {"step", "exp_avg", "exp_avg_sq"},
                     "AdamW materialized moment/step slots changed")
            step = slots["step"]
            _require(type(step) is torch.Tensor and step.dtype == torch.float32 and step.ndim == 0
                     and bool(torch.isfinite(step)),
                     "AdamW scalar step must be finite nonnegative integral FP32")
            step_value = step.item()
            _require(step_value >= 0 and step_value.is_integer(),
                     "AdamW scalar step must be finite nonnegative integral FP32")
        _validate_optimizer_layout(self.engine.optimizer, saved, "LiveWan AdamW")
        for group, live in zip(saved["param_groups"], self.engine.optimizer.param_groups):
            for parameter, parameter_id in zip(live["params"], group["params"]):
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = saved["state"][parameter_id][name]
                    _require(type(moment) is torch.Tensor and moment.layout == torch.strided
                             and moment.stride() == parameter.stride(), "AdamW moment storage layout changed")

    def state_dict(self):
        value = {**super().state_dict(), "identity": deepcopy(self.engine.identity)}
        self._admit(value)
        return value

    def _stage(self, value):
        self._admit(value)
        return super()._stage(value)


def model_bundle_state(bundle, *, source_root):
    """Return named views over one actual resident G or C rank's existing owners.

    Requires native-loader base receipts and clean pinned original LoRA source.
    The caller retains physical progress, transport, shadow and input ownership;
    these views alone must not enable public native W4 checkpoint/resume.
    """
    import torch
    from distillschedule.runtime.livewan_crossscale_worker import ModelBundle
    _require(type(bundle) is ModelBundle, "actual ModelBundle owner required")
    pairs = [("generator", bundle.student, bundle.generator_optimizer),
             ("critic", bundle.critic, bundle.critic_optimizer)]
    active = [(role, module, optimizer) for role, module, optimizer in pairs
              if module is not None or optimizer is not None]
    _require(len(active) == 1 and isinstance(active[0][1], torch.nn.Module)
             and type(active[0][2]) is torch.optim.AdamW, "one actual split-pool resident G/C AdamW owner required")
    source = _source(source_root)
    role = _ResidentRole(bundle, active[0][0], source, _reference_lora(source))
    return {role.role + ".weights": _Weights(role), role.role + ".optimizer": _Optimizer(role)}
