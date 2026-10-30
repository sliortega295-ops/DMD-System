"""Serialization views over actual ModelBundle owners, not training engines.

These are not engines, clocks, a training loop, or physical checkpoint admission.
The local boundary check does NOT drain transport, acknowledge input consumption,
or reconstruct critic shadows. The separate reference factory below is NOT a
relaxation of resident split-pool admission or a C51 pipeline restart seam.
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
    def __init__(self, engine, *, in_place_restore=False):
        # Only the C51 resident owner opts in. The generic/default serialization
        # and the reference FSDP/DTensor adapter retain their original staging.
        _require(type(in_place_restore) is bool
                 and (not in_place_restore or type(engine) is _ResidentRole),
                 "in-place Adam restore requires the exact resident owner")
        super().__init__(engine)
        self._in_place_restore = in_place_restore

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
                    self._moment_layout(parameter, moment)

    def _moment_layout(self, parameter, moment):
        import torch
        _require(type(moment) is torch.Tensor and moment.layout == torch.strided
                 and moment.stride() == parameter.stride(), "AdamW moment storage layout changed")

    def state_dict(self):
        if self._in_place_restore:
            self._in_place_parameters()
        value = {**super().state_dict(), "identity": deepcopy(self.engine.identity)}
        self._admit(value)
        return value

    def _in_place_parameters(self):
        self.engine.validate()
        optimizer = self.engine.optimizer
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        _require({id(p) for p in optimizer.state} == {id(p) for p in parameters},
                 "live materialized AdamW parameter inventory changed")
        return parameters

    def _stage(self, value):
        self._admit(value)
        return super()._stage(value)

    def _in_place_slots(self, value):
        import torch
        from distillschedule.open_workload.torch_backend import storage_key
        # Run the original C51/source/clock/finite/raw-layout checks before any
        # copy. _stage's ownership check must not be lost when staging is absent.
        self._admit(value)
        optimizer = self.engine.optimizer
        _require(value["parameters"] == self._names() and value["class"] == _class(optimizer),
                 "native optimizer parameter ownership changed")
        parameters = self._in_place_parameters()
        occupied = {storage_key(t)[:2] for t in
                    (*self.engine.module.parameters(), *self.engine.module.buffers(),
                     *(p.grad for p in parameters if p.grad is not None)) if t.numel()}
        slots = []
        for group, saved_group in zip(optimizer.param_groups, value["optimizer"]["param_groups"], strict=True):
            for parameter, parameter_id in zip(group["params"], saved_group["params"], strict=True):
                live = optimizer.state[parameter]
                _require(type(live) is dict and set(live) == {"step", "exp_avg", "exp_avg_sq"},
                         "live materialized AdamW slots changed")
                saved = value["optimizer"]["state"][parameter_id]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    source, target = saved[key], live[key]
                    device = (parameter.device if key != "step" or group.get("fused")
                              else torch.device("cpu"))
                    _require(type(source) is torch.Tensor and source.device.type == "cpu"
                             and source.layout == torch.strided
                             and type(target) is torch.Tensor and target.layout == torch.strided
                             and target.device == device and not target.requires_grad
                             and target.shape == source.shape and target.dtype == source.dtype
                             and target.stride() == source.stride(),
                             "live materialized AdamW slot layout/device changed")
                    key_storage = storage_key(target)[:2]
                    _require(not target.numel() or key_storage not in occupied,
                             "live materialized AdamW slot aliases owned storage")
                    if target.numel():
                        occupied.add(key_storage)
                    slots.append((source, target))
        # CPU checkpoint tensors must not alias a destination that a previous
        # copy could overwrite. Real pack/load already creates separate storage.
        _require(all(not source.numel() or storage_key(source)[:2] not in occupied
                     for source, _ in slots), "saved AdamW slot aliases live owned storage")
        return slots

    def validate_state_dict(self, value):
        if not self._in_place_restore:
            return super().validate_state_dict(value)
        self._in_place_slots(value)

    def load_state_dict(self, value):
        if not self._in_place_restore:
            return super().load_state_dict(value)
        import torch
        slots = self._in_place_slots(value)
        # checkpoint.unpack_state has completed its all-rank premutation gate.
        # No .to(device), load_state_dict clone, state.clear or group replacement:
        # copy all moments/scalar steps into their already-materialized storage.
        with torch.no_grad():
            for source, target in slots:
                target.copy_(source)


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


class _ReferenceRole(_ResidentRole):
    """All-role reference serialization view; private mesh-level CPU test seam.

    The public factory separately enforces physical world16 CUDA/NCCL. This
    view can also qualify serialization on a genuine tiny CPU DeviceMesh; that
    is explicitly not FSDP2 execution or a physical W4 topology proof.
    """
    def __init__(self, bundle, role, source, original_lora, versions, restore_versions=None):
        self.versions, self.restore_versions = versions, restore_versions
        self.teacher = bundle.teacher
        self.teacher_parameters = [(n, id(p)) for n, p in self.teacher.named_parameters()]
        self.owners = (bundle.generator_group, bundle.score_group, bundle.critic_group)
        super().__init__(bundle, role, source, original_lora)

    def _base(self):
        metadata = self.bundle.metadata.get(self.module_field)
        placement = "world16_fsdp2_keep_unsharded" if self.role == "generator" else "gpu_resident_replica"
        _require(isinstance(metadata, dict) and metadata.get("placement") == placement,
                 "reference G FSDP2/C replicated placement changed")
        if self.role == "generator":
            value = metadata.get("state")
            _require(isinstance(value, dict) and value.get("kind") in {"base_weights", "stage1_state"},
                     "missing admitted reference Student receipt")
        else:
            value = {"kind": "critic_base", "sha256": metadata.get("base_sha256")}
            _require(metadata.get("lora_rank") == LORA_RANK, "reference critic LoRA rank changed")
        _require(isinstance(value.get("sha256"), str) and SHA256_RE.fullmatch(value["sha256"]),
                 "missing admitted reference base identity")
        return deepcopy(value)

    def _lora(self):
        result = super()._lora()
        _require(all(item["operator"] == "original" and item["forward_sha256"] == _source_sha256(self.original_lora.forward)
                     for item in result.values()), "reference forbids optimized/patched LoRA implementations")
        return result

    def _identity(self):
        import torch
        from torch.distributed.tensor import DTensor, Shard
        from .tensor_state import layout
        from distillschedule.runtime.livewan_crossscale_worker import _bind_trajectory_optimizer_parameter_names, _trajectory_livewan_state_name
        bundle, metadata = self.bundle, self.bundle.metadata
        _require(bundle.student.training and not bundle.critic.training and not self.teacher.training,
                 "reference original Generator/Critic/Teacher train/eval policy changed")
        ranks = metadata.get("generator_ranks")
        _require(isinstance(ranks, list) and ranks == list(range(len(ranks))) and ranks
                 and metadata.get("score_ranks") == metadata.get("critic_ranks") == ranks
                 and (bundle.generator_group_size, bundle.score_group_size, bundle.critic_group_size) == (len(ranks),) * 3,
                 "reference all-role topology changed")
        _require(getattr(bundle, self.module_field) is self.module
                 and getattr(bundle, self.role + "_optimizer") is self.optimizer
                 and bundle.teacher is self.teacher
                 and [(n, id(p)) for n, p in self.teacher.named_parameters()] == self.teacher_parameters
                 and all(a is b for a, b in zip(self.owners, (bundle.generator_group, bundle.score_group, bundle.critic_group))),
                 "reference native owner changed")
        _require(bundle.critic_shadow is None and not bundle.critic_shadow_lora_params
                 and all(getattr(bundle, name) is None for name in ("critic_gradient_reducer", "score_local_group",
                     "score_leader_group", "runahead_host_group", "critic_shadow_sync_group"))
                 and not bundle.score_local_ranks and not bundle.score_leader_ranks,
                 "reference acquired a pipeline/reducer/shadow owner")
        named = list(self.module.named_parameters(remove_duplicate=False))
        _require([(n, id(p)) for n, p in named] == self.parameters and len({id(p) for _, p in named}) == len(named)
                 and [[id(p) for p in g["params"]] for g in self.optimizer.param_groups] == self.groups,
                 "reference named parameter/Adam ownership changed (reshard before snapshot)")
        trainables = {id(p) for _, p in named if p.requires_grad}
        _require(trainables and all(p.dtype == torch.float32 for _, p in named if p.requires_grad),
                 "reference trainables must retain FP32")
        names = _bind_trajectory_optimizer_parameter_names(named_parameters=named, optimizer=self.optimizer,
                                                           expected_parameter_ids=trainables)
        _require(names == self.binding == getattr(bundle, self.binding_field)
                 and all(names[id(p)] == _trajectory_livewan_state_name(n) for n, p in named if id(p) in names),
                 "reference construction-time canonical name binding changed")
        def sharded(parameter, dtype):
            return (isinstance(parameter, DTensor) and parameter.dtype == dtype
                    and parameter.device_mesh.mesh.tolist() == ranks and parameter.device_mesh.ndim == 1
                    and len(parameter.placements) == 1 and isinstance(parameter.placements[0], Shard)
                    and parameter.placements[0].dim == 0)
        teacher_named = list(self.teacher.named_parameters())
        _require(teacher_named and not self.teacher.training and all(not p.requires_grad and sharded(p, torch.bfloat16)
                 for _, p in teacher_named), "reference Teacher must be frozen BF16 FSDP2 shards")
        _require(metadata.get("teacher", {}).get("placement") == "fsdp2_group16_reshard"
                 and metadata["teacher"].get("mesh_ranks") == ranks,
                 "reference Teacher construction receipt changed")
        if self.role == "generator":
            _require(len(trainables) == len(named) and all(sharded(p, torch.float32) for _, p in named),
                     "reference Generator must own every original FP32 DTensor shard")
        else:
            _require(all(type(p) is torch.nn.Parameter and p.layout == torch.strided for _, p in named)
                     and trainables == {id(p) for p in bundle.critic_lora_params}
                     and all(p.dtype == torch.bfloat16 for _, p in named if not p.requires_grad),
                     "reference critic must own replicated FP32 LoRA/BF16 frozen parameters")
        return {"role": self.role, "source": _source(self.source["root"]), "base": self._base(),
                "model_class": _class(self.module), "lora": self._lora(), "optimizer_options": self._options(),
                "ranks": ranks, "teacher": [{"name": n, "layout": layout(p)} for n, p in teacher_named],
                "parameters": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype), "requires_grad": p.requires_grad,
                                **({"dtensor": layout(p)} if isinstance(p, DTensor) else {"stride": list(p.stride())})}
                               for n, p in named],
                "buffers": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in self.module.named_buffers()]}

    def validate_checkpoint_boundary(self):
        self.validate()
        cycle = self.versions.get("generator")
        _require(type(cycle) is int and 0 <= cycle <= 64 and self.versions == {"generator": cycle, "critic": 5 * cycle, "ema": 0}
                 and all(type(v) is int for v in self.versions.values()), "reference actual Gk/C5k/EMA0 versions changed")
        _require(all(p.grad is None for model in (self.bundle.student, self.bundle.critic, self.teacher)
                     for p in model.parameters()) and all(p.grad is None for group in self.optimizer.param_groups for p in group["params"]),
                 "reference native owner has unconsumed gradients")
        _ReferenceOptimizer(self).state_dict()


class _ReferenceWeights(_Weights):
    def validate_state_dict(self, value):
        import torch
        from distillschedule.open_workload.torch_backend import physical_tensor, unwrap_module
        super().validate_state_dict(value)
        model = unwrap_module(self.engine.module)
        _require(value["training"] == {name: module.training for name, module in model.named_modules()},
                 "reference original module training modes changed")
        _require(all(bool(torch.isfinite(physical_tensor(t)).all()) for t in
                     (*value["tensors"].values(), *value["buffers"].values())),
                 "reference checkpoint has nonfinite weights/buffers")


class _ReferenceOptimizer(_Optimizer):
    def _admit(self, value):
        import torch
        from distillschedule.open_workload.torch_backend import physical_tensor
        super()._admit(value)
        _require(all(slots["step"].device.type == "cpu" and slots["step"].layout == torch.strided
                     and bool((physical_tensor(slots["exp_avg_sq"]) >= 0).all())
                     for slots in value["optimizer"]["state"].values()),
                 "reference AdamW scalar device or negative second moment changed")

    def _moment_layout(self, parameter, moment):
        import torch
        from torch.distributed.tensor import DTensor
        from .tensor_state import layout
        from distillschedule.open_workload.torch_backend import physical_tensor
        if isinstance(parameter, DTensor):
            _require(isinstance(moment, DTensor) and layout(moment) == layout(parameter),
                     "reference AdamW DTensor moment layout changed")
        else:
            super()._moment_layout(parameter, moment)
        _require(bool(torch.isfinite(physical_tensor(moment)).all()), "reference AdamW nonfinite moment")

    def _steps(self, value, versions):
        from distillschedule.open_workload.torch_backend import physical_tensor
        expected = versions[self.engine.role]
        saved = value["optimizer"]
        for group, live in zip(saved["param_groups"], self.engine.optimizer.param_groups, strict=True):
            for parameter_id, parameter in zip(group["params"], live["params"], strict=True):
                name = self.engine.binding[id(parameter)]
                slots = saved["state"][parameter_id]
                # The original rollout projects text under no_grad, and its
                # differentiable recompute reuses that detached context. Adam
                # owns/materializes these parameters but never updates them.
                detached = self.engine.role == "generator" and name.startswith("text_embedding.")
                _require(slots["step"].item() == (0 if detached else expected),
                         "reference AdamW step/version mismatch: " + name)
                if detached:
                    _require(all(not bool(physical_tensor(slots[key]).count_nonzero())
                                 for key in ("exp_avg", "exp_avg_sq")),
                             "reference detached text_embedding AdamW moment must remain zero")

    def state_dict(self):
        value = super().state_dict()
        self._steps(value, self.engine.versions)
        return value

    def _stage(self, value):
        self._admit(value)
        self._steps(value, self.engine.restore_versions if self.engine.restore_versions is not None else self.engine.versions)
        return OptimizerState._stage(self, value)


def reference_model_bundle_state(bundle, *, source_root, versions, restore_versions=None):
    """Named in-place native views for ORIGINAL world16 baseline, NOT C51."""
    import torch
    import torch.distributed as dist
    from distillschedule.runtime.livewan_crossscale_worker import ModelBundle
    _require(type(bundle) is ModelBundle and isinstance(bundle.student, torch.nn.Module)
             and isinstance(bundle.critic, torch.nn.Module) and isinstance(bundle.teacher, torch.nn.Module)
             and type(bundle.generator_optimizer) is type(bundle.critic_optimizer) is torch.optim.AdamW,
             "actual all-role ModelBundle/AdamW owners required")
    _require(dist.is_initialized() and dist.get_world_size() == 16 and str(dist.get_backend()) == "nccl"
             and bundle.metadata.get("generator_ranks") == list(range(16)),
             "reference factory requires actual world16 NCCL owners")
    _require(all(group is not None and str(dist.get_backend(group)) == "nccl"
                 and dist.get_world_size(group) == 16 and dist.get_process_group_ranks(group) == list(range(16))
                 for group in (bundle.generator_group, bundle.score_group, bundle.critic_group)),
             "reference actual role groups must all cover world16 NCCL")
    from torch.distributed.tensor import DTensor
    _require(all(isinstance(p, DTensor) and p.device.type == "cuda" and p.device_mesh.device_type == "cuda"
                 for module in (bundle.student, bundle.teacher) for p in module.parameters())
             and all(p.device.type == "cuda" for p in bundle.critic.parameters()),
             "reference physical owners must retain actual CUDA placement")
    source = _source(source_root)
    original = _reference_lora(source)
    state = {}
    for name in ("generator", "critic"):
        role = _ReferenceRole(bundle, name, source, original, versions, restore_versions)
        state.update({name + ".weights": _ReferenceWeights(role), name + ".optimizer": _ReferenceOptimizer(role)})
    return state
