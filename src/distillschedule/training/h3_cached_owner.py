"""One actual original DMD2Trainer.update; persistent ownership, never a loop.

The original helpers own every loss/backward/clip/Adam step. This owner supplies
private native RNG continuity and a safe, staged common-checkpoint state view.
Frozen base bytes stay in the explicitly pinned full local safetensors assets.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
import math
import json
import importlib
import inspect
from pathlib import Path

from distillschedule.workloads.best_known.registry import fingerprint
from .checkpoint import cpu_tree, rng_state, restore_rng, validate_rng
from .h3_cached_assets import (require, checked_pin, validate_topology, content_digest,
                               tensor_content, runtime_identity, BACKEND)

SCHEMA = "distillschedule.h3_cached_update_owner.v1"


class H3CachedOwner:
    """The only mutable Program state owner for one complete original G1F1 cycle."""

    def __init__(self, model, config, assets, asset_pin):
        from distillschedule.reference import diffsynth_h3_cached as provider
        self.api = provider.load_cpu(backend=BACKEND)
        self.assets, self.asset_pin = deepcopy(assets), deepcopy(asset_pin)
        self.config = config
        self.adapter = self.api.dmd2.H3Adapter(model, config)
        self.discriminator = self.api.dmd2.VelocityDiscriminator(config)
        self.optimizers, self.parameters = self.api.training.make_optimizers(self.adapter, self.discriminator, config)
        self._step_binding = self._optimizer_step
        self.trainer = self.api.dmd2.DMD2Trainer(self.adapter, self.discriminator, config, self._step_binding)
        self._rng = cpu_tree(rng_state())
        self._config = deepcopy(asdict(config))
        self._schedule = cpu_tree(self.trainer.schedule)
        self._named = self._named_parameters()
        require(len({id(p) for p in self._named.values()}) == len(self._named), "parameter alias ownership is unsupported")
        self._parameter_ids = {n: id(p) for n, p in self._named.items()}
        self._module_ids = {n: id(m) for n, m in self._modules().items()}
        self._modes = {n: m.training for n, m in self._modules().items()}
        self._module_options = self._module_policy()
        self._frozen = {n: tensor_content(p) for n, p in self._named.items()
                        if n.startswith("adapter.") and ".adapters." not in n}
        self._groups = {role: self._optimizer_value(role)["groups"] for role in self.optimizers}
        self._defaults = {role: cpu_tree(opt.defaults) for role, opt in self.optimizers.items()}
        self._bindings = self._live_bindings()
        self._public_functions = self._public_bindings()
        from distillschedule.workloads.best_known.registry import file_sha256
        self._source_files = {str(Path(m.__file__).resolve()): file_sha256(Path(m.__file__).resolve())
                              for m in self._public_modules()}
        self._consumed, self._last_metrics = [], {}
        self._input_rows = None
        self._busy = self._poisoned = False
        self.identity = dict(schema=SCHEMA, assets=deepcopy(assets), asset_manifest=deepcopy(asset_pin),
            frozen=deepcopy(self._frozen), trainable_layout={n: dict(shape=list(p.shape), dtype=str(p.dtype))
                for n, p in self._named.items() if n not in self._frozen},
            optimizer_groups=deepcopy(self._groups), modes=deepcopy(self._modes),
            optimizer_defaults=deepcopy(self._defaults),
            source_files=deepcopy(self._source_files), module_options=deepcopy(self._module_options),
            scope="CPU1_ORIGINAL_CACHED_AV_REFERENCE_NOT_PRETRAINED_OR_QUALITY",
            common_checkpoint="distillschedule.training_checkpoint.v2")
        # Common manifest identity is JSON, unlike the safe checkpoint's tuple
        # preserving optimizer options. Keep the public metadata JSON-stable.
        self.identity = json.loads(json.dumps(self.identity, sort_keys=True, allow_nan=False))
        self._identity_digest = fingerprint(self.identity)
        self._validate_live()

    @property
    def cycle(self):
        return self.trainer.generator_step

    def _modules(self):
        return {**{"adapter." + n: m for n, m in self.adapter.named_modules()},
                **{"head." + n: m for n, m in self.discriminator.named_modules()}}

    def _named_parameters(self):
        return {**{"adapter." + n: p for n, p in self.adapter.named_parameters(remove_duplicate=False)},
                **{"head." + n: p for n, p in self.discriminator.named_parameters(remove_duplicate=False)}}

    def _module_policy(self):
        def scalar(value):
            return value is None or type(value) in (int, float, bool, str) or (
                isinstance(value, (list, tuple)) and all(scalar(v) for v in value))
        return {name: {k: deepcopy(v) for k, v in vars(module).items()
                       if not k.startswith("_") and k not in ("training", "role") and scalar(v)}
                for name, module in self._modules().items()}

    @staticmethod
    def _public_modules():
        return [importlib.import_module(name) for name in (
            "distillschedule.training.h3_cached_owner", "distillschedule.training.h3_cached_public",
            "distillschedule.training.h3_cached_assets", "distillschedule.data.h3_cached_inputs",
            "distillschedule.training.checkpoint", "distillschedule.training.tensor_state")]

    @classmethod
    def _public_bindings(cls):
        result = {}
        for module in cls._public_modules():
            for name, value in vars(module).items():
                if inspect.isfunction(value):
                    result[(module.__name__, name)] = (value, value.__code__)
                elif inspect.isclass(value) and value.__module__ == module.__name__:
                    result[(module.__name__, name)] = (value, None)
                    for member, item in vars(value).items():
                        if isinstance(item, (staticmethod, classmethod)):
                            item = item.__func__
                        if inspect.isfunction(item):
                            result[(module.__name__, name, member)] = (item, item.__code__)
        return result

    def _live_bindings(self):
        import torch
        functions = [self.api.training.optimizer_step, self.api.training.make_optimizers,
            torch.optim.AdamW.step, torch.optim.AdamW.load_state_dict, torch.autograd.grad,
            torch.nn.Module._call_impl, torch.nn.functional.linear, torch.nn.functional.scaled_dot_product_attention,
            torch.nn.utils.clip_grad_norm_, torch.Tensor.backward]
        return [(fn, getattr(fn, "__code__", None)) for fn in functions]

    def _optimizer_step(self, role, loss):
        self.api.training.optimizer_step(role, loss, self.optimizers, self.parameters, self.config)

    def _phase(self):
        return dict(roles={n: layer.role for n, layer in self.adapter.layers},
            adapter_requires_grad={n: p.requires_grad for n, p in self.adapter.named_parameters()},
            head_requires_grad={n: p.requires_grad for n, p in self.discriminator.named_parameters()},
            training={n: m.training for n, m in self._modules().items()})

    def _expected_phase(self, cycle):
        return dict(roles={n: "student" if cycle else "teacher" for n, _ in self.adapter.layers},
            adapter_requires_grad={n: bool(cycle and ".adapters.student." in n)
                                   for n, _ in self.adapter.named_parameters()},
            head_requires_grad={n: not bool(cycle) for n, _ in self.discriminator.named_parameters()},
            training=deepcopy(self._modes))

    def _optimizer_value(self, role):
        names = {id(p): n for n, p in self._named_parameters().items()}
        optimizer = self.optimizers[role]
        require(all(id(p) in names for g in optimizer.param_groups for p in g["params"])
                and all(id(p) in names for p in optimizer.state), "foreign optimizer parameter ownership")
        groups = [{**{k: cpu_tree(v) for k, v in group.items() if k != "params"},
                   "params": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups]
        return dict(groups=groups, defaults=cpu_tree(optimizer.defaults),
                    state={names[id(p)]: cpu_tree(slot) for p, slot in optimizer.state.items()})

    def _check_hooks(self):
        import torch
        import torch.nn.modules.module as module_hooks
        from torch.optim import optimizer as optimizer_hooks
        for registry in (module_hooks, optimizer_hooks):
            require(not any(value for key, value in vars(registry).items()
                            if key.startswith("_global_") and isinstance(value, dict)), "injected global numerical hooks")
        for module in self._modules().values():
            require(not any(value for key, value in vars(module).items() if "hook" in key and isinstance(value, dict)),
                    "injected module hooks")
            require(not any(callable(value) and callable(getattr(type(module), key, None))
                            for key, value in vars(module).items()), "instance numerical method override")
        require(not hasattr(self.adapter, "_distributed_predict"), "injected adapter predictor")
        for optimizer in self.optimizers.values():
            require(type(optimizer) is torch.optim.AdamW
                    and not any(value for key, value in vars(optimizer).items() if "hook" in key and isinstance(value, dict)),
                    "original AdamW ownership/no hooks required")
            require(not any(callable(v) and callable(getattr(type(optimizer), k, None))
                            for k, v in vars(optimizer).items()), "optimizer numerical method override")

    def _validate_live(self):
        import torch
        from distillschedule.reference import diffsynth_h3_cached as provider
        from distillschedule.workloads.best_known.registry import file_sha256
        require(not self._busy and not self._poisoned, "incomplete/poisoned native update cannot be reused")
        require(fingerprint(self.identity) == self._identity_digest, "owner source/input identity changed")
        validate_topology()
        require(self.api is provider.load_cpu(backend=BACKEND), "original provider binding changed")
        require(self._public_bindings() == self._public_functions
                and all(getattr(fn, "__code__", None) is code for fn, code in self._public_functions.values()
                        if code is not None)
                and all(file_sha256(Path(path)) == sha for path, sha in self._source_files.items()),
                "public source/callable binding changed")
        require(runtime_identity() == self.assets["runtime"], "dependency/runtime identity changed")
        checked_pin(self.asset_pin)
        for pin in [self.assets["base"]["config"], *self.assets["base"]["weights"]]:
            checked_pin(pin)
        require(asdict(self.config) == self._config and self.trainer.config is self.config
                and self.adapter.config is self.config and self.trainer.adapter is self.adapter
                and self.trainer.discriminator is self.discriminator and self.trainer.step_fn is self._step_binding,
                "original trainer/config/object ownership changed")
        require(type(self.trainer) is self.api.dmd2.DMD2Trainer and type(self.adapter) is self.api.dmd2.H3Adapter
                and type(self.discriminator) is self.api.dmd2.VelocityDiscriminator, "original model/trainer classes required")
        require(self._live_bindings() == self._bindings
                and all(getattr(fn, "__code__", None) is code for fn, code in self._bindings),
                "live numerical binding changed")
        self._check_hooks()
        named = self._named_parameters()
        require({n: id(p) for n, p in named.items()} == self._parameter_ids
                and {n: id(m) for n, m in self._modules().items()} == self._module_ids,
                "actual module/parameter ownership or aliases changed")
        require(content_digest(self._module_policy()) == content_digest(self._module_options),
                "native LoRA/model/head scalar options changed")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32 and bool(p.isfinite().all())
                    and p.grad is None for p in named.values()), "finite FP32 zero-gradient complete boundary required")
        require({n: tensor_content(named[n]) for n in self._frozen} == self._frozen, "frozen full base changed")
        t = self.cycle
        require(type(t) is int and 0 <= t <= self.assets["budget"]["max_cycles"]
                and type(self.trainer.critic_step) is int and self.trainer.critic_step == t
                and len(self._consumed) == t, "native G/C/consumption clocks changed")
        previous = "0" * 64
        for i, consumed in enumerate(self._consumed):
            desc = self._consumption(i)
            chain = fingerprint(dict(previous=previous, **desc))
            require(consumed == dict(**desc, chain=chain), "live consumed-source chain changed")
            previous = chain
        require(self._phase() == self._expected_phase(t), "initial teacher/completed student phase changed")
        require(content_digest(self.trainer.schedule) == content_digest(self._schedule), "original continuous schedule changed")
        for role, optimizer in self.optimizers.items():
            expected = (self.adapter.role_parameters("student") if role == "student" else
                        self.adapter.role_parameters("fake_score") + list(self.discriminator.parameters()))
            require(len(expected) == len(self.parameters[role])
                    and all(a is b for a, b in zip(expected, self.parameters[role])), "original optimizer parameter list changed")
            self._check_optimizer(self._optimizer_value(role), role, t)

    def bind_inputs(self, prepared_identity, rows):
        self._validate_live()
        require(self.cycle == 0 and self._input_rows is None and isinstance(prepared_identity, str)
                and len(prepared_identity) == 64 and rows, "fresh owner/prepared admission required")
        self._input_rows = deepcopy(rows)
        self.identity["prepared"] = dict(identity=prepared_identity, rows=deepcopy(rows))
        self._identity_digest = fingerprint(self.identity)

    def _consumption(self, index):
        require(self._input_rows is not None and 0 <= index < len(self._input_rows), "unadmitted/exhausted prepared data")
        return deepcopy(self._input_rows[index])

    def update(self, row, *, cycle):
        self._validate_live()
        require(type(cycle) is int and cycle == self.cycle and cycle < self.assets["budget"]["max_cycles"],
                "original update cursor/capacity changed")
        desc = {k: row[k] for k in ("identity", "absolute_step", "source_sha256", "content_sha256")}
        require(desc == self._consumption(cycle), "prepared input chain/order/content changed")
        require(content_digest(row["raw"]) == desc["content_sha256"], "raw cached content changed")
        inputs, negative = self.api.dmd2.prepare_batch(row["raw"], self.config.teacher_cfg_scale)
        before = content_digest((inputs, negative))
        outer = rng_state()
        self._busy = True
        try:
            restore_rng(self._rng)
            metrics = self.trainer.update((inputs["input_latents"], inputs["audio_input_latents"]), inputs, negative)
            require(content_digest((inputs, negative)) == before, "original clean/reference input mutation")
            require(all(type(v) in (bool, int, float) and math.isfinite(v) for v in metrics.values()), "nonfinite native metrics")
            desc["chain"] = fingerprint(dict(previous=self._consumed[-1]["chain"] if self._consumed else "0" * 64, **desc))
            self._consumed.append(desc)
            self._last_metrics = deepcopy(metrics)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)
            self._busy = False
        try:
            self._validate_live()
        except BaseException:
            self._poisoned = True
            raise
        return deepcopy(metrics)

    def state_dict(self):
        self._validate_live()
        return dict(schema=SCHEMA, identity=deepcopy(self.identity), generator_step=self.cycle,
            critic_step=self.trainer.critic_step, config=deepcopy(self._config), schedule=cpu_tree(self.trainer.schedule),
            adapters=cpu_tree(self.api.training.adapter_state(self.adapter)), heads=cpu_tree(self.discriminator.state_dict()),
            optimizers={role: self._optimizer_value(role) for role in self.optimizers}, phase=self._phase(),
            rng=cpu_tree(self._rng), consumed=deepcopy(self._consumed), last_metrics=deepcopy(self._last_metrics))

    def _check_tensor(self, value, wanted, name):
        import torch
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.layout == torch.strided
                and value.dtype == wanted.dtype and value.shape == wanted.shape and bool(value.isfinite().all())
                and not value.requires_grad and value.grad_fn is None, "checkpoint tensor shape/dtype/nonfinite: " + name)

    def _check_optimizer(self, value, role, cycle):
        import torch
        require(isinstance(value, dict) and set(value) == {"groups", "state", "defaults"}
                and content_digest(value["groups"]) == content_digest(self._groups[role])
                and content_digest(value["defaults"]) == content_digest(self._defaults[role]),
                "checkpoint original Adam groups/options/names changed")
        clocks = {name: (max(0, cycle - self.config.gan_start_step) if name.startswith("head.") else cycle)
                  for group in self._groups[role] for name in group["params"]}
        require(isinstance(value["state"], dict) and set(value["state"]) == {n for n, step in clocks.items() if step},
                "checkpoint Adam slot presence/curriculum changed")
        for name, slot in value["state"].items():
            require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "checkpoint Adam slot fields changed")
            self._check_tensor(slot["step"], torch.tensor(0.0), name + ".step")
            require(slot["step"].item() == clocks[name], "checkpoint Adam per-parameter clock changed")
            for field in ("exp_avg", "exp_avg_sq"):
                # Parameters can require grad; the restored slot cannot.
                self._check_tensor(slot[field], self._named[name], name + "." + field)
            require(bool((slot["exp_avg_sq"] >= 0).all()), "checkpoint Adam negative variance")

    def _stage(self, value):
        self._validate_live()
        fields = {"schema", "identity", "generator_step", "critic_step", "config", "schedule", "adapters", "heads",
                  "optimizers", "phase", "rng", "consumed", "last_metrics"}
        require(isinstance(value, dict) and set(value) == fields and value["schema"] == SCHEMA
                and value["identity"] == self.identity and value["config"] == self._config,
                "checkpoint source/data/config/asset identity changed")
        t = value["generator_step"]
        require(type(t) is int and 0 <= t <= self.assets["budget"]["max_cycles"]
                and type(value["critic_step"]) is int and value["critic_step"] == t,
                "checkpoint native G/C clocks changed")
        require(content_digest(value["phase"]) == content_digest(self._expected_phase(t))
                and content_digest(value["schedule"]) == content_digest(self._schedule),
                "checkpoint phase/modes/continuous sigma schedule changed")
        require(isinstance(value["rng"], dict) and value["rng"].keys() == self._rng.keys(), "checkpoint native RNG owners changed")
        validate_rng(value["rng"])
        require(isinstance(value["consumed"], list) and len(value["consumed"]) == t, "checkpoint consumed cursor changed")
        previous = "0" * 64
        for index, row in enumerate(value["consumed"]):
            desc = self._consumption(index)
            chain = fingerprint(dict(previous=previous, **desc))
            require(row == dict(**desc, chain=chain), "checkpoint consumed source/content chain changed")
            previous = chain
        expected_metrics = {"dmd_loss", "gan_loss", "critic_loss", "gan_active", "generator_step", "critic_step"}
        metrics = value["last_metrics"]
        require(isinstance(metrics, dict) and (metrics == {} if not t else set(metrics) == expected_metrics
                and type(metrics["generator_step"]) is int and metrics["generator_step"] == t
                and type(metrics["critic_step"]) is int and metrics["critic_step"] == t
                and type(metrics["gan_active"]) is bool and metrics["gan_active"] == (t > self.config.gan_start_step)
                and all(type(metrics[k]) is float and math.isfinite(metrics[k]) for k in ("dmd_loss", "gan_loss", "critic_loss"))),
                "checkpoint native branch/metrics/clocks changed")
        for key, current in (("adapters", self.api.training.adapter_state(self.adapter)),
                             ("heads", self.discriminator.state_dict())):
            require(isinstance(value[key], dict) and value[key].keys() == current.keys(), "checkpoint named " + key + " changed")
            for name, tensor in current.items():
                self._check_tensor(value[key][name], tensor, key + "." + name)
        require(isinstance(value["optimizers"], dict) and value["optimizers"].keys() == self.optimizers.keys(),
                "checkpoint original optimizer owners changed")
        staged = {}
        for role, optimizer in self.optimizers.items():
            spec = value["optimizers"][role]
            self._check_optimizer(spec, role, t)
            native = optimizer.state_dict()
            mapping = {name: pid for group, saved in zip(spec["groups"], native["param_groups"], strict=True)
                       for name, pid in zip(group["params"], saved["params"], strict=True)}
            restore = dict(state={mapping[n]: cpu_tree(slot) for n, slot in spec["state"].items()},
                param_groups=[{**{k: cpu_tree(v) for k, v in g.items() if k != "params"},
                               "params": [mapping[n] for n in g["params"]]} for g in spec["groups"]])
            candidate = copy(optimizer)
            candidate.defaults = cpu_tree(spec["defaults"])
            candidate.param_groups = [dict(g, params=list(g["params"])) for g in optimizer.param_groups]
            candidate.state = {}
            candidate.load_state_dict(restore)
            staged[role] = candidate
        return staged

    def validate_state_dict(self, value):
        self._stage(value)

    def load_state_dict(self, value):
        import torch
        staged = self._stage(value)  # ALL owner/RNG/phase/slot checks precede live mutation.
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                if ".adapters." in n:
                    p.copy_(value["adapters"][n])
            self.discriminator.load_state_dict(value["heads"], strict=True)
        for role, candidate in staged.items():
            self.optimizers[role].state = candidate.state
            self.optimizers[role].param_groups = candidate.param_groups
            self.optimizers[role].defaults = candidate.defaults
        self.trainer.generator_step, self.trainer.critic_step = value["generator_step"], value["critic_step"]
        for name, layer in self.adapter.layers:
            layer.role = value["phase"]["roles"][name]
        for n, p in self.adapter.named_parameters():
            p.requires_grad_(value["phase"]["adapter_requires_grad"][n])
        for n, p in self.discriminator.named_parameters():
            p.requires_grad_(value["phase"]["head_requires_grad"][n])
        for n, module in self._modules().items():
            module.training = value["phase"]["training"][n]
        self._rng = cpu_tree(value["rng"])
        self._consumed, self._last_metrics = deepcopy(value["consumed"]), deepcopy(value["last_metrics"])
        self._validate_live()

    def validate_checkpoint_boundary(self):
        self._validate_live()
        require(self.cycle > 0 and self._input_rows is not None, "checkpoint requires complete admitted native cycle")

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        t = value["generator_step"]
        require(isinstance(progress, dict) and type(progress.get("cycle")) is int and progress["cycle"] > 0
                and progress["cycle"] == t and type(progress.get("cursor")) is int and progress["cursor"] == t
                and progress.get("versions") == {"native_update": t}
                and type(progress["versions"]["native_update"]) is int,
                "inner/outer checkpoint progress disagrees")
