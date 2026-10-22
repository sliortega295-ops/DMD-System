"""Bounded original Helios native-step ownership, not a public W3 trainer.

One call performs the original trainer's selectors, optional G update, full C
rollout/update, EMA predicate and global-step increment. All loss, rollout,
Adam/clipping and EMA math remains in the existing original helpers/owners.
Only CPU1, resident, dense tensor-logit/final-head execution is admitted here.
Five calls make a checkpoint boundary; this is deliberately not Program wiring.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
import hashlib
import functools
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random

from .checkpoint import (cpu_tree, rng_state, restore_rng, validate_rng, _class,
                         save_checkpoint, read_checkpoint, unpack_state, _validate_layout)
from .helios_gan_state import CriticPhaseWeights
from .native_state import EngineClocks, engine_state, _named


_SCHEMA = "distillschedule.original_helios_step_owner.v1"


def _require(condition, message):
    if not condition:
        raise ValueError("original Helios native step: " + message)


def _plain(value):
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, dict):
        # Diffusers materializes this metadata list from a set; order is not
        # configuration semantics and differs with the process hash seed.
        return {str(k): _plain(sorted(v) if k == "_use_default_values" else v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("original Helios native step: identity needs explicit primitive values")


def _tensor_identity(value):
    import torch
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and not value.requires_grad and torch.isfinite(value).all().item(),
             "finite detached CPU input/frozen tensor required")
    tensor = value.detach().contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def _content(value):
    import torch
    if isinstance(value, torch.Tensor):
        return _tensor_identity(value)
    if isinstance(value, dict):
        return {key: _content(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_content(v) for v in value]
    return _plain(value)


def expected_clocks(global_step, *, ema_start_step=750, ratio=5):
    """Original source predicates, not invented bootstrap counters."""
    _require(type(global_step) is int and global_step >= 0 and type(ema_start_step) is int
             and ema_start_step >= 0 and ratio == 5, "invalid source curriculum clocks")
    generators = lambda n: (n + ratio - 1) // ratio
    before = min(global_step, ema_start_step)
    return {"generator": generators(global_step), "critic": global_step,
            "ema": before + generators(global_step) - generators(before)}


class _SparseEMAView:
    """Serialization only: the existing SparseShardEMA owns its recurrence."""

    def __init__(self, ema, generator):
        from runtime.sparse_ema import SparseShardEMA
        _require(type(ema) is SparseShardEMA, "existing SparseShardEMA required")
        self.ema, self.generator = ema, generator
        self.parameters = {n: id(p) for n, p in generator.module.named_parameters()}
        self.policy = self._policy()

    def _policy(self):
        e = self.ema
        return dict(names=list(e.names), indices=list(e.indices), parameter_count=e.parameter_count,
                    decay=e.decay, min_decay=e.min_decay, update_after_step=e.update_after_step,
                    base_model=e.base_model, format_version=e.format_version)

    def state_dict(self):
        e = self.ema
        parameters = list(self.generator.module.named_parameters())
        _require({n: id(p) for n, p in parameters} == self.parameters
                 and self._policy() == self.policy
                 and [parameters[i][0] for i in e.indices] == e.names
                 and {n for n, p in parameters if p.requires_grad} == set(e.names),
                 "EMA parameter/name/policy ownership changed")
        _require(len(e.shadows) == len(e.names) and all(s.device.type == "cpu"
                 and s.shape == parameters[i][1].shape and s.dtype == parameters[i][1].dtype
                 for s, i in zip(e.shadows, e.indices)), "EMA shadow layout changed")
        return dict(policy=deepcopy(self.policy), shadows=[s.detach() for s in e.shadows],
                    optimization_step=e.optimization_step, cur_decay_value=e.cur_decay_value)

    def validate_state_dict(self, value):
        current = self.state_dict()
        _require(value.keys() == current.keys() and value["policy"] == self.policy,
                 "EMA fields/ownership/policy changed")
        _validate_layout(current["shadows"], value["shadows"], "original sparse EMA")
        step = value["optimization_step"]
        _require(type(step) is int and step >= 0 and value["cur_decay_value"] ==
                 (self.ema.get_decay(step) if step else None), "EMA clock/decay changed")
        _require(all(s.isfinite().all().item() for s in value["shadows"]), "nonfinite EMA shadows")

    def load_state_dict(self, value):
        import torch
        self.validate_state_dict(value)
        with torch.no_grad():
            for shadow, restored in zip(self.ema.shadows, value["shadows"], strict=True):
                shadow.copy_(restored)
        self.ema.optimization_step = value["optimization_step"]
        self.ema.cur_decay_value = value["cur_decay_value"]


class _SchedulerView:
    """Exact original scheduler attributes; no alternate denoising recurrence."""

    _DYNAMIC = {"timesteps", "sigmas", "num_inference_steps"}

    def __init__(self, scheduler, native, args):
        training = args.training_config
        _require(type(scheduler) is native.scheduler, "exact original scheduler required")
        self.scheduler = scheduler
        self.config = _plain(dict(scheduler.config))
        _require(scheduler.config.shift == training.stage2_timestep_shift
                 and scheduler.config.stages == training.stage2_num_stages
                 and list(scheduler.config.stage_range) == list(training.stage2_stage_range)
                 and scheduler.config.gamma == training.stage2_scheduler_gamma
                 and scheduler.solver_p is None, "source scheduler configuration mismatch")
        self.initial = self._state()
        budget = args.validation_config.stage2_simulated_inference_steps[-1]
        self.allowed_horizons = {budget + 1}
        if training.is_amplify_first_chunk:
            self.allowed_horizons.add(2 * budget + 1)

    def _state(self):
        return cpu_tree({k: v for k, v in vars(self.scheduler).items() if k != "_internal_dict"})

    def state_dict(self):
        return dict(config=_plain(dict(self.scheduler.config)), state=self._state())

    def validate_state_dict(self, value):
        import torch
        _require(value.keys() == {"config", "state"} and value["config"] == self.config
                 and _plain(dict(self.scheduler.config)) == self.config,
                 "scheduler config changed")
        state = value["state"]
        _require(set(self.initial) <= set(state) <= set(self.initial) | {"num_inference_steps"},
                 "unsupported scheduler transient/cache state")
        # The original helpers use set_timesteps and their existing flow step,
        # never HeliosScheduler.step; its UniPC history must stay empty.
        fixed = {k: v for k, v in self.initial.items() if k not in self._DYNAMIC}
        _require(_content(fixed) == _content({k: state[k] for k in fixed}),
                 "scheduler fixed state/cache changed")
        for name in ("timesteps", "sigmas"):
            tensor = state[name]
            _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                     and tensor.dtype == torch.float64 and tensor.ndim == 1
                     and 1 <= tensor.numel() <= 1001 and tensor.isfinite().all().item(),
                     "invalid scheduler timestep/sigma state")
        if "num_inference_steps" in state:
            _require(type(state["num_inference_steps"]) is int
                     and state["num_inference_steps"] in self.allowed_horizons
                     and state["timesteps"].numel() == state["num_inference_steps"] - 1
                     and state["sigmas"].numel() == state["num_inference_steps"],
                     "invalid scheduler horizon")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        # Preserve the original FrozenDict config and live scheduler object.
        for key in set(vars(self.scheduler)) - {"_internal_dict"}:
            delattr(self.scheduler, key)
        self.scheduler.__dict__.update(deepcopy(value["state"]))


class HeliosStepOwner:
    """Already-built G/C owners plus exact original helper/curriculum state.

    Inputs are a finite, content-bound, separately identified native-step
    cohort. History tensors are cloned at the call boundary because original
    preparation/corruption mutates its batch views. No precomputed rollout is
    accepted. The owned RNG survives outer Program random_scope calls.

    This is not a constructor for the original trainer's datasets, assets,
    logging, epochs or validation/export. Those Args sections are identity
    only. Actual supplied FP32/rank/batch/GAS, Adam, LR scheduler and PEFT
    construction is checked explicitly; no claim is made to owning loaders.
    """

    def __init__(self, *, generator, critic, ema, args, generator_scheduler, critic_scheduler,
                 negative_prompt_embeds, noise_shape, inputs, cohort,
                 gan_base_critic_trainable_params, gan_extra_critic_trainable_params,
                 frozen_parameter_names, backend):
        import torch
        from omegaconf import OmegaConf
        from distillschedule.adapters.helios.reference_model import (
            load_reference_model_class, reference_model_identity)
        from distillschedule.strategies.wan_capture import native_primitives
        from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
        from runtime.gan_rolephase_official import HeliosAcceleratorAdapter
        self.generator, self.critic, self.ema = generator, critic, ema
        self.args = args
        self.config = OmegaConf.to_container(args, resolve=True)
        self._validate_config()
        self._validate_topology()
        self.native = native_primitives()
        self.provider_identity = reference_model_identity(backend=backend)
        self.backend = backend
        cls = load_reference_model_class(backend=backend)
        self._engines = {"generator": generator, "critic": critic}
        for role, engine in self._engines.items():
            _require(type(engine) is FSDP2EngineAdapter and type(engine.module) is cls
                     and engine.dp_world_size == 1 and engine.train_batch_size() == 1
                     and type(engine.optimizer) in (torch.optim.Adam, torch.optim.AdamW)
                     and engine.lr_scheduler is not None, "CPU1 full reference model/Adam engine required")
            _require(all(p.device.type == "cpu" and p.dtype == torch.float32
                         for p in engine.module.parameters()), "CPU1 FP32 resident owner required")
            _require(engine._gradient_clip == getattr(args.training_config,
                     "max_grad_norm" if role == "generator" else "max_grad_norm_critic"),
                     "engine-owned clipping differs from source config")
            self._validate_construction(role, engine)
        self.noise_shape = tuple(noise_shape)
        _require(len(self.noise_shape) == 5 and self.noise_shape[:2] == (1, 16)
                 and self.noise_shape[2] == args.training_config.num_critic_input_frames
                 and all(type(x) is int and x > 0 for x in self.noise_shape), "invalid native noise shape")
        self.inputs, self.cohort, self.negative = list(inputs), cohort, negative_prompt_embeds
        _require(isinstance(cohort, str) and cohort and self.inputs and len(self.inputs) % 5 == 0,
                 "explicit complete five-step input cohort required")
        self._validate_inputs()
        self.global_step = 0
        self._busy, self._poisoned = False, False
        self._rng = cpu_tree(rng_state())
        self._views = engine_state(self._engines)
        self._views["critic.weights"] = CriticPhaseWeights(critic,
            base=gan_base_critic_trainable_params, extra=gan_extra_critic_trainable_params,
            frozen=frozen_parameter_names, separate=args.training_config.is_separate_gan_grad)
        self.base = set(gan_base_critic_trainable_params)
        self.extra = set(gan_extra_critic_trainable_params)
        self._views["ema"] = _SparseEMAView(ema, generator)
        _require(ema.update_after_step == args.training_config.ema_start_step
                 and ema.decay == args.training_config.ema_decay, "EMA curriculum differs from config")
        self._views["generator.noise_scheduler"] = _SchedulerView(generator_scheduler, self.native, args)
        self._views["critic.noise_scheduler"] = _SchedulerView(critic_scheduler, self.native, args)
        self._accelerators = {r: HeliosAcceleratorAdapter(torch.device("cpu"), e)
                              for r, e in self._engines.items()}
        for accelerator in self._accelerators.values():
            accelerator.is_main_process = True
        self._helpers = {name: getattr(self.native.post, name) for name in ("_generator_loss", "_critic_loss",
                         "run_generator", "sample_dynamic_dmd_num_latent_sections")}
        helper_path = Path(self.native.post.__file__).resolve()
        _require(all(Path(fn.__code__.co_filename).resolve() == helper_path for fn in self._helpers.values()),
                 "original helper binding changed")
        self._ownership = self._live_ownership()
        self._noise_owners = {r: id(self._views[r + ".noise_scheduler"].scheduler) for r in self._engines}
        self._adapters = self._adapter_state()
        self._generator_flags = {n: p.requires_grad for n, p in generator.module.named_parameters()}
        self._scheduler_policy = {r: self._schedule_policy(e.lr_scheduler.state_dict())
                                  for r, e in self._engines.items()}
        self._optimizer_policy = {r: self._optim_policy(e.optimizer.state_dict())
                                  for r, e in self._engines.items()}
        self.identity = self._identity()
        self._identity_digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self._validate_live()
        _require(self.versions == dict(generator=0, critic=0, ema=0), "source owners must start at genuine zero")

    def _validate_config(self):
        c, v = self.args.training_config, self.args.validation_config
        _require(c.mixed_precision == "no" and c.train_batch_size == 1
                 and c.gradient_accumulation_steps == 1,
                 "CPU1 owner requires explicit FP32 precision, batch=1 and accumulation=1")
        _require(c.optimizer in ("adam", "adamw") and not c.use_8bit_adam
                 and not c.scale_lr and not c.enable_xformers_memory_efficient_attention
                 and not c.enable_npu_flash_attention,
                 "unsupported CPU1 optimizer/resource construction config")
        _require(c.is_train_dmd is True and c.is_use_gan is True and c.use_ema is True
                 and c.dfake_gen_update_ratio == 5 and c.is_enable_stage2 is True
                 and c.stage2_num_stages == 3 and c.is_gan_low_vram_mode is True
                 and c.is_use_gan_final is True and c.is_use_gan_hooks is False
                 and c.has_multi_term_memory_patch is True, "unsupported original owner curriculum/model branch")
        disabled = ("dmd_is_low_vram_mode", "dmd_is_offload_grad", "is_use_reward_model",
            "is_use_ode_regression", "is_only_ode_regression", "use_error_recycling",
            "is_smoothness_loss", "is_dmd_vae_decode", "is_multi_pyramid_stage_backward_simulated",
            "is_consistency_align", "is_decouple_dmd", "is_mean_var_regular", "is_x0_mean_var_regular",
            "is_chunk_mean_var_regular", "is_chunk_x0_mean_var_regular", "corrupt_model_input",
            "is_navit_pyramid", "dmd_teacher_forcing", "is_use_gt_coherence_dmd", "efficient_sample",
            "is_amplify_history", "zero_history_timestep", "restrict_self_attn", "guidance_cross_attn",
            "restrict_lora", "is_train_restrict_lora")
        _require(not any(getattr(c, k) for k in disabled) and v.use_kv_cache is False,
                 "unsupported optional reward/ODE/recycle/corruption/layout/cache branch")
        _require(c.dmd_num_latent_sections_min == c.dmd_num_latent_sections_max == 1
                 and not c.is_enable_cold_start, "only native one-section curriculum admitted")
        _require(all(type(x) is int and x >= 0 for x in (c.ema_start_step, c.gan_start_step))
                 and type(c.is_separate_gan_grad) is bool and 0 <= c.use_gt_history_ratio <= 1,
                 "invalid GAN/EMA/GT-history curriculum")
        _require(len(v.stage2_simulated_inference_steps) == 3 and all(type(x) is int and x > 0
                 for x in v.stage2_simulated_inference_steps), "invalid original stage horizons")

    @staticmethod
    def _validate_topology():
        import torch
        dist = torch.distributed
        _require(int(os.environ.get("WORLD_SIZE", "1")) == 1
                 and (not dist.is_initialized() or dist.get_world_size() == 1),
                 "CPU1 owner refuses an unadmitted actual process group/topology")

    def _validate_construction(self, role, engine):
        import torch
        from diffusers.optimization import get_scheduler
        c = self.args.training_config
        expected_type = torch.optim.AdamW if c.optimizer == "adamw" else torch.optim.Adam
        rate = c.learning_rate if role == "generator" else c.critic_learning_rate
        _require(type(engine.optimizer) is expected_type and len(engine.optimizer.param_groups) == 1
                 and not engine.optimizer.state, "optimizer construction must match original role config at zero")
        group = engine.optimizer.param_groups[0]
        _require(group.get("initial_lr") == rate and tuple(group["betas"]) == (c.adam_beta1, c.adam_beta2)
                 and group["eps"] == c.adam_epsilon and group["weight_decay"] == c.adam_weight_decay
                 and not any(group.get(k, False) for k in ("amsgrad", "maximize", "capturable", "differentiable"))
                 and group.get("fused") is None and group.get("foreach") is None,
                 "optimizer construction differs from original role config")
        # Reconstruct only the original scheduler over copied group metadata.
        # Its factory/initial scheduler step never touches live groups, weights,
        # moments, clocks or RNG and creates no replacement training owner.
        staged_optimizer = copy(engine.optimizer)
        staged_optimizer.param_groups = [dict(group, params=list(group["params"]))]
        staged_optimizer.state = {}
        expected = get_scheduler(c.lr_scheduler, optimizer=staged_optimizer,
            num_warmup_steps=c.lr_warmup_steps, num_training_steps=c.max_train_steps,
            num_cycles=c.lr_num_cycles, power=c.lr_power)
        _require(type(engine.lr_scheduler) is type(expected)
                 and engine.lr_scheduler.state_dict() == expected.state_dict()
                 and [self._callable_identity(fn) for fn in engine.lr_scheduler.lr_lambdas] ==
                     [self._callable_identity(fn) for fn in expected.lr_lambdas]
                 and group["lr"] == staged_optimizer.param_groups[0]["lr"],
                 "native scheduler construction differs from original role config")
        configs = engine.module.peft_config
        model_config = self.args.model_config
        prefix = "" if role == "generator" else "critic_"
        _require(set(configs) == {"default"} and list(engine.module.active_adapters()) == ["default"],
                 "original single active PEFT adapter required")
        peft = configs["default"]
        _require(peft.r == getattr(model_config, prefix + "lora_rank")
                 and peft.lora_alpha == getattr(model_config, prefix + "lora_alpha")
                 and peft.lora_dropout == getattr(model_config, prefix + "lora_dropout"),
                 "actual PEFT rank/alpha/dropout differs from original role config")

    def _validate_inputs(self):
        import torch
        ids = []
        dtype = self.generator.module.dtype
        for row in self.inputs:
            _require(isinstance(row, dict) and row.keys() == {"identity", "prompt_embeds", "gan_prompt_embeds",
                     "gan_vae_latents", "generator_history", "critic_history"}, "native input fields changed")
            _require(isinstance(row["identity"], str) and row["identity"], "explicit source input identity required")
            ids.append(row["identity"])
            for name in ("prompt_embeds", "gan_prompt_embeds"):
                x = row[name]
                _require(isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1
                         and x.shape[-1] == self.generator.module.config.text_dim and x.dtype == dtype,
                         "native prompt shape/dtype changed")
            _require(tuple(row["gan_vae_latents"].shape) == self.noise_shape
                     and row["gan_vae_latents"].dtype == dtype, "native GAN latent shape/dtype changed")
            for role in ("generator", "critic"):
                history = row[role + "_history"]
                _require(history.keys() == {"gt_history_latents", "gt_target_latents", "gt_x0_latents"},
                         "native GT history fields changed")
                for key, frames in (("gt_history_latents", sum(self.args.training_config.history_sizes)),
                                    ("gt_target_latents", self.noise_shape[2]), ("gt_x0_latents", 1)):
                    expected = (*self.noise_shape[:2], frames, *self.noise_shape[-2:])
                    _require(tuple(history[key].shape) == expected and history[key].dtype == dtype,
                             "native GT history shape/dtype changed")
            _content(row)
        _require(len(set(ids)) == len(ids), "each native step requires a distinct source input identity")
        _require(tuple(self.negative.shape) == tuple(self.inputs[0]["prompt_embeds"].shape)
                 and self.negative.dtype == dtype, "negative prompt shape/dtype changed")

    def _live_ownership(self):
        return {r: {"parameters": {n: id(p) for n, p in e.module.named_parameters()},
                    "groups": [[id(p) for p in g["params"]] for g in e.optimizer.param_groups],
                    "optimizer": id(e.optimizer), "scheduler": id(e.lr_scheduler),
                    "module": id(e.module)} for r, e in self._engines.items()}

    @staticmethod
    def _schedule_policy(value):
        return {k: v for k, v in value.items() if k not in ("last_epoch", "_step_count", "_last_lr")}

    @staticmethod
    def _optim_policy(value):
        return [{k: v for k, v in g.items() if k not in ("params", "lr")}
                for g in value["param_groups"]]

    @staticmethod
    def _callable_identity(function):
        """Bind LambdaLR's real native callable, not only its empty state dict."""
        from distillschedule.workloads.best_known.registry import file_sha256
        if isinstance(function, functools.partial):
            return dict(function=HeliosStepOwner._callable_identity(function.func),
                        args=_plain(function.args), kwargs=_plain(function.keywords))
        _require(inspect.isfunction(function), "unsupported opaque native scheduler callable")
        path = Path(inspect.getfile(function)).resolve()
        _require(path.is_file(), "native scheduler callable source is unavailable")
        return dict(module=function.__module__, name=function.__qualname__, path=str(path),
                    sha256=file_sha256(path), line=function.__code__.co_firstlineno,
                    defaults=_plain(function.__defaults__), kwdefaults=_plain(function.__kwdefaults__),
                    closure=_plain([cell.cell_contents for cell in function.__closure__ or ()]))

    def _adapter_state(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        result = {}
        for role, engine in self._engines.items():
            model = engine.module
            config = {n: _plain(c.to_dict()) for n, c in model.peft_config.items()}
            layers = {n: {"active": list(m.active_adapters), "disabled": m.disable_adapters}
                      for n, m in model.named_modules() if isinstance(m, BaseTunerLayer)}
            _require(config and layers and all(not r["disabled"] for r in layers.values()),
                     "enabled actual PEFT adapters required at native boundary")
            result[role] = {"config": config, "layers": layers, "active": list(model.active_adapters())}
        return result

    def _cache_boundary(self):
        for role, engine in self._engines.items():
            model = engine.module
            c = self.args.training_config
            _require(model.gradient_checkpointing == c.gradient_checkpointing
                     and not model.config.is_use_gan_hooks
                     and all(getattr(model.config, key) == getattr(c, key) for key in
                         ("has_multi_term_memory_patch", "zero_history_timestep", "restrict_self_attn",
                          "guidance_cross_attn", "is_train_restrict_lora", "restrict_lora", "is_amplify_history")),
                     "model execution/config branch changed")
            # The original trainer constructs GAN heads on Score only. Do not
            # require an unused discriminator on the actual Generator.
            if role == "critic":
                _require(model.config.is_use_gan and model.config.is_use_gan_final
                         and model.config.gan_cond_map_dim == c.gan_cond_map_dim,
                         "original critic final GAN head config changed")
            for block in model.blocks:
                processor = block.attn1.processor
                _require(processor.kv_cache is None and not processor.cache_enabled,
                         "native kv cache/reuse is not admitted")

    def _identity(self):
        import torch
        from distillschedule.adapters.helios.reference_model import reference_model_identity
        models = {}
        for role, engine in self._engines.items():
            model, selected, _ = _named(engine)
            models[role] = dict(class_name=_class(model), config=_plain(dict(model.config)),
                trainables=list(selected), frozen={n: _tensor_identity(p.detach()) for n, p in
                    model.named_parameters() if n not in selected},
                layouts={n: {"shape": list(p.shape), "dtype": str(p.dtype)} for n, p in model.named_parameters()})
        from distillschedule.workloads.best_known.registry import file_sha256
        from runtime.gan_rolephase_official import common_official_call
        from .tensor_state import decode
        owners = {"step_owner": Path(__file__), "ema": Path(inspect.getfile(type(self.ema))),
                  "engine": Path(inspect.getfile(type(self.generator))),
                  "phase_weights": Path(inspect.getfile(CriticPhaseWeights)),
                  "native_state": Path(inspect.getfile(EngineClocks)),
                  "checkpoint": Path(inspect.getfile(save_checkpoint)),
                  "tensor_state": Path(inspect.getfile(decode)),
                  "model_provider": Path(inspect.getfile(reference_model_identity)),
                  "call_binding": Path(inspect.getfile(common_official_call))}
        dependencies = {}
        for name in ("torch", "diffusers", "peft", "numpy", "omegaconf", "transformers"):
            module = importlib.import_module(name)
            dependencies[name] = dict(distribution=importlib.metadata.version(name),
                imported_version=str(module.__version__), path=str(Path(module.__file__).resolve()))
        return dict(schema=_SCHEMA, scope="CPU1_FULL_SOURCE_NATIVE_STEPS_NOT_PUBLIC_W3",
            source=reference_model_identity(backend=self.backend), config=self.config,
            native_owner_sources={name: {"path": str(path), "sha256": file_sha256(path)}
                                  for name, path in owners.items()},
            models=models, adapters=self._adapters, noise_shape=list(self.noise_shape),
            negative=_tensor_identity(self.negative), cohort=self.cohort, inputs=_content(self.inputs),
            engines={r: {"class": _class(e), "policy": EngineClocks(e).state_dict()["policy"],
                         "optimizer": _class(e.optimizer), "scheduler": _class(e.lr_scheduler),
                         "optimizer_policy": _plain(self._optimizer_policy[r]),
                         "scheduler_policy": _plain(self._scheduler_policy[r]),
                         "scheduler_callables": [self._callable_identity(fn)
                             for fn in getattr(e.lr_scheduler, "lr_lambdas", [])]}
                     for r, e in self._engines.items()},
            runtime=dict(dependencies=dependencies, torch_num_threads=torch.get_num_threads(),
                         matmul_precision=torch.get_float32_matmul_precision()))

    @property
    def versions(self):
        return dict(generator=self.generator.global_steps, critic=self.critic.global_steps,
                    ema=self.ema.optimization_step)

    def _validate_adam(self, role, value, steps, global_step):
        import torch
        optimizer = value["optimizer"]
        names = {name: index for named, group in zip(value["parameters"], optimizer["param_groups"], strict=True)
                 for name, index in zip(named, group["params"], strict=True)}
        expected = {}
        for name, index in names.items():
            count = (max(0, global_step - self.args.training_config.gan_start_step)
                     if role == "critic" and name in self.extra else steps)
            if count:
                expected[index] = count
        _require(set(optimizer["state"]) == set(expected),
                 "native Adam missing/extra parameter state for completed role updates")
        for index, slots in optimizer["state"].items():
            _require(set(slots) == {"step", "exp_avg", "exp_avg_sq"}
                     and all(isinstance(v, torch.Tensor) and v.isfinite().all().item() for v in slots.values())
                     and slots["step"].dtype == torch.float32 and slots["step"].ndim == 0
                     and slots["step"].item() == expected[index],
                     "native Adam moments/step disagree with source role curriculum")

    def _validate_live(self):
        from omegaconf import OmegaConf
        _require(not self._busy and not self._poisoned, "outstanding native step/backward or failed owner")
        self._validate_topology()
        _require(OmegaConf.to_container(self.args, resolve=True) == self.config,
                 "source config/curriculum changed")
        _require(self._live_ownership() == self._ownership, "live Parameter/optimizer ownership changed")
        _require(self._engines == dict(generator=self.generator, critic=self.critic)
                 and self._views["ema"].ema is self.ema and self._views["ema"].generator is self.generator
                 and {r: id(self._views[r + ".noise_scheduler"].scheduler) for r in self._engines}
                 == self._noise_owners, "native engine/EMA/noise-scheduler owner changed")
        _require({n: p.requires_grad for n, p in self.generator.module.named_parameters()} == self._generator_flags
                 and self.base == set(self._views["critic.weights"].base)
                 and self.extra == set(self._views["critic.weights"].extra), "role trainability/name ownership changed")
        _require(self._adapter_state() == self._adapters, "PEFT adapter state changed")
        _require(self._identity() == self.identity, "source/frozen/input/cohort/runtime identity changed")
        _require(hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest() ==
                 self._identity_digest, "bound checkpoint identity changed")
        _require(all(getattr(self.native.post, k) is v for k, v in self._helpers.items()),
                 "source helper binding changed")
        self._cache_boundary()
        expected = expected_clocks(self.global_step, ema_start_step=self.args.training_config.ema_start_step)
        _require(self.versions == expected, "native progress/update versions disagree with source curriculum")
        for role, engine in self._engines.items():
            clocks = EngineClocks(engine)
            clocks.validate_checkpoint_boundary()
            row = clocks.state_dict()
            _require(all(v == expected[role] for v in row["counters"].values())
                     and engine.lr_scheduler.last_epoch == expected[role], "native engine/scheduler clocks changed")
            _require(self._schedule_policy(engine.lr_scheduler.state_dict()) == self._scheduler_policy[role]
                     and self._optim_policy(engine.optimizer.state_dict()) == self._optimizer_policy[role],
                     "native scheduler/optimizer policy changed")
            self._validate_adam(role, self._views[role + ".optimizer"].state_dict(), expected[role], self.global_step)
            _require(all(p.detach().isfinite().all().item() for p in _named(engine)[1].values()),
                     "nonfinite native trainables")
            for slots in engine.optimizer.state.values():
                _require(all(not hasattr(v, "isfinite") or v.isfinite().all().item() for v in slots.values()),
                         "nonfinite native Adam state")
        phase = self._views["critic.weights"].state_dict()["phase"]
        complete = self.args.training_config.is_separate_gan_grad and self.global_step > self.args.training_config.gan_start_step
        _require(phase == ("separate_gan_complete" if complete else "ready"), "critic phase/curriculum mismatch")
        for view in (self._views["ema"], self._views["generator.noise_scheduler"],
                     self._views["critic.noise_scheduler"]):
            view.validate_state_dict(view.state_dict())

    @contextmanager
    def _native_random_scope(self):
        outer = rng_state()
        try:
            restore_rng(self._rng)
            yield
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)

    @contextmanager
    def _observe(self, engine, events):
        backward, step = engine.backward, engine.step
        def observed_backward(loss):
            events.append(["backward", engine._gradient_accumulation_boundary])
            return backward(loss)
        def observed_step():
            events.append(["commit", engine._gradient_accumulation_boundary])
            return step()
        engine.backward, engine.step = observed_backward, observed_step
        try:
            yield
        finally:
            engine.backward, engine.step = backward, step

    def _kwargs(self, *, role, row, use_gt, use_gan, sections):
        from runtime.gan_rolephase_official import common_official_call
        c, t = self.args.training_config, self.global_step
        kwargs = common_official_call(self.args)
        kwargs.update(args=self.args, transformer=self.generator.module,
            scheduler=self._views[role + ".noise_scheduler"].scheduler,
            prompt_embeds=row["prompt_embeds"], is_use_gt_history=use_gt,
            num_rollout_sections=sections, is_use_gan=use_gan, is_gan_low_vram_mode=c.is_gan_low_vram_mode,
            gan_prompt_embeds=row["gan_prompt_embeds"], global_step=t,
            dynamic_alpha=getattr(c, role + "_dynamic_alpha"),
            dynamic_beta=getattr(c, role + "_dynamic_beta"),
            dynamic_sample_type=getattr(c, role + "_dynamic_sample_type"),
            dynamic_step=getattr(c, role + "_dynamic_step"),
            **cpu_tree(row[role + "_history"]))
        if role == "generator":
            kwargs.update(accelerator=self._accelerators[role], real_fake_score_model=self.critic.module,
                negative_prompt_embeds=self.negative, fake_guidance_scale=c.fake_guidance_scale,
                real_guidance_scale=c.real_guidance_scale, gan_g_weight=c.gan_g_weight,
                is_forcing_low_renoise=c.generator_is_forcing_low_renoise)
        else:
            kwargs.update(critic_accelerator=self._accelerators[role], fake_score_model=self.critic.module,
                is_separate_gan_grad=c.is_separate_gan_grad,
                gan_base_critic_trainable_params=self.base, gan_extra_critic_trainable_params=self.extra,
                gan_vae_latents=row["gan_vae_latents"], gan_d_weight=c.gan_d_weight,
                aprox_r1=c.aprox_r1, aprox_r2=c.aprox_r2, r1_weight=c.r1_weight, r2_weight=c.r2_weight,
                r1_sigma=c.r1_sigma, r2_sigma=c.r2_sigma)
        return kwargs

    def step(self):
        """One nonreplayable original training step; no new loss/rollout loop."""
        import torch
        self._validate_live()
        _require(self.global_step < len(self.inputs), "input cohort exhausted")
        t, c = self.global_step, self.args.training_config
        row = self.inputs[t]
        train_generator = t % c.dfake_gen_update_ratio == 0
        use_gan = c.is_use_gan and t >= c.gan_start_step
        receipt = dict(global_step=t, input_identity=row["identity"], train_generator=train_generator,
                       use_gan=use_gan, generator_events=[], critic_events=[])
        self._busy = True
        try:
            with self._native_random_scope():
                use_gt = c.is_use_gt_history and random.random() < c.use_gt_history_ratio
                sections = self.native.post.sample_dynamic_dmd_num_latent_sections(
                    min_sections=c.dmd_num_latent_sections_min, max_sections=c.dmd_num_latent_sections_max,
                    dmd_dynamic_alpha=c.dmd_dynamic_alpha, dmd_dynamic_beta=c.dmd_dynamic_beta,
                    dmd_dynamic_sample_type=c.dmd_dynamic_sample_type, global_step=t,
                    dmd_dynamic_step=c.dmd_dynamic_step, device="cpu")
                receipt.update(use_gt_history=use_gt, num_rollout_sections=sections)
                if train_generator:
                    if use_gan:
                        for name, p in self.critic.module.named_parameters():
                            if name in self.base | self.extra:
                                p.requires_grad = False
                    kwargs = self._kwargs(role="generator", row=row, use_gt=use_gt, use_gan=use_gan, sections=sections)
                    # Preserve the trainer's entry draw even though run_generator
                    # performs its own original resampling inside the helper.
                    kwargs["noise"] = torch.randn(self.noise_shape, dtype=self.generator.module.dtype)
                    inspect.signature(self._helpers["_generator_loss"]).bind(**kwargs)
                    with self._observe(self.generator, receipt["generator_events"]):
                        loss, logs = self._helpers["_generator_loss"](**kwargs)
                        _require(torch.isfinite(loss).all().item(), "nonfinite generator loss")
                        self.generator.backward(loss)
                        _require(all(p.grad is None for p in self.critic.module.parameters()),
                                 "original Score derivative leaked critic gradients")
                        self.generator.step()
                    receipt.update(generator_loss=loss.detach().item(), generator_logs=_content(logs))
                if use_gan:
                    for name, p in self.critic.module.named_parameters():
                        if name in self.base | self.extra:
                            p.requires_grad = True
                kwargs = self._kwargs(role="critic", row=row, use_gt=use_gt, use_gan=use_gan, sections=sections)
                kwargs["noise"] = torch.randn(self.noise_shape, dtype=self.critic.module.dtype)
                inspect.signature(self._helpers["_critic_loss"]).bind(**kwargs)
                with self._observe(self.critic, receipt["critic_events"]):
                    loss, logs = self._helpers["_critic_loss"](**kwargs)
                    _require(torch.isfinite(loss).all().item(), "nonfinite critic loss")
                    if not use_gan:
                        self.critic.backward(loss)
                        self.critic.step()
                receipt.update(critic_loss=loss.detach().item(), critic_logs=_content(logs))
                _require(receipt["generator_events"] == ([["backward", True], ["commit", True]]
                         if train_generator else []), "generator update boundary drift")
                _require(receipt["critic_events"] == ([["backward", False], ["backward", True], ["commit", True]]
                         if use_gan else [["backward", True], ["commit", True]]),
                         "original critic backward/commit boundary drift")
                if t < c.ema_start_step or train_generator:
                    self.ema.step(self.generator.module.parameters())
                self.global_step += 1
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._busy = False
        self._validate_live()
        receipt["versions"] = self.versions
        return receipt

    def state_dict(self):
        self._validate_live()
        return dict(schema=_SCHEMA, identity=deepcopy(self.identity), global_step=self.global_step,
                    versions=self.versions, rng=cpu_tree(self._rng), adapters=self._adapter_state(),
                    native={k: cpu_tree(v.state_dict()) for k, v in self._views.items()})

    def validate_checkpoint_boundary(self):
        self._validate_live()
        _require(self.global_step > 0 and self.global_step % 5 == 0,
                 "checkpoint requires a complete five-native-step cycle")

    def validate_state_dict(self, value):
        self._validate_live()
        _require(isinstance(value, dict) and value.keys() == {"schema", "identity", "global_step", "versions",
                 "rng", "adapters", "native"}, "native checkpoint fields changed")
        _require(value["schema"] == _SCHEMA and value["identity"] == self.identity
                 and value["adapters"] == self._adapters, "source/config/frozen/cohort/adapter identity changed")
        t = value["global_step"]
        _require(type(t) is int and 0 < t <= len(self.inputs) and t % 5 == 0,
                 "native checkpoint cursor is not a completed cycle")
        expected = expected_clocks(t, ema_start_step=self.args.training_config.ema_start_step)
        _require(value["versions"] == expected and value["native"].keys() == self._views.keys(),
                 "checkpoint progress/versions/state owners changed")
        _require(value["rng"].keys() == self._rng.keys(), "native RNG stream owners changed")
        validate_rng(value["rng"])
        for key, view in self._views.items():
            validate = getattr(view, "validate_state_dict", None)
            if callable(validate):
                validate(value["native"][key])
            else:
                _validate_layout(view.state_dict(), value["native"][key], key)
        for role in self._engines:
            counters = value["native"][role + ".clocks"]["counters"]
            _require(all(v == expected[role] for v in counters.values())
                     and value["native"][role + ".scheduler"]["last_epoch"] == expected[role],
                     "checkpoint native engine/scheduler progress disagrees")
            schedule = value["native"][role + ".scheduler"]
            optimizer = value["native"][role + ".optimizer"]["optimizer"]
            _require(self._schedule_policy(schedule) == self._scheduler_policy[role]
                     and self._optim_policy(optimizer) == self._optimizer_policy[role],
                     "checkpoint scheduler/optimizer policy changed")
            _require(schedule.get("_step_count") == expected[role] + 1,
                     "checkpoint scheduler step count disagrees")
            for group, rate in zip(optimizer["param_groups"], schedule["_last_lr"], strict=True):
                _require(group["lr"] == rate, "checkpoint Adam/scheduler learning rate disagrees")
            self._validate_adam(role, value["native"][role + ".optimizer"], expected[role], t)
            weights = value["native"][role + ".weights"]
            _require(all(x.isfinite().all().item() for x in weights["tensors"].values()),
                     "checkpoint nonfinite trainables")
            _require("num_inference_steps" in value["native"][role + ".noise_scheduler"]["state"],
                     "checkpoint lacks a completed original rollout scheduler")
        _require(value["native"]["ema"]["optimization_step"] == expected["ema"],
                 "checkpoint EMA progress disagrees")
        complete = self.args.training_config.is_separate_gan_grad and t > self.args.training_config.gan_start_step
        _require(value["native"]["critic.weights"]["phase"] ==
                 ("separate_gan_complete" if complete else "ready"), "checkpoint critic phase/curriculum mismatch")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        for key, view in self._views.items():
            view.load_state_dict(value["native"][key])
        self.global_step = value["global_step"]
        self._rng = cpu_tree(value["rng"])
        self._validate_live()

    def save(self, path):
        """The common atomic checkpoint writer; publication only every five calls."""
        return save_checkpoint(path, {"native_step": self}, identity=self.identity,
            cycle=self.global_step // 5, cursor=self.global_step, versions=self.versions)

    def restore(self, path):
        """Stage all native owners AND outer progress/RNG before any mutation."""
        payload = read_checkpoint(path, identity=self.identity)
        _require(payload["state"].keys() == {"native_step"}, "checkpoint native state owner changed")
        from .tensor_state import decode
        value = decode(payload["state"]["native_step"]["value"], {})
        self.validate_state_dict(value)
        _require(payload["cycle"] == value["global_step"] // 5 and payload["cursor"] == value["global_step"]
                 and payload["versions"] == value["versions"] and payload["policy"] is None,
                 "checkpoint envelope progress/curriculum disagrees with native owner")
        validate_rng(payload["rng"])
        unpack_state(payload["state"], {"native_step": self})
        restore_rng(payload["rng"])
        return payload
