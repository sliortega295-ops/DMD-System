"""Measured official Self-Forcing mechanisms with continuous-cycle boundaries.

This module consumes a real, source-bound FSDP2 owner. It does not construct a
trainer, replace the upstream loss/rollout, start a process group, or admit a
new GPU experiment. Imported kernel bodies are unchanged. Adaptations here
only bind private imports, finite metadata, host indices and cache lifetime.
The public owner's same-topology checkpoint remains responsible for all live
parameters, optimizer/EMA state, training RNG and data cursors.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from collections.abc import Mapping
import textwrap
import threading
from types import MethodType
import weakref

from ..official_profiles import resolve
from ..self_forcing_source import REVISION, SOURCE_FILES, require
from .bound_source import SourceBoundHelpers

PROFILE = "self-forcing-wan-fsdp2-r6-v1"
_ACTIVE_SOURCE_PATCH = threading.Lock()


def admission_requirements():
    """Metadata-only capability; this is not GPU or numerical acceptance."""
    return dict(profile=PROFILE, runtime="SOURCE_BOUND_NATIVE_HELPERS_WITH_CONTINUOUS_BOUNDARY_API",
        native_checkpoint_and_backward="REQUIRED_UNCHANGED", integration_gpu_validation="NOT_RUN",
        numerical_status="REVIEW", transient_state="REBUILD_AFTER_RESTORE; DRAIN_EACH_COMPLETE_CYCLE",
        contract=resolve(PROFILE)["contract"], native_owner="REQUIRED_NOT_CONSTRUCTED_BY_HELPERS",
        concurrency="ONE_OWNER_PER_SOURCE_NAMESPACE; SCOPED_GLOBAL_PROVIDER_PATCHES")


def _host_initializer():
    """Port only the two integer device literals in the public geometry port."""
    from ..self_forcing_source import GeometryCache
    function = GeometryCache._initialize_kv_cache
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    changed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg not in ("global_end_index", "local_end_index"):
            continue
        call = node.value
        require(isinstance(call, ast.Call) and ast.unparse(call.func) == "torch.zeros"
                and len(call.args) == 1 and ast.literal_eval(call.args[0]) == 1,
                "host geometry integer initializer ABI changed")
        keys = {key.arg: key for key in call.keywords}
        require(set(keys) == {"dtype", "device"} and ast.unparse(keys["dtype"].value) == "torch.long"
                and ast.unparse(keys["device"].value) == "device", "host geometry index dtype/device ABI changed")
        keys["device"].value = ast.Constant("cpu")
        changed.append(node.arg)
    require(sorted(changed) == ["global_end_index", "local_end_index"], "host index rewrite census changed")
    scope = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), "exec"), scope)
    return scope[function.__name__]


class HostGeometryIndices:
    """The geometry-aware initializer, with only integer metadata placed on CPU.

    Unlike the historical two-cycle installer, this permits a new pipeline at
    every Program boundary and does not retain old pipeline/cache objects.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.original = trainer.model._initialize_inference_pipeline
        self.function = _host_initializer()
        self.pipelines, self.handles = [], []
        self.record = dict(pipeline_initializations=0, cache_initializations=0,
                           index_tensors_constructed=0, native_attention_calls=0,
                           index_tensor_gpu_value_reads=0, native_kv_geometry_changed=False)
        require(trainer.model.inference_pipeline is None, "host indices must bind before pipeline creation")
        require(trainer.model.generator.model.local_attn_size == -1, "host indices local eviction not qualified")

        def initialize(*args, **kwargs):
            result = self.original(*args, **kwargs)
            pipeline = trainer.model.inference_pipeline
            require(pipeline is not None and pipeline.kv_cache1 is None,
                    "host indices require a new, empty native pipeline")
            original = pipeline._initialize_kv_cache
            from ..self_forcing_source import GeometryCache
            require(getattr(original, "__func__", None) is GeometryCache._initialize_kv_cache
                    and "_initialize_kv_cache" not in vars(pipeline), "unexpected public cache initializer")
            self.pipelines = [reference for reference in self.pipelines if reference() is not None]
            self.pipelines.append(weakref.ref(pipeline))
            self.record["pipeline_initializations"] += 1

            def allocate(this, *args, **kwargs):
                value = self.function(this, *args, **kwargs)
                require(len(this.kv_cache1) == len(trainer.model.generator.model.blocks),
                        "host index native block census changed")
                for cache in this.kv_cache1:
                    self.validate_indices(cache)
                self.record["cache_initializations"] += 1
                self.record["index_tensors_constructed"] += 2 * len(this.kv_cache1)
                return value
            pipeline._initialize_kv_cache = MethodType(allocate, pipeline)
            return result

        self.installed = initialize
        trainer.model._initialize_inference_pipeline = initialize
        for block in trainer.model.generator.model.blocks:
            def before(module, args, kwargs):
                cache = kwargs.get("kv_cache", args[5] if len(args) > 5 else None)
                require(cache is not None, "host index native causal cache required")
                self.validate_indices(cache)
                self.record["native_attention_calls"] += 1
            self.handles.append(block.self_attn.register_forward_pre_hook(before, with_kwargs=True))

    @staticmethod
    def validate_indices(cache):
        import torch
        for name in ("global_end_index", "local_end_index"):
            value = cache[name]
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype is torch.int64
                    and tuple(value.shape) == (1,) and not value.requires_grad,
                    "host index execution contract changed: " + name)

    def receipt(self):
        return dict(self.record, index_devices=["cpu"], state_role="DERIVED_INTEGER_METADATA",
                    native_k_v_placement="UNCHANGED")

    def close(self):
        require(self.trainer.model._initialize_inference_pipeline is self.installed,
                "host index initializer patch ownership changed")
        self.trainer.model._initialize_inference_pipeline = self.original
        for reference in self.pipelines:
            pipeline = reference()
            if pipeline is not None:
                del pipeline._initialize_kv_cache
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pipelines.clear()


def _tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _storage(tensor):
    from torch.distributed.tensor import DTensor
    tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes()) if storage.nbytes() else None


class EarlyConsumedKV:
    """The R6 early-C/after-commit-G policy, without benchmark audit objects.

    Only physical storage metadata is read. The upstream fwdbwd/loss/backward
    and optimizer calls stay intact; every release binds to its actual native
    step and role. Failed work is abandoned, never reused on the next call.
    """
    def __init__(self, trainer):
        import torch
        self.trainer = trainer
        self.pending, self.backward_complete, self.early = None, False, False
        self.original = trainer.fwdbwd_one_step
        self.original_simulation = trainer.model._consistency_backward_simulation
        self.record = dict(generator_commit_releases=0, critic_early_releases=0, critic_commits=0,
                           released_owner_storage_bytes=0, host_tensor_value_reads=0)
        self.handles = []

        def fwdbwd(batch, train_generator):
            require(self.pending is None, "PREVIOUS_KV_PHASE_INCOMPLETE")
            self.pending = (trainer.step, "generator" if train_generator else "critic")
            self.backward_complete = self.early = False
            result = self.original(batch, train_generator)
            self.backward_complete = True
            return result

        def simulate(*args, **kwargs):
            result = self.original_simulation(*args, **kwargs)
            if not torch.is_grad_enabled():
                require(self.pending == (trainer.step, "critic") and not self.backward_complete
                        and not self.early, "EARLY_RELEASE_ROLE_OR_VERSION_CHANGED")
                owned = self._owned()
                for tensor in _tensors(result):
                    require(not tensor.requires_grad and tensor.grad_fn is None,
                            "NOGRAD_CRITIC_RESULT_HAS_AUTOGRAD_GRAPH")
                    require(_storage(tensor) not in owned, "RETURNED_KV_STORAGE_ALIAS")
                self._release(owned)
                self.early = True
                self.record["critic_early_releases"] += 1
            return result

        self.installed, self.installed_simulation = fwdbwd, simulate
        trainer.fwdbwd_one_step = fwdbwd
        trainer.model._consistency_backward_simulation = simulate
        for name, role in (("generator_optimizer", "generator"), ("critic_optimizer", "critic")):
            optimizer = getattr(trainer, name)

            def before(opt, args, kwargs, role=role):
                require(self.pending == (trainer.step, role) and self.backward_complete,
                        "OPTIMIZER_WITH_INCOMPLETE_NATIVE_BACKWARD_OR_WRONG_ROLE")

            def after(opt, args, kwargs, role=role):
                if role == "generator":
                    require(not self.early, "GENERATOR_EARLY_RELEASE_NOT_ADMITTED")
                    self._release(self._owned())
                    self.record["generator_commit_releases"] += 1
                else:
                    require(self.early and not self._live(), "CRITIC_KV_REBUILT_OR_NOT_RELEASED")
                    self.record["critic_commits"] += 1
                self.pending, self.backward_complete, self.early = None, False, False
            self.handles.extend((optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)))

    def _live(self):
        pipeline = self.trainer.model.inference_pipeline
        return pipeline is not None and any(getattr(pipeline, name) is not None
            for name in ("kv_cache1", "kv_cache2", "crossattn_cache"))

    def _owned(self):
        pipeline = self.trainer.model.inference_pipeline
        require(pipeline is not None and pipeline.kv_cache1 is not None
                and pipeline.crossattn_cache is not None, "MISSING_COMPLETED_NATIVE_CACHE")
        owned = {_storage(value) for value in _tensors(
            [pipeline.kv_cache1, pipeline.kv_cache2, pipeline.crossattn_cache])}
        return {key for key in owned if key is not None}

    def _release(self, owned):
        pipeline = self.trainer.model.inference_pipeline
        pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
        self.record["released_owner_storage_bytes"] += sum(key[2] for key in owned)

    def receipt(self):
        require(self.pending is None and not self._live(), "KV_PHASE_INCOMPLETE_AT_BOUNDARY")
        return dict(self.record, pending=None, live_cache_at_boundary=False,
                    mechanism="C_AFTER_NOGRAD_ROLLOUT_G_AFTER_BACKWARD_AND_COMMIT")

    def close(self, *, abort=False):
        if abort:
            pipeline = self.trainer.model.inference_pipeline
            if pipeline is not None:
                pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
            self.pending, self.backward_complete, self.early = None, False, False
        else:
            self.receipt()
        require(self.trainer.fwdbwd_one_step is self.installed
                and self.trainer.model._consistency_backward_simulation is self.installed_simulation,
                "consumed KV wrapper ownership changed")
        self.trainer.fwdbwd_one_step = self.original
        self.trainer.model._consistency_backward_simulation = self.original_simulation
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def phase_boundary(phase):
    """Drain physical execution weights without the legacy step==10 receipt.

    Keeping the helper installed preserves actual optimizer-version counters
    and first-cycle canary knowledge across arbitrary subsequent cycles.
    """
    phase.assert_released()
    require(not phase.closed, "phase retention closed before a live cycle boundary")
    for role in phase.roles:
        phase._release(role)
        phase._master_check(role)
    require(not any(phase.live.values()), "phase execution weights still live at boundary")
    return dict(mechanism="FSDP2_TRAINABLE_EXECUTION_WEIGHT_PHASE_RETENTION",
        native_step=int(phase.trainer.step), counters=deepcopy(phase.counts), physical_state=deepcopy(phase.record),
        retained_compute_weight_bytes=deepcopy(phase.bytes), maximum_retained_bytes_per_role=phase.max_bytes,
        generator_version=phase.g_version, critic_version=phase.c_version, live_at_boundary=deepcopy(phase.live),
        closed=phase.closed, merged_weights=False, checkpoint_ephemeral_state="NOT_SERIALIZED")


class _CanaryClock:
    """Local startup clock for unchanged helper canaries, not the training clock.

    A restored owner requalifies rebuilt ephemeral kernels/caches in its first
    physical cycle. Algorithm/EMA/scheduler calls use the original owner and
    native_step; helpers only observe this source-compatible read-only view.
    """
    def __init__(self, owner):
        self.owner, self.origin = owner, owner.step

    @property
    def step(self):
        return self.owner.step - self.origin

    def __getattr__(self, name):
        return getattr(self.owner, name)


class _PatchSnapshot:
    """Rollback partially installed wrappers; never snapshot tensor values.

    Failed construction/forward poisons the owner and requires reconstruction.
    On abort, hook tables are restored too; on normal close preserve FSDP's own
    hooks created lazily during training and remove only helper-owned handles.
    """
    def __init__(self, owner, api):
        from torch.utils import checkpoint
        self.rows, self.hooks = [], []
        methods = ("forward", "_forward", "clip_grad_norm_", "_initialize_inference_pipeline",
                   "_consistency_backward_simulation", "_compute_kl_grad", "fwdbwd_one_step")
        objects = [owner, *owner.model.modules()]
        for obj in objects:
            for name in methods:
                self.rows.append((obj, name, name in vars(obj), vars(obj).get(name)))
            for name, value in vars(obj).items():
                if "hooks" in name and isinstance(value, dict):
                    self.hooks.append((obj, name, value.copy()))
        for optimizer in (owner.generator_optimizer, owner.critic_optimizer):
            for name in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks"):
                self.hooks.append((optimizer, name, getattr(optimizer, name).copy()))
        for obj, names in ((api.wan, ("rope_apply", "flash_attention")),
                           (api.causal, ("causal_rope_apply", "attention")),
                           (checkpoint, ("checkpoint",))):
            for name in names:
                self.rows.append((obj, name, name in vars(obj), vars(obj).get(name)))

    def restore(self, *, abort):
        for obj, name, existed, value in reversed(self.rows):
            if existed:
                setattr(obj, name, value)
            elif name in vars(obj):
                delattr(obj, name)
        if abort:
            for obj, name, value in self.hooks:
                setattr(obj, name, value)


class SelfForcingOptimizedRuntime:
    """Install the complete R6 bundle on an already constructed native owner.

    API: construct once, ``begin_cycle(cycle)``, execute the unchanged native
    cycle, ``finish_cycle(cycle)`` BEFORE discarding its inference pipeline.
    Finish drains transient storage but does not reset canaries/version counts.
    On any exception call ``abort()``; that owner must be rebuilt/restored.
    Close before source/model disposal. No training math is implemented here.
    """
    def __init__(self, owner, api):
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        require(getattr(owner, "optimization_profile", None) == PROFILE, "explicit optimized owner profile required")
        require(owner.source is api.source and owner.source.files == SOURCE_FILES,
                "source-bound official Self-Forcing revision changed")
        require(owner.identity["upstream"]["revision"] == REVISION, "upstream revision changed")
        require(dist.is_initialized() and dist.get_world_size() == 16 and str(dist.get_backend()) == "nccl"
                and owner.device.type == "cuda" and owner.dtype is torch.bfloat16,
                "optimized runtime requires actual world16 CUDA/NCCL BF16 owner")
        require(owner.args.image_or_video_shape == [1, 21, 16, 46, 80]
                and owner.ratio == 5 and owner.args.gradient_checkpointing is True,
                "optimized R6 native shape/update/checkpoint contract changed")
        for role, dim, blocks in (("generator", 1536, 30), ("fake_score", 1536, 30), ("real_score", 5120, 40)):
            root = getattr(owner.model, role)
            require(isinstance(root, FSDPModule) and root.model.dim == dim and len(root.model.blocks) == blocks
                    and all(isinstance(parameter, DTensor) and parameter.dtype is torch.float32
                            for parameter in root.parameters()), "actual native FSDP2 role/master geometry required: " + role)
        require(all(not parameter.requires_grad for parameter in owner.model.real_score.parameters()),
                "teacher must remain frozen")
        require(hasattr(owner, "generator_optimizer") and hasattr(owner, "critic_optimizer")
                and callable(getattr(owner, "fwdbwd_one_step", None)), "native owner runtime seams missing")
        require(_ACTIVE_SOURCE_PATCH.acquire(blocking=False), "another source-bound SF runtime already owns provider patches")
        self.owner, self.api = owner, api
        self.helpers, self.active_cycle, self.closed, self.poisoned = [], None, False, False
        self.completed_cycles = 0
        self.canary = _CanaryClock(owner)
        self.version_origin = dict(generator=owner.generator_updates, critic=owner.critic_updates)
        self.saved_attention = (api.wan.flash_attention, api.causal.attention)
        self.attention = None
        self.patch_snapshot = None
        try:
            self.patch_snapshot = _PatchSnapshot(owner, api)
            self.modules = SourceBoundHelpers(api.source)
            # The public FP32 SDPA provider is NEVER used by this BF16 path.
            self.attention = api.source.import_module("wan.modules.attention")
            require(self.attention.FLASH_ATTN_2_AVAILABLE and not self.attention.FLASH_ATTN_3_AVAILABLE,
                    "actual native A100 FA2 provider required; no SDPA fallback")
            api.wan.flash_attention = self.attention.flash_attention
            api.causal.attention = self.attention.attention
            native = self.modules.module
            self.phase = self._install(native("sf_phase_retention").PhaseRetention(owner,
                max_retained_bytes=resolve(PROFILE)["contract"]["phase_retention_max_bytes"]))
            self.teacher = self._install(native("sf_teacher_block_pair").TeacherBlockPair(self.canary))
            self.pointwise = self._install(native("sf_grad_pointwise").GradPointwise(self.canary))
            self.rms = self._install(native("sf_grad_rms").GradRMS(self.canary))
            self.rope = self._install(native("sf_grad_rope").AllModeRoPE(self.canary))
            self.modulation = self._install(native("sf_nograd_modulation").NoGradModulation(self.canary,
                ops=native("sf_native_ln_pointwise").build(torch)))
            self.nograd_rms = self._install(native("sf_nograd_rms").NoGradRMS(self.canary))
            self.dense = self._install(native("sf_dense_attention").DenseFA2())
            self.host_indices = self._install(HostGeometryIndices(owner))
            self.replay = self._install(native("sf_fake_projection_replay").CriticAttentionReplay(self.canary))
            self.context = self._install(native("sf_rollout_context").RolloutContext(self.canary))
            self.kv = self._install(EarlyConsumedKV(owner))
        except BaseException:
            self.poisoned = True
            self._close(abort=True)
            raise

    def _install(self, helper):
        self.helpers.append(helper)
        return helper

    def begin_cycle(self, cycle):
        require(not self.closed and not self.poisoned and self.active_cycle is None,
                "optimized runtime closed, poisoned or already executing")
        require(type(cycle) is int and cycle == self.owner.cycle, "optimized native cycle reordered")
        self.modules.validate()
        self.phase.assert_released()
        require(not any(self.phase.live.values()), "execution weights retained across Program boundary")
        self._validate_versions()
        self.active_cycle = cycle

    def finish_cycle(self, cycle):
        require(self.active_cycle == cycle and not self.closed and not self.poisoned,
                "optimized cycle boundary mismatched")
        require(self.owner.cycle == cycle + 1, "native update cycle did not complete")
        self._validate_versions()
        row = dict(profile=PROFILE, native_step=self.owner.step, completed_process_cycles=self.completed_cycles + 1,
            canary_clock_origin=self.canary.origin, numerical_status="REVIEW", integration_gpu_validation="NOT_RUN",
            parameter_versions=dict(generator=self.owner.generator_updates, critic=self.owner.critic_updates),
            helper_parameter_version_origin=deepcopy(self.version_origin),
            phase_retention=phase_boundary(self.phase))
        for name in ("teacher", "pointwise", "rms", "rope", "modulation", "nograd_rms", "dense",
                     "host_indices", "replay", "context", "kv"):
            row[name] = getattr(self, name).receipt()
        self.completed_cycles += 1
        self.active_cycle = None
        return row

    def _validate_versions(self):
        require(self.phase.g_version + self.version_origin["generator"] == self.owner.generator_updates
                and self.phase.c_version + self.version_origin["critic"] == self.owner.critic_updates,
                "helper and owner parameter update versions diverged")

    def abort(self):
        self.poisoned = True
        self._close(abort=True)

    def close(self):
        require(self.active_cycle is None, "close inside an optimized native cycle")
        self._close(abort=False)

    def _close(self, *, abort):
        if self.closed:
            return
        failures = []
        for helper in reversed(self.helpers):
            try:
                helper.close(abort=abort) if isinstance(helper, EarlyConsumedKV) else helper.close()
            except BaseException as exc:
                failures.append(str(exc))
        self.helpers.clear()
        if self.patch_snapshot is not None:
            self.patch_snapshot.restore(abort=abort)
        self.api.wan.flash_attention, self.api.causal.attention = self.saved_attention
        if abort:
            # These are ephemeral checkpoint records, never optimizer/model
            # state. A poisoned owner cannot continue or checkpoint them.
            replay = getattr(self, "replay", None)
            if replay is not None:
                replay.pending.clear()
            context = getattr(self, "context", None)
            if context is not None:
                context.kv.clear()
                context.projected = context.condition = None
                context.active = False
            pipeline = self.owner.model.inference_pipeline
            if pipeline is not None:
                pipeline.kv_cache1 = pipeline.kv_cache2 = pipeline.crossattn_cache = None
        self.active_cycle, self.closed = None, True
        _ACTIVE_SOURCE_PATCH.release()
        self.cleanup_errors = failures
        require(abort or not failures, "optimized runtime cleanup failed: " + "; ".join(failures))
