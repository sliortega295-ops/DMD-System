"""R3 complete-B adapters for the *original* all-role FSDP16 call graph.

This module is opt-in, does not lower plans, and never installs on import.
Canonical C51/R2 installers and their source/precision guards are unchanged.
An installer is first run against pristine symbols, its guarded callable is
captured, and every symbol/marker is restored before any role dispatcher is
installed. This is essential when several installers share WanRMSNorm,
``_modulate``, ``_gate`` or the three flash_attention aliases.

Caller contract: enter ``helper.scope(torch, student)`` under no_grad for ONE
prepare_rollout; enter ``helper.role('student')`` for differentiable G calls;
enter Critic/Teacher roles for their calls (the R2 score_scope API is supported).
Call ``generator_committed(version)`` only after the rollout scope has ended.
The existing FSDP units, groups, hooks and call order are not replaced. Their
first normal forward gathers lazily; only the no-grad reshard window changes.

CPU tests prove dispatch, autograd/RNG and bounded lifecycle behavior, not
target CUDA kernels, target FSDP collectives, numerical admission or speedup.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
from pathlib import Path
import textwrap
from typing import Any, Callable, Sequence


NAME = "fsdp16_all_role_complete_B_r3_v1"
ATTRIBUTE = "_livewan_factorial_r3_b"
FSDP_SOURCE_SHA256 = "207165ff7c5443a4a18ef93d5ac91cee308a32e1afe7a17a8737a9cd9691bb4f"
ROLES = ("student", "critic", "teacher")
_MISSING = object()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("R3 B: " + message)


def _source(value: Any) -> str:
    return hashlib.sha256(textwrap.dedent(inspect.getsource(value)).strip().encode()).hexdigest()


def capture_installer(installer: Callable, targets: Sequence[tuple[Any, str]]):
    """Capture an unchanged source-bound installer, then restore its globals.

    Original installers use only ``_distillschedule*`` bookkeeping on these
    owners. Snapshot that namespace as well as the functions, including on a
    failed install. No dispatch wrapper is visible to a subsequent source hash.
    """
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    owners = {id(owner): owner for owner, _ in targets}.values()
    markers = [(owner, {key: value for key, value in vars(owner).items()
                        if key.startswith("_distillschedule")}) for owner in owners]
    try:
        receipt = installer()
        captured = [getattr(owner, name) for owner, name in targets]
        _require(all(fn is not native for fn, (_, _, native) in zip(captured, originals)),
                 "requested installer did not install its callable")
        return captured, receipt
    finally:
        for owner, name, native in originals:
            setattr(owner, name, native)
        for owner, before in markers:
            for name in tuple(vars(owner)):
                if name.startswith("_distillschedule") and name not in before:
                    delattr(owner, name)
            for name, value in before.items():
                setattr(owner, name, value)


class RoleDispatcher:
    """Explicit role dispatch, including an explicit None checkpoint binding."""

    def __init__(self, torch: Any):
        self.torch = torch
        self._role = ContextVar("livewan_r3_b_role", default=None)
        self.entries = {role: 0 for role in ROLES}
        self.failures = 0
        self.targets: list[tuple[Any, str, Any, Any]] = []
        self.attempts = {role: {} for role in (*ROLES, "original")}
        self.completed = {role: {} for role in (*ROLES, "original")}
        self.original_calls: dict[str, int] = {}

    @property
    def current_role(self):
        return self._role.get()

    @contextmanager
    def role(self, role: str | None):
        _require(role is None or role in ROLES, "unsupported role")
        token = self._role.set(role)
        if role is not None:
            self.entries[role] += 1
        try:
            yield
        except Exception:
            self.failures += 1
            raise
        finally:
            self._role.reset(token)

    @contextmanager
    def captured_role(self, role: str | None):
        # Autograd workers do not inherit the original thread's ContextVar.
        # None means original Student dispatch, not the replay thread's role.
        _require(role is None or role in ROLES, "unsupported captured role")
        token = self._role.set(role)
        try:
            yield
        finally:
            self._role.reset(token)

    def add(self, key: str, owner: Any, name: str, *, score: Any = None,
            student: Any = None, student_nograd_only: bool = False,
            student_eligible: Any = None):
        _require(not any(o is owner and n == name for o, n, _, _ in self.targets),
                 "duplicate role-dispatch target: " + key)
        native = getattr(owner, name)
        for counters in (self.attempts, self.completed):
            for values in counters.values():
                values[key] = 0
        self.original_calls[key] = 0

        def dispatch(*args, **kwargs):
            role = self.current_role
            selected, label = native, "original"
            if role in ("critic", "teacher") and score is not None:
                selected, label = score, role
            elif (role == "student" and student is not None
                  and (not student_nograd_only or not self.torch.is_grad_enabled())
                  and (student_eligible is None or student_eligible(*args, **kwargs))):
                selected, label = student, role
            self.attempts[label][key] += 1
            if label == "original":
                self.original_calls[key] += 1
            result = selected(*args, **kwargs)
            # Early-stop replay may interrupt the leaf; never invent a hit.
            self.completed[label][key] += 1
            return result

        setattr(owner, name, dispatch)
        self.targets.append((owner, name, native, dispatch))

    def restore(self):
        for owner, name, native, installed in reversed(self.targets):
            _require(getattr(owner, name) is installed, "dispatch target replaced before restoration")
            setattr(owner, name, native)
        self.targets.clear()

    def receipt(self):
        return dict(name=NAME, current_role=self.current_role,
            scope_entries=dict(self.entries), scope_failures=self.failures,
            dispatch_attempts={role: dict(values) for role, values in self.attempts.items()},
            dispatch_completed={role: dict(values) for role, values in self.completed.items()},
            original_calls=dict(self.original_calls),
            counter_scope="ACTUAL_DISPATCH_ATTEMPTS_AND_COMPLETIONS_INCLUDING_REPLAY")


class RoleCheckpointProxy:
    """Original checkpoint/RNG policy, with per-call captured role and SAC.

    ``score_contexts`` is the unchanged pinned C51 FA3 policy factory. It creates
    fresh GPU SAC storage per Critic checkpoint with mutation protection. Other
    roles use the original recompute policy, never Critic attention storage.
    """

    def __init__(self, module: Any, torch: Any, dispatcher: RoleDispatcher,
                 *, score_contexts: Any = None):
        self.module, self.torch, self.dispatcher = module, torch, dispatcher
        self.contexts = score_contexts
        self.calls = self.function_entries = self.replay_entries = 0
        self.sac_calls = 0
        self.role_calls = {role: 0 for role in (*ROLES, "original")}
        self.install_target = None

    def checkpoint(self, function, *args, **kwargs):
        _require(kwargs.get("use_reentrant") is False and "context_fn" not in kwargs
                 and "preserve_rng_state" not in kwargs,
                 "native non-reentrant checkpoint/RNG/context call site changed")
        role = self.dispatcher.current_role
        self.calls += 1
        self.role_calls[role or "original"] += 1
        entries = 0

        def captured(*values, **options):
            nonlocal entries
            self.function_entries += 1
            self.replay_entries += int(entries != 0)
            entries += 1
            with self.dispatcher.captured_role(role):
                return function(*values, **options)

        if role == "critic" and self.contexts is not None and self.torch.is_grad_enabled():
            self.sac_calls += 1
            return self.module.checkpoint(captured, *args, context_fn=self.contexts, **kwargs)
        return self.module.checkpoint(captured, *args, **kwargs)

    def receipt(self):
        installed = (self.install_target is not None
                     and getattr(self.install_target, "ckpt", None) is self)
        result = dict(name="fsdp16_captured_role_fa3_sac_r3_v1", checkpoint_calls=self.calls,
            function_entries=self.function_entries, replay_entries=self.replay_entries,
            sac_checkpoint_calls=self.sac_calls, role_checkpoint_calls=dict(self.role_calls),
            rng_policy="UNCHANGED_CHECKPOINT_DEFAULT_TRUE", all_role_rank_gate=False,
            production_installed=installed)
        if self.contexts is not None:
            result["fa3_sac"] = dict(self.contexts.receipt(), production_installed=installed)
        return result


def _tensor_signature(tensor: Any):
    return (id(tensor), int(tensor._version), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), bool(tensor.requires_grad))


def _cross_parameter_signature(cross: Any):
    # Host metadata only; retain no parameter objects, storage pointers or views.
    return tuple((prefix, name, _tensor_signature(p))
                 for prefix in ("k", "v", "norm_k")
                 for name, p in getattr(cross, prefix).named_parameters())


class LazyTextKV:
    """Detached owned text K/V, born only inside an entered gathered block."""

    def __init__(self, torch: Any, dispatcher: RoleDispatcher,
                 original_forward: Any, attention: Any):
        self.torch, self.dispatcher = torch, dispatcher
        self.original_forward, self.attention = original_forward, attention
        self._state = ContextVar("livewan_r3_lazy_text_kv", default=None)
        self._gathered = ContextVar("livewan_r3_gathered_unit", default=None)
        self.counters = dict(cache_build_context_count=0, cache_build_block_count=0,
            cache_build_bytes=0, cache_hit_count=0, grad_bypass_count=0,
            uncached_fallback_count=0, cache_invalidation_count=0,
            contract_failure_count=0, scope_entries=0, scope_exits=0)
        self._active_state = None

    @property
    def active(self):
        return self._state.get() is not None

    @property
    def entries(self):
        state = self._state.get()
        return state["entries"] if state is not None else {}

    @contextmanager
    def scope(self, *, generator_version: int):
        _require(not self.active and self._active_state is None, "recursive text-KV scope")
        _require(type(generator_version) is int and generator_version >= 0, "invalid G version")
        state = dict(version=generator_version, context=None, context_signature=None,
                     streamer=None, entries={})
        token = self._state.set(state)
        self._active_state = state
        self.counters["scope_entries"] += 1
        try:
            yield
        finally:
            state["entries"].clear()
            state["context"] = state["streamer"] = None
            self._active_state = None
            self._state.reset(token)
            self.counters["scope_exits"] += 1

    @contextmanager
    def gathered(self, layer: Any):
        token = self._gathered.set(layer)
        try:
            yield
        finally:
            self._gathered.reset(token)

    def _fail(self, message):
        self.counters["contract_failure_count"] += 1
        if self.entries:
            self.entries.clear()
            self.counters["cache_invalidation_count"] += 1
        raise RuntimeError("R3 B text-KV: " + message)

    def _validate_context(self, context, lens):
        _require(context.is_cuda and context.dtype == self.torch.float32
                 and not context.requires_grad and tuple(context.shape) == (1, 512, 1536)
                 and context.is_contiguous() and lens is None,
                 "full-length FP32 text context/device/shape/length changed")

    def _validate_query(self, cross, x, context):
        _require(x.is_cuda and x.device == context.device and x.dtype == self.torch.float32
                 and not x.requires_grad and x.ndim == 3 and x.shape[0] == 1
                 and x.shape[-1] == 1536 and cross.num_heads == 12 and cross.head_dim == 128,
                 "text-KV query geometry/dtype changed")
        _require(all(not hasattr(p, "to_local") and p.is_cuda and p.dtype == self.torch.float32
                     for module in (cross.k, cross.v, cross.norm_k) for p in module.parameters()),
                 "text K/V projection parameters were not materialized inside gathered block")

    def _validate_outputs(self, key, value):
        _require(key.dtype == self.torch.float32 and value.dtype == self.torch.bfloat16
                 and tuple(key.shape) == tuple(value.shape) == (1, 512, 12, 128)
                 and key.is_contiguous() and value.is_contiguous(),
                 "projected text K/V dtype/layout changed")

    def _autocast(self):
        return self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16)

    def bind(self, streamer, context, context_lens):
        state = self._state.get()
        _require(state is not None, "set_text binding outside rollout")
        try:
            self._validate_context(context, context_lens)
        except RuntimeError as error:
            self._fail(str(error))
        if state["context"] is not None:
            self.counters["cache_invalidation_count"] += 1
            state["entries"].clear()
        state.update(context=context, context_signature=_tensor_signature(context),
                     streamer=streamer)
        self.counters["cache_build_context_count"] += 1
        # Deliberately NO block traversal, parameter access, projection or gather.

    def forward(self, cross, x, context, context_lens):
        if self.torch.is_grad_enabled():
            self.counters["grad_bypass_count"] += 1
            return self.original_forward(cross, x, context, context_lens)
        if self.dispatcher.current_role != "student" or not self.active:
            self.counters["uncached_fallback_count"] += 1
            return self.original_forward(cross, x, context, context_lens)
        state = self._state.get()
        layer = self._gathered.get()
        if layer is None or getattr(getattr(layer, "blk", None), "cross_attn", None) is not cross:
            self._fail("cache access outside its entered gathered CausalBlock")
        if (state["context"] is not context
                or _tensor_signature(context) != state["context_signature"]):
            self._fail("prompt/context identity, contents or metadata changed")
        try:
            self._validate_context(context, context_lens)
            self._validate_query(cross, x, context)
        except RuntimeError as error:
            self._fail(str(error))
        signature = _cross_parameter_signature(cross)
        entry = state["entries"].get(id(cross))
        if entry is not None and (entry["parameter_signature"] != signature
                                  or entry["version"] != state["version"]):
            self._fail("materialized projection parameter identity/version changed")
        with self._autocast():
            if entry is None:
                # Original projection/norm functions, only after the FSDP hook
                # has made this block's tensors available. Own detached output
                # allocations; never retain FSDP parameter/all-gather views.
                b, length = context.shape[:2]
                n, d = cross.num_heads, cross.head_dim
                key = cross.norm_k(cross.k(context)).view(b, length, n, d).detach().clone()
                value = cross.v(context).view(b, length, n, d).detach().clone()
                try:
                    self._validate_outputs(key, value)
                except RuntimeError as error:
                    self._fail(str(error))
                entry = dict(key=key, value=value, parameter_signature=signature,
                             version=state["version"])
                state["entries"][id(cross)] = entry
                self.counters["cache_build_block_count"] += 1
                self.counters["cache_build_bytes"] += sum(t.numel() * t.element_size()
                                                           for t in (key, value))
            query = cross.norm_q(cross.q(x)).view(x.shape[0], -1, cross.num_heads, cross.head_dim)
            output = self.attention(query, entry["key"], entry["value"], k_lens=None)
            output = cross.o(output.flatten(2))
        self.counters["cache_hit_count"] += 1
        return output

    def receipt(self):
        state = self._active_state
        entries = state["entries"].values() if state is not None else ()
        return dict(self.counters, name="fsdp_gathered_lazy_text_kv_r3_v1", active=state is not None,
            retained_cache_bytes=sum(t.numel() * t.element_size() for e in entries
                                     for t in (e["key"], e["value"])),
            parameter_views_cached=0, cache_cross_optimizer=False,
            construction_scope="ENTERED_CAUSAL_BLOCK_ONLY")


def _model_versions(student):
    # Called before entering and after restoring original sharded parameters.
    return tuple((name, int(p._version)) for name, p in student.named_parameters())


class FSDPRolloutRetention:
    """Retain original FSDP-unit materializations for one no-grad rollout.

    Uses only target-verified Torch2.10 public FSDPModule APIs. No eager unshard,
    no new group, shard policy, parameter copies, prefetch order or grad policy.
    ``unit_type``/``device_type`` support explicit CPU lifecycle stand-ins in
    tests; the production installer pins FSDPModule, CUDA and Torch2.10.
    """

    def __init__(self, torch, student, *, unit_type=None, device_type="cuda"):
        if unit_type is None:
            from torch.distributed.fsdp import FSDPModule
            unit_type = FSDPModule
        self.torch, self.student = torch, student
        self.unit_type, self.device_type = unit_type, device_type
        self.units = tuple(m for m in student.modules() if isinstance(m, unit_type))
        _require(bool(self.units), "no original FSDP materialization units")
        _require(all(callable(getattr(unit, "set_reshard_after_forward", None))
                     and callable(getattr(unit, "reshard", None)) for unit in self.units),
                 "target FSDP retention API unavailable")
        self.active = False
        self.entries = self.exits = self.failures = 0
        self.retain_calls = self.reshard_calls = self.restore_calls = 0
        self.cache_clears = self.version_failures = 0

    @contextmanager
    def scope(self, torch, student):
        _require(torch is self.torch and student is self.student, "retention owner changed")
        _require(not self.active, "recursive FSDP rollout retention")
        _require(not torch.is_grad_enabled(), "FSDP rollout retention requires no_grad")
        _require(not torch.is_inference_mode_enabled(), "retention requires no_grad, not inference_mode")
        _require(not torch.is_autocast_enabled(self.device_type)
                 and torch.is_autocast_cache_enabled(), "outer autocast/cache contract changed")
        _require(tuple(m for m in student.modules() if isinstance(m, self.unit_type)) == self.units,
                 "original FSDP units changed")
        before = _model_versions(student)
        changed = []
        self.active = True
        self.entries += 1
        body_error = None
        try:
            for unit in self.units:
                changed.append(unit)
                unit.set_reshard_after_forward(False, recurse=False)
                self.retain_calls += 1
            # Exact C28 lifetime primitive; existing inner BF16 autocast scopes
            # still decide actual compute dtype. Clear before reshard below.
            from .livewan_rollout_cast_cache import rollout_cast_cache_scope
            with rollout_cast_cache_scope(torch, enabled=True, device_type=self.device_type):
                yield
        except BaseException as error:
            self.failures += 1
            body_error = error
            raise
        finally:
            cleanup_errors = []
            try:
                torch.clear_autocast_cache()
                self.cache_clears += 1
            except Exception as error:
                cleanup_errors.append(error)
            for unit in changed:
                try:
                    unit.reshard()
                    self.reshard_calls += 1
                except Exception as error:
                    cleanup_errors.append(error)
                finally:
                    try:
                        unit.set_reshard_after_forward(True, recurse=False)
                        self.restore_calls += 1
                    except Exception as error:
                        cleanup_errors.append(error)
            if _model_versions(student) != before:
                self.version_failures += 1
                cleanup_errors.append(RuntimeError("R3 B: parameter versions changed inside rollout"))
            self.active = False
            self.exits += 1
            if cleanup_errors:
                self.failures += 1
                if body_error is None:
                    raise RuntimeError("R3 B retention cleanup: " + "; ".join(map(str, cleanup_errors)))
                if hasattr(body_error, "add_note"):
                    body_error.add_note("R3 B cleanup errors: " + "; ".join(map(str, cleanup_errors)))

    def receipt(self):
        return dict(name="fsdp16_nograd_lazy_materialization_retention_r3_v1",
            unit_count=len(self.units), scope_entries=self.entries, scope_exits=self.exits,
            active=self.active, contract_failures=self.failures,
            retain_policy_calls=self.retain_calls, reshard_calls=self.reshard_calls,
            restore_policy_calls=self.restore_calls, autocast_cache_clear_count=self.cache_clears,
            parameter_version_failures=self.version_failures,
            eager_unshard_calls=0, changes_effective_dtype=False, cache_cross_optimizer=False)


class CompleteB:
    """Installed R3 runtime handles; receipts report execution, never plan flags."""

    def __init__(self, torch, student):
        self.torch, self.student = torch, student
        self.dispatch = RoleDispatcher(torch)
        self.kernel_receipts: dict[str, Any] = {}
        self.retention = FSDPRolloutRetention(torch, student)
        self.kv: LazyTextKV | None = None
        self.checkpoint: RoleCheckpointProxy | None = None
        self.generator_version = 0
        self.commits = 0
        self._restorations: list[tuple[Any, str, Any]] = []

    def role(self, role):
        return self.dispatch.role(role)

    @contextmanager
    def scope(self, torch, student, *, generator_version=None):
        _require(torch is self.torch and student is self.student, "rollout owner changed")
        version = self.generator_version if generator_version is None else generator_version
        _require(version == self.generator_version, "stale logical Generator version")
        _require(self.kv is not None, "lazy KV not installed")
        # KV exits/clears before autocast is cleared and FSDP storage is resharded.
        with self.dispatch.role("student"), self.retention.scope(torch, student), \
                self.kv.scope(generator_version=version):
            yield

    def generator_committed(self, version):
        _require(not self.retention.active and self.kv is not None and not self.kv.active,
                 "Generator commit attempted inside rollout/cache scope")
        _require(type(version) is int and version == self.generator_version + 1,
                 "Generator commit version skipped or repeated")
        self.generator_version = version
        self.commits += 1

    def patch(self, owner, name, function):
        self._restorations.append((owner, name, getattr(owner, name, _MISSING)))
        setattr(owner, name, function)

    def uninstall(self):
        _require(not self.retention.active, "cannot uninstall active rollout")
        for owner, name, original in reversed(self._restorations):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self._restorations.clear()
        self.dispatch.restore()

    def receipt(self):
        kernels = {}
        for key, raw in self.kernel_receipts.items():
            row = dict(raw)
            count = sum(int(row.get(name, 0)) for name in
                        ("calls", "forward_calls", "invocation_count",
                         "autocast_input_elision_invocation_count"))
            row["execution_status"] = "PASS" if count > 0 else "NOT_RUN"
            kernels[key] = row
        gate = kernels.get("generator_gate", {})
        kernels["generator_residual_layout"] = dict(
            name="generator_nograd_row_major_residual_v1", output_layout=gate.get("output_layout"),
            calls=gate.get("calls", 0), execution_status=gate.get("execution_status", "NOT_RUN"),
            same_kernel_as="generator_gate")
        splitkv = dict(kernels.get("generator_attention_backend", {}).get("readonly_splitkv", {}))
        splitkv["execution_status"] = "PASS" if splitkv.get("eligible_nograd_calls", 0) > 0 else "NOT_RUN"
        kernels["generator_readonly_splitkv"] = splitkv
        return dict(name=NAME, role_dispatch=self.dispatch.receipt(),
            generator_cross_attention_kv_cache=self.kv.receipt() if self.kv else {},
            generator_rollout_cast_cache=self.retention.receipt(),
            critic_attention_checkpoint=self.checkpoint.receipt() if self.checkpoint else {},
            generator_commit_version=self.generator_version, observed_generator_commits=self.commits,
            kernels=kernels, placement="UNCHANGED_ORIGINAL_FSDP16_UNITS_AND_GROUPS",
            target_validation="PENDING_CLUSTER_VALIDATION")


def install(torch: Any, *, bundle: Any) -> CompleteB:
    """Install every C51 operator under explicit all-role R3 FSDP16 dispatch.

    Must run after model construction/FSDP wrapping, before any other Cxx/R2
    installer. The caller owns exact plan/source/topology/asset and timing gates.
    This function fails closed on an already-patched upstream namespace.
    """
    _require(str(torch.__version__) == "2.10.0+cu128", "validated target Torch identity changed")
    from wan.modules import attention as attention_module, model as model_module
    from wanstreamer import blockcausal as bc
    from wanstreamer.fsdp import CausalBlock, CausalHead
    from wanstreamer.lora import LoRALinear
    from wanstreamer.stream import FewStepStreamer
    from .livewan_attention_backend import install_generator_flash_attn2
    from .livewan_cross_kv_cache import EXPECTED_CROSS_FORWARD_SHA256, EXPECTED_SET_TEXT_SHA256
    from .livewan_generator_fused_rms import build as build_g_rms
    from .livewan_generator_rms_epilogue import REFERENCE_CLASS_SHA256
    from .livewan_generator_modulation import build as build_g_mod, SOURCE_SHA256 as MOD_SOURCE
    from .livewan_generator_gate import build as build_g_gate, SOURCE_SHA256 as GATE_SOURCE
    from .livewan_lora_kernel import install_scale_one_lora_epilogue
    from .livewan_modulation_kernel import install_score_modulation_kernel
    from .livewan_score_rmsnorm import install_frozen_score_rmsnorm
    from .livewan_rope_kernel import install_score_rope_kernel
    from .livewan_score_gate import install_score_gate
    from .livewan_score_layernorm import install_frozen_score_layernorm
    from .livewan_score_fa3 import install_score_flash_attn3, load_flash_attn3, CriticFA3CheckpointProxy

    student = bundle.student
    _require(all(getattr(bundle, role, None) is not None for role in ROLES), "all-role models missing")
    _require(not hasattr(student, ATTRIBUTE)
             and not any(hasattr(getattr(bundle, role), "_livewan_fsdp_score_scope") for role in ROLES),
             "R3 or another role dispatcher already installed")
    _require(len(student.blocks) == len(student.causal_layers) == 30
             and all(isinstance(layer, CausalBlock) and layer.blk is block
                     for layer, block in zip(student.causal_layers, student.blocks))
             and isinstance(student.causal_head, CausalHead), "original gathered-unit topology changed")
    fsdp_source = Path(inspect.getfile(CausalBlock))
    _require(hashlib.sha256(fsdp_source.read_bytes()).hexdigest() == FSDP_SOURCE_SHA256,
             "pinned CausalBlock/FSDP hook source changed")
    owners = (bc, attention_module, model_module, model_module.WanRMSNorm,
              model_module.WanLayerNorm, LoRALinear, FewStepStreamer,
              model_module.WanT2VCrossAttention)
    _require(not any(key.startswith("_distillschedule") for owner in owners for key in vars(owner)),
             "pristine source required before overlapping installers")
    _require(_source(FewStepStreamer.set_text) == EXPECTED_SET_TEXT_SHA256
             and _source(model_module.WanT2VCrossAttention.forward) == EXPECTED_CROSS_FORWARD_SHA256
             and _source(bc._modulate) == MOD_SOURCE and _source(bc._gate) == GATE_SOURCE
             and hashlib.sha256(inspect.getsource(model_module.WanRMSNorm).encode()).hexdigest()
             == REFERENCE_CLASS_SHA256, "pinned Generator math source changed")
    loras = [m for m in bundle.critic.modules() if isinstance(m, LoRALinear)]
    _require(len(loras) == 180 and all(m.scale == 1.0 for m in loras),
             "original Critic LoRA topology/scale changed")
    helper = CompleteB(torch, student)
    _require(len(helper.retention.units) == 35, "original Generator FSDP unit count changed")
    definitions = (
        ("critic_lora_kernel", LoRALinear, "forward",
         lambda: install_scale_one_lora_epilogue(torch, autocast_input_elision=True)),
        ("score_modulation_kernel", bc, "_modulate", lambda: install_score_modulation_kernel(torch)),
        ("score_rmsnorm_kernel", model_module.WanRMSNorm, "forward",
         lambda: install_frozen_score_rmsnorm(torch)),
        ("score_rope_kernel", bc, "apply_rope", lambda: install_score_rope_kernel(torch)),
        ("score_residual_gate", bc, "_gate", lambda: install_score_gate(torch)),
        ("score_layernorm", model_module.WanLayerNorm, "forward",
         lambda: install_frozen_score_layernorm(torch)),
    )
    captured = {}
    for key, owner, name, installer in definitions:
        functions, helper.kernel_receipts[key] = capture_installer(installer, [(owner, name)])
        captured[key] = functions[0]
    attention_targets = [(owner, "flash_attention") for owner in
                         (attention_module, model_module, bc)]
    score_attention, helper.kernel_receipts["score_attention_backend"] = capture_installer(
        lambda: install_score_flash_attn3(torch), attention_targets)
    g_attention, helper.kernel_receipts["generator_attention_backend"] = capture_installer(
        lambda: install_generator_flash_attn2(torch, readonly_splitkv=True), attention_targets)
    g_rms, helper.kernel_receipts["generator_rms_epilogue"] = build_g_rms(torch)
    g_mod, helper.kernel_receipts["generator_modulation"] = build_g_mod(torch)
    g_gate, helper.kernel_receipts["generator_gate"] = build_g_gate(torch, output_layout="row_major")

    # All source-bound capturing above completed with pristine originals. Only
    # now install the shared role dispatchers and composition wrappers.
    kv = helper.kv = LazyTextKV(torch, helper.dispatch,
        model_module.WanT2VCrossAttention.forward,
        lambda *a, **k: model_module.flash_attention(*a, **k))
    student_units = {id(layer) for layer in student.causal_layers} | {id(student.causal_head)}

    def gathered_eligible(*args, **kwargs):
        _require(not helper.retention.active or not torch.is_grad_enabled(),
                 "grad-enabled execution entered retained no-grad rollout")
        return kv.active and kv._gathered.get() is not None

    def rms_forward(module, x):
        return g_rms(x, module.weight, module.eps)

    native_mod, native_gate = bc._modulate, bc._gate

    def modulate(x, norm, shift, scale, per_frame, n_frames):
        if per_frame or n_frames != 1:
            return native_mod(x, norm, shift, scale, per_frame, n_frames)
        return g_mod(norm(x), scale, shift)

    def gate(x, y, g, per_frame, n_frames):
        if per_frame or n_frames != 1:
            return native_gate(x, y, g, per_frame, n_frames)
        return g_gate(x, y, g)

    g_variants = dict(score_modulation_kernel=modulate, score_rmsnorm_kernel=rms_forward,
                      score_residual_gate=gate)
    try:
        for key, owner, name, _ in definitions:
            helper.dispatch.add(key, owner, name, score=captured[key],
                student=g_variants.get(key), student_nograd_only=True,
                student_eligible=gathered_eligible)
        for index, (owner, name) in enumerate(attention_targets):
            helper.dispatch.add("attention_" + str(index), owner, name,
                score=score_attention[index], student=g_attention[index])
        helper.dispatch.add("lazy_cross_kv", model_module.WanT2VCrossAttention, "forward",
                            student=kv.forward)
        for cls in (CausalBlock, CausalHead):
            native = cls.forward

            def gathered_forward(layer, *args, _native=native, **kwargs):
                if (helper.dispatch.current_role == "student" and kv.active
                        and id(layer) in student_units):
                    _require(not torch.is_grad_enabled(), "grad execution inside retained rollout")
                    with kv.gathered(layer):
                        return _native(layer, *args, **kwargs)
                return _native(layer, *args, **kwargs)

            helper.patch(cls, "forward", gathered_forward)
        native_set_text = FewStepStreamer.set_text

        def set_text(streamer, context, context_lens=None):
            result = native_set_text(streamer, context, context_lens)
            if helper.dispatch.current_role == "student" and kv.active:
                _require(streamer.model is student and streamer.ctx is context,
                         "rollout streamer/context owner changed")
                kv.bind(streamer, context, streamer.ctx_lens)
            return result

        helper.patch(FewStepStreamer, "set_text", set_text)
        flash_module, _ = load_flash_attn3()
        original_proxy = CriticFA3CheckpointProxy(bc.ckpt, torch, flash_module)
        helper.checkpoint = RoleCheckpointProxy(bc.ckpt, torch, helper.dispatch,
                                               score_contexts=original_proxy.contexts)
        helper.patch(bc, "ckpt", helper.checkpoint)
        helper.checkpoint.install_target = bc
    except BaseException:
        helper.uninstall()
        raise
    # Existing worker score_scope can use these without canonical R2 changes.
    for role in ROLES:
        helper.patch(getattr(bundle, role), "_livewan_fsdp_score_scope", helper.dispatch)
    helper.patch(student, ATTRIBUTE, helper)
    helper.patch(bundle, "system_factorial_r3_b", helper)
    return helper


def model_scope(torch, student):
    """No implicit activation: the worker must install the exact R3 plan first."""
    helper = getattr(student, ATTRIBUTE, None)
    _require(helper is not None, "R3 rollout helper not installed")
    return helper.scope(torch, student)


__all__ = ["NAME", "ATTRIBUTE", "CompleteB", "RoleDispatcher", "RoleCheckpointProxy",
           "LazyTextKV", "FSDPRolloutRetention", "capture_installer", "install", "model_scope"]
