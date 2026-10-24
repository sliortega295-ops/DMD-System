"""Default-off Resident Critic forward/backward CUDA graphs, original reducer.

Only the five adapter-on grad DiTs are graphed. Optimizer, EMA and gradient
reduction remain outside graphs. Captures use autograd.grad, not loss.backward,
so they cannot fire the engine's post-accumulate communication hooks. Graph
backward returns gradients to the original autograd accumulators and hooks.
"""
from __future__ import annotations
import functools
import os
from typing import Any
from .wan13b_rollout_cuda_graph import clone_tree, signature

ENV = "DISTILLSCHEDULE_WAN13B_CRITIC_GRAD_CUDA_GRAPH"
PLACEMENT = "resident-async128-tree-chunk2048"


def requested() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def check_admission(placement: str, *, enabled: bool, role: str) -> bool:
    if not enabled:
        return False
    if placement != PLACEMENT:
        raise ValueError("Critic grad graphs require full Resident placement")
    return role == "critic"


class CriticGradGraph:
    def __init__(self, source: Any, *, reference_checks: bool) -> None:
        self.source = source
        self.original_forward = source.forward
        self.reference_checks = reference_checks
        self.state = getattr(source, "_distillschedule_resident_clean_state", None)
        if self.state is None:
            raise ValueError("Critic grad graph requires original mode state machine")
        self.pointer_contract = self.parameter_identity()
        self.cache: dict[Any, Any] = {}
        self._stream_binding = None
        self._cold_version = None
        self._stream_storage = None
        self._cycle_start = 0
        self.version_reader = None
        self._audit_source()
        self.receipt: dict[str, Any] = dict(
            requested=True, enabled=True, role="critic",
            scope="FIVE_CRITIC_GRAD_DIT_FORWARD_BACKWARD_ONLY",
            weights_merged=False, parameter_dtype_unchanged=True,
            optimizer_reducer_ema_unchanged=True,
            replay_calls=0, backward_replay_calls=0, reference_calls=0,
            capture_entries=0, cold_original_forward_calls=0,
            cold_original_backward_calls=0, reference_parity=[],
            captured_parameter_gradients_return_to_original_accumulators=True,
            rng_ops="NONE_ZERO_DROPOUT",
            collective_overlap_risk="ALL_GRAPH_GRADS_READY_AFTER_BACKWARD_REPLAY",
        )

    def parameter_identity(self) -> Any:
        return tuple((name, value.data_ptr(), tuple(value.shape), str(value.dtype),
                      str(value.device)) for name, value in self.source.named_parameters())

    def _audit_source(self) -> None:
        import torch
        for name, p in self.source.named_parameters():
            if not p.is_cuda or hasattr(p, "device_mesh"):
                raise ValueError(name + ": Resident CUDA parameters required")
            if p.requires_grad and ("lora_" not in name or p.dtype != torch.float32):
                raise ValueError(name + ": original FP32 LoRA trainables required")
        for name, child in self.source.named_modules():
            if isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.p:
                raise ValueError(name + ": stochastic dropout forbidden")

    def bind_stream_cycle(self, *, version, logical_steps, source):
        from .wan_bundle_lifecycle import validate_capture_transition, capture_storage_identity
        self._audit_source()
        if (self.receipt['replay_calls'] != self.receipt['backward_replay_calls']
                or self.parameter_identity() != self.pointer_contract):
            raise ValueError('Critic stream has pending backward or changed parameter addresses')
        self._stream_binding = validate_capture_transition(self._stream_binding,
            version=version, logical_steps=logical_steps, source=source, increment=5)
        self._cycle_start = self.receipt['replay_calls']
        if self._cold_version is None:
            if self.cache or self._cycle_start:
                raise ValueError('stream admission requires a fresh uncaptured Critic controller')
            self._cold_version = version
            self._stream_storage = capture_storage_identity(self.source)
            self.receipt['stream_lifecycle'] = dict(source=source, first_version=version,
                                                   versions=[], rng_checked_captures=0)

    def _capture(self, keys: Any, values: Any) -> Any:
        import torch
        owner = self

        class NativeTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = owner.source  # Expose ORIGINAL, shared parameter objects.

            def forward(self, *xs):
                output = owner.original_forward(**dict(zip(keys, xs, strict=True)),
                                                return_dict=False)
                return output[0]

        static_values = clone_tree(values)
        wrapped = NativeTensorModule()
        # First-cycle only. This captures BOTH original checkpointed forward
        # and autograd.grad backward. No optimizer/reducer/EMA is invoked here.
        graphed = torch.cuda.make_graphed_callables(
            wrapped, static_values, num_warmup_iters=3, allow_unused_input=False)
        self.receipt["capture_entries"] += 1
        self.receipt["cold_original_forward_calls"] += 4
        self.receipt["cold_original_backward_calls"] += 4
        return graphed

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        import torch
        if not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if args or kwargs.get("return_dict") is not False:
            raise ValueError("Pinned Critic kwargs / tuple output required")
        if not self.source.training or not self.state.training or not self.state.adapter_enabled:
            raise ValueError("Critic grad graph must use original adapter-on train scope")
        if self._stream_binding is not None:
            from .wan_bundle_lifecycle import capture_storage_identity
            offset = self.receipt['replay_calls'] - self._cycle_start
            version = self._stream_binding['version'] + offset
            if (offset >= 5 or self.version_reader is None or self.version_reader() != version
                    or self.receipt['replay_calls'] != self.receipt['backward_replay_calls']):
                raise ValueError('Critic stream version/order/pending backward changed')
            if (self.parameter_identity() != self.pointer_contract
                    or capture_storage_identity(self.source) != self._stream_storage):
                raise ValueError('Critic stream parameter addresses changed')
        if self.receipt["replay_calls"] % 5 == 0 and self.parameter_identity() != self.pointer_contract:
            raise ValueError("Critic grad graph parameter addresses changed")
        keys = tuple(sorted(k for k in kwargs if k != "return_dict"))
        values = tuple(kwargs[k] for k in keys)
        if any(not torch.is_tensor(v) or v.requires_grad for v in values):
            raise ValueError("Detached tensor inputs required")
        key = (keys, signature(values))
        if key not in self.cache:
            cold = (self.receipt['replay_calls'] < 5 if self._stream_binding is None else
                    self._stream_binding['version'] == self._cold_version)
            if not cold or len(self.cache) >= 3:
                raise ValueError("New Critic grad graph signature after cold cycle")
            if self._stream_binding is None:
                self.cache[key] = self._capture(keys, values)
            else:
                from .wan_bundle_lifecycle import capture_rng_guard
                with capture_rng_guard(self.source):
                    entry = self._capture(keys, values)
                self.cache[key] = entry
                self.receipt['stream_lifecycle']['rng_checked_captures'] += 1
        output = self.cache[key](*values)
        if not output.requires_grad:
            raise ValueError("Critic graph output lost autograd")

        def backward_count(gradient):
            self.receipt["backward_replay_calls"] += 1
            return gradient

        output.register_hook(backward_count)
        self.receipt["replay_calls"] += 1
        if self._stream_binding is not None:
            self.receipt['stream_lifecycle']['versions'].append(version)
        if self.reference_checks:
            from .p0_parity import require_tensor_parity
            reference = self.original_forward(*args, **kwargs)[0]
            self.receipt["reference_parity"].append(require_tensor_parity(
                reference, output, label="Resident Critic grad CUDA graph"))
            self.receipt["reference_calls"] += 1
        return (output,)


def install(source: Any, *, role: str, placement: str,
            reference_checks: bool) -> dict[str, Any]:
    enabled = requested()
    if not check_admission(placement, enabled=enabled, role=role):
        return dict(requested=enabled, enabled=False, role=role)
    controller = CriticGradGraph(source, reference_checks=reference_checks)

    @functools.wraps(source.forward)
    def forward(*args, **kwargs):
        return controller.forward(*args, **kwargs)

    source.forward = forward
    object.__setattr__(source, "_distillschedule_critic_grad_cuda_graph", controller)
    return controller.receipt


def validate_build(build: dict[str, Any], *, cycles: int,
                   reference_checks: bool) -> dict[str, Any]:
    g = build.get("generator", {}).get("metadata", {}).get("critic_grad_cuda_graph", {})
    c = build.get("critic", {}).get("metadata", {}).get("critic_grad_cuda_graph", {})
    if g.get("enabled"):
        raise AssertionError("Critic graph must not enter Generator")
    if not requested():
        if c.get("enabled"):
            raise AssertionError("Unrequested Critic grad graph executed")
        return dict(generator=g, critic=c)
    for key, value in dict(enabled=True, replay_calls=5*cycles,
                           backward_replay_calls=5*cycles,
                           reference_calls=5*cycles if reference_checks else 0).items():
        if c.get(key) != value:
            raise AssertionError(f"Critic graph census changed {key}")
    entries = c.get("capture_entries", 0)
    if not 1 <= entries <= 3 or any(c.get(k) != 4*entries for k in
            ("cold_original_forward_calls", "cold_original_backward_calls")):
        raise AssertionError("Critic graph cold capture inventory changed")
    return dict(generator=g, critic=c)
