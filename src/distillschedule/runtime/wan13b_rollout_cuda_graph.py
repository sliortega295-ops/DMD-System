"""Default-off, exact Resident CUDA Graph replay for post-commit B1 rollouts.

Training/reduction/optimizer/EMA and FP32 LoRA storage remain unchanged. Graph
inputs are copied and outputs cloned on the caller's stream. Python hook counts
are NOT GPU replay counts: warmup/capture/reference work is reported separately.
"""
from __future__ import annotations

import os
from typing import Any

ENV = "DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_CUDA_GRAPH"
PLACEMENT = "resident-async128-tree-chunk2048"
MAX_GRAPH_ENTRIES = 12
WARMUP_CALLS = 3


def requested() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def signature(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        if value.requires_grad:
            raise ValueError("graph input must be detached")
        return ("tensor", tuple(value.shape), tuple(value.stride()),
                str(value.dtype), str(value.device))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("graph dictionary keys must be strings")
        return ("dict", tuple((key, signature(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(signature(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return (type(value).__name__, str(value))
    raise ValueError(f"unsupported CUDA Graph input metadata: {type(value)}")


def clone_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        # Preserve non-dense views as well as dense layouts; values are never
        # read by the host, and input copies remain stream-ordered.
        result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                     device=value.device)
        result.copy_(value)
        return result
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    signature(value)  # Fail closed on unsupported non-tensor objects.
    return value


def copy_tree(destination: Any, source: Any) -> None:
    import torch

    if torch.is_tensor(source):
        destination.copy_(source)
    elif isinstance(source, dict):
        for key, item in source.items():
            copy_tree(destination[key], item)
    elif isinstance(source, (list, tuple)):
        for target, item in zip(destination, source, strict=True):
            copy_tree(target, item)


class RolloutCudaGraph:
    def __init__(self, source: Any) -> None:
        from .wan13b_rollout_graph_pool import IndependentRolloutPool

        self.source = source
        self.graph_pool = IndependentRolloutPool()
        self._audit_source()
        self._version: int | None = None
        self._pointer_contract = self._parameter_identity()
        self._cache: dict[Any, tuple[Any, Any, Any, Any]] = {}
        self.receipt: dict[str, Any] = {
            "requested": True, "enabled": True,
            "scope": "FIVE_POST_GENERATOR_CRITIC_INPUT_B1_ROLLOUTS_ONLY",
            "weights_merged": False, "training_master_unchanged": True,
            "optimizer_reducer_ema_unchanged": True,
            "input_transport": "STREAM_ORDERED_COPY_INTO_STATIC_TENSORS",
            "output_transport": "STREAM_ORDERED_FRESH_CLONE_NO_ALIAS",
            "cache_entry_count": 0, "cache_entries": [],
            "warmup_model_call_count": 0, "capture_model_call_count": 0,
            "graph_replay_model_call_count": 0, "model_call_count": 0,
            "rollout_entry_count": 0, "versions": [], "entries": [],
            "diagnostic_reference_model_call_count": 0,
            "diagnostic_reference_entry_count": 0,
            "diagnostic_reference_parity": [], "active": False,
            "cold_capture_scope": "FIRST_CYCLE_ONLY_WARMUP_DIAGNOSTIC",
            "cold_calls_are_extra_not_application_forward": True,
            "rng_ops": "NONE_REQUIRE_ZERO_DROPOUT_PINNED_SOURCE",
            "scratch_pool": self.graph_pool.receipt,
        }

    def _audit_source(self) -> None:
        import torch

        from peft.tuners.lora.layer import LoraLayer

        if not self.source.training:
            raise ValueError("Generator CUDA Graph requires unchanged train mode")
        for name, parameter in self.source.named_parameters():
            if parameter.device.type != "cuda" or hasattr(parameter, "device_mesh"):
                raise ValueError("CUDA Graph requires full Resident CUDA parameters")
            if parameter.requires_grad and ("lora_" not in name or parameter.dtype != torch.float32):
                raise ValueError("CUDA Graph must preserve FP32 LoRA trainables")
        for name, layer in self.source.named_modules():
            if isinstance(layer, torch.nn.modules.dropout._DropoutNd) and layer.p:
                raise ValueError(f"{name}: stochastic dropout is outside graph contract")
            if isinstance(layer, LoraLayer):
                if len(layer.active_adapters) != 1 or layer.disable_adapters or layer.merged:
                    raise ValueError(f"{name}: Generator adapter state changed")

    def _parameter_identity(self) -> tuple[Any, ...]:
        # Host metadata only. Rechecked at each G commit, never read CUDA values.
        return tuple((name, parameter.data_ptr(), tuple(parameter.shape),
                      str(parameter.dtype), str(parameter.device))
                     for name, parameter in self.source.named_parameters())

    def begin_rollout(self, *, version: int, logical_steps: tuple[int, ...]) -> None:
        if self.receipt["active"] or len(logical_steps) != 1:
            raise ValueError("CUDA Graph requires non-nested post-G B1 scope")
        version = int(version)
        if self._version != version:
            if version != (3 if self._version is None else self._version + 1):
                raise ValueError("CUDA Graph must bind consecutive G3/G4 commits")
            self._audit_source()
            if self._parameter_identity() != self._pointer_contract:
                raise ValueError("CUDA Graph parameter addresses/layout changed after G commit")
            self._version = version
            self.receipt["versions"].append(version)
        self.receipt["active"] = True
        self.receipt["rollout_entry_count"] += 1
        self.receipt["entries"].append({"version": version, "logical_steps": list(logical_steps)})

    def end_rollout(self) -> None:
        if not self.receipt["active"]:
            raise ValueError("CUDA Graph scope was not active")
        self.receipt["active"] = False

    def _capture(self, args: Any, kwargs: Any) -> tuple[Any, Any, Any, Any]:
        import torch

        static_args, static_kwargs = clone_tree(args), clone_tree(kwargs)
        options = self.graph_pool.capture_options(torch)
        stream = options.get("stream") or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_CALLS):
                self.source(*static_args, **static_kwargs)
                self.receipt["warmup_model_call_count"] += 1
        torch.cuda.current_stream().wait_stream(stream)
        # Cold, first-cycle capture only. No synchronization in steady replay.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream,
                              **{k: v for k, v in options.items() if k != "stream"}):
            output = self.source(*static_args, **static_kwargs)
        self.graph_pool.verify_capture(graph)
        self.receipt["capture_model_call_count"] += 1
        return static_args, static_kwargs, graph, output

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if torch.is_grad_enabled() or not self.receipt["active"]:
            raise ValueError("CUDA Graph can execute only post-G no-grad B1")
        self.graph_pool.before_call(torch)
        key = signature((args, kwargs))
        if key not in self._cache:
            if self._version != 3:
                raise ValueError("new CUDA Graph signature after warmup cycle is forbidden")
            if len(self._cache) >= MAX_GRAPH_ENTRIES:
                raise ValueError("CUDA Graph cache capacity exceeded")
            self._cache[key] = self._capture(args, kwargs)
            self.receipt["cache_entry_count"] += 1
            self.receipt["cache_entries"].append({
                "input_signature": repr(key),
                "latent_spatial_shape": list(kwargs["hidden_states"].shape[-2:]),
            })
        static_args, static_kwargs, graph, output = self._cache[key]
        copy_tree(static_args, args)
        copy_tree(static_kwargs, kwargs)
        graph.replay()
        self.receipt["model_call_count"] += 1
        self.receipt["graph_replay_model_call_count"] += 1
        # Cached graph outputs must not be returned by reference: another replay
        # may overwrite them while earlier Critic inputs are still retained.
        return clone_tree(output)

    def enable_adapters(self) -> None:
        if not self.receipt["active"]:
            raise ValueError("CUDA Graph adapter request outside rollout scope")

    def train(self, mode: bool = True) -> Any:
        if bool(mode) != bool(self.source.training):
            raise ValueError("CUDA Graph cannot change Generator mode")
        return self


def install(source: Any, *, role: str, placement: str) -> dict[str, Any]:
    enabled = requested()
    receipt = {"requested": enabled, "enabled": False, "role": role}
    if not enabled:
        return {**receipt, "disabled_reason": "NOT_REQUESTED"}
    if placement != PLACEMENT:
        raise ValueError("CUDA Graph requires the registered Resident control")
    # Do not import or register the rejected merged-LoRA execution path.
    if os.environ.get("DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_MERGED_SHADOW", "0") != "0":
        raise ValueError("CUDA Graph cannot be combined with merged execution weights")
    if role != "generator":
        return {**receipt, "disabled_reason": "GENERATOR_ONLY"}
    controller = RolloutCudaGraph(source)
    object.__setattr__(source, "_distillschedule_rollout_cuda_graph", controller)
    controller.receipt["role"] = role
    return controller.receipt


def validate_receipt(receipt: dict[str, Any], *, enabled: bool, cycles: int,
                     diagnostic_reference: bool) -> None:
    if not enabled:
        if receipt.get("enabled"):
            raise AssertionError("unrequested CUDA Graph executed")
        return
    expected = {"requested": True, "enabled": True, "weights_merged": False,
                "active": False, "model_call_count": 27 * cycles,
                "graph_replay_model_call_count": 27 * cycles,
                "rollout_entry_count": 5 * cycles,
                "versions": list(range(3, 3 + cycles))}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise AssertionError(f"CUDA Graph census changed {key}: {receipt.get(key)} != {value}")
    entries = [{"version": 3 + cycle, "logical_steps": [step]}
               for cycle in range(cycles) for step in range(10, 15)]
    if receipt.get("entries") != entries:
        raise AssertionError("CUDA Graph sample/version routing changed")
    count = receipt.get("cache_entry_count", 0)
    if not 3 <= count <= MAX_GRAPH_ENTRIES:
        raise AssertionError("CUDA Graph three-shape cache is missing/overflowed")
    if (receipt.get("capture_model_call_count") != count
            or receipt.get("warmup_model_call_count") != WARMUP_CALLS * count):
        raise AssertionError("CUDA Graph cold extra call inventory changed")
    spatial_shapes = {tuple(row["latent_spatial_shape"]) for row in receipt.get("cache_entries", [])}
    if spatial_shapes != {(12, 20), (24, 40), (48, 80)}:
        raise AssertionError("CUDA Graph actual pyramid shapes changed")
    ref_count = 5 * cycles if diagnostic_reference else 0
    if (receipt.get("diagnostic_reference_entry_count") != ref_count
            or receipt.get("diagnostic_reference_model_call_count") != (27 * cycles if diagnostic_reference else 0)):
        raise AssertionError("CUDA Graph explicit reference census changed")
    parity = receipt.get("diagnostic_reference_parity", [])
    if len(parity) != ref_count or any(row.get("status") != "PASS" for row in parity):
        raise AssertionError("CUDA Graph trained G3/G4 B1 parity failed/missing")
    if diagnostic_reference and [row.get("identity") for row in parity] != entries:
        raise AssertionError("CUDA Graph reference parity identity changed")


def validate(build: dict[str, Any], *, enabled: bool, cycles: int,
             diagnostic_reference: bool = False) -> dict[str, Any]:
    generator = build.get("generator", {}).get("metadata", {}).get("generator_rollout_cuda_graph", {})
    critic = build.get("critic", {}).get("metadata", {}).get("generator_rollout_cuda_graph", {})
    validate_receipt(generator, enabled=enabled, cycles=cycles,
                     diagnostic_reference=diagnostic_reference)
    if critic.get("enabled") or (enabled and critic.get("disabled_reason") != "GENERATOR_ONLY"):
        raise AssertionError("Critic must not use Generator CUDA Graphs")
    return {"enabled": enabled, "generator": generator, "critic": critic}
