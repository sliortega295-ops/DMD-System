"""Source-bound W2 SafeBundle policy, independent of frozen benchmark assets.

Only physical controls live here. Original P0 owns model construction, kernels,
losses, Adam/LR/EMA and attention binary admission; helios_lifecycle owns data
and restart. This policy never calls the historical implementation.environment.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from distillschedule.search.helios_implementation import HeliosImplementation, current_implementation
from distillschedule.search.helios_safe_bundle import HeliosSafeBundle, FLAG_ENV

PRESET_ID = "helios_nogan.safe_stream_ed8.v1"
SCHEMA = "distillschedule.helios_safe_stream_policy.v1"
TOPOLOGY = {"mesh_policy": "node-hsdp", "mesh": [list(range(8)), list(range(8, 16))],
            "mesh_dim_names": ["replicate", "shard"], "device_type": "cuda",
            "placements": [{"kind": "replicate"}, {"kind": "shard", "dim": 0}]}


def implementation():
    return HeliosImplementation("no-gan", "node-hsdp", 512, 5, False, HeliosSafeBundle.safe())


def effective_flags():
    from .helios_worker import _OFF
    enabled = set(FLAG_ENV) | {"all_role_lora_scale1_add", "optimizer_dispatch_warmup",
                              "generator_nograd_residual_fusion", "exact_grad_residual_fusion"}
    return {name.replace("-", "_"): name.replace("-", "_") in enabled for name in _OFF}


def descriptor():
    impl = implementation()
    return {"schema_version": SCHEMA, "preset": PRESET_ID, "implementation": impl.assignment,
            "candidate_bundle": impl.candidate_bundle.cost_identity(), "flags": effective_flags(),
            "attention": "local-fa2", "selective_flash_checkpoint_stride": 0,
            "optimizer": "original_restored_AdamW", "topology": _topology(),
            "numerical_status": "REVIEW", "gpu_execution": "NOT_RUN"}


def _topology():
    from copy import deepcopy
    return deepcopy(TOPOLOGY)


def validate_environment(environment=None, *, require_kernel=False):
    """Reject inherited winners; never install, download or invent LOCAL_KERNELS.

    Existence is only an early dependency gate. Original P0 subsequently checks
    the exact extension SHA256 AND the actually imported FA2 callable unchanged.
    """
    env = os.environ if environment is None else environment
    require(current_implementation(env) is None,
            "safe stream cannot inherit a generated benchmark plan", path="helios.safe_stream")
    require(not env.get("PYTORCH_CUDA_ALLOC_CONF") and not env.get("PYTORCH_ALLOC_CONF"),
            "safe stream allocator environment differs", path="helios.safe_stream")
    flags = effective_flags()
    common = {"elide_explicit_rollout_sync", "elide_critic_scalar_logs", "elide_official_generator_free_memory",
              "optimizer_dispatch_warmup", "cuda_allocator_expandable_segments"}
    wanted = {"DISTILLSCHEDULE_" + ("" if name in common else "HELIOS_") + name.upper():
              "on" if value else "off" for name, value in flags.items()}
    wanted.update(DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS="512",
                  DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE="0",
                  DISTILLSCHEDULE_HELIOS_ATTENTION_BACKEND="local-fa2",
                  DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD_FLUSH_ALLOCATOR_CACHE="on",
                  DISTILLSCHEDULE_EMA_DEVICE="cuda",
                  DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE="cuda_out_of_place",
                  DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE="cuda_out_of_place")
    wanted["DISTILLSCHEDULE_HELIOS_PLANNER_MODE"] = "off"
    for key, value in env.items():
        if key.startswith("DISTILLSCHEDULE_"):
            require(key in wanted and value == wanted[key],
                    "safe stream inherited unbound physical/asset control: " + key, path="helios.safe_stream")
    if require_kernel:
        prefix = "kernels-community/flash-attn2="
        mapping = env.get("LOCAL_KERNELS", "")
        require(mapping.startswith(prefix) and Path(mapping[len(prefix):]).is_absolute(),
                "safe stream requires explicit existing LOCAL_KERNELS=kernels-community/flash-attn2=<absolute path>",
                path="helios.safe_stream")
        root = Path(mapping[len(prefix):])
        require(root.is_dir() and (root / "_flash_attn2_cuda_a511a4c.abi3.so").is_file(),
                "local FlashAttention extension is missing; native GPU dependency NOT_RUN",
                path="helios.safe_stream")


def bind_effective(*, args, policy, environment=None):
    validate_environment(environment)
    require(args.variant == "no-gan" and args.timing_mode and not args.profile_mode and not args.profile_diagnostics
            and not getattr(args, "validate_only", False) and args.critic_only_step is None and not args.operation_source,
            "safe stream requires complete unprofiled original No-GAN helpers", path="helios.safe_stream")
    for name, expected in effective_flags().items():
        require(getattr(args, name, None) is expected,
                "safe stream effective argument differs: " + name, path="helios.safe_stream")
        if hasattr(policy, name):
            require(getattr(policy, name) is expected,
                    "safe stream effective policy differs: " + name, path="helios.safe_stream")
    require(policy.mesh_policy == args.mesh_policy == "node-hsdp"
            and policy.modulation_chunk_tokens == args.modulation_chunk_tokens == 512
            and policy.rollout_batch_size == args.rollout_batch_size == 1
            and policy.rollout_backend == args.rollout_backend == "native-fsdp2-batch"
            and policy.generator_nograd_persistent_unshard_flush_allocator_cache is args.generator_nograd_persistent_unshard_flush_allocator_cache is True
            and type(policy.selective_flash_checkpoint_stride) is int
            and policy.selective_flash_checkpoint_stride == args.selective_flash_checkpoint_stride == 0
            and args.attention_backend == "local-fa2" and args.ema_device == "cuda"
            and args.generator_residual_storage == args.critic_residual_storage == "cuda_out_of_place",
            "safe stream physical policy changed (node-HSDP/C512/p0/window5/stride0)", path="helios.safe_stream")
    result = descriptor()
    result["candidate_bundle"] = implementation().candidate_bundle.bind_effective(args=args, policy=policy)
    result["effective_configuration_sha256"] = fingerprint(result)
    return result


def owner_layout(generator, critic):
    """Validate real native local-shard ownership before restore/checkpoint IO.

    No tensor copies or generic trial clones. Common checkpoint stores each
    rank's local layout and rejects changed meshes before mutating live owners.
    """
    from .native_state import _named
    from .tensor_state import layout
    from torch.distributed.tensor import DTensor
    wanted = _topology()
    rows = {}
    for role, engine in (("generator", generator), ("critic", critic)):
        _, parameters, _ = _named(engine)
        require(bool(parameters), "safe stream missing native trainables", path="helios.safe_stream")
        rows[role] = {}
        for name, parameter in parameters.items():
            require(isinstance(parameter, DTensor), "safe stream trainable is not a native DTensor", path="helios.safe_stream")
            info = layout(parameter)
            require(all(info[key] == wanted[key] for key in ("mesh", "mesh_dim_names", "device_type", "placements")),
                    "safe stream node-HSDP owner topology/layout changed", path="helios.safe_stream")
            require(info["dtype"] == "torch.float32", "safe stream original FP32 trainable changed", path="helios.safe_stream")
            rows[role][name] = info
    # Uneven local shard sizes are rank-qualified in the checkpoint payload,
    # not compared across different ranks or confused with the global layout.
    global_rows = {role: {name: {key:value for key,value in info.items() if key != "local_shape"}
                         for name, info in parameters.items()} for role, parameters in rows.items()}
    return {"topology": wanted, "trainables_sha256": fingerprint(global_rows)}


def validate_layout_record(value):
    require(isinstance(value, dict) and set(value) == {"topology", "trainables_sha256"}
            and value["topology"] == _topology()
            and isinstance(value["trainables_sha256"], str) and re.fullmatch("[0-9a-f]{64}", value["trainables_sha256"]),
            "safe stream checkpoint native node-HSDP layout missing/changed", path="helios.safe_stream")
