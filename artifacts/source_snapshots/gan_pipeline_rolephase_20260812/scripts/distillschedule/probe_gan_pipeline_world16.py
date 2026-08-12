#!/usr/bin/env python3
"""Real world16 persistent 8G+8Score Helios GAN Full5 executor.

This executable is intentionally narrow: it replays the frozen systems-only
GAN trace for logical steps 1000--1004, uses role-local ZeRO-1/GAS2, and moves
only BF16 latents/latent gradients between paired ranks.  It is the functional
gate for the PR5 schedule, not a generic pipeline implementation.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.activation_offload import install_activation_offload_hooks
from runtime.critic_pool import ChecksumKind, StudentDPGroupAdapter
from runtime.critic_pool_helios import (
    LoadedPoolEntry,
    build_helios_pool_model,
    run_generator_trace_rollout,
)
from runtime.frozen_block_streaming import (
    HostStatePolicy,
    install_frozen_block_streamer,
)
from runtime.official_baseline import SparseShardEMA
from runtime.operation_trace import OperationTrace, replay_stage3_source


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _groups(rank: int):
    control = dist.new_group(list(range(16)), backend="nccl")
    g_group = dist.new_group(list(range(8)), backend="nccl")
    s_group = dist.new_group(list(range(8, 16)), backend="nccl")
    pair = None
    for lane in range(8):
        group = dist.new_group([lane, lane + 8], backend="nccl")
        if rank in (lane, lane + 8):
            pair = group
    singleton = None
    for member in range(16):
        group = dist.new_group([member], backend="nccl")
        if rank == member:
            singleton = group
    assert pair is not None and singleton is not None
    return control, g_group, s_group, pair, singleton


def _ds_config() -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 16,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "overlap_comm": False,
            "contiguous_gradients": False,
            "reduce_bucket_size": 16 * 1024 * 1024,
            "allgather_bucket_size": 16 * 1024 * 1024,
        },
    }


def _resolved(engine: Any, group: Any) -> dict[str, int]:
    def val(name: str) -> int:
        x = getattr(engine, name)
        return int(x() if callable(x) else x)
    got = {
        "dp_world_size": val("dp_world_size"),
        "train_micro_batch_size_per_gpu": val("train_micro_batch_size_per_gpu"),
        "gradient_accumulation_steps": val("gradient_accumulation_steps"),
        "train_batch_size": val("train_batch_size"),
    }
    expected = {
        "dp_world_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 16,
    }
    if got != expected or getattr(engine.optimizer, "dp_process_group", None) is not group:
        raise AssertionError(f"role-local ZeRO-1 contract drift: {got} != {expected}")
    return got


class _Accelerator:
    def __init__(self, device: torch.device, engine: Any | None = None):
        self.device = device
        self.is_main_process = False
        self.deepspeed_engine_wrapped = (
            SimpleNamespace(engine=engine) if engine is not None else None
        )

    def backward(self, loss: torch.Tensor) -> None:
        if self.deepspeed_engine_wrapped is None:
            loss.backward()
        else:
            self.deepspeed_engine_wrapped.engine.backward(loss)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # The probe passes the role-local DeepSpeed ``engine.module`` directly,
        # so no wrapper remains to remove.  This mirrors Accelerate's public
        # interface without changing the model object or adapter state.
        return model


def _scheduler(device: torch.device):
    from helios.scheduler.scheduling_helios import HeliosScheduler
    return HeliosScheduler(shift=1.0, stages=3, stage_range=[0, 1/3, 2/3, 1], gamma=1/3)


def _trace(root: Path, slot: int) -> OperationTrace:
    return OperationTrace.load(root / f"rank{slot}")


def _source(trace: OperationTrace, device: torch.device) -> dict[str, torch.Tensor]:
    return replay_stage3_source(trace, device=device, dtype=torch.bfloat16)


def _load_official_config(path: Path) -> Any:
    """Hydrate the YAML through the same dataclass schema as train_helios.py."""

    from helios.utils.train_config import Args

    return OmegaConf.merge(OmegaConf.structured(Args), OmegaConf.load(path))


@torch.no_grad()
def _load_logical_trainables(model: torch.nn.Module, package: Path, component: str) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text())
    spec = manifest["components"][component]
    # Restore the BF16 module values used by the sequential denominator.  The
    # FP32 master and Adam buffers are restored separately from their logical
    # payloads below; putting the master payload directly in
    # the module changes both memory residency and training arithmetic.
    payload = spec["payloads"]["module"]
    if payload.get("dtype") != "torch.bfloat16":
        raise AssertionError(f"{component} module payload is not BF16: {payload}")
    raw = torch.from_file(
        str(package / payload["path"]), shared=False,
        size=int(payload["bytes"]) // 2, dtype=torch.bfloat16,
    )
    trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
    names = list(spec["parameter_names"])
    if set(trainables) != set(names):
        missing = sorted(set(names) - set(trainables)); extra = sorted(set(trainables) - set(names))
        raise AssertionError(f"{component} logical inventory drift missing={missing[:3]} extra={extra[:3]}")
    if list(trainables) != names:
        raise AssertionError(f"{component} logical parameter order drift")
    for name in names:
        rec = spec["records"][name]
        value = raw.narrow(0, int(rec["offset_elements"]), int(rec["numel"])).view(tuple(rec["shape"]))
        parameter = trainables[name]
        if rec["module"]["dtype"] != "torch.bfloat16":
            raise AssertionError(f"{component}/{name} module record is not BF16")
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.bfloat16()
        parameter.copy_(value)
    del raw
    return spec


@torch.no_grad()
def _restore_logical_zero1_optimizer(
    engine: Any,
    package: Path,
    spec: dict[str, Any],
    *,
    role_rank: int,
    role_world_size: int = 8,
) -> dict[str, Any]:
    """Repartition logical FP32 Adam state into the live BF16 ZeRO-1 layout.

    The target package's native shards were templated with FP32 module tensors,
    whose DeepSpeed alignment differs by one element per rank from the official
    BF16 module layout for the odd-sized GAN Critic inventory.  Loading those
    shards directly therefore cannot be correct.  The logical package is the
    authority: slice its flat master/moment payloads at the live partition
    boundaries and leave only alignment padding as zero.
    """

    zero = engine.optimizer
    partitions = list(zero.single_partition_of_fp32_groups)
    if len(partitions) != 1:
        raise AssertionError("GAN pipeline expects one ZeRO-1 optimizer group")
    partition = partitions[0]
    if partition.device.type != "cpu" or partition.dtype != torch.float32:
        raise AssertionError(
            f"unexpected ZeRO-1 master placement: {partition.device}/{partition.dtype}"
        )
    if not 0 <= role_rank < role_world_size:
        raise AssertionError(f"invalid role-local rank {role_rank}")
    partition_numel = int(partition.numel())
    flat_numel = int(zero.bit16_groups_flat[0].numel())
    if flat_numel != partition_numel * role_world_size:
        raise AssertionError(
            f"live ZeRO partition geometry drift: {flat_numel} != "
            f"{partition_numel}*{role_world_size}"
        )
    logical_numel = int(spec["logical_numel"])
    padding_numel = flat_numel - logical_numel
    if padding_numel < 0 or padding_numel >= 2 * role_world_size:
        raise AssertionError(
            f"unexpected live ZeRO alignment padding: {padding_numel}"
        )
    start = role_rank * partition_numel
    valid_numel = max(0, min(partition_numel, logical_numel - start))

    def local_payload(kind: str) -> torch.Tensor:
        payload = spec["payloads"][kind]
        if payload.get("dtype") != "torch.float32":
            raise AssertionError(f"{kind} payload is not FP32: {payload}")
        if int(payload["bytes"]) != logical_numel * 4:
            raise AssertionError(f"{kind} payload length changed")
        source = torch.from_file(
            str(package / payload["path"]),
            shared=False,
            size=logical_numel,
            dtype=torch.float32,
        )
        local = torch.zeros(partition_numel, dtype=torch.float32, device="cpu")
        if valid_numel:
            local[:valid_numel].copy_(source.narrow(0, start, valid_numel))
        del source
        return local

    master = local_payload("master")
    partition.copy_(master)
    del master
    exp_avg = local_payload("first_moment")
    exp_avg_sq = local_payload("second_moment")
    base_optimizer = zero.optimizer
    base_params = base_optimizer.param_groups[0]["params"] if len(base_optimizer.param_groups) == 1 else []
    if len(base_params) != 1 or base_params[0] is not partition:
        raise AssertionError("DeepSpeed base optimizer partition identity drift")
    state = base_optimizer.state[partition]
    state.clear()
    state.update(
        step=torch.tensor(
            float(spec["optimizer_step"]), dtype=torch.float32, device="cpu"
        ),
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )
    return {
        "source": "logical_package_repartition",
        "role_rank": role_rank,
        "role_world_size": role_world_size,
        "logical_numel": logical_numel,
        "flat_numel": flat_numel,
        "partition_numel": partition_numel,
        "padding_numel": padding_numel,
        "valid_numel": valid_numel,
        "optimizer_step": int(spec["optimizer_step"]),
    }


def _common_call(args_cfg: Any) -> dict[str, Any]:
    t, v = args_cfg.training_config, args_cfg.validation_config
    return dict(
        is_keep_x0=True,
        history_sizes=list(t.history_sizes),
        is_enable_stage2=bool(t.is_enable_stage2),
        stage2_num_stages=int(t.stage2_num_stages),
        stage2_num_inference_steps_list=list(v.stage2_simulated_inference_steps),
        denoising_step_list=list(t.dmd_denoising_step_list),
        last_step_only=bool(t.dmd_last_step_only),
        last_section_grad_only=bool(t.dmd_last_section_grad_only),
        timestep_shift=float(t.dmd_timestep_shift),
        use_dynamic_shifting=bool(t.use_dynamic_shifting),
        time_shift_type=str(t.time_shift_type),
        num_critic_input_frames=int(t.num_critic_input_frames),
        num_rollout_sections=1,
        is_skip_first_section=bool(getattr(t, "is_skip_first_section", False)),
        is_amplify_first_chunk=bool(t.is_amplify_first_chunk),
        is_corrupt_history_latents=bool(t.corrupt_history),
        is_add_saturation=bool(t.is_add_saturation),
        is_use_gt_history=True,
        is_dmd_vae_decode=False,
        is_multi_pyramid_stage_backward_simulated=bool(t.is_multi_pyramid_stage_backward_simulated),
        use_kv_cache=bool(getattr(v, "use_kv_cache", False)),
    )


def _loaded_critic(trace: OperationTrace, source: dict[str, torch.Tensor], step: int) -> LoadedPoolEntry:
    tensors = {
        "entry_noise": trace.tensor(f"step/{step}/critic/entry_noise").to(source["text_embedding"].device, dtype=torch.bfloat16),
        "text_embedding": source["text_embedding"],
    }
    for out, key in (
        ("indices_hidden_states", "indices_hidden_states"),
        ("indices_history_short", "indices_history_short"),
        ("indices_history_mid", "indices_history_mid"),
        ("indices_history_long", "indices_history_long"),
        ("history_short", "history_short"),
        ("history_mid", "history_mid"),
        ("history_long", "history_long"),
        ("continuous_history", "continuous_history"),
    ):
        value = trace.tensor(f"step/{step}/critic/{key}")
        tensors[out] = value.to(
            source["text_embedding"].device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
    entry = SimpleNamespace(
        rollout_operation=f"step/{step}/critic/rollout",
        logical_step=step,
        trace_key=f"rank-qualified/{trace.trace_hash}/step/{step}",
        identity_sha256=trace.trace_hash,
    )
    return LoadedPoolEntry(entry, trace, tensors)


class _ResidentFrozenBlocks:
    """No-op lifecycle handle for the explicit k=0 capacity candidate."""

    def __init__(self) -> None:
        self.streamed_block_indices = frozenset()
        self.stats: dict[str, int | float] = {
            "captured_parameter_count": 0,
            "captured_bytes": 0,
            "streamed_block_count": 0,
            "materialize_count": 0,
            "materialized_bytes": 0,
            "release_count": 0,
            "released_bytes": 0,
            "peak_resident_bytes": 0,
            "resident_bytes": 0,
            "pinned_host_bytes": 0,
            "h2d_wall_seconds": 0.0,
            "phase_resident_enter_count": 0,
            "phase_resident_exit_count": 0,
        }

    @contextlib.contextmanager
    def phase_resident_no_grad(self, *, phase: str):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "resident frozen blocks require an enclosing torch.no_grad()"
            )
        if not str(phase).strip():
            raise ValueError("resident phase must be non-empty")
        self.stats["phase_resident_enter_count"] += 1
        try:
            yield
        finally:
            self.stats["phase_resident_exit_count"] += 1

    def remove(self) -> None:
        if int(self.stats["resident_bytes"]) != 0:
            raise AssertionError("resident-only frozen-block handle leaked state")


def _parse_tail_streaming_spec(
    *, count_name: str, indices_name: str
) -> tuple[int, tuple[int, ...]]:
    raw_count = os.environ.get(count_name)
    raw_indices = os.environ.get(indices_name)
    if raw_count is None or raw_indices is None:
        raise ValueError(
            f"role-aware streaming requires both {count_name} and {indices_name}"
        )
    count = int(raw_count)
    if not 0 <= count <= 40:
        raise ValueError(f"{count_name} must be in [0,40]")
    indices = tuple(int(value) for value in raw_indices.split(",") if value.strip())
    expected = tuple(range(40 - count, 40)) if count else ()
    if indices != expected:
        raise ValueError(
            f"{indices_name} requires nested tail indices {expected}, got {indices}"
        )
    return count, indices


def _capacity_streaming_spec(role: str) -> tuple[int, tuple[int, ...]]:
    enabled = os.environ.get("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH", "0").strip()
    if enabled not in {"0", "1"}:
        raise ValueError("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH must be 0 or 1")
    if role not in {"g", "s"}:
        raise ValueError(f"unknown pipeline role: {role}")

    prefix = "GENERATOR" if role == "g" else "SCORE"
    role_count = f"DISTILLSCHEDULE_GAN_{prefix}_STREAMED_BLOCK_COUNT"
    role_indices = f"DISTILLSCHEDULE_GAN_{prefix}_STREAMED_BLOCK_INDICES"
    role_specific_names = tuple(
        f"DISTILLSCHEDULE_GAN_{component}_{suffix}"
        for component in ("GENERATOR", "SCORE")
        for suffix in ("STREAMED_BLOCK_COUNT", "STREAMED_BLOCK_INDICES")
    )
    role_specific_present = any(
        os.environ.get(name) is not None for name in role_specific_names
    )
    if role_specific_present:
        if any(
            os.environ.get(name) is not None
            for name in (
                "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT",
                "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES",
            )
        ):
            raise ValueError(
                "legacy symmetric and role-aware streaming variables cannot be mixed"
            )
        # Validate both role contracts before either 14B role begins compute.
        resolved = {}
        for check_prefix in ("GENERATOR", "SCORE"):
            resolved[check_prefix] = _parse_tail_streaming_spec(
                count_name=f"DISTILLSCHEDULE_GAN_{check_prefix}_STREAMED_BLOCK_COUNT",
                indices_name=f"DISTILLSCHEDULE_GAN_{check_prefix}_STREAMED_BLOCK_INDICES",
            )
        if not _optimized_policy_authorized() and any(
            count != 40 for count, _indices in resolved.values()
        ):
            raise ValueError(
                "role-aware non-default streaming requires a capacity run or "
                "validated formal policy"
            )
        return _parse_tail_streaming_spec(
            count_name=role_count, indices_name=role_indices
        )

    if enabled == "0":
        return 40, tuple(range(40))
    return _parse_tail_streaming_spec(
        count_name="DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT",
        indices_name="DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES",
    )


def _generator_critic_phase_resident_enabled() -> bool:
    value = os.environ.get(
        "DISTILLSCHEDULE_GAN_GENERATOR_CRITIC_PHASE_RESIDENT", "0"
    ).strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "DISTILLSCHEDULE_GAN_GENERATOR_CRITIC_PHASE_RESIDENT must be 0 or 1"
        )
    if value == "1" and not _optimized_policy_authorized():
        raise ValueError(
            "Generator Critic-phase residency requires a capacity run or "
            "validated formal policy"
        )
    return value == "1"


def _score_critic_activation_offload_min_bytes() -> int:
    raw = os.environ.get(
        "DISTILLSCHEDULE_GAN_SCORE_CRITIC_ACTIVATION_OFFLOAD_MIN_BYTES", "0"
    ).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "DISTILLSCHEDULE_GAN_SCORE_CRITIC_ACTIVATION_OFFLOAD_MIN_BYTES "
            "must be a non-negative integer"
        ) from exc
    if value < 0:
        raise ValueError("Score Critic activation-offload threshold must be non-negative")
    if value and not _optimized_policy_authorized():
        raise ValueError(
            "Score Critic activation offload requires a capacity run or "
            "validated formal policy"
        )
    return value


def _optimized_policy_authorized() -> bool:
    capacity = os.environ.get("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH", "0").strip()
    if capacity not in {"0", "1"}:
        raise ValueError("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH must be 0 or 1")
    digest = os.environ.get(
        "DISTILLSCHEDULE_GAN_VALIDATED_POLICY_DIGEST", ""
    ).strip()
    if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "DISTILLSCHEDULE_GAN_VALIDATED_POLICY_DIGEST must be a lowercase SHA256"
        )
    if capacity == "1" and digest:
        raise ValueError(
            "capacity search and validated formal policy cannot be active together"
        )
    return capacity == "1" or bool(digest)


def _install_streamer(model: torch.nn.Module, *, role: str, device: torch.device, args: argparse.Namespace, rank: int):
    streamed_count, streamed_indices = _capacity_streaming_spec(role)
    if streamed_count == 0:
        return _ResidentFrozenBlocks()
    return install_frozen_block_streamer(
        model,
        logical_component=("generator" if role == "g" else "score"),
        device=device,
        window_size=1,
        streamed_block_count=streamed_count,
        streamed_block_indices=streamed_indices,
        host_policy=HostStatePolicy.HOST_SHARED_IMMUTABLE,
        optimizer_owned_parameter_ids={id(p) for p in model.parameters() if p.requires_grad},
        checkpoint_identity=f"{args.generator_root if role == 'g' else args.score_root}::bf16",
        event_path=args.output_dir / "block_streaming" / f"rank_{rank:03d}_{'generator' if role == 'g' else 'score'}.jsonl",
        mmap_cache_root=args.mmap_cache / ("generator" if role == "g" else "score"),
        detailed_events=bool(args.detailed_events),
    )


def _load_generator_ema(
    model: torch.nn.Module, checkpoint_dir: Path
) -> tuple[SparseShardEMA, tuple[torch.nn.Parameter, ...]]:
    """Load the official ``module.``-qualified sparse-EMA identity."""

    ema_view = torch.nn.Module()
    ema_view.add_module("module", model)
    ema = SparseShardEMA.load_pretrained(ema_view, checkpoint_dir)
    parameters = tuple(ema_view.parameters())
    model_parameters = tuple(model.parameters())
    if len(model_parameters) != len(parameters) or any(
        left is not right
        for left, right in zip(model_parameters, parameters, strict=True)
    ):
        raise AssertionError("EMA naming view changed Generator Parameter identity")
    return ema, parameters


def _initialize_engine(role: str, args: argparse.Namespace, rank: int, group: Any, singleton: Any, device: torch.device):
    import deepspeed
    ranks = tuple(range(8)) if role == "g" else tuple(range(8, 16))
    model, model_meta = build_helios_pool_model(
        role=("generator" if role == "g" else "critic"),
        model_root=(args.generator_root if role == "g" else args.score_root),
        subfolder="",
        enable_gan=(role == "s"),
    )
    component = "generator" if role == "g" else "critic"
    logical_spec = _load_logical_trainables(model, args.logical_package, component)
    # Load EMA against the pre-DeepSpeed logical module inventory.  ZeRO's
    # flattened parameter wrappers are an execution detail and must not become
    # part of the sparse-EMA checkpoint identity.
    if role == "g":
        # The immutable systems bootstrap was exported from the official
        # accelerator wrapper, whose logical parameter names are prefixed with
        # ``module.``.  Recreate only that naming view here; the wrapped
        # Parameters and their order are unchanged.
        ema, ema_parameters = _load_generator_ema(
            model, args.checkpoint / "model_ema"
        )
    else:
        ema = None
        ema_parameters = ()
    # Keep the exact pre-DeepSpeed parameter order used by the sparse-EMA
    # manifest.  DeepSpeed may wrap the module or alter enumeration details,
    # but it preserves these Parameter objects.  EMA is a logical training
    # state and must not key its identity to that runtime wrapper.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=(2e-6 if role == "g" else 4e-7),
        betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    mpu = StudentDPGroupAdapter(
        data_parallel_group=group,
        data_parallel_ranks=ranks,
        global_rank=rank,
        singleton_model_parallel_group=singleton,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=_ds_config(), mpu=mpu, dist_init_required=False,
    )
    resolved = _resolved(engine, group)
    role_rank = rank if role == "g" else rank - 8
    optimizer_restore = _restore_logical_zero1_optimizer(
        engine,
        args.logical_package,
        logical_spec,
        role_rank=role_rank,
    )
    engine.lr_scheduler.load_state_dict(logical_spec["scheduler_state"])
    engine.global_steps = int(logical_spec["optimizer_step"])
    # DeepSpeed moves the full module to the role GPU during initialization.
    # Install the immutable owner afterwards, matching the validated real-14B
    # pool substrate; installing it before DeepSpeed lets that move bypass the
    # streamer's materialization accounting and causes a false double release.
    streamer = _install_streamer(
        engine.module, role=role, device=device, args=args, rank=rank
    )
    activation = None
    if role == "g":
        dynamic_exclusions = (
            (streamer.dynamic_excluded_storage_owner,)
            if hasattr(streamer, "dynamic_excluded_storage_owner")
            else ()
        )
        activation = install_activation_offload_hooks(
            engine.module, logical_component="generator_selected",
            event_path=args.output_dir / "activation_offload" / f"rank_{rank:03d}_generator_selected.jsonl",
            pin_memory=True, min_offload_bytes=32 * 1024 * 1024,
            detailed_events=bool(args.detailed_events),
            dynamic_excluded_storage_providers=dynamic_exclusions,
        )
    model_meta["post_restore_trainable_dtype_inventory"] = sorted(
        {str(parameter.dtype) for parameter in engine.module.parameters() if parameter.requires_grad}
    )
    return engine, streamer, activation, ema, ema_parameters, {
        "model": model_meta,
        "resolved": resolved,
        "optimizer_restore": optimizer_restore,
        "logical_step": int(logical_spec["optimizer_step"]),
    }


def _generator_kwargs(cfg: Any, trace: OperationTrace, source: dict[str, torch.Tensor], step: int, negative: torch.Tensor, callback: Any) -> dict[str, Any]:
    t = cfg.training_config
    return dict(
        args=cfg, accelerator=_Accelerator(source["text_embedding"].device),
        real_fake_score_model=None, scheduler=_scheduler(source["text_embedding"].device),
        noise=trace.tensor(f"step/{step}/generator/entry_noise").to(source["text_embedding"].device, dtype=torch.bfloat16),
        prompt_embeds=source["text_embedding"], negative_prompt_embeds=negative,
        dmd_is_low_vram_mode=False, is_gan_low_vram_mode=True,
        fake_guidance_scale=float(t.fake_guidance_scale), real_guidance_scale=float(t.real_guidance_scale),
        gt_history_latents=source["primary_history"], gt_target_latents=source["primary_target"], gt_x0_latents=source["primary_x0"],
        is_use_gan=True, gan_prompt_embeds=source["text_embedding"], gan_g_weight=float(t.gan_g_weight),
        operation_trace=trace, trace_prefix=f"step/{step}/generator", global_step=step,
        remote_score_gradient_service=callback,
        **_common_call(cfg),
    )


def _score_official_kwargs(cfg: Any, trace: OperationTrace, source: dict[str, torch.Tensor], step: int, negative: torch.Tensor, device: torch.device) -> dict[str, Any]:
    t = cfg.training_config
    # Prepared-history stochastic values are replayed by the official function;
    # ts_schedule is disabled in the frozen configuration, so denoised bounds
    # are not inputs to timestep selection.
    from helios.utils.utils_base import apply_schedule_shift
    sigmas = torch.linspace(1.0, 1.0/1000, 1000, device=device, dtype=torch.float64)
    sigmas = apply_schedule_shift(sigmas, torch.empty((1,16,9,48,80)), base_seq_len=t.base_seq_len, max_seq_len=t.max_seq_len, base_shift=t.base_shift, max_shift=t.max_shift, time_shift_type=t.time_shift_type)
    from helios.utils.utils_helios_post import replay_operation_trace_prepared_history
    history, short, mid, long, indices, indices_short, indices_mid, indices_long = replay_operation_trace_prepared_history(
        trace, trace_prefix=f"step/{step}/generator", device=device, dtype=torch.bfloat16,
    )
    gt_all_data = (
        source["primary_x0"], indices, indices_short, indices_mid, indices_long,
        short, mid, long, history,
    )
    return dict(
        accelerator=_Accelerator(device), scheduler=_scheduler(device),
        prompt_embeds=source["text_embedding"], negative_prompt_embeds=negative,
        dmd_is_low_vram_mode=False, is_gan_low_vram_mode=True,
        fake_guidance_scale=float(t.fake_guidance_scale), real_guidance_scale=float(t.real_guidance_scale),
        is_enable_stage2=True, is_use_gt_history=True, gt_all_data=gt_all_data,
        is_use_gan=True, is_decouple_dmd=False,
        ts_schedule=False, ts_schedule_max=False, sigmas=sigmas, timesteps=sigmas*1000,
        gradient_mask=None, denoised_timestep_from=None, denoised_timestep_to=None,
        is_forcing_low_renoise=bool(t.generator_is_forcing_low_renoise),
        dynamic_alpha=float(t.generator_dynamic_alpha), dynamic_beta=float(t.generator_dynamic_beta),
        dynamic_sample_type=str(t.generator_dynamic_sample_type), global_step=step,
        dynamic_step=int(t.generator_dynamic_step), operation_trace=trace,
        trace_prefix=f"step/{step}/generator",
    )


def _critic_kwargs(cfg: Any, trace: OperationTrace, source: dict[str, torch.Tensor], step: int, x0: torch.Tensor, engine: Any, wave: int) -> dict[str, Any]:
    t = cfg.training_config
    return dict(
        args=cfg, critic_accelerator=_Accelerator(x0.device, engine),
        transformer=None, scheduler=_scheduler(x0.device),
        noise=trace.tensor(f"step/{step}/critic/entry_noise").to(x0.device, dtype=torch.bfloat16),
        prompt_embeds=source["text_embedding"],
        dmd_is_low_vram_mode=False, is_gan_low_vram_mode=True,
        gt_history_latents=source["secondary_history"], gt_target_latents=source["secondary_target"], gt_x0_latents=source["secondary_x0"],
        is_use_gan=True, is_separate_gan_grad=bool(t.is_separate_gan_grad),
        gan_base_critic_trainable_params=None, gan_extra_critic_trainable_params=None,
        gan_vae_latents=source["primary_target"], gan_prompt_embeds=source["text_embedding"],
        gan_d_weight=float(t.gan_d_weight), aprox_r1=bool(t.aprox_r1), aprox_r2=bool(t.aprox_r2),
        r1_weight=float(t.r1_weight), r2_weight=float(t.r2_weight), r1_sigma=float(t.r1_sigma), r2_sigma=float(t.r2_sigma),
        dynamic_alpha=float(t.critic_dynamic_alpha), dynamic_beta=float(t.critic_dynamic_beta),
        dynamic_sample_type=str(t.critic_dynamic_sample_type), global_step=step,
        dynamic_step=int(t.critic_dynamic_step), operation_trace=trace,
        trace_prefix=f"step/{step}/critic", precomputed_generated_image_or_video=x0,
        pipeline_micro_wave=wave, **_common_call(cfg),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--logical-package", type=Path, required=True)
    p.add_argument("--trace-bundle", type=Path, required=True)
    p.add_argument("--generator-root", type=Path, required=True)
    p.add_argument("--score-root", type=Path, required=True)
    p.add_argument("--negative-prompt", type=Path, required=True)
    p.add_argument("--mmap-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-nonce", type=int, default=811001)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--detailed-events", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("persistent GAN executor is frozen to world16")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    control, g_group, s_group, pair_group, singleton = _groups(rank)
    result: dict[str, Any] = {"rank": rank, "role": "generator" if rank < 8 else "score", "status": "RUNNING", "updates": {"generator": 0, "critic": 0, "ema": 0}}
    try:
        if args.validate_only:
            result.update(status="PASS", topology="8G+8Score", batch="DP8_MB1_GAS2_B16")
            _atomic_json(args.output_dir / f"rank_{rank:03d}.json", result)
            dist.barrier(group=control); return 0
        cfg = _load_official_config(args.config)
        role = "g" if rank < 8 else "s"; lane = rank if rank < 8 else rank - 8
        requested_streamed_count, requested_streamed_indices = (
            _capacity_streaming_spec(role)
        )
        requested_phase_resident = _generator_critic_phase_resident_enabled()
        score_critic_activation_min_bytes = (
            _score_critic_activation_offload_min_bytes()
        )
        engine, streamer, activation, ema, ema_parameters, init = _initialize_engine(role, args, rank, g_group if role == "g" else s_group, singleton, device)
        result["initialization"] = init
        negative = torch.load(args.negative_prompt, map_location="cpu", weights_only=True).to(device=device, dtype=torch.bfloat16)
        traces = [_trace(args.trace_bundle, lane), _trace(args.trace_bundle, lane + 8)]
        sources = [_source(t, device) for t in traces]
        from distillschedule.runtime.gan_pipeline_transport import (
            GanCriticRequestHeader, GanGradientP2PTransport,
            GanGradientReplyHeader, GanGradientRequestHeader,
        )
        from distillschedule.runtime.gan_pipeline_helios import (
            ScoreBoundaryResult, compute_official_score_boundary_grad,
        )
        from distillschedule.runtime.gan_timing_range import GanFull5TimingRange
        from distillschedule.runtime.nsys_range import GanFull5NsightRange
        from runtime.critic_pool_transport import tensor_identity
        from helios.utils.utils_helios_post import _critic_loss, _generator_loss
        transport = GanGradientP2PTransport(group=pair_group, device=device)
        exact_timing_enabled = os.environ.get(
            "DISTILLSCHEDULE_GAN_EXACT_TIMING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        exact_timing = (
            GanFull5TimingRange.from_env() if exact_timing_enabled else None
        )
        nsys_range = GanFull5NsightRange.from_env()
        dist.barrier(group=control)
        if nsys_range.enabled:
            # All roles enter the node-local capture before Generator wave0.
            # Starting only at Score's first model call would deadlock because
            # Score must first receive x_fake from Generator.
            nsys_range.before_model_compute(1000)
            dist.barrier(group=control)
        result["timing_start_unix_ns"] = time.time_ns()
        records = []
        if role == "g":
            for wave, (trace, source) in enumerate(zip(traces, sources, strict=True)):
                if wave == 0 and exact_timing is not None:
                    exact_timing.before_model_compute(1000)
                def callback(*, generated_latent: torch.Tensor, **_ignored: Any):
                    checksum = tensor_identity(generated_latent.detach(), ChecksumKind.GPU_FINGERPRINT_V1)
                    header = GanGradientRequestHeader(args.run_nonce, 1000, wave, rank, rank+8, 200, 1000, bytes.fromhex(trace.trace_hash), checksum)
                    transport.wait_and_reap(transport.send_request(header, generated_latent, destination_global_rank=rank+8))
                    _, grad = transport.receive_reply(source_global_rank=rank+8, request=header)
                    return ScoreBoundaryResult(grad, 0.0, 0.0, 0.0, 0.0, {"remote": True})
                loss, logs = _generator_loss(transformer=engine.module, **_generator_kwargs(cfg, trace, source, 1000, negative, callback))
                engine.set_gradient_accumulation_boundary(is_boundary=(wave == 1)); engine.backward(loss)
                if wave == 1: engine.step(); result["updates"]["generator"] += 1
                records.append({"phase": "generator", "step": 1000, "wave": wave, "trace_hash": trace.trace_hash, "loss": float(loss.detach())})
        else:
            for wave, (trace, source) in enumerate(zip(traces, sources, strict=True)):
                header, xfake = transport.receive_request(source_global_rank=lane)
                if header.logical_step != 1000 or header.micro_wave != wave or header.trace_sha256 != bytes.fromhex(trace.trace_hash):
                    raise AssertionError("Generator/Score gradient request identity mismatch")
                if wave == 0 and exact_timing is not None:
                    exact_timing.before_model_compute(1000)
                boundary = compute_official_score_boundary_grad(score_model=engine.module, generated_latent=xfake, gan_g_weight=float(cfg.training_config.gan_g_weight), official_kwargs=_score_official_kwargs(cfg, trace, source, 1000, negative, device))
                grad_sum = tensor_identity(boundary.gradient, ChecksumKind.GPU_FINGERPRINT_V1)
                reply = GanGradientReplyHeader(args.run_nonce, 1000, wave, lane, rank, 200, 1000, 1000, bytes.fromhex(trace.trace_hash), grad_sum)
                transport.wait_and_reap(transport.send_reply(reply, boundary.gradient, destination_global_rank=lane))
                records.append({"phase": "score_gradient", "step": 1000, "wave": wave, "trace_hash": trace.trace_hash, "total_loss": boundary.total_loss, "dmd_loss": boundary.dmd_loss, "gan_g_loss": boundary.gan_g_loss_raw})
        dist.barrier(group=control)
        # Score gradient service has now finished. Install the audited saved-
        # tensor primitive only for the Critic phase so DMD/GAN-G input-gradient
        # execution stays on its frozen path. The threshold selects large packed
        # fake/real/R1 checkpoint boundaries rather than small temporaries.
        if role == "s" and score_critic_activation_min_bytes:
            dynamic_exclusions = (
                (streamer.dynamic_excluded_storage_owner,)
                if hasattr(streamer, "dynamic_excluded_storage_owner")
                else ()
            )
            activation = install_activation_offload_hooks(
                engine.module,
                logical_component="score_critic_train",
                event_path=(
                    args.output_dir
                    / "activation_offload"
                    / f"rank_{rank:03d}_score_critic.jsonl"
                ),
                pin_memory=True,
                min_offload_bytes=score_critic_activation_min_bytes,
                detailed_events=bool(args.detailed_events),
                dynamic_excluded_storage_providers=dynamic_exclusions,
            )
        # The updated G201 now owns all five Critic rollouts. Pair-local sends
        # are one-way and may naturally overlap Score work; version commits stay
        # strictly serialized by the Score DP8 collectives. In the optimized
        # capacity candidate, the Generator's already-managed immutable blocks
        # are materialized exactly once for this entire no-grad phase. Step1000
        # differentiable execution above retains its original streaming policy.
        phase_resident = (
            role == "g" and requested_phase_resident
        )
        critic_phase_context = (
            contextlib.ExitStack() if phase_resident else contextlib.nullcontext()
        )
        with critic_phase_context as stack:
            if phase_resident:
                assert isinstance(stack, contextlib.ExitStack)
                stack.enter_context(torch.no_grad())
                stack.enter_context(
                    streamer.phase_resident_no_grad(
                        phase="gan_full5_critic_rollouts_1000_1004"
                    )
                )
            for step in range(1000, 1005):
                if role == "g":
                    pending = []
                    for wave, (trace, source) in enumerate(zip(traces, sources, strict=True)):
                        loaded = _loaded_critic(trace, source, step)
                        x0, details = run_generator_trace_rollout(model=engine.module, loaded=loaded, device=device, detailed_profile=False, record_output_sha256=False, empty_cache_before_rollout=False)
                        digest = tensor_identity(x0, ChecksumKind.GPU_FINGERPRINT_V1)
                        header = GanCriticRequestHeader(args.run_nonce, step, wave, rank, rank+8, 201, 1000+(step-1000), 1001+(step-1000), bytes.fromhex(trace.trace_hash), digest)
                        pending.append(transport.send_critic_request(header, x0, destination_global_rank=rank+8))
                        records.append({"phase": "critic_rollout", "step": step, "wave": wave, "trace_hash": trace.trace_hash, "call_count": details["call_count"]})
                    for item in pending: transport.wait_and_reap(item)
                else:
                    for wave, (trace, source) in enumerate(zip(traces, sources, strict=True)):
                        header, x0 = transport.receive_critic_request(source_global_rank=lane)
                        if (header.logical_step, header.micro_wave, header.trace_sha256) != (step, wave, bytes.fromhex(trace.trace_hash)):
                            raise AssertionError("Critic request identity mismatch")
                        loss, logs = _critic_loss(fake_score_model=engine.module, **_critic_kwargs(cfg, trace, source, step, x0, engine, wave))
                        records.append({"phase": "critic_train", "step": step, "wave": wave, "trace_hash": trace.trace_hash, "loss": float(loss.detach()), "committed": bool(logs.get("critic_optimizer_committed"))})
                    result["updates"]["critic"] += 1
                if step == 1000 and role == "g":
                    assert ema is not None; ema.step(ema_parameters); result["updates"]["ema"] += 1
        transport.flush(); transport.assert_flushed()
        dist.barrier(group=control)
        # This is the existing mandatory Full5 flush/version boundary, not a
        # timing-only barrier.  Recording after it gives Generator ranks proof
        # that the remote step1004 Critic commit has completed without adding
        # an ACK or changing the pipeline schedule.
        if exact_timing is not None:
            exact_timing.after_critic_commit(1004)
            exact_timing.assert_complete()
        if nsys_range.enabled:
            nsys_range.after_critic_commit(1004)
            nsys_range.assert_complete()
        result["timing_stop_unix_ns"] = time.time_ns()
        expected = {"generator": 1, "critic": 0, "ema": 1} if role == "g" else {"generator": 0, "critic": 5, "ema": 0}
        if result["updates"] != expected:
            raise AssertionError(f"update contract drift: {result['updates']} != {expected}")
        activation_stats = (
            dict(activation.offloader.stats) if activation is not None else None
        )
        if activation is not None:
            activation.remove()
        result.update(status="PASS", records=records, streamer_stats=dict(streamer.stats), activation_offload_stats=activation_stats, streaming_policy={"streamed_block_count": requested_streamed_count, "streamed_block_indices": list(requested_streamed_indices), "generator_critic_phase_resident": bool(phase_resident), "score_critic_activation_offload_min_bytes": score_critic_activation_min_bytes}, max_allocated_bytes=int(torch.cuda.max_memory_allocated(device)), max_reserved_bytes=int(torch.cuda.max_memory_reserved(device)), ema_optimization_step=(ema.optimization_step if ema else None))
        streamer.remove()
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        _atomic_json(args.output_dir / f"rank_{rank:03d}.json", result)
        raise
    _atomic_json(args.output_dir / f"rank_{rank:03d}.json", result)
    dist.destroy_process_group(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
