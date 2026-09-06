"""Shared-backbone MiniMax-H3 model adapter for LightX2V-Train."""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.utils import convert_state_dict_to_diffusers
from loguru import logger
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from lightx2v_train.model_zoo.minimax_h3_t2av import MiniMaxH3T2AVModel
from lightx2v_train.model_zoo.native.minimax_h3 import load_minimax_h3_transformer
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import is_fsdp2_module
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .adaln_cache import (
    AdaLNCacheController,
    adaln_bank,
    adaln_controller,
    install_adaln_cache,
)

STUDENT_ADAPTER = "student"
FAKE_ADAPTER = "fake"
BASE_ROLE = "teacher"


class _H3CheckpointSegment(torch.nn.Module):
    """A contiguous block group used by the bounded SAC candidate.

    The modules inside the group remain individually ``fully_shard``-ed.  Only
    the activation-checkpoint boundary changes: the upstream H3 forward sees a
    segment as one callable and therefore saves one input boundary instead of
    one boundary per block.  This is deliberately a small wrapper rather than
    a replacement transformer implementation, so the block math and call
    sequence remain upstream-defined.
    """

    def __init__(self, blocks: list[torch.nn.Module], start: int, controller):
        super().__init__()
        if not blocks:
            raise ValueError("An H3 activation-checkpoint segment cannot be empty")
        self.blocks = torch.nn.ModuleList(blocks)
        self.start = int(start)
        object.__setattr__(self, "_adaln_controller", controller)
        self._replay_scope = (None, False)

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb):
        controller = self._adaln_controller
        active = controller.current_scope()
        if active.key is None:
            active = controller.last_scope()
        if active.key is not None:
            # The original checkpoint forward has the exact request scope.
            # Save it on this segment so replay can restore it even though
            # contextvars are not propagated by non-reentrant checkpointing.
            self._replay_scope = (active.key, active.persistent)
        key, persistent = self._replay_scope
        scope = controller.scope(key, persistent=persistent) if key is not None else contextlib.nullcontext()
        with scope:
            for block in self.blocks:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
            return hidden_states


def _configure_local_flash_attn3() -> None:
    """Bind the pinned local FlashAttention-3 build without Hub access.

    The pinned Diffusers ``_flash_3_hub`` backend normally resolves version 1
    through the Hugging Face Hub on every fresh process.  The BAAI nodes are
    intentionally offline, but the exact kernel snapshot is already present
    in the shared benchmark assets.  Binding the same module through
    ``get_local_kernel`` changes only how the immutable kernel is located; it
    does not change the attention backend or workload semantics.
    """

    local_path = os.environ.get("H3_FLASH_ATTN3_LOCAL_PATH")
    if not local_path:
        return
    from kernels.utils import get_local_kernel
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _HUB_KERNELS_REGISTRY,
    )

    config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    if config.kernel_fn is not None:
        return
    module = get_local_kernel(Path(local_path), "flash_attn3")
    config.kernel_fn = module.flash_attn_func
    config.wrapped_forward_fn = module.flash_attn_interface._flash_attn_forward
    config.wrapped_backward_fn = module.flash_attn_interface._flash_attn_backward
    logger.info("[h3-a100] bound local FlashAttention-3 kernel: {}", local_path)


def _ensure_fsdp_root_initialized(module: torch.nn.Module) -> None:
    """Initialize the FSDP2 root before touching a sharded child directly.

    AdaLN precomputation intentionally calls the sharded ``time_embedder``
    before the first full transformer forward.  FSDP2 normally discovers the
    root from that first forward; entering the child first makes the child look
    like a root and the subsequent real forward fails.  Calling the same
    private lazy-init routine on the already-sharded root is metadata/stream
    setup only: it does not run a collective or move a tensor, and preserves
    the normal first-forward ownership semantics.
    """

    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )
    except ImportError:
        return
    state = _get_module_fsdp_state(module)
    if state is None or state._is_root is not None:
        return
    state._lazy_init()


def _lora_config(mapping: Mapping[str, Any]) -> LoraConfig:
    rank = int(mapping["rank"])
    alpha = int(mapping.get("alpha", rank))
    targets = mapping.get("target_modules")
    if not targets:
        raise ValueError("H3 shared-backbone LoRA requires target_modules")
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(targets),
    )


def _adapter_parameter(name: str, adapter_name: str) -> bool:
    return "lora_" in name and (
        f".{adapter_name}." in name
        or name.endswith(f".{adapter_name}")
        or f"_{adapter_name}." in name
    )


@MODEL_REGISTER("minimax_h3_t2av_a100")
class MiniMaxH3A100Model(MiniMaxH3T2AVModel):
    """One physical H3 backbone with student/fake LoRA role adapters."""

    def __init__(self, config):
        super().__init__(config)
        self._role: contextvars.ContextVar[str] = contextvars.ContextVar(
            "h3_a100_role", default=STUDENT_ADAPTER
        )
        # PEFT adapter changes and ``Module.train()`` both recurse through the
        # full 50-block H3 tree.  The workload enters many logically nested
        # scopes whose requested state is already active, so keep an explicit
        # physical-state cache and make those transitions idempotent.  This is
        # host-side bookkeeping only; every real role/mode change still calls
        # the pinned Diffusers/PEFT APIs.
        self._physical_role: str | None = None
        self._physical_training_mode: bool | None = None
        self._runtime_state_stats = {
            "role_transitions": 0,
            "role_noops": 0,
            "mode_transitions": 0,
            "mode_noops": 0,
        }
        self._shared_backbone_ready = False

    def load_components(self, transformer_only=False, reference_model=None):
        """Load full H3 on CPU; FSDP2/HSDP moves only local shards to CUDA.

        Upstream LightX2V calls ``transformer.to(cuda)`` before sharding. A
        66-GB BF16 H3 checkpoint cannot survive that transient peak on an
        A100-40GB. PyTorch FSDP2 supports CPU initialization and moves the
        sharded DTensor parameters to the device mesh during ``fully_shard``.
        """

        del transformer_only, reference_model
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.transformer_param_dtype = get_running_dtype(
            config.get("transformer_param_dtype", "bf16")
        )
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.video_latent_channels = int(config.get("video_latent_channels", 24))
        self.audio_latent_channels = int(config.get("audio_latent_channels", 32))
        self.vae_spatial_scale_factor = int(config.get("vae_spatial_scale_factor", 16))
        self.use_autocast = bool(config.get("use_autocast", False))
        _configure_local_flash_attn3()
        self.transformer = load_minimax_h3_transformer(
            self.pretrained_model_path,
            torch_dtype=self.transformer_param_dtype,
            local_files_only=bool(config.get("local_files_only", True)),
            attention_backend=config.get("attention_backend"),
        )
        if not bool(config.get("fsdp_load_on_cpu", True)):
            self.transformer.to(self.device)
        else:
            logger.info(
                "[h3-a100] full H3 loaded on CPU; FSDP2/HSDP will move local shards to {}",
                self.device,
            )

    def prepare_shared_backbone(
        self,
        *,
        student_lora: Mapping[str, Any],
        fake_lora: Mapping[str, Any],
        cache_enabled: bool = True,
        max_dynamic_cache_keys: int = 2,
    ) -> None:
        if self._shared_backbone_ready:
            return
        transformer = self.denoiser_module()
        transformer.requires_grad_(False)

        # Extract AdaLN before PEFT/FSDP so each giant projection can be sharded
        # independently and bypassed after its modulation is cached.
        install_adaln_cache(
            transformer,
            enabled=cache_enabled,
            max_dynamic_keys=max_dynamic_cache_keys,
        )
        self._add_named_adapter(STUDENT_ADAPTER, _lora_config(student_lora))
        self._add_named_adapter(FAKE_ADAPTER, _lora_config(fake_lora))

        counts = self._mark_all_adapters_trainable()
        if not all(counts.values()):
            raise RuntimeError(f"Failed to discover named H3 LoRA parameters: {counts}")

        self._activate_role(STUDENT_ADAPTER)
        self._shared_backbone_ready = True
        logger.info(
            "[h3-a100] shared backbone ready student_lora_params={} fake_lora_params={}",
            counts[STUDENT_ADAPTER],
            counts[FAKE_ADAPTER],
        )

    def configure_activation_checkpoint_segments(self, segment_size: int) -> None:
        """Install the bounded SAC grouping before FSDP2 wrapping.

        ``segment_size=1`` is the pinned upstream behavior.  Larger values
        only alter checkpoint boundaries; every contained transformer block is
        still passed to FSDP2 separately by :meth:`fsdp2_shard_plan`.
        """

        segment_size = int(segment_size)
        if segment_size < 1:
            raise ValueError(f"activation checkpoint segment_size must be >= 1, got {segment_size}")
        transformer = self.denoiser_module()
        if self.is_fsdp2_wrapped():
            raise RuntimeError("Activation-checkpoint segmentation must be configured before FSDP2 wrapping")
        blocks = list(transformer.transformer_blocks)
        if not blocks:
            raise RuntimeError("MiniMax-H3 transformer has no transformer blocks")
        if segment_size == 1:
            logger.info("[h3-a100] activation checkpoint segment_size=1 (upstream per-block policy)")
            self._activation_checkpoint_segment_size = 1
            return
        if any(isinstance(block, _H3CheckpointSegment) for block in blocks):
            raise RuntimeError("Activation checkpoint segmentation was configured more than once")
        segments = [
            _H3CheckpointSegment(blocks[start : start + segment_size], start, self.adaln_cache())
            for start in range(0, len(blocks), segment_size)
        ]
        transformer.transformer_blocks = torch.nn.ModuleList(segments)
        self._activation_checkpoint_segment_size = segment_size
        logger.info(
            "[h3-a100] SAC activation checkpoint segments enabled segment_size={} blocks={} segments={}",
            segment_size,
            len(blocks),
            len(segments),
        )

    def _add_named_adapter(self, adapter_name: str, config: LoraConfig) -> None:
        transformer = self.denoiser_module()
        if not hasattr(transformer, "add_adapter"):
            raise RuntimeError(
                "Installed Diffusers H3 model has no add_adapter API. Use the "
                "LightX2V-pinned Diffusers revision with PeftAdapterMixin."
            )
        transformer.add_adapter(config, adapter_name=adapter_name)

    def role_parameters(self, role: str) -> list[torch.nn.Parameter]:
        if role not in {STUDENT_ADAPTER, FAKE_ADAPTER}:
            raise ValueError(f"Role {role!r} has no trainable parameters")
        parameters = [
            parameter
            for name, parameter in self.denoiser_module().named_parameters()
            if _adapter_parameter(name, role)
        ]
        if not parameters:
            raise RuntimeError(f"No trainable parameters found for H3 role {role!r}")
        return parameters

    def _activate_role(self, role: str) -> None:
        if role == self._physical_role:
            self._runtime_state_stats["role_noops"] += 1
            return
        transformer = self.denoiser_module()
        if role == BASE_ROLE:
            if not hasattr(transformer, "disable_adapters"):
                raise RuntimeError("Diffusers H3 cannot disable adapters for the teacher role")
            transformer.disable_adapters()
            self._physical_role = role
            self._runtime_state_stats["role_transitions"] += 1
            return
        if role not in {STUDENT_ADAPTER, FAKE_ADAPTER}:
            raise ValueError(f"Unknown H3 shared-backbone role: {role!r}")
        if hasattr(transformer, "enable_adapters"):
            transformer.enable_adapters()
        if not hasattr(transformer, "set_adapter"):
            raise RuntimeError("Diffusers H3 cannot switch named PEFT adapters")
        transformer.set_adapter(role)
        # PEFT's adapter switch may mark inactive adapters frozen. FSDP2 is
        # configured once, so keep both optimizer-owned adapter sets trainable
        # and let the active adapter determine which one participates in the
        # forward graph.
        self._mark_all_adapters_trainable()
        self._physical_role = role
        self._runtime_state_stats["role_transitions"] += 1

    def set_transformer_training(self, training: bool) -> None:
        """Apply a full-module train/eval transition only when it changes.

        MiniMax-H3 has no stochastic dropout under the controlled workload,
        but its train/eval state remains part of the frozen contract.  This
        helper preserves that state exactly while avoiding repeated recursive
        walks requested by adjacent rollout or Fake-update calls.
        """

        training = bool(training)
        if training == self._physical_training_mode:
            self._runtime_state_stats["mode_noops"] += 1
            return
        self.denoiser_module().train(training)
        self._physical_training_mode = training
        self._runtime_state_stats["mode_transitions"] += 1

    def runtime_state_stats(self) -> dict[str, int | str | bool | None]:
        return {
            **self._runtime_state_stats,
            "physical_role": self._physical_role,
            "physical_training_mode": self._physical_training_mode,
        }

    def _mark_all_adapters_trainable(self) -> dict[str, int]:
        transformer = self.denoiser_module()
        transformer.requires_grad_(False)
        counts = {STUDENT_ADAPTER: 0, FAKE_ADAPTER: 0}
        for name, parameter in transformer.named_parameters():
            for adapter_name in counts:
                if _adapter_parameter(name, adapter_name):
                    parameter.requires_grad_(True)
                    counts[adapter_name] += parameter.numel()
                    break
        return counts

    @contextlib.contextmanager
    def role_scope(self, role: str) -> Iterator[None]:
        previous = self._role.get()
        token = self._role.set(role)
        self._activate_role(role)
        try:
            yield
        finally:
            self._role.reset(token)
            self._activate_role(previous)

    def adaln_cache(self) -> AdaLNCacheController:
        controller = adaln_controller(self.denoiser_module())
        if controller is None:
            raise RuntimeError("H3 AdaLN cache was not installed before FSDP wrapping")
        return controller

    @contextlib.contextmanager
    def adaln_scope(self, key: Any, *, persistent: bool = False) -> Iterator[None]:
        with self.adaln_cache().scope(key, persistent=persistent):
            yield

    @torch.no_grad()
    def precompute_adaln(self, key: Any, timesteps: torch.Tensor, *, persistent: bool) -> None:
        """Materialize all block modulations for one exact timestep table."""

        transformer = self.denoiser_module()
        _ensure_fsdp_root_initialized(transformer)
        bank = adaln_bank(transformer)
        controller = self.adaln_cache()
        if bank is None:
            raise RuntimeError("H3 AdaLN projection bank is missing")
        if controller.has_complete_key(key, len(bank.projections)):
            return

        with controller.scope(key, persistent=persistent):
            temb = transformer.time_proj(timesteps)
            temb = transformer.time_embedder(
                temb.to(get_parameter_dtype(transformer.time_embedder))
            )
            for block_index, projection in enumerate(bank.projections):
                controller.get_or_compute(block_index, temb, projection)

    def drop_adaln_key(self, key: Any) -> None:
        self.adaln_cache().drop(key)

    def save_role_lora(self, save_dir: str, role: str) -> None:
        self.save_lora_weights(save_dir, adapter_name=role)

    def save_all_role_loras(self, student_dir: str, fake_dir: str) -> None:
        """Gather only trainable adapter tensors once and write both roles."""

        denoiser = self.denoiser_module()
        if is_fsdp2_module(denoiser):
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
                strict=False,
            )
            state_dict, _ = get_state_dict(denoiser, (), options=options)
        elif is_main_process():
            state_dict = denoiser.state_dict()
        else:
            state_dict = {}
        if not is_main_process():
            return

        for role, output_dir in (
            (STUDENT_ADAPTER, student_dir),
            (FAKE_ADAPTER, fake_dir),
        ):
            os.makedirs(output_dir, exist_ok=True)
            role_state = get_peft_model_state_dict(
                denoiser, state_dict=state_dict, adapter_name=role
            )
            role_state = convert_state_dict_to_diffusers(role_state)
            role_state = {key: value.detach().cpu().contiguous() for key, value in role_state.items()}
            save_file(
                role_state,
                os.path.join(output_dir, "pytorch_lora_weights.safetensors"),
            )

    def load_role_lora(self, load_dir: str, role: str) -> None:
        self.load_lora_weights_for_resume(load_dir, adapter_name=role)

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config.get("reshard_after_forward", {})
        transformer = self.denoiser_module()
        bank = adaln_bank(transformer)
        if bank is None:
            return super().fsdp2_shard_plan(fsdp_config)

        token_refiner_blocks = list(transformer.token_refiner.refiner_blocks)
        transformer_blocks = []
        for block_or_segment in transformer.transformer_blocks:
            if isinstance(block_or_segment, _H3CheckpointSegment):
                transformer_blocks.extend(list(block_or_segment.blocks))
            else:
                transformer_blocks.append(block_or_segment)
        return [
            # precompute_adaln calls the time MLP directly after FSDP wrapping.
            {"module": transformer.time_embedder, "reshard_after_forward": True},
            # Each ~260M AdaLN projection is its own FSDP unit. Fixed rollout
            # keys touch it once; one dynamic key is shared by fake + teacher.
            {"modules": list(bank.projections), "reshard_after_forward": True},
            {
                "modules": token_refiner_blocks + transformer_blocks,
                "reshard_after_forward": reshard.get("block_reshard", True),
            },
            {
                "module": transformer,
                "reshard_after_forward": reshard.get("root_reshard", False),
            },
        ]

    def log_adaln_stats(self, prefix: str = "[h3-a100]") -> None:
        stats = self.adaln_cache().stats()
        logger.info(
            "{} AdaLN cache hits={} misses={} stores={} evictions={} "
            "persistent_keys={} dynamic_keys={} memory_mib={:.2f}",
            prefix,
            stats.hits,
            stats.misses,
            stats.stores,
            stats.evictions,
            stats.persistent_keys,
            stats.dynamic_keys,
            stats.bytes / 1024**2,
        )
