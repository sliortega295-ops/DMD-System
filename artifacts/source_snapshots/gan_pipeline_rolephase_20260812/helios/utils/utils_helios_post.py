import contextlib
import math
import os
import random
import time
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from einops import rearrange

from diffusers.training_utils import free_memory
from diffusers.utils.torch_utils import is_compiled_module
from runtime.official_baseline import swap_wave_ranges

from .utils_base import apply_schedule_shift
from .utils_helios_base import (
    add_saturation_to_history_latents,
    corrupt_history_latents,
    prepare_stage1_clean_input_from_latents,
)


logger = get_logger(__name__)


def replay_operation_trace_prepared_history(
    operation_trace,
    *,
    trace_prefix: str,
    device,
    dtype: torch.dtype,
):
    """Replay the complete processed history contract at its consumption site.

    History values alone are insufficient for a schedule-invariant workload:
    the four index tensors select the packed token/history layout consumed by
    every DiT/Score call.  Correctness runs therefore look up all eight values
    from the immutable operation trace.  With trace-access receipts disabled
    this adds no instrumentation and retains the same tensor math.
    """

    float_roles = (
        "continuous_history",
        "history_short",
        "history_mid",
        "history_long",
    )
    index_roles = (
        "indices_hidden_states",
        "indices_history_short",
        "indices_history_mid",
        "indices_history_long",
    )
    floats = tuple(
        operation_trace.tensor(f"{trace_prefix}/{role}").to(device, dtype=dtype)
        for role in float_roles
    )
    indices = tuple(
        operation_trace.tensor(f"{trace_prefix}/{role}").to(device)
        for role in index_roles
    )
    return (*floats, *indices)


@contextlib.contextmanager
def _helios_stage_range(global_step, role, op, **metadata):
    """Emit one parseable semantic NVTX range when Nsight annotation is on.

    The helper is intentionally inert in ordinary training.  Labels carry the
    global step and distributed rank so a multi-process Nsight report can be
    reconstructed without guessing process-to-rank ownership.
    """

    enabled = (
        os.environ.get("DISTILLSCHEDULE_NSYS_ANNOTATE", "0") == "1"
        and torch.cuda.is_available()
    )
    pushed = False
    if enabled:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        fields = {
            "step": int(global_step),
            "rank": rank,
            "role": str(role),
            "op": str(op),
            **{key: value for key, value in metadata.items() if value is not None},
        }
        label = "helios_stage|" + "|".join(f"{key}={value}" for key, value in fields.items())
        torch.cuda.nvtx.range_push(label)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


# ======================================== ODE Loss ========================================


def _ode_regression_loss(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    weight_dtype,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For Stage 2
    stage2_num_stages: int = 3,
    # For ODE Main
    last_step_only: bool = False,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    is_backward_grad: bool = False,
    ode_regression_weight: float = 0.25,
    ode_latents: torch.Tensor = None,
    ode_prompt_embeds: torch.Tensor = None,
    ode_num_latent_sections_min: int = 3,
    ode_num_latent_sections_max: int = 3,
    # For Dynamic Num Sections
    ode_dynamic_alpha: float = 1.5,
    ode_dynamic_beta: float = 4.0,
    ode_dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    ode_dynamic_step: int = 1000,
):
    _, num_channels_latents, latent_window_size, height, width = noise.shape
    batch_size, _, _, _, _ = ode_latents[0][0]["latents"][0].shape

    history_sizes = sorted(history_sizes, reverse=True)  # From large to small
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    history_latents = torch.zeros(
        batch_size,
        num_channels_latents,
        sum(history_sizes),
        height,
        width,
        device=accelerator.device,
        dtype=torch.float32,
    )
    max_history_frames = sum(history_sizes) + 1

    ode_stage2_num_stages = len(ode_latents[0])
    assert ode_stage2_num_stages == stage2_num_stages

    total_ode_num_latent_sections = len(ode_latents)
    assert ode_num_latent_sections_min <= ode_num_latent_sections_max
    ode_num_latent_sections = sample_dynamic_dmd_num_latent_sections(
        min_sections=ode_num_latent_sections_min,
        max_sections=ode_num_latent_sections_max,
        dmd_dynamic_alpha=ode_dynamic_alpha,
        dmd_dynamic_beta=ode_dynamic_beta,
        dmd_dynamic_sample_type=ode_dynamic_sample_type,
        global_step=global_step,
        dmd_dynamic_step=ode_dynamic_step,
        device=accelerator.device,
    )

    # Step 1: Denoising loop
    ode_loss_list = []
    image_latents = None
    total_generated_latent_frames = 0
    selected_sections = sorted(random.sample(range(total_ode_num_latent_sections), ode_num_latent_sections))
    for k in range(total_ode_num_latent_sections):
        should_compute_grad = k in selected_sections
        is_first_section = k == 0
        if is_keep_x0:
            if is_first_section:
                history_sizes_first_section = [1] + history_sizes.copy()
                history_latents_first_section = torch.zeros(
                    batch_size,
                    num_channels_latents,
                    sum(history_sizes_first_section),
                    height,
                    width,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_1x,
                    indices_hidden_states,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                    history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                        history_sizes_first_section, dim=2
                    )
                )
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                history_latents_first_section = None

                del history_latents_first_section, indices
            else:
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_1x,
                    indices_hidden_states,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix = image_latents
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)

                del indices
        else:
            raise NotImplementedError

        if should_compute_grad:
            for i_s in range(stage2_num_stages):
                exit_flag = generate_and_sync_flag(
                    accelerator, ode_latents[k][i_s]["timesteps"].shape[0], last_step_only, is_sync=False
                )
                noisy_model_input = ode_latents[k][i_s]["latents"][exit_flag].to(
                    accelerator.device, dtype=weight_dtype
                )
                gt_x0 = ode_latents[k][i_s]["latents"][-1].to(accelerator.device, dtype=weight_dtype)
                timestep = ode_latents[k][i_s]["timesteps"][exit_flag].unsqueeze(0).to(accelerator.device)

                timesteps_per_stage = scheduler.timesteps_per_stage[i_s]
                sigmas_per_stage = scheduler.sigmas_per_stage[i_s]
                if use_dynamic_shifting:
                    temp_sigmas_per_stage = apply_schedule_shift(
                        sigmas_per_stage,
                        noisy_model_input,
                        base_seq_len=args.training_config.base_seq_len,
                        max_seq_len=args.training_config.max_seq_len,
                        base_shift=args.training_config.base_shift,
                        max_shift=args.training_config.max_shift,
                        time_shift_type=time_shift_type,
                    )
                    temp_timesteps_per_stage = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas_per_stage * (
                        scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                    )
                    sigmas_per_stage = temp_sigmas_per_stage
                    timesteps_per_stage = temp_timesteps_per_stage

                    del temp_sigmas_per_stage, temp_timesteps_per_stage

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timestep,
                    encoder_hidden_states=ode_prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(ode_prompt_embeds.dtype),
                    latents_history_mid=latents_history_mid.to(ode_prompt_embeds.dtype),
                    latents_history_long=latents_history_long.to(ode_prompt_embeds.dtype),
                    return_dict=False,
                )[0]
                pred_x0 = convert_flow_pred_to_x0(
                    flow_pred=model_pred,
                    xt=noisy_model_input,
                    timestep=timestep,
                    sigmas=sigmas_per_stage,
                    timesteps=timesteps_per_stage,
                )

                temp_mse_loss = 0.5 * F.mse_loss(pred_x0.float(), gt_x0.float(), reduction="mean")
                ode_loss_list.append(temp_mse_loss)

                del noisy_model_input, timestep, model_pred, pred_x0, temp_mse_loss
        else:
            gt_x0 = ode_latents[k][-1]["latents"][-1].to(accelerator.device, dtype=weight_dtype)

        if is_first_section and is_keep_x0:
            image_latents = gt_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, gt_x0], dim=2)
        history_latents = history_latents[:, :, -max_history_frames:, :, :].contiguous()

        del gt_x0
        del latents_prefix, latents_history_long, latents_history_mid, latents_history_1x, latents_history_short
        del indices_prefix, indices_latents_history_long, indices_latents_history_mid
        del indices_latents_history_1x, indices_hidden_states, indices_latents_history_short
        free_memory()

    ode_loss = torch.stack(ode_loss_list).mean() * ode_regression_weight

    del ode_loss_list
    free_memory()

    assert ode_loss.requires_grad, f"ODE loss should have gradient! Got {ode_loss.requires_grad}"
    assert ode_loss.grad_fn is not None, "ODE loss should have grad_fn!"

    logs = {
        "ode_loss": ode_loss.detach().item(),
        # "lr": lr_scheduler.get_last_lr()[0],
    }

    if is_backward_grad:
        accelerator.backward(ode_loss)

        # Check if the gradient of each model parameter contains NaN
        for name, param in transformer.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"Gradient for {name} contains NaN!")

        grad_norm = None
        if accelerator.sync_gradients:
            params_to_clip = transformer.parameters()
            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training_config.max_grad_norm)

        if grad_norm is not None:
            logs["ode_grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

        ode_loss = None
        grad_norm = None
        del ode_loss
        del grad_norm

        return logs["ode_loss"], logs
    else:
        return ode_loss, logs


# ======================================== VRAM management ========================================


class OptimizedLowVRAMManager:
    def __init__(self, phase_tracer=None, trim_host_fn=None, swap_wave_size=0):
        self.pinned_models = set()
        self.grad_cache = {}
        self.phase_tracer = phase_tracer
        self.trim_host_fn = trim_host_fn
        self.model_labels = {}
        self.block_streamers = {}
        self.immutable_model_backings = {}
        self.engine_state_controllers = {}
        self.move_sequence = 0
        self.swap_wave_size = int(swap_wave_size)
        if self.swap_wave_size < 0:
            raise ValueError("swap_wave_size must be non-negative")

    @property
    def wave_swap_enabled(self):
        return self.swap_wave_size > 0

    def register_model(self, model, label):
        """Associate both an engine and its wrapped module with one logical name."""
        self.model_labels[id(model)] = label
        if hasattr(model, "module"):
            self.model_labels[id(model.module)] = label

    def register_engine_state_controller(self, model, controller):
        """Bind typed optimizer-owned state to an engine and its module."""

        model_to_move = model.module if hasattr(model, "module") else model
        if id(model) in self.engine_state_controllers or id(model_to_move) in self.engine_state_controllers:
            raise ValueError("engine state controller is already registered")
        self.engine_state_controllers[id(model)] = controller
        self.engine_state_controllers[id(model_to_move)] = controller

    def register_block_streamer(self, model, streamer):
        """Interlock whole-model moves with an immutable block streamer.

        A streamed backbone must never be passed through ``Module.to(cuda)``:
        doing so would materialize every frozen block and defeat B3. Both the
        DeepSpeed engine and its wrapped module map to the same streamer.
        """
        model_to_move = model.module if hasattr(model, "module") else model
        immutable_backing = self._immutable_model_backing(model, model_to_move)
        if immutable_backing is not None:
            self._validate_hybrid_pair(
                model_to_move=model_to_move,
                block_streamer=streamer,
                immutable_backing=immutable_backing,
            )
        self.block_streamers[id(model)] = streamer
        self.block_streamers[id(model_to_move)] = streamer

    def register_immutable_model_backing(self, model, backing):
        """Interlock whole-model swaps with node-shared immutable host state."""

        model_to_move = model.module if hasattr(model, "module") else model
        block_streamer = self._block_streamer(model, model_to_move)
        if block_streamer is not None:
            self._validate_hybrid_pair(
                model_to_move=model_to_move,
                block_streamer=block_streamer,
                immutable_backing=backing,
            )
        self.immutable_model_backings[id(model)] = backing
        self.immutable_model_backings[id(model_to_move)] = backing

    @staticmethod
    def _validate_hybrid_pair(*, model_to_move, block_streamer, immutable_backing):
        """Accept only an explicit, disjoint composite residency contract."""

        try:
            managed_ids = frozenset(block_streamer.managed_parameter_ids)
            external_ids = frozenset(
                immutable_backing.externally_managed_parameter_ids
            )
            backed_ids = frozenset(immutable_backing.backed_parameter_ids)
        except AttributeError as error:
            raise ValueError(
                "a model cannot use both block streaming and whole-model backing "
                "without an explicit hybrid ownership contract"
            ) from error
        if getattr(block_streamer, "model", None) is not model_to_move:
            raise ValueError("block streamer is registered against a different model")
        if getattr(immutable_backing, "model", None) is not model_to_move:
            raise ValueError("immutable backing is registered against a different model")
        if not managed_ids:
            raise ValueError("hybrid block streamer owns no parameters")
        if managed_ids != external_ids:
            raise ValueError(
                "hybrid residency ownership mismatch between streamer and backing"
            )
        if managed_ids & backed_ids:
            raise ValueError("hybrid residency owners overlap")

    def _immutable_model_backing(self, model, model_to_move):
        return self.immutable_model_backings.get(
            id(model), self.immutable_model_backings.get(id(model_to_move))
        )

    def _block_streamer(self, model, model_to_move):
        return self.block_streamers.get(
            id(model), self.block_streamers.get(id(model_to_move))
        )

    def compute_state_is_on_device(self, model, device):
        """Return whether the model state needed outside streamed blocks is resident.

        Streamed frozen block parameters intentionally remain on CPU between block
        calls, so checking ``next(model.parameters()).device`` is not a valid
        residency test.  Whole-model swaps, however, must materialize every
        parameter not owned by the block streamer (patch/input/output modules and
        trainable adapters) on the compute device.  This narrow predicate lets a
        phase entry repair a stale role placement without treating streamed CPU
        storage as a miss.
        """

        model_to_move = model.module if hasattr(model, "module") else model
        block_streamer = self._block_streamer(model, model_to_move)
        streamed_parameter_ids = (
            frozenset(block_streamer.managed_parameter_ids)
            if block_streamer is not None
            else frozenset()
        )
        target = torch.device(device)
        inspected = 0
        for parameter in model_to_move.parameters():
            if id(parameter) in streamed_parameter_ids:
                continue
            inspected += 1
            if parameter.device != target:
                return False
        if inspected == 0:
            raise RuntimeError(
                "cannot determine model compute residency: no non-streamed parameters"
            )
        return True

    def _engine_state_controller(self, model, model_to_move):
        return self.engine_state_controllers.get(
            id(model), self.engine_state_controllers.get(id(model_to_move))
        )

    def assert_engine_state_controllers_balanced(self):
        seen = set()
        for controller in self.engine_state_controllers.values():
            if id(controller) in seen:
                continue
            seen.add(id(controller))
            controller.assert_balanced()

    @contextlib.contextmanager
    def phase_resident_frozen_blocks(self, model, *, phase):
        """Retain a model's selected immutable blocks for one no-grad phase."""

        model_to_move = model.module if hasattr(model, "module") else model
        block_streamer = self._block_streamer(model, model_to_move)
        if block_streamer is None:
            yield
            return
        self._record(
            "phase_resident_frozen_blocks_start",
            logical_component=self._label(model, model_to_move),
            resident_phase=str(phase),
            streamed_block_indices=sorted(block_streamer.streamed_block_indices),
        )
        try:
            with block_streamer.phase_resident_no_grad(phase=str(phase)):
                yield
        finally:
            self._record(
                "phase_resident_frozen_blocks_end",
                logical_component=self._label(model, model_to_move),
                resident_phase=str(phase),
                resident_bytes=int(block_streamer.stats["resident_bytes"]),
            )

    def _label(self, model, model_to_move):
        return self.model_labels.get(
            id(model),
            self.model_labels.get(id(model_to_move), type(model_to_move).__name__),
        )

    def _record(self, phase, **extra):
        if self.phase_tracer is not None:
            self.phase_tracer.record(phase, **extra)

    def _finish_move(self, *, model, model_to_move, direction, started_at, **extra):
        # The original manager uses the default stream and immediately consumes
        # the destination model.  Synchronizing only while tracing exposes the
        # actual transfer rather than its enqueue latency.
        if self.phase_tracer is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        host_trimmed = self.trim_host_fn() if self.trim_host_fn is not None else False
        self._record(
            "model_move_end",
            move_sequence=self.move_sequence,
            logical_component=self._label(model, model_to_move),
            direction=direction,
            elapsed_seconds=time.perf_counter() - started_at,
            host_trimmed=host_trimmed,
            **extra,
        )

    def move_to_cpu(
        self,
        model,
        non_blocking=True,
        offload_grad=False,
        offload_engine_lp_state=False,
    ):
        model_to_move = model.module if hasattr(model, "module") else model
        engine_state_controller = self._engine_state_controller(model, model_to_move)
        if offload_engine_lp_state and engine_state_controller is None:
            raise RuntimeError("requested LP-state offload without a registered controller")
        block_streamer = self._block_streamer(model, model_to_move)
        immutable_backing = self._immutable_model_backing(model, model_to_move)
        if block_streamer is not None and immutable_backing is not None:
            self._validate_hybrid_pair(
                model_to_move=model_to_move,
                block_streamer=block_streamer,
                immutable_backing=immutable_backing,
            )
        direction = (
            "rebind_to_shared_cpu_backing"
            if immutable_backing is not None
            else "d2h"
        )
        self.move_sequence += 1
        started_at = time.perf_counter()
        self._record(
            "model_move_start",
            move_sequence=self.move_sequence,
            logical_component=self._label(model, model_to_move),
            direction=direction,
            non_blocking=non_blocking,
            offload_grad=offload_grad,
            offload_engine_lp_state=offload_engine_lp_state,
        )
        if block_streamer is not None and immutable_backing is not None:
            block_streamer.release_all(reason="whole_model_move_to_cpu")
            immutable_backing.move_to_cpu(non_blocking=non_blocking)
        elif immutable_backing is not None:
            immutable_backing.move_to_cpu(non_blocking=non_blocking)
        elif block_streamer is not None:
            block_streamer.move_non_streamed_state(
                "cpu", non_blocking=non_blocking
            )
        else:
            model_to_move.to("cpu", non_blocking=non_blocking)

        lp_state_bytes = 0
        if offload_engine_lp_state:
            lp_state_bytes = int(engine_state_controller.offload().logical_bytes)

        if id(model) not in self.pinned_models:
            for buffer in model_to_move.buffers():
                if buffer.device.type == "cpu" and not buffer.is_pinned():
                    buffer.data = buffer.data.pin_memory()
            self.pinned_models.add(id(model))

        if offload_grad:
            model_id = id(model)

            if model_id not in self.grad_cache:
                self.grad_cache[model_id] = {}

            for i, param in enumerate(model_to_move.parameters()):
                if param.grad is not None:
                    if i not in self.grad_cache[model_id]:
                        self.grad_cache[model_id][i] = torch.empty_like(param.grad, device="cpu", pin_memory=True)

                    self.grad_cache[model_id][i].copy_(param.grad, non_blocking=non_blocking)
                    param.grad = None

        free_memory()
        self._finish_move(
            model=model,
            model_to_move=model_to_move,
            direction=direction,
            started_at=started_at,
            non_blocking=non_blocking,
            offload_grad=offload_grad,
            offload_engine_lp_state=offload_engine_lp_state,
            offloaded_engine_lp_state_bytes=lp_state_bytes,
            frozen_block_streaming=block_streamer is not None,
            immutable_model_backing=immutable_backing is not None,
            hybrid_residency=(
                block_streamer is not None and immutable_backing is not None
            ),
            frozen_parameter_rebound_without_d2h_bytes=(
                immutable_backing.shared_immutable_bytes
                if immutable_backing is not None
                else 0
            ),
        )

    def move_to_gpu(self, model, device, non_blocking=True, load_grad=False):
        model_to_move = model.module if hasattr(model, "module") else model
        engine_state_controller = self._engine_state_controller(model, model_to_move)
        reloaded_engine_lp_state_bytes = 0
        if engine_state_controller is not None and engine_state_controller.is_offloaded:
            # Restore the flat buffer before the generic module move.  DeepSpeed
            # rebinds trainable Parameters to this flat storage, avoiding a
            # transient second rank-private GPU copy of every LoRA tensor.
            reloaded_engine_lp_state_bytes = int(
                engine_state_controller.reload().logical_bytes
            )
        self.move_sequence += 1
        started_at = time.perf_counter()
        self._record(
            "model_move_start",
            move_sequence=self.move_sequence,
            logical_component=self._label(model, model_to_move),
            direction="h2d",
            non_blocking=non_blocking,
            load_grad=load_grad,
        )
        block_streamer = self._block_streamer(model, model_to_move)
        immutable_backing = self._immutable_model_backing(model, model_to_move)
        if block_streamer is not None and immutable_backing is not None:
            self._validate_hybrid_pair(
                model_to_move=model_to_move,
                block_streamer=block_streamer,
                immutable_backing=immutable_backing,
            )
        if block_streamer is not None and immutable_backing is not None:
            immutable_backing.move_to_device(
                device, non_blocking=non_blocking
            )
            block_streamer.assert_idle()
        elif immutable_backing is not None:
            immutable_backing.move_to_device(
                device, non_blocking=non_blocking
            )
        elif block_streamer is not None:
            block_streamer.move_non_streamed_state(
                device, non_blocking=non_blocking
            )
        else:
            model_to_move.to(device, non_blocking=non_blocking)

        if load_grad:
            model_id = id(model)
            if model_id in self.grad_cache:
                for i, param in enumerate(model_to_move.parameters()):
                    if i in self.grad_cache[model_id]:
                        if param.grad is None:
                            param.grad = self.grad_cache[model_id][i].to(device, non_blocking=non_blocking)
                        else:
                            param.grad.copy_(self.grad_cache[model_id][i], non_blocking=non_blocking)

        self._finish_move(
            model=model,
            model_to_move=model_to_move,
            direction="h2d",
            started_at=started_at,
            non_blocking=non_blocking,
            load_grad=load_grad,
            reloaded_engine_lp_state_bytes=reloaded_engine_lp_state_bytes,
            frozen_block_streaming=block_streamer is not None,
            immutable_model_backing=immutable_backing is not None,
            hybrid_residency=(
                block_streamer is not None and immutable_backing is not None
            ),
            materialized_from_shared_bytes=(
                immutable_backing.shared_immutable_bytes
                if immutable_backing is not None
                else 0
            ),
        )

    def swap_models(
        self,
        *,
        incoming_model,
        outgoing_model,
        device,
        outgoing_offload_grad=False,
        incoming_load_grad=False,
        outgoing_offload_engine_lp_state=False,
    ):
        """Exchange two full models in node-local rank waves.

        The original rank-local policy lets every rank first materialize the
        outgoing model on CPU.  Eight 14B replicas then hit the shared host
        cgroup before any rank has released its incoming CPU copy.  A wave
        completes D2H and H2D on a bounded subset of local ranks before the
        next subset starts, preserving model order while bounding node-level
        transient residency.
        """
        if not self.wave_swap_enabled:
            self.move_to_cpu(
                outgoing_model,
                offload_grad=outgoing_offload_grad,
                offload_engine_lp_state=outgoing_offload_engine_lp_state,
            )
            self.move_to_gpu(incoming_model, device, load_grad=incoming_load_grad)
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = int(
            os.environ.get(
                "LOCAL_WORLD_SIZE",
                str(torch.cuda.device_count() if torch.cuda.is_available() else 1),
            )
        )
        if local_world_size <= 0 or local_rank < 0 or local_rank >= local_world_size:
            raise RuntimeError(
                f"invalid local rank topology: rank={local_rank}, world={local_world_size}"
            )
        started_at = time.perf_counter()
        self._record(
            "model_swap_start",
            incoming_component=self._label(
                incoming_model,
                incoming_model.module if hasattr(incoming_model, "module") else incoming_model,
            ),
            outgoing_component=self._label(
                outgoing_model,
                outgoing_model.module if hasattr(outgoing_model, "module") else outgoing_model,
            ),
            swap_wave_size=self.swap_wave_size,
            local_world_size=local_world_size,
        )
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        for start, end in swap_wave_ranges(local_world_size, self.swap_wave_size):
            active = start <= local_rank < end
            if active:
                self.move_to_cpu(
                    outgoing_model,
                    offload_grad=outgoing_offload_grad,
                    offload_engine_lp_state=outgoing_offload_engine_lp_state,
                )
                self.move_to_gpu(incoming_model, device, load_grad=incoming_load_grad)
            if distributed:
                torch.distributed.barrier()
        self._record(
            "model_swap_end",
            incoming_component=self._label(
                incoming_model,
                incoming_model.module if hasattr(incoming_model, "module") else incoming_model,
            ),
            outgoing_component=self._label(
                outgoing_model,
                outgoing_model.module if hasattr(outgoing_model, "module") else outgoing_model,
            ),
            swap_wave_size=self.swap_wave_size,
            local_world_size=local_world_size,
            elapsed_seconds=time.perf_counter() - started_at,
        )


class Gan_D_Loss_With_Cached_Grad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        latent,
        discriminator,
        timestep,
        prompt_embeds,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        label,
    ):
        latent_copy = latent.detach().requires_grad_(True)

        with torch.enable_grad():
            _, logits = discriminator(
                hidden_states=latent_copy,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                gan_mode=True,
                return_dict=False,
            )
            temp_loss = cal_gan_loss(logits, label=label)
            del logits
            free_memory()

            grad = torch.autograd.grad(
                temp_loss,
                latent_copy,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0].detach()

        del latent_copy
        free_memory()

        ctx.save_for_backward(grad)
        return temp_loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output, None, None, None, None, None, None, None, None, None, None, None


# ======================================== GAN Related ========================================


def cal_gan_loss(logit, label=1):
    if logit is None:
        return 0
    elif isinstance(logit, list):
        gan_loss = torch.tensor(0, device=torch.cuda.current_device())
        for logit_item in logit:
            gan_loss = gan_loss + torch.mean(F.softplus(logit_item * label))
        return gan_loss / len(logit)
    else:
        return torch.mean(F.softplus(logit * label).float())


def gan_crop_video_spatial(x, scale=0.5, offsets=None):
    B, C, T, H, W = x.shape
    H2 = int(H * scale)
    W2 = int(W * scale)
    if offsets is None:
        tops = torch.randint(0, H - H2 + 1, (B,), device=x.device)
        lefts = torch.randint(0, W - W2 + 1, (B,), device=x.device)
    else:
        offsets = offsets.to(device=x.device, dtype=torch.long)
        if tuple(offsets.shape) != (B, 2):
            raise ValueError(f"GAN crop offsets must have shape {(B, 2)}, got {tuple(offsets.shape)}")
        tops, lefts = offsets[:, 0], offsets[:, 1]
        if bool(((tops < 0) | (tops > H - H2) | (lefts < 0) | (lefts > W - W2)).any()):
            raise ValueError("GAN crop offsets are outside the legal spatial range")
    x2 = torch.zeros(B, C, T, H2, W2, device=x.device, dtype=x.dtype)
    for i in range(B):
        x2[i] = x[i, :, :, tops[i] : tops[i] + H2, lefts[i] : lefts[i] + W2]
    return x2


def prepare_real_latents_for_gan(
    accelerator,
    vae,
    clean_all_latent,
    latent_window_size,
    history_sizes,
    num_critic_input_frames,
    dmd_is_low_vram_mode=False,
    vram_manager=None,
):
    if dmd_is_low_vram_mode:
        vram_manager.move_to_gpu(vae, accelerator.device)
    else:
        vae.to(accelerator.device)
    vae.requires_grad_(False)
    vae.eval()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    clean_all_latent = clean_all_latent[:, :, sum(history_sizes) :, :, :]
    num_sections = math.ceil(clean_all_latent.shape[2] / latent_window_size)
    total_frame_latent = []
    for i in range(num_sections):
        start_idx = i * latent_window_size
        end_idx = min((i + 1) * latent_window_size, clean_all_latent.shape[2])
        cur_section = clean_all_latent[:, :, start_idx:end_idx, :, :]
        with torch.no_grad():
            decoded = vae.decode(
                cur_section.to(vae.device, dtype=vae.dtype) / latents_std + latents_mean, return_dict=False
            )[0]
        total_frame_latent.append(decoded)

    num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
    combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)
    max_start_idx = combined_frames.shape[2] - num_rgb_frames
    start_idx = random.randint(0, max_start_idx)
    selected_frames = combined_frames[:, :, start_idx : start_idx + num_rgb_frames, :, :]
    with torch.no_grad():
        reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
        gan_vae_latents = (reconstructed_latent - latents_mean) * latents_std

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(vae)

    latents_mean = None
    latents_std = None
    decoded = None
    total_frame_latent = None
    combined_frames = None
    selected_frames = None
    reconstructed_latent = None
    del latents_mean
    del latents_std
    del decoded
    del total_frame_latent
    del combined_frames
    del selected_frames
    del reconstructed_latent
    free_memory()

    return gan_vae_latents


# ======================================== Coarse to Fine Learning ========================================


def sample_dynamic_dmd_num_latent_sections(
    min_sections: int = 3,
    max_sections: int = 3,
    dmd_dynamic_alpha: float = 1.5,
    dmd_dynamic_beta: float = 4.0,
    dmd_dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dmd_dynamic_step: int = 1000,
    device: str = "cuda",
):
    assert min_sections >= 1
    if min_sections == max_sections:
        return min_sections

    dmd_dynamic_step = float(dmd_dynamic_step)
    global_step = float(global_step)

    # Sample a value between 0 and 1
    if dmd_dynamic_sample_type == "uniform":
        t = torch.rand(1, device=device).item()
    elif dmd_dynamic_sample_type == "beta":
        # Adjust alpha and beta based on training progress
        if dmd_dynamic_step > 0:
            progress = min(global_step / dmd_dynamic_step, 1.0)
            # Cosine decay: starts at 1.0, decays to 0.0
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            # Gradually reduce alpha and beta towards 1.0 (uniform distribution)
            alpha = 1.0 + (dmd_dynamic_alpha - 1.0) * cosine_decay
            beta = 1.0 + (dmd_dynamic_beta - 1.0) * cosine_decay
        else:
            alpha = dmd_dynamic_alpha
            beta = dmd_dynamic_beta

        t = torch.distributions.Beta(alpha, beta).sample((1,)).to(device).item()
    else:
        raise ValueError(f"Unsupported sample_type: {dmd_dynamic_sample_type}. Choose from ['uniform', 'beta'].")

    # Map to the range [min_sections, max_sections]
    num_sections = min_sections + t * (max_sections - min_sections)

    # Round to nearest integer and clamp
    num_sections = int(round(num_sections))
    num_sections = max(min_sections, min(max_sections, num_sections))

    return num_sections


def sample_dynamic_timestep(
    B: int,
    num_train_timestep: int = 1000,
    min_timestep: int = 0,
    max_timestep: int = 1000,
    min_step: int = 20,
    max_step: int = 980,
    timestep_shift: float = 1.0,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    device: str = "cuda",
):
    dynamic_step = float(dynamic_step)
    global_step = float(global_step)

    # dynamic timestep
    if dynamic_sample_type == "uniform":
        t = torch.rand(B, device=device) * (1.0 - 0.001) + 0.001
    elif dynamic_sample_type == "beta":
        if dynamic_step > 0:
            progress = min(global_step / dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            dynamic_alpha = 1.0 + (dynamic_alpha - 1.0) * cosine_decay
            dynamic_beta = 1.0 + (dynamic_beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(dynamic_alpha, dynamic_beta).sample((B,)).to(device)
    else:
        raise ValueError(f"Unsupported dynamic_sample_type: {dynamic_sample_type}. Choose from ['uniform', 'beta'].")

    # timestep warping
    timestep = min_timestep + t * (max_timestep - min_timestep)
    if timestep_shift > 1:
        timestep = (
            timestep_shift
            * (timestep / num_train_timestep)
            / (1 + (timestep_shift - 1) * (timestep / num_train_timestep))
            * num_train_timestep
        )
    timestep = timestep.clamp(min_step, max_step)

    return timestep.round().long()


# ======================================== Helper ========================================


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only=False, is_sync=True):
    if is_sync:
        if accelerator.is_main_process:
            if last_step_only:
                step = num_denoising_steps - 1
            else:
                step = torch.randint(low=0, high=num_denoising_steps, size=(), device=accelerator.device).item()
            step_tensor = torch.tensor(step, dtype=torch.long, device=accelerator.device)
        else:
            step_tensor = torch.empty((), dtype=torch.long, device=accelerator.device)

        broadcast(step_tensor, from_process=0)
        return step_tensor.item()
    else:
        if last_step_only:
            step = num_denoising_steps - 1
        else:
            step = torch.randint(low=0, high=num_denoising_steps, size=(), device=accelerator.device).item()
        return step


def sample_block_noise(scheduler, batch_size, channel, num_frames, height, width):
    gamma = scheduler.config.gamma
    block_number = batch_size * channel * num_frames * (height // 2) * (width // 2)
    if os.environ.get("DISTILLSCHEDULE_STABLE_BLOCK_NOISE") == "1":
        if not math.isclose(gamma, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stable block-noise compatibility path requires gamma=1/3")
        from runtime.official_baseline import stable_block_noise

        noise = stable_block_noise(block_number)
    else:
        cov = torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma
        dist = torch.distributions.MultivariateNormal(torch.zeros(4, device=cov.device), covariance_matrix=cov)
        noise = dist.sample((block_number,))  # [block number, 4]
    noise = noise.view(batch_size, channel, num_frames, height // 2, width // 2, 2, 2)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
    return noise


def add_noise(original_samples, noise, timestep, sigmas, timesteps):
    sigmas = sigmas.to(noise.device)
    timesteps = timesteps.to(noise.device)
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    sample = (1 - sigma) * original_samples + sigma * noise
    return sample.type_as(noise)


def convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps):
    # use higher precision for calculations
    original_dtype = flow_pred.dtype
    device = flow_pred.device
    flow_pred, xt, sigmas, timesteps = (x.double().to(device) for x in (flow_pred, xt, sigmas, timesteps))

    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(original_dtype)


def convert_xt_pred_to_x0(noise, xt, timestep, sigmas, timesteps):
    # use higher precision for calculations
    original_dtype = xt.dtype
    device = xt.device
    noise, xt, sigmas, timesteps = (x.double().to(device) for x in (noise, xt, sigmas, timesteps))

    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = (xt - sigma_t * noise) / (1 - sigma_t)
    return x0_pred.to(original_dtype)


# ======================================== Staged Backward Simulation ========================================


def inference_with_trajectory_stage1(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # For GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    # For VAE Re-Encode
    is_dmd_vae_decode: bool = False,
    # For Consistency Align
    is_consistency_align: bool = False,
    # For KV Cache
    use_kv_cache: bool = True,
):
    raise NotImplementedError
    batch_size, num_channels_latents, latent_window_size, height, width = noise.shape
    num_denoising_steps = len(denoising_step_list)
    init_exit_flag = generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only)
    denoising_step_list = torch.tensor(denoising_step_list)
    if timestep_shift > 1:
        denoising_step_list = (
            timestep_shift
            * (denoising_step_list / 1000)
            / (1 + (timestep_shift - 1) * (denoising_step_list / 1000))
            * 1000
        )

    consistency_align_loss = torch.tensor(0.0)
    if is_consistency_align:
        consistentcy_align_loss_list = []

    history_sizes = sorted(history_sizes, reverse=True)  # From large to small
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        ) = gt_all_data
    else:
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height,
            width,
            device=accelerator.device,
            dtype=torch.float32,
        )

    assert num_rollout_sections * latent_window_size >= num_critic_input_frames

    dmd_num_input_frames_sections = (num_critic_input_frames + latent_window_size - 1) // latent_window_size
    if num_rollout_sections <= dmd_num_input_frames_sections:
        start_gradient_section_index = 0
    elif last_section_grad_only:
        start_gradient_section_index = num_rollout_sections - 1
    else:
        start_gradient_section_index = num_rollout_sections - dmd_num_input_frames_sections

    # Step 1: Denoising loop
    image_latents = None
    total_generated_latent_frames = 0
    for k in range(num_rollout_sections):
        noisy_model_input = torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype)
        is_first_section = k == 0
        is_second_section = k == 1
        if not is_use_gt_history:
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height,
                        width,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                raise NotImplementedError

        if not is_use_gt_history and is_corrupt_history_latents:
            latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                # choose mode
                corrupt_mode=args.training_config.corrupt_mode_history,
                noise_mode_prob=args.training_config.corrupt_mode_prob_history,
                # for noise
                is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
                is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
                corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
                corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
                corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
                noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
                # for downsample
                downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
                downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
            )

        if is_add_saturation:
            latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                saturation_ratio_min=args.training_config.saturation_ratio_min,
                saturation_ratio_max=args.training_config.saturation_ratio_max,
                saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
            )

        should_compute_grad = k >= start_gradient_section_index
        if is_consistency_align and should_compute_grad:
            pred_x0_list = []
        for index, current_timestep in enumerate(denoising_step_list):
            is_first_step = index == 0
            exit_flag = index == init_exit_flag
            timestep = torch.ones([batch_size], device=accelerator.device, dtype=torch.int64) * current_timestep

            if not exit_flag:
                with torch.no_grad():
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                        latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                        return_dict=False,
                        is_first_denoising_step=is_first_step,
                    )[0]
                    pred_x0 = convert_flow_pred_to_x0(
                        flow_pred=model_pred,
                        xt=noisy_model_input,
                        timestep=timestep,
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                    next_timestep = denoising_step_list[index + 1]
                    noisy_model_input = add_noise(
                        pred_x0,
                        torch.randn_like(pred_x0, device=accelerator.device, dtype=noise.dtype),
                        next_timestep * torch.ones([batch_size], device=accelerator.device, dtype=torch.long),
                        sigmas,
                        timesteps,
                    )

                    if is_consistency_align and should_compute_grad:
                        pred_x0_list.append(pred_x0)
            else:
                # for getting real output
                with torch.set_grad_enabled(should_compute_grad):
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                        latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                        return_dict=False,
                        is_first_denoising_step=is_first_step,
                    )[0]
                    pred_x0 = convert_flow_pred_to_x0(
                        flow_pred=model_pred,
                        xt=noisy_model_input,
                        timestep=timestep,
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                    if is_consistency_align and should_compute_grad:
                        pred_x0_list.append(pred_x0)
                break

            if is_consistency_align and should_compute_grad and len(pred_x0_list) > 1:
                prev_x0s = torch.stack(pred_x0_list[:-1])
                last_x0 = pred_x0_list[-1]
                temp_mse_loss = 0.5 * F.mse_loss(prev_x0s, last_x0.unsqueeze(0).expand_as(prev_x0s), reduction="mean")
                consistentcy_align_loss_list.append(temp_mse_loss)

        if use_kv_cache:
            transformer.clear_kv_cache()

        if is_keep_x0 and (is_first_section or (is_skip_first_section and is_second_section)):
            image_latents = pred_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, pred_x0], dim=2)

    # Step 2: record the model's output
    total_available_frames = history_latents.shape[2] - sum(history_sizes)
    max_start_section_idx = max(0, (total_available_frames - num_critic_input_frames) // latent_window_size)
    # ---------------
    # Way 1, random
    # start_section_idx = torch.randint(0, max_start_section_idx + 1, (1,)).item()
    # Way 2, fix
    start_section_idx = max_start_section_idx
    # ---------------
    start_frame = sum(history_sizes) + start_section_idx * latent_window_size

    if is_dmd_vae_decode:
        end_frame = history_latents.shape[2]
    else:
        end_frame = start_frame + num_critic_input_frames
        end_frame = min(end_frame, history_latents.shape[2])

    output = history_latents[:, :, start_frame:end_frame, :, :]

    # Step 3: Return the denoised timestep
    if init_exit_flag == len(denoising_step_list) - 1:
        denoised_timestep_to = 0
        denoised_timestep_from = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag]).abs(), dim=0).item()
        )
    else:
        denoised_timestep_to = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag + 1]).abs(), dim=0).item()
        )
        denoised_timestep_from = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag]).abs(), dim=0).item()
        )

    if is_consistency_align and len(consistentcy_align_loss_list) > 0:
        consistency_align_loss = torch.stack(consistentcy_align_loss_list).mean()

    if return_sim_step:
        return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss, init_exit_flag + 1

    return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss


def inference_with_trajectory_stage2(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For Stage 2
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # For GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    # For VAE Re-Encode
    is_dmd_vae_decode: bool = False,
    # For Multi Stage Backward Simulated
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    # For Consistency Align
    is_consistency_align: bool = False,
    # For KV Cache
    use_kv_cache: bool = True,
    operation_trace=None,
    trace_prefix: str = None,
):
    batch_size, num_channels_latents, latent_window_size, height, width = noise.shape

    if operation_trace is not None:
        schedule = operation_trace.operation(f"{trace_prefix}/rollout")["control"][
            "pyramid_stage_call_sequence"
        ]
        init_exit_flag_list = [int(item["denoising_exit"]) for item in schedule]
        if len(init_exit_flag_list) != stage2_num_stages:
            raise ValueError("operation trace pyramid stage count mismatch")
    else:
        init_exit_flag_list = []
        for i_s in range(stage2_num_stages):
            num_denoising_steps = stage2_num_inference_steps_list[i_s]
            init_exit_flag_list.append(generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only))

    if is_multi_pyramid_stage_backward_simulated:
        divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
        pyramid_stage_videos = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // divisor,
            width // divisor,
            device=accelerator.device,
            dtype=torch.float32,
        )

    consistency_align_loss = torch.tensor(0.0)
    if is_consistency_align:
        consistentcy_align_loss_list = []

    history_sizes = sorted(history_sizes, reverse=True)  # From large to small
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        ) = gt_all_data
    else:
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height,
            width,
            device=accelerator.device,
            dtype=torch.float32,
        )

    assert num_rollout_sections * latent_window_size >= num_critic_input_frames

    dmd_num_input_frames_sections = (num_critic_input_frames + latent_window_size - 1) // latent_window_size
    if num_rollout_sections <= dmd_num_input_frames_sections:
        start_gradient_section_index = 0
    elif last_section_grad_only:
        start_gradient_section_index = num_rollout_sections - 1
    else:
        start_gradient_section_index = num_rollout_sections - dmd_num_input_frames_sections

    # Step 1: Denoising loop
    image_latents = None
    total_generated_latent_frames = 0
    for k in range(num_rollout_sections):
        if operation_trace is not None:
            if k != 0:
                raise ValueError("M0 operation trace currently supports one rollout section")
            noisy_model_input = operation_trace.tensor(f"{trace_prefix}/rollout_noise").to(
                accelerator.device, dtype=noise.dtype
            )
        else:
            noisy_model_input = torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype)

        num_frmaes_pyramid, height_pyramid, width_pyramid = (
            noisy_model_input.shape[-3],
            noisy_model_input.shape[-2],
            noisy_model_input.shape[-1],
        )
        noisy_model_input = rearrange(noisy_model_input, "b c t h w -> (b t) c h w")
        # by default, we needs to start from the block noise
        for _ in range(stage2_num_stages - 1):
            height_pyramid //= 2
            width_pyramid //= 2
            noisy_model_input = (
                F.interpolate(
                    noisy_model_input,
                    size=(height_pyramid, width_pyramid),
                    mode="bilinear",
                )
                * 2
            )
        noisy_model_input = rearrange(noisy_model_input, "(b t) c h w -> b c t h w", t=num_frmaes_pyramid)

        is_first_section = k == 0
        is_second_section = k == 1
        if not is_use_gt_history:
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height,
                        width,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                raise NotImplementedError

        if not is_use_gt_history and is_corrupt_history_latents:
            latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                # choose mode
                corrupt_mode=args.training_config.corrupt_mode_history,
                noise_mode_prob=args.training_config.corrupt_mode_prob_history,
                # for noise
                is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
                is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
                corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
                corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
                corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
                noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
                # for downsample
                downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
                downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
            )

        if is_add_saturation and operation_trace is None:
            latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                saturation_ratio_min=args.training_config.saturation_ratio_min,
                saturation_ratio_max=args.training_config.saturation_ratio_max,
                saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
            )

        pred_x0 = None
        start_point_list = [noisy_model_input]
        should_compute_grad = k >= start_gradient_section_index
        for i_s in range(stage2_num_stages):
            if is_consistency_align and should_compute_grad:
                pred_x0_list = []

            if is_amplify_first_chunk and is_first_section:
                if not is_use_gt_history:
                    scheduler.set_timesteps(
                        stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=accelerator.device
                    )
                elif (
                    latents_history_short.sum() == 0
                    and latents_history_mid.sum() == 0
                    and latents_history_long.sum() == 0
                ):
                    scheduler.set_timesteps(
                        stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=accelerator.device
                    )
                else:
                    scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=accelerator.device)
            else:
                scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=accelerator.device)

            original_timestep = scheduler.timesteps
            scheduler.timesteps = scheduler.timesteps[:-1]
            scheduler.sigmas = torch.cat([scheduler.sigmas[:-2], scheduler.sigmas[-1:]])

            timesteps_per_stage = scheduler.timesteps_per_stage[i_s]
            sigmas_per_stage = scheduler.sigmas_per_stage[i_s]

            if i_s > 0:
                # important here !!!
                assert pred_x0 is not None, "pred_x0 should be set in previous iteration"
                noisy_model_input = pred_x0
                height_pyramid *= 2
                width_pyramid *= 2
                num_frames = noisy_model_input.shape[2]
                noisy_model_input = rearrange(noisy_model_input, "b c t h w -> (b t) c h w")
                noisy_model_input = F.interpolate(
                    noisy_model_input, size=(height_pyramid, width_pyramid), mode="nearest"
                )
                noisy_model_input = rearrange(noisy_model_input, "(b t) c h w -> b c t h w", t=num_frames)
                # Fix the stage
                ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]  # the original coeff of signal
                gamma = scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, channel, num_frames, height_pyramid, width_pyramid = noisy_model_input.shape
                if operation_trace is not None:
                    noise = operation_trace.tensor(f"{trace_prefix}/block_noise/{i_s}").to(
                        accelerator.device, dtype=noisy_model_input.dtype
                    )
                else:
                    noise = sample_block_noise(scheduler, batch_size, channel, num_frames, height_pyramid, width_pyramid)
                    noise = noise.to(device=accelerator.device, dtype=noisy_model_input.dtype)
                noisy_model_input = alpha * noisy_model_input + beta * noise  # To fix the block artifact

                start_point_list.append(noisy_model_input)

            if use_dynamic_shifting:
                temp_sigmas, temp_sigmas_per_stage = apply_schedule_shift(
                    scheduler.sigmas,
                    noisy_model_input,
                    sigmas_two=sigmas_per_stage,
                    base_seq_len=args.training_config.base_seq_len,
                    max_seq_len=args.training_config.max_seq_len,
                    base_shift=args.training_config.base_shift,
                    max_shift=args.training_config.max_shift,
                    time_shift_type=time_shift_type,
                )

                temp_timesteps = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas[:-1] * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                scheduler.sigmas = temp_sigmas
                scheduler.timesteps = temp_timesteps

                temp_timesteps_per_stage = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas_per_stage * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                sigmas_per_stage = temp_sigmas_per_stage
                timesteps_per_stage = temp_timesteps_per_stage

            denoising_step_list = scheduler.timesteps

            if operation_trace is not None:
                init_exit_flag = init_exit_flag_list[i_s]
            elif is_amplify_first_chunk and is_first_section:
                if not is_use_gt_history:
                    init_exit_flag = generate_and_sync_flag(
                        accelerator, stage2_num_inference_steps_list[i_s] * 2, last_step_only
                    )
                elif (
                    latents_history_short.sum() == 0
                    and latents_history_mid.sum() == 0
                    and latents_history_long.sum() == 0
                ):
                    init_exit_flag = generate_and_sync_flag(
                        accelerator, stage2_num_inference_steps_list[i_s] * 2, last_step_only, is_sync=False
                    )
                else:
                    init_exit_flag = init_exit_flag_list[i_s]
            else:
                init_exit_flag = init_exit_flag_list[i_s]

            for index, current_timestep in enumerate(denoising_step_list):
                is_first_step = i_s == 0 and index == 0
                exit_flag = index == init_exit_flag
                timestep = torch.ones([batch_size], device=accelerator.device, dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                            latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                            return_dict=False,
                            is_first_denoising_step=is_first_step,
                        )[0]
                        pred_x0 = convert_flow_pred_to_x0(
                            flow_pred=model_pred,
                            xt=noisy_model_input,
                            timestep=timestep,
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        noisy_model_input = add_noise(
                            pred_x0,
                            start_point_list[i_s],
                            next_timestep * torch.ones([batch_size], device=accelerator.device, dtype=torch.long),
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )

                        if is_consistency_align and should_compute_grad:
                            pred_x0_list.append(pred_x0)
                else:
                    # for getting real output
                    with torch.set_grad_enabled(should_compute_grad):
                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                            latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                            return_dict=False,
                            is_first_denoising_step=is_first_step,
                        )[0]
                        pred_x0 = convert_flow_pred_to_x0(
                            flow_pred=model_pred,
                            xt=noisy_model_input,
                            timestep=timestep,
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )
                        if is_consistency_align and should_compute_grad:
                            pred_x0_list.append(pred_x0)
                    break

            if is_multi_pyramid_stage_backward_simulated and i_s == init_pyramid_stage_flag:
                if i_s != stage2_num_stages - 1:
                    pred_x0 = convert_xt_pred_to_x0(
                        noise=torch.randn_like(pred_x0, device=accelerator.device, dtype=pred_x0.dtype),
                        xt=pred_x0,
                        timestep=torch.ones([batch_size], device=accelerator.device, dtype=torch.int64)
                        * original_timestep[-1],
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                pyramid_stage_videos = torch.cat([pyramid_stage_videos, pred_x0], dim=2)

            if is_consistency_align and should_compute_grad and len(pred_x0_list) > 1:
                prev_x0s = torch.stack(pred_x0_list[:-1])
                last_x0 = pred_x0_list[-1]
                temp_mse_loss = 0.5 * F.mse_loss(prev_x0s, last_x0.unsqueeze(0).expand_as(prev_x0s), reduction="mean")
                consistentcy_align_loss_list.append(temp_mse_loss)

        if use_kv_cache:
            transformer.clear_kv_cache()

        if is_keep_x0 and (is_first_section or (is_skip_first_section and is_second_section)):
            image_latents = pred_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, pred_x0], dim=2)

    # Step 2: record the model's output
    total_available_frames = history_latents.shape[2] - sum(history_sizes)
    max_start_section_idx = max(0, (total_available_frames - num_critic_input_frames) // latent_window_size)
    # ---------------
    # Way 1, random
    # start_section_idx = torch.randint(0, max_start_section_idx + 1, (1,)).item()
    # Way 2, fix
    start_section_idx = max_start_section_idx
    # ---------------
    start_frame = sum(history_sizes) + start_section_idx * latent_window_size

    if is_dmd_vae_decode:
        end_frame = history_latents.shape[2]
    else:
        end_frame = start_frame + num_critic_input_frames
        end_frame = min(end_frame, history_latents.shape[2])

    # Step 3: Return the denoised timestep
    if is_multi_pyramid_stage_backward_simulated:
        output = pyramid_stage_videos[:, :, start_frame:end_frame, :, :]

        stage_exit_flag = init_exit_flag_list[init_pyramid_stage_flag]
        scheduler.set_timesteps(
            stage2_num_inference_steps_list[init_pyramid_stage_flag] + 1,
            init_pyramid_stage_flag,
            device=accelerator.device,
        )
        original_timestep = scheduler.timesteps
        stage_denoising_step_list = scheduler.timesteps[:-1]
        if stage_exit_flag == len(stage_denoising_step_list) - 1:
            denoised_timestep_to = original_timestep[-1]
        else:
            denoised_timestep_to = stage_denoising_step_list[stage_exit_flag + 1]
        denoised_timestep_from = stage_denoising_step_list[stage_exit_flag]
    else:
        output = history_latents[:, :, start_frame:end_frame, :, :]
        if init_exit_flag == len(denoising_step_list) - 1:
            denoised_timestep_to = original_timestep[-1]
        else:
            denoised_timestep_to = denoising_step_list[init_exit_flag + 1]
        denoised_timestep_from = denoising_step_list[init_exit_flag]

    if is_consistency_align and len(consistentcy_align_loss_list) > 0:
        consistency_align_loss = torch.stack(consistentcy_align_loss_list).mean()

    if return_sim_step:
        return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss, init_exit_flag + 1

    return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss


def consistency_backward_simulation(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # Stage 2
    is_enable_stage2: bool = False,
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    # For VAE Re-Encode
    is_dmd_vae_decode: bool = False,
    # For Multi Stage Backward Simulated
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    # For Consistency Align
    is_consistency_align: bool = False,
    # For KV Cache
    use_kv_cache: bool = True,
    operation_trace=None,
    trace_prefix: str = None,
) -> torch.Tensor:
    common_kwargs = {
        "args": args,
        "accelerator": accelerator,
        "transformer": transformer,
        "scheduler": scheduler,
        "noise": noise,
        "prompt_embeds": prompt_embeds,
        # For Stage 1
        "is_keep_x0": is_keep_x0,
        "history_sizes": history_sizes,
        # For DMD Main
        "denoising_step_list": denoising_step_list,
        "last_step_only": last_step_only,
        "last_section_grad_only": last_section_grad_only,
        "return_sim_step": return_sim_step,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "num_critic_input_frames": num_critic_input_frames,
        "num_rollout_sections": num_rollout_sections,
        "is_skip_first_section": is_skip_first_section,
        "is_amplify_first_chunk": is_amplify_first_chunk,
        # Easy Anti-Drifting
        "is_corrupt_history_latents": is_corrupt_history_latents,
        "is_add_saturation": is_add_saturation,
        # For VAE Re-Encode
        "is_dmd_vae_decode": is_dmd_vae_decode,
        # Consistency Align
        "is_consistency_align": is_consistency_align,
        # For KV Cache
        "use_kv_cache": use_kv_cache,
        "operation_trace": operation_trace,
        "trace_prefix": trace_prefix,
    }

    if is_enable_stage2:
        stage2_kwargs = {
            "use_dynamic_shifting": use_dynamic_shifting,
            "time_shift_type": time_shift_type,
            # Stage 2
            "stage2_num_stages": stage2_num_stages,
            "stage2_num_inference_steps_list": stage2_num_inference_steps_list,
            # GT History
            "is_use_gt_history": is_use_gt_history,
            "gt_all_data": gt_all_data,
            # Multi Stage Backward Simulated
            "is_multi_pyramid_stage_backward_simulated": is_multi_pyramid_stage_backward_simulated,
            "init_pyramid_stage_flag": init_pyramid_stage_flag,
        }
        return inference_with_trajectory_stage2(**common_kwargs, **stage2_kwargs)
    else:
        stage1_kwargs = {
            "timestep_shift": timestep_shift,
        }
        return inference_with_trajectory_stage1(**common_kwargs, **stage1_kwargs)


def run_generator(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    # For VRAM manager
    dmd_is_low_vram_mode: bool = False,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For Stage 2
    is_enable_stage2: bool = False,
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # For GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    # For VAE Re-Encode
    is_dmd_vae_decode: bool = False,
    # For Multi Stage Backward Simulated
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    # For Consistency Align
    is_consistency_align: bool = False,
    # For KV Cache
    use_kv_cache: bool = True,
    operation_trace=None,
    trace_prefix: str = None,
):
    if use_kv_cache:
        transformer.disable_kv_cache()

    pred_image_or_video, denoised_timestep_from, denoised_timestep_to, consistency_align_loss = (
        consistency_backward_simulation(
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            scheduler=scheduler,
            noise=(
                operation_trace.tensor(f"{trace_prefix}/entry_noise").to(
                    accelerator.device, dtype=noise.dtype
                )
                if operation_trace is not None
                else torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype)
            ),
            prompt_embeds=prompt_embeds,
            # For Stage 1
            is_keep_x0=is_keep_x0,
            history_sizes=history_sizes,
            # For Stage 2
            is_enable_stage2=is_enable_stage2,
            stage2_num_stages=stage2_num_stages,
            stage2_num_inference_steps_list=stage2_num_inference_steps_list,
            # For DMD Main
            denoising_step_list=denoising_step_list,
            last_step_only=last_step_only,
            last_section_grad_only=last_section_grad_only,
            return_sim_step=return_sim_step,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            num_critic_input_frames=num_critic_input_frames,
            num_rollout_sections=num_rollout_sections,
            is_skip_first_section=is_skip_first_section,
            is_amplify_first_chunk=is_amplify_first_chunk,
            # For Easy Anti-Drifting
            is_corrupt_history_latents=is_corrupt_history_latents,
            is_add_saturation=is_add_saturation,
            # For GT History
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,
            # For VAE Re-Encode
            is_dmd_vae_decode=is_dmd_vae_decode,
            # For Multi Stage Backward Simulated
            is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
            init_pyramid_stage_flag=init_pyramid_stage_flag,
            # Consistency Align
            is_consistency_align=is_consistency_align,
            # For KV Cache
            use_kv_cache=use_kv_cache,
            operation_trace=operation_trace,
            trace_prefix=trace_prefix,
        )
    )

    if use_kv_cache and dmd_is_low_vram_mode:
        transformer.disable_kv_cache()

    pred_image_or_video_last_21 = pred_image_or_video
    gradient_mask = None

    return (
        pred_image_or_video_last_21,
        gradient_mask,
        denoised_timestep_from,
        denoised_timestep_to,
        consistency_align_loss,
    )


# ======================================== Generator Loss ========================================


def compute_kl_grad(
    accelerator,
    scheduler,
    real_fake_score_model,
    noisy_image_or_video,
    estimated_clean_image_or_video,
    prompt_embeds,
    negative_prompt_embeds,
    # For DMD Main
    timestep,
    sigmas,
    timesteps,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    normalization: bool = True,
    # For Decouple DMD
    is_decouple_dmd: bool = False,
    ca_noisy_image_or_video: torch.Tensor = None,
    dm_noisy_image_or_video: torch.Tensor = None,
    ca_timestep: torch.Tensor = None,
    dm_timestep: torch.Tensor = None,
    # For GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    global_step: int = 0,
):
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            _,
        ) = gt_all_data
    else:
        indices_hidden_states = None
        indices_latents_history_short = None
        indices_latents_history_mid = None
        indices_latents_history_long = None
        latents_history_short = None
        latents_history_mid = None
        latents_history_long = None

    # Step 1: Compute the fake score
    with _helios_stage_range(global_step, "critic", "dmd_fake_score_cond", adapter="fake"):
        pred_fake_image_cond = real_fake_score_model(
            hidden_states=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else dm_timestep,
            encoder_hidden_states=prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )[0]
    pred_fake_image_cond = convert_flow_pred_to_x0(
        flow_pred=pred_fake_image_cond,
        xt=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else dm_timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )

    if fake_guidance_scale != 0.0 and not is_decouple_dmd:
        with _helios_stage_range(global_step, "critic", "dmd_fake_score_uncond", adapter="fake"):
            pred_fake_image_uncond = real_fake_score_model(
                hidden_states=noisy_image_or_video,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
            )[0]
        pred_fake_image_uncond = convert_flow_pred_to_x0(
            flow_pred=pred_fake_image_uncond,
            xt=noisy_image_or_video,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        pred_fake_image = pred_fake_image_cond + (pred_fake_image_cond - pred_fake_image_uncond) * fake_guidance_scale
    else:
        pred_fake_image = pred_fake_image_cond

    # Step 2: Compute the real score
    # We compute the conditional and unconditional prediction
    # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
    unwrap_model(real_fake_score_model).disable_adapters()

    if is_decouple_dmd:
        with _helios_stage_range(global_step, "teacher", "real_score_cond_dm", adapter="disabled"):
            pred_real_image_cond_dm = real_fake_score_model(
                hidden_states=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
                timestep=timestep if not is_decouple_dmd else dm_timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
            )[0]
        pred_real_image_cond_dm = convert_flow_pred_to_x0(
            flow_pred=pred_real_image_cond_dm,
            xt=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else dm_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )

    with _helios_stage_range(global_step, "teacher", "real_score_cond", adapter="disabled"):
        pred_real_image_cond = real_fake_score_model(
            hidden_states=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else ca_timestep,
            encoder_hidden_states=prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )[0]
    pred_real_image_cond = convert_flow_pred_to_x0(
        flow_pred=pred_real_image_cond,
        xt=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else ca_timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )

    if real_guidance_scale != 0.0 or is_decouple_dmd:
        with _helios_stage_range(global_step, "teacher", "real_score_uncond", adapter="disabled"):
            pred_real_image_uncond = real_fake_score_model(
                hidden_states=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
                timestep=timestep if not is_decouple_dmd else ca_timestep,
                encoder_hidden_states=negative_prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
            )[0]
        pred_real_image_uncond = convert_flow_pred_to_x0(
            flow_pred=pred_real_image_uncond,
            xt=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else ca_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        if not is_decouple_dmd:
            pred_real_image = (
                pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * real_guidance_scale
            )
    else:
        pred_real_image = pred_real_image_cond

    unwrap_model(real_fake_score_model).enable_adapters()

    if is_decouple_dmd:
        assert real_guidance_scale != 0.0
        ca_grad = real_guidance_scale * (pred_real_image_cond - pred_real_image_uncond)
        dm_grad = pred_real_image_cond_dm - pred_fake_image_cond

        if normalization:
            ca_normalizer = torch.abs(estimated_clean_image_or_video - pred_real_image_cond).mean(
                dim=[1, 2, 3, 4], keepdim=True
            )
            ca_grad = ca_grad / ca_normalizer
            dm_normalizer = torch.abs(estimated_clean_image_or_video - pred_real_image_cond_dm).mean(
                dim=[1, 2, 3, 4], keepdim=True
            )
            dm_grad = dm_grad / dm_normalizer

        ca_grad = torch.nan_to_num(ca_grad)
        dm_grad = torch.nan_to_num(dm_grad)

        return (
            None,
            ca_grad,
            dm_grad,
            {
                "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
                "dmdtrain_ca_noisy_latent": ca_noisy_image_or_video.detach(),
                "dmdtrain_dm_noisy_latent": dm_noisy_image_or_video.detach(),
                "dmdtrain_pred_real_image": pred_real_image_cond.detach(),
                "dmdtrain_pred_fake_image": pred_fake_image_cond.detach(),
                "dmdtrain_ca_gradient_norm": torch.mean(torch.abs(ca_grad)).detach(),
                "dmdtrain_dm_gradient_norm": torch.mean(torch.abs(dm_grad)).detach(),
                "ca_timestep": ca_timestep.detach(),
                "dm_timestep": dm_timestep.detach(),
            },
        )
    else:
        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = pred_fake_image - pred_real_image

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return (
            grad,
            None,
            None,
            {
                "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
                "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
                "dmdtrain_pred_real_image": pred_real_image.detach(),
                "dmdtrain_pred_fake_image": pred_fake_image.detach(),
                "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
                "timestep": timestep.detach(),
            },
        )


def compute_distribution_matching_loss(
    accelerator,
    scheduler,
    real_fake_score_model,
    image_or_video,
    prompt_embeds,
    negative_prompt_embeds,
    # For VRAM manager
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    is_gan_low_vram_mode: bool = False,
    # For Stage 2
    is_enable_stage2: bool = False,
    # For DMD Main
    gradient_mask: Optional[torch.Tensor] = None,
    denoised_timestep_from: int = 0,
    denoised_timestep_to: int = 0,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,
    num_train_timestep: int = 1000,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    # For GT History
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    # For GAN
    is_use_gan: bool = False,
    # For Decouple DMD
    is_decouple_dmd: bool = False,
    decouple_ca_start_step: int = 2000,
    decouple_ca_end_step: int = 3000,
    # For Dynamic Timestep
    is_forcing_low_renoise: bool = False,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    operation_trace=None,
    trace_prefix: str = None,
):
    original_latent = image_or_video
    batch_size = image_or_video.shape[0]

    timestep = None
    ca_timestep = None
    dm_timestep = None
    noisy_fake_latent = None
    ca_noisy_image_or_video = None
    dm_noisy_image_or_video = None
    with torch.no_grad():
        # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
        min_timestep = denoised_timestep_to if ts_schedule and denoised_timestep_to is not None else min_score_timestep
        if is_forcing_low_renoise:
            max_timestep = 500
        else:
            max_timestep = (
                denoised_timestep_from
                if ts_schedule_max and denoised_timestep_from is not None
                else num_train_timestep
            )
        min_step = int(0.02 * num_train_timestep)
        max_step = int(0.98 * num_train_timestep)

        if operation_trace is not None:
            timestep = operation_trace.tensor(f"{trace_prefix}/dmd_timestep").to(accelerator.device)
            noise = operation_trace.tensor(f"{trace_prefix}/dmd_noise").to(
                accelerator.device, dtype=image_or_video.dtype
            )
        else:
            timestep = sample_dynamic_timestep(
                B=batch_size,
                num_train_timestep=num_train_timestep,
                min_timestep=min_timestep,
                max_timestep=max_timestep,
                min_step=min_step,
                max_step=max_step,
                timestep_shift=timestep_shift,
                dynamic_alpha=dynamic_alpha,
                dynamic_beta=dynamic_beta,
                dynamic_sample_type=dynamic_sample_type,
                global_step=global_step,
                dynamic_step=dynamic_step,
                device=accelerator.device,
            )
            noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)
        noisy_fake_latent = add_noise(
            image_or_video,
            noise,
            timestep,
            sigmas,
            timesteps,
        ).detach()

        noisy_fake_latent = noisy_fake_latent.to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)
        prompt_embeds = prompt_embeds.to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(
            real_fake_score_model.device, dtype=real_fake_score_model.dtype
        )
        if negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            negative_prompt_embeds = negative_prompt_embeds.repeat(prompt_embeds.shape[0], 1, 1)

        if is_decouple_dmd:
            assert decouple_ca_start_step >= dynamic_step
            assert decouple_ca_end_step >= dynamic_step

            # For dm
            dm_noisy_image_or_video = noisy_fake_latent
            dm_timestep = timestep

            # For ca
            ca_min_timestep = min_score_timestep
            if global_step < decouple_ca_start_step:
                ca_max_timestep = max_timestep
            elif decouple_ca_start_step <= global_step < decouple_ca_end_step:
                ca_max_timestep = 565  # approx 564.6138
            else:
                ca_max_timestep = int(denoised_timestep_from)

            ca_timestep = sample_dynamic_timestep(
                B=batch_size,
                num_train_timestep=num_train_timestep,
                min_timestep=ca_min_timestep,
                max_timestep=ca_max_timestep,
                min_step=min_step,
                max_step=max_step,
                timestep_shift=timestep_shift if not is_enable_stage2 and timestep_shift > 1 else 1.0,
                dynamic_alpha=dynamic_alpha,
                dynamic_beta=dynamic_beta,
                dynamic_sample_type=dynamic_sample_type,
                global_step=global_step,
                dynamic_step=dynamic_step,
                device=accelerator.device,
            )

            ca_noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)
            ca_noisy_image_or_video = add_noise(
                image_or_video,
                ca_noise,
                ca_timestep,
                sigmas,
                timesteps,
            ).detach()
            ca_noisy_image_or_video = ca_noisy_image_or_video.to(
                real_fake_score_model.device, dtype=real_fake_score_model.dtype
            )

        # Step 2: Compute the KL grad
        grad, ca_grad, dm_grad, dmd_log_dict = compute_kl_grad(
            accelerator,
            scheduler,
            real_fake_score_model,
            noisy_image_or_video=noisy_fake_latent,
            estimated_clean_image_or_video=original_latent,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # For DMD Main
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
            fake_guidance_scale=fake_guidance_scale,
            real_guidance_scale=real_guidance_scale,
            # For Decouple DMD
            is_decouple_dmd=is_decouple_dmd,
            ca_noisy_image_or_video=ca_noisy_image_or_video,
            dm_noisy_image_or_video=dm_noisy_image_or_video,
            ca_timestep=ca_timestep,
            dm_timestep=dm_timestep,
            # For GT History
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,
            global_step=global_step,
        )

    ca_dmd_loss = torch.tensor(0.0)
    dm_dmd_loss = torch.tensor(0.0)
    if is_decouple_dmd:
        if gradient_mask is not None:
            ca_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() + ca_grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
            dm_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() + dm_grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            ca_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() + ca_grad.double()).detach(), reduction="mean"
            )
            dm_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() + dm_grad.double()).detach(), reduction="mean"
            )
        dmd_loss = ca_dmd_loss + dm_dmd_loss
    else:
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() - grad.double()).detach(), reduction="mean"
            )

    gan_G_loss = torch.tensor(0.0)
    if is_use_gan:
        ca_noisy_image_or_video = None
        dm_noisy_image_or_video = None
        ca_grad = None
        dm_grad = None
        grad = None
        noisy_fake_latent = None
        del ca_noisy_image_or_video
        del dm_noisy_image_or_video
        del ca_grad
        del dm_grad
        del grad
        del noisy_fake_latent
        free_memory()

        if operation_trace is not None:
            noise = operation_trace.tensor(f"{trace_prefix}/gan_g_noise").to(
                accelerator.device, dtype=image_or_video.dtype
            )
            gan_g_crop = operation_trace.tensor(f"{trace_prefix}/gan_g_crop").to(
                accelerator.device
            )
        else:
            noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)
            gan_g_crop = None

        noisy_fake_latent_for_gan = add_noise(
            image_or_video.clone(),
            noise,
            timestep,
            sigmas,
            timesteps,
        ).to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)

        if is_use_gt_history:
            (
                _,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                _,
            ) = gt_all_data
        else:
            indices_hidden_states = None
            indices_latents_history_short = None
            indices_latents_history_mid = None
            indices_latents_history_long = None
            latents_history_short = None
            latents_history_mid = None
            latents_history_long = None

        if is_gan_low_vram_mode:
            gan_G_loss = Gan_D_Loss_With_Cached_Grad.apply(
                gan_crop_video_spatial(noisy_fake_latent_for_gan, offsets=gan_g_crop),
                real_fake_score_model,
                timestep,
                prompt_embeds,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                1,
            )
            del noisy_fake_latent_for_gan
        else:
            _, noisy_fake_logits = real_fake_score_model(
                hidden_states=noisy_fake_latent_for_gan,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                gan_mode=True,
                return_dict=False,
            )
            gan_G_loss = cal_gan_loss(noisy_fake_logits, label=1)
            del noisy_fake_latent_for_gan, noisy_fake_logits

        free_memory()

    return dmd_loss, ca_dmd_loss, dm_dmd_loss, gan_G_loss, dmd_log_dict


def _generator_loss(
    args,
    accelerator,
    real_fake_score_model,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    negative_prompt_embeds,
    # For VRAM manager
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    dmd_is_offload_grad: bool = False,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For Stage 2
    is_enable_stage2: bool = False,
    stage2_num_stages: int = None,
    stage2_num_inference_steps_list: list = None,
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,
    num_train_timestep: int = 1000,
    timestep_shift: float = 1,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # For GT History
    is_use_gt_history: bool = False,
    gt_history_latents: torch.Tensor = None,
    gt_target_latents: torch.Tensor = None,
    gt_x0_latents: torch.Tensor = None,
    # For VAE Re-Encode
    vae=None,
    is_dmd_vae_decode: bool = False,
    # For Multi Stage Backward Simulated
    is_multi_pyramid_stage_backward_simulated: bool = False,
    # For Consistency Align
    is_consistency_align: bool = False,
    consistentcy_align_weight: float = 0.25,
    # For Smoothness
    is_smoothness_loss: bool = False,
    smoothness_loss_weight: float = 1e-2,
    # For KV Cache
    use_kv_cache: bool = True,
    # For Mean-Variance Regularization
    is_mean_var_regular: bool = False,
    mean_var_regular_weight: float = 1.0,
    regular_mean: float = 0.00657021,
    regular_var: float = 0.85126512,
    is_x0_mean_var_regular: bool = False,
    mean_var_regular_x0_weight: float = 1.0,
    regular_x0_mean: float = -0.01618061,
    regular_x0_var: float = 0.27996052,
    #
    is_chunk_mean_var_regular: bool = False,
    chunk_mean_var_regular_weight: float = 1.0,
    chunk_regular_mean: float = 0.01906107,
    chunk_regular_var: float = 0.81397036,
    is_chunk_x0_mean_var_regular: bool = False,
    chunk_mean_var_regular_x0_weight: float = 1.0,
    chunk_regular_x0_mean: float = -0.01578601,
    chunk_regular_x0_var: float = 0.29913200,
    # For GAN
    is_use_gan: bool = False,
    is_gan_low_vram_mode: bool = False,
    gan_prompt_embeds: torch.Tensor = None,
    gan_g_weight: float = 1e-2,
    # For Reward
    is_use_reward_model: bool = False,
    reward_model=None,
    reward_weight_vq: float = 1.0,
    reward_weight_mq: float = 1.0,
    reward_weight_ta: float = 1.0,
    reward_texts: Optional[List[str]] = None,
    # For Decouple DMD
    is_decouple_dmd: bool = False,
    decouple_ca_start_step: int = 2000,
    decouple_ca_end_step: int = 3000,
    # For Dynamic Timestep
    is_forcing_low_renoise: bool = False,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    operation_trace=None,
    trace_prefix: str = None,
    remote_score_gradient_service=None,
):
    if remote_score_gradient_service is not None and dmd_is_low_vram_mode:
        raise ValueError(
            "remote Score gradient service owns placement; whole-model low-VRAM swapping is forbidden"
        )
    if is_use_gt_history:
        assert gan_prompt_embeds is not None
        prompt_embeds = gan_prompt_embeds

    can_wave_swap = (
        dmd_is_low_vram_mode
        and vram_manager.wave_swap_enabled
        and not is_smoothness_loss
        and not is_dmd_vae_decode
        and not is_use_reward_model
    )
    if dmd_is_low_vram_mode:
        if can_wave_swap:
            vram_manager.swap_models(
                incoming_model=transformer,
                outgoing_model=real_fake_score_model,
                device=accelerator.device,
            )
        else:
            vram_manager.move_to_cpu(real_fake_score_model)
            if (is_smoothness_loss or is_dmd_vae_decode) and vae is not None:
                vram_manager.move_to_cpu(vae)
            if is_use_reward_model:
                vram_manager.move_to_cpu(reward_model.model)
            vram_manager.move_to_gpu(transformer, accelerator.device)

    init_pyramid_stage_flag = None
    if is_multi_pyramid_stage_backward_simulated:
        assert is_multi_pyramid_stage_backward_simulated, (
            "use_dynamic_shifting must be True when is_multi_pyramid_stage_backward_simulated is True"
        )
        init_pyramid_stage_flag = random.randint(0, stage2_num_stages - 1)

    # Prepare all sigmas and timesteps
    sigmas = torch.linspace(
        1.0, 1.0 / num_train_timestep, num_train_timestep, device=accelerator.device, dtype=torch.float64
    )
    if use_dynamic_shifting:
        base_height, base_width = noise.shape[-2:]
        if is_multi_pyramid_stage_backward_simulated:
            divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
            temp_height, temp_width = base_height // divisor, base_width // divisor
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, temp_height, temp_width)
        else:
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, base_height, base_width)

        sigmas, timestep_shift = apply_schedule_shift(
            sigmas,
            temp_tenosr,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
            time_shift_type=time_shift_type,
            return_mu=True,
        )
    elif timestep_shift > 1:
        sigmas = timestep_shift * sigmas / (1 + (timestep_shift - 1) * sigmas)
    timesteps = sigmas * num_train_timestep

    gt_all_data = None
    if is_use_gt_history:
        latent_window_size = noise.shape[2]
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=gt_history_latents,
            target_latents=gt_target_latents,
            x0_latents=gt_x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=args.training_config.is_random_drop,
            random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
            random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
            random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
            is_keep_x0=True,
            dtype=noise.dtype,
            device=accelerator.device,
        )
        history_latents = torch.cat(
            [latents_history_long, latents_history_mid, latents_history_short[:, :, 1:]], dim=2
        )
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )
        if operation_trace is not None:
            (
                history_latents,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
            ) = replay_operation_trace_prepared_history(
                operation_trace,
                trace_prefix=trace_prefix,
                device=accelerator.device,
                dtype=noise.dtype,
            )

        gt_all_data = (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        )
        assert num_critic_input_frames == latent_window_size
        assert num_rollout_sections == 1
        assert not is_smoothness_loss and not is_dmd_vae_decode

    # Step 1: Unroll generator to obtain fake videos
    with _helios_stage_range(global_step, "generator", "train_rollout", grad="enabled"):
        (
            pred_image_or_video,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
            consistency_align_loss,
        ) = run_generator(
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            scheduler=scheduler,
            noise=noise,
            prompt_embeds=prompt_embeds,
            # For VRAM manager
            dmd_is_low_vram_mode=dmd_is_low_vram_mode,
            # For Stage 1
            is_keep_x0=is_keep_x0,
            history_sizes=history_sizes,
            # For Stage 2
            is_enable_stage2=is_enable_stage2,
            stage2_num_stages=stage2_num_stages,
            stage2_num_inference_steps_list=stage2_num_inference_steps_list,
            # For DMD Main
            denoising_step_list=denoising_step_list,
            last_step_only=last_step_only,
            last_section_grad_only=last_section_grad_only,
            return_sim_step=return_sim_step,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            num_critic_input_frames=num_critic_input_frames,
            num_rollout_sections=num_rollout_sections,
            is_skip_first_section=is_skip_first_section,
            is_amplify_first_chunk=is_amplify_first_chunk,
            # Easy Anti-Drifting
            is_corrupt_history_latents=is_corrupt_history_latents,
            is_add_saturation=is_add_saturation,
            # GT History
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,
            # For VAE Re-Encode
            is_dmd_vae_decode=is_dmd_vae_decode,
            # For Multi Stage Backward Simulated
            is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
            init_pyramid_stage_flag=init_pyramid_stage_flag,
            # Consistency Align
            is_consistency_align=is_consistency_align,
            # KV Cache
            use_kv_cache=use_kv_cache,
            operation_trace=operation_trace,
            trace_prefix=trace_prefix,
        )

    if remote_score_gradient_service is not None:
        # Persistent 8G+8Score execution changes only the physical location of
        # official DMD/GAN-G.  All other optional Generator objectives are
        # deliberately rejected here so this seam cannot silently redefine the
        # frozen GAN Full5 workload.
        if any(
            (
                is_smoothness_loss,
                is_dmd_vae_decode,
                is_use_reward_model,
                is_consistency_align,
                is_mean_var_regular,
                is_x0_mean_var_regular,
                is_chunk_mean_var_regular,
                is_chunk_x0_mean_var_regular,
                is_decouple_dmd,
            )
        ):
            raise ValueError(
                "remote Score gradient service supports only the frozen DMD+GAN-G workload"
            )
        boundary = remote_score_gradient_service(
            generated_latent=pred_image_or_video,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            gt_all_data=gt_all_data,
            sigmas=sigmas,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            operation_trace=operation_trace,
            trace_prefix=trace_prefix,
            global_step=global_step,
        )
        gradient = getattr(boundary, "gradient", None)
        if gradient is None and isinstance(boundary, dict):
            gradient = boundary.get("gradient")
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("remote Score gradient service returned no tensor gradient")
        if tuple(gradient.shape) != tuple(pred_image_or_video.shape):
            raise ValueError("remote Score gradient shape differs from Generator output")
        gradient = gradient.to(
            device=pred_image_or_video.device,
            dtype=pred_image_or_video.dtype,
        ).detach()
        # The returned tensor is already d(DMD + gan_g_weight * GAN-G)/dx.
        # A sum surrogate preserves that exact latent gradient; DeepSpeed owns
        # DP8/GAS2 normalization and the single optimizer commit outside here.
        surrogate = (pred_image_or_video.float() * gradient.float()).sum()
        if not surrogate.requires_grad or surrogate.grad_fn is None:
            raise AssertionError("remote Score surrogate lost Generator autograd")
        logs = dict(getattr(boundary, "logs", {}) or {})
        for name in (
            "total_loss",
            "dmd_loss",
            "gan_g_loss_raw",
            "gan_g_loss_weighted",
        ):
            value = getattr(boundary, name, None)
            if value is not None:
                logs[name] = float(value)
        logs["remote_score_gradient"] = True
        return surrogate, logs

    if dmd_is_low_vram_mode and not can_wave_swap:
        vram_manager.move_to_cpu(transformer, offload_grad=dmd_is_offload_grad)

    # Step 2: Compute the Smoothness loss
    selected_frames = None
    smooth_count = 0
    smoothness_loss = torch.tensor(0.0, device=pred_image_or_video.device)
    if is_smoothness_loss or is_dmd_vae_decode:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(vae, accelerator.device)
        else:
            vae.to(accelerator.device)
        vae.requires_grad_(False)
        vae.eval()

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        latent_window_size = noise.shape[2]
        assert pred_image_or_video.shape[2] % latent_window_size == 0
        num_sections = math.ceil(pred_image_or_video.shape[2] / latent_window_size)

        total_frame_latent = []
        prev_last_frame_latent = None
        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, pred_image_or_video.shape[2])
            cur_section = pred_image_or_video[:, :, start_idx:end_idx, :, :]

            if is_smoothness_loss:
                cur_first_frame_latent = cur_section[:, :, :1, :, :].clone()

                if prev_last_frame_latent is not None:
                    prev_lat = prev_last_frame_latent.double()
                    cur_lat = cur_first_frame_latent.double()

                    mse_loss = 0.5 * F.mse_loss(prev_lat, cur_lat, reduction="mean")
                    smoothness_loss += mse_loss
                    smooth_count += 1

            with torch.no_grad():
                decoded = vae.decode(cur_section.to(vae.dtype) / latents_std + latents_mean, return_dict=False)[0]

            if is_dmd_vae_decode:
                total_frame_latent.append(decoded)

            if is_smoothness_loss:
                with torch.no_grad():
                    prev_last_frame_latent = (
                        vae.encode(decoded[:, :, -1:, :, :].to(vae.dtype)).latent_dist.sample() - latents_mean
                    ) * latents_std

        del prev_last_frame_latent
        free_memory()

        if is_dmd_vae_decode:
            num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
            combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)

            begin_flag = random.random() < 0.5
            if begin_flag:
                selected_frames = combined_frames[:, :, :num_rgb_frames, :, :]
            else:
                selected_frames = combined_frames[:, :, -num_rgb_frames:, :, :]

            with torch.no_grad():
                reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
                reconstructed_latent = (reconstructed_latent - latents_mean) * latents_std

            # Straight-Through Estimator
            if begin_flag:
                pred_image_or_video = (
                    pred_image_or_video[:, :, :num_critic_input_frames, :, :]
                    + (reconstructed_latent - pred_image_or_video[:, :, :num_critic_input_frames, :, :]).detach()
                )
            else:
                pred_image_or_video = (
                    pred_image_or_video[:, :, -num_critic_input_frames:, :, :]
                    + (reconstructed_latent - pred_image_or_video[:, :, -num_critic_input_frames:, :, :]).detach()
                )

        if smooth_count > 1:
            smoothness_loss = smoothness_loss / smooth_count

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(vae)

    # Step 3: Compute the Reward score
    if is_use_reward_model:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(reward_model.model, accelerator.device)

        processed_frames = ((selected_frames + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 1, 3, 4)
        processed_frames = list(processed_frames)

        with torch.no_grad():
            reward = reward_model.reward(
                videos=processed_frames,
                prompts=reward_texts,
                use_norm=True,
                return_batch_score=True,
                device=accelerator.device,
                dtype=torch.float32,
            )

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(reward_model.model)

        processed_frames = None
        del processed_frames

    # Step 4: Compute the DMD loss
    if dmd_is_low_vram_mode:
        if can_wave_swap:
            vram_manager.swap_models(
                incoming_model=real_fake_score_model,
                outgoing_model=transformer,
                device=accelerator.device,
                outgoing_offload_grad=dmd_is_offload_grad,
            )
        else:
            vram_manager.move_to_gpu(real_fake_score_model, accelerator.device)

    with _helios_stage_range(global_step, "score", "distribution_matching"):
        dmd_loss, ca_dmd_loss, dm_dmd_loss, gan_G_loss, dmd_log_dict = compute_distribution_matching_loss(
            accelerator,
            scheduler,
            real_fake_score_model,
            image_or_video=pred_image_or_video,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # For VRAM manager
            dmd_is_low_vram_mode=dmd_is_low_vram_mode,
            vram_manager=vram_manager,
            is_gan_low_vram_mode=is_gan_low_vram_mode,
            # For Stage 2
            is_enable_stage2=is_enable_stage2,
            # For DMD Main
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            ts_schedule=ts_schedule,
            ts_schedule_max=ts_schedule_max,
            min_score_timestep=min_score_timestep,
            num_train_timestep=num_train_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            fake_guidance_scale=fake_guidance_scale,
            real_guidance_scale=real_guidance_scale,
            # For GT History
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,
            # For GAN
            is_use_gan=is_use_gan,
            # For Decouple DMD
            is_decouple_dmd=is_decouple_dmd,
            decouple_ca_start_step=decouple_ca_start_step,
            decouple_ca_end_step=decouple_ca_end_step,
            # For Dynamic Timestep
            is_forcing_low_renoise=is_forcing_low_renoise,
            dynamic_alpha=dynamic_alpha,
            dynamic_beta=dynamic_beta,
            dynamic_sample_type=dynamic_sample_type,
            global_step=global_step,
            dynamic_step=dynamic_step,
            operation_trace=operation_trace,
            trace_prefix=trace_prefix,
        )

    if dmd_is_low_vram_mode:
        if vram_manager.wave_swap_enabled:
            vram_manager.swap_models(
                incoming_model=transformer,
                outgoing_model=real_fake_score_model,
                device=accelerator.device,
                incoming_load_grad=dmd_is_offload_grad,
            )
        else:
            vram_manager.move_to_cpu(real_fake_score_model)
            vram_manager.move_to_gpu(transformer, accelerator.device, load_grad=dmd_is_offload_grad)

    if is_smoothness_loss or is_use_gan or is_use_reward_model or is_consistency_align:
        dmd_log_dict["dmd_loss_raw"] = dmd_loss.detach().item()

    if is_consistency_align:
        if consistency_align_loss != 0:
            assert consistency_align_loss.requires_grad, (
                f"Consistentcy Align loss should have gradient! Got {consistency_align_loss.requires_grad}"
            )
            assert consistency_align_loss.grad_fn is not None, "Consistentcy Align loss should have grad_fn!"
        consistency_align_loss = consistency_align_loss * consistentcy_align_weight
        dmd_log_dict["consistency_align_loss"] = consistency_align_loss.detach().item()
        dmd_loss = dmd_loss + consistency_align_loss

    if is_smoothness_loss:
        assert smoothness_loss.requires_grad, (
            f"Smoothness loss should have gradient! Got {smoothness_loss.requires_grad}"
        )
        assert smoothness_loss.grad_fn is not None, "Smoothness loss should have grad_fn!"
        smoothness_loss = smoothness_loss * smoothness_loss_weight
        dmd_log_dict["smoothness_loss"] = smoothness_loss.detach().item()
        dmd_loss = dmd_loss + smoothness_loss

    if is_mean_var_regular:
        latent_window_size = noise.shape[2]
        dims = list(range(1, pred_image_or_video.ndim))

        pred_mean = pred_image_or_video.mean(dim=dims)
        pred_variance = pred_image_or_video.var(dim=dims, unbiased=False)
        pred_variance = pred_variance.clamp(min=1e-6)

        kl_mean_var_loss = (
            0.5
            * (
                pred_variance / regular_var
                + (pred_mean - regular_mean) ** 2 / regular_var
                - 1.0
                - torch.log(pred_variance / regular_var)
            ).mean()
        )

        kl_mean_var_loss = kl_mean_var_loss * mean_var_regular_weight
        dmd_log_dict["kl_mean_var_loss"] = kl_mean_var_loss.detach().item()
        dmd_log_dict["pred_mean_avg"] = pred_mean.mean().detach().item()
        dmd_log_dict["pred_var_avg"] = pred_variance.mean().detach().item()

        if is_x0_mean_var_regular:
            x0 = pred_image_or_video[:, :, :1, :, :]
            pred_x0_mean = x0.mean(dim=dims)
            pred_x0_variance = x0.var(dim=dims, unbiased=False)
            pred_x0_variance = pred_x0_variance.clamp(min=1e-6)

            kl_mean_var_x0_loss = (
                0.5
                * (
                    pred_x0_variance / regular_x0_var
                    + (pred_x0_mean - regular_x0_mean) ** 2 / regular_x0_var
                    - 1.0
                    - torch.log(pred_x0_variance / regular_x0_var)
                ).mean()
            )

        if is_x0_mean_var_regular:
            kl_mean_var_x0_loss = kl_mean_var_x0_loss * mean_var_regular_x0_weight
            dmd_log_dict["kl_mean_var_x0_loss"] = kl_mean_var_x0_loss.detach().item()
            dmd_log_dict["pred_x0_mean_avg"] = pred_x0_mean.mean().detach().item()
            dmd_log_dict["pred_x0_var_avg"] = pred_x0_variance.mean().detach().item()
            kl_mean_var_loss = 0.7 * kl_mean_var_loss + 0.3 * kl_mean_var_x0_loss

        dmd_loss = dmd_loss + kl_mean_var_loss
        assert kl_mean_var_loss != 0, "kl_mean_var_loss should be non-zero when there are valid sections"
        assert kl_mean_var_loss.requires_grad, (
            f"kl_mean_var_loss should have gradient! Got {kl_mean_var_loss.requires_grad}"
        )
        assert kl_mean_var_loss.grad_fn is not None, "kl_mean_var_loss should have grad_fn!"

    if is_chunk_mean_var_regular:
        latent_window_size = noise.shape[2]
        num_sections = math.ceil(pred_image_or_video.shape[2] / latent_window_size)

        kl_chunk_mean_var_loss = 0
        total_chunk_pred_mean = 0
        total_chunk_pred_var = 0
        valid_sections_count = 0

        if is_chunk_x0_mean_var_regular:
            kl_chunk_mean_var_x0_loss = 0
            total_pred_x0_mean = 0
            total_pred_x0_var = 0

        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, pred_image_or_video.shape[2])

            cur_section = pred_image_or_video[:, :, start_idx:end_idx, :, :]

            if cur_section.shape[2] >= latent_window_size:
                dims = list(range(1, cur_section.ndim))
                pred_mean = cur_section.mean(dim=dims)
                pred_variance = cur_section.var(dim=dims, unbiased=False)
                pred_variance = pred_variance.clamp(min=1e-6)

                section_kl_loss = 0.5 * (
                    pred_variance / chunk_regular_var
                    + (pred_mean - chunk_regular_mean) ** 2 / chunk_regular_var
                    - 1.0
                    - torch.log(pred_variance / chunk_regular_var)
                )
                kl_chunk_mean_var_loss += section_kl_loss.mean()
                total_chunk_pred_mean += pred_mean.mean().item()
                total_chunk_pred_var += pred_variance.mean().item()
                valid_sections_count += 1

            if is_chunk_x0_mean_var_regular:
                x0_cur_section = cur_section[:, :, :1, :, :]
                pred_x0_mean = x0_cur_section.mean(dim=dims)
                pred_x0_variance = x0_cur_section.var(dim=dims, unbiased=False)
                pred_x0_variance = pred_x0_variance.clamp(min=1e-6)

                section_x0_kl_loss = 0.5 * (
                    pred_x0_variance / chunk_regular_x0_var
                    + (pred_x0_mean - chunk_regular_x0_mean) ** 2 / chunk_regular_x0_var
                    - 1.0
                    - torch.log(pred_x0_variance / chunk_regular_x0_var)
                )
                kl_chunk_mean_var_x0_loss += section_x0_kl_loss.mean()
                total_pred_x0_mean += pred_x0_mean.mean().item()
                total_pred_x0_var += pred_x0_variance.mean().item()

        if valid_sections_count > 0:
            kl_chunk_mean_var_loss = (kl_chunk_mean_var_loss / valid_sections_count) * chunk_mean_var_regular_weight
            dmd_log_dict["kl_chunk_mean_var_loss"] = kl_chunk_mean_var_loss.detach().item()
            dmd_log_dict["pred_chunk_mean_avg"] = total_chunk_pred_mean / valid_sections_count
            dmd_log_dict["pred_chunk_var_avg"] = total_chunk_pred_var / valid_sections_count
        else:
            kl_chunk_mean_var_loss = 0
            dmd_log_dict["kl_chunk_mean_var_loss"] = 0
            dmd_log_dict["pred_chunk_mean_avg"] = 0
            dmd_log_dict["pred_chunk_var_avg"] = 0

        if is_chunk_x0_mean_var_regular:
            kl_chunk_mean_var_x0_loss = (kl_chunk_mean_var_x0_loss / num_sections) * chunk_mean_var_regular_x0_weight

            if valid_sections_count > 0:
                kl_chunk_mean_var_loss = 0.7 * kl_chunk_mean_var_loss + 0.3 * kl_chunk_mean_var_x0_loss
            else:
                kl_chunk_mean_var_loss = kl_chunk_mean_var_x0_loss

            dmd_log_dict["kl_chunk_mean_var_x0_loss"] = kl_chunk_mean_var_x0_loss.detach().item()
            dmd_log_dict["pred_chunk_x0_mean_avg"] = total_pred_x0_mean / num_sections
            dmd_log_dict["pred_chunk_x0_var_avg"] = total_pred_x0_var / num_sections

        dmd_loss = dmd_loss + kl_chunk_mean_var_loss
        assert kl_chunk_mean_var_loss != 0, "kl_chunk_mean_var_loss should be non-zero when there are valid sections"
        assert kl_chunk_mean_var_loss.requires_grad, (
            f"kl_chunk_mean_var_loss should have gradient! Got {kl_chunk_mean_var_loss.requires_grad}"
        )
        assert kl_chunk_mean_var_loss.grad_fn is not None, "kl_chunk_mean_var_loss should have grad_fn!"

    if is_use_gan:
        assert gan_G_loss.requires_grad, f"GAN G loss should have gradient! Got {gan_G_loss.requires_grad}"
        assert gan_G_loss.grad_fn is not None, "GAN G loss should have grad_fn!"
        gan_G_loss = gan_G_loss * gan_g_weight
        dmd_log_dict["gan_G_loss"] = gan_G_loss.detach().item()
        dmd_loss = dmd_loss + gan_G_loss

    if is_use_reward_model:
        reward_scores = []
        if reward_weight_vq != 0:
            reward_score_vq = reward_weight_vq * reward["VQ"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_vq)
            dmd_log_dict["reward_score_vq"] = reward["VQ"].detach().mean().item()
            assert not reward_score_vq.requires_grad, (
                f"Reward Score VQ should not have gradient! Got {reward_score_vq.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_vq"] = 0

        if reward_weight_mq != 0:
            reward_score_mq = reward_weight_mq * reward["MQ"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_mq)
            dmd_log_dict["reward_score_mq"] = reward["MQ"].detach().mean().item()
            assert not reward_score_mq.requires_grad, (
                f"Reward Score MQ should not have gradient! Got {reward_score_mq.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_mq"] = 0

        if reward_weight_ta != 0:
            reward_score_ta = reward_weight_ta * reward["TA"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_ta)
            dmd_log_dict["reward_score_ta"] = reward["TA"].detach().mean().item()
            assert not reward_score_ta.requires_grad, (
                f"Reward Score TA should not have gradient! Got {reward_score_ta.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_ta"] = 0

        reward_score = torch.stack(reward_scores).mean()
        reward_score = torch.exp(reward_score)

        dmd_loss = dmd_loss * reward_score

    if is_decouple_dmd:
        assert ca_dmd_loss.requires_grad, f"CA DMD loss should have gradient! Got {ca_dmd_loss.requires_grad}"
        assert dm_dmd_loss.requires_grad, f"DM DMD loss should have gradient! Got {dm_dmd_loss.requires_grad}"
        assert ca_dmd_loss.grad_fn is not None, "CA DMD loss should have grad_fn!"
        assert dm_dmd_loss.grad_fn is not None, "DM DMD loss should have grad_fn!"
        dmd_log_dict["ca_dmd_loss"] = ca_dmd_loss.detach().item()
        dmd_log_dict["dm_dmd_loss"] = dm_dmd_loss.detach().item()

    assert dmd_loss.requires_grad, f"Final DMD loss should have gradient! Got {dmd_loss.requires_grad}"
    assert dmd_loss.grad_fn is not None, "Final DMD loss should have grad_fn!"

    return dmd_loss, dmd_log_dict


# ======================================== Critic Loss ========================================


def _critic_loss(
    args,
    critic_accelerator,
    fake_score_model,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    # For VRAM manager
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    is_gan_low_vram_mode: bool = False,
    # For Stage 1
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    # For Stage 2
    is_enable_stage2: bool = False,
    stage2_num_stages: int = None,
    stage2_num_inference_steps_list: list = None,
    # For DMD Main
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,
    num_train_timestep: int = 1000,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    # For Easy Anti-Drifting
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    # For GT History
    is_use_gt_history: bool = False,
    gt_history_latents: torch.Tensor = None,
    gt_target_latents: torch.Tensor = None,
    gt_x0_latents: torch.Tensor = None,
    # For VAE Re-Encode
    vae=None,
    is_dmd_vae_decode: bool = False,
    # For Multi Stage Backward Simulated
    is_multi_pyramid_stage_backward_simulated: bool = False,
    # For KV Cache
    use_kv_cache: bool = True,
    # For GAN
    is_use_gan: bool = False,
    is_separate_gan_grad: bool = False,
    gan_base_critic_trainable_params: dict = None,
    gan_extra_critic_trainable_params: dict = None,
    gan_vae_latents: torch.Tensor = None,
    gan_prompt_embeds: torch.Tensor = None,
    gan_d_weight: float = 1e-2,
    aprox_r1: bool = False,
    aprox_r2: bool = False,
    r1_weight: float = 0.0,
    r2_weight: float = 0.0,
    r1_sigma: float = 0.01,
    r2_sigma: float = 0.01,
    # For Dynamic Timestep
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    operation_trace=None,
    trace_prefix: str = None,
    phase_resident_generator_rollout: bool = False,
    critic_phase_offload_generator_lp_state: bool = False,
    precomputed_generated_image_or_video: torch.Tensor = None,
    pipeline_micro_wave: int = None,
):
    if pipeline_micro_wave not in (None, 0, 1):
        raise ValueError("persistent GAN pipeline micro-wave must be 0 or 1")
    if pipeline_micro_wave is not None and precomputed_generated_image_or_video is None:
        raise ValueError("pipeline Critic micro-wave requires a precomputed Generator rollout")
    if critic_phase_offload_generator_lp_state and (
        not dmd_is_low_vram_mode or vram_manager is None
    ):
        raise ValueError(
            "Critic-phase Generator LP-state offload requires the low-VRAM manager"
        )
    if phase_resident_generator_rollout and (
        not dmd_is_low_vram_mode or vram_manager is None
    ):
        raise ValueError(
            "phase-resident Critic Generator requires the low-VRAM manager"
        )
    if is_use_gt_history:
        assert gan_prompt_embeds is not None
        prompt_embeds = gan_prompt_embeds

    can_wave_swap = (
        dmd_is_low_vram_mode
        and vram_manager.wave_swap_enabled
        and not is_dmd_vae_decode
    )
    if dmd_is_low_vram_mode:
        if can_wave_swap:
            # A GAN-active Critic-only step above ema_start_step leaves Score on
            # GPU because EMA is skipped.  The next Critic rollout must therefore
            # restore Generator before its first patch-embedding call.  Earlier
            # workloads hid this stale placement because bootstrap EMA swapped
            # Generator back after every step.
            if not vram_manager.compute_state_is_on_device(
                transformer, critic_accelerator.device
            ):
                vram_manager.swap_models(
                    incoming_model=transformer,
                    outgoing_model=fake_score_model,
                    device=critic_accelerator.device,
                )
        else:
            vram_manager.move_to_cpu(fake_score_model)
            if is_dmd_vae_decode:
                vram_manager.move_to_cpu(vae)
            vram_manager.move_to_gpu(transformer, critic_accelerator.device)

    init_pyramid_stage_flag = None
    if is_multi_pyramid_stage_backward_simulated:
        assert is_multi_pyramid_stage_backward_simulated, (
            "use_dynamic_shifting must be True when is_multi_pyramid_stage_backward_simulated is True"
        )
        init_pyramid_stage_flag = random.randint(0, stage2_num_stages - 1)

    # Prepare all sigmas and timesteps
    sigmas = torch.linspace(
        1.0, 1.0 / num_train_timestep, num_train_timestep, device=critic_accelerator.device, dtype=torch.float64
    )
    if use_dynamic_shifting:
        base_height, base_width = noise.shape[-2:]
        if is_multi_pyramid_stage_backward_simulated:
            divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
            temp_height, temp_width = base_height // divisor, base_width // divisor
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, temp_height, temp_width)
        else:
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, base_height, base_width)

        sigmas, timestep_shift = apply_schedule_shift(
            sigmas,
            temp_tenosr,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
            time_shift_type=time_shift_type,
            return_mu=True,
        )
    elif timestep_shift > 1:
        sigmas = timestep_shift * sigmas / (1 + (timestep_shift - 1) * sigmas)
    timesteps = sigmas * num_train_timestep

    noise = (
        operation_trace.tensor(f"{trace_prefix}/entry_noise").to(critic_accelerator.device, dtype=noise.dtype)
        if operation_trace is not None
        else torch.randn(noise.shape, device=critic_accelerator.device, dtype=noise.dtype)
    )
    batch_size = noise.shape[0]

    if is_use_gt_history:
        latent_window_size = noise.shape[2]
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=gt_history_latents,
            target_latents=gt_target_latents,
            x0_latents=gt_x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=args.training_config.is_random_drop,
            random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
            random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
            random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
            is_keep_x0=True,
            dtype=noise.dtype,
            device=critic_accelerator.device,
        )
        history_latents = torch.cat(
            [latents_history_long, latents_history_mid, latents_history_short[:, :, 1:]], dim=2
        )
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )
        if operation_trace is not None:
            (
                history_latents,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
            ) = replay_operation_trace_prepared_history(
                operation_trace,
                trace_prefix=trace_prefix,
                device=critic_accelerator.device,
                dtype=noise.dtype,
            )

        gt_all_data = (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        )
        assert num_critic_input_frames == latent_window_size
        assert num_rollout_sections == 1
        assert not is_dmd_vae_decode
    else:
        gt_all_data = None
        indices_hidden_states = None
        indices_latents_history_short = None
        indices_latents_history_mid = None
        indices_latents_history_long = None
        latents_history_short = None
        latents_history_mid = None
        latents_history_long = None

    # Step 1: Run Generator, or consume the exact paired-rank rollout produced
    # by the persistent Generator pool.  The latter is a placement seam only;
    # all Critic denoising/GAN-D/R1 math below remains the upstream path.
    if precomputed_generated_image_or_video is None:
        with torch.no_grad():
            resident_context = (
                vram_manager.phase_resident_frozen_blocks(
                    transformer,
                    phase=f"critic_rollout_step_{global_step}",
                )
                if phase_resident_generator_rollout
                else contextlib.nullcontext()
            )
            with resident_context:
                with _helios_stage_range(global_step, "generator", "critic_rollout", grad="disabled"):
                    generated_image_or_video, _, denoised_timestep_from, denoised_timestep_to, _ = run_generator(
                        args=args,
                        accelerator=critic_accelerator,
                        transformer=transformer,
                        scheduler=scheduler,
                        noise=noise,
                        prompt_embeds=prompt_embeds,
                        # For VRAM manager
                        dmd_is_low_vram_mode=dmd_is_low_vram_mode,
                        # For Stage 1
                        is_keep_x0=is_keep_x0,
                        history_sizes=history_sizes,
                        # For Stage 2
                        is_enable_stage2=is_enable_stage2,
                        stage2_num_stages=stage2_num_stages,
                        stage2_num_inference_steps_list=stage2_num_inference_steps_list,
                        # For DMD Main
                        denoising_step_list=denoising_step_list,
                        last_step_only=last_step_only,
                        last_section_grad_only=last_section_grad_only,
                        return_sim_step=return_sim_step,
                        sigmas=sigmas,
                        timesteps=timesteps,
                        timestep_shift=timestep_shift,
                        use_dynamic_shifting=use_dynamic_shifting,
                        time_shift_type=time_shift_type,
                        num_critic_input_frames=num_critic_input_frames,
                        num_rollout_sections=num_rollout_sections,
                        is_skip_first_section=is_skip_first_section,
                        is_amplify_first_chunk=is_amplify_first_chunk,
                        # Easy Anti-Drifting
                        is_corrupt_history_latents=is_corrupt_history_latents,
                        is_add_saturation=is_add_saturation,
                        # GT History
                        is_use_gt_history=is_use_gt_history,
                        gt_all_data=gt_all_data,
                        # For VAE Re-Encode
                        is_dmd_vae_decode=is_dmd_vae_decode,
                        # For Multi Stage Backward Simulated
                        is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
                        init_pyramid_stage_flag=init_pyramid_stage_flag,
                        # KV Cache
                        use_kv_cache=use_kv_cache,
                        operation_trace=operation_trace,
                        trace_prefix=trace_prefix,
                    )
    else:
        if is_dmd_vae_decode:
            raise ValueError("precomputed Critic rollout forbids VAE re-encode")
        generated_image_or_video = precomputed_generated_image_or_video.detach().to(
            critic_accelerator.device,
            dtype=noise.dtype,
        )
        if tuple(generated_image_or_video.shape) != tuple(noise.shape):
            raise ValueError("precomputed Critic rollout violates official latent shape")
        denoised_timestep_from = None
        denoised_timestep_to = None

    if dmd_is_low_vram_mode and not can_wave_swap:
        vram_manager.move_to_cpu(
            transformer,
            offload_engine_lp_state=critic_phase_offload_generator_lp_state,
        )

    # Step 2: Compute the Smoothness loss
    if is_dmd_vae_decode:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(vae, critic_accelerator.device)
        else:
            vae.to(critic_accelerator.device)
        vae.requires_grad_(False)
        vae.eval()

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        latent_window_size = noise.shape[2]
        assert generated_image_or_video.shape[2] % latent_window_size == 0
        num_sections = math.ceil(generated_image_or_video.shape[2] / latent_window_size)
        total_frame_latent = []
        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, generated_image_or_video.shape[2])
            cur_section = generated_image_or_video[:, :, start_idx:end_idx, :, :]

            with torch.no_grad():
                decoded = vae.decode(cur_section.to(vae.dtype) / latents_std + latents_mean, return_dict=False)[0]
            total_frame_latent.append(decoded)

        num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
        combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)

        max_start_idx = combined_frames.shape[2] - num_rgb_frames
        start_idx = random.randint(0, max_start_idx)
        selected_frames = combined_frames[:, :, start_idx : start_idx + num_rgb_frames, :, :]

        with torch.no_grad():
            reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
            reconstructed_latent = (reconstructed_latent - latents_mean) * latents_std

        generated_image_or_video = reconstructed_latent

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(vae)

        free_memory()

    # Step 3: Compute the fake prediction
    if dmd_is_low_vram_mode:
        if can_wave_swap:
            vram_manager.swap_models(
                incoming_model=fake_score_model,
                outgoing_model=transformer,
                device=critic_accelerator.device,
                outgoing_offload_engine_lp_state=(
                    critic_phase_offload_generator_lp_state
                ),
            )
        else:
            vram_manager.move_to_gpu(fake_score_model, critic_accelerator.device)

    min_timestep = denoised_timestep_to if ts_schedule and denoised_timestep_to is not None else min_score_timestep
    max_timestep = (
        denoised_timestep_from if ts_schedule_max and denoised_timestep_from is not None else num_train_timestep
    )
    min_step = int(0.02 * num_train_timestep)
    max_step = int(0.98 * num_train_timestep)

    if operation_trace is not None:
        critic_timestep = operation_trace.tensor(f"{trace_prefix}/critic_timestep").to(
            critic_accelerator.device
        )
        critic_noise = operation_trace.tensor(f"{trace_prefix}/critic_noise").to(
            critic_accelerator.device, dtype=noise.dtype
        )
    else:
        critic_timestep = sample_dynamic_timestep(
            B=batch_size,
            num_train_timestep=num_train_timestep,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            min_step=min_step,
            max_step=max_step,
            timestep_shift=timestep_shift,
            dynamic_alpha=dynamic_alpha,
            dynamic_beta=dynamic_beta,
            dynamic_sample_type=dynamic_sample_type,
            global_step=global_step,
            dynamic_step=dynamic_step,
            device=critic_accelerator.device,
        )
        critic_noise = torch.randn_like(
            generated_image_or_video, device=critic_accelerator.device, dtype=noise.dtype
        )
    noisy_fake_latent = add_noise(
        generated_image_or_video,
        critic_noise,
        critic_timestep,
        sigmas,
        timesteps,
    )

    gan_D_loss = torch.tensor(0.0)
    r1_loss = torch.tensor(0.0)
    r2_loss = torch.tensor(0.0)
    if is_use_gan:
        if gan_prompt_embeds is None:
            gan_prompt_embeds = prompt_embeds

        if is_gan_low_vram_mode:
            # Under accelerate's DeepSpeed wrapper, every accelerator.backward()
            # triggers engine.step() (i.e. optimizer.step + zero_grad). Calling
            # backward twice in this block would step the critic optimizer twice,
            # so denoising_loss and gan/r1/r2 losses would never reach the
            # optimizer in the same step. Drive the engine directly to defer the
            # step until both backwards have populated their gradients. Falls
            # back to accelerator.backward() when DeepSpeed isn't in use.
            critic_engine = getattr(
                getattr(critic_accelerator, "deepspeed_engine_wrapped", None),
                "engine",
                None,
            )
            if pipeline_micro_wave is not None and critic_engine is None:
                raise RuntimeError(
                    "pipeline GAS2 requires direct DeepSpeed boundary control"
                )

            if is_separate_gan_grad:
                for name, param in fake_score_model.named_parameters():
                    if name in gan_extra_critic_trainable_params:
                        param.requires_grad = False

            flow_fake_pred = fake_score_model(
                hidden_states=noisy_fake_latent,
                timestep=critic_timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
            )[0]
            denoising_loss = torch.mean(
                (flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2
            )

            assert denoising_loss.requires_grad, (
                f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
            )
            assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"
            if critic_engine is not None:
                critic_engine.set_gradient_accumulation_boundary(is_boundary=False)
                critic_engine.backward(denoising_loss)
            else:
                critic_accelerator.backward(denoising_loss)

            if is_separate_gan_grad:
                for name, param in fake_score_model.named_parameters():
                    if name in gan_base_critic_trainable_params:
                        param.requires_grad = False
                    if name in gan_extra_critic_trainable_params:
                        param.requires_grad = True

            noisy_real_latent = add_noise(
                gan_vae_latents,
                critic_noise,
                critic_timestep,
                sigmas,
                timesteps,
            )
            hidden_states_list = [noisy_fake_latent, noisy_real_latent]
            timestep_list = [critic_timestep, critic_timestep]
            embeds_list = [prompt_embeds, gan_prompt_embeds]

            if is_use_gt_history:
                indices_latents_list = [indices_hidden_states, indices_hidden_states]
                indices_latents_history_short_list = [indices_latents_history_short, indices_latents_history_short]
                indices_latents_history_mid_list = [indices_latents_history_mid, indices_latents_history_mid]
                indices_latents_history_long_list = [indices_latents_history_long, indices_latents_history_long]
                latents_history_short_list = [latents_history_short, latents_history_short]
                latents_history_mid_list = [latents_history_mid, latents_history_mid]
                latents_history_long_list = [latents_history_long, latents_history_long]

            # Prepare R1 perturbed input
            r1_enabled = r1_weight > 0.0
            if r1_enabled:
                noisy_real_latent_perturbed = noisy_real_latent.clone()
                epsilon_real = r1_sigma * (
                    operation_trace.tensor(f"{trace_prefix}/r1_epsilon").to(
                        critic_accelerator.device,
                        dtype=noisy_real_latent_perturbed.dtype,
                    )
                    if operation_trace is not None
                    else torch.randn_like(noisy_real_latent_perturbed)
                )
                noisy_real_latent_perturbed = noisy_real_latent_perturbed + epsilon_real
                hidden_states_list.append(noisy_real_latent_perturbed)
                timestep_list.append(critic_timestep)
                embeds_list.append(gan_prompt_embeds)
                if is_use_gt_history:
                    indices_latents_list.append(indices_hidden_states)
                    indices_latents_history_short_list.append(indices_latents_history_short)
                    indices_latents_history_mid_list.append(indices_latents_history_mid)
                    indices_latents_history_long_list.append(indices_latents_history_long)
                    latents_history_short_list.append(latents_history_short)
                    latents_history_mid_list.append(latents_history_mid)
                    latents_history_long_list.append(latents_history_long)

            # Prepare R2 perturbed input
            r2_enabled = r2_weight > 0.0
            if r2_enabled:
                noisy_fake_latent_perturbed = noisy_fake_latent.clone()
                epsilon_generated = r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
                noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epsilon_generated
                hidden_states_list.append(noisy_fake_latent_perturbed)
                timestep_list.append(critic_timestep)
                embeds_list.append(prompt_embeds)
                if is_use_gt_history:
                    indices_latents_list.append(indices_hidden_states)
                    indices_latents_history_short_list.append(indices_latents_history_short)
                    indices_latents_history_mid_list.append(indices_latents_history_mid)
                    indices_latents_history_long_list.append(indices_latents_history_long)
                    latents_history_short_list.append(latents_history_short)
                    latents_history_mid_list.append(latents_history_mid)
                    latents_history_long_list.append(latents_history_long)

            # Single forward pass for everything
            if operation_trace is not None:
                crop_roles = ["gan_d_fake_crop", "gan_d_real_crop"]
                if r1_enabled:
                    crop_roles.append("gan_d_r1_crop")
                if r2_enabled:
                    crop_roles.append("gan_d_r2_crop")
                crop_offsets = [
                    operation_trace.tensor(f"{trace_prefix}/{role}").to(
                        critic_accelerator.device
                    )
                    for role in crop_roles
                ]
            else:
                crop_offsets = [None] * len(hidden_states_list)
            hidden_states_list = [
                gan_crop_video_spatial(value, offsets=offsets)
                for value, offsets in zip(hidden_states_list, crop_offsets)
            ]
            _, all_logits = fake_score_model(
                hidden_states=torch.cat(hidden_states_list, dim=0),
                timestep=torch.cat(timestep_list, dim=0),
                encoder_hidden_states=torch.cat(embeds_list, dim=0),
                indices_hidden_states=torch.cat(indices_latents_list, dim=0) if is_use_gt_history else None,
                indices_latents_history_short=torch.cat(indices_latents_history_short_list, dim=0)
                if is_use_gt_history
                else None,
                indices_latents_history_mid=torch.cat(indices_latents_history_mid_list, dim=0)
                if is_use_gt_history
                else None,
                indices_latents_history_long=torch.cat(indices_latents_history_long_list, dim=0)
                if is_use_gt_history
                else None,
                latents_history_short=torch.cat(latents_history_short_list, dim=0) if is_use_gt_history else None,
                latents_history_mid=torch.cat(latents_history_mid_list, dim=0) if is_use_gt_history else None,
                latents_history_long=torch.cat(latents_history_long_list, dim=0) if is_use_gt_history else None,
                gan_mode=True,
                return_dict=False,
            )

            # Split outputs
            num_outputs = 2 + int(r1_enabled) + int(r2_enabled)
            logits_split = all_logits.chunk(num_outputs, dim=0)
            noisy_fake_logits = logits_split[0]
            noisy_real_logits = logits_split[1]

            idx = 2
            if r1_enabled:
                noisy_real_logit_perturbed = logits_split[idx]
                idx += 1
            if r2_enabled:
                noisy_fake_logit_perturbed = logits_split[idx]

            # Calculate GAN losses
            gan_D_fake_loss = cal_gan_loss(noisy_fake_logits, -1) * gan_d_weight
            gan_D_real_loss = cal_gan_loss(noisy_real_logits, 1) * gan_d_weight
            gan_D_loss = gan_D_fake_loss.detach() + gan_D_real_loss.detach()

            assert gan_D_fake_loss.requires_grad
            assert gan_D_fake_loss.grad_fn is not None
            assert gan_D_real_loss.requires_grad
            assert gan_D_real_loss.grad_fn is not None

            # Calculate regularization losses
            total_regular_loss = None

            if r1_enabled:
                if aprox_r1:
                    r1_loss = r1_weight * torch.nn.functional.mse_loss(
                        noisy_real_logits.float(), noisy_real_logit_perturbed.float(), reduction="mean"
                    )
                else:
                    r1_grad = (noisy_real_logit_perturbed.float() - noisy_real_logits.float()) / r1_sigma
                    r1_loss = r1_weight * torch.mean(r1_grad**2)
                total_regular_loss = r1_loss

            if r2_enabled:
                if aprox_r2:
                    r2_loss = r2_weight * torch.nn.functional.mse_loss(
                        noisy_fake_logits.float(), noisy_fake_logit_perturbed.float(), reduction="mean"
                    )
                else:
                    r2_grad = (noisy_fake_logit_perturbed.float() - noisy_fake_logits.float()) / r2_sigma
                    r2_loss = r2_weight * torch.mean(r2_grad**2)
                total_regular_loss = r2_loss if total_regular_loss is None else total_regular_loss + r2_loss

            if total_regular_loss is not None:
                assert total_regular_loss.requires_grad
                assert total_regular_loss.grad_fn is not None
                second_loss = total_regular_loss + gan_D_real_loss + gan_D_fake_loss
            else:
                second_loss = gan_D_real_loss + gan_D_fake_loss

            if critic_engine is not None:
                final_micro_wave = pipeline_micro_wave in (None, 1)
                critic_engine.set_gradient_accumulation_boundary(
                    is_boundary=final_micro_wave
                )
                critic_engine.backward(second_loss)
                if final_micro_wave:
                    critic_engine.step()
            else:
                critic_accelerator.backward(second_loss)

        else:
            raise NotImplementedError
            noisy_real_latent = add_noise(
                gan_vae_latents,
                critic_noise,
                critic_timestep,
                sigmas,
                timesteps,
            )
            flow_preds, noisy_logits = fake_score_model(
                hidden_states=torch.cat((noisy_fake_latent, noisy_real_latent), dim=0),
                timestep=torch.cat((critic_timestep, critic_timestep), dim=0),
                encoder_hidden_states=torch.cat((prompt_embeds, gan_prompt_embeds), dim=0),
                gan_mode=True,
                return_dict=False,
            )
            flow_fake_pred, flow_real_pred = flow_preds.chunk(2, dim=0)
            noisy_fake_logits, noisy_real_logits = noisy_logits.chunk(2, dim=0)

            denoising_loss = torch.mean(
                (flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2
            )
            gan_D_loss = (cal_gan_loss(noisy_fake_logits, -1) + cal_gan_loss(noisy_real_logits, 1)) * gan_d_weight

            assert denoising_loss.requires_grad, (
                f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
            )
            assert gan_D_loss.requires_grad, f"GAN D loss should have gradient! Got {gan_D_loss.requires_grad}"
            assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"
            assert gan_D_loss.grad_fn is not None, "GAN D loss should have grad_fn!"

            # R1 & R2 regularization
            if r1_weight > 0.0 or r2_weight > 0.0:
                perturbed_latents = []
                perturbed_timesteps = []
                perturbed_embeds = []

                # Prepare R1 perturbed input
                if r1_weight > 0.0:
                    noisy_real_latent_perturbed = noisy_real_latent.clone()
                    epsilon_real = r1_sigma * torch.randn_like(noisy_real_latent_perturbed)
                    noisy_real_latent_perturbed = noisy_real_latent_perturbed + epsilon_real
                    perturbed_latents.append(noisy_real_latent_perturbed)
                    perturbed_timesteps.append(critic_timestep)
                    perturbed_embeds.append(gan_prompt_embeds)

                # Prepare R2 perturbed input
                if r2_weight > 0.0:
                    noisy_fake_latent_perturbed = noisy_fake_latent.clone()
                    epsilon_generated = r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
                    noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epsilon_generated
                    perturbed_latents.append(noisy_fake_latent_perturbed)
                    perturbed_timesteps.append(critic_timestep)
                    perturbed_embeds.append(prompt_embeds)

                # Batch forward pass
                batched_latents = torch.cat(perturbed_latents, dim=0)
                batched_timesteps = (
                    torch.cat(perturbed_timesteps, dim=0)
                    if isinstance(critic_timestep, torch.Tensor)
                    else critic_timestep
                )
                batched_embeds = torch.cat(perturbed_embeds, dim=0)

                _, batched_logits = fake_score_model(
                    hidden_states=batched_latents,
                    timestep=batched_timesteps,
                    encoder_hidden_states=batched_embeds,
                    gan_mode=True,
                    return_dict=False,
                )

                # Split results and compute losses
                idx = 0
                if r1_weight > 0.0:
                    batch_size = noisy_real_latent.shape[0]
                    noisy_real_logit_perturbed = batched_logits[idx : idx + batch_size]
                    if aprox_r1:
                        r1_loss = r1_weight * torch.nn.functional.mse_loss(
                            noisy_real_logits.float(), noisy_real_logit_perturbed.float(), reduction="mean"
                        )
                    else:
                        r1_grad = (noisy_real_logit_perturbed.float() - noisy_real_logits.float()) / r1_sigma
                        r1_loss = r1_weight * torch.mean(r1_grad**2)

                    assert r1_loss.requires_grad, f"R1 loss should have gradient! Got {r1_loss.requires_grad}"
                    assert r1_loss.grad_fn is not None, "R1 loss should have grad_fn!"
                    idx += batch_size

                if r2_weight > 0.0:
                    batch_size = noisy_fake_latent.shape[0]
                    noisy_fake_logit_perturbed = batched_logits[idx : idx + batch_size]
                    if aprox_r2:
                        r2_loss = r2_weight * torch.nn.functional.mse_loss(
                            noisy_fake_logits.float(), noisy_fake_logit_perturbed.float(), reduction="mean"
                        )
                    else:
                        r2_grad = (noisy_fake_logit_perturbed.float() - noisy_fake_logits.float()) / r2_sigma
                        r2_loss = r2_weight * torch.mean(r2_grad**2)

                    assert r2_loss.requires_grad, f"R2 loss should have gradient! Got {r2_loss.requires_grad}"
                    assert r2_loss.grad_fn is not None, "R2 loss should have grad_fn!"
    else:
        with _helios_stage_range(global_step, "critic", "denoise_forward", adapter="fake"):
            flow_fake_pred = fake_score_model(
                hidden_states=noisy_fake_latent,
                timestep=critic_timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
            )[0]
        denoising_loss = torch.mean((flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2)

        assert denoising_loss.requires_grad, f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
        assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"

    pred_fake_image = convert_flow_pred_to_x0(
        flow_pred=flow_fake_pred,
        xt=noisy_fake_latent,
        timestep=critic_timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )

    final_loss = denoising_loss + gan_D_loss + r1_loss + r2_loss
    assert final_loss.requires_grad, f"Final loss should have gradient! Got {final_loss.requires_grad}"
    assert final_loss.grad_fn is not None, "Final loss should have grad_fn!"

    # Step 5: Debugging Log
    critic_log_dict = {
        "critictrain_latent": generated_image_or_video.detach(),
        "critictrain_noisy_latent": noisy_fake_latent.detach(),
        "critictrain_pred_image": pred_fake_image.detach(),
        "critic_timestep": critic_timestep.detach(),
    }

    if is_use_gan:
        critic_log_dict["denoising_loss"] = denoising_loss.detach().item()
        critic_log_dict["gan_D_loss"] = gan_D_loss.detach().item()
        critic_log_dict["r1_loss"] = r1_loss.detach().item()
        critic_log_dict["r2_loss"] = r2_loss.detach().item()
        critic_log_dict["pipeline_micro_wave"] = pipeline_micro_wave
        critic_log_dict["critic_optimizer_committed"] = pipeline_micro_wave in (
            None,
            1,
        )

    return final_loss, critic_log_dict
