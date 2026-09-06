import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import contextlib
import copy
import json
import logging
import math
import random
import time
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

# Bind each torchrun child before importing Helios CUDA-kernel modules. Some
# import-time capability probes otherwise create a persistent context on GPU 0
# in every local rank. This is initialization-only and remains outside the
# measured training range.
_EARLY_CUDA_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "-1"))
if _EARLY_CUDA_LOCAL_RANK >= 0:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_RANK is set but CUDA is unavailable")
    torch.cuda.set_device(_EARLY_CUDA_LOCAL_RANK)

import torch.distributed.checkpoint as dcp
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    broadcast,
    set_seed,
)
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.utils.create_ema_zero3_lora import create_ema_final, gather_zero3ema
from helios.utils.train_config import Args
from helios.utils.utils_base import (
    NORM_LAYER_PREFIXES,
    compare_configs,
    encode_prompt,
    get_optimizer,
    load_extra_components,
    load_model_checkpoint,
    save_extra_components,
    save_model_checkpoint,
)
from helios.utils.utils_helios_base import (
    _flow_loss,
    prepare_stage1_clean_input_from_latents,
    prepare_stage1_noise_input,
    prepare_stage2_noise_input,
)
from helios.utils.utils_helios_post import (
    OptimizedLowVRAMManager,
    _critic_loss,
    _generator_loss,
    _helios_stage_range,
    _ode_regression_loss,
    merge_dict_list,
    sample_dynamic_dmd_num_latent_sections,
)
from helios.utils.utils_recycle_batch import get_timesteps
from helios.videoalign.inference import VideoVLMRewardInference
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from runtime.operation_trace import (
    STAGE3_SOURCE_KEYS,
    OperationTrace,
    replay_stage3_source,
    resolve_rank_trace_root,
    tensor_sha256,
)
from runtime.generator_update_gate import GeneratorUpdateGateOptions
from runtime.deepspeed_phase_state import DeepSpeedZero1LPStateController
from runtime.fresh_init_rng import (
    FRESH_INIT_RNG_CONTRACT,
    isolated_component_rng,
)
from runtime.official_baseline import (
    OfficialBaselineOptions,
    PhaseTracer,
    SparseShardEMA,
    TrainableGradientObserver,
    construct_in_local_rank_waves,
    optimizer_state_checksum,
    parameter_gradient_diagnostics,
    parameter_summary,
    should_restore_dataloader_state,
    suspend_transformers_zero3_init,
    trainable_checksum,
    trainable_sha256,
    trim_host_allocator,
    write_oom_report,
)
from transformers import (
    AutoTokenizer,
    UMT5EncoderModel,
)

import diffusers
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def main(args):
    baseline_options = OfficialBaselineOptions.from_env()
    baseline_options.validate()
    precast_paths_configured = any(
        path is not None
        for path in (
            baseline_options.precast_generator_path,
            baseline_options.precast_score_path,
        )
    )
    precast_fresh_init = bool(
        precast_paths_configured
        and not args.training_config.resume_from_checkpoint
    )
    if (
        precast_fresh_init
        and not baseline_options.allow_precast_fresh_init
    ):
        raise ValueError(
            "pre-cast checkpoint loading changes fresh-initialization RNG progression; "
            "resume a real checkpoint or explicitly set "
            "DISTILLSCHEDULE_ALLOW_PRECAST_FRESH_INIT=1"
        )
    if precast_fresh_init and args.seed is None:
        raise ValueError("pre-cast fresh initialization requires an explicit seed")
    generator_update_gate = GeneratorUpdateGateOptions.from_env()
    generator_update_gate.validate_baseline(baseline_options)
    if (
        "generator" in baseline_options.immutable_model_backing_components
        and not args.training_config.gradient_checkpointing
    ):
        raise ValueError(
            "Generator whole-model immutable backing requires gradient checkpointing "
            "so frozen CUDA storage can be discarded before backward recomputation"
        )
    phase_tracer = PhaseTracer(baseline_options.trace_dir)
    operation_trace_root = baseline_options.operation_trace_dir
    operation_trace_rank = int(os.environ.get("RANK", "0"))
    operation_trace_dir = (
        resolve_rank_trace_root(operation_trace_root, operation_trace_rank)
        if operation_trace_root is not None
        else None
    )
    operation_trace = OperationTrace.load(operation_trace_dir) if operation_trace_dir is not None else None
    phase_tracer.record(
        "main_start",
        baseline_options=baseline_options.__dict__,
        early_cuda_local_rank=_EARLY_CUDA_LOCAL_RANK,
        early_cuda_current_device=(
            torch.cuda.current_device() if torch.cuda.is_initialized() else None
        ),
        operation_trace_hash=(operation_trace.trace_hash if operation_trace is not None else None),
        operation_trace_rank=operation_trace_rank,
        operation_trace_dir=(str(operation_trace_dir) if operation_trace_dir is not None else None),
    )

    def construct_backbone_in_wave(logical_component, factory):
        wave_size = baseline_options.construction_wave_size
        if not wave_size:
            return factory()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

        def record(phase, wave_index, start, end):
            phase_tracer.record(
                phase,
                logical_component=logical_component,
                construction_wave_size=wave_size,
                construction_wave_index=wave_index,
                construction_wave_start_rank=start,
                construction_wave_end_rank=end,
                local_world_size=local_world_size,
            )

        return construct_in_local_rank_waves(
            factory,
            local_rank=local_rank,
            local_world_size=local_world_size,
            wave_size=wave_size,
            barrier=accelerator.wait_for_everyone,
            on_wave_start=lambda i, start, end: record(
                "backbone_construction_wave_start", i, start, end
            ),
            on_wave_end=lambda i, start, end: record(
                "backbone_construction_wave_end", i, start, end
            ),
        )

    if args.data_config.use_stage3_dataset:
        from helios.dataset.dataloader_dmd import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    elif args.data_config.use_stage1_dataset:
        from helios.dataset.dataloader_history_latents_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    else:
        from helios.dataset.dataloader_mp4_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )

    if torch.backends.mps.is_available() and args.training_config.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # load dmd reward model
    reward_model = None
    if args.training_config.is_use_reward_model:
        reward_model = VideoVLMRewardInference(args.model_config.reward_model_name_or_path)
        reward_model.model.requires_grad_(False)
        reward_model.model.eval()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    # Support 2 models training using deepspeed.
    # https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model
    deepspeed_plugins = None
    dmd_deepspeed_training = (
        args.training_config.is_train_dmd
        and args.training_config.dmd_generator_deepspeed_config is not None
        and args.training_config.dmd_critic_deepspeed_config is not None
    )
    if dmd_deepspeed_training:
        generator_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_generator_deepspeed_config)
        critic_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_critic_deepspeed_config)
        deepspeed_plugins = {"generator": generator_zero_plugin, "critic_model": critic_zero_plugin}

    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=None if baseline_options.skip_reporting else args.report_to.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins,
        kwargs_handlers=[kwargs, init_kwargs],
    )
    if (
        accelerator.distributed_type == DistributedType.DEEPSPEED
        and args.training_config.is_train_dmd
        and not args.training_config.dmd_generator_deepspeed_config
        and not args.training_config.dmd_critic_deepspeed_config
    ):
        raise ValueError("`--deepspeed_config` is required for DMD distillation.")

    if dmd_deepspeed_training:
        critic_accelerator = Accelerator()

    phase_tracer.record(
        "accelerators_ready",
        distributed_type=str(accelerator.distributed_type),
        world_size=accelerator.num_processes,
        device=str(accelerator.device),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        current_conf = OmegaConf.to_container(args, resolve=True)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing_conf = json.load(f)

            ignore_keys = {"training_config.local_rank"}
            mismatches = compare_configs(existing_conf, current_conf, ignore_keys=ignore_keys)
            if mismatches:
                print("Config mismatches found:")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                raise ValueError("Configuration mismatch detected!")
        else:
            with open(config_path, "w") as f:
                json.dump(current_conf, f, indent=4)

    if args.training_config.use_ema:
        args.training_config.ema_zero3_port = os.environ.get("MASTER_PORT", "12345")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if not baseline_options.skip_reporting and args.report_to.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model_config.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    if args.training_config.is_enable_stage2:
        noise_scheduler = HeliosScheduler(
            shift=args.training_config.stage2_timestep_shift,
            stages=args.training_config.stage2_num_stages,
            stage_range=args.training_config.stage2_stage_range,
            gamma=args.training_config.stage2_scheduler_gamma,
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        noise_scheduler = UniPCMultistepScheduler.from_pretrained("scripts/accelerate_configs/scheduler_config.json")
        noise_scheduler_copy = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        if args.training_config.is_train_dmd:
            noise_scheduler.config.flow_shift = args.training_config.dmd_timestep_shift

    if args.training_config.is_train_dmd:
        if args.training_config.is_enable_stage2:
            critic_noise_scheduler = HeliosScheduler(
                shift=args.training_config.stage2_timestep_shift,
                stages=args.training_config.stage2_num_stages,
                stage_range=args.training_config.stage2_stage_range,
                gamma=args.training_config.stage2_scheduler_gamma,
            )
        else:
            critic_noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    preprocessor_context = (
        suspend_transformers_zero3_init()
        if baseline_options.staged_zero3_prepare
        else contextlib.nullcontext()
    )
    with preprocessor_context:
        vae = AutoencoderKLWan.from_pretrained(
            args.model_config.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.model_config.revision,
            variant=args.model_config.variant,
            torch_dtype=torch.float32,
            device_map=accelerator.device,
        )
        if args.model_config.enable_slicing:
            vae.enable_slicing()
        if args.model_config.enable_tiling:
            vae.enable_tiling()

        text_encoder = UMT5EncoderModel.from_pretrained(
            args.model_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.model_config.revision,
            variant=args.model_config.variant,
            dtype=weight_dtype,
            device_map=accelerator.device,
        )
        # For negative prompt
        with torch.no_grad():
            negative_prompt_embeds, _ = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=args.data_config.negative_prompt,
                device=accelerator.device,
            )
    phase_tracer.record("preprocessors_and_negative_prompt_ready")

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
        "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
        "restrict_lora": args.training_config.restrict_lora,
        "restrict_lora_rank": args.training_config.restrict_lora_rank,
        "is_amplify_history": args.training_config.is_amplify_history,
        "history_scale_mode": args.training_config.history_scale_mode,
    }
    backbone_load_kwargs = {}
    if baseline_options.direct_bf16_load:
        backbone_load_kwargs["torch_dtype"] = weight_dtype

    def resolve_backbone_load(component, original_path, subfolder):
        precast_path = (
            baseline_options.precast_generator_path
            if component == "generator"
            else baseline_options.precast_score_path
        )
        load_kwargs = dict(backbone_load_kwargs)
        if precast_path is None:
            return original_path, subfolder, load_kwargs
        load_kwargs["precast_direct_load"] = True
        source_path = Path(original_path)
        if subfolder is not None:
            source_path = source_path / subfolder
        load_kwargs["precast_expected_source_path"] = str(source_path)
        return str(precast_path), None, load_kwargs

    generator_path, generator_subfolder, generator_load_kwargs = resolve_backbone_load(
        "generator",
        args.model_config.transformer_model_name_or_path,
        args.model_config.subfolder or "transformer",
    )
    transformer = construct_backbone_in_wave(
        "generator",
        lambda: HeliosTransformer3DModel.from_pretrained(
            generator_path,
            subfolder=generator_subfolder,
            transformer_additional_kwargs=transformer_additional_kwargs,
            **generator_load_kwargs,
        ),
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()
    phase_tracer.record("generator_backbone_loaded", parameters=parameter_summary(transformer))

    # load dmd real score model
    if args.training_config.is_train_dmd:
        if args.model_config.real_score_model_name_or_path is None:
            args.model_config.real_score_model_name_or_path = args.model_config.transformer_model_name_or_path
        critic_transformer_additional_kwargs = {
            "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
            "zero_history_timestep": args.training_config.zero_history_timestep,
            "restrict_self_attn": args.training_config.restrict_self_attn,
            "guidance_cross_attn": args.training_config.guidance_cross_attn,
            "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
            "restrict_lora": args.training_config.restrict_lora,
            "restrict_lora_rank": args.training_config.restrict_lora_rank,
            "is_use_gan": args.training_config.is_use_gan,
            "is_use_gan_hooks": args.training_config.is_use_gan_hooks,
            "is_use_gan_final": args.training_config.is_use_gan_final,
            "gan_cond_map_dim": args.training_config.gan_cond_map_dim,
            "gan_hooks": args.training_config.gan_hooks,
        }

        score_path, score_subfolder, score_load_kwargs = resolve_backbone_load(
            "score",
            args.model_config.real_score_model_name_or_path,
            args.model_config.critic_subfolder or "transformer",
        )
        real_score_model = construct_backbone_in_wave(
            "score",
            lambda: HeliosTransformer3DModel.from_pretrained(
                score_path,
                subfolder=score_subfolder,
                transformer_additional_kwargs=critic_transformer_additional_kwargs,
                **score_load_kwargs,
            ),
        )
        real_score_model = replace_rmsnorm_with_fp32(real_score_model)
        real_score_model = replace_all_norms_with_flash_norms(real_score_model)
        phase_tracer.record("score_backbone_loaded", parameters=parameter_summary(real_score_model))

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    if args.training_config.is_train_dmd:
        real_score_model.requires_grad_(False)

    if args.model_config.lora_layers is not None:
        if args.model_config.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.model_config.lora_layers.split(",")]
            # add the input layer to the mix.
            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            # add multi-term memory patches to the mix
            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)
        elif args.model_config.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
            # add the input layer to the mix.
            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            # add multi-term memory patches to the mix
            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)
        target_modules = [t for t in target_modules if "norm" not in t]
    else:
        target_modules = args.model_config.lora_target_modules

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        exclude_modules=list(args.model_config.lora_exclude_modules),
    )
    precast_fresh_init_component_seeds = {}
    if precast_fresh_init:
        with isolated_component_rng(
            args.seed, "generator_adapter"
        ) as component_seed:
            transformer.add_adapter(transformer_lora_config)
        precast_fresh_init_component_seeds["generator_adapter"] = component_seed
    else:
        transformer.add_adapter(transformer_lora_config)
    phase_tracer.record("generator_adapter_added", parameters=parameter_summary(transformer))

    if args.model_config.train_norm_layers:
        for name, param in transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    # set trainable parameter
    trainable_modules = []
    if args.training_config.is_train_full_multi_term_memory_patchg:
        trainable_modules.extend(["patch_short", "patch_mid", "patch_long"])
    if args.training_config.is_train_full_patch_embedding:
        trainable_modules.append("patch_embedding")
    if args.training_config.is_train_restrict_lora:
        trainable_modules.extend(["q_loras", "k_loras", "v_loras"])
    if args.training_config.is_amplify_history:
        trainable_modules.append("history_key_scale")
    for name, param in transformer.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    transformer_cpu = None
    ds_config = None
    if args.training_config.use_ema and not baseline_options.sparse_ema:
        model_cls = HeliosTransformer3DModel
        transformer_cpu = copy.deepcopy(transformer)
        with open(args.training_config.ema_deepspeed_config_file, "r") as f:
            ds_config = json.load(f)
        phase_tracer.record("dense_ema_source_copied", parameters=parameter_summary(transformer_cpu))
    elif args.training_config.use_ema:
        phase_tracer.record("dense_ema_source_skipped_for_sparse_policy")

    # get fake score model
    if args.training_config.is_train_dmd:
        critic_target_modules = [
            m for m in target_modules if m not in ["patch_short", "patch_mid", "patch_long", "patch_embedding"]
        ]
        critic_exclude_modules = list(args.model_config.lora_exclude_modules) + [
            "patch_short",
            "patch_mid",
            "patch_long",
            "patch_embedding",
            "gan_heads",
            "gan_final_head",
        ]
        critic_transformer_lora_config = LoraConfig(
            r=args.model_config.critic_lora_rank,
            lora_alpha=args.model_config.critic_lora_alpha,
            lora_dropout=args.model_config.critic_lora_dropout,
            init_lora_weights="gaussian",
            target_modules=critic_target_modules,
            exclude_modules=critic_exclude_modules,
        )

        if precast_fresh_init:
            with isolated_component_rng(
                args.seed, "score_adapter"
            ) as component_seed:
                real_score_model.add_adapter(critic_transformer_lora_config)
            precast_fresh_init_component_seeds["score_adapter"] = component_seed
        else:
            real_score_model.add_adapter(critic_transformer_lora_config)
        phase_tracer.record("score_adapter_added", parameters=parameter_summary(real_score_model))

        if args.model_config.train_norm_layers:
            for name, param in real_score_model.named_parameters():
                if any(k in name for k in NORM_LAYER_PREFIXES):
                    param.requires_grad = True

        if args.training_config.is_use_gan:
            critic_trainable_modules = ["gan_heads", "gan_final_head"]
            for name, param in real_score_model.named_parameters():
                for trainable_module_name in critic_trainable_modules:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break

    if precast_fresh_init:
        set_seed(args.seed)
        phase_tracer.record(
            "precast_fresh_init_rng_rebased",
            contract=FRESH_INIT_RNG_CONTRACT,
            base_seed=args.seed,
            component_seeds=precast_fresh_init_component_seeds,
        )

    if baseline_options.split_modulation:
        b0_compute_path = baseline_options.b0_compute_path
        if b0_compute_path:
            modulation_chunk_tokens = None
            ffn_chunk_tokens = None
            lora_output_chunk_size = None
            lora_min_output_features = None
        else:
            from runtime.chunked_lora import install_chunked_lora_b
            from runtime.safe_tiled_ffn import install_safe_tiled_ffn

            modulation_chunk_tokens = int(
                os.environ.get("DISTILLSCHEDULE_MODULATION_CHUNK_TOKENS", "16")
            )
            ffn_chunk_tokens = int(
                os.environ.get("DISTILLSCHEDULE_FFN_CHUNK_TOKENS", "512")
            )
            lora_output_chunk_size = int(
                os.environ.get("DISTILLSCHEDULE_LORA_OUTPUT_CHUNK_SIZE", "512")
            )
            lora_min_output_features = int(
                os.environ.get("DISTILLSCHEDULE_LORA_MIN_OUTPUT_FEATURES", "4096")
            )
            if (
                modulation_chunk_tokens <= 0
                or ffn_chunk_tokens <= 0
                or lora_output_chunk_size <= 0
                or lora_min_output_features <= 0
            ):
                raise ValueError("split-modulation chunk sizes must be positive")
        physical_models = [("generator", transformer)]
        if args.training_config.is_train_dmd:
            physical_models.append(("score", real_score_model))
        for physical_model, model in physical_models:
            blocks = getattr(model, "blocks", ())
            model._distillschedule_split_modulation = True
            model._distillschedule_b0_compute_path = b0_compute_path
            for block in blocks:
                block._distillschedule_split_modulation = True
                block._distillschedule_b0_compute_path = b0_compute_path
                block._distillschedule_physical_model = physical_model
                if not b0_compute_path:
                    block._distillschedule_modulation_chunk_tokens = (
                        modulation_chunk_tokens
                    )
            if b0_compute_path:
                chunked_lora_b_count = 0
                safe_tiled_ffn_count = 0
            else:
                chunked_lora_b_count = install_chunked_lora_b(
                    model,
                    output_chunk_size=lora_output_chunk_size,
                    min_output_features=lora_min_output_features,
                )
                safe_tiled_ffn_count = install_safe_tiled_ffn(
                    model,
                    chunk_tokens=ffn_chunk_tokens,
                    logical_component=physical_model,
                )
            direct_lora_b_accumulator_parameter_count = sum(
                len(
                    getattr(
                        module,
                        "_distillschedule_direct_lora_b_parameter_names",
                        (),
                    )
                )
                for module in model.modules()
            )
            model._distillschedule_direct_lora_b_accumulator_parameter_count = (
                direct_lora_b_accumulator_parameter_count
            )
            phase_tracer.record(
                "split_modulation_enabled",
                physical_model=physical_model,
                block_count=len(blocks),
                compute_path=(
                    "b0_vectorized_split_modulation"
                    if b0_compute_path
                    else "memory_bounded_split_modulation"
                ),
                b0_compute_path=b0_compute_path,
                lazy_timestep_projection=True,
                modulation_chunk_tokens=modulation_chunk_tokens,
                chunked_lora_b_count=chunked_lora_b_count,
                lora_output_chunk_size=lora_output_chunk_size,
                lora_min_output_features=lora_min_output_features,
                safe_tiled_ffn_count=safe_tiled_ffn_count,
                ffn_chunk_tokens=ffn_chunk_tokens,
                direct_lora_b_accumulator_parameter_count=(
                    direct_lora_b_accumulator_parameter_count
                ),
            )

    if args.model_config.load_checkpoints_custom:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.load_model_path,
            transformer=transformer,
            pipeline_class=HeliosPipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )
        if args.training_config.is_train_dmd:
            assert args.model_config.critic_lora_name_or_path is not None
            assert args.model_config.load_dcp

    if args.model_config.critic_lora_name_or_path is not None:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.critic_lora_name_or_path,
            transformer=real_score_model,
            pipeline_class=HeliosPipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    target_device = (
        "cpu" if (args.data_config.use_stage1_dataset or args.data_config.use_stage3_dataset) else accelerator.device
    )
    vae.to(target_device)
    text_encoder.to(target_device)
    if args.training_config.is_use_reward_model:
        reward_model.model.to(target_device)
    free_memory()

    if baseline_options.release_preprocessors:
        safe_to_release_vae = (
            (
                not args.training_config.is_dmd_vae_decode
                and not args.training_config.is_use_reward_model
                and not args.training_config.is_smoothness_loss
            )
            or args.training_config.is_use_gt_history
        )
        if not safe_to_release_vae:
            raise ValueError("preprocessor release requested while an enabled branch still consumes the VAE")
        vae = None
        text_encoder = None
        free_memory()
        phase_tracer.record("preprocessors_released")

    # we never offload the transformer to CPU, so we can just use the accelerator device
    for name, param in transformer.named_parameters():
        should_keep_fp32 = any(pattern in name for pattern in transformer.__class__._keep_in_fp32_modules)
        if should_keep_fp32:
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)
    if not baseline_options.staged_zero3_prepare:
        transformer.to(accelerator.device)
    else:
        phase_tracer.record("generator_pre_device_move_skipped", parameters=parameter_summary(transformer))

    if args.training_config.is_train_dmd:
        for name, param in real_score_model.named_parameters():
            should_keep_fp32 = any(pattern in name for pattern in real_score_model.__class__._keep_in_fp32_modules)
            if should_keep_fp32:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(weight_dtype)
        if not baseline_options.staged_zero3_prepare:
            real_score_model.to(accelerator.device)
        else:
            phase_tracer.record("score_pre_device_move_skipped", parameters=parameter_summary(real_score_model))
    free_memory()
    phase_tracer.record(
        "backbones_cast_and_placed",
        generator=parameter_summary(transformer),
        score=parameter_summary(real_score_model) if args.training_config.is_train_dmd else None,
    )

    if args.training_config.enable_npu_flash_attention:
        if is_torch_npu_available():
            accelerator.print("npu flash attention enabled.")
            transformer.enable_npu_flash_attention()
            if args.training_config.is_train_dmd:
                real_score_model.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
            if args.training_config.is_train_dmd:
                real_score_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.training_config.is_train_dmd:
            real_score_model.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    if args.model_config.train_norm_layers:
                        transformer_norm_layers_to_save = {
                            f"transformer.{name}": param
                            for name, param in model.named_parameters()
                            if any(k in name for k in NORM_LAYER_PREFIXES)
                        }
                        transformer_lora_layers_to_save = {
                            **transformer_lora_layers_to_save,
                            **transformer_norm_layers_to_save,
                        }
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            HeliosPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

            save_extra_components(args, model=unwrap_model(model), output_dir=output_dir)

    def load_model_hook(models, input_dir):
        transformer_ = None
        is_critic_checkpoint = Path(input_dir).name == "critic"
        dcp_dir = os.path.join(input_dir, "distributed_checkpoint")

        # DeepSpeed restores the complete module before Accelerate invokes this hook.
        # Reconstructing another 14B backbone here is both redundant and unsafe for
        # offline critic checkpoints.  The saved DeepSpeed module contains LoRA and
        # the partial patch modules; only the custom dataloader DCP remains to load.
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            dataloader_state_policy = "not_applicable"
            if not is_critic_checkpoint:
                checkpoint_step = int(Path(input_dir).name.rsplit("-", 1)[-1])
                if should_restore_dataloader_state(checkpoint_step, num_update_steps_per_epoch):
                    dcp.load({"dataloader": train_dataloader}, checkpoint_id=dcp_dir)
                    dataloader_state_policy = "restore_mid_epoch_cursor"
                else:
                    dataloader_state_policy = "reset_completed_epoch_cursor"
            phase_tracer.record(
                "deepspeed_redundant_model_hook_skipped",
                checkpoint_component="critic" if is_critic_checkpoint else "generator",
                checkpoint_path=str(input_dir),
                dataloader_state_policy=dataloader_state_policy,
            )
            return

        while len(models) > 0:
            model = models.pop()

            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = HeliosPipeline.lora_state_dict(
            input_dir, weight_name="pytorch_lora_weights.safetensors"
        )

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.model_config.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = HeliosPipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        load_extra_components(args, transformer_, os.path.join(input_dir, "transformer_partial.pth"))

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.training_config.mixed_precision != "fp32":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

        if not is_critic_checkpoint:
            states = {
                "dataloader": train_dataloader,
            }
            dcp.load(states, checkpoint_id=dcp_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.training_config.is_train_dmd:
        critic_accelerator.register_save_state_pre_hook(save_model_hook)
        critic_accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.training_config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training_config.scale_lr:
        args.training_config.learning_rate = (
            args.training_config.learning_rate
            * args.training_config.gradient_accumulation_steps
            * args.training_config.train_batch_size
            * accelerator.num_processes
        )

        if args.training_config.is_train_dmd:
            args.training_config.critic_learning_rate = (
                args.training_config.critic_learning_rate
                * args.training_config.gradient_accumulation_steps
                * args.training_config.train_batch_size
                * accelerator.num_processes
            )

    # Make sure the trainable params are in float32.
    if args.training_config.mixed_precision != "fp32":
        models = [transformer]
        if args.training_config.is_train_dmd:
            models.append(real_score_model)
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.training_config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, accelerator, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)
    phase_tracer.record("generator_optimizer_created")

    if args.training_config.is_train_dmd:
        critic_model_lora_parameters = list(filter(lambda p: p.requires_grad, real_score_model.parameters()))
        critic_model_lr_parameters_with_lr = {
            "params": critic_model_lora_parameters,
            "lr": args.training_config.critic_learning_rate,
        }
        critic_model_params_to_optimize = [critic_model_lr_parameters_with_lr]
        critic_optimizer = get_optimizer(
            args, critic_accelerator, critic_model_params_to_optimize, use_deepspeed=use_deepspeed_optimizer
        )
        phase_tracer.record("critic_optimizer_created")

    # Dataset and DataLoaders creation:
    dataset_sampling_ratios = {}
    if args.data_config.dataset_sampling_ratios:
        for temp_key, temp_value in zip(args.data_config.instance_data_root, args.data_config.dataset_sampling_ratios):
            clean_path = temp_key.rstrip("/")
            dataset_sampling_ratios[clean_path] = temp_value

    if args.data_config.use_stage3_dataset:
        dataset_kwargs = {
            "gan_folders": args.data_config.gan_data_root
            if args.training_config.is_use_gan or args.training_config.is_use_gt_history
            else None,
            "ode_folders": args.data_config.ode_data_root if args.training_config.is_use_ode_regression else None,
            "text_folders": args.data_config.text_data_root
            if not args.training_config.is_only_ode_regression
            else None,
            "is_use_gt_history": args.training_config.is_use_gt_history,
            "return_secondary": args.training_config.is_use_gt_history,
            "single_res": args.data_config.single_res,
            "single_length": args.data_config.single_length,
            "single_num_frame": args.data_config.single_num_frame,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "force_rebuild": args.data_config.force_rebuild,
            "seed": args.seed,
        }
        assert any(
            [
                dataset_kwargs["gan_folders"],
                dataset_kwargs["ode_folders"],
                dataset_kwargs["text_folders"],
            ]
        ), "Invalid dataset config: at least one of `gan_folders`, `ode_folders`, or `text_folders` must be non-empty."
    elif args.data_config.use_stage1_dataset:
        dataset_kwargs = {
            "feature_folders": args.data_config.instance_data_root,
            "single_res": args.data_config.single_res,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "return_prompt_raw": args.training_config.is_use_reward_model,
            "return_all_vae_latent": (
                args.training_config.dmd_teacher_forcing and args.training_config.dmd_teacher_forcing_ratio > 0
            )
            or args.training_config.is_use_gan,
            "history_sizes": args.training_config.history_sizes,
            "is_keep_x0": True,
            "force_rebuild": args.data_config.force_rebuild,
            "seed": args.seed,
        }
    else:
        raise NotImplementedError
        dataset_kwargs = {
            "json_files": args.data_config.instance_data_root,
            "video_folders": args.data_config.instance_video_root,
            "force_rebuild": args.data_config.force_rebuild,
            "stride": args.data_config.stride,
            "resolution": args.data_config.resolution,
            "single_res": args.data_config.single_res,
            "single_length": args.data_config.single_length,
            "single_num_frame": args.data_config.single_num_frame,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "multi_res": args.data_config.multi_res,
            "id_token": args.data_config.id_token,
        }

    train_dataset = BucketedFeatureDataset(**dataset_kwargs)
    phase_tracer.record("stage3_dataset_built", dataset_length=len(train_dataset))

    sampler = BucketedSampler(
        train_dataset,
        batch_size=args.training_config.train_batch_size,
        drop_last=True,  # TODO need to be true now
        shuffle=args.data_config.use_shuffle,
        seed=args.seed,
        dataset_sampling_ratios=dataset_sampling_ratios,
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=args.data_config.pin_memory,
        prefetch_factor=args.data_config.prefetch_factor if args.data_config.prefetch_factor > 0 else None,
        persistent_workers=args.data_config.persistent_workers,
        collate_fn=collate_fn,
        num_workers=args.data_config.dataloader_num_workers,
    )

    if args.model_config.load_dcp:
        if args.model_config.load_dcp_path is not None:
            dcp_dir = os.path.join(args.model_config.load_dcp_path, "distributed_checkpoint")
        else:
            dcp_dir = os.path.join(args.model_config.load_model_path, "distributed_checkpoint")
        states = {
            "dataloader": train_dataloader,
        }
        dcp.load(states, checkpoint_id=dcp_dir)
        print(f"load dcp from {dcp_dir} successfully!")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        )

        if args.training_config.is_train_dmd:
            critic_lr_scheduler = DummyScheduler(
                name=args.training_config.lr_scheduler,
                optimizer=critic_optimizer,
                total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

        if args.training_config.is_train_dmd:
            critic_lr_scheduler = get_scheduler(
                args.training_config.lr_scheduler,
                optimizer=critic_optimizer,
                num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
                num_cycles=args.training_config.lr_num_cycles,
                power=args.training_config.lr_power,
            )

    # Prepare everything with our `accelerator`.
    accelerator.wait_for_everyone()
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.training_config.train_batch_size
        )
    if args.training_config.is_train_dmd:
        if dmd_deepspeed_training:
            accelerator.state.select_deepspeed_plugin("generator")
        phase_tracer.record("generator_prepare_start", parameters=parameter_summary(transformer))
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
        phase_tracer.record(
            "generator_prepare_end",
            parameters=parameter_summary(transformer),
            checksum=trainable_checksum(transformer),
        )
        if baseline_options.swap_generator_before_score_prepare:
            generator_module = transformer.module if hasattr(transformer, "module") else transformer
            generator_module.to("cpu", non_blocking=False)
            free_memory()
            phase_tracer.record(
                "generator_swapped_before_score_prepare",
                parameters=parameter_summary(transformer),
            )
        if dmd_deepspeed_training:
            critic_accelerator.state.select_deepspeed_plugin("critic_model")
            critic_accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
                args.training_config.train_batch_size
            )
        phase_tracer.record("score_prepare_start", parameters=parameter_summary(real_score_model))
        real_score_model, critic_optimizer, critic_lr_scheduler = critic_accelerator.prepare(
            real_score_model, critic_optimizer, critic_lr_scheduler
        )
        phase_tracer.record(
            "score_prepare_end",
            parameters=parameter_summary(real_score_model),
            checksum=trainable_checksum(real_score_model),
        )
    else:
        phase_tracer.record("generator_prepare_start", parameters=parameter_summary(transformer))
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
        phase_tracer.record(
            "generator_prepare_end",
            parameters=parameter_summary(transformer),
            checksum=trainable_checksum(transformer),
        )

    if baseline_options.stop_after_prepare:
        accelerator.wait_for_everyone()
        phase_tracer.record("requested_stop_after_prepare")
        return

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and not baseline_options.skip_reporting:
        tracker_name = args.report_to.tracker_name or "wanvideo-train"
        wandb_name = args.report_to.wandb_name or "custom-wandb-run-name"
        accelerator.init_trackers(
            tracker_name,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={"wandb": {"name": wandb_name}},
        )

    # Train!
    total_batch_size = (
        args.training_config.train_batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    if args.training_config.is_train_dmd:
        critic_num_trainable_parameters = sum(
            param.numel() for model in critic_model_params_to_optimize for param in model["params"]
        )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num generator trainable parameters = {num_trainable_parameters}")
    if args.training_config.is_train_dmd:
        accelerator.print(f"  Num fake_score_model trainable parameters = {critic_num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.training_config.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.training_config.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    ema_transformer = None
    vram_manager = None
    if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
        vram_manager = OptimizedLowVRAMManager(
            phase_tracer=phase_tracer,
            trim_host_fn=(
                trim_host_allocator
                if baseline_options.trim_host_after_model_move
                else None
            ),
            swap_wave_size=baseline_options.swap_wave_size,
        )
        vram_manager.register_model(transformer, "generator")
        vram_manager.register_model(real_score_model, "score")
        if baseline_options.critic_phase_offload_generator_lp_state:
            generator_lp_state_controller = DeepSpeedZero1LPStateController(
                transformer,
                logical_component="generator",
                event_sink=phase_tracer.record,
            )
            vram_manager.register_engine_state_controller(
                transformer, generator_lp_state_controller
            )
        if vae is not None:
            vram_manager.register_model(vae, "vae")
        if reward_model is not None:
            vram_manager.register_model(reward_model.model, "reward")

    # Potentially load in the weights and states from a previous save
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            resume_path = args.training_config.resume_from_checkpoint
            if os.path.isabs(resume_path):
                path = resume_path
            else:
                path = os.path.join(args.output_dir, resume_path)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path is None or not os.path.exists(path):
            if precast_paths_configured:
                raise RuntimeError(
                    "configured resume checkpoint does not exist; refusing to combine "
                    "pre-cast loading with an implicit fresh start"
                )
            accelerator.print(
                f"Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path, load_kwargs={"weights_only": False})
            if args.training_config.is_train_dmd:
                critic_accelerator.load_state(os.path.join(path, "critic"), load_kwargs={"weights_only": False})
            global_step = int(os.path.basename(path).split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            if args.training_config.use_ema:
                if baseline_options.sparse_ema:
                    ema_transformer = SparseShardEMA.load_pretrained(
                        transformer,
                        os.path.join(path, "model_ema"),
                    )
                    phase_tracer.record(
                        "sparse_ema_resumed",
                        checkpoint_path=os.path.join(path, "model_ema"),
                        ema=ema_transformer.checksum(),
                    )
                else:
                    if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                        vram_manager.move_to_cpu(transformer, non_blocking=False)
                        vram_manager.move_to_cpu(real_score_model, non_blocking=False)

                    transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
                    ema_transformer = create_ema_final(
                        accelerator=accelerator,
                        args=args,
                        transformer_cpu=transformer_cpu,
                        model_cls=model_cls,
                        ds_config=ds_config,
                        transformer_lora_config=transformer_lora_config,
                        resume_checkpoint_path=os.path.join(path, "model_ema"),
                        transformer_additional_kwargs=transformer_additional_kwargs,
                    )
                    accelerator.wait_for_everyone()

                    transformer_cpu = None
                    del transformer_cpu

                    if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                        vram_manager.move_to_gpu(transformer, accelerator.device)
                        vram_manager.move_to_gpu(real_score_model, accelerator.device)
            phase_tracer.record(
                "checkpoint_resume_complete",
                global_step=global_step,
                generator=trainable_checksum(transformer),
                critic=(
                    trainable_checksum(real_score_model)
                    if args.training_config.is_train_dmd
                    else None
                ),
                generator_optimizer=optimizer_state_checksum(optimizer),
                critic_optimizer=(
                    optimizer_state_checksum(critic_optimizer)
                    if args.training_config.is_train_dmd
                    else None
                ),
                generator_scheduler=lr_scheduler.state_dict(),
                critic_scheduler=(
                    critic_lr_scheduler.state_dict()
                    if args.training_config.is_train_dmd
                    else None
                ),
            )
    else:
        initial_global_step = 0

    if args.model_config.load_checkpoints_custom:
        assert initial_global_step == 0

    if baseline_options.logical_start_step:
        if args.training_config.resume_from_checkpoint:
            raise ValueError("logical start step cannot be combined with checkpoint resume")
        if args.training_config.max_train_steps <= baseline_options.logical_start_step:
            raise ValueError("max_train_steps must exceed the logical start step")
        global_step = baseline_options.logical_start_step
        initial_global_step = global_step
        first_epoch = 0
        phase_tracer.record(
            "logical_start_step_applied",
            global_step=global_step,
            synthetic_boundary_probe=True,
            operation_trace_hash=(
                operation_trace.trace_hash if operation_trace is not None else None
            ),
        )

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if (
        args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
    ) or args.data_config.use_stage3_dataset:
        if (
            not args.training_config.is_dmd_vae_decode
            and not args.training_config.is_use_reward_model
            and not args.training_config.is_smoothness_loss
        ) or args.training_config.is_use_gt_history:
            vae = None
        text_encoder = None
        free_memory()

    # initial ema
    if ema_transformer is None and args.training_config.use_ema:
        if baseline_options.sparse_ema:
            if args.training_config.resume_from_checkpoint:
                raise NotImplementedError("sparse EMA resume is not implemented in the H1 baseline")
            ema_transformer = SparseShardEMA.from_module(
                transformer,
                decay=args.training_config.ema_decay,
                update_after_step=args.training_config.ema_start_step,
                base_model=args.model_config.transformer_model_name_or_path,
            )
            phase_tracer.record("sparse_ema_initialized", ema=ema_transformer.checksum())
            if baseline_options.synthetic_ema_optimization_step is not None:
                ema_transformer.optimization_step = (
                    baseline_options.synthetic_ema_optimization_step
                )
                ema_transformer.cur_decay_value = None
                phase_tracer.record(
                    "sparse_ema_synthetic_boundary_applied",
                    optimization_step=ema_transformer.optimization_step,
                    synthetic_boundary_probe=True,
                )
        else:
            if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                vram_manager.move_to_cpu(transformer, non_blocking=False)
                vram_manager.move_to_cpu(real_score_model, non_blocking=False)
            else:
                transformer.to("cpu", non_blocking=False)

            transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
            ema_transformer = create_ema_final(
                accelerator=accelerator,
                args=args,
                transformer_cpu=transformer_cpu,
                model_cls=model_cls,
                ds_config=ds_config,
                transformer_lora_config=transformer_lora_config,
                update_after_step=args.training_config.ema_start_step,
            )
            accelerator.wait_for_everyone()

            transformer_cpu = None
            del transformer_cpu

            if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                vram_manager.move_to_gpu(transformer, accelerator.device)
                vram_manager.move_to_gpu(real_score_model, accelerator.device)
            else:
                transformer.to(accelerator.device, non_blocking=False)

    immutable_model_backings = {}
    if baseline_options.immutable_model_backing_components:
        if baseline_options.trace_dir is None:
            raise ValueError("immutable model backing requires a trace directory")
        if vram_manager is None:
            raise ValueError("immutable model backing requires the low-VRAM manager")
        from runtime.immutable_model_backing import (
            install_node_shared_immutable_model_backing,
            prepare_node_shared_immutable_cache,
        )

        backing_event_dir = (
            baseline_options.trace_dir.parent / "immutable_model_backing"
        )
        backing_event_dir.mkdir(parents=True, exist_ok=True)
        backing_specs = {
            "generator": (
                transformer,
                unwrap_model(transformer),
                transformer_lora_parameters,
                (
                    f"{Path(args.model_config.transformer_model_name_or_path).resolve()}"
                    f"::{args.model_config.subfolder}::bf16"
                ),
            ),
            "score": (
                real_score_model,
                unwrap_model(real_score_model),
                critic_model_lora_parameters,
                (
                    f"{Path(args.model_config.real_score_model_name_or_path).resolve()}"
                    "::transformer::bf16"
                ),
            ),
        }
        hybrid_streamed_parameter_ids = {}
        if baseline_options.hybrid_residency:
            from runtime.frozen_block_streaming import (
                selected_frozen_block_parameter_ids,
            )

            for logical_component in (
                baseline_options.frozen_block_streaming_components
            ):
                (
                    _engine,
                    physical_model,
                    optimizer_parameters,
                    _checkpoint_identity,
                ) = backing_specs[logical_component]
                component_streamed_indices = (
                    baseline_options.frozen_block_indices_for(logical_component)
                )
                component_streamed_count = (
                    baseline_options.frozen_block_count_for(logical_component)
                )
                hybrid_streamed_parameter_ids[logical_component] = (
                    selected_frozen_block_parameter_ids(
                        physical_model,
                        streamed_block_count=component_streamed_count,
                        streamed_block_indices=component_streamed_indices,
                        optimizer_owned_parameter_ids={
                            id(parameter) for parameter in optimizer_parameters
                        },
                    )
                )
                phase_tracer.record(
                    "hybrid_residency_ownership_resolved",
                    logical_component=logical_component,
                    externally_managed_parameter_count=len(
                        hybrid_streamed_parameter_ids[logical_component]
                    ),
                    streamed_block_count=component_streamed_count,
                    streamed_block_indices=list(component_streamed_indices),
                )
        selected_backing_specs = [
            (logical_component, *backing_specs[logical_component])
            for logical_component in baseline_options.immutable_model_backing_components
        ]
        if accelerator.is_local_main_process:
            for (
                logical_component,
                _engine,
                physical_model,
                optimizer_parameters,
                checkpoint_identity,
            ) in selected_backing_specs:
                phase_tracer.record(
                    "immutable_model_cache_prepare_start",
                    logical_component=logical_component,
                    checkpoint_identity=checkpoint_identity,
                )
                cache_stats = prepare_node_shared_immutable_cache(
                    physical_model,
                    logical_component=logical_component,
                    checkpoint_identity=checkpoint_identity,
                    cache_root=baseline_options.immutable_model_mmap_cache,
                    optimizer_owned_parameter_ids={
                        id(parameter) for parameter in optimizer_parameters
                    },
                    externally_managed_parameter_ids=(
                        hybrid_streamed_parameter_ids.get(logical_component, ())
                    ),
                )
                phase_tracer.record(
                    "immutable_model_cache_prepare_end",
                    logical_component=logical_component,
                    checkpoint_identity=checkpoint_identity,
                    cache_stats=cache_stats,
                )
        # The two local-rank-0 writers operate on node-qualified roots. A global
        # barrier is intentional: consumers never observe a partial manifest.
        accelerator.wait_for_everyone()
        for (
            logical_component,
            engine,
            physical_model,
            optimizer_parameters,
            checkpoint_identity,
        ) in selected_backing_specs:
            event_path = backing_event_dir / (
                f"rank_{accelerator.process_index:03d}_{logical_component}.jsonl"
            )
            backing = install_node_shared_immutable_model_backing(
                physical_model,
                logical_component=logical_component,
                checkpoint_identity=checkpoint_identity,
                cache_root=baseline_options.immutable_model_mmap_cache,
                optimizer_owned_parameter_ids={
                    id(parameter) for parameter in optimizer_parameters
                },
                externally_managed_parameter_ids=(
                    hybrid_streamed_parameter_ids.get(logical_component, ())
                ),
                event_path=event_path,
                detailed_events=baseline_options.detailed_hotpath_trace,
            )
            immutable_model_backings[logical_component] = backing
            vram_manager.register_immutable_model_backing(engine, backing)
            # Establish the initial CPU-backed state. Frozen parameters are
            # rebound without D2H; only rank-private state is copied to CPU.
            vram_manager.move_to_cpu(engine, non_blocking=False)
            free_memory()
            phase_tracer.record(
                "immutable_model_backing_install_end",
                logical_component=logical_component,
                checkpoint_identity=checkpoint_identity,
                event_path=str(event_path),
                stats=dict(backing.stats),
            )
        accelerator.wait_for_everyone()

    block_streamers = {}
    if baseline_options.frozen_block_streaming_window:
        if baseline_options.trace_dir is None:
            raise ValueError("B3 frozen-block streaming requires a trace directory")
        if vram_manager is None:
            raise ValueError("B3 frozen-block streaming requires the low-VRAM manager")
        from runtime.frozen_block_streaming import install_frozen_block_streamer

        streaming_event_dir = baseline_options.trace_dir.parent / "block_streaming"
        streaming_event_dir.mkdir(parents=True, exist_ok=True)
        streaming_models = [
            (
                "generator",
                transformer,
                unwrap_model(transformer),
                transformer_lora_parameters,
                (
                    f"{Path(args.model_config.transformer_model_name_or_path).resolve()}"
                    f"::{args.model_config.subfolder}::bf16"
                ),
            ),
            (
                "score",
                real_score_model,
                unwrap_model(real_score_model),
                critic_model_lora_parameters,
                f"{Path(args.model_config.real_score_model_name_or_path).resolve()}::transformer::bf16",
            ),
        ]
        streaming_models = [
            record
            for record in streaming_models
            if record[0] in baseline_options.frozen_block_streaming_components
        ]
        for logical_component, engine, physical_model, optimizer_parameters, checkpoint_identity in streaming_models:
            component_streamed_indices = baseline_options.frozen_block_indices_for(
                logical_component
            )
            component_streamed_count = baseline_options.frozen_block_count_for(
                logical_component
            )
            pin_component_host_state = (
                baseline_options.frozen_block_pin_host_state
                and baseline_options.frozen_block_pin_host_component
                in {"all", logical_component}
            )
            shared_immutable_block_indices = (
                baseline_options.frozen_block_shared_generator_indices
                if logical_component == "generator"
                else baseline_options.frozen_block_shared_score_indices
            )
            phase_tracer.record(
                "frozen_block_streamer_install_start",
                logical_component=logical_component,
                window_size=baseline_options.frozen_block_streaming_window,
                streamed_block_count=component_streamed_count,
                streamed_block_indices=list(component_streamed_indices),
                host_policy=baseline_options.frozen_block_host_policy,
                shared_immutable_block_indices=list(shared_immutable_block_indices),
                pin_host_state=pin_component_host_state,
            )
            event_path = streaming_event_dir / (
                f"rank_{accelerator.process_index:03d}_{logical_component}.jsonl"
            )
            streamer = install_frozen_block_streamer(
                physical_model,
                logical_component=logical_component,
                device=accelerator.device,
                window_size=baseline_options.frozen_block_streaming_window,
                streamed_block_count=component_streamed_count,
                streamed_block_indices=component_streamed_indices,
                host_policy=baseline_options.frozen_block_host_policy,
                shared_immutable_block_indices=shared_immutable_block_indices,
                optimizer_owned_parameter_ids={id(parameter) for parameter in optimizer_parameters},
                checkpoint_identity=checkpoint_identity,
                event_path=event_path,
                mmap_cache_root=baseline_options.frozen_block_mmap_cache,
                pin_host_state=pin_component_host_state,
                detailed_events=baseline_options.detailed_hotpath_trace,
                memory_snapshot_events=(
                    baseline_options.frozen_block_memory_snapshots
                ),
            )
            block_streamers[logical_component] = streamer
            vram_manager.register_block_streamer(engine, streamer)
            free_memory()
            phase_tracer.record(
                "frozen_block_streamer_install_end",
                logical_component=logical_component,
                window_size=baseline_options.frozen_block_streaming_window,
                streamed_block_count=component_streamed_count,
                streamed_block_indices=sorted(streamer.streamed_block_indices),
                host_policy=baseline_options.frozen_block_host_policy,
                shared_immutable_block_indices=list(shared_immutable_block_indices),
                pin_host_state=pin_component_host_state,
                memory_snapshot_events=(
                    baseline_options.frozen_block_memory_snapshots
                ),
                event_path=str(event_path),
                stats=dict(streamer.stats),
            )

    activation_offload_registrations = []
    if baseline_options.full_activation_offload:
        if baseline_options.trace_dir is None:
            raise ValueError("B2 activation offload requires a trace directory")
        from runtime.activation_offload import install_activation_offload_hooks

        activation_event_dir = baseline_options.trace_dir.parent / "activation_offload"
        activation_event_dir.mkdir(parents=True, exist_ok=True)
        activation_models = [
            ("generator_selected", unwrap_model(transformer), block_streamers.get("generator")),
            ("critic_train", unwrap_model(real_score_model), block_streamers.get("score")),
        ]
        activation_models = [
            record
            for record in activation_models
            if record[0] in baseline_options.activation_offload_components
        ]
        for logical_component, physical_model, block_streamer in activation_models:
            event_path = activation_event_dir / (
                f"rank_{accelerator.process_index:03d}_{logical_component}.jsonl"
            )
            registration = install_activation_offload_hooks(
                physical_model,
                logical_component=logical_component,
                event_path=event_path,
                pin_memory=True,
                min_offload_bytes=baseline_options.activation_offload_min_bytes,
                detailed_events=baseline_options.detailed_hotpath_trace,
                dynamic_excluded_storage_providers=(
                    (block_streamer.dynamic_excluded_storage_owner,)
                    if block_streamer is not None
                    else ()
                ),
            )
            activation_offload_registrations.append(registration)
            phase_tracer.record(
                "activation_offload_installed",
                logical_component=logical_component,
                event_path=str(event_path),
                save_generation="ORIGINAL_FORWARD",
                checkpoint_recompute_offload=False,
                synchronous_unpack=True,
                min_offload_bytes=baseline_options.activation_offload_min_bytes,
                detailed_events=baseline_options.detailed_hotpath_trace,
            )

    # initial gan
    gan_critic_trainable_params = None
    gan_base_critic_trainable_params = None
    gan_extra_critic_trainable_params = None
    if args.training_config.is_use_gan:
        gan_critic_trainable_params = {
            name for name, param in real_score_model.named_parameters() if param.requires_grad
        }
        gan_extra_critic_trainable_params = {
            name
            for name, param in real_score_model.named_parameters()
            if param.requires_grad and any(module in name for module in critic_trainable_modules)
        }
        gan_base_critic_trainable_params = gan_critic_trainable_params - gan_extra_critic_trainable_params

    # initial recycle noise
    recycle_vars = None
    if args.training_config.use_error_recycling:
        from types import SimpleNamespace

        num_grids = args.training_config.num_grids

        recycle_vars = SimpleNamespace()
        recycle_vars.recycle_inferece_timesteps, recycle_vars.recycle_sigmas = get_timesteps(
            num_inference_steps=num_grids, denoising_strength=1, shift=1.0
        )

        resolutions = set()
        for t, h, w in sampler.buckets.keys():
            base_h = h // 8
            base_w = w // 8
            resolutions.add((base_h, base_w))
            if args.training_config.is_enable_stage2:
                resolutions.add((base_h // 2, base_w // 2))
                resolutions.add((base_h // 4, base_w // 4))

        recycle_vars.latent_error_buffer = {
            resolution: {i: [] for i in range(num_grids)} for resolution in resolutions
        }
        recycle_vars.y_error_buffer = {resolution: {i: [] for i in range(num_grids)} for resolution in resolutions}

    def safe_item(value):
        return value.item() if hasattr(value, "item") else value

    generator_update_ratio = max(1, args.training_config.dfake_gen_update_ratio)
    generator_parameter_version = (
        (initial_global_step + generator_update_ratio - 1) // generator_update_ratio
        if args.training_config.is_train_dmd
        else initial_global_step
    )
    critic_parameter_version = (
        initial_global_step if args.training_config.is_train_dmd else None
    )
    semantic_state = {
        "generator_update_verified": False,
        "critic_update_verified": False,
        "critic_only_verified": False,
    }
    generator_gradient_observer = (
        TrainableGradientObserver(transformer)
        if baseline_options.semantic_assertions
        else None
    )
    critic_gradient_observer = (
        TrainableGradientObserver(real_score_model)
        if baseline_options.semantic_assertions and args.training_config.is_train_dmd
        else None
    )
    phase_tracer.record(
        "parameter_versions_initialized",
        global_step=global_step,
        generator_version=generator_parameter_version,
        critic_version=critic_parameter_version,
        synthetic_boundary_probe=bool(baseline_options.logical_start_step),
    )

    accelerator.wait_for_everyone()

    prof = None
    if args.training_config.profile_out_dir is not None:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.training_config.profile_out_dir),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        )

    nsys_capture_start_step = int(
        os.environ.get("DISTILLSCHEDULE_NSYS_CAPTURE_START_STEP", "-1")
    )
    nsys_capture_stop_step = int(
        os.environ.get("DISTILLSCHEDULE_NSYS_CAPTURE_STOP_STEP", "-1")
    )
    nsys_range_label = os.environ.get(
        "DISTILLSCHEDULE_NSYS_RANGE_LABEL", "distillschedule_profile"
    ).strip()
    if (nsys_capture_start_step >= 0) != (nsys_capture_stop_step >= 0):
        raise ValueError(
            "DISTILLSCHEDULE_NSYS_CAPTURE_START_STEP and "
            "DISTILLSCHEDULE_NSYS_CAPTURE_STOP_STEP must be set together"
        )
    if (
        nsys_capture_start_step >= 0
        and nsys_capture_stop_step <= nsys_capture_start_step
    ):
        raise ValueError("Nsight capture stop step must be greater than start step")
    if nsys_capture_start_step >= 0 and not nsys_range_label:
        raise ValueError("DISTILLSCHEDULE_NSYS_RANGE_LABEL must not be empty")
    nsys_range_open = False
    nsys_gate_capture_range = os.environ.get(
        "DISTILLSCHEDULE_NSYS_CAPTURE_RANGE", ""
    ).strip()
    if nsys_gate_capture_range not in {"", "cudaProfilerApi"}:
        raise ValueError(
            "DISTILLSCHEDULE_NSYS_CAPTURE_RANGE must be empty or cudaProfilerApi"
        )

    previous_step_end = time.perf_counter()
    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        # A resumed or synthetic logical-step run may start beyond the epoch
        # implied by the dataloader.  Guard both loop levels so reaching the
        # logical max step cannot enter an untraced extra batch.
        if global_step >= args.training_config.max_train_steps:
            break
        transformer.train()
        if args.training_config.is_train_dmd:
            real_score_model.train()
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.training_config.max_train_steps:
                break
            if global_step == nsys_capture_start_step:
                # Profiling-only synchronization keeps every distributed rank
                # inside its node-local Nsight capture window.  Normal runs
                # never execute this branch.
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push(nsys_range_label)
                    nsys_range_open = True
            if phase_tracer.timing_only and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_wall_start = time.perf_counter()
            data_wait_seconds = step_wall_start - previous_step_end
            phase_tracer.record(
                "training_step_start",
                global_step=global_step,
                epoch=epoch,
                data_wait_seconds=data_wait_seconds,
            )
            models_to_accumulate = [transformer]
            if args.training_config.is_train_dmd:
                models_to_accumulate.append(real_score_model)

            with torch.no_grad():
                latent_window_size = args.training_config.latent_window_size[0]

                # Get data samples
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None
                history_latents = None
                target_latents = None
                x0_latents = None
                model_input = None
                prompt_raws = None
                prompt_embeds = None
                indices_hidden_states = None
                indices_latents_history_short = None
                indices_latents_history_mid = None
                indices_latents_history_long = None
                latents_history_short = None
                latents_history_mid = None
                latents_history_long = None
                gan_vae_latents = None
                gan_prompt_embeds = None
                ode_latents = None
                ode_prompt_embeds = None
                text_prompt_raws = None
                text_prompt_embeds = None
                trace_source = None

                if args.data_config.use_stage3_dataset:
                    noisy_model_input_shape = (
                        args.training_config.train_batch_size,
                        16,
                        latent_window_size,
                        args.data_config.single_height // 8,
                        args.data_config.single_width // 8,
                    )

                    # For ODE
                    if args.training_config.is_use_ode_regression:
                        ode_latent_window_size = batch["ode_latent_window_size"][0]
                        ode_latents = batch["ode_latents"][0]
                        ode_prompt_embeds = batch["ode_prompt_embeds"][:1].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        assert args.training_config.train_batch_size == 1
                        assert ode_latent_window_size == latent_window_size

                    # For Text
                    if dataset_kwargs["text_folders"] and not args.training_config.is_only_ode_regression:
                        text_prompt_raws = batch["text_prompt_raws"]
                        text_prompt_embeds = batch["text_prompt_embeds"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )

                    # For GAN / ground-truth history.  The operation trace is
                    # authoritative: restoring StatefulDataLoader moves its sampler
                    # position, which must not silently substitute a different source
                    # sample for an operation-keyed timing replay.
                    if args.training_config.is_use_gan or args.training_config.is_use_gt_history:
                        trace_source = None
                        if operation_trace is not None:
                            trace_source = replay_stage3_source(
                                operation_trace, device=accelerator.device, dtype=weight_dtype
                            )
                            gan_vae_latents = trace_source["primary_target"]
                            gan_prompt_embeds = trace_source["text_embedding"]
                        else:
                            gan_vae_latents = batch["gan_vae_latents"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            gan_prompt_embeds = batch["gan_prompt_embeds"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                        if args.training_config.is_use_gt_history:
                            text_prompt_raws = batch["gan_prompt_raws"]
                            text_prompt_embeds = gan_prompt_embeds
                            if trace_source is not None:
                                gt_target_latents = trace_source["primary_target"]
                                gt_x0_latents = trace_source["primary_x0"]
                                gt_history_latents = trace_source["primary_history"]
                                gt_target_latents_2 = trace_source["secondary_target"]
                                gt_x0_latents_2 = trace_source["secondary_x0"]
                                gt_history_latents_2 = trace_source["secondary_history"]
                            else:
                                gt_target_latents = gan_vae_latents.to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                                gt_x0_latents = batch["gan_x0_latents"].to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                                gt_history_latents = batch["gan_history_latents"].to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                                gt_target_latents_2 = batch["gan_vae_latents_2"].to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                                gt_x0_latents_2 = batch["gan_x0_latents_2"].to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                                gt_history_latents_2 = batch["gan_history_latents_2"].to(
                                    accelerator.device, dtype=weight_dtype, non_blocking=True
                                )
                            assert gt_target_latents_2.shape[2] == args.training_config.num_critic_input_frames
                        assert gan_vae_latents.shape[2] == args.training_config.num_critic_input_frames
                        if (
                            trace_source is not None
                            and global_step == initial_global_step
                            and os.environ.get("DISTILLSCHEDULE_VALIDATE_TRACE_INPUT") == "1"
                        ):
                            source_hashes = {}
                            for name, value in trace_source.items():
                                actual_hash = tensor_sha256(value.squeeze(0))
                                expected_hash = operation_trace.manifest["tensors"][
                                    STAGE3_SOURCE_KEYS[name]
                                ]["sha256"]
                                if actual_hash != expected_hash:
                                    raise ValueError(
                                        f"operation-trace source materialization changed {name}"
                                    )
                                source_hashes[name] = actual_hash
                            phase_tracer.record(
                                "operation_trace_input_validated",
                                trace_hash=operation_trace.trace_hash,
                                source_authority="operation_trace",
                                source_hashes=source_hashes,
                            )

                elif args.data_config.use_stage1_dataset:
                    # Prepare prompt embeds
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)

                    # Prepare stage1 clean data
                    history_latents = batch["history_latents"].to(accelerator.device)
                    target_latents = batch["target_latents"].to(accelerator.device)
                    x0_latents = batch["x0_latents"].to(accelerator.device)
                    (
                        model_input,  # torch.Size([2, 16, 9, 60, 104])
                        indices_hidden_states,  # torch.Size([2, 9])
                        indices_latents_history_short,  # torch.Size([2, 2])
                        indices_latents_history_mid,  # torch.Size([2, 2])
                        indices_latents_history_long,  # torch.Size([2, 16])
                        latents_history_short,  # torch.Size([2, 16, 2, 60, 104])
                        latents_history_mid,  # torch.Size([2, 16, 2, 60, 104])
                        latents_history_long,  # torch.Size([2, 16, 16, 60, 104])
                    ) = prepare_stage1_clean_input_from_latents(
                        history_latents=history_latents,
                        target_latents=target_latents,
                        x0_latents=x0_latents,
                        latent_window_size=latent_window_size,
                        history_sizes=args.training_config.history_sizes,
                        is_random_drop=args.training_config.is_random_drop,
                        random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
                        random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
                        random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
                        is_keep_x0=True,
                        dtype=weight_dtype,
                        device=accelerator.device,
                    )
                    history_latents = None
                    target_latents = None
                    x0_latents = None
                    del history_latents
                    del target_latents
                    del x0_latents
                else:
                    raise NotImplementedError

                batch = None
                del batch

                if not args.data_config.use_stage3_dataset and (
                    args.training_config.offload or args.data_config.use_stage1_dataset
                ):
                    if vae is not None:
                        vae.to("cpu", non_blocking=True)
                    if text_encoder is not None:
                        text_encoder.to("cpu", non_blocking=True)
                    free_memory()

                # Set NULL Text
                if prompt_embeds is not None:
                    dropout_mask = (
                        torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device)
                        < args.data_config.caption_dropout_p
                    )
                    prompt_embeds[dropout_mask] = 0

                # To device
                if not args.training_config.is_train_dmd and not args.training_config.is_use_ode_regression:
                    model_input = model_input.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                    indices_hidden_states = indices_hidden_states.to(accelerator.device, non_blocking=True)
                    indices_latents_history_short = indices_latents_history_short.to(
                        accelerator.device, non_blocking=True
                    )
                    indices_latents_history_mid = indices_latents_history_mid.to(accelerator.device, non_blocking=True)
                    indices_latents_history_long = indices_latents_history_long.to(
                        accelerator.device, non_blocking=True
                    )
                    latents_history_short = latents_history_short.to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    latents_history_mid = latents_history_mid.to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    latents_history_long = latents_history_long.to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                if prompt_embeds is not None:
                    prompt_embeds = prompt_embeds.to(accelerator.device, non_blocking=True)

                # Prepare final data for training
                use_clean_input = False
                if args.training_config.is_train_dmd or args.training_config.is_use_ode_regression:
                    noisy_model_input_list = None
                    sigmas_list = None
                    timesteps_list = None
                    targets_list = None
                    latents_history_short = None
                    latents_history_mid = None
                    latents_history_long = None
                else:
                    if args.training_config.is_enable_stage2:
                        (
                            noisy_model_input_list,
                            sigmas_list,
                            timesteps_list,
                            targets_list,
                            latents_history_short,
                            latents_history_mid,
                            latents_history_long,
                        ) = prepare_stage2_noise_input(
                            args=args,
                            scheduler=noise_scheduler_copy,
                            latents=model_input,
                            pyramid_stage_num=args.training_config.stage2_num_stages,
                            stage2_sample_ratios=args.training_config.stage2_sample_ratios,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid,
                            latents_history_long=latents_history_long,
                            latent_window_size=latent_window_size,
                            is_navit_pyramid=args.training_config.is_navit_pyramid,
                            is_efficient_sample=args.training_config.efficient_sample,
                        )
                    else:
                        (
                            noisy_model_input_list,
                            sigmas_list,
                            timesteps_list,
                            targets_list,
                            latents_history_short,
                            latents_history_mid,
                            latents_history_long,
                            use_clean_input,
                        ) = prepare_stage1_noise_input(
                            args=args,
                            model_input=model_input,
                            noise_scheduler=noise_scheduler_copy,
                            recycle_vars=recycle_vars,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid,
                            latents_history_long=latents_history_long,
                            latent_window_size=latent_window_size,
                            is_keep_x0=True,
                        )

            with accelerator.accumulate(models_to_accumulate):
                # Predict the noise residual
                if not args.training_config.is_train_dmd and not args.training_config.is_use_ode_regression:
                    assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)
                    logs = _flow_loss(
                        args=args,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        transformer=transformer,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_masks=None,
                        noisy_model_input_list=noisy_model_input_list,
                        sigmas_list=sigmas_list,
                        timesteps_list=timesteps_list,
                        targets_list=targets_list,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        recycle_vars=recycle_vars,
                        global_step=global_step,
                        noise_scheduler_copy=noise_scheduler_copy,
                        use_clean_input=use_clean_input,
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                elif args.training_config.is_use_ode_regression and args.training_config.is_only_ode_regression:
                    if vae is not None:
                        vae.to("cpu", non_blocking=True)
                    if text_encoder is not None:
                        text_encoder.to("cpu", non_blocking=True)

                    _, logs = _ode_regression_loss(
                        args=args,
                        accelerator=accelerator,
                        transformer=transformer,
                        scheduler=noise_scheduler_copy,
                        noise=torch.randn(noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype),
                        weight_dtype=weight_dtype,
                        # For Stage 1
                        is_keep_x0=True,
                        history_sizes=args.training_config.history_sizes,
                        # For Stage 2
                        stage2_num_stages=args.training_config.stage2_num_stages,
                        # For ODE Main
                        last_step_only=args.training_config.dmd_last_step_only,
                        use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                        time_shift_type=args.training_config.time_shift_type,
                        is_backward_grad=True,
                        ode_regression_weight=args.training_config.ode_regression_weight,
                        ode_latents=ode_latents,
                        ode_prompt_embeds=ode_prompt_embeds,
                        ode_num_latent_sections_min=args.training_config.ode_num_latent_sections_min,
                        ode_num_latent_sections_max=args.training_config.ode_num_latent_sections_max,
                        # For Dynamic Num Sections
                        ode_dynamic_alpha=args.training_config.ode_dynamic_alpha,
                        ode_dynamic_beta=args.training_config.ode_dynamic_beta,
                        ode_dynamic_sample_type=args.training_config.ode_dynamic_sample_type,
                        global_step=global_step,
                        ode_dynamic_step=args.training_config.ode_dynamic_step,
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                else:
                    TRAIN_GENERATOR = global_step % args.training_config.dfake_gen_update_ratio == 0
                    semantic_critic_only_generator_sha = None
                    if (
                        baseline_options.semantic_assertions
                        and not TRAIN_GENERATOR
                        and not semantic_state["critic_only_verified"]
                    ):
                        generator_gradients = parameter_gradient_diagnostics(transformer)
                        if generator_gradients["trainable_grad_present"] != 0:
                            raise AssertionError(
                                "Generator gradients must be None at Critic-only step entry"
                            )
                        semantic_critic_only_generator_sha = trainable_sha256(transformer)
                        phase_tracer.record(
                            "semantic_critic_only_entry",
                            global_step=global_step,
                            generator_sha256=semantic_critic_only_generator_sha,
                            generator_gradients=generator_gradients,
                            generator_version=generator_parameter_version,
                        )
                    USE_GAN = args.training_config.is_use_gan and global_step >= args.training_config.gan_start_step
                    USE_REWARD = (
                        args.training_config.is_use_reward_model
                        and global_step >= args.training_config.reward_start_step
                    )
                    USE_GT_HIST = (
                        args.training_config.is_use_gt_history
                        and random.random() < args.training_config.use_gt_history_ratio
                    )

                    VISUALIZE = (
                        global_step % args.training_config.log_iters == 0 and not args.training_config.no_visualize
                    )
                    logs = {}

                    if accelerator.is_main_process:
                        if (
                            args.training_config.is_enable_cold_start
                            and global_step < args.training_config.cold_start_step
                        ):
                            num_rollout_sections = (
                                args.training_config.dmd_num_latent_sections_min + 1
                                if args.training_config.stage_cold_start_step is not None
                                and global_step >= args.training_config.stage_cold_start_step
                                else args.training_config.dmd_num_latent_sections_min
                            )
                        else:
                            num_rollout_sections = sample_dynamic_dmd_num_latent_sections(
                                min_sections=args.training_config.dmd_num_latent_sections_min,
                                max_sections=args.training_config.dmd_num_latent_sections_max,
                                dmd_dynamic_alpha=args.training_config.dmd_dynamic_alpha,
                                dmd_dynamic_beta=args.training_config.dmd_dynamic_beta,
                                dmd_dynamic_sample_type=args.training_config.dmd_dynamic_sample_type,
                                global_step=global_step,
                                dmd_dynamic_step=args.training_config.dmd_dynamic_step,
                                device=accelerator.device,
                            )
                        num_rollout_sections = torch.tensor(num_rollout_sections, device=accelerator.device)
                    else:
                        num_rollout_sections = torch.tensor(0, device=accelerator.device)

                    num_rollout_sections = broadcast(num_rollout_sections, from_process=0).item()
                    logs["num_rollout_sections"] = num_rollout_sections

                    if args.data_config.use_stage3_dataset:
                        prompt_raws = text_prompt_raws
                        prompt_embeds = text_prompt_embeds

                    if TRAIN_GENERATOR:
                        generator_update_gate_range_open = False
                        if generator_update_gate.enabled:
                            generator_update_gate.validate_runtime(
                                global_step=global_step,
                                world_size=accelerator.num_processes,
                                global_batch=total_batch_size,
                                gradient_checkpointing=args.training_config.gradient_checkpointing,
                                is_train_dmd=args.training_config.is_train_dmd,
                                dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                                is_use_gan=args.training_config.is_use_gan,
                            )
                            phase_tracer.record(
                                "generator_update_gate_start",
                                global_step=global_step,
                                backing_components=list(
                                    baseline_options.immutable_model_backing_components
                                ),
                            )
                            accelerator.wait_for_everyone()
                            if (
                                os.environ.get("DISTILLSCHEDULE_NSYS_ANNOTATE", "0")
                                == "1"
                                and torch.cuda.is_available()
                                and accelerator.is_local_main_process
                            ):
                                if nsys_gate_capture_range == "cudaProfilerApi":
                                    torch.cuda.cudart().cudaProfilerStart()
                                torch.cuda.nvtx.range_push(
                                    generator_update_gate.nvtx_label
                                )
                                generator_update_gate_range_open = True
                        extras_list = []

                        if USE_GAN:
                            for name, param in real_score_model.named_parameters():
                                if name in gan_critic_trainable_params:
                                    param.requires_grad = False

                        if args.training_config.is_use_ode_regression:
                            if args.training_config.dmd_is_low_vram_mode:
                                vram_manager.move_to_cpu(real_score_model)
                                vram_manager.move_to_gpu(transformer, accelerator.device)

                            _, ode_log_dict = _ode_regression_loss(
                                args=args,
                                accelerator=accelerator,
                                transformer=transformer,
                                scheduler=noise_scheduler_copy,
                                noise=torch.randn(
                                    noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype
                                ),
                                weight_dtype=weight_dtype,
                                # For Stage 1
                                is_keep_x0=True,
                                history_sizes=args.training_config.history_sizes,
                                # For Stage 2
                                stage2_num_stages=args.training_config.stage2_num_stages,
                                stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                                # For ODE Main
                                last_step_only=args.training_config.dmd_last_step_only,
                                use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                                time_shift_type=args.training_config.time_shift_type,
                                is_backward_grad=True,
                                ode_regression_weight=args.training_config.ode_regression_weight,
                                ode_latents=ode_latents,
                                ode_prompt_embeds=ode_prompt_embeds,
                                ode_num_latent_sections_min=args.training_config.ode_num_latent_sections_min,
                                ode_num_latent_sections_max=args.training_config.ode_num_latent_sections_max,
                                # For Dynamic ODE Length
                                ode_dynamic_alpha=args.training_config.ode_dynamic_alpha,
                                ode_dynamic_beta=args.training_config.ode_dynamic_beta,
                                ode_dynamic_sample_type=args.training_config.ode_dynamic_sample_type,
                                global_step=global_step,
                                ode_dynamic_step=args.training_config.ode_dynamic_step,
                            )
                            logs.update(ode_log_dict)

                            ode_log_dict = None
                            del ode_log_dict

                        phase_tracer.record(
                            "generator_loss_start",
                            global_step=global_step,
                            num_rollout_sections=num_rollout_sections,
                            use_gt_history=USE_GT_HIST,
                        )
                        generator_loss, generator_log_dict = _generator_loss(
                            args=args,
                            accelerator=accelerator,
                            real_fake_score_model=real_score_model,
                            transformer=transformer,
                            scheduler=noise_scheduler_copy,
                            noise=(
                                operation_trace.tensor(f"step/{global_step}/generator/entry_noise").to(
                                    accelerator.device, dtype=weight_dtype
                                )
                                if operation_trace is not None
                                else torch.randn(noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype)
                            ),
                            prompt_embeds=prompt_embeds,
                            operation_trace=operation_trace,
                            trace_prefix=f"step/{global_step}/generator",
                            negative_prompt_embeds=negative_prompt_embeds,
                            # For VRAM manager
                            dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                            vram_manager=vram_manager,
                            dmd_is_offload_grad=args.training_config.dmd_is_offload_grad,
                            is_gan_low_vram_mode=args.training_config.is_gan_low_vram_mode,
                            # For Stage 1
                            is_keep_x0=True,
                            history_sizes=args.training_config.history_sizes,
                            # For Stage 2
                            is_enable_stage2=args.training_config.is_enable_stage2,
                            stage2_num_stages=args.training_config.stage2_num_stages,
                            stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                            # For DMD Main
                            denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                            last_step_only=args.training_config.dmd_last_step_only,
                            last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                            timestep_shift=args.training_config.dmd_timestep_shift,
                            use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                            time_shift_type=args.training_config.time_shift_type,
                            fake_guidance_scale=args.training_config.fake_guidance_scale,
                            real_guidance_scale=args.training_config.real_guidance_scale,
                            num_critic_input_frames=args.training_config.num_critic_input_frames,
                            num_rollout_sections=num_rollout_sections,
                            is_skip_first_section=args.training_config.is_skip_first_section,
                            is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                            # For Easy Anti-Drifting
                            is_corrupt_history_latents=args.training_config.corrupt_history,
                            is_add_saturation=args.training_config.is_add_saturation,
                            # For GT History
                            is_use_gt_history=USE_GT_HIST,
                            gt_history_latents=gt_history_latents,
                            gt_target_latents=gt_target_latents,
                            gt_x0_latents=gt_x0_latents,
                            # For VAE Re-Encode
                            vae=vae,
                            is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                            # For Multi Stage Backward Simulated
                            is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                            # For Consistency Align
                            is_consistency_align=args.training_config.is_consistency_align,
                            consistentcy_align_weight=args.training_config.consistentcy_align_weight,
                            # For Smoothness
                            is_smoothness_loss=args.training_config.is_smoothness_loss,
                            smoothness_loss_weight=args.training_config.smoothness_loss_weight,
                            # For KV Cache
                            use_kv_cache=args.validation_config.use_kv_cache,
                            # For Mean-Variance Regularization
                            is_mean_var_regular=args.training_config.is_mean_var_regular,
                            mean_var_regular_weight=args.training_config.mean_var_regular_weight,
                            regular_mean=args.training_config.regular_mean,
                            regular_var=args.training_config.regular_var,
                            is_x0_mean_var_regular=args.training_config.is_x0_mean_var_regular,
                            mean_var_regular_x0_weight=args.training_config.mean_var_regular_x0_weight,
                            regular_x0_mean=args.training_config.regular_x0_mean,
                            regular_x0_var=args.training_config.regular_x0_var,
                            #
                            is_chunk_mean_var_regular=args.training_config.is_chunk_mean_var_regular,
                            chunk_mean_var_regular_weight=args.training_config.chunk_mean_var_regular_weight,
                            chunk_regular_mean=args.training_config.chunk_regular_mean,
                            chunk_regular_var=args.training_config.chunk_regular_var,
                            is_chunk_x0_mean_var_regular=args.training_config.is_chunk_x0_mean_var_regular,
                            chunk_mean_var_regular_x0_weight=args.training_config.chunk_mean_var_regular_x0_weight,
                            chunk_regular_x0_mean=args.training_config.chunk_regular_x0_mean,
                            chunk_regular_x0_var=args.training_config.chunk_regular_x0_var,
                            # For GAN
                            is_use_gan=USE_GAN,
                            gan_prompt_embeds=gan_prompt_embeds,
                            gan_g_weight=args.training_config.gan_g_weight,
                            # For Reward
                            is_use_reward_model=USE_REWARD,
                            reward_model=reward_model,
                            reward_weight_vq=args.training_config.reward_weight_vq,
                            reward_weight_mq=args.training_config.reward_weight_mq,
                            reward_weight_ta=args.training_config.reward_weight_ta,
                            reward_texts=prompt_raws,
                            # For Decouple DMD
                            is_decouple_dmd=args.training_config.is_decouple_dmd,
                            decouple_ca_start_step=args.training_config.decouple_ca_start_step,
                            decouple_ca_end_step=args.training_config.decouple_ca_end_step,
                            # For Dynamic Timestep
                            is_forcing_low_renoise=args.training_config.generator_is_forcing_low_renoise,
                            dynamic_alpha=args.training_config.generator_dynamic_alpha,
                            dynamic_beta=args.training_config.generator_dynamic_beta,
                            dynamic_sample_type=args.training_config.generator_dynamic_sample_type,
                            global_step=global_step,
                            dynamic_step=args.training_config.generator_dynamic_step,
                        )

                        phase_tracer.record(
                            "generator_loss_end",
                            global_step=global_step,
                            loss=float(generator_loss.detach().mean().item()),
                        )
                        semantic_generator_sha_before = None
                        semantic_generator_gradient_observation = None
                        if (
                            baseline_options.semantic_assertions
                            and not semantic_state["generator_update_verified"]
                        ):
                            generator_gradients = parameter_gradient_diagnostics(transformer)
                            if generator_gradients["trainable_grad_present"] != 0:
                                raise AssertionError(
                                    "Generator gradients were live before backward"
                                )
                            if generator_gradients["frozen_grad_present"] != 0:
                                raise AssertionError(
                                    "frozen Generator backbone had a gradient before backward"
                                )
                            semantic_generator_sha_before = trainable_sha256(transformer)
                            assert generator_gradient_observer is not None
                            generator_gradient_observer.begin()
                            phase_tracer.record(
                                "semantic_generator_pre_backward",
                                global_step=global_step,
                                generator_sha256=semantic_generator_sha_before,
                                generator_gradients=generator_gradients,
                                generator_version=generator_parameter_version,
                            )
                        if os.environ.get("DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD", "0") == "1":
                            with _helios_stage_range(
                                global_step,
                                "runtime",
                                "allocator_cache_release",
                                target="generator_backward",
                            ):
                                torch.cuda.empty_cache()
                        with _helios_stage_range(global_step, "generator", "train_backward"):
                            accelerator.backward(generator_loss)
                        generator_module = unwrap_model(transformer)
                        expected_direct_lora_b_accumulators = int(
                            getattr(
                                generator_module,
                                "_distillschedule_direct_lora_b_accumulator_parameter_count",
                                0,
                            )
                        )
                        if expected_direct_lora_b_accumulators:
                            contribution_counts = []
                            observed_direct_lora_b_accumulators = 0
                            observed_direct_lora_b_modules = 0
                            for module in generator_module.modules():
                                counts = getattr(
                                    module,
                                    "_distillschedule_direct_lora_b_last_contribution_counts",
                                    None,
                                )
                                if counts is None:
                                    continue
                                observed_direct_lora_b_modules += 1
                                observed_direct_lora_b_accumulators += len(counts)
                                contribution_counts.extend(counts.values())
                            if (
                                observed_direct_lora_b_accumulators
                                != expected_direct_lora_b_accumulators
                            ):
                                raise AssertionError(
                                    "Generator direct LoRA-B accumulation coverage drift: "
                                    f"expected {expected_direct_lora_b_accumulators}, "
                                    f"observed {observed_direct_lora_b_accumulators}"
                                )
                            phase_tracer.record(
                                "generator_direct_lora_b_accumulation_verified",
                                global_step=global_step,
                                module_count=observed_direct_lora_b_modules,
                                parameter_count=observed_direct_lora_b_accumulators,
                                contribution_count_min=min(contribution_counts),
                                contribution_count_max=max(contribution_counts),
                            )
                        if semantic_generator_sha_before is not None:
                            assert generator_gradient_observer is not None
                            semantic_generator_gradient_observation = (
                                generator_gradient_observer.end()
                            )
                            if semantic_generator_gradient_observation["hook_calls"] == 0:
                                raise AssertionError(
                                    "Generator backward delivered no trainable gradients"
                                )
                            phase_tracer.record(
                                "semantic_generator_gradient_observed",
                                global_step=global_step,
                                observation=semantic_generator_gradient_observation,
                            )
                        phase_tracer.record("generator_backward_end", global_step=global_step)

                        generator_grad_norm = None
                        if accelerator.sync_gradients:
                            generator_params_to_clip = transformer.parameters()
                            generator_grad_norm = accelerator.clip_grad_norm_(
                                generator_params_to_clip, args.training_config.max_grad_norm
                            )

                        generator_log_dict["generator_loss"] = generator_loss
                        if generator_grad_norm is not None:
                            generator_log_dict["generator_grad_norm"] = generator_grad_norm

                        extra = generator_log_dict
                        extras_list.append(extra)
                        generator_log_dict = merge_dict_list(extras_list)
                        with _helios_stage_range(global_step, "generator", "optimizer_step"):
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                        generator_parameter_version += 1
                        if semantic_generator_sha_before is not None:
                            semantic_generator_sha_after = trainable_sha256(transformer)
                            generator_gradients_after = parameter_gradient_diagnostics(transformer)
                            if semantic_generator_sha_after == semantic_generator_sha_before:
                                raise AssertionError(
                                    "Generator trainable parameters did not change"
                                )
                            if generator_gradients_after["trainable_grad_present"] != 0:
                                raise AssertionError(
                                    "Generator gradients were not cleared with set_to_none"
                                )
                            semantic_state["generator_update_verified"] = True
                            phase_tracer.record(
                                "semantic_generator_post_step",
                                global_step=global_step,
                                before_sha256=semantic_generator_sha_before,
                                after_sha256=semantic_generator_sha_after,
                                generator_gradients=generator_gradients_after,
                                gradient_observation=semantic_generator_gradient_observation,
                                generator_version=generator_parameter_version,
                            )
                        generator_step_event = {
                            "global_step": global_step,
                            "loss": float(generator_log_dict["generator_loss"].mean().item()),
                            "generator_version": generator_parameter_version,
                        }
                        if not phase_tracer.timing_only:
                            generator_step_event["checksum"] = trainable_checksum(transformer)
                        phase_tracer.record("generator_optimizer_step", **generator_step_event)

                        base_logs = {
                            # "generator_lr": lr_scheduler.get_last_lr()[0],
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": safe_item(generator_log_dict["generator_grad_norm"]),
                        }
                        if args.training_config.is_decouple_dmd:
                            base_logs.update(
                                {
                                    "dmdtrain_ca_gradient_norm": safe_item(
                                        generator_log_dict["dmdtrain_ca_gradient_norm"]
                                    ),
                                    "dmdtrain_dm_gradient_norm": safe_item(
                                        generator_log_dict["dmdtrain_dm_gradient_norm"]
                                    ),
                                }
                            )
                        else:
                            base_logs["dmdtrain_gradient_norm"] = safe_item(
                                generator_log_dict["dmdtrain_gradient_norm"]
                            )
                        logs.update(base_logs)
                        base_logs = None
                        del base_logs

                        if args.training_config.is_smoothness_loss or USE_GAN or USE_REWARD:
                            logs["dmd_loss_raw"] = generator_log_dict["dmd_loss_raw"]

                        if args.training_config.is_consistency_align:
                            logs["consistency_align_loss"] = generator_log_dict["consistency_align_loss"]

                        if args.training_config.is_smoothness_loss:
                            logs["smoothness_loss"] = generator_log_dict["smoothness_loss"]

                        if args.training_config.is_mean_var_regular:
                            logs["kl_mean_var_loss"] = generator_log_dict["kl_mean_var_loss"]
                            logs["pred_mean_avg"] = generator_log_dict["pred_mean_avg"]
                            logs["pred_var_avg"] = generator_log_dict["pred_var_avg"]

                            if args.training_config.is_x0_mean_var_regular:
                                logs["kl_mean_var_x0_loss"] = generator_log_dict["kl_mean_var_x0_loss"]
                                logs["pred_x0_mean_avg"] = generator_log_dict["pred_x0_mean_avg"]
                                logs["pred_x0_var_avg"] = generator_log_dict["pred_x0_var_avg"]

                        if args.training_config.is_chunk_mean_var_regular:
                            logs["kl_chunk_mean_var_loss"] = generator_log_dict["kl_chunk_mean_var_loss"]
                            logs["pred_chunk_mean_avg"] = generator_log_dict["pred_chunk_mean_avg"]
                            logs["pred_chunk_var_avg"] = generator_log_dict["pred_chunk_var_avg"]

                            if args.training_config.is_chunk_x0_mean_var_regular:
                                logs["kl_chunk_mean_var_x0_loss"] = generator_log_dict["kl_chunk_mean_var_x0_loss"]
                                logs["pred_chunk_x0_mean_avg"] = generator_log_dict["pred_chunk_x0_mean_avg"]
                                logs["pred_chunk_x0_var_avg"] = generator_log_dict["pred_chunk_x0_var_avg"]

                        if USE_GAN:
                            logs["gan_G_loss"] = generator_log_dict["gan_G_loss"]

                        if USE_REWARD:
                            logs["reward_score_vq"] = generator_log_dict["reward_score_vq"]
                            logs["reward_score_mq"] = generator_log_dict["reward_score_mq"]
                            logs["reward_score_ta"] = generator_log_dict["reward_score_ta"]

                        generator_loss = None
                        generator_grad_norm = None
                        del generator_loss
                        del generator_grad_norm
                        free_memory()

                        if generator_update_gate.enabled:
                            if (
                                semantic_generator_sha_before is None
                                or semantic_generator_sha_after is None
                                or semantic_generator_gradient_observation is None
                            ):
                                raise AssertionError(
                                    "Generator update gate requires pre/post checksums and gradient evidence"
                                )
                            if len(activation_offload_registrations) != 1:
                                raise AssertionError(
                                    "Generator update gate requires exactly one activation offloader"
                                )
                            activation_registration = activation_offload_registrations[0]
                            if (
                                activation_registration.offloader.logical_component
                                != "generator_selected"
                            ):
                                raise AssertionError(
                                    "Score/Teacher saved-activation offload is forbidden"
                                )
                            activation_stats = dict(
                                activation_registration.offloader.stats
                            )
                            if activation_stats["cpu_copy_count"] <= 0:
                                raise AssertionError(
                                    "Generator activation policy made no CPU copy"
                                )
                            if activation_stats["unpack_count"] <= 0:
                                raise AssertionError(
                                    "Generator backward unpacked no saved activation"
                                )
                            backing_evidence = {}
                            frozen_block_streaming_evidence = {}
                            if immutable_model_backings:
                                if set(immutable_model_backings) != {
                                    "generator",
                                    "score",
                                }:
                                    raise AssertionError(
                                        "Generator update gate requires both immutable model backings"
                                    )
                                backing_evidence = {
                                    name: backing.evidence_summary()
                                    for name, backing in immutable_model_backings.items()
                                }
                                for name, evidence in backing_evidence.items():
                                    positive_phase_releases = [
                                        item
                                        for item in evidence["release_observations"]
                                        if item["reason"] == "whole_model_move_to_cpu"
                                        and item["cuda_rebound_without_d2h_bytes"] > 0
                                    ]
                                    if not positive_phase_releases:
                                        raise AssertionError(
                                            f"{name} never rebound a CUDA frozen backbone to shared CPU backing"
                                        )
                                    if not any(
                                        item["cuda_allocator_drop_bytes"] > 0
                                        for item in positive_phase_releases
                                    ):
                                        raise AssertionError(
                                            f"{name} rebind produced no physical CUDA allocator drop"
                                        )
                                if backing_evidence["generator"]["stats"]["materialize_count"] < 2:
                                    raise AssertionError(
                                        "Generator was not reloaded for backward recomputation"
                                    )
                                if backing_evidence["score"]["stats"]["materialize_count"] < 1:
                                    raise AssertionError("Score was never materialized for no-grad")
                            elif baseline_options.immutable_model_backing_components:
                                raise AssertionError(
                                    "configured immutable model backings were not installed"
                                )

                            if baseline_options.frozen_block_streaming_window:
                                expected_streamers = set(
                                    baseline_options.frozen_block_streaming_components
                                )
                                if set(block_streamers) != expected_streamers:
                                    raise AssertionError(
                                        "Generator update streamer set drift: "
                                        f"expected={sorted(expected_streamers)}, "
                                        f"observed={sorted(block_streamers)}"
                                    )
                                for name, streamer in block_streamers.items():
                                    stats = dict(streamer.stats)
                                    if (
                                        stats["captured_bytes"] <= 0
                                        or stats["materialize_count"] <= 0
                                        or stats["release_count"] <= 0
                                        or stats["resident_bytes"] != 0
                                    ):
                                        raise AssertionError(
                                            f"{name} B3 streamer evidence is incomplete: {stats}"
                                        )
                                    if streamer.host_policy.value != "HOST_SHARED_IMMUTABLE":
                                        raise AssertionError(
                                            f"{name} B3 host policy drift"
                                        )
                                    frozen_block_streaming_evidence[name] = {
                                        "window_size": int(streamer.window_size),
                                        "host_policy": streamer.host_policy.value,
                                        "streamed_block_indices": sorted(
                                            streamer.streamed_block_indices
                                        ),
                                        "stats": stats,
                                    }

                            torch.cuda.synchronize(accelerator.device)
                            accelerator.wait_for_everyone()
                            if generator_update_gate_range_open:
                                torch.cuda.nvtx.range_pop()
                                if nsys_gate_capture_range == "cudaProfilerApi":
                                    torch.cuda.cudart().cudaProfilerStop()
                                generator_update_gate_range_open = False

                            result = {
                                "schema_version": (
                                    "distillschedule.generator_update.hybrid_residency.rank.v1"
                                    if baseline_options.hybrid_residency
                                    else (
                                        "distillschedule.generator_update.frozen_block_streaming.rank.v1"
                                        if baseline_options.frozen_block_streaming_window
                                        else "distillschedule.generator_update.shared_backing.rank.v1"
                                    )
                                ),
                                "status": "PASS",
                                "rank": accelerator.process_index,
                                "local_rank": accelerator.local_process_index,
                                "world_size": accelerator.num_processes,
                                "global_step": global_step,
                                "contract": {
                                    "scope": "isolated synthetic-boundary Generator update",
                                    "global_batch": total_batch_size,
                                    "whole_model_switching": True,
                                    "hybrid_residency": bool(
                                        baseline_options.hybrid_residency
                                    ),
                                    "immutable_model_backing_components": list(
                                        baseline_options.immutable_model_backing_components
                                    ),
                                    "frozen_block_streaming": bool(
                                        baseline_options.frozen_block_streaming_window
                                    ),
                                    "frozen_block_streaming_window": int(
                                        baseline_options.frozen_block_streaming_window
                                    ),
                                    "frozen_block_streaming_components": list(
                                        baseline_options.frozen_block_streaming_components
                                    ),
                                    "frozen_block_streaming_block_count": int(
                                        baseline_options.frozen_block_streaming_block_count
                                    ),
                                    "frozen_block_streaming_indices": list(
                                        sorted(
                                            next(iter(block_streamers.values())).streamed_block_indices
                                        )
                                        if block_streamers
                                        else ()
                                    ),
                                    "frozen_block_memory_snapshots": bool(
                                        baseline_options.frozen_block_memory_snapshots
                                    ),
                                    "frozen_block_host_policy": (
                                        baseline_options.frozen_block_host_policy
                                        if baseline_options.frozen_block_streaming_window
                                        else None
                                    ),
                                    "gradient_checkpointing": True,
                                    "compute_path": (
                                        "b0_vectorized_split_modulation"
                                        if baseline_options.b0_compute_path
                                        else "memory_bounded_split_modulation"
                                    ),
                                    "memory_bounded_compute_tiling": not (
                                        baseline_options.b0_compute_path
                                    ),
                                    "generator_activation_offload_min_bytes": baseline_options.activation_offload_min_bytes,
                                    "score_teacher_activation_offload": False,
                                    "step15b": "Not Run",
                                },
                                "generator_sha256_before": semantic_generator_sha_before,
                                "generator_sha256_after": semantic_generator_sha_after,
                                "generator_parameter_version": generator_parameter_version,
                                "generator_gradient_observation": semantic_generator_gradient_observation,
                                "activation_offload": activation_stats,
                                "immutable_model_backings": backing_evidence,
                                "frozen_block_streaming_evidence": (
                                    frozen_block_streaming_evidence
                                ),
                                "generator_parameters_after": parameter_summary(transformer),
                                "score_parameters_after": parameter_summary(real_score_model),
                                "cuda": {
                                    "allocated_bytes": torch.cuda.memory_allocated(
                                        accelerator.device
                                    ),
                                    "reserved_bytes": torch.cuda.memory_reserved(
                                        accelerator.device
                                    ),
                                    "max_allocated_bytes": torch.cuda.max_memory_allocated(
                                        accelerator.device
                                    ),
                                    "max_reserved_bytes": torch.cuda.max_memory_reserved(
                                        accelerator.device
                                    ),
                                },
                            }
                            destination = generator_update_gate.write_rank_result(
                                accelerator.process_index, result
                            )
                            phase_tracer.record(
                                "generator_update_gate_complete",
                                global_step=global_step,
                                result_path=str(destination),
                                step15b="Not Run",
                            )
                            accelerator.wait_for_everyone()
                            activation_registration.remove()
                            phase_tracer.record(
                                "requested_stop_after_generator_update",
                                global_step=global_step,
                                step15b="Not Run",
                            )
                            for logical_component, streamer in block_streamers.items():
                                streamer.assert_idle()
                                phase_tracer.record(
                                    "frozen_block_streamer_final",
                                    logical_component=logical_component,
                                    stats=dict(streamer.stats),
                                )
                                streamer.remove()
                            for logical_component, backing in immutable_model_backings.items():
                                backing.assert_integrity()
                                phase_tracer.record(
                                    "immutable_model_backing_final",
                                    logical_component=logical_component,
                                    evidence=backing.evidence_summary(),
                                )
                            phase_tracer.record(
                                "training_complete_skip_final_save",
                                global_step=global_step,
                            )
                            return

                    if USE_GAN:
                        for name, param in real_score_model.named_parameters():
                            if name in gan_critic_trainable_params:
                                param.requires_grad = True

                    # Train the critic
                    extras_list = []
                    phase_tracer.record(
                        "critic_loss_start",
                        global_step=global_step,
                        num_rollout_sections=num_rollout_sections,
                        use_gt_history=USE_GT_HIST,
                    )
                    semantic_critic_sha_before = None
                    semantic_critic_gradient_observation = None
                    if (
                        baseline_options.semantic_assertions
                        and not semantic_state["critic_update_verified"]
                    ):
                        critic_gradients = parameter_gradient_diagnostics(real_score_model)
                        if critic_gradients["trainable_grad_present"] != 0:
                            raise AssertionError("Critic gradients were live before backward")
                        if critic_gradients["frozen_grad_present"] != 0:
                            raise AssertionError(
                                "frozen Score backbone had a gradient before backward"
                            )
                        semantic_critic_sha_before = trainable_sha256(real_score_model)
                        assert critic_gradient_observer is not None
                        # GAN low-VRAM performs both Critic backwards and the
                        # DeepSpeed commit inside _critic_loss.  Begin before
                        # entering the official function so the same
                        # lifetime-neutral hooks cover both that path and the
                        # ordinary outer-backward path.
                        critic_gradient_observer.begin()
                        phase_tracer.record(
                            "semantic_critic_pre_backward",
                            global_step=global_step,
                            critic_sha256=semantic_critic_sha_before,
                            critic_gradients=critic_gradients,
                            critic_version=critic_parameter_version,
                        )
                    critic_loss, critic_log_dict = _critic_loss(
                        args=args,
                        critic_accelerator=critic_accelerator,
                        fake_score_model=real_score_model,
                        transformer=transformer,
                        scheduler=critic_noise_scheduler,
                        noise=(
                            operation_trace.tensor(f"step/{global_step}/critic/entry_noise").to(
                                critic_accelerator.device, dtype=weight_dtype
                            )
                            if operation_trace is not None
                            else torch.randn(
                                noisy_model_input_shape, device=critic_accelerator.device, dtype=weight_dtype
                            )
                        ),
                        prompt_embeds=prompt_embeds,
                        operation_trace=operation_trace,
                        trace_prefix=f"step/{global_step}/critic",
                        phase_resident_generator_rollout=(
                            baseline_options.phase_resident_critic_generator
                        ),
                        critic_phase_offload_generator_lp_state=(
                            baseline_options.critic_phase_offload_generator_lp_state
                        ),
                        # For VRAM manager
                        dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                        vram_manager=vram_manager,
                        is_gan_low_vram_mode=args.training_config.is_gan_low_vram_mode,
                        # For Stage 1
                        is_keep_x0=True,
                        history_sizes=args.training_config.history_sizes,
                        # For Stage 2
                        is_enable_stage2=args.training_config.is_enable_stage2,
                        stage2_num_stages=args.training_config.stage2_num_stages,
                        stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                        # For DMD Main
                        denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                        last_step_only=args.training_config.dmd_last_step_only,
                        last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                        timestep_shift=args.training_config.dmd_timestep_shift,
                        use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                        time_shift_type=args.training_config.time_shift_type,
                        num_critic_input_frames=args.training_config.num_critic_input_frames,
                        num_rollout_sections=num_rollout_sections,
                        is_skip_first_section=args.training_config.is_skip_first_section,
                        is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                        # For Easy Anti-Drifting
                        is_corrupt_history_latents=args.training_config.corrupt_history,
                        is_add_saturation=args.training_config.is_add_saturation,
                        # GT History
                        is_use_gt_history=USE_GT_HIST,
                        gt_history_latents=gt_history_latents_2,
                        gt_target_latents=gt_target_latents_2,
                        gt_x0_latents=gt_x0_latents_2,
                        # For VAE Re-Encode
                        vae=vae,
                        is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                        # For Multi Stage Backward Simulated
                        is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                        # For KV Cache
                        use_kv_cache=args.validation_config.use_kv_cache,
                        # For GAN
                        is_use_gan=USE_GAN,
                        is_separate_gan_grad=args.training_config.is_separate_gan_grad,
                        gan_base_critic_trainable_params=gan_base_critic_trainable_params,
                        gan_extra_critic_trainable_params=gan_extra_critic_trainable_params,
                        gan_vae_latents=gan_vae_latents,
                        gan_prompt_embeds=gan_prompt_embeds,
                        gan_d_weight=args.training_config.gan_d_weight,
                        aprox_r1=args.training_config.aprox_r1,
                        aprox_r2=args.training_config.aprox_r2,
                        r1_weight=args.training_config.r1_weight,
                        r2_weight=args.training_config.r2_weight,
                        r1_sigma=args.training_config.r1_sigma,
                        r2_sigma=args.training_config.r2_sigma,
                        # For Dynamic Timestep
                        dynamic_alpha=args.training_config.critic_dynamic_alpha,
                        dynamic_beta=args.training_config.critic_dynamic_beta,
                        dynamic_sample_type=args.training_config.critic_dynamic_sample_type,
                        global_step=global_step,
                        dynamic_step=args.training_config.critic_dynamic_step,
                    )
                    phase_tracer.record(
                        "critic_loss_end",
                        global_step=global_step,
                        loss=float(critic_loss.detach().mean().item()),
                    )
                    if not (
                        USE_GAN
                        and args.training_config.is_gan_low_vram_mode
                    ):
                        if os.environ.get("DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD", "0") == "1":
                            with _helios_stage_range(
                                global_step,
                                "runtime",
                                "allocator_cache_release",
                                target="critic_backward",
                            ):
                                torch.cuda.empty_cache()
                        with _helios_stage_range(global_step, "critic", "train_backward"):
                            critic_accelerator.backward(critic_loss)
                        if semantic_critic_sha_before is not None:
                            assert critic_gradient_observer is not None
                            semantic_critic_gradient_observation = (
                                critic_gradient_observer.end()
                            )
                            if semantic_critic_gradient_observation["hook_calls"] == 0:
                                raise AssertionError(
                                    "Critic backward delivered no trainable gradients"
                                )
                            phase_tracer.record(
                                "semantic_critic_gradient_observed",
                                global_step=global_step,
                                observation=semantic_critic_gradient_observation,
                            )
                        phase_tracer.record("critic_backward_end", global_step=global_step)
                    elif semantic_critic_sha_before is not None:
                        assert critic_gradient_observer is not None
                        semantic_critic_gradient_observation = (
                            critic_gradient_observer.end()
                        )
                        if semantic_critic_gradient_observation["hook_calls"] == 0:
                            raise AssertionError(
                                "GAN low-VRAM Critic backwards delivered no trainable gradients"
                            )
                        phase_tracer.record(
                            "semantic_critic_gradient_observed",
                            global_step=global_step,
                            observation=semantic_critic_gradient_observation,
                            backward_owner="official_gan_low_vram_critic_loss",
                        )
                        phase_tracer.record(
                            "critic_backward_end",
                            global_step=global_step,
                            backward_owner="official_gan_low_vram_critic_loss",
                        )

                    critic_grad_norm = None
                    if critic_accelerator.sync_gradients:
                        critic_params_to_clip = real_score_model.parameters()
                        critic_grad_norm = critic_accelerator.clip_grad_norm_(
                            critic_params_to_clip, args.training_config.max_grad_norm_critic
                        )

                    critic_log_dict["critic_loss"] = critic_loss
                    if critic_grad_norm is not None:
                        critic_log_dict["critic_grad_norm"] = critic_grad_norm

                    extra = critic_log_dict
                    extras_list.append(extra)
                    critic_log_dict = merge_dict_list(extras_list)
                    with _helios_stage_range(global_step, "critic", "optimizer_step"):
                        critic_optimizer.step()
                        critic_lr_scheduler.step()
                        critic_optimizer.zero_grad(set_to_none=True)
                    critic_parameter_version += 1
                    if semantic_critic_sha_before is not None:
                        semantic_critic_sha_after = trainable_sha256(real_score_model)
                        if semantic_critic_sha_after == semantic_critic_sha_before:
                            raise AssertionError("Critic trainable parameters did not change")
                        semantic_state["critic_update_verified"] = True
                        phase_tracer.record(
                            "semantic_critic_post_step",
                            global_step=global_step,
                            before_sha256=semantic_critic_sha_before,
                            after_sha256=semantic_critic_sha_after,
                            gradient_observation=semantic_critic_gradient_observation,
                            critic_version=critic_parameter_version,
                        )
                    if semantic_critic_only_generator_sha is not None:
                        semantic_critic_only_generator_after = trainable_sha256(transformer)
                        generator_gradients_after_critic = parameter_gradient_diagnostics(
                            transformer
                        )
                        if (
                            semantic_critic_only_generator_after
                            != semantic_critic_only_generator_sha
                        ):
                            raise AssertionError(
                                "Critic-only step changed Generator trainable parameters"
                            )
                        if generator_gradients_after_critic["trainable_grad_present"] != 0:
                            raise AssertionError(
                                "Critic-only step left Generator gradients allocated"
                            )
                        semantic_state["critic_only_verified"] = True
                        phase_tracer.record(
                            "semantic_critic_only_exit",
                            global_step=global_step,
                            generator_sha256=semantic_critic_only_generator_after,
                            generator_gradients=generator_gradients_after_critic,
                            generator_version=generator_parameter_version,
                        )
                    critic_step_event = {
                        "global_step": global_step,
                        "loss": float(critic_log_dict["critic_loss"].mean().item()),
                        "critic_version": critic_parameter_version,
                    }
                    if not phase_tracer.timing_only:
                        critic_step_event["checksum"] = trainable_checksum(real_score_model)
                    phase_tracer.record("critic_optimizer_step", **critic_step_event)

                    if args.training_config.use_ema and ema_transformer is not None:
                        if (
                            global_step < args.training_config.ema_start_step
                            or not args.training_config.is_train_dmd
                            or TRAIN_GENERATOR
                        ):
                            if args.training_config.dmd_is_low_vram_mode:
                                if vram_manager.wave_swap_enabled:
                                    vram_manager.swap_models(
                                        incoming_model=transformer,
                                        outgoing_model=real_score_model,
                                        device=accelerator.device,
                                    )
                                else:
                                    vram_manager.move_to_cpu(real_score_model)
                                    vram_manager.move_to_gpu(transformer, accelerator.device)

                    logs.update(
                        {
                            # "critic_lr": critic_lr_scheduler.get_last_lr()[0],
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": safe_item(critic_log_dict["critic_grad_norm"]),
                        }
                    )
                    if USE_GAN:
                        logs.update(
                            {
                                "denoising_loss": critic_log_dict["denoising_loss"],
                                "gan_D_loss": critic_log_dict["gan_D_loss"],
                                "r1_loss": critic_log_dict["r1_loss"],
                                "r2_loss": critic_log_dict["r2_loss"],
                            }
                        )

                    critic_loss = None
                    critic_grad_norm = None
                    del critic_loss
                    del critic_grad_norm
                    free_memory()

                batch = None
                model_input = None
                prompt_embeds = None
                indices_hidden_states = None
                indices_latents_history_short = None
                indices_latents_history_mid = None
                indices_latents_history_long = None
                latents_history_short = None
                latents_history_mid = None
                latents_history_long = None
                gan_vae_latents = None
                gan_prompt_embeds = None
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None
                ode_latents = None
                ode_prompt_embeds = None
                text_prompt_raws = None
                text_prompt_embeds = None
                trace_source = None
                del batch
                del model_input
                del prompt_embeds
                del indices_hidden_states
                del indices_latents_history_short
                del indices_latents_history_mid
                del indices_latents_history_long
                del latents_history_short
                del latents_history_mid
                del latents_history_long
                del gan_vae_latents
                del gan_prompt_embeds
                del gt_history_latents
                del gt_target_latents
                del gt_x0_latents
                del gt_history_latents_2
                del gt_target_latents_2
                del gt_x0_latents_2
                del ode_latents
                del ode_prompt_embeds
                del text_prompt_raws
                del text_prompt_embeds
                del trace_source
                free_memory()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.training_config.use_ema and ema_transformer is not None:
                    ema_before_start = global_step < args.training_config.ema_start_step
                    ema_generator_update = bool(TRAIN_GENERATOR)
                    ema_should_step = (
                        ema_before_start
                        or not args.training_config.is_train_dmd
                        or ema_generator_update
                    )
                    if ema_should_step:
                        ema_optimization_step_before = ema_transformer.optimization_step
                        ema_transformer.step(transformer.parameters())
                        ema_comparison = None
                        if (
                            baseline_options.semantic_assertions
                            and isinstance(ema_transformer, SparseShardEMA)
                        ):
                            ema_comparison = ema_transformer.compare_to_parameters(
                                transformer.parameters()
                            )
                            if (
                                ema_transformer.cur_decay_value == 0.0
                                and not ema_comparison["matches_exact"]
                            ):
                                raise AssertionError(
                                    "zero-decay EMA must exactly match the Generator source"
                                )
                            if (
                                ema_transformer.optimization_step
                                != ema_optimization_step_before + 1
                            ):
                                raise AssertionError("EMA optimization_step did not advance once")
                        if phase_tracer.timing_only:
                            ema_checksum = {
                                "optimization_step": ema_transformer.optimization_step,
                                "decay": ema_transformer.cur_decay_value,
                            }
                        else:
                            ema_checksum = (
                                ema_transformer.checksum()
                                if isinstance(ema_transformer, SparseShardEMA)
                                else {"optimization_step": ema_transformer.optimization_step}
                            )
                        phase_tracer.record(
                            "ema_step",
                            global_step=global_step,
                            ema=ema_checksum,
                            optimization_step_before=ema_optimization_step_before,
                            generator_version=generator_parameter_version,
                            critic_version=critic_parameter_version,
                            predicate={
                                "before_ema_start": ema_before_start,
                                "not_train_dmd": not args.training_config.is_train_dmd,
                                "generator_update": ema_generator_update,
                            },
                            source_comparison=ema_comparison,
                        )
                    else:
                        phase_tracer.record(
                            "ema_skipped",
                            global_step=global_step,
                            optimization_step=ema_transformer.optimization_step,
                            generator_version=generator_parameter_version,
                            critic_version=critic_parameter_version,
                            predicate={
                                "before_ema_start": ema_before_start,
                                "not_train_dmd": not args.training_config.is_train_dmd,
                                "generator_update": ema_generator_update,
                            },
                        )

                if phase_tracer.timing_only and torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_wall_end = time.perf_counter()
                phase_tracer.record(
                    "training_step_end",
                    global_step=global_step,
                    train_generator=bool(locals().get("TRAIN_GENERATOR", True)),
                    generator_version=generator_parameter_version,
                    critic_version=critic_parameter_version,
                    elapsed_seconds=step_wall_end - step_wall_start,
                    data_wait_seconds=data_wait_seconds,
                )
                if global_step + 1 == nsys_capture_stop_step:
                    accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process:
                        if not nsys_range_open:
                            raise AssertionError("Nsight NVTX range was not opened")
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.cudart().cudaProfilerStop()
                        nsys_range_open = False
                previous_step_end = step_wall_end
                progress_bar.update(1)
                global_step += 1

                if args.training_config.is_train_dmd:
                    if accelerator.is_main_process and VISUALIZE:
                        phase_name = "dmd_visualize"
                        if args.training_config.dmd_is_low_vram_mode:
                            vram_manager.move_to_cpu(transformer)
                            vram_manager.move_to_cpu(real_score_model)

                        if vae is None:
                            vae = AutoencoderKLWan.from_pretrained(
                                args.model_config.pretrained_model_name_or_path,
                                subfolder="vae",
                                revision=args.model_config.revision,
                                variant=args.model_config.variant,
                                torch_dtype=torch.float32,
                                device_map=accelerator.device,
                            )
                            if args.model_config.enable_slicing:
                                vae.enable_slicing()
                            if args.model_config.enable_tiling:
                                vae.enable_tiling()

                        if args.training_config.dmd_is_low_vram_mode and args.training_config.is_dmd_vae_decode:
                            vram_manager.move_to_gpu(vae, accelerator.device)
                        else:
                            vae.to(accelerator.device, non_blocking=True)
                        latents_mean = (
                            torch.tensor(vae.config.latents_mean)
                            .view(1, vae.config.z_dim, 1, 1, 1)
                            .to(vae.device, vae.dtype)
                        )
                        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                            vae.device, vae.dtype
                        )

                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                video_logs = []

                                def decode_latent(latent):
                                    with torch.no_grad():
                                        latent = latent[0:1]  # [1, C, T, H, W]
                                        latent = latent / latents_std + latents_mean
                                        return vae.decode(latent)[0]  # [1, C, T, H, W]

                                def prepare_for_saving(tensor, fps=30, caption=None):
                                    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()
                                    tensor = tensor.permute(0, 2, 1, 3, 4)
                                    video_array = (tensor * 255).cpu().numpy().astype(np.uint8)
                                    return wandb.Video(video_array, fps=fps, format="mp4", caption=caption)

                                log_configs = [
                                    (
                                        critic_log_dict,
                                        ["critictrain_latent", "critictrain_noisy_latent", "critictrain_pred_image"],
                                    ),
                                ]
                                generator_keys = [
                                    "dmdtrain_clean_latent",
                                    "dmdtrain_pred_real_image",
                                    "dmdtrain_pred_fake_image",
                                ]
                                if args.training_config.is_decouple_dmd:
                                    generator_keys.extend(["dmdtrain_ca_noisy_latent", "dmdtrain_dm_noisy_latent"])
                                else:
                                    generator_keys.append("dmdtrain_noisy_latent")
                                log_configs.append((generator_log_dict, generator_keys))
                                for log_dict, keys in log_configs:
                                    for key in keys:
                                        if key in log_dict:
                                            with torch.no_grad():
                                                decoded = decode_latent(log_dict[key])
                                            video_logs.append(prepare_for_saving(decoded, fps=30, caption=key))
                                            del decoded

                                tracker.log({phase_name: video_logs}, step=global_step)

                        if (
                            args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
                        ) or args.data_config.use_stage3_dataset:
                            if (
                                not args.training_config.is_dmd_vae_decode
                                and not args.training_config.is_use_reward_model
                                and not args.training_config.is_smoothness_loss
                            ):
                                vae = None
                            free_memory()

                        if vae is not None:
                            vae.to("cpu", non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    if "generator_log_dict" in locals():
                        generator_log_dict.clear()
                        del generator_log_dict
                    if "critic_log_dict" in locals():
                        critic_log_dict.clear()
                        del critic_log_dict
                    if "video_logs" in locals():
                        del video_logs
                    if "log_configs" in locals():
                        del log_configs
                    free_memory()

                if global_step % args.training_config.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    states = {
                        "dataloader": train_dataloader,
                    }
                    dcp_dir = os.path.join(save_path, "distributed_checkpoint")
                    dcp.save(states, checkpoint_id=dcp_dir)
                    states = None
                    del states
                    free_memory()

                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.training_config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.training_config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.training_config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        if args.training_config.save_checkpoints_custom:
                            if accelerator.is_main_process:
                                save_model_checkpoint(
                                    transformer=transformer,
                                    args=args,
                                    save_path=save_path,
                                    weight_dtype=weight_dtype,
                                    unwrap_model_fn=unwrap_model,
                                    get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                    collate_lora_metadata_fn=_collate_lora_metadata,
                                    save_extra_components_fn=save_extra_components,
                                    pipeline_class=HeliosPipeline,
                                    norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                )
                                if args.training_config.is_train_dmd:
                                    save_model_checkpoint(
                                        transformer=real_score_model,
                                        args=args,
                                        save_path=os.path.join(save_path, "critic"),
                                        weight_dtype=weight_dtype,
                                        unwrap_model_fn=unwrap_model,
                                        get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                        collate_lora_metadata_fn=_collate_lora_metadata,
                                        save_extra_components_fn=save_extra_components,
                                        pipeline_class=HeliosPipeline,
                                        norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                    )
                        else:
                            accelerator.save_state(save_path)
                            if args.training_config.is_train_dmd:
                                critic_accelerator.save_state(os.path.join(save_path, "critic"))
                        accelerator.print(f"Saved state to {save_path}")

                    if args.training_config.use_ema and ema_transformer is not None:
                        if isinstance(ema_transformer, SparseShardEMA):
                            # ZeRO-1 replicas hold identical sparse CPU EMA state.  A
                            # single global writer avoids sixteen ranks racing on the
                            # same atomic shard names; barriers publish a complete set
                            # before any rank continues or resumes.
                            accelerator.wait_for_everyone()
                            phase_tracer.record(
                                "sparse_ema_checkpoint_save_start",
                                global_step=global_step,
                                save_path=os.path.join(save_path, "model_ema"),
                                writer=accelerator.is_main_process,
                            )
                            if accelerator.is_main_process:
                                ema_transformer.save_pretrained(
                                    args,
                                    os.path.join(save_path, "model_ema"),
                                    args.model_config.transformer_model_name_or_path,
                                    lora_config=transformer_lora_config,
                                    transformer_additional_kwargs=transformer_additional_kwargs,
                                )
                            accelerator.wait_for_everyone()
                            phase_tracer.record(
                                "sparse_ema_checkpoint_save_end",
                                global_step=global_step,
                                save_path=os.path.join(save_path, "model_ema"),
                                writer=accelerator.is_main_process,
                            )
                        else:
                            ema_transformer.save_pretrained(
                                args,
                                os.path.join(save_path, "model_ema"),
                                args.model_config.transformer_model_name_or_path,
                                lora_config=transformer_lora_config,
                                transformer_additional_kwargs=transformer_additional_kwargs,
                            )

                if not baseline_options.skip_validation and (
                    (
                        args.validation_config.validation_prompts is not None
                        and global_step % args.validation_config.validation_steps == 0
                    )
                    or (
                        args.validation_config.first_step_valid
                        and global_step == (initial_global_step + 1)
                    )
                ):
                    if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                        vram_manager.move_to_cpu(real_score_model)

                    if args.training_config.is_train_dmd:
                        optimizer.zero_grad(set_to_none=True)
                        critic_optimizer.zero_grad(set_to_none=True)

                        if "generator_log_dict" in locals():
                            generator_log_dict.clear()
                            del generator_log_dict
                        if "critic_log_dict" in locals():
                            critic_log_dict.clear()
                            del critic_log_dict

                        free_memory()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.print("Starting EMA store and copy_to...")
                        ema_transformer.store(transformer.parameters())
                        ema_state_dict = gather_zero3ema(accelerator, ema_transformer)
                        transformer.load_state_dict({"module." + k: v for k, v in ema_state_dict.items()})
                        accelerator.print("EMA store and copy_to completed")
                        ema_state_dict = None
                        del ema_state_dict

                    free_memory()
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            if vae is None:
                                vae = AutoencoderKLWan.from_pretrained(
                                    args.model_config.pretrained_model_name_or_path,
                                    subfolder="vae",
                                    revision=args.model_config.revision,
                                    variant=args.model_config.variant,
                                    torch_dtype=torch.float32,
                                    device_map=accelerator.device,
                                )
                                if args.model_config.enable_slicing:
                                    vae.enable_slicing()
                                if args.model_config.enable_tiling:
                                    vae.enable_tiling()

                            if text_encoder is None:
                                text_encoder = UMT5EncoderModel.from_pretrained(
                                    args.model_config.pretrained_model_name_or_path,
                                    subfolder="text_encoder",
                                    revision=args.model_config.revision,
                                    variant=args.model_config.variant,
                                    dtype=weight_dtype,
                                    device_map=accelerator.device,
                                )

                            if args.data_config.use_stage1_dataset or args.training_config.offload:
                                vae.to(accelerator.device, non_blocking=True)
                                text_encoder.to(accelerator.device, non_blocking=True)

                            pipe = HeliosPipeline.from_pretrained(
                                args.model_config.pretrained_model_name_or_path,
                                vae=vae,
                                transformer=unwrap_model(transformer),
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                scheduler=noise_scheduler,
                                revision=args.model_config.revision,
                                variant=args.model_config.variant,
                                torch_dtype=weight_dtype,
                            )

                            all_videos = []
                            all_prompts = []
                            for validation_prompt in args.validation_config.validation_prompts:
                                pipeline_args = {
                                    "prompt": args.data_config.id_token + validation_prompt,
                                    "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                                    "guidance_scale": args.validation_config.validation_guidance_scale,
                                    "num_frames": args.validation_config.validation_max_num_frames,
                                    "height": args.validation_config.validation_height,
                                    "width": args.validation_config.validation_width,
                                    "num_inference_steps": args.validation_config.num_inference_steps,
                                    # ---- Dynamic Shifting ----
                                    "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
                                    "time_shift_type": args.validation_config.time_shift_type,
                                    # For Stage 1
                                    "history_sizes": args.training_config.history_sizes,
                                    "latent_window_size": args.validation_config.validation_latent_window_size[0],
                                    "is_keep_x0": True,
                                    "use_kv_cache": args.validation_config.use_kv_cache,
                                    # For Stage 2
                                    "is_enable_stage2": args.training_config.is_enable_stage2,
                                    "stage2_num_stages": args.training_config.stage2_num_stages,
                                    "stage2_num_inference_steps_list": args.validation_config.stage2_simulated_inference_steps,
                                    "vae_decode_type": args.training_config.vae_decode_type,
                                    # For Stage 3
                                    "use_dmd": args.training_config.is_train_dmd,
                                    "is_amplify_first_chunk": args.training_config.is_amplify_first_chunk,
                                }

                                videos, prompt = log_validation(
                                    pipe=pipe,
                                    args=args,
                                    accelerator=accelerator,
                                    pipeline_args=pipeline_args,
                                )

                                all_videos.extend(videos)
                                all_prompts.extend([prompt] * len(videos))

                            for tracker in accelerator.trackers:
                                phase_name = "validation"
                                if tracker.name == "wandb":
                                    video_logs = []

                                    for i, (video, prompt) in enumerate(zip(all_videos, all_prompts)):
                                        filename = os.path.join(
                                            args.output_dir,
                                            f"global_step{global_step}_{phase_name}_video_{i}_{prompt[:25].replace(' ', '_')}.mp4",
                                        )
                                        export_to_video(video, filename, fps=30)
                                        video_logs.append(
                                            wandb.Video(filename, caption=f"{i}: {prompt}", format="mp4")
                                        )

                                    tracker.log({phase_name: video_logs}, step=global_step)

                            videos = None
                            prompt = None
                            all_videos = None
                            all_prompts = None
                            video_logs = None
                            del videos
                            del prompt
                            del all_videos
                            del all_prompts
                            del video_logs
                            free_memory()

                            if (
                                args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
                            ) or args.data_config.use_stage3_dataset:
                                if (
                                    not args.training_config.is_dmd_vae_decode
                                    and not args.training_config.is_use_reward_model
                                    and not args.training_config.is_smoothness_loss
                                ):
                                    vae = None
                                text_encoder = None
                                free_memory()

                            del pipe
                            free_memory()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.wait_for_everyone()
                        ema_transformer.restore(transformer.parameters())

            if args.data_config.use_stage1_dataset:
                if vae is not None:
                    vae.to("cpu", non_blocking=True)
                if text_encoder is not None:
                    text_encoder.to("cpu", non_blocking=True)
                free_memory()

            if args.training_config.offload:
                if vae is not None:
                    vae.to(accelerator.device, non_blocking=True)
                if text_encoder is not None:
                    text_encoder.to(accelerator.device, non_blocking=True)

            if prof is not None:
                prof.step()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

            logs = None
            del logs
            free_memory()

    if prof is not None:
        prof.stop()
        print(f"Profiler stopped. Check results in: {args.training_config.profile_out_dir}")

    if generator_gradient_observer is not None:
        generator_gradient_observer.close()
    if critic_gradient_observer is not None:
        critic_gradient_observer.close()

    if baseline_options.semantic_assertions:
        missing_semantic_checks = [
            name for name, verified in semantic_state.items() if not verified
        ]
        phase_tracer.record(
            "semantic_contract_complete",
            global_step=global_step,
            checks=dict(semantic_state),
            missing=missing_semantic_checks,
            generator_version=generator_parameter_version,
            critic_version=critic_parameter_version,
        )
        if missing_semantic_checks:
            raise AssertionError(
                f"semantic contract did not exercise: {missing_semantic_checks}"
            )

    if baseline_options.skip_final_save:
        accelerator.wait_for_everyone()
        if vram_manager is not None:
            vram_manager.assert_engine_state_controllers_balanced()
        for registration in activation_offload_registrations:
            registration.offloader.assert_scope_closed()
            phase_tracer.record(
                "activation_offload_final",
                logical_component=registration.offloader.logical_component,
                stats=dict(registration.offloader.stats),
            )
        for logical_component, streamer in block_streamers.items():
            streamer.assert_idle()
            phase_tracer.record(
                "frozen_block_streamer_final",
                logical_component=logical_component,
                stats=dict(streamer.stats),
            )
            streamer.remove()
        for logical_component, backing in immutable_model_backings.items():
            backing.assert_integrity()
            phase_tracer.record(
                "immutable_model_backing_final",
                logical_component=logical_component,
                stats=dict(backing.stats),
            )
        phase_tracer.record("training_complete_skip_final_save", global_step=global_step)
        return

    # Save the lora layers
    if args.training_config.is_train_dmd:
        if vram_manager is not None:
            vram_manager.move_to_cpu(real_score_model, non_blocking=True)
        else:
            real_score_model.to("cpu", non_blocking=True)
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
    if args.training_config.use_ema and ema_transformer is not None:
        ema_transformer.save_pretrained(
            args,
            os.path.join(save_path, "model_ema"),
            args.model_config.transformer_model_name_or_path,
            lora_config=transformer_lora_config,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )
    if accelerator.is_main_process:
        modules_to_save = {}
        model_to_save = unwrap_model(transformer)
        original_dtype = next(model_to_save.parameters()).dtype
        if args.model_config.bnb_quantization_config_path is None:
            if args.training_config.upcast_before_saving:
                model_to_save.to(torch.float32)
            else:
                model_to_save.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(model_to_save)
        if args.model_config.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in model_to_save.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {
                **transformer_lora_layers,
                **transformer_norm_layers,
            }
        modules_to_save["transformer"] = model_to_save

        HeliosPipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        save_extra_components(args, model=model_to_save, output_dir=save_path)
        model_to_save.to(original_dtype)

        if args.training_config.use_ema and ema_transformer is not None:
            ema_state_dict = gather_zero3ema(accelerator, ema_transformer)
            transformer.load_state_dict(ema_state_dict)

        # Run a final round of validation.
        # Setting `vae`, `unet`, and `controlnet` to None to load automatically from `args.output_dir`.
        if args.validation_config.validation_prompts is not None:
            with torch.no_grad():
                if vae is None:
                    vae = AutoencoderKLWan.from_pretrained(
                        args.model_config.pretrained_model_name_or_path,
                        subfolder="vae",
                        revision=args.model_config.revision,
                        variant=args.model_config.variant,
                        torch_dtype=torch.float32,
                        device_map=accelerator.device,
                    )
                    if args.model_config.enable_slicing:
                        vae.enable_slicing()
                    if args.model_config.enable_tiling:
                        vae.enable_tiling()

                if text_encoder is None:
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        args.model_config.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        revision=args.model_config.revision,
                        variant=args.model_config.variant,
                        dtype=weight_dtype,
                        device_map=accelerator.device,
                    )

                if args.data_config.use_stage1_dataset:
                    vae.to(accelerator.device, non_blocking=True)
                    text_encoder.to(accelerator.device, non_blocking=True)

                pipe = HeliosPipeline.from_pretrained(
                    args.model_config.pretrained_model_name_or_path,
                    vae=vae,
                    transformer=unwrap_model(transformer),
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler=noise_scheduler,
                    revision=args.model_config.revision,
                    variant=args.model_config.variant,
                    torch_dtype=weight_dtype,
                )

                all_videos = []
                all_prompts = []
                for validation_prompt in args.validation_config.validation_prompts:
                    pipeline_args = {
                        "prompt": args.data_config.id_token + validation_prompt,
                        "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        "guidance_scale": args.validation_config.validation_guidance_scale,
                        "num_frames": args.validation_config.validation_max_num_frames,
                        "height": args.validation_config.validation_height,
                        "width": args.validation_config.validation_width,
                        "num_inference_steps": args.validation_config.num_inference_steps,
                        # ---- Dynamic Shifting ----
                        "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
                        "time_shift_type": args.validation_config.time_shift_type,
                        # For Stage 1
                        "history_sizes": args.training_config.history_sizes,
                        "latent_window_size": args.validation_config.validation_latent_window_size[0],
                        "is_keep_x0": True,
                        "use_kv_cache": args.validation_config.use_kv_cache,
                        # For Stage 2
                        "is_enable_stage2": args.training_config.is_enable_stage2,
                        "stage2_num_stages": args.training_config.stage2_num_stages,
                        "stage2_num_inference_steps_list": args.validation_config.stage2_simulated_inference_steps,
                        "vae_decode_type": args.training_config.vae_decode_type,
                        # For Stage 3
                        "use_dmd": args.training_config.is_train_dmd,
                        "is_amplify_first_chunk": args.training_config.is_amplify_first_chunk,
                    }
                    videos, prompt = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                    )

                    all_videos.extend(videos)
                    all_prompts.extend([prompt] * len(videos))

                for tracker in accelerator.trackers:
                    phase_name = "final_step_validation"
                    if tracker.name == "wandb":
                        video_logs = []

                        for i, (video, prompt) in enumerate(zip(all_videos, all_prompts)):
                            filename = os.path.join(
                                args.output_dir,
                                f"global_step{global_step}_{phase_name}_video_{i}_{prompt[:25].replace(' ', '_')}.mp4",
                            )
                            export_to_video(video, filename, fps=30)
                            video_logs.append(wandb.Video(filename, caption=f"{i}: {prompt}", format="mp4"))

                        tracker.log({phase_name: video_logs}, step=global_step)

    accelerator.end_training()


@torch.no_grad()
def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
):
    logger.info(
        f"Running validation... \n Generating {args.validation_config.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.validation_config.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    del pipe
    free_memory()

    return videos, pipeline_args["prompt"]


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    global_rank = int(os.environ.get("RANK", -1))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    assert (
        len(conf.validation_config.validation_latent_window_size) == 1
        and len(conf.validation_config.validation_stream_chunk_size) == 1
    ), "Only a single value is currently supported for validation_latent_window_size and validation_stream_chunk_size"

    assert not (conf.data_config.use_stage1_dataset and conf.training_config.offload), (
        "use_stage1_dataset and offload cannot both be True"
    )

    assert not (conf.data_config.use_stage1_dataset and conf.training_config.offload), (
        "use_stage1_dataset and offload cannot both be True"
    )

    if conf.model_config.lora_layers is not None:
        assert len(conf.model_config.lora_target_modules) == 0, (
            f"Error: lora_target_modules length is {len(conf.model_config.lora_target_modules)}, expected 0 when lora_layers is not None."
        )

    if conf.training_config.efficient_sample:
        assert conf.training_config.pyramid_sample_mode == "full", (
            f"efficient_sample requires pyramid_sample_mode='full', got '{conf.training_config.pyramid_sample_mode}'"
        )

    if conf.data_config.dataset_sampling_ratios:
        assert conf.data_config.use_stage1_dataset, (
            "dataset_sampling_ratios is only supported when use_stage1_dataset=True"
        )
        if len(conf.data_config.instance_data_root) != len(conf.data_config.dataset_sampling_ratios):
            raise ValueError(
                f"Length mismatch: instance_data_root ({len(conf.data_config.instance_data_root)}) "
                f"vs dataset_sampling_ratios ({len(conf.data_config.dataset_sampling_ratios)})"
            )

        basenames = []
        for temp_key, temp_value in zip(conf.data_config.instance_data_root, conf.data_config.dataset_sampling_ratios):
            basename = temp_key.rstrip("/")
            if basename in basenames:
                raise ValueError(f"Duplicate dataset name: {basename}")
            basenames.append(basename)

    if conf.data_config.single_res:
        assert conf.data_config.force_rebuild, "force_rebuild must be True when single_res is enabled"

    # ---------------------- For Wan ----------------------
    if (
        conf.training_config.is_train_full_multi_term_memory_patchg
        or conf.training_config.is_train_lora_multi_term_memory_patchg
        or conf.training_config.zero_history_timestep
    ):
        assert conf.training_config.has_multi_term_memory_patch, "Missing clean patch embedding configuration."
        assert conf.training_config.is_enable_stage1, (
            "is_enable_stage1 must be enabled when using clean patch embedding."
        )

    if conf.training_config.restrict_lora:
        assert conf.training_config.restrict_self_attn, (
            "Self-attention restriction must be enabled when restricting LoRA."
        )

    if conf.training_config.is_train_restrict_lora:
        assert conf.training_config.restrict_lora, (
            "LoRA restriction must be enabled when training with LoRA restriction."
        )

    assert not (
        conf.training_config.is_train_full_multi_term_memory_patchg
        and conf.training_config.is_train_lora_multi_term_memory_patchg
    ), (
        "Both 'is_train_full_multi_term_memory_patchg' and 'is_train_lora_multi_term_memory_patchg' cannot be True at the same time."
    )
    assert not (
        conf.training_config.is_train_full_patch_embedding and conf.training_config.is_train_lora_patch_embedding
    ), "Both 'is_train_full_patch_embedding' and 'is_train_lora_patch_embedding' cannot be True at the same time."

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_history), (
        "Both 'use_error_recycling' and 'corrupt_history' cannot be True at the same time."
    )

    if conf.training_config.is_enable_stage2:
        if not conf.training_config.is_train_dmd and not conf.training_config.is_use_ode_regression:
            assert conf.training_config.use_dynamic_shifting is False, (
                "Dynamic shifting cannot be used with pyramid sampling unless is_train_dmd or is_use_ode_regression is True."
            )

    if conf.training_config.is_use_ode_regression:
        assert conf.training_config.use_dynamic_shifting, (
            "use_dynamic_shifting must be True when is_use_ode_regression is enabled."
        )

    if conf.validation_config.use_kv_cache:
        assert conf.training_config.restrict_self_attn, "When use_kv_cache=True, restrict_self_attn must also be True!"

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_history), (
        "Both 'use_error_recycling' and 'corrupt_history' cannot be True at the same time."
    )

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_model_input), (
        "Both 'use_error_recycling' and 'corrupt_model_input' cannot be True at the same time."
    )

    if conf.training_config.is_multi_pyramid_stage_backward_simulated:
        assert conf.training_config.is_enable_stage2, (
            "Multi_Pyramid_Stage_Backward_Simulated requires is_enable_stage2 to be enabled"
        )

    if conf.training_config.use_ema_validation:
        assert conf.training_config.use_ema, "EMA validation requires use_ema to be enabled"

    if conf.training_config.is_use_reward_model:
        assert conf.training_config.reward_weight_vq > 0 or conf.training_config.reward_weight_mq > 0, (
            "At least one of reward_weight_vq or reward_weight_mq must be greater than 0 when using reward model"
        )

    if conf.training_config.is_use_gan:
        assert conf.training_config.is_train_dmd, "GAN training requires is_train_dmd to be enabled"
        assert conf.training_config.is_use_gan_hooks or conf.training_config.is_use_gan_final, (
            "GAN training requires either is_use_gan_hooks or is_use_gan_final to be enabled"
        )

    if conf.training_config.stage_cold_start_step is not None:
        assert conf.training_config.stage_cold_start_step <= conf.training_config.cold_start_step, (
            f"stage_cold_start_step ({conf.training_config.stage_cold_start_step}) must be less than or equal to cold_start_step ({conf.training_config.cold_start_step})"
        )

    if conf.training_config.is_decouple_dmd:
        assert conf.training_config.decouple_ca_start_step >= conf.training_config.generator_dynamic_step, (
            "decouple_ca_start_step must be greater than or equal to generator_dynamic_step"
        )

        assert conf.training_config.decouple_ca_end_step >= conf.training_config.generator_dynamic_step, (
            "decouple_ca_end_step must be greater than or equal to generator_dynamic_step"
        )

    try:
        main(conf)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            report_path = os.environ.get("DISTILLSCHEDULE_OOM_REPORT")
            if report_path:
                write_oom_report(report_path, error)
        raise
