"""Compatibility adapter that routes typed plans into the validated Helios entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from distillschedule.adapters.base import (
    CompiledAdapterPlan,
    DMDModelAdapter,
    LegacyInvocation,
)
from distillschedule.core.capabilities import AdapterCapabilities
from distillschedule.core.contracts import ContractViolation
from distillschedule.core.enums import (
    HostStatePolicy,
    InstrumentationMode,
    RolloutEngineKind,
    TrainingBackendKind,
)
from distillschedule.core.plan import ExecutionPlan
from distillschedule.workloads.dmd import DMDProtocolCompiler


_TRUE = "1"


def _resolve(path: str | None, *, base_dir: Path) -> Path | None:
    """Resolve every relative path against the plan that declares it."""

    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _multi_node_rank(plan: ExecutionPlan) -> int:
    """Return this host's node rank and fail closed for multi-node plans."""

    if plan.hardware.nodes == 1:
        return 0
    raw = os.environ.get("DISTILLSCHEDULE_NODE_RANK")
    if raw is None:
        raise ContractViolation(
            "multi-node Helios compilation requires DISTILLSCHEDULE_NODE_RANK",
            path="hardware.nodes",
        )
    try:
        node_rank = int(raw)
    except ValueError as exc:
        raise ContractViolation(
            "DISTILLSCHEDULE_NODE_RANK must be an integer",
            path="hardware.nodes",
        ) from exc
    if not 0 <= node_rank < plan.hardware.nodes:
        raise ContractViolation(
            "DISTILLSCHEDULE_NODE_RANK is outside the plan node range",
            path="hardware.nodes",
        )
    return node_rank


class HeliosDMDAdapter(DMDModelAdapter):
    """Strangler adapter: new plans, existing source-faithful Helios algorithm."""

    name = "helios"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            training_backends=(
                TrainingBackendKind.NATIVE,
                TrainingBackendKind.DEEPSPEED,
            ),
            rollout_engines=(
                RolloutEngineKind.NATIVE,
                RolloutEngineKind.VLLM,
                RolloutEngineKind.SGLANG,
                RolloutEngineKind.REMOTE,
            ),
            supports_dynamic_shape=False,
            supports_exact_version_sync=True,
            supports_external_rollout=True,
            features={
                "shared_score_modes": True,
                "nested_video_rollout": True,
                "operation_trace": True,
                "legacy_entrypoint": True,
            },
        )

    def validate(self, plan: ExecutionPlan) -> None:
        if plan.workload.generator.family != "helios":
            raise ContractViolation(
                "Generator family must be 'helios'",
                path="workload.generator.family",
            )
        if plan.workload.score.family != "helios":
            raise ContractViolation(
                "Score family must be 'helios'",
                path="workload.score.family",
            )
        shape = plan.workload.shape
        if shape.pyramid and shape.pyramid[-1] != shape.latent.spatial:
            raise ContractViolation(
                "Helios final pyramid stage must match latent spatial size",
                path="workload.shape.pyramid",
            )
        for component, backend in plan.training_backends.items():
            if not self.capabilities.supports_training(backend.kind):
                raise ContractViolation(
                    f"Helios adapter does not support {backend.kind.value!r} "
                    f"for {component}",
                    path=f"training_backends.{component}.kind",
                )
        if plan.rollout.engine in {
            RolloutEngineKind.VLLM,
            RolloutEngineKind.SGLANG,
        }:
            compatible = bool(
                plan.workload.metadata.get("external_rollout_compatible", False)
            )
            if not compatible:
                raise ContractViolation(
                    "current Helios diffusion rollout is not directly compatible "
                    "with vLLM/SGLang; use native/remote or provide an adapter "
                    "declaring external_rollout_compatible=true",
                    path="rollout.engine",
                )
        schedule = plan.workload.schedule
        if schedule.critic_update_every != 1:
            raise ContractViolation(
                "the current Helios entrypoint updates Critic every logical step",
                path="workload.schedule.critic_update_every",
            )
        if schedule.ema_update_every != 1:
            raise ContractViolation(
                "the current Helios entrypoint evaluates EMA every logical step",
                path="workload.schedule.ema_update_every",
            )
        if not schedule.ema_after_critic:
            raise ContractViolation(
                "the current Helios entrypoint orders EMA after Critic",
                path="workload.schedule.ema_after_critic",
            )
        if plan.legacy is None:
            raise ContractViolation(
                "the first Helios refactor version requires a legacy entrypoint "
                "while operations are migrated incrementally",
                path="legacy",
            )
        allow_logical_offset = bool(
            plan.evidence.metadata.get("allow_logical_offset_without_checkpoint", False)
        )
        if (
            plan.evidence.start_step > 0
            and plan.evidence.checkpoint is None
            and not allow_logical_offset
        ):
            raise ContractViolation(
                "a nonzero start_step requires a checkpoint or an explicit "
                "allow_logical_offset_without_checkpoint contract",
                path="evidence.checkpoint",
            )

    def _validate_backend_config(
        self,
        *,
        component: str,
        plan: ExecutionPlan,
        source_dir: Path,
    ) -> None:
        backend = plan.training_backends[component]
        if backend.kind != TrainingBackendKind.DEEPSPEED or not backend.config_path:
            return
        config_path = _resolve(backend.config_path, base_dir=source_dir)
        if config_path is None or not config_path.is_file():
            raise ContractViolation(
                "DeepSpeed config does not exist",
                path=f"training_backends.{component}.config_path",
            )
        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            try:
                from omegaconf import OmegaConf
            except ImportError as exc:  # pragma: no cover
                raise ContractViolation(
                    "YAML DeepSpeed config validation requires OmegaConf",
                    path=f"training_backends.{component}.config_path",
                ) from exc
            payload = OmegaConf.to_container(
                OmegaConf.load(config_path), resolve=True
            )
        if not isinstance(payload, dict):
            raise ContractViolation(
                "DeepSpeed config must be a mapping",
                path=f"training_backends.{component}.config_path",
            )
        zero = payload.get("zero_optimization")
        if not isinstance(zero, dict):
            raise ContractViolation(
                "DeepSpeed config is missing zero_optimization",
                path=f"training_backends.{component}.config_path",
            )
        observed_stage = int(zero.get("stage", -1))
        if observed_stage != backend.zero_stage:
            raise ContractViolation(
                f"plan declares ZeRO-{backend.zero_stage}, config contains "
                f"ZeRO-{observed_stage}",
                path=f"training_backends.{component}.zero_stage",
            )
        observed_optimizer = zero.get("offload_optimizer", {})
        observed_optimizer_device = (
            str(observed_optimizer.get("device", "none")).lower()
            if isinstance(observed_optimizer, dict)
            else "none"
        )
        if observed_optimizer_device != backend.optimizer_offload:
            raise ContractViolation(
                f"plan declares optimizer_offload={backend.optimizer_offload!r}, "
                f"config contains {observed_optimizer_device!r}",
                path=f"training_backends.{component}.optimizer_offload",
            )

    def _environment(
        self,
        plan: ExecutionPlan,
        *,
        source_dir: Path,
    ) -> dict[str, str]:
        if plan.legacy is None:
            raise ContractViolation("missing validated legacy entrypoint", path="legacy")
        lifecycle = plan.lifecycle
        evidence = plan.evidence
        logical_start_step = (
            evidence.start_step if evidence.checkpoint is None else 0
        )
        environment: dict[str, str] = {
            "DISTILLSCHEDULE_OFFICIAL_CPU_OFFLOAD": _TRUE,
            "DISTILLSCHEDULE_SPLIT_MODULATION": (
                _TRUE if lifecycle.split_modulation else "0"
            ),
            "DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE": str(
                lifecycle.construction_wave_size
            ),
            "DISTILLSCHEDULE_SWAP_WAVE_SIZE": str(lifecycle.swap_wave_size),
            "DISTILLSCHEDULE_TRIM_HOST_AFTER_MODEL_MOVE": (
                _TRUE if lifecycle.trim_host_after_move else "0"
            ),
            "DISTILLSCHEDULE_M0_SEMANTIC_ASSERTIONS": (
                _TRUE if evidence.semantic_assertions else "0"
            ),
            "DISTILLSCHEDULE_LOGICAL_START_STEP": str(logical_start_step),
        }
        if evidence.instrumentation == InstrumentationMode.TIMING:
            environment["DISTILLSCHEDULE_TIMING_ONLY_TRACE"] = _TRUE

        output_dir = _resolve(evidence.output_dir, base_dir=source_dir)
        if output_dir is None:
            raise ContractViolation("output_dir did not resolve", path="evidence.output_dir")
        environment["DISTILLSCHEDULE_PHASE_TRACE_DIR"] = str(
            output_dir / "phase_trace"
        )
        operation_trace = _resolve(evidence.operation_trace, base_dir=source_dir)
        if operation_trace is not None:
            environment["DISTILLSCHEDULE_OPERATION_TRACE_DIR"] = str(
                operation_trace
            )

        activation = lifecycle.activation_offload
        environment["DISTILLSCHEDULE_FULL_ACTIVATION_OFFLOAD"] = (
            _TRUE if activation.enabled else "0"
        )
        if activation.enabled:
            environment["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_MIN_BYTES"] = str(
                activation.threshold_bytes
            )
            environment["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"] = ",".join(
                activation.components
            )

        streaming = lifecycle.frozen_block_streaming
        environment["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] = str(
            streaming.window if streaming.enabled else 0
        )
        if streaming.enabled:
            environment[
                "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"
            ] = ",".join(streaming.components)
            environment[
                "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"
            ] = str(streaming.block_count)
            if streaming.block_indices:
                environment[
                    "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"
                ] = ",".join(str(index) for index in streaming.block_indices)
            environment["DISTILLSCHEDULE_FROZEN_BLOCK_HOST_POLICY"] = (
                "HOST_SHARED_IMMUTABLE"
                if streaming.host_policy == HostStatePolicy.SHARED_IMMUTABLE
                else "HOST_PRIVATE"
            )
            mmap_cache = _resolve(streaming.mmap_cache, base_dir=source_dir)
            if mmap_cache is not None:
                environment["DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE"] = str(
                    mmap_cache
                )
            environment["DISTILLSCHEDULE_FROZEN_BLOCK_PIN_HOST_STATE"] = (
                _TRUE if streaming.pin_host_state else "0"
            )

        if lifecycle.immutable_model_backing_components:
            environment[
                "DISTILLSCHEDULE_IMMUTABLE_MODEL_BACKING_COMPONENTS"
            ] = ",".join(lifecycle.immutable_model_backing_components)
        environment.update(plan.legacy.extra_env)
        return environment

    def _render_config(
        self,
        plan: ExecutionPlan,
        *,
        template_path: Path,
        output_dir: Path,
        source_dir: Path,
    ) -> str:
        if not template_path.is_file():
            raise ContractViolation(
                "legacy Helios config template does not exist",
                path=str(template_path),
            )
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - full Helios env includes it
            raise ContractViolation(
                "Helios config rendering requires OmegaConf",
                path="legacy.config_path",
            ) from exc

        config = OmegaConf.load(template_path)
        workload = plan.workload
        generator_backend = plan.training_backends["generator"]
        score_backend = plan.training_backends["score"]

        config.output_dir = str(output_dir)
        config.data_config.single_height = workload.shape.resolution.height
        config.data_config.single_width = workload.shape.resolution.width
        config.validation_config.validation_height = workload.shape.resolution.height
        config.validation_config.validation_width = workload.shape.resolution.width
        config.validation_config.validation_max_num_frames = workload.shape.frames

        config.model_config.transformer_model_name_or_path = (
            workload.generator.checkpoint
        )
        if workload.generator.subfolder is not None:
            config.model_config.subfolder = workload.generator.subfolder
        # Helios uses the Base checkpoint both as the common pretrained root and
        # as the physical Real/Fake/Critic Score backbone.  Keeping both fields
        # plan-derived avoids leaking a machine-specific path from the template.
        config.model_config.pretrained_model_name_or_path = workload.score.checkpoint
        config.model_config.real_score_model_name_or_path = workload.score.checkpoint
        if workload.score.subfolder is not None:
            config.model_config.critic_subfolder = workload.score.subfolder
        if workload.generator.adapter.rank:
            config.model_config.lora_rank = workload.generator.adapter.rank
            config.model_config.lora_alpha = float(
                workload.generator.adapter.alpha
                or workload.generator.adapter.rank
            )
        if workload.score.adapter.rank:
            config.model_config.critic_lora_rank = workload.score.adapter.rank
            config.model_config.critic_lora_alpha = float(
                workload.score.adapter.alpha or workload.score.adapter.rank
            )

        training = config.training_config
        data_roots = workload.metadata.get("data_roots")
        if data_roots is not None:
            if not isinstance(data_roots, (list, tuple)) or not data_roots:
                raise ContractViolation(
                    "Helios workload.metadata.data_roots must be a non-empty list",
                    path="workload.metadata.data_roots",
                )
            if any(not isinstance(item, str) or not item.strip() for item in data_roots):
                raise ContractViolation(
                    "Helios data roots must be non-empty strings",
                    path="workload.metadata.data_roots",
                )
            config.data_config.gan_data_root = list(data_roots)
        training.max_train_steps = plan.evidence.start_step + plan.evidence.steps
        checkpointing_steps = plan.evidence.metadata.get("checkpointing_steps")
        if checkpointing_steps is not None:
            if isinstance(checkpointing_steps, bool) or not isinstance(checkpointing_steps, int) or checkpointing_steps <= 0:
                raise ContractViolation(
                    "evidence.metadata.checkpointing_steps must be a positive integer",
                    path="evidence.metadata.checkpointing_steps",
                )
            training.checkpointing_steps = checkpointing_steps
        training.train_batch_size = workload.shape.batch_size
        training.latent_window_size = [workload.shape.latent.frames]
        if workload.shape.pyramid:
            training.stage2_num_stages = len(workload.shape.pyramid)
        config.validation_config.validation_latent_window_size = [
            workload.shape.latent.frames
        ]
        checkpoint = _resolve(plan.evidence.checkpoint, base_dir=source_dir)
        training.resume_from_checkpoint = str(checkpoint) if checkpoint else None
        training.gradient_checkpointing = generator_backend.gradient_checkpointing
        training.dfake_gen_update_ratio = workload.schedule.generator_update_every
        training.ema_start_step = workload.schedule.ema_start_step
        training.use_ema = workload.branches.ema
        training.is_train_dmd = workload.branches.dmd
        training.is_dmd_vae_decode = workload.branches.vae_decode
        training.is_use_gan = workload.branches.gan
        training.is_use_reward_model = workload.branches.reward

        if generator_backend.config_path:
            backend_path = _resolve(
                generator_backend.config_path,
                base_dir=source_dir,
            )
            training.dmd_generator_deepspeed_config = str(backend_path)
        if score_backend.config_path:
            backend_path = _resolve(score_backend.config_path, base_dir=source_dir)
            training.dmd_critic_deepspeed_config = str(backend_path)

        return OmegaConf.to_yaml(config, resolve=False)

    def _invocation(
        self,
        plan: ExecutionPlan,
        *,
        source_dir: Path,
        repo_root: Path,
    ) -> LegacyInvocation:
        if plan.legacy is None:
            raise ContractViolation("missing validated legacy entrypoint", path="legacy")
        legacy = plan.legacy
        template = _resolve(legacy.config_path, base_dir=source_dir)
        entrypoint = _resolve(legacy.entrypoint, base_dir=source_dir)
        cwd = _resolve(legacy.cwd, base_dir=source_dir) if legacy.cwd else repo_root
        output_dir = _resolve(plan.evidence.output_dir, base_dir=source_dir)
        if template is None or entrypoint is None or cwd is None or output_dir is None:
            raise ContractViolation("failed to resolve a legacy path", path="legacy")

        node_rank = _multi_node_rank(plan)
        config_suffix = f".node{node_rank}" if plan.hardware.nodes > 1 else ""
        rendered_config = output_dir / f"resolved_helios_config{config_suffix}.yaml"
        rendered_text = self._render_config(
            plan,
            template_path=template,
            output_dir=output_dir,
            source_dir=source_dir,
        )
        master_addr = os.environ.get("DISTILLSCHEDULE_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("DISTILLSCHEDULE_MASTER_PORT", "29500")
        placeholders = {
            "python": legacy.python_executable,
            "entrypoint": str(entrypoint),
            "config": str(rendered_config),
            "repo_root": str(repo_root),
            "plan_id": plan.plan_id,
            "nodes": str(plan.hardware.nodes),
            "local_world_size": str(plan.hardware.local_world_size),
            "world_size": str(plan.hardware.world_size),
            "node_rank": str(node_rank),
            "master_addr": master_addr,
            "master_port": master_port,
        }
        if legacy.command:
            command = tuple(token.format(**placeholders) for token in legacy.command)
        else:
            command = (
                legacy.python_executable,
                str(entrypoint),
                "--config",
                str(rendered_config),
            )
        return LegacyInvocation(
            command=command,
            cwd=cwd,
            environment=self._environment(plan, source_dir=source_dir),
            generated_files={rendered_config: rendered_text},
        )

    def compile(
        self,
        plan: ExecutionPlan,
        *,
        source_dir: Path,
        repo_root: Path,
    ) -> CompiledAdapterPlan:
        self.validate(plan)
        for component in ("generator", "score"):
            self._validate_backend_config(
                component=component, plan=plan, source_dir=source_dir
            )
        protocol = DMDProtocolCompiler().compile(plan)
        warnings: list[str] = []
        if plan.lifecycle.prefetch_distance > 0:
            warnings.append(
                "prefetch_distance is represented in the plan, but the current "
                "Helios legacy entrypoint remains synchronous"
            )
        if plan.rollout.engine == RolloutEngineKind.REMOTE:
            warnings.append(
                "remote rollout is protocol-ready but requires a remote client "
                "integration before legacy execution"
            )
        invocation = (
            self._invocation(
                plan,
                source_dir=source_dir,
                repo_root=repo_root,
            )
            if plan.rollout.engine == RolloutEngineKind.NATIVE
            else None
        )
        if invocation is None:
            warnings.append(
                "external rollout plans compile to a versioned protocol only; "
                "register a remote role runner before execution"
            )
        return CompiledAdapterPlan(
            protocol=protocol,
            invocation=invocation,
            warnings=tuple(warnings),
        )
