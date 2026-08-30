"""Typed adapter for LiveWan's Wan1.3B <- Wan14B SF-DMD math."""

from __future__ import annotations

from pathlib import Path
import os

from distillschedule.adapters.base import CompiledAdapterPlan, DMDModelAdapter, LegacyInvocation
from distillschedule.core.capabilities import AdapterCapabilities
from distillschedule.core.contracts import ContractViolation
from distillschedule.core.enums import ModelRole, Precision, RolloutEngineKind, TrainingBackendKind
from distillschedule.core.plan import ExecutionPlan
from distillschedule.workloads.livewan_crossscale import (
    CRITIC_SIZE,
    FULL5_STEPS,
    GENERATOR_SIZE,
    GLOBAL_BATCH,
    LIVEWAN_COMMIT,
    LORA_RANK,
    PIPELINE_6G10S_LAYOUT,
    PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
    PIPELINE_6G10S_SCORE_LEADERS,
    PIPELINE_6G10S_SCORE_NODE_GROUPS,
    PIPELINE_8G8S_LAYOUT,
    TEACHER_SIZE,
    WAN21_COMMIT,
    WORLD_SIZE,
)
from distillschedule.workloads.dmd import DMDProtocolCompiler


class LiveWanSFDMDAdapter(DMDModelAdapter):
    """Compile one source-pinned workload into the DMD-System worker."""

    name = "livewan-sfdmd"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            training_backends=(TrainingBackendKind.NATIVE, TrainingBackendKind.FSDP),
            rollout_engines=(RolloutEngineKind.NATIVE,),
            supports_dynamic_shape=False,
            supports_exact_version_sync=True,
            supports_external_rollout=False,
            features={
                "cross_scale_teacher": True,
                "resident_role_pools": True,
                "fsdp2_teacher_submesh": True,
                "operation_keyed_trace": True,
            },
        )

    @staticmethod
    def _fail(message: str, path: str) -> None:
        raise ContractViolation(message, path=path)

    def validate(self, plan: ExecutionPlan) -> None:
        workload = plan.workload
        if plan.hardware.world_size != WORLD_SIZE:
            self._fail("cross-scale formal workload requires world16", "hardware.world_size")
        if workload.shape.batch_size != GLOBAL_BATCH:
            self._fail("cross-scale formal workload requires global batch16", "workload.shape.batch_size")
        if plan.evidence.start_step != 0 or plan.evidence.steps != FULL5_STEPS:
            self._fail("formal unit is logical steps 0..4", "evidence")
        if workload.schedule.generator_update_every != 5:
            self._fail("G1/C5 requires generator_update_every=5", "workload.schedule.generator_update_every")
        if workload.schedule.critic_update_every != 1:
            self._fail("G1/C5 requires critic_update_every=1", "workload.schedule.critic_update_every")
        if workload.branches.ema:
            self._fail("EMA is measured separately and must be disabled in Full5", "workload.branches.ema")
        if workload.generator.family != "wan2.1" or workload.generator.size != GENERATOR_SIZE:
            self._fail("Generator must be Wan2.1-1.3B", "workload.generator")
        if workload.generator.role != ModelRole.GENERATOR:
            self._fail("Generator role mismatch", "workload.generator.role")
        if workload.generator.precision != Precision.FP32:
            self._fail("source Student parameters must remain FP32", "workload.generator.precision")
        if workload.generator.adapter.kind.lower() != "none":
            self._fail("source Student is full-parameter trainable, not LoRA", "workload.generator.adapter")
        if workload.score.family != "wan2.1" or workload.score.size != CRITIC_SIZE:
            self._fail("Fake Score must be Wan2.1-1.3B", "workload.score")
        if workload.score.role not in {ModelRole.SCORE, ModelRole.CRITIC}:
            self._fail("Fake Score must own the Score/Critic role", "workload.score.role")
        if workload.score.precision != Precision.BF16:
            self._fail("Fake Score frozen base must remain BF16", "workload.score.precision")
        if workload.score.adapter.kind.lower() != "lora" or workload.score.adapter.rank != LORA_RANK:
            self._fail("Fake Score requires trainable LoRA rank32", "workload.score.adapter")
        teacher = workload.teacher
        if teacher is None:
            self._fail("cross-scale workload requires an explicit Teacher", "workload.teacher")
        if teacher.family != "wan2.1" or teacher.size != TEACHER_SIZE or teacher.role != ModelRole.TEACHER:
            self._fail("Teacher must be frozen Wan2.1-14B", "workload.teacher")
        if teacher.adapter.trainable:
            self._fail("Teacher adapter must not be trainable", "workload.teacher.adapter.trainable")
        if teacher.precision != Precision.BF16:
            self._fail("Teacher must remain BF16", "workload.teacher.precision")
        branches = workload.branches
        if not branches.dmd or branches.gan or branches.reward or branches.vae_decode:
            self._fail("only the source SF-DMD branch is authorized", "workload.branches")
        if not branches.classifier_free_guidance:
            self._fail("Teacher CFG must remain enabled", "workload.branches.classifier_free_guidance")
        if [workload.shape.resolution.height, workload.shape.resolution.width] != [368, 640]:
            self._fail("source workload is fixed at 368x640", "workload.shape.resolution")
        latent = workload.shape.latent
        if [latent.channels, latent.frames, latent.height, latent.width] != [16, 21, 46, 80]:
            self._fail("source latent contract is Bx16x21x46x80", "workload.shape.latent")
        meta = dict(workload.metadata)
        if meta.get("livewan_commit") != LIVEWAN_COMMIT:
            self._fail("LiveWan source revision mismatch", "workload.metadata.livewan_commit")
        if meta.get("wan21_commit") != WAN21_COMMIT:
            self._fail("Wan2.1 source revision mismatch", "workload.metadata.wan21_commit")
        if meta.get("sigma_distribution") != "shifted_uniform_continuous_0.02_0.98_shift5":
            self._fail("continuous source sigma contract changed", "workload.metadata.sigma_distribution")
        for component, backend in plan.training_backends.items():
            if not self.capabilities.supports_training(backend.kind):
                self._fail(f"unsupported backend {backend.kind.value}", f"training_backends.{component}.kind")
            if backend.parameter_offload != "none":
                self._fail("parameter offload is outside the frozen workload", f"training_backends.{component}.parameter_offload")
        if plan.rollout.engine != RolloutEngineKind.NATIVE:
            self._fail("formal workload uses the native LiveWan rollout", "rollout.engine")
        if plan.lifecycle.activation_offload.enabled or plan.lifecycle.frozen_block_streaming.enabled:
            self._fail("offload/streaming is outside the controlled comparison", "lifecycle")
        mode = str(plan.metadata.get("execution_mode", "baseline"))
        groups = {group.name: tuple(group.ranks) for group in plan.rank_groups}
        if mode == "baseline":
            expected = tuple(range(WORLD_SIZE))
            if any(groups.get(name) != expected for name in ("generator", "score", "teacher")):
                self._fail("baseline requires all three roles on world16", "rank_groups")
            if plan.training_backends["generator"].kind != TrainingBackendKind.FSDP:
                self._fail("baseline Student must use FSDP2", "training_backends.generator.kind")
            if plan.training_backends["teacher"].kind != TrainingBackendKind.FSDP:
                self._fail("baseline Teacher must use FSDP2", "training_backends.teacher.kind")
        elif mode == "pipeline":
            pool_policy = str(plan.metadata.get("pool_policy", ""))
            layout = {
                "persistent_8g_8score_teacher_fsdp8": PIPELINE_8G8S_LAYOUT,
                "persistent_8g_8score_teacher_fsdp8_score_b2": PIPELINE_8G8S_LAYOUT,
                "persistent_6g_10score_teacher_fsdp10": PIPELINE_6G10S_LAYOUT,
                "persistent_6g_10score_teacher_node_local_hierarchical": PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
            }.get(pool_policy)
            if layout is None:
                self._fail("pipeline pool policy is not registered", "metadata.pool_policy")
            if groups.get("generator") != layout.generator_ranks:
                self._fail("pipeline Generator pool differs from policy", "rank_groups.generator")
            if (
                groups.get("score") != layout.score_ranks
                or groups.get("teacher") != layout.teacher_ranks
            ):
                self._fail("pipeline Score/Teacher pool differs from policy", "rank_groups")
            if plan.training_backends["generator"].kind != TrainingBackendKind.NATIVE:
                self._fail("pipeline Student must be GPU-resident", "training_backends.generator.kind")
            if plan.training_backends["teacher"].kind != TrainingBackendKind.FSDP:
                self._fail("pipeline Teacher must use its score-pool FSDP2 mesh", "training_backends.teacher.kind")
            if pool_policy == "persistent_6g_10score_teacher_node_local_hierarchical":
                generator_options = dict(plan.training_backends["generator"].options)
                teacher_options = dict(plan.training_backends["teacher"].options)
                score_options = dict(plan.training_backends["score"].options)
                optimizer_impl = generator_options.get(
                    "optimizer_impl", "foreach_adamw"
                )
                if optimizer_impl not in {"foreach_adamw", "fused_adamw"}:
                    self._fail(
                        "node-local candidate has an unknown Generator optimizer implementation",
                        "training_backends.generator.options.optimizer_impl",
                    )
                metadata_impl = plan.metadata.get("generator_optimizer_impl")
                if metadata_impl is not None and metadata_impl != optimizer_impl:
                    self._fail(
                        "Generator optimizer metadata differs from its runtime option",
                        "metadata.generator_optimizer_impl",
                    )
                if teacher_options.get("mesh_ranks_by_node") != [
                    list(ranks) for ranks in PIPELINE_6G10S_SCORE_NODE_GROUPS
                ]:
                    self._fail(
                        "node-local Teacher shard groups differ from the registered topology",
                        "training_backends.teacher.options.mesh_ranks_by_node",
                    )
                if teacher_options.get("mesh_ranks") not in (None, (), []):
                    self._fail(
                        "node-local Teacher candidate must disable the flat Score10 mesh",
                        "training_backends.teacher.options.mesh_ranks",
                    )
                if score_options.get("gradient_sync") != "hierarchical_node_local_sum_leader_sum_broadcast_divide":
                    self._fail(
                        "node-local candidate requires exact hierarchical Critic gradient synchronization",
                        "training_backends.score.options.gradient_sync",
                    )
                if score_options.get("leader_ranks") != list(PIPELINE_6G10S_SCORE_LEADERS):
                    self._fail(
                        "hierarchical Critic leader ranks differ from the registered topology",
                        "training_backends.score.options.leader_ranks",
                    )
        else:
            self._fail("execution_mode must be baseline or pipeline", "metadata.execution_mode")

    def compile(
        self,
        plan: ExecutionPlan,
        *,
        source_dir: Path,
        repo_root: Path,
    ) -> CompiledAdapterPlan:
        self.validate(plan)
        protocol = DMDProtocolCompiler().compile(plan)
        mode = str(plan.metadata.get("execution_mode", "baseline"))
        if mode not in {"baseline", "pipeline"}:
            self._fail("execution_mode must be baseline or pipeline", "metadata.execution_mode")
        output_dir = Path(plan.evidence.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (source_dir / output_dir).resolve()
        node_rank = int(os.environ.get("DISTILLSCHEDULE_NODE_RANK", "0"))
        suffix = f".node{node_rank}" if plan.hardware.nodes > 1 else ""
        command = (
            plan.legacy.python_executable if plan.legacy else "python",
            str(repo_root / "scripts/distillschedule/livewan_crossscale_entrypoint.py"),
            "--plan",
            str(output_dir / f"resolved_plan{suffix}.json"),
            "--mode",
            mode,
        )
        invocation = LegacyInvocation(
            command=command,
            cwd=repo_root,
            environment={
                "DISTILLSCHEDULE_LIVEWAN_COMMIT": LIVEWAN_COMMIT,
                "DISTILLSCHEDULE_WAN21_COMMIT": WAN21_COMMIT,
            },
        )
        return CompiledAdapterPlan(protocol=protocol, invocation=invocation)
