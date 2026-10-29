"""Independent W3 native-history ABI over the existing original GAN owners.

Only the G action uses aligned complete-block materialization through backward.
Detached r_all keeps the original persistent-unshard/window5 owner: every rank
executes its first complete forward before any rank-local repeated forward.
Never nest that owner in an already-materialized phase. No GAN math lives here.
"""
from contextlib import contextmanager

from distillschedule.core.contracts import require
from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY

SCHEMA = "distillschedule.helios_gan_native_history_execution.v1"
SEMANTICS = "original_helios_gan_rank_local_amplified.v1"
REQUIREMENT = "GAN_G_ALIGNED_COMPLETE_BLOCK_PHASE_AND_ORIGINAL_PERSISTENT_R_ALL"


def descriptor():
    return dict(schema_version=SCHEMA, semantics=SEMANTICS, stage_schedule_policy=NATIVE_HISTORY_POLICY,
        execution_requirement=REQUIREMENT, gan=True, n70=False,
        checkpoint_block_class="helios.modules.transformer_helios.HeliosTransformerBlock",
        checkpoint_blocks=40, fsdp_units=41, acquire="G_update_before_original_GAN_helper",
        release="before_original_G_clip_step",
        detached_owner="original_generator_nograd_persistent_unshard_scope_window5",
        original_optimizer_scheduler_ema=True, padding_calls=0,
        numerical_equivalence="NOT_RUN", gpu_capacity_validation="NOT_RUN")


def require_task(task):
    from distillschedule.training.helios_gan_inputs import GANCohortTask, REQUEST
    require(type(task) is GANCohortTask and task.request == REQUEST and task.native_policy == NATIVE_HISTORY_POLICY,
            "independently versioned native GAN history task required", path="helios.gan.native_history")


def validate_contract(task, trace):
    from distillschedule.training.helios_gan_inputs import TRACE_KEY
    require_task(task)
    expected = {TRACE_KEY: task.to_dict(), "resolution": [384, 640],
        "full_latent_shape": list(task.latent_shape), "stage_spatial_shapes": [list(s) for s in task.stage_spatial_shapes],
        "dtype": "bfloat16", "dfake_gen_update_ratio": 5}
    require(trace.manifest["contract"] == expected, "native GAN history geometry/schema changed", path="helios.gan.native_history")
    for step in task.source_steps:
        branch = dict(global_step=step, dfake_gen_update_ratio=5, train_generator=step == task.first_step,
                      train_critic=True, gan=True, reward=False, vae_decode=False)
        require(trace.operation(f"step/{step}/branch")["control"] == branch,
                "native GAN history branch/update contract changed", path="helios.gan.native_history")


def validate_schedules(task, controls):
    from distillschedule.strategies.wan_trace import _validate_native_history_schedules
    require_task(task)
    for rank, trace in controls.items():
        for step in task.source_steps:
            for phase in (("generator", "critic") if step == task.first_step else ("critic",)):
                c = trace.operation(f"step/{step}/{phase}/rollout")["control"]
                require(set(c) == {"pyramid_stage_call_sequence", "call_count", "initial_synchronized_pyramid_stage_exit",
                        "stage_schedule_scope", "stage_schedule_policy", "all_history_zero", "rank"}
                        and c.get("stage_schedule_policy") == NATIVE_HISTORY_POLICY
                        and type(c.get("all_history_zero")) is bool,
                        "native GAN history predicate/policy missing", path="helios.gan.native_history")
                require(all(set(s) == {"stage", "denoising_exit", "call_count", "denoising_step_budget", "latent_shape", "timestep"}
                            for s in c["pyramid_stage_call_sequence"]),
                        "native GAN history stage schema changed", path="helios.gan.native_history")
                require(len(c["pyramid_stage_call_sequence"]) == 3 and all(
                    s["latent_shape"] == [*task.latent_shape[:3], *hw]
                    for s, hw in zip(c["pyramid_stage_call_sequence"], task.stage_spatial_shapes, strict=True)),
                    "native GAN history stage geometry changed", path="helios.gan.native_history")
    # Shared PURE native initial-exit/rank-local-exception checker, not W2's
    # training capability, Resident execution claim, or benchmark request ABI.
    checked = _validate_native_history_schedules(task, controls)
    return {**checked, "semantics": SEMANTICS, "execution_requirement": REQUIREMENT}


def admit_engine(engine):
    # Reuse the strict physical whole-Helios-block/checkpoint validator only.
    # This is not an assertion that W2 math/input/lifecycle covers W3.
    from .helios_native_history import admit_engine as admit_complete_blocks
    return {**admit_complete_blocks(engine), "schema_version": SCHEMA}


@contextmanager
def generator_phase(engine, *, task, with_grad, action_id):
    from helios.modules.transformer_helios import HeliosTransformerBlock
    from .fsdp2_variable_forward import fsdp2_variable_forward_phase
    require_task(task)
    require(with_grad is True and action_id == "g", "GAN history phase is G-only; preserve original detached persistence",
            path="helios.gan.native_history")
    capability = admit_engine(engine)
    with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=True,
            expected_units=capability["fsdp_units"], checkpoint_block_type=HeliosTransformerBlock) as (view, receipt):
        receipt.update(schema_version=SCHEMA, action_id=action_id, task_sha256=task.digest,
                       capability=capability, execution_requirement=REQUIREMENT)
        yield view, receipt


def validate_action(row):
    entries = row["trace_entries"]
    native = [e for e in entries if e.get("native_history_policy") == NATIVE_HISTORY_POLICY]
    if not native:
        return
    require(len(native) == len(entries), "mixed GAN native/normal history entries", path="helios.gan.native_history")
    for entry in entries:
        require(entry["source_rank"] == row["source_rank"] and entry.get("native_history_semantics") == SEMANTICS,
                "native GAN observed rank/semantics changed", path="helios.gan.native_history")
        zero = entry.get("all_history_zero")
        require(type(zero) is bool, "native GAN observed predicate missing", path="helios.gan.native_history")
        sequence = entry["stage_schedule"]
        require([s["stage"] for s in sequence] == [0, 1, 2]
                and all(s["denoising_step_budget"] == (4 if zero else 2) and type(s["call_count"]) is int
                        and 1 <= s["call_count"] <= s["denoising_step_budget"]
                        and s["call_count"] == s["denoising_exit"] + 1 for s in sequence),
                "native GAN observed horizon/exits changed", path="helios.gan.native_history")
    if row["action_id"] == "g":
        phase = row.get("native_history_phase", {})
        require(phase.get("schema_version") == SCHEMA and phase.get("action_id") == "g"
                and phase.get("execution_requirement") == REQUIREMENT and phase.get("with_grad") is True
                and all(e.get("native_task_sha256") == phase.get("task_sha256") for e in entries)
                and phase.get("unshard_count") == phase.get("reshard_count") == 41
                and phase.get("restored") is True and phase.get("backward_count") == phase.get("commit_count") == 1,
                "native GAN actual G phase census changed", path="helios.gan.native_history")
    elif row["action_id"].startswith("r"):
        p = row.get("persistence", {})
        require(row["action_id"] == "r_all" and "native_history_phase" not in row
                and p.get("enabled") is p.get("restored") is True
                and p.get("rollout_batch_count") == 5
                and p.get("scope") == "GENERATOR_NOGRAD_ROLLOUTS_ONLY"
                and p.get("fsdp_module_count") == p.get("parameter_group_count") == 41
                and p.get("set_no_reshard_count") == p.get("restore_reshard_policy_count")
                    == p.get("explicit_reshard_count") == p.get("fully_unsharded_before_exit_count") == 41
                and p.get("unsharded_after_exit_count") == 0,
                "native GAN requires original drained window5 persistence", path="helios.gan.native_history")
