"""W2-only native rank-local history protocol and aligned FSDP phase owner.

The original data/scheduler/helpers own all random decisions and training math.
This opt-in ABI permits unequal non-updating prefixes only while the complete
Generator is materialized. It does not widen a benchmark request or plan.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial

from distillschedule.core.contracts import require
from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY

SCHEMA = "distillschedule.helios_native_history_execution.v1"
SEMANTICS = "original_helios_nogan_rank_local_amplified.v1"
REQUIREMENT = "FSDP2_ALIGNED_MATERIALIZATION_WITH_ORIGINAL_REDUCE_AND_STEP"


def descriptor():
    return dict(schema_version=SCHEMA, stage_schedule_policy=NATIVE_HISTORY_POLICY,
        execution_requirement=REQUIREMENT, checkpoint_block_class="helios.modules.transformer_helios.HeliosTransformerBlock",
        checkpoint_blocks=40, fsdp_units=41, acquire="each_G_update_or_detached_rollout_action",
        release="before_original_G_clip_step_or_at_detached_rollout_exit",
        original_optimizer_scheduler_ema=True, padding_calls=0, numerical_equivalence="NOT_RUN",
        gpu_capacity_validation="NOT_RUN")


def validate_contract(task, trace):
    from distillschedule.training.helios_inputs import TRACE_KEY
    expected = dict(resolution=[384,640], full_latent_shape=list(task.latent_shape),
        stage_spatial_shapes=[list(s) for s in task.stage_spatial_shapes], dtype="bfloat16", dfake_gen_update_ratio=5)
    require(set(trace.manifest["contract"]) == set(expected) | {TRACE_KEY}
            and all(trace.manifest["contract"].get(key) == value for key, value in expected.items()),
            "native history original geometry/precision changed", path="helios.native_history")
    for step in task.source_steps:
        branch = dict(global_step=step, dfake_gen_update_ratio=5, train_generator=step == task.first_step,
                      train_critic=True, gan=False, reward=False, vae_decode=False)
        require(trace.operation(f"step/{step}/branch")["control"] == branch,
                "native history original branch/update contract changed", path="helios.native_history")


def validate_schedules(task, controls):
    from distillschedule.strategies.wan_trace import _validate_native_history_schedules
    for rank, trace in controls.items():
        for step in task.source_steps:
            for phase in (("generator", "critic") if step == task.first_step else ("critic",)):
                control = trace.operation(f"step/{step}/{phase}/rollout")["control"]
                require(set(control) == {"pyramid_stage_call_sequence", "call_count",
                        "initial_synchronized_pyramid_stage_exit", "stage_schedule_scope",
                        "stage_schedule_policy", "all_history_zero", "rank"}
                        and control.get("stage_schedule_policy") == NATIVE_HISTORY_POLICY
                        and type(control.get("all_history_zero")) is bool,
                        "native history policy/predicate missing", path="helios.native_history")
                require(all(set(item) == {"stage", "denoising_exit", "call_count", "denoising_step_budget",
                                          "latent_shape", "timestep"}
                            for item in control["pyramid_stage_call_sequence"]),
                        "native history stage schema changed", path="helios.native_history")
    # Pure common-initial-exit/local-exception validation; its Resident-only
    # execution requirement is NOT inherited by this independently bound ABI.
    checked = _validate_native_history_schedules(task, controls)
    return {**checked, "semantics":SEMANTICS, "execution_requirement":REQUIREMENT}


def admit_engine(engine):
    """Bind actual FSDP owners, pinned whole blocks and installed checkpoint.

    Model/source/FP32 master/mixed precision admission remains original P0's
    responsibility. This check cannot be satisfied by sharding interior LoRA
    Linear modules, by substituting a block type, or by merely matching 41.
    """
    from torch.distributed.fsdp import FSDPModule
    from helios.modules.transformer_helios import HeliosTransformerBlock
    from .fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    root = engine.module
    require(isinstance(root, FSDPModule), "native history requires actual FSDP root", path="helios.native_history")
    modules = tuple(root.modules())
    blocks = [m for m in modules if m is not root and isinstance(m, HeliosTransformerBlock)]
    units = [m for m in modules if isinstance(m, FSDPModule)]
    require(len(blocks) == 40 and len(units) == 41
            and {id(m) for m in units if m is not root} == {id(m) for m in blocks},
            "native history requires exactly the pinned 40 whole checkpoint blocks and root",
            path="helios.native_history")
    owners = [m for m in modules if "_gradient_checkpointing_func" in vars(m)]
    require(len(owners) == 1 and getattr(owners[0], "gradient_checkpointing", False) is True,
            "native history requires one active original checkpoint owner", path="helios.native_history")
    owner, options = owners[0], {}
    fn = owner._gradient_checkpointing_func
    while isinstance(fn, partial):
        require(not fn.args and not (set(fn.keywords) & set(options)),
                "ambiguous native checkpoint partial", path="helios.native_history")
        options.update(fn.keywords); fn = fn.func
    require(fn is fsdp2_metadata_tolerant_checkpoint
            and not set(options) - {"operation_source", "function_wrapper"}
            and type(options.get("operation_source", False)) is bool
            and ("function_wrapper" not in options or options["function_wrapper"] is
                 getattr(owner, "_distillschedule_lora_checkpoint_function_wrapper", None)),
            "native history original checkpoint function changed", path="helios.native_history")
    return dict(schema_version=SCHEMA, checkpoint_blocks=len(blocks), fsdp_units=len(units),
                checkpoint_block_class=descriptor()["checkpoint_block_class"], checkpoint_policy="original_metadata_tolerant")


@contextmanager
def generator_phase(engine, *, task, with_grad, action_id):
    """Only aligned graph action boundaries acquire; original step releases."""
    from .fsdp2_variable_forward import fsdp2_variable_forward_phase
    from helios.modules.transformer_helios import HeliosTransformerBlock
    require(task.native_policy == NATIVE_HISTORY_POLICY
            and type(with_grad) is bool and isinstance(action_id, str)
            and (action_id == "g" if with_grad else action_id.startswith("r")),
            "native history phase must bind an original G action", path="helios.native_history")
    capability = admit_engine(engine)
    with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=with_grad,
            expected_units=capability["fsdp_units"], checkpoint_block_type=HeliosTransformerBlock) as (view, receipt):
        receipt.update(schema_version=SCHEMA, action_id=action_id, task_sha256=task.digest,
                       capability=capability, execution_requirement=REQUIREMENT)
        yield view, receipt


def validate_action(row):
    """Executed per-rank census, not configuration-derived success counters."""
    entries = row["trace_entries"]
    native = [e for e in entries if e.get("native_history_policy") == NATIVE_HISTORY_POLICY]
    if not native:
        return
    require(len(native) == len(entries), "mixed native and old W2 action entries", path="helios.native_history")
    for entry in entries:
        require(entry["source_rank"] == row["source_rank"]
                and entry.get("native_history_semantics") == SEMANTICS,
                "native history observed rank/semantics changed", path="helios.native_history")
        zero = entry.get("all_history_zero")
        require(type(zero) is bool, "native history observed predicate missing", path="helios.native_history")
        sequence = entry["stage_schedule"]
        require([item["stage"] for item in sequence] == [0,1,2]
                and all(item["denoising_step_budget"] == (4 if zero else 2)
                        and type(item["call_count"]) is int
                        and 1 <= item["call_count"] <= item["denoising_step_budget"]
                        and item["call_count"] == item["denoising_exit"] + 1
                        for item in sequence),
                "native history observed horizon/exits changed", path="helios.native_history")
    if row["action_id"] == "g" or row["action_id"].startswith("r"):
        phase = row.get("native_history_phase", {})
        grad = row["action_id"] == "g"
        require(phase.get("schema_version") == SCHEMA and phase.get("action_id") == row["action_id"]
                and phase.get("execution_requirement") == REQUIREMENT and phase.get("with_grad") is grad
                and all(entry.get("native_task_sha256") == phase.get("task_sha256") for entry in entries)
                and phase.get("unshard_count") == phase.get("reshard_count") == 41
                and phase.get("restored") is True
                and phase.get("backward_count") == phase.get("commit_count") == int(grad),
                "native history actual phase acquire/release/commit census changed", path="helios.native_history")
