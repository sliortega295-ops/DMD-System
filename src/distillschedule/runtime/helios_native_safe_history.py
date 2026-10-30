"""W2 node-HSDP native-history action phases; no training math or new sampler."""
from contextlib import contextmanager

from distillschedule.core.contracts import require
from . import helios_native_history as original

SCHEMA = 'distillschedule.helios_native_safe_history_execution.v1'
REQUIREMENT = 'NODE_HSDP_ALIGNED_COMPLETE_BLOCK_PHASE_WITH_FULL_SAFE_BUNDLE'


def descriptor():
    return {**original.descriptor(), 'schema_version':SCHEMA, 'execution_requirement':REQUIREMENT,
            'mesh_policy':'node-hsdp', 'mesh_shape':[2,8], 'rollout_window':5,
            'persistent_unshard':False, 'candidate_bundle':'safe-nogan-ed8-v1'}


def admit_engine(engine):
    from distillschedule.training.helios_safe_stream import owner_layout
    # Whole blocks/checkpoint callable AND actual FP32 DTensor 2x8 layout.
    # No CPU fake, equal-unit-count substitute, or metadata-only admission.
    capability = original.admit_engine(engine)
    layout = owner_layout(engine, engine)
    return {**capability, 'schema_version':SCHEMA, 'generator_topology':layout['topology']}


@contextmanager
def generator_phase(engine, *, task, with_grad, action_id):
    from distillschedule.training import helios_native_safe_stream, helios_checkpoint_cpu
    from distillschedule.training.helios_lifecycle import current_training
    binding = current_training()
    candidate = (binding is not None and binding["config"].execution.get("preset") == helios_checkpoint_cpu.PRESET_ID)
    require_binding = helios_checkpoint_cpu.require_binding if candidate else helios_native_safe_stream.require_binding
    from distillschedule.training.helios_inputs import HeliosCohortTask
    from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
    from .fsdp2_variable_forward import fsdp2_variable_forward_phase
    from helios.modules.transformer_helios import HeliosTransformerBlock
    require_binding()
    require(type(task) is HeliosCohortTask and task.native_policy == NATIVE_HISTORY_POLICY and type(with_grad) is bool
            and (action_id == 'g' if with_grad else action_id in {f'r{i}' for i in range(5)}),
            'native SafeBundle requires original aligned Generator action', path='helios.native_safe_stream')
    capability = admit_engine(engine)
    registration = None
    if candidate and with_grad:
        from .helios_checkpoint_inputs_cpu import generator_scope
        # Admission must see optimizer-owned sharded FP32 DTensor slots.
        # FSDP materialization temporarily installs different compute Parameters;
        # only installation of the already-admitted wrapper belongs inside it.
        registration = generator_scope(engine, task=task)
    with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=with_grad,
            expected_units=capability['fsdp_units'], checkpoint_block_type=HeliosTransformerBlock) as (view, receipt):
        receipt.update(schema_version=SCHEMA, action_id=action_id, task_sha256=task.digest,
            capability=capability, execution_requirement=REQUIREMENT, mesh_policy='node-hsdp', mesh_shape=[2,8])
        if registration is not None:
            try:
                with registration:
                    yield view, receipt
            finally:
                receipt["checkpoint_input_cpu"] = registration.receipt()
                # Shared with the rank result before dispatch, so an original
                # helper error still leaves executed/copy/lifetime evidence.
                binding.setdefault("checkpoint_input_executions", []).append(receipt["checkpoint_input_cpu"])
        else:
            if candidate:
                from .helios_checkpoint_inputs_cpu import SCHEMA as CPU_SCHEMA
                receipt["checkpoint_input_cpu"] = dict(schema_version=CPU_SCHEMA, policy=helios_checkpoint_cpu.PRESET_ID, enabled=False)
            yield view, receipt


def validate_action(row):
    phase = row.get('native_history_phase')
    if phase is None:
        return original.validate_action(row)
    require(phase.get('schema_version') == SCHEMA and phase.get('execution_requirement') == REQUIREMENT
            and phase.get('mesh_policy') == 'node-hsdp' and phase.get('mesh_shape') == [2,8],
            'native SafeBundle actual complete-block phase changed', path='helios.native_safe_stream')
    # Validate the original rank-local source trace before deriving the new
    # restore census from its actual exits. Never replace or rewrite a schedule.
    original.validate_action({**row, 'native_history_phase':{**phase,
        'schema_version':original.SCHEMA, 'execution_requirement':original.REQUIREMENT}})
    from distillschedule.training.helios_lifecycle import current_training
    from distillschedule.training import helios_checkpoint_cpu
    binding = current_training()
    if "checkpoint_input_cpu" in phase or (binding is not None and
            binding["config"].execution.get("preset") == helios_checkpoint_cpu.PRESET_ID):
        from .helios_checkpoint_inputs_cpu import validate_phase_receipt
        with_grad = row['action_id'] == 'g'
        require(not with_grad or len(row['trace_entries']) == 1,
                'checkpoint CPU requires one original Generator trace', path='helios.native_safe_stream')
        validate_phase_receipt(phase.get("checkpoint_input_cpu"), with_grad=row["action_id"] == 'g',
                               task_sha256=phase.get("task_sha256"),
                               generator_version=row.get("physical_versions_before",{}).get("generator"), rank=row.get("source_rank"),
                               stage_schedule=row['trace_entries'][0]['stage_schedule'] if with_grad else None)
