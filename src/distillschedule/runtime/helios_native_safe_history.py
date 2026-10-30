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
    from distillschedule.training.helios_native_safe_stream import require_binding
    from distillschedule.training.helios_inputs import HeliosCohortTask
    from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
    from .fsdp2_variable_forward import fsdp2_variable_forward_phase
    from helios.modules.transformer_helios import HeliosTransformerBlock
    require_binding()
    require(type(task) is HeliosCohortTask and task.native_policy == NATIVE_HISTORY_POLICY and type(with_grad) is bool
            and (action_id == 'g' if with_grad else action_id in {f'r{i}' for i in range(5)}),
            'native SafeBundle requires original aligned Generator action', path='helios.native_safe_stream')
    capability = admit_engine(engine)
    with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=with_grad,
            expected_units=capability['fsdp_units'], checkpoint_block_type=HeliosTransformerBlock) as (view, receipt):
        receipt.update(schema_version=SCHEMA, action_id=action_id, task_sha256=task.digest,
            capability=capability, execution_requirement=REQUIREMENT, mesh_policy='node-hsdp', mesh_shape=[2,8])
        yield view, receipt


def validate_action(row):
    phase = row.get('native_history_phase')
    if phase is None:
        return original.validate_action(row)
    require(phase.get('schema_version') == SCHEMA and phase.get('execution_requirement') == REQUIREMENT
            and phase.get('mesh_policy') == 'node-hsdp' and phase.get('mesh_shape') == [2,8],
            'native SafeBundle actual complete-block phase changed', path='helios.native_safe_stream')
    # Reuse the original rank-local schedule/call/commit validator after the
    # independent new physical capability has been checked. Never mutate a row.
    original.validate_action({**row, 'native_history_phase':{**phase,
        'schema_version':original.SCHEMA, 'execution_requirement':original.REQUIREMENT}})
