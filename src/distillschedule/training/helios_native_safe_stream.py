"""Explicit W2 native-history/full SafeBundle lifecycle identity.

The original SafeBundle, input sampler, P0 model owners and training helpers
remain authoritative. This policy only composes their already separate ABIs.
"""
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from . import helios_safe_stream as safe

PRESET_ID = 'helios_nogan.native_history_safe_ed8.v1'
SCHEMA = 'distillschedule.helios_native_safe_stream_policy.v1'
IDENTITY = 'distillschedule.helios_native_safe_stream_identity.v1'
CHECKPOINT = 'distillschedule.helios_native_safe_stream_checkpoint_policy.v1'
implementation = safe.implementation
effective_flags = safe.effective_flags
validate_environment = safe.validate_environment
owner_layout = safe.owner_layout
validate_layout_record = safe.validate_layout_record


def history_descriptor():
    from distillschedule.runtime.helios_native_safe_history import descriptor
    return descriptor()


def descriptor():
    return {**safe.descriptor(), 'schema_version':SCHEMA, 'preset':PRESET_ID,
            'native_history':history_descriptor()}


def validate_config(config):
    from .helios_inputs import NATIVE_KIND
    require(config.data.get('input_kind') == NATIVE_KIND
            and config.execution.get('mode') == 'manual'
            and config.execution.get('preset') == PRESET_ID
            and not any(key in config.execution for key in ('plan','implementation','transport','capacity_policy')),
            'native SafeBundle requires its explicit native-history stream/preset only', path='helios.native_safe_stream')


def bind_effective(*, args, policy, environment=None):
    actual = safe.bind_effective(args=args, policy=policy, environment=environment)
    value = {**descriptor(), 'candidate_bundle':actual['candidate_bundle']}
    value['effective_configuration_sha256'] = fingerprint(value)
    return value


def validate_identity(identity, *, optimized):
    require(optimized is True and identity.get('schema_version') == IDENTITY
            and identity.get('native_history') == history_descriptor()
            and identity.get('execution_policy') == descriptor(),
            'native SafeBundle checkpoint execution identity changed', path='helios.native_safe_stream')


def require_binding():
    from .helios_lifecycle import current_training
    binding = current_training()
    require(binding is not None, 'native SafeBundle requires its admitted public lifecycle', path='helios.native_safe_stream')
    validate_config(binding['config'])
    validate_identity(binding['identity'], optimized=True)
    require(binding.get('implementation') == implementation() and binding.get('effective_configuration') is not None,
            'native SafeBundle effective implementation missing', path='helios.native_safe_stream')
    return binding
