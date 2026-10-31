"""Explicit W2 capacity candidate: original G checkpoint hidden inputs on CPU.

No parameter/optimizer offload, approximation, new sampler or alternate trainer.
The original native-history/full SafeBundle remains on. Host observations are
fresh admission checks, not reservations against unrelated processes.
"""
from copy import deepcopy
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from . import helios_native_safe_stream as original

PRESET_ID = 'helios_nogan.native_history_safe_ed8_checkpoint_cpu.v1'
SCHEMA = 'distillschedule.helios_checkpoint_hidden_cpu_policy.v1'
IDENTITY = 'distillschedule.helios_checkpoint_hidden_cpu_identity.v1'
CHECKPOINT = 'distillschedule.helios_checkpoint_hidden_cpu_checkpoint.v1'
CAPACITY = 'distillschedule.helios_checkpoint_hidden_cpu_capacity.v1'
STARTUP_RESERVE = 512 * 2**30
RUNNING_RESERVE = 64 * 2**30
GPU_RESERVE = 512 * 2**20
implementation = original.implementation
effective_flags = original.effective_flags
validate_environment = original.validate_environment
owner_layout = original.owner_layout
validate_layout_record = original.validate_layout_record
history_descriptor = original.history_descriptor


def native_contract():
    # Original B1 current latents: 9 x {12x20,24x40,48x80}, patch (1,2,2).
    # The unchanged full-resolution short/mid/long histories add respectively
    # 2*24*40 + 1*12*20 + 4*6*10 = 2400 tokens in every pyramid stage.
    return dict(schema_version='distillschedule.helios_native_g_hidden_bound.v1',
        shapes=[[1,2940,5120],[1,4560,5120],[1,11040,5120]], dtype='torch.bfloat16',
        checkpoint_blocks=40, max_connected_grad_stages=3)


def _rank_bytes():
    spec = native_contract()
    # Production BF16, synthetic double fixtures only replace native_contract.
    size = {'torch.bfloat16':2, 'torch.float64':8}[spec['dtype']]
    return sum(a*b*c*size for a,b,c in spec['shapes']) * spec['checkpoint_blocks']


def capacity_template(storage='pageable'):
    require(storage in ('pageable','pinned'), 'explicit pageable or pinned storage required', path='helios.checkpoint_cpu')
    size = _rank_bytes()
    return dict(schema_version=CAPACITY, storage=storage,
        rank_pageable_limit_bytes=size if storage=='pageable' else 0,
        rank_pinned_limit_bytes=size if storage=='pinned' else 0,
        node_pageable_limit_bytes=8*size if storage=='pageable' else 0,
        node_pinned_limit_bytes=8*size if storage=='pinned' else 0)


def validate_capacity(value):
    require(type(value) is dict and value.get('storage') in ('pageable','pinned')
            and value == capacity_template(value['storage'])
            and all(type(v) is int for k,v in value.items() if k.endswith('_bytes')),
            'exact original three-stage rank/node pinned/pageable capacity declaration required', path='helios.checkpoint_cpu')
    return deepcopy(value)


def load_capacity(config):
    path = Path(config.execution['capacity_policy']).expanduser()
    if not path.is_absolute(): path = config.source.parent / path
    return validate_capacity(json.loads(path.read_text()))


def checkpoint_input_preservation():
    """Static capability metadata; native source costs belong to descriptor()."""
    return dict(schema_version=SCHEMA,
            scope='exact_first_hidden_input_of_original_whole_G_block_checkpoint_only',
            native_contract=native_contract(), copies='synchronous_same_dtype_device_restore',
            other_checkpoint_inputs='KEEP_UNCHANGED_shared_encoder_rope_not_duplicated',
            parameter_optimizer_offload=False, original_checkpoint_callable=True,
            original_fresh_world16_two_node_process_group_required=True,
            startup_host_reserve_bytes=STARTUP_RESERVE, running_host_reserve_bytes=RUNNING_RESERVE,
            gpu_reserve_bytes=GPU_RESERVE, target_torch_2_10='NOT_RUN', gpu_capacity='NOT_RUN')


def descriptor():
    return {**original.descriptor(), 'schema_version':SCHEMA, 'preset':PRESET_ID,
            'checkpoint_input_preservation':checkpoint_input_preservation()}


def validate_config(config):
    from .helios_inputs import NATIVE_KIND
    require(config.data.get('input_kind') == NATIVE_KIND
            and config.execution.get('mode') == 'manual' and config.execution.get('preset') == PRESET_ID
            and isinstance(config.execution.get('capacity_policy'),str) and bool(config.execution['capacity_policy'])
            and not any(key in config.execution for key in ('plan','implementation','transport')),
            'checkpoint CPU candidate requires explicit native history/full bundle and capacity file', path='helios.checkpoint_cpu')


def bind_effective(*, args, policy, environment=None):
    actual = original.bind_effective(args=args, policy=policy, environment=environment)
    binding = _binding()
    capacity = load_capacity(binding['config'])
    require(capacity == binding['identity']['checkpoint_input_capacity'],
            'checkpoint CPU capacity changed after public admission', path='helios.checkpoint_cpu')
    value = {**descriptor(), 'candidate_bundle':actual['candidate_bundle'], 'checkpoint_input_capacity':capacity}
    value['effective_configuration_sha256'] = fingerprint(value)
    return value


def validate_identity(identity, *, optimized):
    require(optimized is True and identity.get('schema_version') == IDENTITY
            and identity.get('native_history') == history_descriptor()
            and identity.get('execution_policy') == descriptor(),
            'checkpoint CPU execution identity changed', path='helios.checkpoint_cpu')
    validate_capacity(identity.get('checkpoint_input_capacity'))


def _binding():
    from .helios_lifecycle import current_training
    binding = current_training()
    require(binding is not None, 'checkpoint CPU requires admitted public lifecycle', path='helios.checkpoint_cpu')
    validate_config(binding['config'])
    validate_identity(binding['identity'], optimized=True)
    return binding


def admit_startup(binding, *, rank):
    from .host_memory import admit_host_memory
    validate_identity(binding['identity'], optimized=True)
    require(type(rank) is int and 0<=rank<16 and 'verified_immutable_assets' in binding['identity']
            and binding.get('owns_process_group') is True and binding.get('metadata_group') is not None,
            'checkpoint CPU startup requires original verified source/assets/rank admission', path='helios.checkpoint_cpu')
    capacity = load_capacity(binding['config'])
    require(capacity == binding['identity']['checkpoint_input_capacity'],
            'checkpoint CPU capacity changed before construction', path='helios.checkpoint_cpu')
    needed = capacity['node_pageable_limit_bytes'] + capacity['node_pinned_limit_bytes']
    observation = admit_host_memory(needed, STARTUP_RESERVE)
    binding['checkpoint_input_startup'] = dict(identity_sha256=fingerprint({k:v for k,v in binding['identity'].items() if k != 'native_owner_layout'}), rank=rank,
        node_rank=rank//8, extra_retained_node_bytes=needed, observation=observation,
        admission='BEFORE_ORIGINAL_MODEL_CONSTRUCTION_NOT_RESERVATION')
    return binding['checkpoint_input_startup']


def require_binding():
    binding = _binding()
    require(binding.get('implementation') == implementation() and binding.get('effective_configuration') is not None
            and binding.get('checkpoint_input_startup',{}).get('identity_sha256') == fingerprint(
                {k:v for k,v in binding['identity'].items() if k != 'native_owner_layout'}),
            'checkpoint CPU preconstruction capacity/effective source binding missing', path='helios.checkpoint_cpu')
    return binding
