"""Default-OFF W3 exact G-input capacity policy over the original full bundle.

Only first hidden tensors of G's original whole-block checkpoints use CPU
storage. C's cropped GAN-G graph/FA3 saves remain on GPU, then drain before G
backward. Limits are admission refusals, not a reservation or native fit proof.
"""
from copy import deepcopy
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, repository_root
from . import helios_gan_training as original

PRESET_ID = 'helios_gan.native_history_fa3_down40_checkpoint_cpu.v1'
SCHEMA = 'distillschedule.helios_gan_checkpoint_hidden_cpu_policy.v1'
IDENTITY = 'distillschedule.helios_gan_checkpoint_hidden_cpu_identity.v1'
CHECKPOINT = 'distillschedule.helios_gan_checkpoint_hidden_cpu_checkpoint.v1'
CAPACITY = 'distillschedule.helios_gan_checkpoint_hidden_cpu_capacity.v1'
SESSION_SCHEMA = 'distillschedule.helios_gan_g_checkpoint_hidden_cpu_execution.v1'
STARTUP_RESERVE = 512 * 2**30
RUNNING_RESERVE = 64 * 2**30
GPU_RESERVE = 512 * 2**20
implementation = original.implementation


def native_contract():
    # W3 same original B1 G geometry; these are NOT W2 node-HSDP owners.
    return dict(schema_version='distillschedule.helios_gan_native_g_hidden_bound.v1',
        shapes=[[1,2940,5120],[1,4560,5120],[1,11040,5120]], dtype='torch.bfloat16',
        checkpoint_blocks=40, max_connected_grad_stages=3)


def critic_boundary_contract():
    return dict(role='original_GAN_G_cached_input_derivative', latent_shape=[1,16,9,24,40],
        hidden_shape=[1,4560,5120], dtype='torch.bfloat16', checkpoint_blocks=40,
        patch_modules=['patch_embedding','patch_short','patch_mid','patch_long'],
        auxiliary_heads=5, checkpoint_dispatches=49, selective_contexts=20,
        selective_FA3_results=40, hidden_input_bytes=1_867_776_000,
        storage='UNCHANGED_GPU', down40='INACTIVE_OUTSIDE_ORIGINAL_C_UPDATE',
        drain='ORIGINAL_AUTOGRAD_GRAD_BEFORE_ORIGINAL_G_BACKWARD')


def capacity_template(storage='pageable'):
    require(storage in ('pageable','pinned'), 'explicit W3 pageable or pinned storage required', path='helios.gan.checkpoint_cpu')
    spec=native_contract()
    size=sum(a*b*c*{'torch.bfloat16':2,'torch.float64':8}[spec['dtype']] for a,b,c in spec['shapes'])*spec['checkpoint_blocks']
    return dict(schema_version=CAPACITY,storage=storage,
        rank_pageable_limit_bytes=size if storage=='pageable' else 0,
        rank_pinned_limit_bytes=size if storage=='pinned' else 0,
        node_pageable_limit_bytes=8*size if storage=='pageable' else 0,
        node_pinned_limit_bytes=8*size if storage=='pinned' else 0)


def validate_capacity(value):
    require(type(value) is dict and value.get('storage') in ('pageable','pinned')
            and value==capacity_template(value['storage'])
            and all(type(v) is int for k,v in value.items() if k.endswith('_bytes')),
            'exact W3 G-only bounded rank/node capacity declaration required', path='helios.gan.checkpoint_cpu')
    return deepcopy(value)


def validate_config(config):
    from .helios_gan_inputs import NATIVE_KIND
    require(config.execution.get('preset')==PRESET_ID and config.execution.get('mode')=='manual'
            and config.data.get('input_kind')==NATIVE_KIND
            and isinstance(config.execution.get('capacity_policy'),str) and bool(config.execution['capacity_policy'])
            and not any(k in config.execution for k in ('plan','implementation','transport')),
            'W3 checkpoint CPU needs explicit native history/full bundle/capacity; no fallback', path='helios.gan.checkpoint_cpu')


def load_capacity(config):
    validate_config(config)
    path=Path(config.execution['capacity_policy']).expanduser()
    if not path.is_absolute():path=config.source.parent/path
    return validate_capacity(json.loads(path.read_text()))


def checkpoint_input_preservation():
    return dict(schema_version=SCHEMA, scope='exact_first_hidden_input_of_original_whole_G_block_checkpoint_only',
        native_contract=native_contract(), critic_boundary=critic_boundary_contract(),
        copies='synchronous_same_dtype_device_restore', parameter_optimizer_offload=False,
        original_checkpoint_callable=True, original_fresh_world16_two_node_process_group_required=True,
        startup_host_reserve_bytes=STARTUP_RESERVE,running_host_reserve_bytes=RUNNING_RESERVE,
        gpu_reserve_bytes=GPU_RESERVE,target_torch_2_10='NOT_RUN',gpu_capacity='NOT_RUN')


def descriptor():
    from distillschedule.strategies.wan_task import NATIVE_HISTORY_POLICY
    value=original.descriptor(NATIVE_HISTORY_POLICY)
    owners=('training/helios_gan_checkpoint_cpu.py','runtime/helios_gan_checkpoint_cpu.py',
            'runtime/helios_checkpoint_inputs_cpu.py')
    return {**value,'schema_version':SCHEMA,'preset':PRESET_ID,
        'checkpoint_input_preservation':checkpoint_input_preservation(),
        'checkpoint_input_source_sha256':{p:file_sha256(repository_root()/'src/distillschedule'/p) for p in owners}}


def validate_identity(identity):
    from distillschedule.runtime.helios_gan_native_history import descriptor as history
    require(type(identity) is dict and identity.get('schema_version')==IDENTITY
            and identity.get('gan_native_history')==history() and identity.get('execution_policy')==descriptor(),
            'W3 checkpoint CPU source/policy identity changed',path='helios.gan.checkpoint_cpu')
    validate_capacity(identity.get('checkpoint_input_capacity'))
    if 'native_owner_layout' in identity:
        original.validate_layout_record(identity['native_owner_layout'])


def _identity_digest(identity):
    return fingerprint({k:v for k,v in identity.items() if k!='native_owner_layout'})


def admit_startup(binding,*,rank):
    from .host_memory import admit_host_memory
    validate_config(binding['config']);validate_identity(binding['identity'])
    require(type(rank) is int and 0<=rank<16 and binding.get('gan_stream') is True
            and binding.get('owns_process_group') is True and binding.get('metadata_group') is not None
            and 'verified_immutable_assets' in binding['identity'],
            'W3 exact original startup/rank/source/assets ownership missing',path='helios.gan.checkpoint_cpu')
    capacity=load_capacity(binding['config'])
    require(capacity==binding['identity']['checkpoint_input_capacity'],
            'W3 checkpoint capacity changed before construction',path='helios.gan.checkpoint_cpu')
    needed=capacity['node_pageable_limit_bytes']+capacity['node_pinned_limit_bytes']
    observation=admit_host_memory(needed,STARTUP_RESERVE)
    binding['checkpoint_input_startup']=dict(identity_sha256=_identity_digest(binding['identity']),rank=rank,
        node_rank=rank//8,extra_retained_node_bytes=needed,observation=observation,
        admission='BEFORE_ORIGINAL_MODEL_CONSTRUCTION_NOT_RESERVATION')
    return binding['checkpoint_input_startup']


def require_binding():
    from .helios_lifecycle import current_training
    binding=current_training()
    require(binding is not None,'W3 checkpoint CPU requires public lifecycle',path='helios.gan.checkpoint_cpu')
    validate_config(binding['config']);validate_identity(binding['identity'])
    startup=binding.get('checkpoint_input_startup',{})
    require(binding.get('implementation')==implementation() and binding.get('effective_configuration')==descriptor()
            and binding.get('checkpoint_input_startup',{}).get('identity_sha256')==_identity_digest(binding['identity'])
            and type(startup.get('rank')) is int and 0<=startup['rank']<16 and startup.get('node_rank')==startup['rank']//8
            and binding['identity']['stream_sha256']==binding['stream']['stream_sha256']
            and binding['identity']['source_fingerprint']==binding['stream']['identity']['source_fingerprint']
            and load_capacity(binding['config'])==binding['identity']['checkpoint_input_capacity'],
            'W3 checkpoint preconstruction source/capacity/effective binding missing',path='helios.gan.checkpoint_cpu')
    return binding
