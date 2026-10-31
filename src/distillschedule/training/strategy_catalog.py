"""Documentation of actual execution paths; discovery neither loads nor enables them.

Executable applicability remains the existing runtime inventory's responsibility,
using constructed state, precision, dependencies, layout and observed effects.
``automatic_enable=False`` describes discovery, not a ban on measured selection.
"""
from copy import deepcopy


def _cap(category, implementation, applicability, effects, evidence):
    return dict(category=category, implementation=implementation,
                applicability=applicability, observed_effects=effects,
                evidence=evidence, automatic_enable=False)


_STRATEGIES = {
    'resident': dict(backend='torch-callable-resident-v1',
        implementation='distillschedule.open_workload.torch_backend:Session',
        applicability='Single declared device; visible module state or an explicitly admitted native owner',
        selection='execution.mode reference/manual/auto/plan', validation='CPU_REAL_PROGRAM_AND_NATIVE_REFERENCE'),
    'replicated': dict(backend='torch-callable-replicated-v1',
        implementation='distillschedule.open_workload.replicated:ReplicaLifecycle',
        applicability='Explicit initialized group and declared ReplicaSemantics; equal per-rank batch contract',
        selection='post_backward or admitted ddp_overlap', validation='CPU_GLOO_REAL_PROGRAM; GPU_REQUIRES_OWN_ADMISSION'),
    'native-owner-ddp': dict(backend='torch-callable-native-owner-v1',
        implementation='distillschedule.open_workload.collective_owner:NativeOwnerLifecycle',
        applicability='Explicit versioned atomic owner with native role DDP; initialized multi-rank group and rank-strided input',
        selection='Persistent reference or direct manual only; no clone-based calibration or owner overrides',
        validation='CPU_GLOO_NATIVE_OWNER_AND_SD15_SDXL_CONTINUATION; CUDA_NCCL_NOT_RUN'),
    'fsdp2': dict(implementation='distillschedule.open_workload.sharded:ShardedLifecycle',
        applicability='Visible dense module roots, admitted optimizer/state/dtype and forward/backward context',
        selection=['fsdp2_reshard','fsdp2_keep_forward'], validation='CPU_TORCH210_REAL_PROGRAM; NEW_MODEL_GPU_NOT_RUN'),
    'hsdp2': dict(implementation='distillschedule.open_workload.sharded:ShardedLifecycle',
        applicability='FSDP predicates plus actual admitted replicate/shard topology',
        selection=['hsdp2_reshard','hsdp2_keep_forward'], validation='CPU_DISTRIBUTED_PROGRAM; NEW_MODEL_GPU_NOT_RUN'),
    'mixed-layout': dict(implementation='distillschedule.open_workload.mixed_layout:MixedLifecycle',
        applicability='Per-root supported layouts and explicit allocated topology/replica semantics',
        selection='Existing bounded per_module candidate inventory', validation='CPU_DISTRIBUTED_PROGRAM'),
    'role-pools': dict(implementation='distillschedule.open_workload.role_transport:RolePoolExecutor',
        applicability='Explicit existing role-pool Program and transport/capacity contract; not arbitrary opaque owners',
        selection='Existing role-pool calibration/lowering entrypoints', validation='SOURCE_AND_EXISTING_CPU_TESTS; NATIVE_GPU_SOURCE_SCOPED'),
    'native-recipes': dict(implementation='distillschedule.training.native:run_native',
        applicability='W1-W6 original per-recipe source, asset, shape, precision and topology admission',
        selection='Only modes and backends actually exposed by the selected recipe; ZeRO3 is W6-specific',
        validation='HISTORICAL_GPU_RECEIPTS_SEPARATE_FROM_NEW_CPU_INTEGRATION'),
}

_CAPABILITIES = {
    'official-dmd2-fsdp2-reference': _cap('layout_scheduling',
        'distillschedule.training.dmd2_fsdp:DMD2FSDP2Owner',
        'Explicit SD1.5 official DMD2 GAN reference; CUDA world16 A100/Torch2.10, B1/global16 G1/C10/EMA0; native precision and rank-strided inputs',
        ['Actual block/root FSDP2 after all-rank admission; empty Adam remapped before its first update',
         'Native classifier module hooks; global gradient norm; same-topology local-shard checkpoints',
         'No Resident/DDP optimization bundle, fused Adam or clone-based trials'],
        'MEASURED_BASELINE_SOURCE_ABI; CPU_GLOO_REAL_FSDP_NATIVE_STATE; PUBLIC_WORLD16_GPU_QUALIFICATION_PENDING'),
    'official-dmd2-resident': _cap('state_lifecycle_and_execution',
        'distillschedule.training.official_optimizations.dmd2:DMD2OptimizedOwner',
        'Explicit SD1.5 official DMD2 GAN profile; native world16 B1/global16, G1/C10/EMA0; owner must admit mixed precision, DDP, source and assets',
        ['Source-locked Fake epsilon activation retention', 'Packed real/fake classifier trunk with separate native B1 heads',
         'Critic-update cast scope; fused AdamW and native ready-bucket DDP; explicit Tree/min16 environment'],
        'HISTORICAL_NATIVE_GPU_REVIEW; RELOCATED_HELPER_SOURCE_LOCK; PUBLIC_INTEGRATION_GPU_NOT_RUN; NOT_A_SPEED_GUARANTEE'),
    'official-self-forcing-fsdp2': _cap('state_lifecycle_and_execution',
        'distillschedule.training.self_forcing_fsdp:SelfForcingFSDP2Owner',
        'Official Self-Forcing Wan only, not LiveWan; explicit world16 FSDP2/BF16-master-FP32, 368x640/81 output frames, B1/global16 G1/C5/EMA0; source/version/layout admission required',
        ['Source-locked phase-weight and text-KV retention', 'Critic checkpoint attention/FFN/projection reuse and rollout context hoist',
         'Host KV indices, norm/pointwise/RoPE specializations and paired Teacher CFG block execution'],
        'HISTORICAL_NATIVE_GPU_REVIEW; RELOCATED_HELPER_SOURCE_LOCK; PUBLIC_INTEGRATION_GPU_NOT_RUN; NOT_A_SPEED_GUARANTEE'),
    'native-consumed-kv-release': _cap('state_lifecycle',
        'distillschedule.training.self_forcing_owner:SelfForcingOwner._release_consumed_rollout_cache',
        'Explicit official Self-Forcing release-after-commit-v1 model profile; completed native backward/clip and Adam/EMA commit',
        ['Actual Generator/Critic post-commit releases', 'Deduplicated owner storage references relinquished',
         'Native rollout rebuilds caches; no host-visible tensor value reads in release helper'],
        'CPU_TORCH210_CHECKPOINT_ON_OFF_NATIVE_STATE_AND_FRESH_PROCESS_RESTORE; NEW_POLICY_GPU_NOT_RUN'),
    'native-component-checkpointing': _cap('state_lifecycle',
        'distillschedule.training.dmd2_owner:DMD2Owner; distillschedule.open_workload.owner_execution',
        'Opt-in component-v1 on checkpoint-enabled actual SD15/SDXL CPU1 FP32; Generator checkpointing retained',
        ['Independent Generator/Fake flags', 'Real Generator replay and direct Fake epsilon activation retention',
         'Per-role forward/recompute/retention counters; exception cleanup and restart admission'],
        'CPU_TORCH210_LOSS_GRADIENT_ADAM_RNG_AND_FRESH_PROCESS_RESTORE; NOT_DISTRIBUTED_OR_MIXED_GPU_PORTFOLIO'),
    'native-owner-checkpointing': _cap('state_lifecycle',
        'distillschedule.open_workload.owner_execution; distillschedule.training.dmd2_owner:DMD2Owner',
        'Versioned finite stateful owner descriptor; qualified actual SD15/SDXL CPU1 FP32 UNets; paired preupdate admission',
        ['Actual per-G/F checkpoint requests and recompute calls', 'Scoped flags/callback restoration',
         'Source-bound manual/auto/saved-plan and first-resumed-cycle reference gate'],
        'CPU_TORCH210_ORIGINAL_LOSS_GRADIENT_ADAM_RNG_AND_FRESH_PROCESS_RESTORE; NOT_GPU_OR_OPAQUE_FSDP'),
    'frozen-parameter-sharing': _cap('state_lifecycle',
        'distillschedule.open_workload.torch_backend:sharing_inventory',
        'Declared immutable subtrees; identical contents/dtype/layout; no writable/trainable alias',
        ['Actual shared storage','Frozen content validation'], 'CPU_NATIVE_STORAGE_AND_UPDATE_PARITY'),
    'activation-checkpointing': _cap('state_lifecycle',
        'distillschedule.open_workload.torch_backend:Session',
        'Pure replay_safe grad Call, explicit inputs/state versions; never optimizer macrocall replay',
        ['Checkpointed native Call','Real backward recomputation'], 'CPU_OUTPUT_GRADIENT_OPTIMIZER_RNG_PARITY'),
    'selective-result-retention': _cap('state_lifecycle',
        'distillschedule.open_workload.checkpoint_policy:contexts',
        'Checkpointed pure Call plus actual native matmul/attention op and unchanged cache tensors',
        ['selective_save_requests','selective_native_cache_bytes','explicit replay callbacks when present'],
        'CPU_REAL_NATIVE_OP_AND_FULL_STATE_PARITY; NOT_H3_COMPACT_CACHE_REPLACEMENT'),
    'gradient-communication-overlap': _cap('layout_scheduling',
        'distillschedule.open_workload.replicated:ReplicaLifecycle',
        'Explicit replica semantics and admitted single-outstanding forward/backward reducer context',
        ['Actual DDP bucket hook','Completed logical mean reduction'], 'CPU_GLOO_UPDATE_PARITY; GPU_SOURCE_SCOPED'),
    'parameter-reshard-lifetime': _cap('state_lifecycle',
        'distillschedule.open_workload.sharded:ShardedLifecycle',
        'Admitted FSDP2/HSDP2 module roots, optimizer ownership and capacity',
        ['Actual fully_shard parameters','reshard_after_forward','Migrated optimizer moments'], 'CPU_TORCH210_DTENSOR_UPDATE_RESTORE'),
    'legal-call-reordering': _cap('layout_scheduling',
        'distillschedule.search',
        'Existing graph dependencies, parameter versions, resources and native update/sample invariants',
        ['Executed selected operation order','Version ledger and dependency checks'], 'CPU_EXECUTOR_AND_SOURCE_SCOPED_NATIVE_RECEIPTS'),
    'native-state-and-kernel-policies': _cap('state_lifecycle_and_execution',
        'distillschedule.workloads.best_known; distillschedule.strategies',
        'Only original recipe-specific cache/parameter/input/dtype/layout predicates and real kernel backend',
        ['Actual native cache/recompute/transfer counters where provided','Source-bound worker policy'],
        'SEPARATE_W1_W6_NATIVE_RECEIPTS; NOT_AUTOMATIC_NEW_MODEL_SUPPORT'),
}


def describe():
    return dict(strategies=deepcopy(_STRATEGIES), capabilities=deepcopy(_CAPABILITIES))


def resolve(kind, name):
    rows = {'strategy':_STRATEGIES,'capability':_CAPABILITIES}.get(kind)
    if rows is None or name not in rows:
        raise ValueError('unknown ' + str(kind) + ': ' + str(name))
    return deepcopy(rows[name])
