"""D1: pinned original Step10 state to ordinary named CPU G/C/EMA owners.

This is an admission/materialization boundary, not a Program, another binary
checkpoint format, or a replacement optimizer. Native CUDA/FSDP loaders remain
unchanged. A reader fixture is explicitly not original W2 warm-update evidence.
No caller-owned model is accepted: all owners stay private until the complete
G2/C10/EMA10 transaction has passed validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

from .helios_public_assets import (EVIDENCE_KINDS, checked_pin, load_base, local_file,
                                   read_json, require, runtime_identity)
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, repository_root


SCHEMA = 'distillschedule.helios_nogan_cpu_warm_assets.v1'
SCOPE = 'W2_STEP10_CPU1_STATE_MATERIALIZATION_NOT_DISTRIBUTED_OR_LIFECYCLE'
POLICY = MappingProxyType(dict(lora_rank=256,gradient_clip=1.,generator_lr=2e-6,critic_lr=4e-7,
              betas=(0.,.999),eps=1e-8,weight_decay=1e-3,scheduler='constant_LambdaLR',
              ema_decay=.99,ema_update_after_step=750))
READER_ONLY = 'STATE_READER_FIXTURE_ONLY'
ORIGINAL = 'OPERATOR_SUPPLIED_ORIGINAL_W2_STEP10'
CPU_WARM = 'ACTUAL_ORIGINAL_W2_CPU_TEST_WARM_NOT_HISTORICAL_STEP10'
CPU_WARM_RECEIPT = 'distillschedule.helios_nogan_actual_cpu_warm_receipt.v2'


def native_policy():
    """A fresh JSON envelope value, never an alias of the native policy pin."""
    return dict(POLICY,betas=list(POLICY['betas']))


def source_identity():
    """Re-admit the original class and the existing native state/policy owners."""
    from distillschedule.adapters.helios.reference_model import reference_model_identity
    files = ('runtime/critic_pool_helios.py','runtime/critic_pool_checkpoint.py','runtime/sparse_ema.py',
             'src/distillschedule/runtime/fsdp2_static.py',
             'scripts/distillschedule/probe_static_fsdp2_p0_world16.py')
    root = repository_root()
    return dict(model=reference_model_identity(backend='torch-sdpa'),
                native={name:file_sha256(root/name) for name in files})


def _freeze(value):
    if isinstance(value,dict):
        return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class WarmBootstrap:
    engines: object
    ema: object
    identity: object

    def step_ema(self):
        """Bind, without replacing, the native sparse EMA recurrence."""
        return self.ema.step(self.engines['generator'].module.parameters())

    def state_views(self):
        from .helios_step_owner import _SparseEMAView
        from .native_state import engine_state
        views = engine_state(self.engines)
        views['ema'] = _SparseEMAView(self.ema,self.engines['generator'])
        return views


def _cpu1():
    import torch
    require(torch.get_default_dtype()==torch.float32, 'W2 CPU1 requires default FP32 before model construction')
    require(all(os.environ.get(k,default)==default for k,default in
                (('WORLD_SIZE','1'),('LOCAL_WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')))
            and not torch.distributed.is_initialized() and not torch.cuda.is_initialized(),
            'W2 warm materialization requires CPU1/rank0 without distributed/CUDA ownership')


def _inside(root, relative):
    from .helios_reference_assets import _file
    path = _file(root,relative)
    require(path.is_file(), 'missing logical/EMA payload: '+str(path))
    return path


def _admit(path):
    from .helios_reference_assets import asset_config, verify_pins, load_bootstrap
    from distillschedule.search.helios_request import OriginalHeliosRequest
    value = read_json(path)
    require(isinstance(value,dict) and set(value)=={'schema_version','scope','precision','base_evidence_kind',
        'source','runtime','request','native_assets','generator','critic','budget','provenance'},
        'exact versioned W2 warm CPU asset envelope required')
    require(value['schema_version']==SCHEMA and value['scope']==SCOPE
            and value['precision']=='WAN_NATIVE_MIXED_BF16' and value['base_evidence_kind'] in EVIDENCE_KINDS,
            'wrong W2 warm CPU schema/scope/precision/evidence')
    require(OriginalHeliosRequest.from_dict(value['request'])==OriginalHeliosRequest(),
            'only original W2 No-GAN33/222/world16/B1 request')
    require(value['source']==source_identity() and value['runtime']==runtime_identity(),
            'original source or dependency runtime identity changed')
    budget = value['budget']
    require(isinstance(budget,dict) and set(budget)=={'max_parameters','max_loaded_bytes'}
            and type(budget['max_parameters']) is int and 0 < budget['max_parameters'] <= 100000000
            and type(budget['max_loaded_bytes']) is int and 0 < budget['max_loaded_bytes'] <= 2**31,
            'explicit bounded CPU parameter/loaded-byte budget required')
    native = asset_config(value['native_assets'])
    for row in native['assets'].values():
        root = Path(row['path'])
        require(root.resolve()==root and root.is_dir(), 'canonical local native asset directory required')
        read_json(_inside(root,row['identity_file']))
    verify_pins(native)
    row = native['assets']['logical_checkpoint']
    root = Path(row['path'])
    # Reject duplicate JSON keys/nonfinite metadata before the unchanged native
    # package reader (whose ABI intentionally remains untouched) sees it.
    manifest = read_json(root/'manifest.json')
    ema_path = _inside(root,row['ema_manifest']['identity_file'])
    metadata = read_json(ema_path)
    require(set(manifest['components'])=={'generator','critic'}
            and type(manifest['ema']['optimization_step']) is int and manifest['ema']['optimization_step']==10,
            'complete original two-role/EMA10 logical state required')
    paths, loaded_bytes = set(), 0
    for role, native_role in (('generator','generator_model'),('critic','score_model')):
        spec = value[role]
        require(isinstance(spec,dict) and set(spec)=={'config','weights'} and isinstance(spec['weights'],list)
                and spec['weights'], 'explicit complete safe base/config required')
        config = checked_pin(spec['config'])
        native_row = native['assets'][native_role]
        require(config==Path(native_row['path'])/native_row['identity_file']
                and spec['config']['sha256']==native_row['sha256'], 'safe base config is not its pinned native role')
        for pin in spec['weights']:
            payload = checked_pin(pin)
            require(payload not in paths, 'aliased base payloads')
            paths.add(payload); loaded_bytes += pin['bytes']
    for role in ('generator','critic'):
        spec = manifest['components'][role]
        require(set(spec['payloads'])=={'master','first_moment','second_moment'}, 'complete FP32 Adam payloads required')
        ranges = sorted((r['offset_elements'],r['offset_elements']+r['numel']) for r in spec['records'].values())
        require(ranges[0][0]==0 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
                and type(spec['logical_numel']) is int and ranges[-1][1]==spec['logical_numel'],
                'logical records must cover the exact nonoverlapping contiguous payload')
        for payload in spec['payloads'].values():
            file = _inside(root,payload['path'])
            require(file not in paths and payload['bytes']==4*spec['logical_numel'],
                    'aliased/padded/overlapping logical payloads')
            paths.add(file); loaded_bytes += payload['bytes']
    require(isinstance(metadata['shards'],list) and metadata['shards'], 'missing EMA shards')
    for shard in metadata['shards']:
        file = _inside(ema_path.parent,shard['file'])
        require(file not in paths and type(shard['storage_bytes']) is int and shard['storage_bytes']>0,
                'aliased or malformed EMA payload')
        # torch.save can serialize a large backing storage behind a small view;
        # bound actual file bytes before any weights_only deserialization too.
        paths.add(file); loaded_bytes += max(shard['storage_bytes'],file.stat().st_size)
    require(loaded_bytes<=budget['max_loaded_bytes'], 'CPU loaded-byte budget exceeded')
    proof = read_json(checked_pin(value['provenance']))
    require(isinstance(proof,dict) and set(proof)=={'evidence_kind','update_source','source','runtime','request',
        'policy','bases','native_assets','inputs','clocks'}, 'complete pinned warm-state provenance required')
    require(proof['evidence_kind'] in (READER_ONLY,ORIGINAL,CPU_WARM) and proof['update_source']==
            ('DECLARED_READER_GRADIENTS_NOT_W2' if proof['evidence_kind']==READER_ONLY else 'ORIGINAL_W2_HELPERS'),
            'explicit original W2 or STATE_READER_FIXTURE_ONLY evidence required; no W1 relabel')
    require(proof['source']==value['source'] and proof['runtime']==value['runtime']
            and proof['request']==value['request'] and fingerprint(proof['policy'])==fingerprint(native_policy())
            and proof['bases']=={r:value[r] for r in ('generator','critic')}
            and proof['native_assets']==native and proof['clocks']==dict(global_step=10,generator=2,critic=10,ema=10),
            'warm provenance source/base/native policy/clocks changed')
    require(isinstance(proof['inputs'],list) and proof['inputs'], 'explicit warm input evidence pins required')
    inputs = [checked_pin(pin) for pin in proof['inputs']]
    require(len(inputs)==len(set(inputs)), 'duplicate warm input evidence')
    if proof['evidence_kind']==CPU_WARM:
        _validate_cpu_warm_receipt(inputs,value,manifest)
    package, bootstrap = load_bootstrap(native,verify_payloads=True)
    require(package.manifest==manifest, 'logical manifest changed during native admission')
    return value,proof,package,bootstrap,metadata,ema_path.parent


def _validate_cpu_warm_receipt(inputs,value,manifest):
    """Admit a distinct test warm boundary, never invent historical step0 data.

    Pinned actual observer action/forward/backward/commit records and complete
    source documents bind the executed frozen bases, serialized moments,
    masters and EMA. This is a CPU test fixture, not provenance for an
    operator's pretrained Step10.
    """
    from distillschedule.runtime.helios_planner_cycle import _cycle_graph, _validate_action
    from distillschedule.search.helios_implementation import HeliosImplementation
    from .helios_inputs import HeliosCohortTask, source_fingerprint
    require(len(inputs)==1, 'actual original CPU warm receipt required')
    receipt = read_json(inputs[0])
    fields = {'schema_version','evidence_kind','source','runtime','request','framework_source','execution',
              'physical_initial','physical_final','cycles','bases','input_documents','prepared_inputs','logical_payloads','ema_manifest'}
    require(set(receipt)==fields and receipt['schema_version']==CPU_WARM_RECEIPT
            and receipt['evidence_kind']==CPU_WARM and receipt['source']==value['source']
            and receipt['runtime']==value['runtime'] and receipt['request']==value['request']
            and receipt['framework_source']==source_fingerprint()
            and receipt['execution']==dict(device='cpu',world_size=1,source_rank=0,source_world_size=16,
                historical_step0_inputs=False,pretrained_quality='NOT_RUN')
            and receipt['physical_initial']==dict(generator=0,critic=0,ema=0)
            and receipt['physical_final']==dict(generator=2,critic=10,ema=10)
            and isinstance(receipt['cycles'],list) and len(receipt['cycles'])==2,
            'actual original CPU warm receipt/source required')
    require(fingerprint(receipt['bases'])==fingerprint({r:value[r] for r in ('generator','critic')}),
            'actual original CPU warm receipt base binding changed')
    impl = HeliosImplementation('no-gan','world16-fsdp',512,1,False)
    from distillschedule.data import PreparedDataset
    from distillschedule.data.helios_nogan_reference_inputs import decode, descriptor
    prepared_path = checked_pin(receipt['prepared_inputs'])
    dataset = PreparedDataset(prepared_path)
    require(len(dataset)==2, 'actual original CPU warm receipt requires two complete prepared cohorts')
    for index,row in enumerate(receipt['cycles']):
        task = HeliosCohortTask(index)
        trace = decode(dataset.sample(index),task)
        require(row['descriptor']==descriptor(trace.reference_document)
                and row['rank_control_hashes']==[trace._control[r].trace_hash for r in range(16)],
                'actual original CPU warm receipt does not bind its original prepared inputs')
        initial = dict(generator=index,critic=5*index,ema=5*index)
        graph,order = _cycle_graph(impl,task.source_steps,initial)
        require(row['task']==task.to_dict() and row['source_steps']==list(task.source_steps)
                and row['dispatch']['status']=='PASS' and row['dispatch']['initial_versions']==initial
                and row['dispatch']['final_versions']==graph.metadata['final_versions']
                and row['dispatch']['executed_operations']==list(order)
                and [a['action_id'] for a in row['actions']]==list(order)
                and len(row['rank_control_hashes'])==16 and len(set(row['rank_control_hashes']))==16,
                'actual original CPU warm receipt action/rank/source clocks changed')
        operations = {op.operation_id:op for op in graph.operations}
        current = initial
        for action in row['actions']:
            _validate_action(action,impl)
            wanted = {**current,**operations[action['action_id']].produced_versions}
            require(action['versions_before']==current and action['versions_after']==wanted
                    and action['physical_versions_before']==current and action['physical_versions_after']==wanted,
                    'actual original CPU warm receipt native versions changed')
            current = wanted
    require(isinstance(receipt['input_documents'],list) and len(receipt['input_documents'])>=65,
            'actual original CPU warm receipt complete original input pins required')
    documents = [checked_pin(pin) for pin in receipt['input_documents']]
    pins = {pin['path']:pin for pin in receipt['input_documents']}
    expected = [dict(path=str(prepared_path),sha256=receipt['prepared_inputs']['sha256']),
                *dataset.manifest['provenance']['documents'],
                *[dict(path=str(prepared_path.parent/r['features']['path']),sha256=r['features']['sha256'])
                  for r in dataset.manifest['records']]]
    require(all(row['path'] in pins and row['sha256']==pins[row['path']]['sha256'] for row in expected),
            'actual original CPU warm receipt input document/payload coverage changed')
    require(len(documents)==len(set(documents)) and receipt['logical_payloads']=={
        role:{name:payload['sha256'] for name,payload in component['payloads'].items()}
        for role,component in manifest['components'].items()}, 'actual original CPU warm receipt logical payload binding changed')
    row = value['native_assets']['assets']['logical_checkpoint']
    require(checked_pin(receipt['ema_manifest'])==Path(row['path'])/row['ema_manifest']['identity_file']
            and receipt['ema_manifest']['sha256']==row['ema_manifest']['sha256'],
            'actual original CPU warm receipt EMA binding changed')


def _model(cls, value, role, remaining):
    from peft import LoraConfig
    from runtime.critic_pool_helios import _all_linear_targets, TRANSFORMER_ADDITIONAL_KWARGS
    model = load_base(cls,value[role],evidence_kind=value['base_evidence_kind'],
                      remaining_parameters=remaining,precision='WAN_NATIVE_MIXED_BF16')
    remaining -= sum(p.numel() for p in model.parameters())
    require(all(getattr(model.config,k)==v for k,v in TRANSFORMER_ADDITIONAL_KWARGS.items())
            and model.config.text_dim==4096 and model.config.in_channels==model.config.out_channels==16
            and tuple(model.config.patch_size)==(1,2,2)
            and not any(getattr(model.config,k) for k in ('is_use_gan','is_use_gan_hooks','is_use_gan_final')),
            'complete original native No-GAN architecture/conditioning/pyramid branch required')
    model.requires_grad_(False)
    targets, excludes = _all_linear_targets(model), ['down','up']
    patches = ['patch_short','patch_mid','patch_long','patch_embedding']
    require(all(hasattr(model,n) for n in patches), 'complete original pyramid patches required')
    if role=='generator':
        targets += patches[:3]
    else:
        targets = [n for n in targets if n not in patches and not n.startswith(('gan_heads.','gan_final_head.'))]
        excludes += sorted(patches)+['gan_heads','gan_final_head']
    bound = sum(256*(layer.weight[0].numel()+layer.weight.shape[0])
                for name,layer in model.named_modules() if name in targets)
    require(0 < bound <= remaining, 'native LoRA256 parameter budget exceeded')
    model.add_adapter(LoraConfig(r=256,lora_alpha=256.,lora_dropout=0.,init_lora_weights='gaussian',
        target_modules=sorted(set(targets)),exclude_modules=excludes))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)==bound,
            'native LoRA256 inventory/budget changed')
    model.enable_gradient_checkpointing(); model.train()
    return model,remaining-bound


def _stage_engine(model, package, role):
    import torch
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter, _logical_tensor
    from .native_state import OptimizerState
    spec = package.component(role)
    selected = {n:p for n,p in model.named_parameters() if p.requires_grad}
    require(list(selected)==spec['parameter_names'], 'original named LoRA256 trainable inventory/order changed: '+role)
    rate = POLICY[role+'_lr']
    optimizer = torch.optim.AdamW(list(selected.values()),lr=rate,betas=(0.,.999),eps=1e-8,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    step = 2 if role=='generator' else 10
    state, masters = optimizer.state_dict(), {}
    for index,(name,p) in enumerate(selected.items()):
        require(p.device.type=='cpu' and p.dtype==torch.float32
                and list(p.shape)==spec['records'][name]['shape'], 'logical named parameter shape/dtype changed: '+name)
        slots = {}
        for payload,key in (('master','master'),('first_moment','exp_avg'),('second_moment','exp_avg_sq')):
            tensor = _logical_tensor(package,role,payload,name).clone()
            require(tensor.dtype==torch.float32 and tuple(tensor.shape)==tuple(p.shape)
                    and bool(tensor.isfinite().all()), 'nonfinite or malformed logical state: '+role+'/'+payload+'/'+name)
            if key=='master':
                masters[name] = tensor
            else:
                require(key!='exp_avg_sq' or bool((tensor>=0).all()), 'negative Adam second moment')
                slots[key] = tensor
        # v3 stores a physical component step, not a separate per-tensor file.
        # This is precisely the native loader's singleton FP32 step conversion.
        slots['step'] = torch.tensor(float(spec['optimizer_step']),dtype=torch.float32,device='cpu')
        require(slots['step'].ndim==0 and slots['step'].item()==step, 'invalid original Adam step tensor')
        state['state'][index] = slots
    engine = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=step,global_batch=1,dp_world_size=1,gradient_clip=1.)
    view = OptimizerState(engine)
    named_state = view.state_dict(); named_state['optimizer'] = state
    view.validate_state_dict(named_state)
    current = scheduler.state_dict()
    expected = deepcopy(current)
    expected.update(last_epoch=step,_step_count=step+1,_last_lr=[rate])
    # _is_initial was added by newer Torch. Absence in a native older version's
    # otherwise complete state is lossless; arbitrary policy/transient keys are not.
    if '_is_initial' in expected:
        expected['_is_initial'] = False
    restored = spec['scheduler_state']
    require(isinstance(restored,dict) and set(restored)==set(expected)-({'_is_initial'} if '_is_initial' not in restored else set())
            and fingerprint(restored)==fingerprint({k:expected[k] for k in restored})
            and type(restored['last_epoch']) is int and type(restored['_step_count']) is int,
            'complete constant LambdaLR state/policy/counters changed: '+role)
    return engine,named_state,deepcopy(restored),masters


def _stage_ema(model, metadata, root, package):
    import torch
    from runtime.sparse_ema import SparseShardEMA
    require(type(metadata.get('format_version')) is int and metadata['format_version']==1
            and metadata.get('frozen_state_policy')=='reconstruct_from_base_model'
            and type(metadata.get('optimization_step')) is int and metadata['optimization_step']==10
            and metadata.get('decay')==POLICY['ema_decay']
            and type(metadata.get('update_after_step')) is int
            and metadata['update_after_step']==POLICY['ema_update_after_step']
            and isinstance(metadata.get('base_model'),str) and bool(metadata['base_model']),
            'original sparse EMA policy/clock changed')
    require(type(metadata.get('parameter_count')) is int
            and all(type(i) is int and i>=0 for i in metadata['tracked_indices']), 'invalid sparse EMA ownership indices')
    ema = SparseShardEMA.load_pretrained(model,root)
    require(ema.cur_decay_value==ema.get_decay(10) and type(ema.cur_decay_value) is float,
            'original EMA10 current decay changed')
    normalized = [n[7:] if n.startswith('module.') else n for n in ema.names]
    require(normalized==package.component('generator')['parameter_names'], 'sparse EMA named order changed')
    parameters = list(model.parameters())
    for shadow,index in zip(ema.shadows,ema.indices,strict=True):
        parameter = parameters[index]
        require(isinstance(shadow,torch.Tensor) and shadow.device.type=='cpu' and shadow.dtype==torch.float32
                and shadow.shape==parameter.shape and bool(shadow.isfinite().all()), 'nonfinite/malformed EMA shadow')
    by_name = dict(zip(ema.names,ema.shadows,strict=True))
    require(all(row['storage_bytes']==sum(by_name[n].numel()*by_name[n].element_size() for n in row['names'])
                for row in metadata['shards']), 'EMA shard storage coverage changed')
    # Same single-prefix projection admitted by the existing native loader.
    # Keep serialized names in immutable identity, normalize only the CPU view.
    ema.names = normalized
    return ema


def materialize_bootstrap(asset_manifest):
    """Return all actual owners only after complete decoded all-role admission.

    The operator supplies a new explicitly pinned CPU envelope referencing the
    *unchanged* six-role native asset document and LogicalCheckpointPackage v3.
    Native world16 sample clocks are provenance, not fictitious CPU reductions.
    """
    import torch
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from runtime.critic_pool_checkpoint import load_logical_master_into_trainables
    from .native_state import OptimizerState
    _cpu1()
    path = local_file(str(asset_manifest))
    asset_pin = dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path))
    value,proof,package,bootstrap,metadata,ema_root = _admit(path)
    cls, staged, engines = load_reference_model_class(backend='torch-sdpa'), {}, {}
    remaining = value['budget']['max_parameters']
    # Adapter initialization is subsequently overwritten by admitted masters;
    # it must not consume the caller's future training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for role in ('generator','critic'):
            model,remaining = _model(cls,value,role,remaining)
            staged[role] = _stage_engine(model,package,role)
            engines[role] = staged[role][0]
        ema = _stage_ema(engines['generator'].module,metadata,ema_root,package)
    # No master/optimizer/scheduler write above. Every decoded G/C/EMA owner is
    # admitted now. Recheck all pins before the first native master copy.
    checked_pin(asset_pin)
    checked_pin(value['provenance'])
    for pin in proof['inputs']:
        checked_pin(pin)
    for role in ('generator','critic'):
        for pin in [value[role]['config'],*value[role]['weights']]:
            checked_pin(pin)
    from .helios_reference_assets import verify_pins, load_bootstrap
    verify_pins(value['native_assets'])
    _, repeated = load_bootstrap(value['native_assets'],verify_payloads=True)
    require(repeated==bootstrap and value['source']==source_identity() and value['runtime']==runtime_identity(),
            'warm-state identity changed during admission')
    for role in ('generator','critic'):
        engine,optimizer_state,scheduler_state,masters = staged[role]
        before = {n:id(p) for n,p in engine.module.named_parameters()}
        load_logical_master_into_trainables(engine.module,package,component=role,verify_payload_sha256=True)
        require(before=={n:id(p) for n,p in engine.module.named_parameters()}
                and all(torch.equal(p.detach(),masters[n]) for n,p in engine.module.named_parameters() if p.requires_grad),
                'native master copy changed Parameter ownership or staged values')
        OptimizerState(engine).load_state_dict(optimizer_state)
        engine.lr_scheduler.load_state_dict(scheduler_state)
    identity = dict(schema_version=SCHEMA,scope=SCOPE,asset_manifest=asset_pin,assets=value,bootstrap=bootstrap,policy=native_policy(),
        evidence_kind=proof['evidence_kind'],original_w2_warm_updates_verified=False,
        cpu_test_warm_receipt_admitted=proof['evidence_kind']==CPU_WARM,
        provenance_claim=proof['update_source'],native_protocol_samples=dict(generator=32,critic=160),
        cpu_execution=dict(device='cpu',rank=0,world_size=1,batch=1,distributed_reduction=False),
        serialized_ema_names=metadata['tracked_names'],generation_export_lifecycle='NOT_IMPLEMENTED_BY_D1')
    result = WarmBootstrap(MappingProxyType(engines),ema,_freeze(deepcopy(identity)))
    for view in result.state_views().values():
        if hasattr(view,'validate_state_dict'):
            view.validate_state_dict(view.state_dict())
    _cpu1()
    return result
