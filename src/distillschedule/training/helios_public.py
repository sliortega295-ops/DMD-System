"""Original full-source Helios GAN in the existing public Program lifecycle.

CPU1 FP32 resident reference only. This module constructs declared owners and
FIVE nonreplayable calls, not a trainer, alternate loss, rollout, or solver.
Portable current-Generator export uses a separate inference-only projection.
"""
from __future__ import annotations
from copy import deepcopy

from .adapters import ModelComponents
from .helios_public_assets import (
    admit_assets, checked_pin, load_base, require, SCHEMA)


def build_model(parameters):
    """Strictly load explicit local bases; session alone seeds setup once."""
    import torch
    from peft import LoraConfig
    from safetensors.torch import load_file
    from diffusers.optimization import get_scheduler
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.data.helios_native_inputs import ROW_SCHEMA, PRECISION
    from runtime.sparse_ema import SparseShardEMA
    from types import SimpleNamespace
    from .helios_step_owner import HeliosStepOwner
    path, assets, args = admit_assets(parameters)
    c,m = args.training_config,args.model_config
    require(m.lora_layers is None and not m.train_norm_layers
            and list(m.lora_target_modules) and all('norm' not in n for n in m.lora_target_modules),
            'explicit original LoRA target list required; automatic/norm targeting is not admitted')
    cls = load_reference_model_class(backend='torch-sdpa')
    engines, remaining = {}, assets['budget']['max_parameters']
    for role in ('generator','critic'):
        model = load_base(cls,assets[role],evidence_kind=assets['evidence_kind'],remaining_parameters=remaining)
        remaining -= sum(p.numel() for p in model.parameters())
        require(model.config.text_dim==assets['geometry']['prompt_shape'][-1]
                and tuple(model.config.patch_size)==(1,2,2)
                and model.config.in_channels==model.config.out_channels==16,
                'conditioning/model native patch/channel geometry disagreement')
        require(bool(model.config.is_use_gan)==(role=='critic')
                and bool(model.config.is_use_gan_final)==(role=='critic'), 'original G/C GAN-head construction changed')
        model.requires_grad_(False)
        critic = role=='critic'
        prefix = 'critic_' if critic else ''
        exclude = list(m.lora_exclude_modules)
        targets = list(m.lora_target_modules)
        if critic:
            patch = ['patch_short','patch_mid','patch_long','patch_embedding']
            targets = [name for name in targets if name not in patch]
            exclude += patch + ['gan_heads','gan_final_head']
        rank = getattr(m,prefix+'lora_rank')
        require(type(rank) is int and rank>0, 'positive explicit LoRA rank required')
        adapter_bound = sum(rank*(layer.in_features+layer.out_features) for name,layer in model.named_modules()
                            if isinstance(layer,torch.nn.Linear) and any(name.endswith(t) for t in targets))
        require(adapter_bound>0 and adapter_bound<=remaining, 'LoRA construction exceeds explicit CPU parameter budget')
        remaining -= adapter_bound
        model.add_adapter(LoraConfig(r=rank,lora_alpha=getattr(m,prefix+'lora_alpha'),
            lora_dropout=getattr(m,prefix+'lora_dropout'),init_lora_weights='gaussian',
            target_modules=targets,exclude_modules=exclude))
        if critic:
            for name,p in model.named_parameters():
                if any(module in name for module in ('gan_heads','gan_final_head')):
                    p.requires_grad_(True)
        if c.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        rate = c.critic_learning_rate if critic else c.learning_rate
        optimizer = native_primitives().base.get_optimizer(args,SimpleNamespace(print=lambda *a,**kw:None),
            [{'params':[p for p in model.parameters() if p.requires_grad], 'lr':rate}],use_deepspeed=False)
        scheduler = get_scheduler(c.lr_scheduler,optimizer=optimizer,num_warmup_steps=c.lr_warmup_steps,
            num_training_steps=c.max_train_steps,num_cycles=c.lr_num_cycles,power=c.lr_power)
        engines[role] = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=0,global_batch=1,dp_world_size=1,
            gradient_clip=c.max_grad_norm_critic if critic else c.max_grad_norm)
    g,critic = engines['generator'],engines['critic']
    # Source-equivalent name binding, before GAN can disable the base mask.
    trainable = {name for name,p in critic.module.named_parameters() if p.requires_grad}
    extra = {name for name in trainable if any(module in name for module in ('gan_heads','gan_final_head'))}
    base = trainable - extra
    frozen = {name for name,p in critic.module.named_parameters() if not p.requires_grad}
    ema = SparseShardEMA(g.module.named_parameters(),decay=c.ema_decay,update_after_step=c.ema_start_step,
                        base_model=assets['generator']['weights'][0]['sha256'])
    def noise_scheduler():
        return native_primitives().scheduler(shift=c.stage2_timestep_shift,stages=c.stage2_num_stages,
            stage_range=list(c.stage2_stage_range),gamma=c.stage2_scheduler_gamma)
    negative = load_file(checked_pin(assets['negative']),device='cpu')['negative_prompt_embeds']
    require(negative.isfinite().all().item(), 'nonfinite negative conditioning')
    stream = dict(schema_version='distillschedule.helios_input_stream.v2',row_schema=ROW_SCHEMA,
        precision=PRECISION,capacity=c.max_train_steps,cohort_policy='FINITE_DISTINCT_NATIVE_SOURCES')
    owner = HeliosStepOwner(generator=g,critic=critic,ema=ema,args=args,
        generator_scheduler=noise_scheduler(),critic_scheduler=noise_scheduler(),
        negative_prompt_embeds=negative,noise_shape=assets['geometry']['noise_shape'],input_stream=stream,
        gan_base_critic_trainable_params=base,gan_extra_critic_trainable_params=extra,
        frozen_parameter_names=frozen,backend='torch-sdpa')
    from distillschedule.workloads.best_known.registry import file_sha256
    metadata = dict(schema_version=SCHEMA,scope='CPU1_FP32_ORIGINAL_GAN_REFERENCE_NOT_WORLD16',
        assets=deepcopy(assets),asset_manifest=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path)),
        owner_identity=deepcopy(owner.identity),
        evidence_kind=assets['evidence_kind'],quality_validation='NOT_RUN',claimable_speedup=False,
        portable_generator_export='CURRENT_FULL_UNMERGED_NATIVE_LATENT_DIAGNOSTIC')
    from .helios_inference import project_current_generator
    return ModelComponents(state={'native_step':owner},metadata=metadata,admit_training=admit_training,
                           collect_metrics=collect_metrics, inference_projection=project_current_generator).validate()


def collect_metrics(outputs):
    """Read losses already returned by the five actual original native entries."""
    names = [f'helios.native_step.{i}' for i in range(5)]
    require(set(outputs) == set(names), 'complete five-step output receipt required for metrics')
    rows = [outputs[name] for name in names]
    generator = [row['generator_loss'] for row in rows if row['train_generator']]
    require(len(generator) == 1, 'one actual Generator update required for cycle metrics')
    critic = [row['critic_loss'] for row in rows]
    return {'generator/loss': generator[0], 'critic/mean_loss': sum(critic) / 5,
            **{f'critic/loss_{i+1}': value for i, value in enumerate(critic)}}


def _row_schema(owner):
    """Static tensor-tree schema; validate every prepared record before output."""
    def tensor(shape,dtype='torch.float32'):
        return dict(kind='tensor',shape=list(shape),dtype=dtype)
    def tree(items):
        return dict(kind='dict',items=items)
    noise = owner.noise_shape
    history = tree({name:tensor((*noise[:2],frames,*noise[-2:])) for name,frames in
        (('gt_history_latents',19),('gt_target_latents',noise[2]),('gt_x0_latents',1))})
    inputs = tree(dict(prompt_embeds=tensor(owner.negative.shape),gan_prompt_embeds=tensor(owner.negative.shape),
                       gan_vae_latents=tensor(noise),generator_history=history,critic_history=history))
    return tree(dict(identity=tensor([32],'torch.uint8'),absolute_step=tensor([1],'torch.int64'),
        source_sha256=tensor([32],'torch.uint8'),content_sha256=tensor([32],'torch.uint8'),inputs=inputs))


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    """Generic callback seam; no dataset captured in a factory or native owner."""
    from distillschedule.data.helios_native_inputs import SCHEMA as INPUTS, ROW_SCHEMA, PRECISION, producer_identity
    owner = state['native_step']
    owner._validate_live()
    execution = dict(request.execution)
    require(set(execution)<= {'adapter','mode'} and execution.get('adapter','program')=='program'
            and execution.get('mode','reference')=='reference',
            'public Helios stream-v2 requires reference mode; auto/manual/plan/order are unsupported')
    require(request.trainer.get('device','cpu')=='cpu' and not request.trainer.get('distributed')
            and not request.trainer.get('validation') and request.data.get('sharding','single')=='single',
            'CPU1 resident reference only; distributed/validation not admitted')
    require(request.data.get('sampling',{'kind':'finite','shuffle':False})=={'kind':'finite','shuffle':False},
            'finite sequential sampling only; no epochs/shuffle')
    require(request.algorithm['factory']=='distillschedule.training.helios_public:build_algorithm'
            and not request.algorithm.get('parameters'), 'original atomic algorithm with no override parameters required')
    _,assets,_ = admit_assets(request.model.get('parameters',{}))
    meta = prepared_manifest['metadata']
    require(meta.get('schema_version')==INPUTS and meta.get('row_schema')==ROW_SCHEMA
            and meta.get('precision')==PRECISION and meta.get('producer')==producer_identity()
            and meta.get('args')==assets['args'] and meta.get('geometry')==assets['geometry']
            and meta.get('evidence_kind')==assets['evidence_kind'], 'prepared input/Args/source/geometry/evidence mismatch')
    from distillschedule.data.manifest import validate_source
    for pin in prepared_manifest['provenance']['documents']:
        validate_source(pin)
    records = prepared_manifest['records']
    require(type(meta.get('native_steps')) is int and meta['native_steps']==5*len(records)
            and meta['native_steps']<=owner.capacity and request.trainer['cycles']<=len(records),
            'prepared inventory/target exceeds fixed native capacity')
    require(isinstance(prepared_identity,str) and len(prepared_identity)==64, 'common prepared identity required')
    expected_schema = dict(kind='dict',items={'native_rows':dict(kind='list',items=[_row_schema(owner)]*5)})
    ids,paths,hashes,contents,source_contents = set(),set(),set(),set(),set()
    for cycle,record in enumerate(records):
        require(record['features']['tensors']==expected_schema, 'prepared native tensor layout changed')
        rowmeta = record.get('metadata',{})
        require(rowmeta.get('cycle')==cycle and isinstance(rowmeta.get('native_rows'),list)
                and len(rowmeta['native_rows'])==5, 'prepared native cycle metadata/order changed')
        for step,row in enumerate(rowmeta['native_rows'],start=5*cycle):
            source = checked_pin(row['source'])
            require(row.get('absolute_step')==step and row.get('identity') not in ids
                    and str(source) not in paths and row['source']['sha256'] not in hashes
                    and row.get('content_sha256') not in contents and row.get('raw_content_sha256') not in source_contents,
                    'prepared source reused/reordered; new ID/seed is not fresh data')
            ids.add(row['identity']);paths.add(str(source));hashes.add(row['source']['sha256'])
            contents.add(row['content_sha256']);source_contents.add(row['raw_content_sha256'])


def native_step(ctx):
    """Module-level nonreplayable binding; all mutable native owners are state."""
    owner = ctx.state['native_step']
    index = ctx.attributes['step_in_cycle']
    absolute = 5*ctx.attributes['cycle']+index
    require(owner.global_step==absolute, 'atomic native step version/order drift')
    from distillschedule.data.helios_native_inputs import decode_row
    row = decode_row(ctx.inputs['native_rows'][index],absolute_step=absolute)
    return owner.step_input(row,absolute_step=absolute)


def build_algorithm(components, feeds, parameters, cycle, seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program, Call
    from distillschedule.data.helios_native_inputs import decode_row
    require(not parameters and type(cycle) is int and cycle>=0 and set(components.state)=={'native_step'},
            'original one-owner atomic algorithm required')
    require(isinstance(feeds,dict) and set(feeds)=={'native_rows'} and isinstance(feeds['native_rows'],list)
            and len(feeds['native_rows'])==5, 'one complete five-row native record required')
    owner = components.state['native_step']
    checked_pin(components.metadata['asset_manifest'])
    assets = components.metadata['assets']
    for pin in [assets['args'],assets['negative'],*[
            pin for role in ('generator','critic') for pin in [assets[role]['config'],*assets[role]['weights']]]]:
        checked_pin(pin)
    rows = [decode_row(row,absolute_step=5*cycle+i) for i,row in enumerate(feeds['native_rows'])]
    for row in rows:
        owner._validate_inputs([row])
    from distillschedule.workloads.best_known.registry import fingerprint
    from .helios_step_owner import _content
    contents = [fingerprint(_content({key:value for key,value in row.items() if key!='identity'})) for row in rows]
    consumed_ids = {row['identity'] for row in owner._consumed}
    consumed_contents = {row['content_sha256'] for row in owner._consumed}
    require(len({row['identity'] for row in rows})==5 and len(set(contents))==5
            and not {row['identity'] for row in rows} & consumed_ids
            and not set(contents) & consumed_contents and 5*(cycle+1)<=owner.capacity,
            'native source IDs/content/capacity reused or changed')
    calls = tuple(Call(id=f'helios.native_step.{i}',binding='original_native_step',inputs=('native_rows',),
        reads=('native_step',),writes=('native_step',),after=((f'helios.native_step.{i-1}',) if i else ()),
        grad='external_grad',replay_safe=False,attributes=dict(cycle=cycle,step_in_cycle=i)) for i in range(5))
    return Workload(program=Program(calls,initial_versions={'native_step':5*cycle},feeds=('native_rows',),
        outputs=tuple(c.id for c in calls)),bindings={'original_native_step':native_step},state=components.state,
        feeds=feeds,seed=seed,identity_extra={'scope':'ORIGINAL_HELIOS_CPU1_REFERENCE_NOT_QUALITY'})
