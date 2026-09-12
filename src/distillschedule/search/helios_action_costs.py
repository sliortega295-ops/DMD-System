"""Reusable Helios native costs, not historical recipe times or NCCL discounts."""
from dataclasses import replace

from distillschedule.core.graph import Dependency, ExecutionGraph
from distillschedule.runtime.helios_planner_cycle import _cycle_graph
from distillschedule.workloads.best_known.registry import fingerprint, load_profile
from .contracts import require
from .cost_model import ComponentContract
from .helios_implementation import HeliosImplementation, NO_GAN, GAN

ADAPTER = 'helios_prepared_actions_v1'
RANKS = tuple(range(16))
PRECISION = fingerprint(dict(base_compute='BF16',lora_master='FP32',
    fsdp_unsharded_param_dtype='BF16',reduce_dtype='FP32',ema='cuda_reference',
    attention='pinned_local_fa2',scope='CANONICAL_NATIVE_FSDP_POLICY_NOT_WAN_FP32_POLICY'))
MEMORY = 'PEAK_MINUS_STATIC_MINUS_ENTRY_LATENTS; RETAINED_TRACE_AND_STATE_GROWTH_IN_WORKSPACE'


def anchors(workload):
    require(workload in (NO_GAN,GAN),'not a Helios workload')
    variant='no-gan' if workload==NO_GAN else 'gan'
    p=load_profile(workload)
    env=p['runtime']['environment']
    return (HeliosImplementation(variant,'world16-fsdp',512,5,False),
            HeliosImplementation(variant,p['topology']['placement'],
                int(env['DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS']),5,
                env['DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD']=='on'))


def action_graph(impl, *, selected=True):
    p=load_profile(impl.workload)['contract'];versions=dict(p['initial_versions'])
    ops,edges,ends=[],[],{}
    for cycle in (1,2):
        graph,order=_cycle_graph(impl,tuple(p['logical_steps']),versions)
        by_id={o.operation_id:o for o in graph.operations};prefix=f'cycle{cycle}.'
        for key in order:
            op=by_id[key]
            ops.append(replace(op,operation_id=prefix+key,metadata={**op.metadata,
                'inputs':tuple(prefix+x for x in op.metadata['inputs']),
                'action':key,'cycle':cycle}))
        edges.extend(replace(e,source=prefix+e.source,target=prefix+e.target) for e in graph.dependencies)
        if selected:
            edges.extend(Dependency(prefix+a,prefix+b,'selected_native_order') for a,b in zip(order,order[1:]))
        if cycle==2: edges.append(Dependency(ends['1'],prefix+'g','same_process_cycle_boundary'))
        ends[str(cycle)]=prefix+order[-1];versions=graph.metadata['final_versions']
    return ExecutionGraph(tuple(ops),tuple(edges),dict(adapter=ADAPTER,workload=impl.workload,
        initial_versions=p['initial_versions'],final_versions=versions,cycle_ends=ends,
        source_cohort_policy='fixed_source_cohort_replay',
        canonical_profile_sha256=fingerprint(load_profile(impl.workload)),
        scope='SERIAL_WORLD16_ACTIONS_INTERNAL_OVERLAP_IN_MEASURED_ENVELOPE'))


def observed_work(ranks):
    """All rank layouts/call inventories; sample values remain asset-bound."""
    maps=[{f'cycle{c["cycle"]}.{a["action_id"]}':a for c in r['cycles'] for a in c['actions']} for r in ranks]
    require(maps and all(set(m)==set(maps[0]) for m in maps),'rank action inventory differs')
    return {key:dict(physical_batch=1,logical_batch=16,rank_layouts=[dict(
        counts=m[key]['counts'],
        forwards=[{k:f[k] for k in ('role','grad_enabled','gan_mode','inputs','output')}
                  for f in m[key]['forwards'] if f['application_forward']],
        rollout_stages=[t['expected_stage_call_count'] for t in m[key]['trace_entries']],
        produced=[{k:t[k] for k in ('shape','dtype','bytes','requires_grad')} for t in m[key]['produced']])
        for m in maps]) for key in maps[0]}


def component_contract(op,impl,work):
    action=op.metadata['action'];kind=action[0]
    require(kind in 'grce' and work.get('physical_batch')==1 and work.get('logical_batch')==16,
            'Helios operation work missing')
    local={'family':kind,'variant':impl.variant}
    if kind in 'grc':local['modulation_chunk_tokens']=impl.chunk_tokens
    if kind=='r':local['persistent_generator_unshard']=impl.persistent_unshard
    return ComponentContract(f'helios.{impl.variant}.{action if action=="r_all" else kind}',work,
        {'native_precision_fingerprint':PRECISION},
        {'mesh':impl.mesh,'world_size':16,'placement':'FULL_SHARD','replicate':2 if impl.mesh=='node-hsdp' else 1},
        local,{'warm_state':'RESTORED_FIRST_CYCLE' if op.metadata['cycle']==1 else 'AFTER_ONE_NATIVE_CYCLE',
               'concurrency':'EXCLUSIVE_SERIAL_WORLD16',
               'scope':'ACQUIRE_FIVE_B1_RELEASE' if action=='r_all' else 'NATIVE_ACTION',
               'retained_latents':'MEASURED_LIFETIMES_NOT_AN_OVERLAP_DISCOUNT'})
