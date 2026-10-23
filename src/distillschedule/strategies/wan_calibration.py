"""Parameterized original actions using the existing synchronized CUDA observer."""
from __future__ import annotations
from dataclasses import replace
from distillschedule.core.graph import ExecutionGraph, Dependency
from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph,parameterized_window_order
from distillschedule.search.cost_model import ComponentContract
from distillschedule.search.wan_action_calibration import ActionCalibrator
from distillschedule.search.wan_implementation import PRECISION
from .wan_strategy import WanStrategy


def cycle_graph(task, initial_versions, window):
    graph=build_parameterized_cycle_graph(source_steps=task.source_steps,initial_versions=initial_versions,ema_after=task.ema_after)
    order=parameterized_window_order(graph,max_live_rollouts=window)
    by_id={o.operation_id:o for o in graph.operations}
    return ExecutionGraph(tuple(by_id[k] for k in order),graph.dependencies,graph.metadata)


def two_cycles(task, initial_versions, window):
    versions=dict(initial_versions);ops=[];edges=[];last=None;ends={}
    for cycle in (1,2):
        graph=cycle_graph(task,versions,window);prefix=f'cycle{cycle}.'
        ops.extend(replace(op,operation_id=prefix+op.operation_id,metadata={**op.metadata,
            'inputs':tuple(prefix+i for i in op.metadata['inputs']),'cycle':cycle,'action':op.operation_id}) for op in graph.operations)
        edges.extend(replace(e,source=prefix+e.source,target=prefix+e.target) for e in graph.dependencies)
        if last:edges.append(Dependency(last,prefix+'g','previous_cycle_commit'))
        last=prefix+graph.operations[-1].operation_id;ends[str(cycle)]=last
        versions=graph.metadata['final_versions']
    return ExecutionGraph(tuple(ops),tuple(edges),{'initial_versions':dict(initial_versions),'final_versions':versions,'cycle_ends':ends})


def component(task, action, strategy, *, cycle, work):
    kind=action[0]
    local={'family':kind}
    if strategy.candidate_bundle.enabled:
        local.update(strategy.candidate_bundle.component_identity())
    if kind in 'grc':local['chunk_tokens']=strategy.chunk_tokens
    if kind in 'gr':local['lora_nograd_epilogue']=strategy.lora_epilogue
    context={'warm_state':'cold' if cycle==1 else 'warm',
             'concurrency':'exclusive_world16','input_lifetime':'retained_until_C_return'}
    if strategy.compute_policy:
        # The persistent allocation context can affect every action, including
        # Generator/EMA. Do not relabel old component costs as state-qualified.
        context['state_policy'] = strategy.compute_policy
    return ComponentContract(family='original_wan.'+kind,
        work={'task_sha256':task.digest,'action':action,'trace_work':work[action]},
        precision={'execution_precision':PRECISION.digest},
        placement={'world_size':16,'placement':strategy.placement,
            'gradient_sync':'async128_tree' if strategy.placement=='resident' else 'fsdp_sum_divide16'},
        implementation=local,context=context)


def trace_work(task, trace):
    from runtime.static_world16_trace import _generator_keys,_critic_keys
    work={}
    for action,step,phase in [('g',task.first_step,'generator'),
                             *((f'r{i}',step,'critic') for i,step in enumerate(task.source_steps))]:
        keys=_generator_keys(step,gan=False) if phase=='generator' else _critic_keys(step,gan=False)
        layouts=[];control=[]
        for rank in range(task.world_size):
            t=trace._control[rank]
            layouts.append({name:{k:t.manifest['tensors'][key][k] for k in ('shape','dtype','stride')
                                  if k in t.manifest['tensors'][key]} for name,key in keys.items()})
            control.append(t.operation(f'step/{step}/{phase}/rollout')['control'])
        work[action]={'rank_layouts':layouts,'rollout_control':control,'output_shape':list(task.latent_shape)}
        if phase=='critic':work['c'+action[1:]]={'rank_layouts':layouts,'source_rollout':action,'score_forwards':1}
    for i in task.ema_after:work[f'e{i}']={'ema':'original_reference','lora_rank':task.lora_rank}
    return work


class OriginalWanCalibrator(ActionCalibrator):
    IMPLEMENTATION_TYPE=WanStrategy
    ADAPTER='original_wan_parameterized_actions_v1'
    SCHEMA='distillschedule.original_wan_rank_calibration.v1'

    def __init__(self, *, task, initial_versions, **kwargs):
        self.task,self.initial_versions=task,dict(initial_versions)
        super().__init__(**kwargs)

    def operation_graph(self):
        return two_cycles(self.task,self.initial_versions,self._implementation.rollout_window)

    def contract(self, operation):
        return component(self.task,operation.metadata['action'],self._implementation,
                         cycle=operation.metadata['cycle'],work=self.work).to_dict()
