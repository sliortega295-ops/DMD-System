"""Original Wan task -> measured components -> existing hierarchical planner.

All native actions reserve world16; internal async gradient reduction is inside
measured envelopes, not a fictitious extra pool. Changing a window changes
retained memory, not the calibrated sum of otherwise identical serial work.
"""
from dataclasses import replace
from typing import Mapping
from distillschedule.core.graph import Dependency, ExecutionGraph
from distillschedule.search.contracts import require
from distillschedule.search.cost_model import ComponentContract, CostModel
from distillschedule.search.plan_ir import BindingRule, Lifetime, StateAllocation, PlanningProblem
from distillschedule.search.hierarchical import search_hierarchical
from distillschedule.search.simulator import simulate
from distillschedule.workloads.best_known.registry import fingerprint
from .wan import OriginalWanTask, OriginalWanStrategy, PRECISION, candidates

RANKS=tuple(range(16))
PLAN_SCHEMA='distillschedule.original_wan_selected_plan.v1'


def action_graph(task, initial_versions, window=None):
    versions=dict(initial_versions); operations=[]; edges=[]; ends={}; previous=None
    for cycle in (1,2):
        graph=task.graph(versions)
        order=(task.graph(versions).topological_order() if window is None else
               OriginalWanStrategy(rollout_window=window).order(task,versions))
        prefix=f'cycle{cycle}.'
        by_id={o.operation_id:o for o in graph.operations}
        for key in order:
            op=by_id[key]
            operations.append(replace(op,operation_id=prefix+key,metadata={**op.metadata,
                'cycle':cycle,'action':key,'inputs':tuple(prefix+x for x in op.metadata.get('inputs',()))}))
        edges.extend(replace(e,source=prefix+e.source,target=prefix+e.target) for e in graph.dependencies)
        if window is not None:
            edges.extend(Dependency(prefix+a,prefix+b,'selected_native_order') for a,b in zip(order,order[1:]))
        if previous: edges.append(Dependency(previous,prefix+'g','complete_cycle_boundary'))
        last=f'e{task.fake_updates-1}' if task.fake_updates-1 in task.ema_after else f'c{task.fake_updates-1}'
        previous=prefix+last; ends[str(cycle)]=previous
        versions=graph.metadata['final_versions']
    return ExecutionGraph(tuple(operations),tuple(edges),dict(initial_versions=dict(initial_versions),
           final_versions=versions,cycle_ends=ends,task_sha256=task.digest,
           protocol=task.protocol,native_work='ORIGINAL_P0_HELPERS',inter_role_overlap=False))


def component_contract(task, operation, implementation):
    """Occurrence-specific first version: safe reuse across windows, not tasks."""
    impl=implementation.implementation
    return ComponentContract(family='original_wan.'+operation.metadata['action'][0],
        work={'task_sha256':task.digest,'action':operation.metadata['action'],
              'source_step':operation.logical_step,'latent_shape':list(task.latent_shape),
              'stage2_steps':list(task.stage2_steps),'physical_batch':1,'logical_batch':16},
        precision={'execution':PRECISION.digest},
        placement={'placement':impl.placement,'world_size':16,'gradient':'FP32_SUM_POST_DIVIDE_16'},
        implementation=impl.assignment,
        context={'cycle':operation.metadata['cycle'],'state':'original_bootstrap_then_consecutive_cycles',
                 'resource':'exclusive_world16_with_internal_collectives',
                 'retained_storage':'measured_separately_from_action_workspace'})


def assignment(strategy):
    return {**strategy.implementation.assignment,'rollout_window':str(strategy.rollout_window)}


def strategy_from_assignment(value):
    require(set(value)=={'placement','chunk_tokens','lora_epilogue','rollout_window'},'unknown/missing original strategy decision')
    require(value['chunk_tokens'] in {'512','1024','2048'} and value['lora_epilogue'] in {'on','off'},'invalid original kernel selection')
    require(value['rollout_window'] in {str(i) for i in range(1,65)},'invalid original window')
    return OriginalWanStrategy(value['placement'],int(value['chunk_tokens']),value['lora_epilogue']=='on',int(value['rollout_window']))


def build_problem(task, costs: CostModel, *, initial_versions, variant_memory: Mapping,
                  output_bytes: Mapping, memory_limits: Mapping):
    require(costs.binding['semantic_fingerprint']==task.digest,'costs belong to another task')
    graph=action_graph(task,initial_versions)
    require(set(memory_limits)==set(RANKS),'all sixteen capacity limits required')
    require(set(output_bytes)=={o.operation_id for o in graph.operations if o.metadata['action'].startswith('r')},
            'actual retained output inventory required')
    native_impls={s.implementation.id:s for s in candidates(task) if s.rollout_window==1}
    require(set(variant_memory)<=set(native_impls) and variant_memory,'invalid/empty measured implementation storage')
    states=[];rules=[];forbidden=[{'placement':'fsdp','chunk_tokens':'1024'},
       {'placement':'fsdp','chunk_tokens':'2048'},{'placement':'fsdp','lora_epilogue':'on'}]
    for key,s in native_impls.items():
        when=s.implementation.assignment
        if key not in variant_memory:
            forbidden.append(when)
            continue
        size=variant_memory[key]
        require(type(size) is int and size>0,'invalid measured physical storage')
        states.append(StateAllocation(key,RANKS,size,('native_runtime',),False,
                                      fingerprint({'task':task.digest,'implementation':key})))
        for op in graph.operations:
            contract=component_contract(task,op,s)
            rules.append(BindingRule(op.operation_id,when,contract.digest,RANKS,
                         tuple(f'gpu:{r}' for r in RANKS),(key,),contract.digest))
    lives=tuple(Lifetime('retained:'+k,k,(k.replace('.r','.c'),),RANKS,size,'producer_exit')
                for k,size in output_bytes.items())
    conditional=[]
    for w in range(1,task.fake_updates+1):
        order=OriginalWanStrategy(rollout_window=w).order(task,initial_versions)
        for cycle in (1,2):
            conditional.extend({'when':{'rollout_window':str(w)},'source':f'cycle{cycle}.{a}',
                                'target':f'cycle{cycle}.{b}'} for a,b in zip(order,order[1:]))
    return PlanningProblem(workload='wan13b_original_parameterized',binding=costs.binding,graph=graph,
        decisions={'placement':('fsdp','resident'),'chunk_tokens':('512','1024','2048'),
                   'lora_epilogue':('off','on'),'rollout_window':tuple(str(i) for i in range(1,task.fake_updates+1))},
        rules=tuple(rules),states=tuple(states),lifetimes=lives,memory_limits=dict(memory_limits),
        forbidden=tuple(forbidden),conditional_dependencies=tuple(conditional),seeds=(),
        evidence_scope=costs.scope,lowering={'adapter':'original_wan_parameterized','task':task.to_dict()})


def select_plan(task, costs, problem, *, max_proposals=10000, timeout_seconds=60, beam_width=0):
    require(problem.lowering.get('task')==task.to_dict(),'problem task differs')
    costs.verify_receipts()
    result=search_hierarchical(problem,costs,max_proposals=max_proposals,timeout_seconds=timeout_seconds,
                               beam_width=beam_width,structural_decisions=('placement',))
    require(result['predicted_best'] is not None,'no calibrated feasible original Wan plan')
    # Several serial orders tie in this envelope model. Prefer smaller retained memory
    # among evaluated equal-time plans; do not manufacture a window speedup.
    best=min(result['ranked'],key=lambda r:(r['predicted_cycle2_ns'],r['predicted_total_ns'],
                    max(m['peak_bytes'] for m in r['memory']),int(r['assignment']['rollout_window'])))
    strategy=strategy_from_assignment(best['assignment']).validate(task)
    document={'schema_version':PLAN_SCHEMA,'task':task.to_dict(),'strategy':strategy.to_dict(),
        'binding':dict(costs.binding),'problem_sha256':problem.digest,'cost_model_sha256':costs.digest,
        'predicted':best,'search_diagnostics':result['search_diagnostics'],
        'lowered_order':list(strategy.order(task,problem.graph.metadata['initial_versions'])),
        'environment':strategy.environment(task),'native_placement':strategy.implementation.native_placement,
        'gpu_execution':'NOT_RUN_BY_SEARCH','claimable_speedup':False,
        'evidence_scope':costs.scope,'protocol_substitution':False}
    document['plan_sha256']=fingerprint(document)
    return document


def validate_plan(document, task, problem, costs):
    unsigned=dict(document); sha=unsigned.pop('plan_sha256',None)
    require(document.get('schema_version')==PLAN_SCHEMA and sha==fingerprint(unsigned),'selected plan changed')
    require(document['task']==task.to_dict() and document['problem_sha256']==problem.digest and
            document['cost_model_sha256']==costs.digest and document['binding']==dict(costs.binding),
            'stale task/problem/cost binding')
    require(document['evidence_scope']==costs.scope,'evidence scope changed')
    s=OriginalWanStrategy(**document['strategy']).validate(task)
    expected=simulate(problem,assignment(s),costs)
    require(expected==document['predicted'],'selected plan does not reconcile with measured components')
    require(document['lowered_order']==list(s.order(task,problem.graph.metadata['initial_versions'])) and
            document['environment']==s.environment(task) and document['native_placement']==s.implementation.native_placement,
            'native dispatch differs from selected strategy')
    return s
