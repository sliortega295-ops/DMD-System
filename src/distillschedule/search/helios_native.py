"""Native Helios evidence gates and two-cycle normalization, never a train loop."""
from pathlib import Path
import copy
import json
import math

from distillschedule.workloads.best_known.registry import (
    _object, fingerprint, file_sha256, write_new_json, repository_root, git_head, load_profile,
)
from .contracts import require, finite_number, MEASUREMENT_SCHEMA, validate_measurement
from .helios_implementation import HeliosImplementation, current_implementation, MODE_ENV
from .helios_request import TRACE_KEY, TRACE_HASH_KEY

SUMMARY_SCHEMA='distillschedule.helios_native_two_cycle_summary.v1'
COMPARISON_SCHEMA='distillschedule.helios_native_two_cycle_comparison.v1'
GATE_SCHEMA='distillschedule.helios_native_admission.v1'


def read_ranks(out):
    paths=[Path(out)/'execution/ranks'/f'rank_{r:02d}.json' for r in range(16)]
    return [_object(p) for p in paths],[ref(p) for p in paths]


def ref(path):
    p=Path(path);return dict(path=str(p.resolve()),sha256=file_sha256(p))


def bound(reference):
    p=Path(reference['path'])
    require(p.is_file() and file_sha256(p)==reference['sha256'],'source receipt hash changed/missing')
    return _object(p)


def summarize(rows,*,expected_head,candidate_id):
    from distillschedule.runtime.helios_planner_cycle import SCHEMA,_validate_action,_cycle_graph
    impl=HeliosImplementation.parse(candidate_id);contract=impl.lower_profile()['contract']
    require(len(rows)==16 and {r.get('rank') for r in rows}==set(range(16)),'Helios needs 16/16 ranks')
    require(len({r.get('run_id') for r in rows})==1,'rank run IDs differ')
    mode=rows[0]['mode']
    mapping=impl.version_mapping(contract)
    def physical(logical):
        return {key: value-mapping['logical_minus_physical'][key] for key,value in logical.items()}
    for r in rows:
        require(r.get('schema_version')==SCHEMA and r.get('status')=='PASS'
                and r.get('source_head')==expected_head and r.get('candidate_id')==impl.id
                and r.get('mode')==mode and r.get('world_size')==16
                and r.get('profiled') is False and r.get('profiler_status')=='DISABLED'
                and r.get('same_process') is True and r.get('initializations')==1,
                'Helios native source/mode/rank contract failed')
        require(r.get('planner_binding',{}).get('planner_implementation')==impl.assignment,'effective implementation differs')
        if impl.retained_critic_blocks:
            retention = r.get('checkpoint_block_retention', {})
            g, c = retention.get('generator', {}), retention.get('critic', {})
            require(g.get('enabled') is False and c.get('enabled') is True
                    and c.get('block_indices') == list(impl.retained_critic_blocks)
                    and c.get('retained_forward_calls') == 10*len(impl.retained_critic_blocks)
                    and all(c.get('block_forward_counts', {}).get(str(i)) == 10
                            for i in impl.retained_critic_blocks)
                    and c.get('parameter_dtype_changed') is False and c.get('weights_merged') is False,
                    'native retained-block source/census/effects differ')
        if impl.request is not None:
            require(r['planner_binding'].get(TRACE_KEY)==impl.request.to_dict()
                    and r['planner_binding'].get(TRACE_HASH_KEY)==impl.request.digest,
                    'effective Helios request binding differs')
            actual_hash=r['planner_binding'].get('expected_trace_hash')
            grid=r.get('identities',{}).get('original_helios_native_grid',{})
            require(actual_hash==r.get('identities',{}).get('trace_bundle_hash')
                    and isinstance(actual_hash,str) and len(actual_hash)==64
                    and actual_hash==rows[0]['planner_binding'].get('expected_trace_hash')
                    and grid.get('status')=='PASS' and grid.get('request_sha256')==impl.request.digest
                    and grid.get('operation_count')==6,
                    'Helios exact launch trace/native grid preflight binding differs')
        if impl.candidate_bundle.enabled:
            source = impl.candidate_bundle.cost_identity()
            binding = r['planner_binding'].get('candidate_bundle', {})
            require(binding == source, 'safe No-GAN effective flags/source binding differs')
            effects = r.get('candidate_bundle_effects', {})
            actual = impl.candidate_bundle.executed_effects(r.get('build', {}), effects.get('before', {}))
            require(effects == actual, 'safe No-GAN effects differ from actual build/call receipts')
        require(r['source_steps']==contract['logical_steps'] and r['source_cohort_policy']=='fixed_source_cohort_replay',
                'Helios fixed source cohort changed')
        require(r['initial_versions']==contract['initial_versions'] and r['final_versions']==contract['final_versions'],
                'Helios final/initial versions differ')
        require(r.get('version_mapping')==mapping
                and r.get('initial_physical_versions')==physical(r['initial_versions'])
                and r.get('final_physical_versions')==physical(r['final_versions']),
                'Helios physical versus logical version mapping differs')
        require(r['final']['generator_physical_step']==r['final_physical_versions']['generator']
                and r['final']['critic_physical_step']==r['final_physical_versions']['critic']
                and r['final']['ema_step']==r['final_physical_versions']['ema'],
                'Helios physical final engine counters differ')
        require(r['final']['generator_logical_version']==r['final_versions']['generator']
                and r['final']['critic_logical_version']==r['final_versions']['critic'],
                'Helios logical final engine labels differ')
        require(len(r['cycles'])==2,'Helios requires two cycles')
        previous=contract['initial_versions']
        bundle_previous = r['candidate_bundle_effects']['before'] if impl.candidate_bundle.enabled else None
        for index,c in enumerate(r['cycles'],1):
            if impl.candidate_bundle.enabled:
                effect=c.get('candidate_bundle_effects',{})
                require(effect.get('before')==bundle_previous,'safe No-GAN cycle counter chain differs')
                require(effect==impl.candidate_bundle.effects_between(effect['before'],effect.get('after',{})),
                        'safe No-GAN cycle effects differ')
                bundle_previous=effect['after']
            require(c['cycle']==index and c['status']=='PASS' and c['measured']==(index==2)
                    and c['initial_versions']==previous,'Helios cycle sequence/status differs')
            require(c.get('initial_physical_versions')==physical(c['initial_versions'])
                    and c.get('final_physical_versions')==physical(c['final_versions']),
                    'Helios cycle physical step census differs')
            finite_number(c['wall_seconds'],'cycle wall',positive=True)
            graph,order=_cycle_graph(impl,tuple(contract['logical_steps']),previous)
            require(fingerprint(c['graph'])==fingerprint(graph.to_dict()) and c['selected_order']==list(order),
                    'executed native graph differs from generated order')
            require([a['action_id'] for a in c['actions']]==list(order),'missing/extra native actions')
            by_id={o.operation_id:o for o in graph.operations};versions=dict(previous)
            for a in c['actions']:
                require(a['source_rank']==r['rank'] and all(t['source_rank']==r['rank'] for t in a['trace_entries']),
                        'rank-qualified trace ownership differs')
                _validate_action(a,impl)
                require(a['versions_before']==versions,'action version dependency differs')
                versions.update(by_id[a['action_id']].produced_versions)
                require(a['versions_after']==versions,'action commit versions differ')
                require(a.get('physical_versions_before')==physical(a['versions_before'])
                        and a.get('physical_versions_after')==physical(a['versions_after']),
                        'Helios action physical step census differs')
                require(all(math.isfinite(l['value']) for l in a['losses']),'non-finite native loss')
                require(len(a['losses'])==a['counts']['generator_backward']+a['counts']['critic_backward'],
                        'backward loss observations incomplete')
                if mode=='calibration':finite_number(a['calibration_seconds'],'action calibration',positive=True)
                else:require(a['calibration_seconds'] is None,'calibration contaminated native timing')
                m=a['memory_after'];require(0<=m['allocated_bytes']<=m['reserved_bytes']
                    and m['max_allocated_bytes']<=m['max_reserved_bytes'],'invalid allocator observation')
            require(versions==c['final_versions']==graph.metadata['final_versions'],'cycle version census differs')
            require(len(c['retained_lifetimes'])==5 and c['peak_live_rollouts']==impl.rollout_window,
                    'native retained rollout window differs')
            for life in c['retained_lifetimes']:
                require(0<=life['produced_after_action']<life['released_after_action']<len(order),
                        'rollout lifetime is incomplete')
                consumer=c['actions'][life['released_after_action']]
                require(consumer['action_id'].startswith('c') and consumer['source_steps']==[life['source_step']],
                        'rollout released at wrong native consumer')
            require(all(c['semantic_validation'].get(k)=='PASS' for k in
                        ('census','versions','finite','dtype','sample_rng')),'native semantic gate failed')
            previous=versions
        require(r['input_reuse']['critic_entries_still_cached']==0,'trace cache not drained')
        if impl.candidate_bundle.enabled:
            require(bundle_previous==r['candidate_bundle_effects']['after'],
                    'safe No-GAN final cycle counters differ from actual build')
    return dict(schema_version=SUMMARY_SCHEMA,status='PASS',rank_count=16,source_head=expected_head,
        run_id=rows[0].get('run_id'),candidate_id=impl.id,mode=mode,profiled=False,
        initial_versions=contract['initial_versions'],final_versions=contract['final_versions'],
        version_mapping=mapping,initial_physical_versions=physical(contract['initial_versions']),
        final_physical_versions=physical(contract['final_versions']),
        cycle_wall_seconds={str(i):max(r['cycles'][i-1]['wall_seconds'] for r in rows) for i in (1,2)},
        timer_scope='MAX_RANK_LOCAL_CUDA_FENCED_CYCLE_WALL_OUTSIDE_PROBES',
        source_cohort_policy='fixed_source_cohort_replay',calibration_only=mode=='calibration',
        formal_timing_eligible=False,numerical_status='NOT_RUN',errors=[])


def checked(receipt):
    from .native import checked_summary
    out,summary=checked_summary(receipt);rows,refs=read_ranks(out)
    actual=summarize(rows,expected_head=receipt['integration_head'],candidate_id=receipt['search_candidate']['id'])
    impl=HeliosImplementation.parse(receipt['search_candidate']['id'])
    if impl.request is not None:
        trace=receipt.get('original_helios_trace_binding',{})
        require(trace.get('request_sha256')==impl.request.digest
                and all(r['planner_binding'].get('expected_trace_hash')==trace.get('bundle_hash') for r in rows),
                'Helios native ranks do not match exact asset-bound launch trace')
    require(all(summary.get(k)==v for k,v in actual.items()) and summary.get('native_evidence')==refs,
            'Helios summary differs from raw ranks')
    return out,summary,rows,refs


def compare_rows(left,right,*,head):
    from distillschedule.evidence.trajectory_state_sketch import compare_numeric_state_sketches
    from distillschedule.evidence.trajectory_coarse import COARSE_50C_THRESHOLDS
    left=sorted(left,key=lambda r:r['rank']);right=sorted(right,key=lambda r:r['rank'])
    reports=[];losses=[]
    require(HeliosImplementation.parse(left[0]['candidate_id']).request==
            HeliosImplementation.parse(right[0]['candidate_id']).request,
            'Helios comparison cannot mix request cohorts')
    for l,r in zip(left,right,strict=True):
        require(l['mode']==r['mode']=='correctness','sampled parity requires correctness runs')
        require(l['source_steps']==r['source_steps'] and l['initial_versions']==r['initial_versions'],
                'comparison source cohort/versions differ')
        initial=compare_numeric_state_sketches(l['outside_timer_probes'][0]['state'],r['outside_timer_probes'][0]['state'],
            cycle=1,component='initial_state').evaluate(COARSE_50C_THRESHOLDS)
        require(initial['normalized_l2']==0,'restored initial sampled state differs')
        for a,b in zip(l['cycles'],r['cycles'],strict=True):
            def trace(c):
                return {f"{v['source_rank']}:{v['source_step']}:{v['rollout_operation']}":v
                        for x in c['actions'] for v in x['trace_entries']}
            require(trace(a)==trace(b),'actual rank-qualified trace/RNG identities differ')
            reports.append(dict(rank=l['rank'],**compare_numeric_state_sketches(a['state_sketch'],b['state_sketch'],
                cycle=a['cycle'],component='trainable_optimizer_ema').evaluate(COARSE_50C_THRESHOLDS)))
            la={x['action_id']:x['losses'] for x in a['actions'] if x['losses']}
            lb={x['action_id']:x['losses'] for x in b['actions'] if x['losses']}
            require(la.keys()==lb.keys(),'loss operation inventory differs')
            for key in la:
                require(len(la[key])==len(lb[key]),'loss backward count differs')
                for x,y in zip(la[key],lb[key],strict=True):
                    require(x['role']==y['role'],'loss role differs')
                    losses.append(dict(rank=l['rank'],cycle=a['cycle'],action=key,role=x['role'],
                        reference=x['value'],candidate=y['value'],
                        symmetric_relative_difference=abs(x['value']-y['value'])/max(abs(x['value']),abs(y['value']),1e-12)))
    return dict(schema_version=COMPARISON_SCHEMA,status='REVIEW_SAMPLED_PARITY_ONLY',source_head=head,
        reference_candidate_id=left[0]['candidate_id'],candidate_id=right[0]['candidate_id'],
        semantics_status='PASS',rank_count=16,cycles=2,sample_rng_status='PASS',finite_status='PASS',
        initial_sampled_state_status='PASS',state_metrics=reports,loss_metrics=losses,
        coarse_state_status='PASS' if all(r['status']=='PASS' for r in reports) else 'REVIEW',
        strict_numerical_gate='NOT_ESTABLISHED_BY_SAMPLED_COORDINATES')


def validate_comparison(value):
    require(value.get('schema_version')==COMPARISON_SCHEMA and value.get('status')=='REVIEW_SAMPLED_PARITY_ONLY'
            and value.get('rank_count')==16 and value.get('cycles')==2
            and all(value.get(k)=='PASS' for k in ('semantics_status','sample_rng_status','finite_status','initial_sampled_state_status'))
            and len(value.get('state_metrics',[]))==32 and bool(value.get('loss_metrics')),
            'Helios comparison missing/failed; REVIEW is not strict parity')


def prepare_pair(reference,candidate):
    for k in ('profile_id','integration_head','asset_fingerprint','workload_semantics_hash'):
        require(reference[k]==candidate[k],f'Helios pair {k} differs')
    lo,_,lr,lrefs=checked(reference);ro,_,rr,rrefs=checked(candidate)
    value=compare_rows(lr,rr,head=reference['integration_head'])
    value['sources']={
        'reference':dict(execution_receipt=ref(lo/'effective-contract.node0.json'),native_ranks=lrefs),
        'candidate':dict(execution_receipt=ref(ro/'effective-contract.node0.json'),native_ranks=rrefs),
    }
    target=ro/'search-gate';target.mkdir(exist_ok=False)
    p=target/'comparison.json';write_new_json(p,value);validate_comparison(value)
    ready={}
    for plan,out in ((reference,lo),(candidate,ro)):
        cid=plan['search_candidate']['id']
        gate=dict(schema_version=GATE_SCHEMA,status='PASS',candidate_id=cid,comparison=ref(p),
            execution_receipt=ref(out/'effective-contract.node0.json'),
            native_ranks=read_ranks(out)[1],numerical_status='REVIEW',
            integration_binding={k:plan[k] for k in ('integration_head','profile_sha256','asset_fingerprint')})
        q=target/(cid+'-gate.json');write_new_json(q,gate)
        assets=copy.deepcopy(plan['asset_config']);assets['correctness']=ref(q)
        ready[cid]=dict(source_head=plan['integration_head'],candidate_id=cid,semantics_status='PASS',
            numerical_status='REVIEW',strict_numerical_gate='NOT_ESTABLISHED_BY_SAMPLED_COORDINATES',
            comparison=ref(p),correctness=ref(out/'bound-summary.json'),assets=assets)
        write_new_json(target/(cid+'-preparation.json'),ready[cid])
    return ready


def _correctness_source(source,receipt,candidate_id,root):
    """Bind an executed correctness plan to its own exact rank files/run ID."""
    from .execution import validated_profile
    plan=bound(source['execution_receipt'])
    validated_profile(plan,root=root)
    require(plan['mode']=='correctness' and plan['search_candidate']['id']==candidate_id,
            'Helios source is not this candidate correctness run')
    require(all(plan[k]==receipt[k] for k in ('integration_head','asset_fingerprint','profile_id',
                                            'workload_semantics_hash')),
            'Helios correctness source/assets/semantics differ')
    out=Path(plan['asset_config']['run_root'])/plan['run_id']
    require(Path(source['execution_receipt']['path']).resolve()==(out/'effective-contract.node0.json').resolve(),
            'Helios source plan does not belong to the executed run')
    references=source['native_ranks']
    require(len(references)==16 and {Path(r['path']).resolve() for r in references}==
            {(out/'execution/ranks'/f'rank_{i:02d}.json').resolve() for i in range(16)},
            'Helios native rank paths do not belong to correctness run')
    rows=[bound(r) for r in references]
    summary=summarize(rows,expected_head=receipt['integration_head'],candidate_id=candidate_id)
    require(summary['run_id']==plan['run_id'] and summary['mode']=='correctness',
            'Helios native rank run identity differs from correctness plan')
    return plan,rows


def validate_admission(data,receipt,root=None):
    root=root or repository_root()
    require(data.get('schema_version')==GATE_SCHEMA and data.get('status')=='PASS','Helios native gate missing')
    cid=receipt['search_candidate']['id'];require(data['candidate_id']==cid,'Helios gate candidate differs')
    require(data['integration_binding']=={k:receipt[k] for k in ('integration_head','profile_sha256','asset_fingerprint')},
            'Helios gate source/assets/profile differs')
    comparison=bound(data['comparison']);validate_comparison(comparison)
    require(comparison['source_head']==receipt['integration_head'] and cid in
        (comparison['reference_candidate_id'],comparison['candidate_id']),'Helios comparison binding differs')
    sources=comparison.get('sources',{})
    require(set(sources)=={'reference','candidate'},'Helios comparison executed sources missing')
    field='reference' if cid==comparison['reference_candidate_id'] else 'candidate'
    require(sources[field]=={k:data[k] for k in ('execution_receipt','native_ranks')},
            'Helios gate sources differ from actual comparison inputs')
    left,lrows=_correctness_source(sources['reference'],receipt,comparison['reference_candidate_id'],root)
    right,rrows=_correctness_source(sources['candidate'],receipt,comparison['candidate_id'],root)
    expected=left if field=='reference' else right
    require(expected['profile_sha256']==receipt['profile_sha256'],'Helios admitted implementation differs')
    actual=compare_rows(lrows,rrows,head=receipt['integration_head'])
    require({k:v for k,v in comparison.items() if k!='sources'}==actual,
            'Helios comparison differs from executed rank observations')


def validate_launcher_admission(environment):
    impl=current_implementation(environment)
    require(impl is not None,'Helios native identity/mode missing')
    if impl.candidate_bundle.enabled or impl.request is not None:
        require(all(environment.get(k) == v for k,v in impl.environment().items()
                    if k!=MODE_ENV and not (impl.request is not None and k==impl.trace_hash_environment)),
                'safe No-GAN launcher configuration/flags missing or changed')
    if impl.request is not None:
        import re
        require(re.fullmatch(r'[0-9a-f]{64}',environment.get(impl.trace_hash_environment,'')) is not None,
                'ORIGINAL Helios request requires exact asset-bound trace hash')
    if impl.candidate_bundle.enabled:
        impl.candidate_bundle.source_identity()
    if environment[MODE_ENV]=='correctness':return dict(status='PASS_CORRECTNESS_BOOTSTRAP_ONLY')
    p=environment.get('DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY','')
    require(bool(p) and Path(p).is_file(),'Helios native gate receipt missing')
    data=bound(dict(path=p,sha256=environment.get('DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256','')))
    # Source-owned dispatcher gate is also checked here for direct launcher use.
    plan=bound(data['execution_receipt'])
    require(git_head(repository_root())==plan['integration_head'] and plan['search_candidate']['id']==impl.id,
            'Helios launcher source/identity differs from native gate')
    validate_admission(data,plan)
    if impl.request is not None:
        trace=plan.get('original_helios_trace_binding',{})
        key=load_profile(impl.workload)['required_assets']['trace']['environment']
        require(trace.get('request_sha256')==impl.request.digest
                and environment.get(impl.trace_hash_environment)==trace.get('bundle_hash')
                and environment.get(key)==plan['asset_config']['assets']['trace']['path'],
                'Helios launcher differs from admitted request trace')
    return dict(status='PASS_NATIVE_SOURCE_BOUND_ADMISSION',numerical_status='REVIEW')


def launch_evidence(receipt,out):
    refs=[];resources=[]
    for i in (0,1):
        node=receipt['nodes'][i];p=out/f'effective-contract.node{i}.json';plan=_object(p)
        require(plan['integration_head']==receipt['integration_head'] and plan['nodes'][i]==node,
                'effective launcher differs')
        require(node['environment'][MODE_ENV]=={'diagnostic-timing':'native'}.get(receipt['mode'],receipt['mode'])
                and node['command'][5]=='timing','not direct unprofiled Helios timing launcher')
        resource=_object(out/f'search-resources.node{i}.json')
        require(resource['status']=='PASS' and resource['run_id']==receipt['run_id'] and resource['node_rank']==i
                and len(resource['samples'])>=2,'Helios resource observation missing')
        for event in ('oom','oom_kill'):
            values=[s['cgroup_events'][event] for s in resource['samples']]
            require(max(values)==min(values),'Helios cgroup OOM')
        q=out/'launcher'/f'node_{i}'/'teardown_status.txt'
        flags=dict(line.split('=',1) for line in q.read_text().splitlines() if '=' in line)
        require(flags==dict(status='PASS',gpu_process_count='0',owner_run_id=receipt['run_id']),
                'Helios teardown incomplete')
        resources.append(resource);refs.extend((ref(p),ref(q),ref(out/f'search-resources.node{i}.json')))
    return resources,refs


def normalize_native(receipt,preparation,identity):
    from .helios_action_costs import PRECISION
    out,summary,rows,refs=checked(receipt);impl=HeliosImplementation.parse(receipt['search_candidate']['id'])
    require(receipt['mode']=='diagnostic-timing' and summary['mode']=='native','not native cycle2 timing')
    require(identity['precision_fingerprint']==PRECISION,'Helios precision differs')
    for name,key in (('source_head','integration_head'),('asset_fingerprint','asset_fingerprint'),
                     ('semantic_fingerprint','workload_semantics_hash')):
        require(identity[name]==receipt[key],f'Helios timing {name} differs')
    validate_admission(bound(preparation['assets']['correctness']),receipt)
    resources,launch_refs=launch_evidence(receipt,out);ranks=[]
    for raw in rows:
        rank=raw['rank'];sample=resources[rank//8]['samples']
        mem=[a['memory_after'] for c in raw['cycles'] for a in c['actions']]
        ranks.append(dict(rank=rank,status='PASS',**{k:'PASS' for k in ('census','versions','sample_rng','finite','dtype','memory')},
            cuda_oom_delta=0,cgroup_oom_delta=0,
            peak_allocated_bytes=max(m['max_allocated_bytes'] for m in mem),
            peak_reserved_bytes=max(m['max_reserved_bytes'] for m in mem),
            min_driver_free_bytes=min(g['free_bytes'] for s in sample for g in s['gpus'] if g['device']==rank%8),
            host_peak_bytes=max(s['cgroup_current_bytes'] for s in sample),
            host_peak_scope='MAX_SAMPLED_CGROUP_CURRENT_NOT_CONTINUOUS_PEAK'))
    result=dict(schema_version=MEASUREMENT_SCHEMA,status='PASS',**identity,candidate_id=impl.id,
        execution_fingerprint=fingerprint(impl.candidate().to_dict()),profiled=False,same_process=True,initializations=1,
        semantics_status='PASS',numerical_status='REVIEW',strict_numerical_gate='NOT_RUN',formal_timing_eligible=False,
        profiler_status='DISABLED',resource_status='EXCLUSIVE_LOCKED',cleanup_status='PASS',
        timer_scope='NATIVE_CYCLE_OUTSIDE_PROBES',native_timer_denominator=summary['timer_scope'],
        cycle_wall_seconds=summary['cycle_wall_seconds'],ranks=ranks,
        native_evidence=[ref(out/'bound-summary.json'),preparation['comparison'],*refs,*launch_refs])
    validate_measurement(result,identity,impl.candidate());return result


def main():
    import argparse,os
    p=argparse.ArgumentParser();p.add_argument('action',choices=('summarize','admission'))
    p.add_argument('--run-root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--expected-head');p.add_argument('--candidate-id');args=p.parse_args()
    if args.action=='admission':print(json.dumps(validate_launcher_admission(os.environ)));return
    rows,refs=read_ranks(args.run_root)
    result=summarize(rows,expected_head=args.expected_head,candidate_id=args.candidate_id)
    result['native_evidence']=refs;write_new_json(args.output,result)


if __name__=='__main__':main()
