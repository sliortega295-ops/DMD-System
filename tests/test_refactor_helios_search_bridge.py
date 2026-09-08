"""CPU protocol tests; fixture observations are never real GPU evidence."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from distillschedule.search.helios_implementation import HeliosImplementation, NO_GAN, GAN
from distillschedule.workloads.best_known.registry import fingerprint, load_profile


@pytest.mark.parametrize("workload,variant", [(NO_GAN,"no-gan"),(GAN,"gan")])
def test_both_variants_route_typed_native_actions_not_canonical_recipe(workload,variant):
    from distillschedule.search.model_adapters import adapter_for, SUPPORTED_WORKLOADS
    from distillschedule.search.execution import resolve_candidate, validated_profile
    impl=HeliosImplementation(variant,"world16-fsdp",512,5,False)
    assert workload in SUPPORTED_WORKLOADS
    adapter=adapter_for(workload)
    assert adapter.implementation_type is HeliosImplementation
    plan=resolve_candidate(workload,impl.id,run_id="cpu-not-run",mode="correctness")
    assert plan['native_mode']=='timing'
    assert plan['fully_resolved_profile']['runtime']['planner_implementation']==impl.assignment
    assert validated_profile(plan,root=Path(__file__).parents[1])
    assert adapter.capability_report()['arbitrary_pool_lowering'] is False


def native_rows(monkeypatch, *, variant='no-gan', mode='calibration', window=3):
    from test_refactor_helios_planner_native import _fixture
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f=_fixture(monkeypatch,variant=variant,mode=mode,window=window)
    run_two_cycles(**f.kwargs)
    rows=[]
    for rank in range(16):
        r=deepcopy(f.result);r.update(rank=rank,run_id='fixture-only')
        for c in r['cycles']:
            for a in c['actions']:
                a['source_rank']=rank
                if 'helper_record' in a:a['helper_record']['source_rank']=rank
                for t in a['trace_entries']:t['source_rank']=rank
        rows.append(r)
    return rows,f.impl


@pytest.mark.parametrize('variant,window',[('no-gan',3),('gan',5)])
def test_native_summary_rechecks_actual_inventory_and_two_cycles(monkeypatch,variant,window):
    from distillschedule.search.helios_native import summarize
    rows,impl=native_rows(monkeypatch,variant=variant,window=window)
    value=summarize(rows,expected_head=rows[0]['source_head'],candidate_id=impl.id)
    assert value['status']=='PASS' and value['rank_count']==16
    assert value['calibration_only'] and not value['formal_timing_eligible']
    assert value['cycle_wall_seconds']['2']==max(r['cycles'][1]['wall_seconds'] for r in rows)
    bad=deepcopy(rows);bad[10]['cycles'][1]['actions'][0]['counts']['generator_optimizer']=2
    with pytest.raises(ValueError,match='counts|optimizer'):
        summarize(bad,expected_head=rows[0]['source_head'],candidate_id=impl.id)
    with pytest.raises(ValueError,match='16|rank'):
        summarize(rows[:-1],expected_head=rows[0]['source_head'],candidate_id=impl.id)


@pytest.mark.parametrize('variant,window',[('no-gan',3),('gan',5)])
def test_graph_cost_identity_keeps_cycles_commits_and_persistence(monkeypatch,variant,window):
    from distillschedule.search.helios_action_costs import action_graph, component_contract, observed_work
    from distillschedule.search.plan_ir import validate_versions
    rows,impl=native_rows(monkeypatch,variant=variant,window=window)
    graph=action_graph(impl)
    validate_versions(graph)
    names=[o.operation_id for o in graph.operations]
    assert names==[f'cycle{c["cycle"]}.{a["action_id"]}' for c in rows[0]['cycles'] for a in c['actions']]
    by_id=observed_work(rows)
    op=next(o for o in graph.operations if o.operation_id=='cycle2.g')
    c=component_contract(op,impl,by_id[op.operation_id])
    assert impl.id not in json.dumps(c.to_dict())
    changed=HeliosImplementation(variant,impl.mesh,2048,window,False)
    assert component_contract(op,changed,by_id[op.operation_id]).digest!=c.digest
    if variant=='no-gan':
        moved=HeliosImplementation(variant,impl.mesh,512,1,False)
        assert component_contract(op,moved,by_id[op.operation_id]).digest==c.digest
    else:
        persistent=HeliosImplementation(variant,impl.mesh,512,5,True)
        assert [o.operation_id for o in action_graph(persistent).operations]==names
        op=next(o for o in graph.operations if o.operation_id=='cycle2.r_all')
        assert component_contract(op,persistent,by_id[op.operation_id]).digest!=component_contract(op,impl,by_id[op.operation_id]).digest


def test_helios_launcher_admission_does_not_fall_through_legacy_or_bypass():
    from distillschedule.search.helios_native import validate_launcher_admission
    with pytest.raises(ValueError,match='identity|mode'):
        validate_launcher_admission({})
    i=HeliosImplementation('no-gan','world16-fsdp',512,5,False)
    env=i.environment()|{'DISTILLSCHEDULE_HELIOS_PLANNER_MODE':'correctness'}
    assert validate_launcher_admission(env)['status']=='PASS_CORRECTNESS_BOOTSTRAP_ONLY'
    env['DISTILLSCHEDULE_HELIOS_PLANNER_MODE']='native'
    with pytest.raises(ValueError,match='gate|receipt'):
        validate_launcher_admission(env)


def test_helios_summary_command_is_not_old_single_cycle_summary():
    from distillschedule.search.execution import resolve_candidate
    from distillschedule.workloads.best_known.evidence import summary_command
    i=HeliosImplementation('no-gan','world16-fsdp',512,5,False)
    p=resolve_candidate(NO_GAN,i.id,mode='correctness')
    p['nodes']=[dict(repo_root='/source',python='/python',environment={})]
    p['asset_config']={'run_root':'/evidence'}
    command=summary_command(p)
    assert 'distillschedule.search.helios_native' in command
    assert not any('summarize_static_fsdp2_p0' in part for part in command)


@pytest.mark.parametrize('variant,window',[('no-gan',3),('gan',5)])
def test_component_search_cpu_fixture_preserves_native_decisions_and_missing_costs(tmp_path,monkeypatch,variant,window):
    from dataclasses import replace
    from distillschedule.search import helios_model_adapter as m
    from distillschedule.search.cost_model import CostModel,OperationCost
    from distillschedule.search.helios_action_costs import PRECISION,anchors
    from distillschedule.search.contracts import semantic_contract
    from distillschedule.search.model_search import search_plans
    from test_refactor_model_planning import BINDING
    rows,impl=native_rows(monkeypatch,variant=variant,window=window)
    workload=impl.workload
    for r in rows:
        for mem in [r['post_setup_memory'],*[a[k] for c in r['cycles'] for a in c['actions'] for k in ('memory_before','memory_after')]]:
            mem['driver_total_bytes']=40*2**30;mem['driver_free_bytes']=35*2**30
    identity={**BINDING,'source_head':rows[0]['source_head'],'precision_fingerprint':PRECISION,
        'semantic_fingerprint':fingerprint(semantic_contract(load_profile(workload)))}
    pair=anchors(workload)
    def read(reference,root):
        return {'binding':identity},pair[reference['index']],deepcopy(rows)
    monkeypatch.setattr(m,'_read_fragment',read)
    path=tmp_path/'layout.json';path.write_text(json.dumps({'schema_version':m.SCHEMA,'fragments':[{'index':0},{'index':1}]}))
    problem=m.build_problem(path)
    # No CPU fixture can pass validate_native_problem or native admission.
    problem=replace(problem,evidence_scope='CPU_FIXTURE_ONLY')
    raw=m._component_samples(impl,rows)
    costs=CostModel(identity,tuple(OperationCost(k,tuple(v['samples_ns']),v['workspace_bytes_per_rank'],
        {'component_contract':v['contract'].to_dict()}) for k,v in raw.items()),scope='CPU_FIXTURE_ONLY')
    result=search_plans(problem,costs,max_proposals=40,exhaustive=True)
    assert result['status']=='PASS_CPU_SEARCH'
    assert len(result['ranked'])==(5 if variant=='no-gan' else 1)
    assert all(r['status']=='UNCALIBRATED' for r in result['rejections'])
    assert not result['claimable_speedup']
    with pytest.raises(ValueError,match='measured'):
        m.validate_native_problem(problem)


def test_rehashed_helios_cost_cannot_replace_actual_rank_samples(monkeypatch):
    from distillschedule.search import helios_model_adapter as m
    rows,impl=native_rows(monkeypatch)
    binding={'source_head':rows[0]['source_head']}
    monkeypatch.setattr(m,'_read_fragment',lambda reference,root: ({'binding':binding},impl,rows))
    key,value=next(iter(m._component_samples(impl,rows).items()))
    raw=dict(key=key,binding=binding,samples_ns=value['samples_ns'],
        workspace_bytes_per_rank=value['workspace_bytes_per_rank'],
        component_contract=value['contract'].to_dict(),operation_ids=value['operation_ids'],
        helios_native_fragment={'path':'fixture-only','sha256':'0'*64})
    m.validate_component_receipt(raw,{})
    for field in ('samples_ns','workspace_bytes_per_rank','operation_ids'):
        bad=deepcopy(raw)
        bad[field]={'samples_ns':[1], 'workspace_bytes_per_rank':-1,
                    'operation_ids':['invented']}[field]
        with pytest.raises(ValueError,match='native rank observations'):
            m.validate_component_receipt(bad,{})


def test_native_alias_cannot_enable_profiler_or_silent_recipe_fallback(tmp_path):
    from distillschedule.search.execution import resolve_candidate
    from distillschedule.search.registry import get_candidate
    for workload,variant in ((NO_GAN,'no-gan'),(GAN,'gan')):
        impl=HeliosImplementation(variant,'world16-fsdp',512,5,False)
        assert get_candidate(workload,impl.id)==impl.candidate()
        for mode in ('correctness','calibration','diagnostic-timing'):
            plan=resolve_candidate(workload,impl.id,mode=mode,run_id='fixture-not-run')
            assert plan['native_mode']=='timing'
            assert plan['fully_resolved_profile']['runtime']['planner_implementation']==impl.assignment
        with pytest.raises(ValueError):
            resolve_candidate(workload,impl.id,mode='profile',run_id='fixture-not-run')
