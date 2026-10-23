"""Synthetic CPU fixtures: never empirical timings or GPU validation."""
from pathlib import Path
import pytest

from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation
from distillschedule.search.h3_implementation import H3, H3Implementation


def test_native_macro_graphs_keep_fixed_accounting_scope():
    from distillschedule.search.native_macro_adapter import native_graph
    from distillschedule.search.plan_ir import validate_versions
    for workload, count in ((LIVEWAN, 10), (H3, 14)):
        graph = native_graph(workload)
        assert len(graph.operations) == count
        assert graph.metadata['arbitrary_pool_dag_lowering'] is False
        assert graph.metadata['native_final_versions'] == dict(generator=2, critic=10, ema=0)
        validate_versions(graph)
    # LiveWan partitions are NOT independent G and Score service operations.
    assert native_graph(LIVEWAN).metadata['scope'] == 'COUPLED_COMMIT_ACCOUNTING_PARTITIONS'


def test_implementation_domains_do_not_silently_relax_prerequisites():
    from distillschedule.search.native_macro_adapter import implementations
    assert len(implementations(LIVEWAN)) == 5
    assert len(implementations(H3)) == 4
    for i in implementations(LIVEWAN):
        assert LiveWanImplementation.from_assignment(i.assignment) == i
        profile = i.lower_profile()
        assert profile['contract']['actual_global_batch'] == 16
        assert profile['runtime']['planner_implementation'] == i.assignment
    with pytest.raises(ValueError, match='requires'):
        LiveWanImplementation('pipeline', False, True, False)


def test_async_commit_partition_uses_score_ranks_not_cross_rank_clock_union():
    from distillschedule.search.native_macro_adapter import observed_operations
    rows = [dict(cycle=c, step=s, time_ns=(c-1)*50+(s+1)*10)
            for c in (1, 2) for s in range(5)]
    raw = dict(commit_boundaries=rows)
    obs = observed_operations(LIVEWAN, raw)
    assert [r['duration_ns'] for r in obs] == [10]*10
    assert [r['cycle'] for r in obs] == [1]*5+[2]*5
    with pytest.raises(ValueError, match='boundary'):
        observed_operations(LIVEWAN, dict(commit_boundaries=rows[:-1]))
    with pytest.raises(ValueError, match='clock'):
        observed_operations(LIVEWAN, dict(commit_boundaries=[{**r, 'time_ns': 0} for r in rows]))


def test_h3_duplicate_fake_ranges_not_overwritten():
    from distillschedule.search.native_macro_adapter import observed_operations
    labels = ['h3/student_step', 'h3/critic_prepare_5xG']+['h3/critic_update_F']*5
    raw = dict(rows=[dict(cycle=c, label=l, duration_ns=10, start_ns=j*10, stop_ns=(j+1)*10)
                    for c in (1, 2) for j,l in enumerate(labels)])
    assert len(observed_operations(H3, raw)) == 14


def test_shared_adapter_dispatch_rejects_unknown_and_supports_three_workloads():
    from distillschedule.search.model_adapters import adapter_for
    from distillschedule.search.wan_implementation import WAN
    for w in (WAN, LIVEWAN, H3):
        assert adapter_for(w).capability_report()['workload'] == w
    with pytest.raises(ValueError, match='adapter'):
        adapter_for('helios_unknown')


@pytest.mark.parametrize('workload,new_frames', [(LIVEWAN,False),(LIVEWAN,True),(H3,False)])
def test_partial_calibrated_subspace_never_invents_missing_costs(tmp_path, workload, new_frames):
    import json
    from distillschedule.search.native_macro_adapter import (
        ADAPTER, MEMORY, build_problem, implementations, precision_contract, operation_contract, cost_key,
        lower_prediction,
    )
    from distillschedule.workloads.best_known.registry import git_head, repository_root, fingerprint, load_profile
    from distillschedule.search.contracts import semantic_contract
    from distillschedule.search.cost_model import CostModel, OperationCost
    from distillschedule.search.model_search import search_plans
    from distillschedule.search.livewan_request import LiveWanRequest
    request = LiveWanRequest(frames=69) if new_frames else None
    values = implementations(workload, livewan_request=request)
    anchors = (values[0], values[-1])
    bind = dict(source_head=git_head(repository_root()),
        precision_fingerprint=fingerprint(precision_contract(workload,livewan_request=request)),
        semantic_fingerprint=fingerprint(semantic_contract(request.profile() if request else load_profile(workload))),
        **{k:'a'*64 for k in ('asset_fingerprint','environment_fingerprint','hardware_fingerprint')})
    layout = dict(schema_version='distillschedule.native_macro_layout.v1', adapter=ADAPTER,
        workload=workload, binding=bind, status='PASS', profiled=False, memory_definition=MEMORY,
        seed_implementations=[i.id for i in anchors], memory_limits={str(r):100 for r in range(16)},
        implementations={i.id:dict(status='PASS', ranks=[dict(rank=r,status='PASS',source_head=bind['source_head'],
            post_setup_static_bytes=10) for r in range(16)]) for i in anchors})
    path=tmp_path/'layout.json'; path.write_text(json.dumps(layout))
    problem=build_problem(path)
    # Files are explicitly synthetic test receipts; not publication evidence.
    records=[]
    from distillschedule.workloads.best_known.registry import file_sha256
    for index,impl in enumerate(anchors):
        for op in problem.graph.operations:
            contract=fingerprint(operation_contract(workload,op,impl))
            payload=dict(schema_version='distillschedule.operation_calibration.v1', binding=bind,
                key=cost_key(workload,op,impl),samples_ns=[100-index*10],workspace_bytes_per_rank=20,
                status='PASS',profiled=False,kind='native_operation',operation_contract_sha256=contract)
            p=tmp_path/(impl.id+'-'+op.operation_id+'.json'); p.write_text(json.dumps(payload))
            records.append(OperationCost(payload['key'],tuple(payload['samples_ns']),20,
                {**{k:payload[k] for k in ('status','profiled','kind','operation_contract_sha256')},
                'source_receipt':str(p),'source_receipt_sha256':file_sha256(p)}))
    costs=CostModel(bind,tuple(records)); costs.verify_receipts()
    result=search_plans(problem,costs,exhaustive=True,max_proposals=32)
    assert len(result['ranked'])==2
    assert result['predicted_best']['assignment']==anchors[-1].assignment
    assert result['rejections'] and result['gpu_execution']=='NOT_RUN'
    plan=lower_prediction(result,problem,costs,run_id='cpu-fixture')
    assert plan['search_candidate']==anchors[-1].candidate().to_dict()
    assert problem.graph.metadata['arbitrary_pool_dag_lowering'] is False
    if new_frames:
        assert plan['fully_resolved_profile']['contract']['request'] == request.to_dict()
        assert problem.graph.metadata['workload_request'] == request.to_dict()
        changed = dict(layout)
        changed['implementations'] = {**layout['implementations'], values[0].id.removesuffix('-f69'): layout['implementations'][values[0].id]}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='mix requests'):
            build_problem(path)
