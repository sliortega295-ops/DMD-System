"""CPU-only native graph proof; no invented service costs or executable backend."""
from copy import deepcopy

import pytest

from distillschedule.search.livewan_pool_contract import build_pool_contract, validate_pool_contract
from distillschedule.search.plan_ir import ancestors, graph_from_dict, validate_versions
from test_refactor_livewan_action_observation import action_fixture as base_fixture


def action_fixture():
    rows=base_fixture()
    for r in rows:r["implementation"]="joint-livewan1-pipeline-m1-g1-r1"
    return rows


def contract():
    return build_pool_contract(action_fixture(), expected_head='a'*40)


def test_two_cycle_pool_graph_preserves_actual_helper_batch_and_version_contract():
    c=contract();g=graph_from_dict(c['graph']);validate_versions(g)
    assert c['status']=='CPU_SOURCE_CONTRACT_ONLY'
    assert c['cost_admission']=='UNADMITTED' and c['native_lowering']=='NOT_RUN'
    assert c['graph']['metadata']['final_versions']==dict(generator=2,critic=10,teacher=0,shadow_epoch=1,ema=0)
    ops={o.operation_id:o for o in g.operations}
    assert len([o for o in ops.values() if o.metadata['family']=='rollout'])==20
    assert len([o for o in ops.values() if o.metadata['family']=='critic_backward'])==10
    assert all(o.metadata['physical_batch_per_rank']==2 for o in ops.values() if o.metadata['family']=='critic_backward')
    assert all(len(o.metadata['rank_samples'])==8 for o in ops.values() if o.metadata['family']=='rollout')
    assert ops['s0.c.backward'].metadata['produced_with_generator_version']==0
    assert ops['s1.c.backward'].metadata['produced_with_generator_version']==1
    assert ops['s5.c.backward'].metadata['produced_with_generator_version']==1
    assert all('generator' not in o.required_versions for o in ops.values() if o.metadata['family']=='critic_backward')
    assert ops['shadow.load.C5'].produced_versions=={'shadow_epoch':1}
    assert ops['shadow.load.C5'].metadata['native_critic_version']==5


def test_depth_one_lookahead_is_not_accidentally_serialized_or_extended():
    g=graph_from_dict(contract()['graph']);a=ancestors(g)
    assert 's4.c.commit' not in a['s5.g.rollout.w0']
    assert 's5.g.rollout.w0' in a['shadow.load.C5']
    assert 'shadow.load.C5' in a['s5.g.rollout.w1']
    assert 's4.c.commit' in a['s5.g.fake.w0']
    assert 's0.c.commit' not in a['s0.g.commit']
    assert 's0.g.commit' not in a['s0.c.commit']
    assert not any(k.startswith('s10.') for k in a)
    classes={e.metadata['classification'] for e in g.dependencies}
    assert classes=={'semantic','native_transport_safety','frozen_native_order','depth1_policy'}


def test_transport_census_separates_gloo_latents_teacher_velocities_and_shadow():
    c=contract();m=c['messages']
    host=[x for x in m if x['path']=='pinned_gloo']
    assert len(host)==17 and sum(x['bytes_per_lane'] for x in host)==1700
    assert {tuple(x['key']) for x in host}=={(s,w) for s in range(10) if s%5 for w in (0,1)}|{(5,0)}
    assert len([x for x in m if x['path']=='nccl_latent'])==3
    assert len([x for x in m if x['path']=='nccl_teacher_velocity'])==4
    assert len([x for x in m if x['path']=='paired_nccl_shadow'])==1
    assert not any(x['payload']=='final_dmd_gradient' for x in m)


def test_unknown_storage_is_not_zero_workspace_and_async_holds_are_retained():
    c=contract();s={x['id']:x for x in c['storage']}
    assert s['prepared.s5.w0']['observed_bytes'] is None
    assert s['prepared.s5.w0']['last_semantic_consumer']=='s5.g.backward.w0'
    assert s['prepared.s5.w0']['native_release_boundary']=='cycle2.g.clear_precomputed_first'
    assert 's1.latent.complete.w0' in s['host_source.s1.w0']['async_completion_handles']
    assert s['host_source.s1.w0']['alias_status']=='UNRESOLVED_CONTIGUOUS_MAY_ALIAS_PREPARED_X0'
    assert s['pinned_buffers.G']['observed_bytes']==1700
    assert s['pinned_buffers.G']['allocation_boundary']=='setup'
    assert s['teacher_send.s0.w1']['native_release_boundary']=='s0.teacher.complete.w1'
    assert 's0.c.backward' in ancestors(graph_from_dict(c['graph']))['s0.teacher.complete.w1']
    assert c['memory_feasibility']=='UNRESOLVED'


@pytest.mark.parametrize('bad', ['sample','source','shadow','missing_edge','zero_workspace','fake_cost','native_claim'])
def test_contract_tampering_and_bad_observations_fail_closed(bad):
    raw=action_fixture()
    if bad in ('sample','source'):
        if bad=='sample':raw[8]['rows'][-1]['metadata']['rows'][0]['seeds']['rollout']=-1
        else:raw[0]['source_head']='b'*40
        with pytest.raises(ValueError):build_pool_contract(raw,expected_head='a'*40)
        return
    c=deepcopy(contract())
    if bad=='shadow':
        next(x for x in c['graph']['operations'] if x['operation_id']=='shadow.load.C5')['produced_versions']['shadow_epoch']=5
    elif bad=='missing_edge':c['graph']['dependencies'].pop()
    elif bad=='zero_workspace':next(x for x in c['storage'] if x['id']=='prepared.s5.w0')['observed_bytes']=0
    elif bad=='fake_cost':c['cost_admission']='PASS'
    else:c['native_lowering']='PASS'
    with pytest.raises(ValueError):validate_pool_contract(c)


def test_native_copy_stream_sender_fifo_and_final_drains_are_explicit():
    c=contract();a=ancestors(graph_from_dict(c['graph']))
    assert 's4.latent.complete.w1' in a['s5.latent.complete.w0']
    assert 's4.latent.d2h_ready.w1' in a['s5.latent.d2h_ready.w0']
    assert 's9.latent.complete.w1' in a['transport.finish.G']
    assert 's9.c.commit' in a['transport.finish.Score']
    assert 'transport.finish.G' not in a['s5.g.rollout.w0']


def test_observation_rank_order_does_not_change_the_compiled_contract():
    raw=action_fixture()
    assert build_pool_contract(raw,expected_head='a'*40)==build_pool_contract(raw[::-1],expected_head='a'*40)


def test_shadow_submission_precedes_score_next_entry_and_transport_owners_are_local():
    c=contract();g=graph_from_dict(c['graph']);a=ancestors(g)
    assert 'shadow.pack.C5' in a['s5.latent.ready.w0']
    assert 'cycle2.g.role_return' in a['cycle2.g.clear_precomputed_first']
    assert all(tuple(o.metadata['ranks'])==tuple(range(8)) for o in g.operations
               if o.metadata['family'] in ('latent_d2h_ready','latent_complete'))
    raw=base_fixture()
    with pytest.raises(ValueError,match='C34'):build_pool_contract(raw,expected_head='a'*40)


def test_peer_wait_returns_have_no_invented_happens_before():
    c=contract();a=ancestors(graph_from_dict(c['graph']))
    for stem in ('s1.latent','s0.teacher'):
        assert f'{stem}.complete.w0' not in a[f'{stem}.ready.w0']
        assert f'{stem}.ready.w0' not in a[f'{stem}.complete.w0']
        assert f'{stem}.delivery.w0' in a[f'{stem}.ready.w0']
    c['evidence']['gate_status']='FAIL'
    with pytest.raises(ValueError,match='not PASS'):validate_pool_contract(c)


def test_reference_lifetimes_and_byte_scope_do_not_claim_measured_storage_release():
    c=contract();g=graph_from_dict(c['graph']);s={x['id']:x for x in c['storage']}
    assert not next(o for o in g.operations if o.operation_id=='shadow.source_wait.C5').required_versions
    assert s['pinned_buffers.G']['byte_scope']=='PER_OWNING_RANK_NOT_POOL_SUM'
    assert tuple(s['pinned_buffers.G']['owning_ranks'])==tuple(range(8))
    assert s['b2_inputs.s0']['producer'].startswith('UNRESOLVED')
    assert s['teacher_send.s0.w1']['actual_storage_release']=='UNRESOLVED'


def test_saved_contract_alone_never_revalidates_the_source_receipts():
    raw=action_fixture();c=build_pool_contract(raw,expected_head='a'*40)
    assert validate_pool_contract(c)['native_evidence_check']=='NOT_RECHECKED'
    assert validate_pool_contract(c,observations=raw)['native_evidence_check']=='PASS_REUSED_NATIVE_RECEIPTS'
    raw[0]['rows'][1]['duration_ns']+=1
    with pytest.raises(ValueError,match='content changed'):validate_pool_contract(c,observations=raw)


def test_json_roundtrip_and_original_extraction_reconciliation():
    import json
    raw=action_fixture();c=json.loads(json.dumps(build_pool_contract(raw,expected_head='a'*40)))
    assert validate_pool_contract(c,observations=raw)['native_evidence_check']=='PASS_REUSED_NATIVE_RECEIPTS'
    c['evidence']['latent_contract']['buffer_count']=1
    with pytest.raises(ValueError,match='extracted evidence'):validate_pool_contract(c,observations=raw)
