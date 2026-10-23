"""Diagnostic observations must not become invented asynchronous service costs."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from distillschedule.search.async_range_calibration import AsyncRangeCollector
from distillschedule.search.livewan_action_observation import TransportObserver
from distillschedule.runtime.livewan_host_runahead import PinnedHostRunaheadTransport
from distillschedule.workloads.livewan_crossscale import TRACE_BALANCED_8G8S_SLOT_PAIRS


def observer(*, ready=True):
    ticks = iter(range(10000))
    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
        def record(self, stream):
            self.stream, self.t = stream, next(ticks)
        def query(self):
            return ready
        def elapsed_time(self, other):
            assert self.stream == other.stream
            return other.t - self.t
    cuda = SimpleNamespace(Event=Event, current_stream=lambda device: 'consumer')
    return TransportObserver(torch=SimpleNamespace(cuda=cuda), device='cuda:0', rank=8,
                             clock=lambda: next(ticks))


def test_explicit_copy_stream_and_thread_host_ranges_without_extra_waits():
    o = observer()
    with o.gpu_range('latent_d2h', key=(1, 0), nbytes=100, stream='copy'):
        pass
    with o.host_range('gloo_send_completion', key=(1, 0), nbytes=100):
        pass
    result = o.finish()
    assert result['status'] == 'PASS'
    gpu, host = result['rows']
    assert gpu['clock'] == 'CUDA_EXPLICIT_STREAM_DURATION_ONLY'
    assert host['clock'] == 'HOST_MONOTONIC_LOCAL_PROCESS'
    assert gpu['bytes'] == host['bytes'] == 100
    assert gpu['duration_ns'] > 0 and host['duration_ns'] > 0
    assert 'gpu_start_ns' not in gpu  # No invented common cross-device origin.
    assert result['extra_synchronizations'] == result['extra_collectives'] == result['tensor_value_reads'] == 0
    assert result['formal_timing_eligible'] is False
    assert result['independent_link_service_admitted'] is False
    with pytest.raises(ValueError):
        o.finish()


def test_unfinished_or_failed_transport_is_not_a_cost():
    o = observer(ready=False)
    with o.gpu_range('latent_h2d', key=(1, 1), nbytes=100):
        pass
    with pytest.raises(ValueError, match='terminal'):
        o.finish()
    o = observer()
    with pytest.raises(RuntimeError):
        with o.host_range('gloo_receive_wait', key=(2, 0), nbytes=100):
            raise RuntimeError('real failure')
    assert o.finish()['status'] == 'FAIL'


def test_native_sender_calls_unchanged_and_disabled_observer_is_inert():
    logs = []
    class Spy:
        def host_range(self, kind, **kwargs):
            logs.append(('observe', kind, kwargs))
            return nullcontext()
    def execute(obs):
        calls = []
        t = object.__new__(PinnedHostRunaheadTransport)
        t.observer = obs
        t.torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda d: calls.append(('device', d))))
        t.dist = SimpleNamespace(isend=lambda *a, **kw: (
            calls.append(('isend', a, kw)) or SimpleNamespace(wait=lambda: calls.append(('send_wait',)))))
        t.device, t.peer, t.group, t.waves_per_step = 'cuda:0', 8, 'gloo', 2
        t.buffers, t.payload_bytes = {(1, 0): 'same-buffer'}, 100
        event = SimpleNamespace(synchronize=lambda: calls.append(('existing_copy_wait',)))
        t._send_after_copy((1, 0), event, 'same-source')
        return calls
    assert execute(None) == execute(Spy())
    assert [x[1] for x in logs] == ['sender_copy_ready_wait', 'gloo_send_completion']


def test_sender_thread_uses_thread_safe_observer_not_main_thread_collector():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    o = observer()
    owner = threading.get_ident()
    def work():
        with o.host_range('gloo_send_completion', key=(1, 0), nbytes=100):
            pass
    with ThreadPoolExecutor(1) as pool:
        pool.submit(work).result()
    row = o.finish()['rows'][0]
    assert row['thread_id'] != owner


def test_nested_native_helper_identity_is_not_same_as_parent_send_envelope():
    from test_refactor_async_range_calibration import fake_cuda
    c = AsyncRangeCollector(torch=SimpleNamespace(cuda=fake_cuda()), device='cuda:0', rank=8,
        source_head='a'*40, implementation='fixture', runtime_metadata=lambda: {'critic': 5})
    c.cycle = 2
    with c.range('teacher_with_send'):
        with c.range('native/teacher_velocity', metadata={'source_function': '_dmd_teacher_velocity'}):
            pass
    result = c.finish()
    parent, child = result['rows']
    assert child['parent'] == parent['index']
    assert child['metadata']['source_function'] == '_dmd_teacher_velocity'
    assert child['runtime_before'] == child['runtime_after'] == {'critic': 5}


def test_real_helper_decorator_preserves_return_exception_and_row_identity(monkeypatch):
    from distillschedule.runtime import livewan_crossscale_worker as worker
    captured = []
    class Spy:
        def range(self, label, **kwargs):
            captured.append((label, kwargs))
            return nullcontext()
    marker = object()
    row = SimpleNamespace(operation_key='step/0/slot/3', logical_step=0, logical_slot=3,
                          seeds={'rollout': 17})
    @worker._observe_native_action('generator_rollout')
    def native(*, row, failure=False):
        if failure:
            raise RuntimeError('native failed')
        return marker
    monkeypatch.setattr(worker, '_operation_calibrator', None)
    assert native(row=row) is marker and not captured
    monkeypatch.setattr(worker, '_operation_calibrator', Spy())
    assert native(row=row) is marker
    assert captured[0][1]['metadata']['rows'][0]['operation_key'] == row.operation_key
    assert captured[0][1]['metadata']['rows'][0]['seeds'] == row.seeds
    with pytest.raises(RuntimeError, match='native failed'):
        native(row=row, failure=True)


def test_legacy_envelopes_cannot_be_admitted_as_independent_action_costs():
    from distillschedule.search.livewan_action_observation import validate_action_observations
    raws = [dict(status='PASS', rank=r, source_head='a'*40, implementation='fixture',
                 rows=[], profiled=False) for r in range(16)]
    with pytest.raises(ValueError, match='native helper'):
        validate_action_observations(raws, expected_head='a'*40)


@pytest.mark.parametrize('kind,cycles,allowed', [('trajectory','2',True),('trajectory','50',False),
                                              ('two-cycle-timing','0',True),('timing','0',False)])
def test_observation_launcher_does_not_bypass_formal_timing_gate(kind,cycles,allowed):
    import os
    from pathlib import Path
    import subprocess
    source=Path('scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    fragment=source[source.index('if [[ "${OPERATION_CALIBRATION}" == 1 ]]; then'):source.index('[[ "${PROFILE_TWO_CYCLES}" == 0')]
    env={**os.environ,'OPERATION_CALIBRATION':'1','RUN_KIND':kind,'LIVEWAN_TRAJECTORY_CYCLES':cycles,
         'DISTILLSCHEDULE_SEARCH_CANDIDATE_ID':'fixture'}
    assert (subprocess.run(['bash','-c','set -eu\n'+fragment],env=env).returncode == 0) is allowed
    # The existing timing path still requires current-source paired correctness.
    assert '"expected_head": expected_head,' in source
    assert '"claimable_for_two_cycle_timing": True,' in source


def test_action_observation_is_typed_trajectory_only_not_new_timing_authority():
    from distillschedule.search.execution import resolve_candidate
    from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation
    from distillschedule.workloads.best_known.registry import load_profile
    c=LiveWanImplementation('pipeline',True,True,True)
    p=resolve_candidate(LIVEWAN,c.id,mode='action-observation')
    assert p['native_mode'] == 'trajectory'
    assert p['fully_resolved_profile']['runtime']['environment']['DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION']=='0'
    assert 'action-observation' not in load_profile(LIVEWAN)['allowed_run_modes']
    with pytest.raises(ValueError):
        resolve_candidate(LIVEWAN,LiveWanImplementation('fsdp').id,mode='action-observation')


def test_native_receiver_uses_one_existing_wait_and_same_h2d(monkeypatch):
    class Tensor:
        def copy_(self, source, **kwargs):
            calls.append(('copy', source, kwargs))
    class Spy:
        def host_range(self, kind, **kwargs):
            observed.append(kind)
            return nullcontext()
        gpu_range = host_range
    def execute(obs):
        t = object.__new__(PinnedHostRunaheadTransport)
        t.observer, t.is_generator, t.cycles, t.cross_cycle_lookahead = obs, False, 1, False
        t.waves_per_step, t.shape, t.dtype, t.device, t.payload_bytes = 2, (1, 2), 'fp32', 'cuda:0', 8
        t.received_keys, t.buffers = set(), {(1,0): 'buffer'}
        t.receive_works = {(1,0): SimpleNamespace(wait=lambda: calls.append(('existing_recv_wait',)))}
        t.torch = SimpleNamespace(empty=lambda *a, **kw: calls.append(('allocate',a,kw)) or Tensor())
        t.receive_to_device(step=1,wave=0)
        assert t.received_keys == {(1,0)}
    calls, observed = [], []
    execute(None)
    expected = calls.copy()
    calls.clear()
    execute(Spy())
    assert calls == expected
    assert observed == ['gloo_receive_wait', 'latent_h2d']


def action_fixture():
    sources = dict(generator_rollout='prepare_rollout', generator_backward='generator_backward_for_sample',
        fake_velocity='_dmd_fake_velocity', teacher_velocity='_dmd_teacher_velocity',
        critic_backward_b2='critic_backward_for_batch')
    result = []
    for rank in range(16):
        g, lane = rank < 8, rank % 8
        slots = TRACE_BALANCED_8G8S_SLOT_PAIRS[lane]
        rows = [dict(index=0,parent=None,start_ns=0,stop_ns=1000000,status='PASS')]
        def add(kind,step,slots):
            v = dict(generator=step//5+int(step%5!=0) if g else 0,
                     critic=0 if g else step,ema=0,shadow_loaded_version=5*(step//5))
            rows.append(dict(index=len(rows),parent=0,cycle=step//5+1,start_ns=len(rows)*10,
                stop_ns=len(rows)*10+2,duration_ns=2,status='PASS',runtime_before=v,runtime_after=v.copy(),
                metadata=dict(action_kind=kind,source_function=sources[kind],rows=[dict(
                    operation_key=f'step/{step}/slot/{slot}',logical_step=step,logical_slot=slot,
                    seeds={'rollout':step*16+slot+1}) for slot in slots])))
        for step in range(10):
            if g:
                for slot in slots:
                    add('generator_rollout',step,[slot])
                if step%5 == 0:
                    for _ in (0,1): add('fake_velocity',step,[])
                    for slot in slots: add('generator_backward',step,[slot])
            else:
                if step%5 == 0:
                    for _ in (0,1): add('teacher_velocity',step,[])
                add('critic_backward_b2',step,slots)
        keys = sorted({(s,w) for s in range(10) if s%5 for w in (0,1)} | {(5,0)})
        shadow_kinds = ('shadow_receiver_transfer_wait','shadow_target_load') if g else ('shadow_source_pack','shadow_source_completion_wait')
        host_kinds = ('sender_copy_ready_wait','gloo_send_completion') if g else ('gloo_receive_wait',)
        reuse_keys = sorted({(s,w) for s in (1,2,3,6,7,8) for w in (0,1)} | {(4,0)}) if g else []
        result.append(dict(status='PASS',rank=rank,source_head='a'*40,implementation='fixture',profiled=False,
            rows=rows,transport_observation=dict(status='PASS',rank=rank,independent_link_service_admitted=False,
                rows=[dict(kind='latent_d2h' if g else 'latent_h2d',key=list(k),status='PASS',
                           bytes=100,duration_ns=3) for k in keys] + [dict(kind=k,key=[5,0],status='PASS',
                           bytes=64,duration_ns=2) for k in shadow_kinds] + [dict(kind=kind,key=list(k),
                           status='PASS',bytes=100,duration_ns=1) for kind in host_kinds for k in keys]
                           + [dict(kind='streamer_reuse_copy_wait',key=list(k),status='PASS',bytes=100,duration_ns=1) for k in reuse_keys]),
            native_transport_contract=dict(runahead=dict(execution_status='PASS',buffer_count=17,
                payload_bytes=100,payload_shape=[25],payload_dtype='torch.float32',
                d2h_bytes=1700 if g else 0,h2d_bytes=0 if g else 1700),
                shadow=dict(execution_status='PASS',role='generator_receiver' if g else 'score_source',
                    payload_bytes=64,total_numel=16,parameter_count=2,dtype='torch.float32',layout_sha256='b'*64,
                    events=[dict(version=5,status='PASS')]))))
    return result


def test_action_bijection_versions_and_copies_reconcile_but_are_not_a_cost_model():
    from distillschedule.search.livewan_action_observation import validate_action_observations
    raws = action_fixture()
    # The actual cross-cycle lookahead is observed before entering cycle 2;
    # sample identity, not the outer collector counter, assigns its logical cycle.
    for row in raws[0]['rows'][1:]:
        if row['metadata']['action_kind'] == 'generator_rollout' and row['metadata']['rows'][0]['logical_step'] == 5:
            row['cycle'] = 1
    result = validate_action_observations(raws,expected_head='a'*40)
    assert result['rank_count'] == 16 and result['matched_rank_sample_identities'] == 160
    assert result['planner_costs_ready'] is False and result['formal_timing_eligible'] is False
    assert all(r['byte_residual'] == r['event_residual'] == 0 for r in result['rank_summaries'])
    g = [r for r in result['helpers'] if r['rank']==0 and r['cycle']==2 and r['family']=='generator_rollout']
    assert g[0]['count'] == 10


def test_request_qualified_actions_reject_old_actual_shape():
    from math import prod
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.livewan_implementation import LiveWanImplementation
    from distillschedule.search.livewan_action_observation import validate_action_observations
    req=LiveWanRequest(frames=69);raws=action_fixture();nbytes=prod(req.latent_shape)*4
    for raw in raws:
        raw.update(implementation=LiveWanImplementation('pipeline',True,True,True,request=req).id,
            workload_request=req.to_dict(),actual_latent_shapes=[list(req.latent_shape)],installed_plan_sha256='a'*64)
        for row in raw['rows'][1:]:row['metadata'].update(workload_request=req.to_dict(),latent_shape=list(req.latent_shape))
        transport=raw['native_transport_contract']['runahead']
        transport.update(payload_shape=list(req.latent_shape),payload_bytes=nbytes,
            d2h_bytes=nbytes*17 if raw['rank']<8 else 0,h2d_bytes=0 if raw['rank']<8 else nbytes*17)
        for row in raw['transport_observation']['rows']:
            if not row['kind'].startswith('shadow_'):row['bytes']=nbytes
    assert validate_action_observations(raws,expected_head='a'*40)['rank_count']==16
    raws[0]['actual_latent_shapes']=[[1,16,21,46,80]]
    with pytest.raises(ValueError,match='request|shape'):
        validate_action_observations(raws,expected_head='a'*40)


@pytest.mark.parametrize('failure', ['rank','source','version','sample','bytes','parent','helpers',
                                   'backward_rng','backward_duplicate','shadow_missing','shadow_layout','cycle_distribution',
                                   'host_missing','paired_bytes','arithmetic_lane'])
def test_bad_action_evidence_fail_closed(failure):
    from distillschedule.search.livewan_action_observation import validate_action_observations
    raws = action_fixture()
    if failure == 'rank': raws.pop()
    elif failure == 'source': raws[0]['source_head'] = 'b'*40
    elif failure == 'version': raws[0]['rows'][1]['runtime_before']['generator'] = 7
    elif failure == 'sample': raws[8]['rows'][3]['metadata']['rows'][0]['seeds']['rollout'] = 888
    elif failure == 'bytes': raws[0]['transport_observation']['rows'][0]['bytes'] = 101
    elif failure == 'parent': raws[0]['rows'][1]['parent'] = 888
    elif failure == 'backward_rng':
        r=next(r for r in raws[0]['rows'][1:] if r['metadata']['action_kind']=='generator_backward')
        r['metadata']['rows'][0]['seeds']['rollout']=444
    elif failure == 'backward_duplicate':
        r=[r for r in raws[0]['rows'][1:] if r['metadata']['action_kind']=='generator_backward']
        r[1]['metadata']['rows']=r[0]['metadata']['rows']
    elif failure == 'shadow_missing': raws[0]['native_transport_contract']['shadow']=None
    elif failure == 'shadow_layout': raws[0]['native_transport_contract']['shadow']['layout_sha256']='c'*64
    elif failure == 'cycle_distribution':
        r=next(r for r in raws[8]['rows'][1:] if r['metadata']['action_kind']=='teacher_velocity' and r['cycle']==2)
        r['cycle']=1
    elif failure == 'host_missing':
        raws[0]['transport_observation']['rows']=[r for r in raws[0]['transport_observation']['rows'] if r['kind']!='gloo_send_completion']
    elif failure == 'paired_bytes':
        c=raws[8]['native_transport_contract']['runahead'];c.update(payload_bytes=200,payload_shape=[50],h2d_bytes=3400)
        for r in raws[8]['transport_observation']['rows']:
            if r['kind'] in ('latent_h2d','gloo_receive_wait'): r['bytes']=200
    elif failure == 'arithmetic_lane':
        r=next(r for r in raws[0]['rows'][1:] if r['metadata']['rows'][0]['logical_slot']==4)
        r['metadata']['rows'][0]['logical_slot']=8
    else: raws[0]['rows'].pop()
    with pytest.raises(ValueError): validate_action_observations(raws,expected_head='a'*40)
