"""Opt-in observations on the actual LiveWan transport streams/threads.

These are evidence, not an automatically admitted link cost model. Gloo
completion includes host/runtime/peer waits; CUDA durations include dependencies
already on the observed stream. No cross-node clock alignment is invented.
The owning native runtime must finish its ordinary teardown/synchronization
before calling finish(). This module never synchronizes or copies a tensor.
"""
from contextlib import contextmanager
from time import perf_counter_ns
import threading
from collections import Counter
from math import prod
from statistics import median

from .contracts import require
from distillschedule.workloads.livewan_crossscale import TRACE_BALANCED_8G8S_SLOT_PAIRS


class TransportObserver:
    def __init__(self, *, torch, device, rank, clock=perf_counter_ns):
        self.torch, self.device, self.rank, self.clock = torch, device, rank, clock
        self.rows, self.active, self.finished = [], 0, False
        self.lock = threading.Lock()

    def _begin(self, kind, key, nbytes, clock):
        require(isinstance(kind, str) and kind, 'missing operation kind')
        require(len(key) == 2 and all(type(v) is int and v >= 0 for v in key), 'invalid message identity')
        require(type(nbytes) is int and nbytes >= 0, 'invalid transport byte count')
        with self.lock:
            require(not self.finished, 'observer already finalized')
            row = dict(index=len(self.rows), rank=self.rank, kind=kind, key=list(key),
                       bytes=nbytes, clock=clock, thread_id=threading.get_ident(),
                       host_start_ns=self.clock(), status='RUNNING')
            self.rows.append(row)
            self.active += 1
        return row

    def _end(self, row):
        row['host_stop_ns'] = self.clock()
        with self.lock:
            self.active -= 1

    def _event(self, stream):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    @contextmanager
    def gpu_range(self, kind, *, key, nbytes, stream=None):
        stream = self.torch.cuda.current_stream(self.device) if stream is None else stream
        row = self._begin(kind, key, nbytes, 'CUDA_EXPLICIT_STREAM_DURATION_ONLY')
        try:
            row['stream'] = str(getattr(stream, 'cuda_stream', stream))
            row['start_event'] = self._event(stream)
            yield
            row['status'] = 'PASS'
        except BaseException as error:
            row.update(status='FAIL', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                row['stop_event'] = self._event(stream)
            finally:
                self._end(row)

    @contextmanager
    def host_range(self, kind, *, key, nbytes):
        row = self._begin(kind, key, nbytes, 'HOST_MONOTONIC_LOCAL_PROCESS')
        try:
            yield
            row['status'] = 'PASS'
        except BaseException as error:
            row.update(status='FAIL', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self._end(row)

    def finish(self):
        with self.lock:
            require(not self.finished and not self.active, 'unfinished/already finalized transport observer')
            self.finished = True
        rows = []
        for raw in self.rows:
            row = {k:v for k,v in raw.items() if k not in ('start_event','stop_event')}
            if row['clock'] == 'CUDA_EXPLICIT_STREAM_DURATION_ONLY':
                require(raw['start_event'].query() and raw['stop_event'].query(),
                        'native terminal synchronization has not completed')
                row['duration_ns'] = round(raw['start_event'].elapsed_time(raw['stop_event']) * 1_000_000)
            else:
                row['duration_ns'] = row['host_stop_ns'] - row['host_start_ns']
            require(row['duration_ns'] >= 0, 'negative transport interval')
            rows.append(row)
        return dict(schema_version='distillschedule.livewan_transport_observation.v1',
            status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
            rank=self.rank, rows=rows, extra_synchronizations=0, extra_collectives=0,
            tensor_value_reads=0, calibration_only=True, formal_timing_eligible=False,
            independent_link_service_admitted=False,
            cross_rank_clock_alignment='NOT_ESTABLISHED',
            duration_scope='CONTEXTUAL_OBSERVATIONS_NOT_ISOLATED_LINK_SERVICE',
            bytes_scope='COUNT_ONCE_PER_ACTUAL_COPY; DO_NOT_ADD_WAIT_AND_SEND_PHASE_BYTES')


def validate_action_observations(raws, *, expected_head):
    """Reconcile new helper/message observations with the native B16 protocol.

    This gate does not turn a CUDA stream envelope into isolated service time,
    supply missing workspace/lifetime evidence or claim to execute a new DAG.
    Old coupled-commit receipts intentionally do not satisfy this schema.
    """
    require(len(raws) == 16 and {r['rank'] for r in raws} == set(range(16)), 'sixteen native ranks required')
    require(len({r['implementation'] for r in raws}) == 1, 'mixed implementations')
    helper_sources = dict(generator_rollout='prepare_rollout', fake_velocity='_dmd_fake_velocity',
        teacher_velocity='_dmd_teacher_velocity', generator_backward='generator_backward_for_sample',
        critic_backward_b2='critic_backward_for_batch')
    counts = {True:dict(generator_rollout=20, fake_velocity=4, generator_backward=4),
              False:dict(teacher_velocity=4, critic_backward_b2=10)}
    groups, produced, consumed, backward, summaries = {}, {}, {}, {}, []
    transport_contracts = {}
    message_keys = {(s,w) for s in range(10) if s % 5 for w in (0,1)} | {(5,0)}
    reuse_keys = {(s,w) for s in (1,2,3,6,7,8) for w in (0,1)} | {(4,0)}
    for raw in sorted(raws, key=lambda r:r['rank']):
        rank, is_g = raw['rank'], raw['rank'] < 8
        require(raw.get('status') == 'PASS' and raw.get('source_head') == expected_head
                and raw.get('profiled') is False, 'native action source/status differs')
        rows = raw['rows']
        helpers = [r for r in rows if r.get('metadata',{}).get('action_kind') in helper_sources]
        require(Counter(r['metadata']['action_kind'] for r in helpers) == counts[is_g], 'native helper inventory incomplete')
        require(all(r.get('status') == 'PASS' for r in rows), 'failed native action')
        by_id = {r['index']:r for r in rows}
        require(len(by_id) == len(rows), 'duplicate native action index')
        for row in helpers:
            m, family = row['metadata'], row['metadata']['action_kind']
            require(m['source_function'] == helper_sources[family], 'native helper owner changed')
            parent = by_id.get(row['parent'])
            require(parent is not None and parent['start_ns'] <= row['start_ns'] <= row['stop_ns'] <= parent['stop_ns'],
                    'native helper has no enclosing source range')
            require(row.get('runtime_before') is not None and row['runtime_before'] == row.get('runtime_after'),
                    'helper version snapshot missing or mutable')
            if family in ('generator_rollout','generator_backward','critic_backward_b2'):
                samples = m['rows']
                require(len(samples) == (2 if family == 'critic_backward_b2' else 1), 'physical B1/B2 changed')
                step = samples[0]['logical_step']
                require(step in range(10) and all(s['logical_step'] == step and s.get('seeds') for s in samples),
                        'operation step/RNG identity missing')
                expected_version = step if family == 'critic_backward_b2' else step // 5 + int(step % 5 != 0)
                role = 'critic' if family == 'critic_backward_b2' else 'generator'
                require(row['runtime_before'][role] == expected_version, 'operation parameter version differs')
                if family == 'generator_backward':
                    require(step % 5 == 0, 'G backward outside update step')
                target = backward if family == 'generator_backward' else produced if is_g else consumed
                for sample in samples:
                    require(sample['logical_slot'] in TRACE_BALANCED_8G8S_SLOT_PAIRS[rank % 8],
                            'sample belongs to another trace-balanced lane')
                    key = (step, sample['logical_slot'])
                    require(key not in target, 'duplicate rank-qualified sample')
                    target[key] = dict(rank=rank, operation_key=sample['operation_key'], seeds=sample['seeds'])
                logical_cycle = step // 5 + 1
            else:
                logical_cycle = row['cycle']
                require(logical_cycle in (1,2), 'missing logical cycle')
                if family == 'fake_velocity':
                    require((row['runtime_before']['shadow_loaded_version'] or 0) == 5*(logical_cycle-1),
                            'stale Critic shadow')
            groups.setdefault((rank,logical_cycle,family),[]).append(row['duration_ns'])
        for cycle in (1,2):
            require(all(len(groups.get((rank,cycle,f),[])) == n // 2 for f,n in counts[is_g].items()),
                    'native per-cycle helper inventory differs')
        transport = raw.get('transport_observation',{})
        require(transport.get('status') == 'PASS' and transport.get('rank') == rank
                and transport.get('independent_link_service_admitted') is False,
                'native transport observations missing or falsely admitted')
        contract = raw['native_transport_contract']['runahead']
        require(contract.get('execution_status') == 'PASS' and contract.get('buffer_count') == len(message_keys),
                'transport runtime census incomplete')
        shape = contract.get('payload_shape',[])
        require(shape and all(type(v) is int and v > 0 for v in shape)
                and contract.get('payload_dtype') == 'torch.float32'
                and prod(shape) * 4 == contract.get('payload_bytes'), 'latent payload contract differs')
        shadow = raw['native_transport_contract'].get('shadow')
        require(isinstance(shadow,dict) and shadow.get('execution_status') == 'PASS', 'shadow contract missing')
        require(shadow.get('role') == ('generator_receiver' if is_g else 'score_source')
                and shadow.get('dtype') == 'torch.float32'
                and type(shadow.get('total_numel')) is int and shadow['total_numel'] > 0
                and shadow['total_numel'] * 4 == shadow.get('payload_bytes')
                and type(shadow.get('parameter_count')) is int and shadow['parameter_count'] > 0
                and isinstance(shadow.get('layout_sha256'),str) and len(shadow['layout_sha256']) == 64,
                'shadow payload/layout contract differs')
        require(len(shadow.get('events',[])) == 1 and shadow['events'][0].get('version') == 5
                and shadow['events'][0].get('status') == 'PASS', 'shadow version completion inventory differs')
        transport_contracts[rank] = (contract,shadow)
        expected_phases = ({'latent_d2h': message_keys, 'sender_copy_ready_wait': message_keys,
                            'gloo_send_completion': message_keys, 'streamer_reuse_copy_wait': reuse_keys,
                            'shadow_receiver_transfer_wait': {(5,0)}, 'shadow_target_load': {(5,0)}}
                           if is_g else
                           {'latent_h2d': message_keys, 'gloo_receive_wait': message_keys,
                            'shadow_source_pack': {(5,0)}, 'shadow_source_completion_wait': {(5,0)}})
        observed_phases = Counter(r['kind'] for r in transport['rows'])
        require(observed_phases == Counter({k:len(v) for k,v in expected_phases.items() if v}),
                'native transport phase inventory differs')
        for phase,keys in expected_phases.items():
            observations = [r for r in transport['rows'] if r['kind'] == phase]
            require({tuple(r['key']) for r in observations} == keys, 'native transport phase keys differ')
            nbytes = shadow['payload_bytes'] if phase.startswith('shadow_') else contract['payload_bytes']
            require(all(r.get('status') == 'PASS' and r['bytes'] == nbytes
                        and type(r.get('duration_ns')) is int and r['duration_ns'] >= 0 for r in observations),
                    'native transport phase status/bytes/duration differs')
        kind = 'latent_d2h' if is_g else 'latent_h2d'
        copies = [r for r in transport['rows'] if r['kind'] == kind]
        require(len(copies) == len(message_keys) and {tuple(r['key']) for r in copies} == message_keys,
                'actual transport copy keys differ')
        require(all(r['status'] == 'PASS' and r['bytes'] == contract['payload_bytes'] for r in copies),
                'actual transport bytes/status differ')
        total = sum(r['bytes'] for r in copies)
        require(total == contract['d2h_bytes' if is_g else 'h2d_bytes'], 'transport byte reconciliation failed')
        summaries.append(dict(rank=rank,helper_counts=counts[is_g],copy_kind=kind,
            copy_count=len(copies),copy_bytes=total,copy_sum_duration_ns=sum(r['duration_ns'] for r in copies),
            byte_residual=0,event_residual=0,reconciliation_scope='ACTUAL_LATENT_COPY_ONLY',
            transport_phase_counts=dict(observed_phases),shadow_event_residual=0,host_event_residual=0))
    require(set(produced) == set(consumed) == {(s,i) for s in range(10) for i in range(16)},
            'global batch16 / two-cycle sample bijection changed')
    require(all(produced[k]['operation_key'] == consumed[k]['operation_key']
                and produced[k]['seeds'] == consumed[k]['seeds']
                and consumed[k]['rank'] == produced[k]['rank'] + 8 for k in produced),
            'producer/consumer message identity or route changed')
    require(set(backward) == {(s,i) for s in (0,5) for i in range(16)}
            and all(backward[k] == produced[k] for k in backward), 'G backward sample/RNG differs from rollout')
    for rank in range(8):
        sender, g_shadow = transport_contracts[rank]
        receiver, c_shadow = transport_contracts[rank+8]
        require(all(sender[k] == receiver[k] for k in ('payload_shape','payload_dtype','payload_bytes')),
                'paired latent payload differs')
        require(sender['d2h_bytes'] == receiver['h2d_bytes'] and sender['h2d_bytes'] == receiver['d2h_bytes'] == 0,
                'paired latent transfer byte reconciliation failed')
        require(all(g_shadow[k] == c_shadow[k] for k in
                    ('payload_bytes','total_numel','parameter_count','dtype','layout_sha256')),
                'paired shadow layout/payload differs')
    return dict(status='PASS_NATIVE_ACTION_OBSERVATIONS', source_head=expected_head,
        rank_count=16, rank_summaries=summaries,
        helpers=[dict(rank=r,cycle=c,family=f,count=len(v),sum_duration_ns=sum(v),
                      median_duration_ns=median(v)) for (r,c,f),v in sorted(groups.items())],
        matched_rank_sample_identities=len(produced), formal_timing_eligible=False,
        planner_costs_ready=False, independent_link_service_admitted=False,
        remaining=['WORKSPACE_AND_OUTPUT_LIFETIMES','CONTEXTUAL_SERVICE_COST_ADMISSION',
                   'EXECUTABLE_ACTION_DAG_LOWERING_AND_HELDOUT_NATIVE_FEEDBACK'])
