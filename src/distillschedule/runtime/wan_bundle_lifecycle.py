"""Explicit finite-stream lifecycle over the ORIGINAL Wan incumbent controllers.

No model/loss/update implementation lives here. v1 literal captures/census are
unchanged. v2 is research-only: at most 64 fresh cohorts, fixed original model
geometry, 12 rollout and 3 Critic signatures, captured in the first process
cycle only. Restoring native state starts with new empty controllers.
"""
from __future__ import annotations

from contextlib import contextmanager
import random

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.wan_incumbent_stream_lifecycle.v2'


def capture_storage_identity(model):
    """v2 static storage metadata only; never read or log tensor values."""
    return tuple((kind, name, value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                  value.storage_offset(), str(value.dtype), str(value.device))
                 for kind, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers()))
                 for name, value in values)


def validate_capture_transition(previous, *, version, logical_steps, source, increment):
    require(type(version) is int and version >= 0 and isinstance(logical_steps, tuple)
            and len(logical_steps) == 5 and all(type(s) is int and s >= 0 for s in logical_steps)
            and logical_steps == tuple(range(logical_steps[0], logical_steps[0]+5))
            and isinstance(source, str) and len(source) == 64 and set(source) <= set('0123456789abcdef'),
            'invalid stream capture version/sample/source binding', path='wan.bundle.stream')
    if previous is not None:
        require(source == previous['source'], 'capture source changed', path='wan.bundle.stream')
        require(version == previous['version'] + increment
                and logical_steps[0] == previous['logical_steps'][0] + 5,
                'capture stream must advance exactly one cohort/version', path='wan.bundle.stream')
    return dict(version=version, logical_steps=logical_steps, source=source)


@contextmanager
def capture_rng_guard(model):
    """Cold-only RNG audit, no random draws and no tensor-value telemetry."""
    import torch
    import numpy as np
    cpu, python, numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    cuda = {device: torch.cuda.get_rng_state(device) for device in devices}
    yield
    after = np.random.get_state()
    require(torch.equal(cpu, torch.get_rng_state()) and python == random.getstate()
            and numpy[0] == after[0] and np.array_equal(numpy[1], after[1]) and numpy[2:] == after[2:]
            and all(torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in cuda.items()),
            'cold graph capture consumed training RNG; stochastic capture is unsupported', path='wan.bundle.stream')


class WanBundleLifecycle:
    """Boundary/census checks, never a replacement trainer or saved tensor cache."""
    def __init__(self, session):
        from distillschedule.strategies.wan_planning import source_fingerprint
        self.session = session
        self.bundle = session.strategy.candidate_bundle
        require(self.bundle.continuous, 'explicit stream bundle required', path='wan.bundle.stream')
        self.source_fingerprint = source_fingerprint()
        self.source = fingerprint(dict(source=self.source_fingerprint, bundle=self.bundle.source_identity()))
        self.rollout = getattr(session.g.module, '_distillschedule_rollout_cuda_graph', None)
        self.critic = getattr(session.c.module, '_distillschedule_critic_grad_cuda_graph', None)
        self.cast_quiescence = getattr(session.g.module, '_distillschedule_cast_cache_quiescence', None)
        require(self.rollout is not None and self.critic is not None and callable(self.cast_quiescence),
                'complete actual rollout/Critic/cast controllers required', path='wan.bundle.stream')
        self.critic.version_reader = lambda: session.c.global_steps
        self.active = None
        self.receipt = dict(schema_version=SCHEMA, source=self.source, cycles=[], active=False,
                            capture_policy='FIRST_PROCESS_CYCLE_ONLY; NOVEL_LATER_SIGNATURE_REJECTED',
                            max_rollout_signatures=12, max_critic_signatures=3,
                            checkpoint_cache_policy='REBUILD_TRANSIENT_CAPTURES_AFTER_PERSISTENT_STATE_RESTORE')
        for role in ('generator', 'critic'):
            metadata = session.build[role]['metadata']
            require(metadata['candidate_bundle_sources'] == self.bundle.source_identity(),
                    'installed source differs from stream controller', path='wan.bundle.stream')
            metadata['bundle_lifecycle'] = self.receipt

    def quiescent(self):
        from distillschedule.training.native_state import EngineClocks
        require(not self.session.finite and not self.rollout.receipt['active']
                and self.critic.receipt['replay_calls'] == self.critic.receipt['backward_replay_calls'],
                'stream boundary has pending finite/backward/replay work', path='wan.bundle.stream')
        self.cast_quiescence()
        for engine in (self.session.g, self.session.c):
            EngineClocks(engine).validate_checkpoint_boundary()
            require(all(bucket.work is None or bucket.work.is_completed()
                        for bucket in getattr(engine, '_buckets_by_index', ())),
                    'stream boundary has pending native gradient communication', path='wan.bundle.stream')
        require(self.rollout._parameter_identity() == self.rollout._pointer_contract
                and self.critic.parameter_identity() == self.critic.pointer_contract,
                'stream controller parameter addresses changed', path='wan.bundle.stream')
        for controller in (self.rollout, self.critic):
            if controller._stream_storage is not None:
                require(capture_storage_identity(controller.source) == controller._stream_storage,
                        'stream static parameter/buffer storage changed', path='wan.bundle.stream')

    def begin(self, *, task, trace, document, index):
        from distillschedule.runtime.fsdp2_p0 import expected_trace_rollout_call_inventory
        from distillschedule.strategies.wan_planning import source_fingerprint
        require(self.active is None and source_fingerprint() == self.source_fingerprint
                and fingerprint(dict(source=self.source_fingerprint, bundle=self.bundle.source_identity())) == self.source,
                'active cycle or stale stream source', path='wan.bundle.stream')
        require(document['identity']['source_fingerprint'] == self.source_fingerprint,
                'stream source changed', path='wan.bundle.stream')
        self.quiescent()
        self.bundle.validate_task(task)
        before = self.session.versions()
        self.bundle.validate_versions(before)
        require(before['critic'] == task.first_step and trace.bundle_hash == document['cohorts'][index]['trace_hash'],
                'stream clock/trace does not match next input cohort', path='wan.bundle.stream')
        inventories = [expected_trace_rollout_call_inventory(trace.rank_control_validation,
            operation=f'step/{step}/critic/rollout', expected_rank_count=16) for step in task.source_steps]
        g, c = self.rollout.receipt, self.critic.receipt
        self.active = dict(cohort_index=index, stream_sha256=document['stream_sha256'], trace_hash=trace.bundle_hash,
            before=before, source_steps=list(task.source_steps), inventories=inventories,
            rollout_calls_before=g['graph_replay_model_call_count'], entries_before=len(g['entries']),
            critic_calls_before=c['replay_calls'], critic_backward_before=c['backward_replay_calls'])
        self.rollout.bind_stream_cycle(version=before['generator']+1, logical_steps=task.source_steps, source=self.source)
        self.critic.bind_stream_cycle(version=before['critic'], logical_steps=task.source_steps, source=self.source)
        self.receipt['active'] = True

    def finish(self, record):
        require(self.active is not None, 'stream cycle was not bound', path='wan.bundle.stream')
        self.quiescent()
        row = self.active
        g, c = self.rollout.receipt, self.critic.receipt
        expected = [dict(version=row['before']['generator']+1, logical_steps=[s]) for s in row['source_steps']]
        calls = sum(v['call_count'] for v in row['inventories'])
        require(record['call_census']['status'] == 'PASS_TRACE_DERIVED_CALL_CENSUS'
                and record['call_census']['inventories'][1:] == row['inventories']
                and g['graph_replay_model_call_count']-row['rollout_calls_before'] == calls
                and g['entries'][row['entries_before']:] == expected
                and c['replay_calls']-row['critic_calls_before'] == 5
                and c['backward_replay_calls']-row['critic_backward_before'] == 5,
                'stream graph census differs from actual fresh trace/update work', path='wan.bundle.stream')
        completed = {k: row[k] for k in ('cohort_index', 'stream_sha256', 'trace_hash', 'before', 'source_steps', 'inventories')}
        completed.update(after=self.session.versions(), rollout_calls=calls, rollout_entries=expected,
                         critic_versions=list(range(row['before']['critic'], row['before']['critic']+5)))
        self.receipt['cycles'].append(completed)
        self.receipt['active'] = False
        self.active = None
        # The original dispatcher has already released last-consumer rollouts.
        # Drop helper context/data caches too; Graph-owned static buffers stay
        # bounded in-process, and are never part of engine_state/checkpoints.
        self.session.runtime_cache.clear()
        record['bundle_lifecycle'] = completed

    def validate_checkpoint_boundary(self):
        require(self.active is None and not self.receipt['active'] and not self.session.runtime_cache,
                'checkpoint requires a completed/drained stream cycle', path='wan.bundle.stream')
        self.quiescent()


def validate_stream_receipts(build, *, cycles):
    """v2 census is derived from fresh trace rows, never the v1 fixed 27 calls."""
    g = build['generator']['metadata']['generator_rollout_cuda_graph']
    c = build['critic']['metadata']['critic_grad_cuda_graph']
    receipt = build['generator']['metadata'].get('bundle_lifecycle', {})
    rows = receipt.get('cycles', [])
    require(receipt.get('schema_version') == SCHEMA and receipt.get('active') is False
            and receipt == build['critic']['metadata'].get('bundle_lifecycle')
            and type(cycles) is int and 1 <= cycles <= 64 and len(rows) == cycles,
            'missing/incomplete stream lifecycle census', path='wan.bundle.stream')
    entries, critic_versions = [], []
    for i, row in enumerate(rows):
        before, after = row['before'], row['after']
        require(after == dict(generator=before['generator']+1, critic=before['critic']+5, ema=before['ema']+5)
                and row['source_steps'] == list(range(before['critic'], before['critic']+5))
                and len(row['inventories']) == 5
                and row['rollout_calls'] == sum(v['call_count'] for v in row['inventories'])
                and row['rollout_entries'] == [dict(version=after['generator'], logical_steps=[s]) for s in row['source_steps']]
                and row['critic_versions'] == row['source_steps'], 'stream sample/version census changed', path='wan.bundle.stream')
        if i:
            previous = rows[i-1]
            require(row['before'] == previous['after'] and row['cohort_index'] == previous['cohort_index']+1
                    and row['stream_sha256'] == previous['stream_sha256'] and row['trace_hash'] != previous['trace_hash'],
                    'stream repeated/skipped input cohort', path='wan.bundle.stream')
        entries.extend(row['rollout_entries']); critic_versions.extend(row['critic_versions'])
    calls = sum(row['rollout_calls'] for row in rows)
    require(all(g.get(k) == v for k, v in dict(active=False, model_call_count=calls,
            graph_replay_model_call_count=calls, rollout_entry_count=5*cycles, entries=entries,
            versions=[row['after']['generator'] for row in rows], diagnostic_reference_model_call_count=0,
            diagnostic_reference_entry_count=0, weights_merged=False, optimizer_reducer_ema_unchanged=True).items()),
            'rollout stream graph census/contract mismatch', path='wan.bundle.stream')
    require(all(c.get(k) == v for k, v in dict(replay_calls=5*cycles, backward_replay_calls=5*cycles,
            reference_calls=0, weights_merged=False, optimizer_reducer_ema_unchanged=True,
            captured_parameter_gradients_return_to_original_accumulators=True).items()),
            'Critic stream graph census/contract mismatch', path='wan.bundle.stream')
    gn, cn = g.get('cache_entry_count', 0), c.get('capture_entries', 0)
    require(3 <= gn <= 12 and 1 <= cn <= 3 and g['capture_model_call_count'] == gn
            and g['warmup_model_call_count'] == 3*gn and c['cold_original_forward_calls'] == 4*cn
            and c['cold_original_backward_calls'] == 4*cn
            and {tuple(r['latent_spatial_shape']) for r in g['cache_entries']} == {(12,20),(24,40),(48,80)},
            'bounded stream cold capture inventory changed', path='wan.bundle.stream')
    require(g.get('stream_lifecycle') == dict(source=receipt['source'], first_version=rows[0]['after']['generator'],
                                              rng_checked_captures=gn)
            and c.get('stream_lifecycle') == dict(source=receipt['source'], first_version=rows[0]['before']['critic'],
                                                 rng_checked_captures=cn, versions=critic_versions),
            'stream capture source/version/RNG binding changed', path='wan.bundle.stream')
    return receipt
