"""Fresh original-protocol cohorts, with persistent models and optimizer state.

A selected *policy* may be reused at identical geometry. Timing/capacity
predictions for the calibration trace are NOT asserted for new random exits.
This does not resume a previous training run or adapt layout mid-backward.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json
from .wan_inputs import WanInputSources
from .wan_task import OriginalWanTask, NATIVE_HISTORY_POLICY

SCHEMA = 'distillschedule.original_wan_cohort_stream.v1'


def task_at(task, index):
    require(type(index) is int and index >= 0, 'invalid cohort index', path='wan.stream')
    return replace(task, first_step=task.first_step + index * task.fake_updates)


@dataclass(frozen=True)
class WanCohortSources:
    cohorts: tuple[WanInputSources, ...]

    def __post_init__(self):
        require(2 <= len(self.cohorts) <= 64 and all(isinstance(s, WanInputSources) for s in self.cohorts),
                'two to 64 explicit cohort source contracts required', path='wan.stream')
        require(len({(s.seed, s.cohort) for s in self.cohorts}) == len(self.cohorts),
                'cohorts must have distinct logical RNG identities; no fixed-trace repetition', path='wan.stream')

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == {'cohorts'}
                and isinstance(value['cohorts'], list), 'explicit cohort sequence required', path='wan.stream')
        return cls(tuple(WanInputSources.from_dict(s) for s in value['cohorts']))

    def to_dict(self):
        return {'cohorts': [s.to_dict() for s in self.cohorts]}


def prepare_stream(task, sources, training_config, output, *, execute=False, resume=False):
    from .wan_capture import capture, read_training_config
    from .wan_inputs import preflight_sources
    from .wan_planning import source_fingerprint
    from .wan_prepare import publish_new_json
    from .wan_trace import load_original_trace
    require(isinstance(task, OriginalWanTask) and isinstance(sources, WanCohortSources),
            'typed original task/cohorts required', path='wan.stream')
    read_training_config(training_config)
    inventory = [preflight_sources(task_at(task, i), s) for i, s in enumerate(sources.cohorts)]
    identity = {'task': task.to_dict(), 'sources': sources.to_dict(), 'inventory': inventory,
                'training_config_sha256': file_sha256(Path(training_config)),
                'source_fingerprint': source_fingerprint()}
    key = fingerprint(identity)
    if not execute:
        return {'status': 'VALIDATED_COHORTS_NOT_CAPTURED', 'identity': identity,
                'stream_input_sha256': key, 'gpu_execution': 'NOT_RUN'}
    root = Path(output).resolve()
    if root.exists():
        require(resume, 'stream exists; explicit resume required', path='wan.stream')
        require(json.loads((root/'inputs.json').read_text()) == identity,
                'cohort inputs changed; do not reset or repeat training', path='wan.stream')
    else:
        root.mkdir(parents=True)
        write_new_json(root/'inputs.json', identity)
    with (root/'capture.lock').open('x') as handle:
        handle.write(key+'\n')
    try:
        entries = []
        for i, source in enumerate(sources.cohorts):
            current = task_at(task, i)
            path = root/f'cohort-{i:04d}'
            if (path/'capture.json').exists():
                record = json.loads((path/'capture.json').read_text())
            else:
                require(not path.exists(), 'incomplete cohort preserved; never silently resample', path='wan.stream')
                record = capture(current, source, training_config, path)
            require(record['status'] == 'PASS_CAPTURED_ORIGINAL_INPUTS'
                    and record['task'] == current.to_dict() and record['input_inventory'] == inventory[i]
                    and record['source_fingerprint'] == identity['source_fingerprint']
                    and record['training_config_sha256'] == identity['training_config_sha256'],
                    'cohort capture identity changed', path='wan.stream')
            load_original_trace(current, path, expected_hash=record['expected_trace_hash'])
            entries.append({'index': i, 'task': current.to_dict(), 'trace_bundle': str(path),
                            'trace_hash': record['expected_trace_hash'],
                            'capture_sha256': file_sha256(path/'capture.json')})
        require(source_fingerprint() == identity['source_fingerprint']
                and file_sha256(Path(training_config)) == identity['training_config_sha256']
                and all(s.inventory()['digest'] == old['digest'] for s, old in zip(sources.cohorts, inventory)),
                'inputs changed during stream capture', path='wan.stream')
        document = {'schema_version': SCHEMA, 'task': task.to_dict(), 'identity': identity,
                    'cohorts': entries, 'status': 'READY_FRESH_COHORT_STREAM',
                    'gpu_execution': 'NOT_RUN', 'claimable_speedup': False}
        document['stream_sha256'] = fingerprint(document)
        path = root/'stream.json'
        if path.exists():
            require(json.loads(path.read_text()) == document, 'stream manifest changed', path='wan.stream')
        else:
            publish_new_json(path, document)
        return document
    finally:
        (root/'capture.lock').unlink()


def load_stream(task, path):
    """Validate each complete cohort before any distributed training starts."""
    from .wan_planning import source_fingerprint
    from .wan_trace import load_original_trace
    path = Path(path).resolve()
    document = json.loads(path.read_text()); unsigned = dict(document)
    recorded = unsigned.pop('stream_sha256', None)
    require(document.get('schema_version') == SCHEMA and recorded == fingerprint(unsigned)
            and document.get('status') == 'READY_FRESH_COHORT_STREAM' and document.get('task') == task.to_dict(),
            'changed stream manifest/task', path='wan.stream')
    identity = document['identity']
    require(identity['task'] == task.to_dict() and identity['source_fingerprint'] == source_fingerprint(),
            'stream source/task changed', path='wan.stream')
    sources = WanCohortSources.from_dict(identity['sources'])
    entries = document['cohorts']
    require(len(entries) == len(sources.cohorts) == len(identity['inventory']), 'incomplete cohort stream', path='wan.stream')
    for i, entry in enumerate(entries):
        current = task_at(task, i)
        root = Path(entry['trace_bundle']).resolve()
        require(root == path.parent/f'cohort-{i:04d}' and entry['index'] == i
                and entry['task'] == current.to_dict(), 'cohort order, geometry or source steps changed', path='wan.stream')
        require(file_sha256(root/'capture.json') == entry['capture_sha256'], 'capture record changed', path='wan.stream')
        capture = json.loads((root/'capture.json').read_text())
        require(capture['task'] == current.to_dict() and capture['expected_trace_hash'] == entry['trace_hash']
                and capture['input_inventory'] == identity['inventory'][i]
                and capture['source_fingerprint'] == identity['source_fingerprint']
                and capture['training_config_sha256'] == identity['training_config_sha256'],
                'cohort does not match original captured inputs', path='wan.stream')
        load_original_trace(current, root, expected_hash=entry['trace_hash'])
    return document


def order_for_cohort(selected_order, task):
    """Keep selected G/R/C order; recompute EMA positions from absolute steps."""
    core = tuple(k for k in selected_order if not k.startswith('e'))
    required = {'g', *(f'r{i}' for i in range(task.fake_updates)), *(f'c{i}' for i in range(task.fake_updates))}
    require(len(core) == len(required) and set(core) == required, 'selected policy has wrong logical work', path='wan.stream')
    order = []
    for key in core:
        order.append(key)
        if key.startswith('c') and int(key[1:]) in task.ema_after:
            order.append('e'+key[1:])
    return tuple(order)


def bind_cohort(session, base_task, document, index, selected_order):
    """Called only between completed cycles; never constructs or resets engines."""
    from .wan_trace import load_original_trace, validate_loaded_history
    require(not session.finite, 'unconsumed backward/finite state at cohort switch', path='wan.stream')
    previous = getattr(session, '_stream_binding', None)
    require((previous is None and index == 0 and session.task == base_task) or
            (previous == (document['stream_sha256'], index-1) and index > 0
             and session.task == task_at(base_task, index-1)),
            'cohort transition must advance exactly once without changing task/stream', path='wan.stream')
    entry = document['cohorts'][index]; current = task_at(base_task, index)
    require(entry['task'] == current.to_dict(), 'stream task changed', path='wan.stream')
    trace = load_original_trace(current, entry['trace_bundle'], expected_hash=entry['trace_hash'])
    if current.stage_schedule_policy == NATIVE_HISTORY_POLICY:
        validate_loaded_history(current, trace._load(session.rank))
    session.strategy.validate_for(current)
    lifecycle = getattr(session, 'bundle_lifecycle', None)
    if lifecycle is not None:
        lifecycle.begin(task=current, trace=trace, document=document, index=index)
    session.task, session.trace = current, trace
    session._stream_binding = (document['stream_sha256'], index)
    if session.runtime_cache is not None:
        session.runtime_cache.clear()
    return {'operation_order': order_for_cohort(selected_order, current),
            'receipt': {'stream_sha256': document['stream_sha256'], 'cohort_index': index,
                        'trace_hash': entry['trace_hash'], 'source_steps': list(current.source_steps),
                        'prediction_transfer': 'NOT_CLAIMED_NEW_RANDOM_EXITS'}}


def restore_stream_cursor(session, base_task, document, *, completed_cycles):
    """Restore only the next input binding after native state/RNG restoration.

    No input is replayed, optimizer rebuilt, RNG consumed or CUDA cache restored.
    The caller has already validated the complete source-bound checkpoint.
    """
    require(type(completed_cycles) is int and 0 < completed_cycles < len(document['cohorts'])
            and session.task == base_task and not session.finite
            and getattr(session, '_stream_binding', None) is None,
            'resume needs a fresh session and an incomplete stream cursor', path='wan.stream')
    previous = task_at(base_task, completed_cycles - 1)
    require(document['cohorts'][completed_cycles-1]['task'] == previous.to_dict(),
            'resume stream task changed', path='wan.stream')
    session.task = previous
    session._stream_binding = (document['stream_sha256'], completed_cycles - 1)
    if session.runtime_cache is not None:
        session.runtime_cache.clear()
