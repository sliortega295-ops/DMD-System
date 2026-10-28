"""Cohort lifecycle tests; fixture trace IO does not stand for GPU execution."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from distillschedule.strategies.wan_stream import (
    WanCohortSources, prepare_stream, load_stream, task_at, order_for_cohort, bind_cohort)
from distillschedule.strategies.wan_inputs import WanInputSources
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, write_new_json


def test_absolute_steps_drive_ema_cadence_not_repeated_first_cycle():
    task = OriginalWanTask(fake_updates=3, first_step=6, ema_every=2)
    assert task_at(task, 1).source_steps == (9, 10, 11)
    assert task.ema_after == (1,) and task_at(task, 1).ema_after == (0, 2)
    chosen = ('g', 'r0', 'r1', 'r2', 'c0', 'c1', 'e1', 'c2')
    assert order_for_cohort(chosen, task_at(task, 1)) == (
        'g', 'r0', 'r1', 'r2', 'c0', 'e0', 'c1', 'c2', 'e2')
    with pytest.raises(ValueError, match='wrong logical work'):
        order_for_cohort(chosen, replace(task, fake_updates=1))


def test_no_cohort_rng_repetition_or_single_fixed_trace():
    source = WanInputSources(tuple(f'{i}.pt' for i in range(16)))
    with pytest.raises(ValueError, match='two to 100'):
        WanCohortSources((source,))
    with pytest.raises(ValueError, match='distinct logical RNG'):
        WanCohortSources((source, source))
    values = WanCohortSources((source, replace(source, cohort=1)))
    assert WanCohortSources.from_dict(values.to_dict()).to_dict() == values.to_dict()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import distillschedule.strategies.wan_capture as cap
    import distillschedule.strategies.wan_inputs as inp
    import distillschedule.strategies.wan_trace as trace
    task = OriginalWanTask(fake_updates=3, first_step=6, ema_every=2)
    source = WanInputSources(tuple(str(tmp_path/f'{i}.pt') for i in range(16)))
    cohorts = WanCohortSources((source, replace(source, cohort=1)))
    config = tmp_path/'config.yaml'; config.write_text('input-only-fixture')
    calls = []
    def inventory(s):
        return {'digest': fingerprint(s.to_dict()), 'sources': s.to_dict(), 'files': []}
    monkeypatch.setattr(inp.WanInputSources, 'inventory', inventory)
    def preflight(t, s):
        return {**inventory(s), 'task_sha256': t.digest}
    monkeypatch.setattr(inp, 'preflight_sources', preflight)
    monkeypatch.setattr(cap, 'read_training_config', lambda p: {})
    def capture(t, s, cfg, output):
        from distillschedule.strategies.wan_planning import source_fingerprint
        output = Path(output); output.mkdir()
        record = {'status': 'PASS_CAPTURED_ORIGINAL_INPUTS', 'task': t.to_dict(),
                  'input_inventory': preflight(t, s), 'source_fingerprint': source_fingerprint(),
                  'training_config_sha256': file_sha256(cfg),
                  'expected_trace_hash': fingerprint({'task': t.to_dict(), 'sources': s.to_dict()})}
        write_new_json(output/'capture.json', record)
        calls.append((t, s))
        return record
    monkeypatch.setattr(cap, 'capture', capture)
    monkeypatch.setattr(trace, 'load_original_trace', lambda t, p, **kw: NS(task=t, bundle_hash=kw['expected_hash']))
    return task, cohorts, config, calls


def test_prepare_reload_and_resume_preserve_distinct_cohorts(tmp_path, fixture):
    task, cohorts, config, calls = fixture
    root = tmp_path/'stream'
    assert prepare_stream(task, cohorts, config, root)['gpu_execution'] == 'NOT_RUN'
    assert not root.exists() and not calls
    ready = prepare_stream(task, cohorts, config, root, execute=True)
    assert len(calls) == 2 and calls[1][0].source_steps == (9, 10, 11)
    assert load_stream(task, root/'stream.json') == ready
    assert prepare_stream(task, cohorts, config, root, execute=True, resume=True) == ready
    assert len(calls) == 2
    assert ready['cohorts'][0]['trace_hash'] != ready['cohorts'][1]['trace_hash']
    with pytest.raises(ValueError, match='explicit resume'):
        prepare_stream(task, cohorts, config, root, execute=True)


@pytest.mark.parametrize('mutation', ['order', 'geometry', 'capture'])
def test_stream_rejects_changed_cohorts_even_if_manifest_rehashed(tmp_path, fixture, mutation):
    task, cohorts, config, calls = fixture
    root = tmp_path/'stream'
    doc = prepare_stream(task, cohorts, config, root, execute=True)
    if mutation == 'order': doc['cohorts'].reverse()
    elif mutation == 'geometry': doc['cohorts'][1]['task']['video_height'] *= 2
    else:
        (root/'cohort-0001'/'capture.json').write_text('{}')
    doc.pop('stream_sha256'); doc['stream_sha256'] = fingerprint(doc)
    (root/'stream.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError): load_stream(task, root/'stream.json')


def test_bind_next_cohort_keeps_engines_and_state(tmp_path, fixture):
    task, cohorts, config, calls = fixture
    root = tmp_path/'stream'; doc = prepare_stream(task, cohorts, config, root, execute=True)
    objects = [object(), object(), object()]
    session = NS(task=task, trace=None, finite=[], runtime_cache={'old_input': object()},
                 strategy=NS(validate_for=lambda t: None), g=objects[0], c=objects[1], ema=objects[2])
    chosen = ('g', 'r0', 'r1', 'r2', 'c0', 'c1', 'e1', 'c2')
    bind_cohort(session, task, doc, 0, chosen)
    event = bind_cohort(session, task, doc, 1, chosen)
    assert [session.g, session.c, session.ema] == objects
    assert session.task.first_step == 9 and not session.runtime_cache
    assert event['operation_order'][-1] == 'e2'
    with pytest.raises(ValueError, match='advance exactly once'):
        bind_cohort(session, task, doc, 1, chosen)
    session.finite.append(False)
    with pytest.raises(ValueError, match='unconsumed'):
        bind_cohort(session, task, doc, 0, chosen)
