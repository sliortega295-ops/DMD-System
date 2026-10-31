"""Private shared planning on one real CPU Adam owner; no GPU claims."""
from dataclasses import replace
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time

import pytest
import torch

from distillschedule.open_workload import planning
from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.resident_trial import ResidentTrial, RollbackError
from distillschedule.open_workload.torch_backend import Session, signature
from distillschedule.training.checkpoint import rng_state
from distillschedule.workloads.best_known.registry import write_new_json
from test_resident_trial import TrialOwner, build, budget, real_reference


class ScopedOwner(TrialOwner):
    """Exercise the same active-descriptor identity boundary as native CUDA."""
    def execution_options(self):
        result = super().execution_options()
        active = getattr(self, '_resident_trial_active', None)
        admitted = type(active) is ResidentTrial and active._capacity_admitted
        result['identity']['resident_scope'] = admitted
        if not admitted: result['options'] = [result['baseline']]
        return result

    def run(self, x):
        # Real measured CPU fixture latency makes the exercised branch win the
        # real DP search. No cost record or saved plan is fabricated or edited.
        if not self.enabled: time.sleep(.025)
        return super().run(x)


def snapshot(workload):
    owner = workload.state['owner']
    return dict(state=signature(owner), feeds=signature(workload.feeds),
        outer=signature(rng_state(device='cpu')),
        parameters=[(id(p), p.data_ptr()) for p in owner.model.parameters()],
        optimizers=[id(opt) for opt in owner.optimizers.values()],
        slots=[(id(t), t.data_ptr()) for opt in owner.optimizers.values()
               for row in opt.state.values() for t in row.values()])


def measured_plan(workload, tmp_path):
    manifest = planning.calibrate(workload, tmp_path/'costs', samples=1,
        max_candidates=2, max_program_runs=3, resident_budget=budget())
    document = planning.plan(tmp_path/'costs/calibration.json', memory_limit_bytes=8_000_000)
    write_new_json(tmp_path/'plan.json', document)
    return manifest, document


@pytest.mark.parametrize('warmed', [False, True])
def test_resident_calibration_shared_search_saved_run_preserves_then_adopts(tmp_path, warmed):
    workload = build(ScopedOwner()); owner = workload.state['owner']
    if warmed: real_reference(workload)
    before = snapshot(workload); initial_clock = owner.clock
    manifest, document = measured_plan(workload, tmp_path)
    assert snapshot(workload) == before
    assert manifest['budget']['actual'] == 3 and len(manifest['variants']) == 2
    assert manifest['resident_trial']['closed'] and not manifest['resident_trial']['adopted']
    assert document['resident_budget'] == manifest['resident_budget']
    assert manifest['identity'] != ExecutionBackend(workload).identity(), 'descriptor scope must be real'
    assert json.loads((tmp_path/'costs/calibration.json').read_text()) == manifest
    result, session, outputs = planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
        persistent=True, resident_budget=budget())
    assert result['status'] == 'PASS_SELECTED_PROGRAM' and result['reference_gate']['status'] == 'PASS'
    assert session.state['owner'] is owner and owner.clock == initial_clock + 2
    assert session.persistent and outputs['native'].ndim == 0
    assert result['selected_implementation'] == document['selected']['implementation']
    assert result['executed_order'] == document['selected']['order']
    assert result['resident_trial']['closed'] and result['resident_trial']['adopted']
    assert json.loads((tmp_path/'run/execution.json').read_text()) == result
    assert result['native_capability_evidence'][0]['active']
    after = snapshot(workload)
    for key in ('feeds', 'outer', 'parameters', 'optimizers'): assert after[key] == before[key]
    if warmed: assert after['slots'] == before['slots']
    assert not hasattr(owner, '_resident_trial_active')


@pytest.mark.parametrize('drift', ['feeds', 'shape', 'state', 'seed', 'policy', 'source', 'hardware', 'budget'])
def test_resident_saved_plan_strict_identity_before_any_update(tmp_path, monkeypatch, drift):
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path)
    selected_budget = budget()
    if drift == 'feeds': workload.feeds['x'].add_(.1)
    elif drift == 'shape': workload.feeds['x'] = workload.feeds['x'].repeat(2, 1)
    elif drift == 'state':
        with torch.no_grad(): workload.state['owner'].model.weight.add_(.01)
    elif drift == 'seed': workload.seed += 1
    elif drift == 'policy': workload.state['owner'].version += 1
    elif drift == 'source': monkeypatch.setattr(planning, 'source_identity', lambda: 'changed')
    elif drift == 'hardware': monkeypatch.setattr(ExecutionBackend, 'hardware', lambda self: {'changed': True})
    else: selected_budget = replace(budget(), reserve_bytes=100_001)
    before = snapshot(workload)
    def forbidden(*args, **kwargs): raise AssertionError('drift reached native update')
    monkeypatch.setattr(Session, 'run', forbidden)
    with pytest.raises(ValueError, match='changed|budget|environment|stale'):
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                     persistent=True, resident_budget=selected_budget)
    assert snapshot(workload) == before and not (tmp_path/'run').exists()


@pytest.mark.parametrize('kwargs', [dict(resident_budget=budget()), dict(persistent=True)])
def test_resident_plan_requires_explicit_persistent_ownership_and_budget(tmp_path, kwargs):
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path); before = snapshot(workload)
    with pytest.raises(ValueError, match='persistent|budget'):
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run', **kwargs)
    assert snapshot(workload) == before and not (tmp_path/'run').exists()


@pytest.mark.parametrize('limit', ['runs', 'host', 'snapshot', 'result'])
def test_resident_calibration_budget_rejection_does_not_execute(tmp_path, monkeypatch, limit):
    workload = build(ScopedOwner()); before = snapshot(workload)
    changes = {} if limit == 'runs' else {limit + '_limit_bytes': 1}
    def forbidden(*args, **kwargs): raise AssertionError('budget refusal reached native update')
    monkeypatch.setattr(Session, 'run', forbidden)
    with pytest.raises(ValueError, match='budget|bound|headroom'):
        planning.calibrate(workload, tmp_path/'costs', samples=1, max_candidates=2,
            max_program_runs=2 if limit == 'runs' else 3, resident_budget=budget(**changes))
    assert snapshot(workload) == before and not (tmp_path/'costs').exists()


@pytest.mark.parametrize('fault', ['numeric', 'effect', 'publication', 'final_rng'])
def test_resident_saved_run_failed_gate_never_publishes_success_or_adopts(tmp_path, monkeypatch, fault):
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path); before = snapshot(workload)
    original_run = Session.run; original_publish = planning._publish_resident_success
    original_rng = ResidentTrial._restore_outer_rng; runs = []; restores = []
    def execute(self, *args, **kwargs):
        outputs = original_run(self, *args, **kwargs); runs.append(1)
        if len(runs) == 2:
            if fault == 'numeric': outputs['native'].add_(1)
            elif fault == 'effect':
                row = self.effects['owner_execution']['owner']['components']['native']
                row.update(checkpoint_requests=0, recompute_calls=0, disabled_verified=False)
        return outputs
    def publish(path, value):
        if fault == 'publication' and path.name == 'execution.json': raise OSError('publication refused')
        return original_publish(path, value)
    def restore(self):
        restores.append(1)
        if fault == 'final_rng' and len(restores) == 2: raise MemoryError('final RNG refused')
        return original_rng(self)
    monkeypatch.setattr(Session, 'run', execute)
    monkeypatch.setattr(planning, '_publish_resident_success', publish)
    monkeypatch.setattr(ResidentTrial, '_restore_outer_rng', restore)
    with pytest.raises((ValueError, OSError, MemoryError), match='numeric|reference|exercised|publication|RNG'):
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                     persistent=True, resident_budget=budget())
    assert len(runs) == 2 and snapshot(workload) == before
    assert not (tmp_path/'run/execution.json').exists()
    assert (tmp_path/'run/failure.json').exists()
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']
    assert failure['reference_gate']['rtol'] == 1e-5 and failure['reference_gate']['atol'] == 1e-7
    assert failure['reference_gate']['status'] == ('REVIEW' if fault == 'numeric' else 'PASS')
    if fault == 'numeric': assert '/outputs/native' in failure['reference_gate']['mismatches']
    assert failure['native_capability_evidence'][0]['active'] is (fault != 'effect')
    assert not hasattr(workload.state['owner'], '_resident_trial_active')


@pytest.mark.parametrize('fault', ['publication', 'restore'])
def test_resident_calibration_success_requires_completed_restore_and_publication(tmp_path, monkeypatch, fault):
    workload = build(ScopedOwner()); owner = workload.state['owner']; before = snapshot(workload)
    publish = planning._publish_resident_success
    def fail_publish(path, value):
        if path.name == 'calibration.json':
            assert snapshot(workload) == before and not hasattr(owner, '_resident_trial_active')
            raise OSError('calibration publication refused')
        return publish(path, value)
    if fault == 'publication': monkeypatch.setattr(planning, '_publish_resident_success', fail_publish)
    else:
        def fail_restore(*args, **kwargs): raise MemoryError('native restore refused')
        monkeypatch.setattr(ScopedOwner, '_resident_trial_restore', fail_restore)
    with pytest.raises(OSError if fault == 'publication' else RollbackError):
        planning.calibrate(workload, tmp_path/'costs', samples=1, max_candidates=2,
                           max_program_runs=3, resident_budget=budget())
    assert not (tmp_path/'costs/calibration.json').exists()
    assert (tmp_path/'costs/failure.json').exists()
    assert not hasattr(owner, '_resident_trial_active')
    if fault == 'publication': assert snapshot(workload) == before
    else: assert owner._resident_trial_poisoned


def test_resident_calibration_reports_inactive_candidates_without_adoption(tmp_path):
    workload = build(ScopedOwner()); workload.state['owner'].exercise = False
    before = snapshot(workload)
    manifest = planning.calibrate(workload, tmp_path/'costs', samples=1, max_candidates=2,
        max_program_runs=3, resident_budget=budget())
    assert snapshot(workload) == before and len(manifest['variants']) == 1
    assert manifest['failed'][0]['reason'] == 'INAPPLICABLE_NATIVE_OWNER_CAPABILITY'
    assert manifest['budget']['actual'] == 3


def test_resident_profile_probes_are_explicitly_out_of_budget(tmp_path, monkeypatch):
    from distillschedule.open_workload import hardware_probe
    workload = build(ScopedOwner()); before = snapshot(workload)
    def forbidden(*args, **kwargs): raise AssertionError('unbudgeted hardware probe')
    monkeypatch.setattr(hardware_probe, 'probe', forbidden)
    with pytest.raises(ValueError, match='probe|profile_hardware'):
        planning.calibrate(workload, tmp_path/'refused', resident_budget=budget(), profile_hardware=True)
    assert snapshot(workload) == before and not (tmp_path/'refused').exists()
    manifest, _ = measured_plan(workload, tmp_path)
    assert manifest['hardware_probe']['status'] == 'NOT_RUN_RESIDENT_BUDGET'


def test_cpu_synthetic_measurement_keeps_resident_phase_peak_without_reset(monkeypatch):
    """Synthetic allocator observations only; never a CUDA execution proof."""
    from distillschedule.open_workload import torch_backend
    from test_resident_trial import synthetic_cuda_accounting_on_cpu_trial, option
    workload = build(); before = snapshot(workload)
    with ExecutionBackend(workload).resident_trial(budget()) as trial:
        observed = synthetic_cuda_accounting_on_cpu_trial(trial, monkeypatch)
        trial.reference(); resets = observed['reset_count']
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, stream): pass
            def elapsed_time(self, other): return 1.0
        monkeypatch.setattr(torch.cuda, 'Event', Event)
        monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
        monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: observed['row']['allocated_bytes'])
        monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: observed['row']['peak_allocated_bytes'])
        monkeypatch.setattr(torch_backend, 'random_scope', lambda *args, **kwargs: nullcontext())
        monkeypatch.setattr(Session, 'synchronize', lambda self: None)
        def execute(session):
            session.device = torch.device('cuda:0')  # API observations injected; tensors remain CPU.
            observed['row']['peak_allocated_bytes'] += 1000
            return session.run(measure=True)
        session, _, gate, _ = trial.candidate(option(), execute=execute)
        assert gate['status'] == 'PASS' and observed['reset_count'] == resets + 1
        assert session.profile['native']['capacity_measurement'] == 'resident_phase_high_water_bound'
        assert session.profile['native']['workspace_bytes'] >= 1000
    assert snapshot(workload) == before


@pytest.mark.parametrize('secondary_failure', [False, True])
def test_failed_publication_and_rollback_keep_both_errors_and_poison(tmp_path, monkeypatch, secondary_failure):
    from distillschedule.open_workload import resident_trial
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path)
    outer = signature(rng_state(device='cpu')); publish = planning._publish_resident_success
    original = OSError('saved-plan publication refused')
    def fail_restore(*args): raise MemoryError('snapshot restore refused')
    def fail_publish(path, value):
        if path.name == 'execution.json':
            monkeypatch.setattr(resident_trial, '_restore_tensors', fail_restore)
            raise original
        return publish(path, value)
    monkeypatch.setattr(planning, '_publish_resident_success', fail_publish)
    if secondary_failure:
        write = planning.write_new_json
        def fail_receipt(path, value):
            if path.name == 'failure.json': raise OSError('secondary failure publication refused')
            return write(path, value)
        monkeypatch.setattr(planning, 'write_new_json', fail_receipt)
    with pytest.raises(RollbackError) as captured:
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                     persistent=True, resident_budget=budget())
    assert captured.value.original_error is original
    assert isinstance(captured.value.rollback_error, MemoryError)
    assert workload.state['owner']._resident_trial_poisoned
    assert signature(rng_state(device='cpu')) == outer
    assert not (tmp_path/'run/execution.json').exists()
    if secondary_failure:
        assert 'secondary failure publication refused' in captured.value.planning_failure_publication_error
        return
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']
    assert failure['original_error'] == dict(type='builtins.OSError', message='saved-plan publication refused')
    assert failure['rollback_error'] == dict(type='builtins.MemoryError', message='snapshot restore refused')


def test_double_publication_failure_preserves_primary_without_python311_notes(tmp_path, monkeypatch):
    class Python310PublicationError(OSError):
        add_note = None
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path); before = snapshot(workload)
    publish = planning._publish_resident_success; write = planning.write_new_json
    original = Python310PublicationError('primary execution publication refused')
    def fail_publish(path, value):
        if path.name == 'execution.json': raise original
        return publish(path, value)
    def fail_receipt(path, value):
        if path.name == 'failure.json': raise OSError('secondary failure publication refused')
        return write(path, value)
    monkeypatch.setattr(planning, '_publish_resident_success', fail_publish)
    monkeypatch.setattr(planning, 'write_new_json', fail_receipt)
    with pytest.raises(Python310PublicationError) as captured:
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                     persistent=True, resident_budget=budget())
    assert captured.value is original and snapshot(workload) == before
    assert 'secondary failure publication refused' in captured.value.planning_failure_publication_error
    assert not (tmp_path/'run/execution.json').exists()


def test_resident_calibration_numeric_rejection_restores_and_counts_real_attempts(tmp_path):
    workload = build(ScopedOwner()); before = snapshot(workload)
    def corrupt(active, role, x):
        if active.enabled and role == 'c':
            with torch.no_grad(): active.model.weight.add_(1)
    workload.state['owner'].fault = corrupt
    manifest = planning.calibrate(workload, tmp_path/'costs', samples=2, max_candidates=2,
        max_program_runs=5, resident_budget=budget())
    assert snapshot(workload) == before
    assert manifest['budget']['actual'] == 4 and len(manifest['variants']) == 1
    assert manifest['failed'][0]['reference_comparison']['status'] == 'REVIEW'


def test_resident_deep_profile_requires_separate_budget_before_trial(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('profiler admission reached snapshot')
    monkeypatch.setattr(ResidentTrial, '__enter__', forbidden)
    with pytest.raises(ValueError, match='profiling.*budget'):
        planning.run(build(), tmp_path/'absent-plan.json', tmp_path/'run',
                     resident_budget=budget(), persistent=True, deep_profile=True)


@pytest.mark.parametrize('operation', ['calibration', 'execution'])
@pytest.mark.parametrize('fault', ['mid_write', 'trailing_write', 'flush', 'close'])
def test_resident_success_writer_failure_never_installs_final_receipt(tmp_path, monkeypatch, operation, fault):
    """Break the actual opened writer, including a parseable complete JSON body."""
    workload = build(ScopedOwner())
    if operation == 'execution': measured_plan(workload, tmp_path)
    before = snapshot(workload)
    target = tmp_path/('run/execution.json' if operation == 'execution' else 'costs/calibration.json')
    original_open = Path.open; original_fdopen = os.fdopen; original_mkstemp = tempfile.mkstemp
    staged = {}; observations = []
    def relevant(path):
        return path.parent == target.parent and (path.name == target.name or path.name.startswith('.'+target.name+'.'))
    class Writer:
        def __init__(self, stream, path): self.stream, self.path, self.calls = stream, path, 0
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def __getattr__(self, name): return getattr(self.stream, name)
        def write(self, value):
            self.calls += 1
            if fault == 'mid_write' and self.calls == 4:
                self.stream.flush(); observations.append('partial JSON'); raise OSError('actual mid-write failed')
            if fault == 'trailing_write' and value == '\n':
                self.stream.flush()
                try:
                    with original_open(self.path, encoding='utf-8') as reader: json.load(reader)
                except json.JSONDecodeError: pass
                else:
                    observations.append('complete JSON'); raise OSError('actual trailing write failed')
            return self.stream.write(value)
        def flush(self):
            self.stream.flush()
            if fault == 'flush':
                observations.append('complete JSON'); raise OSError('actual flush failed')
        def close(self):
            try: self.flush()
            finally: self.stream.close()
            if fault == 'close':
                observations.append('complete JSON'); raise OSError('actual close failed')
    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        return Writer(stream, path) if relevant(path) and any(c in mode for c in 'xwa') else stream
    def made(*args, **kwargs):
        fd, name = original_mkstemp(*args, **kwargs); staged[fd] = Path(name); return fd, name
    def fdopened(fd, *args, **kwargs):
        stream = original_fdopen(fd, *args, **kwargs); path = staged.get(fd)
        return Writer(stream, path) if path is not None and relevant(path) else stream
    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(tempfile, 'mkstemp', made)
    monkeypatch.setattr(os, 'fdopen', fdopened)
    with pytest.raises(OSError, match='actual .*failed'):
        if operation == 'execution':
            planning.run(workload, tmp_path/'plan.json', target.parent, persistent=True, resident_budget=budget())
        else:
            planning.calibrate(workload, target.parent, samples=1, max_candidates=2,
                               max_program_runs=3, resident_budget=budget())
    assert observations and snapshot(workload) == before
    assert not target.exists(), 'a failed writer left a final success pathname'
    assert not list(target.parent.glob('.'+target.name+'.*'))
    failure = json.loads((target.parent/'failure.json').read_text())
    assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']


@pytest.mark.parametrize('operation', ['calibration', 'execution'])
@pytest.mark.parametrize('fault', ['install', 'collision', 'cleanup'])
def test_resident_success_atomic_install_and_nonfatal_cleanup(tmp_path, monkeypatch, operation, fault):
    workload = build(ScopedOwner()); owner = workload.state['owner']
    if operation == 'execution': measured_plan(workload, tmp_path)
    before = snapshot(workload); initial_clock = owner.clock
    target = tmp_path/('run/execution.json' if operation == 'execution' else 'costs/calibration.json')
    link = os.link; unlink = os.unlink; observations = []
    def install(source, destination, *args, **kwargs):
        if Path(destination) == target:
            assert json.loads(Path(source).read_text())
            assert not target.exists()
            if operation == 'execution':
                assert not owner._resident_trial_active.receipt['adopted']
            else: assert snapshot(workload) == before
            observations.append('complete_before_install')
            if fault == 'install': raise OSError('atomic install refused')
            if fault == 'collision': target.write_bytes(b'prior publisher content')
        return link(source, destination, *args, **kwargs)
    def cleanup(path, *args, **kwargs):
        path = Path(path)
        if fault == 'cleanup' and path.parent == target.parent and path.name.startswith('.'+target.name+'.'):
            assert target.exists(); observations.append('cleanup_failed_after_install')
            raise OSError('staging cleanup refused after commit')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(os, 'link', install); monkeypatch.setattr(os, 'unlink', cleanup)
    def execute():
        if operation == 'execution':
            return planning.run(workload, tmp_path/'plan.json', target.parent,
                                persistent=True, resident_budget=budget())[0]
        return planning.calibrate(workload, target.parent, samples=1, max_candidates=2,
                                  max_program_runs=3, resident_budget=budget())
    if fault == 'cleanup':
        result = execute()
        assert json.loads(target.read_text()) == result
        assert not (target.parent/'failure.json').exists()
        assert result['resident_trial']['adopted'] is (operation == 'execution')
        assert owner.clock == initial_clock + (2 if operation == 'execution' else 0)
        if operation == 'calibration': assert snapshot(workload) == before
        assert observations == ['complete_before_install', 'cleanup_failed_after_install']
        remaining = list(target.parent.glob('.'+target.name+'.*'))
        assert len(remaining) == 1
        unlink(remaining[0])  # Test-owned non-authoritative staging file only.
    else:
        with pytest.raises(OSError, match='atomic install refused|File exists'): execute()
        assert snapshot(workload) == before
        if fault == 'collision': assert target.read_bytes() == b'prior publisher content'
        else: assert not target.exists()
        assert observations == ['complete_before_install']
        failure = json.loads((target.parent/'failure.json').read_text())
        assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']
        assert not list(target.parent.glob('.'+target.name+'.*'))


@pytest.mark.parametrize('operation', ['calibration', 'execution'])
def test_resident_planning_preserves_preexisting_output_directory(tmp_path, operation):
    workload = build(ScopedOwner())
    if operation == 'execution': measured_plan(workload, tmp_path)
    before = snapshot(workload); output = tmp_path/'already-owned'; output.mkdir()
    sentinel = output/'sentinel'; sentinel.write_bytes(b'preexisting caller evidence')
    with pytest.raises(FileExistsError):
        if operation == 'execution':
            planning.run(workload, tmp_path/'plan.json', output, persistent=True, resident_budget=budget())
        else:
            planning.calibrate(workload, output, samples=1, max_candidates=2,
                               max_program_runs=3, resident_budget=budget())
    assert snapshot(workload) == before
    assert list(output.iterdir()) == [sentinel] and sentinel.read_bytes() == b'preexisting caller evidence'


@pytest.mark.parametrize('drift', ['source', 'hardware'])
def test_final_publication_callback_rechecks_source_and_hardware(tmp_path, monkeypatch, drift):
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path); before = snapshot(workload)
    calls = []
    if drift == 'source':
        original = planning.source_identity
        def changed():
            calls.append(1)
            return original() if len(calls) == 1 else 'final-source-drift'
        monkeypatch.setattr(planning, 'source_identity', changed)
    else:
        original = ExecutionBackend.hardware
        def changed(self):
            calls.append(1)
            return original(self) if len(calls) == 1 else {'final_hardware_drift': True}
        monkeypatch.setattr(ExecutionBackend, 'hardware', changed)
    with pytest.raises(ValueError, match='source changed|environment changed'):
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run', persistent=True, resident_budget=budget())
    assert len(calls) == 2 and snapshot(workload) == before
    assert not (tmp_path/'run/execution.json').exists()
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert failure['reference_gate']['status'] == 'PASS'
    assert failure['native_capability_evidence'][0]['active']
    assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']


def test_cpu_synthetic_capacity_failure_persists_failed_and_rollback_observations(tmp_path, monkeypatch):
    """Real owner rollback with injected CPU-only CUDA accounting, NOT GPU proof."""
    from test_resident_trial import synthetic_cuda_accounting_on_cpu_trial
    entered = ResidentTrial.__enter__; run = Session.run; observed = {}
    def enter(self):
        result = entered(self)
        observed[self] = synthetic_cuda_accounting_on_cpu_trial(self, monkeypatch)
        return result
    monkeypatch.setattr(ResidentTrial, '__enter__', enter)
    workload = build(ScopedOwner()); measured_plan(workload, tmp_path); before = snapshot(workload)
    calls = []
    def exceed(self, *args, **kwargs):
        out = run(self, *args, **kwargs); calls.append(1)
        if len(calls) == 2:
            trial = workload.state['owner']._resident_trial_active
            observed[trial]['row']['peak_allocated_bytes'] = trial._allocated_bound + 1
        return out
    monkeypatch.setattr(Session, 'run', exceed)
    with pytest.raises(ValueError, match='allocator peak exceeded'):
        planning.run(workload, tmp_path/'plan.json', tmp_path/'run', persistent=True, resident_budget=budget())
    assert snapshot(workload) == before and not (tmp_path/'run/execution.json').exists()
    failure = json.loads((tmp_path/'run/failure.json').read_text()); receipt = failure['resident_trial']
    assert receipt['closed'] and not receipt['adopted']
    rows = receipt['cuda_observations']
    assert {'candidate', 'failed', 'rollback'} <= {row['phase'] for row in rows}
    assert next(row for row in rows if row['phase'] == 'failed')['peak_allocated_bytes'] > receipt['device_allocated_bound_bytes']
    assert rows[-1]['phase'] == 'rollback'


@pytest.mark.parametrize('family', ['sd15', 'sdxl', 'self_forcing'])
@pytest.mark.parametrize('warmed', [False, True])
def test_optional_real_native_cpu_resident_shared_planning(tmp_path, monkeypatch, family, warmed):
    """Root-owned optional pinned-native CPU lane; no substituted tiny algorithm."""
    from distillschedule.open_workload.resident_trial import TrialBudget
    from distillschedule.training.checkpoint import cpu_tree
    from distillschedule.training.config import load_training_config
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.session import construct, build_cycle
    from test_training_end_to_end import equal
    env = 'SELF_FORCING_SOURCE_ROOT' if family == 'self_forcing' else 'DMD2_SOURCE_ROOT'
    source = os.environ.get(env)
    if not source: pytest.skip('optional pinned native CPU proof NOT_RUN: set '+env)
    recipe = 'self_forcing_official' if family == 'self_forcing' else 'dmd2_sd15'
    builder, _ = load_callable(str(Path(__file__).resolve().parents[1]/'recipes'/recipe/'create_cpu_fixture.py')+':build_fixture')
    files = builder(tmp_path/'native', Path(source).resolve(strict=True), cycles=3,
                    **({} if family == 'self_forcing' else dict(variant=family, ratio=2)))
    config_path = Path(files['config'])
    if family == 'self_forcing':
        value = json.loads(config_path.read_text())
        value['model']['parameters']['settings']['gradient_checkpointing'] = True
        config_path.write_text(json.dumps(value))
    config = load_training_config(config_path)
    components, data, algorithm, _ = construct(config); owner = components.state['native_cycle']
    first = int(warmed)
    if warmed: owner.run(build_cycle(config, components, data, algorithm, cycle=0, cursor=0).feeds, cycle=0)
    workload = build_cycle(config, components, data, algorithm, cycle=first, cursor=first)
    before = cpu_tree(owner.state_dict())
    expected_output = cpu_tree(owner.run(workload.feeds, cycle=first)); expected = cpu_tree(owner.state_dict())
    owner.load_state_dict(before)  # Prepare the reference boundary before private trials.
    outer = signature(rng_state(device='cpu')); feeds = signature(workload.feeds)
    parameters = [(id(p), p.data_ptr()) for p in owner.model.parameters()]
    slots = [(id(t), t.data_ptr()) for opt in owner.optimizers.values()
             for row in opt.state.values() for t in row.values()]
    def forbidden(*args, **kwargs): raise AssertionError('native resident planning cloned or checkpoint-loaded owner')
    monkeypatch.setattr(type(owner), '__deepcopy__', forbidden)
    monkeypatch.setattr(owner, 'load_state_dict', forbidden)
    for optimizer in owner.optimizers.values(): monkeypatch.setattr(optimizer, 'load_state_dict', forbidden)
    limits = TrialBudget(host_limit_bytes=2**31, snapshot_limit_bytes=2**29,
        result_limit_bytes=2**29, scratch_bytes=2**29, reserve_bytes=2**26)
    manifest = planning.calibrate(workload, tmp_path/'costs', samples=1, max_candidates=2,
                                  max_program_runs=3, resident_budget=limits)
    assert equal(cpu_tree(owner.state_dict()), before)
    assert signature(rng_state(device='cpu')) == outer and signature(workload.feeds) == feeds
    assert len(manifest['variants']) == 2 and manifest['budget']['actual'] == 3
    document = planning.plan(tmp_path/'costs/calibration.json', memory_limit_bytes=2**31)
    write_new_json(tmp_path/'plan.json', document)
    result, session, output = planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                                          resident_budget=limits, persistent=True)
    assert session.state['native_cycle'] is owner and owner.cycle == first + 1
    assert equal(cpu_tree(owner.state_dict()), expected)
    assert equal(cpu_tree(next(iter(output.values()))), expected_output)
    assert result['selected_implementation'] == document['selected']['implementation']
    assert result['resident_trial']['adopted'] and result['reference_gate']['status'] == 'PASS'
    assert parameters == [(id(p), p.data_ptr()) for p in owner.model.parameters()]
    if warmed:
        assert slots == [(id(t), t.data_ptr()) for opt in owner.optimizers.values()
                         for row in opt.state.values() for t in row.values()]
    assert signature(rng_state(device='cpu')) == outer and signature(workload.feeds) == feeds
