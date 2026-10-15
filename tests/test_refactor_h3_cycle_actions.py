"""CPU native-loop integration, not H3 model/GPU numerical evidence."""
import copy
import hashlib
import pickle

import pytest

from distillschedule.backends.training.minimax_h3.cycle_actions import (
    H3ActionTrace, prepare_cycle, validate_cycle_actions, compare_action_inputs,
)
from distillschedule.search.h3_implementation import H3Implementation
from test_refactor_h3_planner_cycle_observer import _native_fixture


def fixture():
    f = _native_fixture(fake_loss_rng=False)
    def snapshot():
        digest = hashlib.sha256(pickle.dumps(f.rng.getstate())).hexdigest()
        return dict(torch_cpu=digest, torch_cuda=digest, python_random=digest)
    trace = H3ActionTrace(rank=0, source_head="a" * 40,
        rng_snapshot=snapshot)
    return f, trace


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
def test_native_helpers_execute_two_cycles_with_last_consumer_release(window):
    f, trace = fixture()
    trainer = f.trainer
    # Binding uses the real native helper methods extracted from trainer_loop.
    iterator = trainer._iter_train_samples()
    for cycle in (0, 1):
        trainer.matched_cycle_census.reset()
        trace.begin_cycle(cycle + 1)
        f.observer.active = cycle
        program = prepare_cycle(trainer, samples=iterator, grad_accum_iters=1,
                                current_iter=cycle, rollout_window=window, trace=trace)
        assert not f.calls, "plan construction must not execute native work"
        result = program.run()
        assert set(result) == {"g", "c0", "c1", "c2", "c3", "c4"}
        receipt = trace.finish_cycle(program.receipt(), f.observer.updates,
                                     trainer.matched_cycle_census.snapshot(), require_draws=False)
        assert receipt["dispatch"]["peak_live_inputs"] == window
        assert receipt["actual_versions"] == dict(generator=cycle + 1, critic=5*(cycle+1), ema=0)
        assert trainer.matched_cycle_census.validate_local() == []
        assert [c[2] for c in f.calls if c[0] == "prepare"] == list(range(6*cycle+1, 6*cycle+6))
        assert [c[1] for c in f.calls if c[0] == "optimizer"] == ["student"] + ["fake"] * 5
        with pytest.raises(Exception, match="one-shot"):
            program.run()
        f.calls.clear()
    assert f.actual_updates == dict(student=2, fake=10)
    assert f.torch.cuda.synchronizations == 0


@pytest.mark.parametrize("site", ["fake_loss", "backward", "optimizer"])
def test_update_rng_consumption_is_not_admitted_even_when_terminal_draw_count_matches(site):
    f, trace = fixture()
    if site == "fake_loss":
        original = f.trainer.fake_loss
        def call(item):
            f.rng.random()
            return original(item)
        f.trainer.fake_loss = call
    elif site == "optimizer":
        original = f.trainer.fake_optimizer.step
        def call():
            f.rng.random()
            return original()
        f.trainer.fake_optimizer.step = call
    else:
        original = f.trainer.fake_loss
        def call(item):
            value = original(item)
            value.backward = lambda: f.rng.random()
            # Scalar's division returns itself in the native fixture.
            return value
        f.trainer.fake_loss = call
    trace.begin_cycle(1)
    f.observer.active = 0
    program = prepare_cycle(f.trainer, samples=f.trainer._iter_train_samples(), grad_accum_iters=1,
                            current_iter=0, rollout_window=3, trace=trace)
    with pytest.raises(RuntimeError, match="RNG.*c0"):
        program.run()
    assert program.receipt()["status"] == "FAIL"
    assert trace.rows[-1]["status"] == "FAIL"
    # The optimizer may have committed before the diagnostic rejected the action.
    # No rollback or automatic replay is advertised.
    with pytest.raises(Exception, match="one-shot"):
        program.run()


def test_legacy_random_fake_fixture_is_rejected_as_a_reordering_reference():
    f = _native_fixture()  # deliberately random in preparation AND fake_loss
    trace = H3ActionTrace(rank=0, source_head="a" * 40,
        rng_snapshot=lambda: {"fixture_rng": repr(f.rng.getstate())})
    trace.begin_cycle(1)
    f.observer.active = 0
    with pytest.raises(RuntimeError, match="RNG.*c0"):
        prepare_cycle(f.trainer, samples=f.trainer._iter_train_samples(), grad_accum_iters=1,
                      current_iter=0, rollout_window=5, trace=trace).run()


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
def test_new_id_is_opt_in_and_cannot_admit_timing_or_old_macro_costs(window):
    legacy = H3Implementation(2, 128)
    impl = H3Implementation(2, 128, rollout_window=window)
    assert legacy.id == "joint-h3v1-d2-n128"
    assert impl.id == f"joint-h3v2-d2-n128-w{window}"
    assert H3Implementation.parse(impl.id) == impl
    assert H3Implementation.from_assignment(impl.assignment) == impl
    profile = impl.lower_profile()
    assert profile["allowed_run_modes"] == ["correctness", "diagnostic-timing", "calibration"]
    assert profile["runtime"]["environment"]["H3_PLANNER_ROLLOUT_WINDOW"] == str(window)
    assert profile["runtime"]["model_config"] == legacy.lower_profile()["runtime"]["model_config"]
    for mode in ("diagnostic-timing", "calibration"):
        env=dict(profile["runtime"]["environment"],H3_PLANNER_CALIBRATION=str(int(mode=='calibration')))
        impl.receipt_binding(profile["runtime"]["model_config"],env,mode=mode)
    from distillschedule.search.h3_action_admission import validate_action_admission
    from pathlib import Path
    with pytest.raises(ValueError,match='prior native action proof'):
        validate_action_admission({}, {}, Path('.'))


def test_dispatch_success_is_not_native_update_success():
    f, trace = fixture()
    trace.begin_cycle(1)
    f.observer.active = 0
    program = prepare_cycle(f.trainer, samples=f.trainer._iter_train_samples(), grad_accum_iters=1,
                            current_iter=0, rollout_window=2, trace=trace)
    program.run()
    wrong_updates = dict(f.observer.updates, fake=4)
    with pytest.raises(ValueError, match="updates"):
        trace.finish_cycle(program.receipt(), wrong_updates,
                           f.trainer.matched_cycle_census.snapshot(), require_draws=False)


def action_receipt(window):
    f, trace = fixture()
    trace.begin_cycle(1)
    f.observer.active = 0
    program = prepare_cycle(f.trainer, samples=f.trainer._iter_train_samples(), grad_accum_iters=1,
                            current_iter=0, rollout_window=window, trace=trace)
    program.run()
    for row in trace.rows:
        if row["action"].startswith("c"):
            continue
        tensor = dict(shape=[1], dtype="torch.float32", device="cuda:0", bytes=4, finite=True,
                      sha256=hashlib.sha256(row["rng_before"]["torch_cpu"].encode()).hexdigest())
        row["draws"] = [dict(site=site, tensors=[copy.deepcopy(tensor) for _ in range(count)])
                        for site, count in (("initial_latents", 2), ("renoise_sigmas", 2),
                                            ("noise_like", 1), ("noise_like", 1))]
    return trace.finish_cycle(program.receipt(), f.observer.updates, f.trainer.matched_cycle_census.snapshot())


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
def test_draw_comparison_is_keyed_not_physical_order(window):
    assert compare_action_inputs(action_receipt(5), action_receipt(window))["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["draw", "shape", "dtype", "site", "step", "update", "input", "census"])
def test_well_formed_changed_draws_and_identity_cannot_inherit_rng_boundary_pass(mutation):
    left, right = action_receipt(5), action_receipt(3)
    row = next(r for r in right["rows"] if r["action"] == "r1")
    if mutation == "draw": row["draws"][0]["tensors"][0]["sha256"] = "b" * 64
    if mutation == "shape": row["draws"][0]["tensors"][0]["shape"] = [2]
    if mutation == "dtype": row["draws"][0]["tensors"][0]["dtype"] = "torch.bfloat16"
    if mutation == "site": row["draws"][0]["site"] = "replacement"
    if mutation == "step": row["logical_fake_step"] += 1
    if mutation == "update": row["actual_updates_after"]["fake"] += 1
    if mutation == "input": row["inputs"] = ["r0"]
    if mutation == "census": right["census_binding"]["samples"][0]["identity"] = "changed"
    assert compare_action_inputs(left, right)["status"] == "FAIL"


def test_native_lowering_releases_original_prepared_objects_after_their_own_consumer():
    import weakref
    f, trace = fixture()
    original_prepare = f.trainer.prepare_fake_update
    refs = []
    class Prepared:
        def __init__(self, key): self.cache_key = key
    def prepare(*args):
        item = Prepared(original_prepare(*args).cache_key)
        refs.append(weakref.ref(item))
        return item
    original_apply = f.trainer._apply_one_fake_group
    applied = 0
    def apply(group):
        nonlocal applied
        assert all(ref() is None for ref in refs[:applied]), "old prepared group retained after last consumer"
        assert group[0] is refs[applied]()
        applied += 1
        return original_apply(group)
    f.trainer.prepare_fake_update, f.trainer._apply_one_fake_group = prepare, apply
    trace.begin_cycle(1)
    f.observer.active = 0
    prepare_cycle(f.trainer, samples=f.trainer._iter_train_samples(), grad_accum_iters=1,
                  current_iter=0, rollout_window=3, trace=trace).run()
    assert all(ref() is None for ref in refs)


def test_failure_evidence_io_error_does_not_mask_native_error(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from distillschedule.backends.training.minimax_h3 import integrated_entrypoint as entry
    from distillschedule.backends.training.minimax_h3 import planner_cycle_observer as observers
    closed = []
    observer = SimpleNamespace(action_trace=SimpleNamespace(rows=[], cycle=1), source_head="a"*40,
        updates={}, cycles=[], action_program=None, close=lambda _: closed.append(True))
    monkeypatch.setattr(observers, "H3PlannerCycleObserver", lambda **_: observer)
    monkeypatch.setenv("EXPECTED_INTEGRATION_HEAD", "a" * 40)
    bad = tmp_path / "not_a_directory"
    bad.write_text("preserve this file")
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ROOT", str(bad))
    def failed(_): raise RuntimeError("original native failure")
    with pytest.raises(RuntimeError, match="original native failure"):
        entry._measure_planner_two_cycle(SimpleNamespace(), failed, torch=object(), rank=0)
    assert closed == [True]
    assert "could not be persisted" in capsys.readouterr().err


@pytest.mark.parametrize("candidate,window,mode,passes", [
    ("joint-h3v2-d2-n128-w3", "3", "correctness", True),
    ("joint-h3v2-d1-n64-w5", "5", "correctness", True),
    ("joint-h3v2-d2-n128-w3", "3", "diagnostic-timing", True),
    ("joint-h3v2-d2-n128-w3", "2", "correctness", False),
    ("joint-h3v1-d2-n128", "3", "correctness", False),
    ("joint-h3v1-d2-n128", "", "correctness", True),
    ("joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128", "", "correctness", True),
    ("joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128", "1", "correctness", False),
    ("joint-h3v3-h480-w832-f124-cuda_no_trim-d2-n128-r3", "3", "correctness", True),
])
def test_real_launcher_identity_guard_without_gpu(candidate, window, mode, passes, tmp_path):
    import os
    import subprocess
    import sys
    import json
    from pathlib import Path
    from distillschedule.search.h3_implementation import H3Implementation
    runtime = H3Implementation.parse(candidate).lower_profile()['runtime']
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(runtime['model_config']))
    text = (Path(__file__).resolve().parents[1] / "scripts/distillschedule/launch_minimax_h3_best_node.sh").read_text()
    # Execute the actual whole candidate guard, not a copied regex/model.
    text = text[text.index('if [[ -n "${DISTILLSCHEDULE_SEARCH_CANDIDATE_ID:-}"'):text.index('mkdir -p "$RUN_ROOT')]
    result = subprocess.run(["bash", "-ec", text], env={**os.environ, **runtime['environment'],
        "PYTHON": sys.executable, "CONFIG": str(config),
        "DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": candidate, "H3_PLANNER_ROLLOUT_WINDOW": window,
        "MODE": mode, "H3_PLANNER_TWO_CYCLE": "1", "H3_PLANNER_CLOCK": "host_fenced",
        "H3_PLANNER_CALIBRATION": "0"})
    assert (result.returncode == 0) == passes
