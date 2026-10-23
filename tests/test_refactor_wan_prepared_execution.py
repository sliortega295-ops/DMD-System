from types import SimpleNamespace as NS
import ast
from pathlib import Path

import pytest

from distillschedule.runtime.wan_cycle_plan import prepare_wan_cycle
from distillschedule.search.wan_implementation import WanImplementation, current_implementation


def make_native(window, *, persistent=False, operation_scope=None):
    calls = []
    g, c, ema = NS(global_steps=2), NS(global_steps=10), NS(optimization_step=10)
    def gupdate(**kw):
        calls.append(("g",)); g.global_steps += 1
        return {"physical_score_forwards":3}
    def rollout(**kw):
        assert len(kw["steps"]) == 1
        step = kw["steps"][0]
        calls.append(("r", step, g.global_steps))
        kw["entry_outputs"][step] = NS(tensors={"trace": step})
        return {step:(g.global_steps,step)}, [{"batch_size":1, "batch_index":0}], {}
    def critic(**kw):
        assert kw["entry"].tensors == {"trace":kw["step"]}
        assert kw["x0"] == (3,kw["step"])
        calls.append(("c",kw["step"],c.global_steps)); c.global_steps += 1
        return {"logical_step":kw["step"],"score_forwards":1}
    def ema_step():
        calls.append(("ema",)); ema.optimization_step += 1
    ema.step = ema_step
    p0 = NS(_run_nogan_generator_update_p0=gupdate, _precompute_critic_rollouts=rollout,
            _run_nogan_critic_precomputed=critic)
    program = prepare_wan_cycle(p0=p0, source_steps=(10,11,12,13,14), max_live_rollouts=window,
        rank=0, device="test", generator_engine=g, critic_engine=c, ema=ema, trace=object(),
        negative=object(), diagnostics=object(), policy=NS(rollout_batch_size=1,
            generator_nograd_persistent_unshard=persistent), clean_runtime=True,
        runtime_cache={}, deferred_finite_checks=[], collect_scalars=False, operation_scope=operation_scope)
    return program, calls


@pytest.mark.parametrize("window",[1,3,5])
def test_native_helpers_consume_selected_order_not_old_clustered_loop(window):
    program,calls = make_native(window)
    assert calls == []  # compilation is before the timer and does not call the model
    raw = program.run()
    assert len(raw["rollouts"]) == len(raw["critic"]) == 5
    assert [x["batch_index"] for x in raw["rollouts"]] == list(range(5))
    assert [x["logical_step"] for x in raw["critic"]] == list(range(10,15))
    first_c = next(i for i,x in enumerate(calls) if x[0] == "c")
    assert first_c == window+1
    receipt = program.receipt()
    assert receipt["final_versions"] == {"generator":3,"critic":15,"ema":15}
    assert receipt["peak_live_inputs"] == window
    assert receipt["status"] == "PASS"


def test_persistent_unshard_cannot_silently_change_scope_with_new_schedule():
    with pytest.raises(ValueError, match="persistent"):
        make_native(3,persistent=True)


def test_native_commit_mismatch_cannot_emit_pass_dispatch_receipt():
    program, _ = make_native(1)
    program.read_versions = lambda: {"generator":3,"critic":14,"ema":15}
    with pytest.raises(ValueError, match="native engine commits"):
        program.run()
    assert program.receipt()["status"] == "FAIL"


def test_schedule_env_cannot_override_registered_candidate():
    from distillschedule.search.wan_implementation import current_rollout_window
    for env in ({"DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW":"3"},
                {"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID":"joint-wan2-resident-c2048-e1-w3",
                 "DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW":"5"}):
        with pytest.raises(ValueError):
            current_rollout_window(env)


def test_missing_or_different_executed_schedule_cannot_pass_coordinator():
    from distillschedule.runtime.wan_cycle_plan import validate_wan_prepared_receipts
    program,_ = make_native(3); program.run()
    rows = [program.receipt()]
    validate_wan_prepared_receipts(rows,window=3,cycles=1)
    for changed in ([], [{**rows[0],"executed_operations":[]}], [{**rows[0],"native_versions":{}}]):
        with pytest.raises(ValueError):
            validate_wan_prepared_receipts(changed,window=3,cycles=1)


def test_generated_schedule_identity_reaches_native_environment():
    from distillschedule.search.registry import get_candidate, candidate_profile
    impl = WanImplementation("resident",2048,True,rollout_window=3)
    assert impl.id == "joint-wan2-resident-c2048-e1-w3"
    assert WanImplementation.parse(impl.id) == impl
    assert WanImplementation.from_assignment(impl.assignment) == impl
    env = impl.environment()
    assert env["DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW"] == "3"
    assert current_implementation({**env,"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID":impl.id}) == impl
    p = candidate_profile(get_candidate("wan13b_nogan_resident_best",impl.id))
    assert p["runtime"]["environment"]["DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW"] == "3"
    legacy = WanImplementation("resident",2048,True)
    assert legacy.id == "joint-wan1-resident-c2048-e1"
    assert "rollout_window" not in legacy.assignment
    assert "DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW" not in legacy.environment()


def test_both_real_workers_consume_prepared_plan_and_timing_compiles_before_start():
    root = Path(__file__).resolve().parents[1] / "scripts/distillschedule"
    for filename in ("probe_wan13b_two_cycle_timing_world16.py", "probe_wan13b_trajectory_world16.py"):
        source = (root/filename).read_text()
        ast.parse(source)
        assert "current_rollout_window()" in source
        assert "prepare_wan_cycle(" in source
        assert "prepared_cycle.run()" in source
        assert "prepared_cycle.receipt()" in source
    timing = (root/"probe_wan13b_two_cycle_timing_world16.py").read_text()
    assert timing.index("prepared_cycle = prepare_wan_cycle(") < timing.index("start_ns = time.time_ns()")


def test_native_rollout_storage_observer_runs_after_calibrated_range():
    from contextlib import contextmanager
    events=[]
    @contextmanager
    def scope(op):
        events.append(("start",op.operation_id))
        yield NS(record_output=lambda value:events.append(("output",op.operation_id,value)))
        events.append(("stop",op.operation_id))
    program,_=make_native(3,operation_scope=scope)
    program.run()
    outputs=[row for row in events if row[0]=="output"]
    assert len(outputs)==5
    for row in outputs:
        assert events.index(("stop",row[1])) < events.index(row)
        assert row[2][1] == {"trace":10+int(row[1][1:])}


def test_prepared_calibration_reaches_native_collector_not_legacy_macros():
    root=Path(__file__).resolve().parents[1]
    worker=(root/"scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py").read_text()
    assert "from distillschedule.search.wan_action_calibration import ActionCalibrator" in worker
    assert "operation_scope=" in worker
    assert "prepared schedules require component calibration, not legacy stage costs" not in worker
