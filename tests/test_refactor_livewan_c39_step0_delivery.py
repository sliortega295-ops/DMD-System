"""C39 opt-in dependency/key inventory and source-routed semantic checks."""
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime.livewan_host_runahead import (
    PinnedHostRunaheadTransport, runahead_tag,
)
from distillschedule.runtime.livewan_step0_delivery import (
    CONTRACT, VARIANT, enabled, runtime_receipt_errors, transport_receipt_errors,
)

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "configs/distillschedule/plans"


@pytest.fixture
def plans(monkeypatch):
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/unread/" + key)
    return [load_plan(P / name).plan.to_dict() for name in (
        "livewan_wan13b_14b_sf_dmd_c38_continuous_b16_score_fa3.json",
        "livewan_wan13b_14b_sf_dmd_c39_continuous_b16_step0_delivery.json")]


def test_parent_unchanged_and_one_registered_factor(plans):
    a, b = copy.deepcopy(plans)
    assert b.pop("id") != a.pop("id")
    assert b["metadata"].pop("step0_wave1_early_delivery") == CONTRACT
    assert a == b
    assert not enabled(plans[0], mode="pipeline")
    assert enabled(plans[1], mode="pipeline")
    # Actual C38 parent is all-five Critic B2, not C37 step1 split.
    assert not worker._critic_step1_overlap_enabled(plans[1], mode="pipeline")
    for plan in plans:
        worker._apply_trajectory_contract(plan, cycles=2, mode="pipeline")
        worker._validate_plan_identity(plan, mode="pipeline")
    with pytest.raises(RuntimeError):
        enabled(plans[1], mode="baseline")


@pytest.mark.parametrize("field,value", [
    ("global_batch", 32), ("payload_dtype", "bfloat16"),
    ("enqueue_position", "before_rollout1"), ("enabled", False),
    ("depth1_lookahead_unchanged", False), ("extra", True),
])
def test_contract_fail_closed(plans, field, value):
    p = copy.deepcopy(plans[1])
    p["metadata"]["step0_wave1_early_delivery"][field] = value
    with pytest.raises(RuntimeError):
        worker._validate_plan_identity(p, mode="pipeline")


@pytest.mark.parametrize("batch,depth", [(32, 1), (16, 2)])
def test_actual_workload_not_only_metadata_is_checked(plans, batch, depth):
    p = copy.deepcopy(plans[1])
    p["workload"]["shape"]["batch_size"] = batch
    p["metadata"]["continuous_pipeline"]["lookahead_depth"] = depth
    with pytest.raises(RuntimeError):
        enabled(p, mode="pipeline")


@pytest.mark.parametrize("rank", [0, 8])
@pytest.mark.parametrize("early_delivery", [False, True])
def test_real_continuous_entry_forwards_validated_opt_in(
    monkeypatch, rank, early_delivery,
):
    """Execute the real CPU entry up to the GPU runner, not source matching.

    This catches an unbound plan lookup before a costly model construction.
    The sentinel intentionally stops before any model/GPU execution; this is
    a routing test, not a substitute for world16 workload correctness.
    """
    calls = []

    class ReachedModelRunner(Exception):
        pass

    def reached_model_runner(**kwargs):
        calls.append(kwargs)
        raise ReachedModelRunner

    monkeypatch.setattr(worker, "_run_pipeline_8x8", reached_model_runner)
    with pytest.raises(ReachedModelRunner):
        worker._run_pipeline_8x8_continuous(
            torch=None, rank=rank,
            trace=SimpleNamespace(cycles=2, global_batch=16),
            loader=None, bundle=None, streamers=(), score_context=None,
            run_kind="two-cycle-timing", evidence=worker.RunEvidence(),
            versions={"generator": 0, "critic": 0, "ema": 0}, device=None,
            runahead_host_transport=object(), critic_shadow_hot_sync=object(),
            slot_pairs_by_lane=tuple((i, i + 8) for i in range(8)),
            full_context_lens_elision=True,
            critic_step1_split_overlap=False, critic_b1_attribution=False,
            inventory=worker.CallInventory(), cycle_endpoint_records=[],
            step0_early_delivery=early_delivery,
        )
    assert len(calls) == 1
    assert calls[0]["rank"] == rank
    assert calls[0]["step0_early_delivery"] is early_delivery
    assert calls[0]["critic_b2_only"] is True
    assert calls[0]["critic_step1_split_overlap"] is False
    assert calls[0]["step_offset"] == 0


def test_default_transport_keys_unchanged_and_tags_unique():
    get = PinnedHostRunaheadTransport.expected_keys.fget
    old = SimpleNamespace(cycles=2, cross_cycle_lookahead=True)
    new = SimpleNamespace(cycles=2, cross_cycle_lookahead=True,
                          waves_per_step=2, step0_wave1_early_delivery=True)
    assert len(get(old)) == 17 and len(get(new)) == 19
    assert get(new) - get(old) == {(0, 1), (5, 1)}
    assert (0, 0) not in get(new) and (5, 0) in get(new)
    tags = [runahead_tag(*k, allow_step0_wave1=True) for k in sorted(get(new))]
    assert len(tags) == len(set(tags)) == 19 and min(tags) == 27101
    with pytest.raises(ValueError):
        runahead_tag(0, 1)
    for k in ((0, 0), (-1, 1)):
        with pytest.raises(ValueError):
            runahead_tag(*k, allow_step0_wave1=True)
    with pytest.raises(ValueError):
        runahead_tag(0, 1, allow_step0_wave1=True, waves_per_step=4)


@pytest.mark.parametrize("rank", [0, 7, 8, 15])
def test_actual_receipts_reject_partial_or_extra_operations(rank):
    is_g = rank < 8
    r = dict(enabled=True, name=CONTRACT["name"], role="generator" if is_g else "score",
             early_latent_sends=2 if is_g else 0,
             early_latent_receives=0 if is_g else 2,
             async_teacher0_sends=0 if is_g else 2,
             teacher0_send_waits=0 if is_g else 2)
    assert not runtime_receipt_errors(r, rank=rank, cycles=2)
    for key in ("early_latent_sends", "early_latent_receives", "async_teacher0_sends", "teacher0_send_waits"):
        bad = dict(r); bad[key] += 1
        assert runtime_receipt_errors(bad, rank=rank, cycles=2)
    assert transport_receipt_errors({}, rank=rank, cycles=2)


def test_reordered_transport_does_not_reorder_math_or_commit():
    s = inspect.getsource(worker._run_pipeline_8x8)
    early = s.index('label="livewan/pipeline/g/step_0/early_host_wave_1"')
    fake0 = s.index('label="livewan/pipeline/g/step_0/fake_velocity_wave_0"')
    recv0 = s.index('label="livewan/pipeline/g/step_0/recv_teacher_velocity_0"')
    fake1 = s.index('label="livewan/pipeline/g/step_0/fake_velocity_wave_1"')
    backward0 = s.index('label="livewan/pipeline/g/step_0/backward_wave_0"')
    backward1 = s.index('label="livewan/pipeline/g/step_0/backward_wave_1"')
    assert early < fake0 < recv0 < fake1 < backward0 < backward1
    assert "sends = ()" in s and "teacher0_work.wait()" in s
    assert s.index("teacher0_work.wait()") < s.index('label=f"livewan/pipeline/score/step_{step}/critic_commit"')
    assert "step0_wave1_early_delivery=step0_early_delivery" in inspect.getsource(worker.run_worker)
    assert "step0_early_delivery=step0_early_delivery" in inspect.getsource(worker.run_worker)


def test_controller_and_launcher_route_child():
    for rel in ("scripts/distillschedule/run_livewan_crossscale_world16.py",
                "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh",
                "scripts/distillschedule/probe_livewan_crossscale_transport.py",
                "scripts/distillschedule/build_livewan_two_cycle_timing_gate.py",
                "scripts/distillschedule/compare_livewan_two_cycle_trajectories.py"):
        assert VARIANT in (ROOT / rel).read_text()
    for rel in ("scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py",
                "scripts/distillschedule/summarize_livewan_trajectory_world16.py"):
        assert "step0_transport_errors" in (ROOT / rel).read_text()
