from __future__ import annotations

from pathlib import Path

from distillschedule.adapters.livewan.adapter import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_host_runahead import (
    PinnedHostRunaheadTransport,
    runahead_tag,
)
from distillschedule.workloads.livewan_crossscale import (
    TRACE_BALANCED_8G8S_SLOT_PAIRS,
    validate_slot_groups,
)


ROOT = Path(__file__).resolve().parents[1]
C23 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c23_continuous_b32.json"


def test_c23_is_explicit_source_reuse_b32_only(monkeypatch) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")
    plan = load_plan(C23).plan
    LiveWanSFDMDAdapter().validate(plan)
    value = plan.to_dict()
    assert value["workload"]["shape"]["batch_size"] == 32
    assert value["workload"]["metadata"]["global_batch"] == 32
    assert value["metadata"]["batch32_diagnostic"] == {
        "enabled": True,
        "name": "matched_b32_source_reuse_systems_v1",
        "source_reuse_within_step": True,
        "unique_source_samples_per_cycle": 80,
        "rank_qualified_samples_per_cycle": 160,
        "training_quality_claimable": False,
    }
    expected = tuple(
        (first, second, first + 16, second + 16)
        for first, second in TRACE_BALANCED_8G8S_SLOT_PAIRS
    )
    assert validate_slot_groups(
        value["metadata"]["slot_groups_by_lane"],
        batch=32,
        slots_per_lane=4,
    ) == expected
    assert value["metadata"]["generator_runahead_transport"]["waves_per_step"] == 4


def test_b32_host_transport_inventory_is_bounded() -> None:
    instance = type(
        "TransportContract",
        (),
        {"cycles": 3, "cross_cycle_lookahead": True, "waves_per_step": 4},
    )()
    keys = PinnedHostRunaheadTransport.expected_keys.fget(instance)
    assert len(keys) == 50
    assert {(5, 0), (10, 0)}.issubset(keys)
    assert (5, 1) not in keys and (10, 1) not in keys
    assert runahead_tag(1, 3, waves_per_step=4) == 27107


def test_b32_worker_keeps_two_streamers_and_single_commits() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _run_pipeline_8x8_b32")
    end = source.index("def _run_pipeline_8x8_continuous", start)
    candidate = source[start:end]
    assert "len(streamers) == 2" in candidate
    assert "sample_weight=0.25" in candidate
    assert "loss_scale=0.5" in candidate
    assert "reducer_accumulation=(microbatch, 2)" in candidate
    assert candidate.count('versions["generator"] += 1') == 1
    assert candidate.count('versions["critic"] += 1') == 1


def test_b32_trace_builder_discloses_non_training_claim() -> None:
    source = (
        ROOT / "scripts/distillschedule/build_livewan_continuous_b32_trace.py"
    ).read_text(encoding="utf-8")
    for text in (
        "source_reuse_within_step=True",
        "training_quality_claimable=False",
        "rank_qualified_samples_per_cycle=160",
        "not convergence",
    ):
        assert text in source
