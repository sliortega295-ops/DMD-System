from __future__ import annotations

from pathlib import Path

from distillschedule.planner import (
    HardwareSpec,
    MemoryPolicy,
    ModelInventory,
    OffloadCandidate,
    PlanSimulator,
    PlannerProfile,
    PrimitiveMeasurement,
    ProfileDB,
    estimate_model_memory,
    load_plan,
    load_profile,
    replay_decomposed_observation,
    select_offload_blocks,
    simulate_two_stage_pipeline,
)
from distillschedule.profiling.planner_profile import (
    PlannerProfileRecorder,
    ProfileIdentity,
    load_profile_events,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/distillschedule/planner/helios_a100_profile_v0.json"
NOGAN_PLAN = ROOT / "configs/distillschedule/planner/plans/no_gan_rolephase_replay.json"
FSDP_PLAN = ROOT / "configs/distillschedule/planner/plans/gan_static_fsdp2_candidate.json"


def test_known_no_gan_components_reconstruct_full5_with_subpercent_error() -> None:
    profile = load_profile(PROFILE)
    observation = next(value for value in profile.plans if value.name == "no_gan_rolephase_measured")
    replay = replay_decomposed_observation(observation)
    assert replay["status"] == "PASS"
    assert replay["component_sum_seconds"] == 280.957
    assert replay["relative_error"] < 0.004

    result = PlanSimulator(ProfileDB(profile)).simulate(load_plan(NOGAN_PLAN))
    assert result.predicted_seconds == 280.957
    assert result.observed_median_seconds == 279.934
    assert result.relative_error is not None and result.relative_error < 0.004
    assert result.claimable_prediction is False
    assert "component" in result.notes[0]


def test_unseen_fsdp2_candidate_fails_closed_without_profile() -> None:
    result = PlanSimulator(ProfileDB(load_profile(PROFILE))).simulate(
        load_plan(FSDP_PLAN), allow_interpolation=False
    )
    assert result.predicted_seconds is None
    assert result.confidence == "insufficient_profile"
    assert result.coverage.required == 7
    assert len(result.coverage.missing) == 7
    assert result.claimable_prediction is False


def test_payload_credit_improves_bounded_pipeline_over_commit_credit() -> None:
    receive = simulate_two_stage_pipeline(
        producer_seconds=2.0,
        consumer_seconds=5.0,
        batches=6,
        queue_depth=1,
        credit="payload_receive",
    )
    commit = simulate_two_stage_pipeline(
        producer_seconds=2.0,
        consumer_seconds=5.0,
        batches=6,
        queue_depth=1,
        credit="consumer_commit",
    )
    assert receive["seconds"] < commit["seconds"]
    assert receive["producer_blocked_seconds"] < commit["producer_blocked_seconds"]


def test_memory_model_exposes_pinned_construction_duplicate_and_cpu_failure() -> None:
    hardware = HardwareSpec(
        name="test",
        nodes=2,
        gpus_per_node=8,
        gpu_hbm_bytes=40 * 2**30,
        node_cpu_limit_bytes=760 * 2**30,
    )
    model = ModelInventory(
        role="generator",
        parameter_bytes=28 * 2**30,
        frozen_parameter_bytes=27 * 2**30,
        trainable_parameter_bytes=1 * 2**30,
        buffer_bytes=1 * 2**30,
        block_parameter_bytes=tuple([690 * 2**20] * 40),
        trainable_numel=100_000_000,
    )
    estimate = estimate_model_memory(
        model,
        MemoryPolicy(
            backend="zero3",
            mesh_size=8,
            optimizer_offload=True,
            activation_cpu_bytes=300 * 2**30,
            persistent_pinned_target_copies=8,
            construction_duplicates_target=True,
        ),
        hardware,
        node_rank_copies=8,
    )
    assert estimate.cpu_breakdown["persistent_pinned_targets"] == 8 * 29 * 2**30
    assert estimate.cpu_breakdown["construction_duplicate"] == 8 * 29 * 2**30
    assert estimate.feasible_cpu is False


def test_offload_knapsack_selects_low_cost_blocks_not_tail_count() -> None:
    selection = select_offload_blocks(
        (
            OffloadCandidate(0, 100, 10.0),
            OffloadCandidate(1, 70, 1.0),
            OffloadCandidate(2, 50, 1.1),
            OffloadCandidate(3, 40, 0.9),
        ),
        required_free_bytes=120,
    )
    assert selection.feasible
    assert selection.indices == (1, 2)
    assert selection.freed_gpu_bytes == 120
    assert selection.added_seconds == 2.1


def test_profile_recorder_round_trip_without_cuda(tmp_path: Path) -> None:
    path = tmp_path / "rank0.jsonl"
    recorder = PlannerProfileRecorder(
        path,
        identity=ProfileIdentity(
            run_id="unit",
            workload="gan",
            rank=0,
            source_sha="a" * 40,
        ),
    )
    with recorder.span(
        operation="dit_forward",
        role="generator",
        backend="fsdp2",
        mesh_size=16,
        local_batch=1,
        stage=2,
        residency="sharded_resident",
    ) as receipt:
        receipt["collective_bytes"] = 1234
        receipt["metadata"] = {"test": True}
    events = load_profile_events([path])
    assert len(events) == 1
    assert events[0]["status"] == "PASS"
    assert events[0]["collective_bytes"] == 1234
    assert events[0]["metadata"]["test"] is True


def test_exact_fsdp2_primitives_turn_candidate_into_claimable_prediction() -> None:
    seed = load_profile(PROFILE)
    plan = load_plan(FSDP_PLAN)
    keys = plan.phases[0].primitive_keys
    measurements = []
    for key in {value.id: value for value in keys}.values():
        if key.operation == "generator_update_full":
            seconds = 50.0
        elif key.operation == "critic_update_full":
            seconds = 20.0
        else:
            seconds = 2.0
        measurements.append(
            PrimitiveMeasurement(
                key=key,
                wall_seconds=seconds,
                samples=3,
                provenance="synthetic unit calibration",
                confidence="measured",
            )
        )
    profile = PlannerProfile(
        hardware=seed.hardware,
        models=seed.models,
        primitives=tuple(measurements),
        transitions=seed.transitions,
        plans=seed.plans,
        metadata=seed.metadata,
    )
    result = PlanSimulator(ProfileDB(profile)).simulate(plan, allow_interpolation=False)
    assert result.predicted_seconds == 152.0
    assert result.coverage.required == 7
    assert result.coverage.exact == 7
    assert result.claimable_prediction is True
