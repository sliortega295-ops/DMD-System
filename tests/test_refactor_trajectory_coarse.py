from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.evidence.trajectory_coarse import (
    COARSE_50C_THRESHOLDS,
    SCHEMA_VERSION,
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
    TrajectoryStateComparison,
    compare_trajectories,
    load_trajectory_jsonl,
    write_trajectory_report,
)
from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY


def _receipt(
    *,
    run_id: str,
    variant: str,
    rank: int,
    cycle: int,
    loss_scale: float = 1.0,
    matched_contract: dict[str, str] | None = None,
    runtime_status: str = "PASS",
    finite: bool = True,
) -> TrajectoryCycleReceipt:
    contract = matched_contract or {
        "bootstrap_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "trace_sha256": "c" * 64,
        "precision": "bf16",
    }
    return TrajectoryCycleReceipt(
        run_id=run_id,
        workload="synthetic-dmd",
        variant=variant,
        rank=rank,
        world_size=2,
        seed=20260817,
        cycle=cycle,
        expected_total_cycles=50,
        logical_step=1000 + cycle - 1,
        matched_contract=contract,
        implementation={
            "source_head": "baseline-head" if variant == "baseline" else "candidate-head",
            "config_sha256": "d" * 64 if variant == "baseline" else "e" * 64,
        },
        sample_identities=(f"rank={rank}/cycle={cycle}/sample=0",),
        rng_identities={"rollout": f"rank={rank}/cycle={cycle}/rng"},
        versions={"generator": cycle + 2, "critic": cycle * 5 + 10},
        cycle_updates={"generator": 1, "critic": 5, "ema": 5},
        cumulative_updates={"generator": cycle, "critic": cycle * 5, "ema": cycle * 5},
        call_census={"generator_forward": 24, "critic_forward": 6, "teacher_forward": 1},
        losses={
            "generator_total": loss_scale * (1.0 / cycle),
            "critic_0": loss_scale * (2.0 / (cycle + 1)),
        },
        gradient_norms={"generator_pre_clip": 2.0, "generator_post_clip": 1.0},
        tensor_contracts={
            "latent": {"shape": [1, 16, 9, 48, 80], "dtype": "torch.bfloat16"}
        },
        state_fingerprints={"generator": f"g-{cycle}", "critic": f"c-{cycle}"},
        finite=finite,
        runtime_status=runtime_status,
    )


def _rows(*, run_id: str, variant: str, loss_scale: float = 1.0):
    return [
        _receipt(
            run_id=run_id,
            variant=variant,
            rank=rank,
            cycle=cycle,
            loss_scale=loss_scale,
        )
        for cycle in range(1, 51)
        for rank in range(2)
    ]


def test_schema_is_independent_from_strict_p0_parity() -> None:
    assert SCHEMA_VERSION == "distillschedule.trajectory_coarse_v1"
    assert COARSE_50C_THRESHOLDS.min_cosine == pytest.approx(0.999)
    assert DEFAULT_P0_PARITY.min_cosine == pytest.approx(0.9999)
    assert COARSE_50C_THRESHOLDS is not DEFAULT_P0_PARITY


def test_headless_report_retains_all_numeric_evidence_without_plot_import(tmp_path, monkeypatch):
    import hashlib
    import sys
    from distillschedule.evidence import trajectory_coarse
    result = compare_trajectories(_rows(run_id="a", variant="baseline"),
                                  _rows(run_id="b", variant="optimized"),
                                  expected_cycles=50, expected_world_size=2)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setattr(trajectory_coarse, "_render_png", lambda *args: pytest.fail("plot called"))
    outputs = write_trajectory_report(result, tmp_path, render_png=False)
    assert set(outputs) == {"json", "csv", "markdown", "sha256"}
    assert json.loads(outputs["json"].read_text()) == result
    assert not (tmp_path / "trajectory_comparison.png").exists()
    hashes = json.loads(outputs["sha256"].read_text())
    for key in ("json", "csv", "markdown"):
        assert hashes[outputs[key].name] == hashlib.sha256(outputs[key].read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_trajectory_report(result, tmp_path, render_png=False)


def test_receipt_round_trip_and_rejects_nonfinite_json_values() -> None:
    receipt = _receipt(run_id="run-a", variant="baseline", rank=0, cycle=1)
    payload = receipt.to_mapping()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert TrajectoryCycleReceipt.from_mapping(payload) == receipt

    payload["losses"]["generator_total"] = float("nan")
    with pytest.raises(ValueError, match="losses.generator_total"):
        TrajectoryCycleReceipt.from_mapping(payload)

    payload = receipt.to_mapping()
    payload["matched_contract"]["trace_sha256"] = None
    with pytest.raises(TypeError, match="matched_contract.trace_sha256"):
        TrajectoryCycleReceipt.from_mapping(payload)


def test_append_only_writer_preserves_rows_and_rejects_rewind(tmp_path: Path) -> None:
    path = tmp_path / "rank_000.trajectory.jsonl"
    writer = AppendOnlyTrajectoryWriter(path, run_id="run-a", rank=0)
    first = _receipt(run_id="run-a", variant="baseline", rank=0, cycle=1)
    second = _receipt(run_id="run-a", variant="baseline", rank=0, cycle=2)
    writer.append(first)
    writer.append(second)

    rows = load_trajectory_jsonl(path)
    assert [row.cycle for row in rows] == [1, 2]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="strictly increase"):
        writer.append(second)
    assert path.read_bytes() == before

    with pytest.raises(ValueError, match="writer identity"):
        writer.append(_receipt(run_id="other", variant="baseline", rank=0, cycle=3))


def test_append_only_writer_fails_closed_on_corrupt_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "rank_000.trajectory.jsonl"
    path.write_text('{"partial":', encoding="utf-8")
    writer = AppendOnlyTrajectoryWriter(path, run_id="run-a", rank=0)
    with pytest.raises(ValueError, match="malformed JSONL|newline"):
        writer.append(_receipt(run_id="run-a", variant="baseline", rank=0, cycle=1))
    assert path.read_text(encoding="utf-8") == '{"partial":'


def test_append_only_writer_rejects_valid_last_row_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "rank_000.trajectory.jsonl"
    first = _receipt(run_id="run-a", variant="baseline", rank=0, cycle=1)
    path.write_text(json.dumps(first.to_mapping()), encoding="utf-8")
    writer = AppendOnlyTrajectoryWriter(path, run_id="run-a", rank=0)
    with pytest.raises(ValueError, match="newline"):
        writer.append(_receipt(run_id="run-a", variant="baseline", rank=0, cycle=2))


def test_compare_full_50_cycle_match_is_post_run_only() -> None:
    states = [
        TrajectoryStateComparison(
            cycle=cycle,
            component="generator",
            state_kind="parameter_delta",
            cosine=0.9995,
            normalized_l2=0.01,
            norm_ratio=1.01,
            min_named_tensor_cosine=0.9992,
        )
        for cycle in (1, 10, 25, 50)
    ]
    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        _rows(run_id="opt", variant="optimized"),
        state_comparisons=states,
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["status"] == "TRAJECTORY_MATCHED_50C_SINGLE_SEED"
    assert result["complete"] is True
    assert result["evaluation_policy"] == "POST_RUN_ONLY_NO_NUMERICAL_EARLY_STOP"
    assert result["contract"]["status"] == "PASS"
    assert result["curves"]["generator_total"]["status"] == "PASS"
    assert result["gradient_norm_curves"]["generator_pre_clip"]["status"] == "PASS"
    assert len(result["curves"]["generator_total"]["points"]) == 50


def test_soft_curve_drift_completes_and_returns_review() -> None:
    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        _rows(run_id="opt", variant="optimized", loss_scale=1.3),
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["status"] == "TRAJECTORY_REVIEW_50C_COMPLETED"
    assert result["complete"] is True
    assert result["curves"]["generator_total"]["normalized_rmse"] > 0.1
    assert len(result["curves"]["generator_total"]["points"]) == 50


def test_contract_mismatch_is_reported_after_complete_run() -> None:
    candidate = _rows(run_id="opt", variant="optimized")
    changed = candidate[20].to_mapping()
    changed["call_census"]["generator_forward"] = 23
    candidate[20] = TrajectoryCycleReceipt.from_mapping(changed)

    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        candidate,
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["status"] == "CONTRACT_MISMATCH_50C_COMPLETED"
    assert result["complete"] is True
    assert any("call_census" in row for row in result["contract"]["mismatches"])
    assert len(result["curves"]["generator_total"]["points"]) == 50


def test_missing_cycle_is_runtime_fail_not_partial_curve_claim() -> None:
    candidate = _rows(run_id="opt", variant="optimized")[:-1]
    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        candidate,
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["status"] == "RUNTIME_FAIL"
    assert result["complete"] is False
    assert result["curves"] == {}


def test_constant_near_zero_curve_marks_correlation_not_applicable() -> None:
    reference = _rows(run_id="base", variant="baseline")
    candidate = _rows(run_id="opt", variant="optimized")
    reference = [
        TrajectoryCycleReceipt.from_mapping(
            {
                **row.to_mapping(),
                "losses": {"zero_loss": 0.0},
            }
        )
        for row in reference
    ]
    candidate = [
        TrajectoryCycleReceipt.from_mapping(
            {
                **row.to_mapping(),
                "losses": {"zero_loss": 0.0},
            }
        )
        for row in candidate
    ]
    result = compare_trajectories(
        reference,
        candidate,
        expected_cycles=50,
        expected_world_size=2,
    )
    curve = result["curves"]["zero_loss"]
    assert curve["status"] == "PASS"
    assert curve["moving_average_correlation"] is None
    assert curve["moving_average_correlation_policy"] == "N/A_NEAR_CONSTANT"
    assert curve["tail_symmetric_relative_difference"] is None
    assert curve["tail_difference_policy"] == "N/A_BOTH_NEAR_ZERO"


def test_report_encodes_near_zero_reference_nonzero_candidate_without_infinity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    reference = [
        TrajectoryCycleReceipt.from_mapping(
            {**row.to_mapping(), "losses": {"near_zero": 0.0}}
        )
        for row in _rows(run_id="base", variant="baseline")
    ]
    candidate = [
        TrajectoryCycleReceipt.from_mapping(
            {**row.to_mapping(), "losses": {"near_zero": 1.0}}
        )
        for row in _rows(run_id="opt", variant="optimized")
    ]
    result = compare_trajectories(
        reference,
        candidate,
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["curves"]["near_zero"]["normalized_rmse"] is None
    paths = write_trajectory_report(result, tmp_path)
    assert "Infinity" not in paths["json"].read_text()


def test_state_cosine_below_point_999_is_review_not_contract_failure() -> None:
    state = TrajectoryStateComparison(
        cycle=50,
        component="generator",
        state_kind="gradient",
        cosine=0.9989,
        normalized_l2=0.01,
        norm_ratio=1.0,
        min_named_tensor_cosine=0.9988,
    )
    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        _rows(run_id="opt", variant="optimized"),
        state_comparisons=[state],
        expected_cycles=50,
        expected_world_size=2,
    )
    assert result["status"] == "TRAJECTORY_REVIEW_50C_COMPLETED"
    assert result["contract"]["status"] == "PASS"
    assert result["state_comparisons"][0]["status"] == "REVIEW"


def test_approximation_comparison_has_distinct_success_claim() -> None:
    result = compare_trajectories(
        _rows(run_id="exact", variant="exact-continuous"),
        _rows(run_id="grid", variant="grid1000"),
        expected_cycles=50,
        expected_world_size=2,
        comparison_kind="approximation",
    )
    assert result["status"] == "APPROXIMATION_CURVE_CLOSE_50C_SINGLE_SEED"
    assert result["claim_boundary"] == "APPROXIMATION_NOT_EXACT_EQUIVALENCE"


def test_report_helper_writes_json_csv_markdown_and_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    result = compare_trajectories(
        _rows(run_id="base", variant="baseline"),
        _rows(run_id="opt", variant="optimized"),
        expected_cycles=50,
        expected_world_size=2,
    )
    paths = write_trajectory_report(result, tmp_path)
    assert set(paths) == {"json", "csv", "markdown", "png", "sha256"}
    assert json.loads(paths["json"].read_text())["status"] == result["status"]
    assert paths["csv"].read_text().splitlines()[0].startswith("metric,cycle")
    assert paths["markdown"].read_text().startswith("# Trajectory coarse comparison")
    assert paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    digest = json.loads(paths["sha256"].read_text())
    assert set(digest) == {
        "trajectory_comparison.json",
        "loss_curves.csv",
        "trajectory_comparison.md",
        "trajectory_comparison.png",
    }

    with pytest.raises(FileExistsError):
        write_trajectory_report(result, tmp_path)
