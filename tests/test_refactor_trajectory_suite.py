from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.evidence.trajectory_coarse import TrajectoryCycleReceipt
from distillschedule.evidence.trajectory_suite import (
    EXPECTED_CYCLES,
    EXPECTED_WORLD_SIZE,
    MATRIX_RECORD_SCHEMA_VERSION,
    WORKLOAD_DEFINITIONS,
    AppendOnlyTrajectoryRunMatrixWriter,
    TrajectoryRunMatrixRecord,
    TrajectorySuiteArm,
    build_trajectory_suite_report,
    load_run_matrix_manifest,
    load_state_comparisons,
    write_trajectory_suite_report,
)


def _arm(
    root: Path | None,
    *,
    run_id: str,
    variant: str,
    head_character: str,
) -> TrajectorySuiteArm:
    return TrajectorySuiteArm(
        run_id=run_id,
        variant=variant,
        source_head=head_character * 40,
        rank_jsonl_root=None if root is None else str(root),
        assets={"config_sha256": head_character * 64},
        implementation_expectations={"arm": variant},
    )


def _record(
    tmp_path: Path,
    workload_id: str,
    *,
    status: str = "NOT_RUN",
    attempt: int = 1,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
) -> TrajectoryRunMatrixRecord:
    return TrajectoryRunMatrixRecord(
        suite_id="suite-50c-seed-20260817",
        workload_id=workload_id,
        attempt=attempt,
        recorded_at_utc=f"2026-09-03T12:00:{attempt:02d}Z",
        execution_status=status,
        reason="both arms completed"
        if status == "PASS"
        else "awaiting bounded GPU run",
        baseline=_arm(
            baseline_root,
            run_id=f"{workload_id}-base",
            variant="exact" if workload_id.startswith("minimax") else "baseline",
            head_character="1",
        ),
        candidate=_arm(
            candidate_root,
            run_id=f"{workload_id}-candidate",
            variant="grid1000" if workload_id.startswith("minimax") else "optimized",
            head_character="2",
        ),
        shared_assets={
            "model_sha256": "a" * 64,
            "trace_sha256": "b" * 64,
        },
    )


def _receipt(
    *,
    run_id: str,
    workload: str,
    variant: str,
    rank: int,
    cycle: int,
) -> TrajectoryCycleReceipt:
    return TrajectoryCycleReceipt(
        run_id=run_id,
        workload=workload,
        variant=variant,
        rank=rank,
        world_size=EXPECTED_WORLD_SIZE,
        seed=20260817,
        cycle=cycle,
        expected_total_cycles=EXPECTED_CYCLES,
        logical_step=10 + (cycle - 1) * 5,
        matched_contract={
            "model_sha256": "a" * 64,
            "trace_sha256": "b" * 64,
            "batch": "world16-b1",
        },
        implementation={"arm": variant},
        sample_identities=(f"cycle={cycle}/rank={rank}/sample=0",),
        rng_identities={"main": f"cycle={cycle}/rank={rank}/rng=main"},
        versions={"generator": cycle, "critic": cycle * 5},
        cycle_updates={"generator": 1, "critic": 5, "ema": 5},
        cumulative_updates={"generator": cycle, "critic": cycle * 5, "ema": cycle * 5},
        call_census={
            "generator_forward": 24,
            "critic_forward": 6,
            "teacher_forward": 1,
        },
        losses={
            "generator_total": 1.0 / (cycle + 1),
            "critic_total": 2.0 / (cycle + 1),
        },
        gradient_norms={"generator_pre_clip": 2.0 / (cycle + 1)},
        tensor_contracts={"latent": {"shape": [1, 16, 9, 48, 80], "dtype": "bf16"}},
        state_fingerprints={"generator": f"g-{cycle}"},
        finite=True,
    )


def _write_rank_root(
    root: Path, record: TrajectoryRunMatrixRecord, *, candidate: bool
) -> None:
    arm = record.candidate if candidate else record.baseline
    root.mkdir(parents=True)
    for rank in range(EXPECTED_WORLD_SIZE):
        rows = [
            _receipt(
                run_id=arm.run_id,
                workload=record.workload_id,
                variant=arm.variant,
                rank=rank,
                cycle=cycle,
            ).to_mapping()
            for cycle in range(1, EXPECTED_CYCLES + 1)
        ]
        (root / f"rank_{rank:02d}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _complete_records(tmp_path: Path) -> list[TrajectoryRunMatrixRecord]:
    records: list[TrajectoryRunMatrixRecord] = []
    for workload_id in WORKLOAD_DEFINITIONS:
        baseline_root = tmp_path / workload_id / "baseline"
        candidate_root = tmp_path / workload_id / "candidate"
        record = _record(
            tmp_path,
            workload_id,
            status="PASS",
            baseline_root=baseline_root,
            candidate_root=candidate_root,
        )
        _write_rank_root(baseline_root, record, candidate=False)
        _write_rank_root(candidate_root, record, candidate=True)
        records.append(record)
    return records


def test_record_schema_fixes_16x50_and_h3_approximation(tmp_path: Path) -> None:
    h3 = _record(tmp_path, "minimax_h3_exact_vs_grid1000")
    assert h3.schema_version == MATRIX_RECORD_SCHEMA_VERSION
    assert h3.expected_world_size == 16
    assert h3.expected_cycles == 50
    assert h3.comparison_kind == "approximation"
    mapping = h3.to_mapping()
    assert mapping["comparison_kind"] == "approximation"
    assert TrajectoryRunMatrixRecord.from_mapping(mapping) == h3

    mapping["comparison_kind"] = "semantic"
    with pytest.raises(ValueError, match="comparison_kind"):
        TrajectoryRunMatrixRecord.from_mapping(mapping)
    mapping = h3.to_mapping()
    mapping.pop("comparison_kind")
    with pytest.raises(ValueError, match="comparison_kind"):
        TrajectoryRunMatrixRecord.from_mapping(mapping)


def test_pass_requires_explicit_absolute_roots_and_exact_heads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit baseline and candidate"):
        _record(tmp_path, "wan1_3b_nogan", status="PASS")
    with pytest.raises(ValueError, match="40-character"):
        TrajectorySuiteArm(
            run_id="run",
            variant="baseline",
            source_head="deadbeef",
            assets={"model": "sha"},
        )
    with pytest.raises(ValueError, match="absolute path"):
        TrajectorySuiteArm(
            run_id="run",
            variant="baseline",
            source_head="a" * 40,
            assets={"model": "sha"},
            rank_jsonl_root="relative/ranks",
        )


def test_append_only_matrix_preserves_failure_and_rejects_identity_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_matrix.jsonl"
    first = _record(tmp_path, "wan1_3b_nogan", status="FAIL")
    writer = AppendOnlyTrajectoryRunMatrixWriter(path, suite_id=first.suite_id)
    writer.append(first)
    second_payload = first.to_mapping()
    second_payload.update(
        {
            "attempt": 2,
            "recorded_at_utc": "2026-09-03T12:00:02Z",
            "execution_status": "NOT_RUN",
            "reason": "retry intentionally not launched",
        }
    )
    writer.append(second_payload)
    loaded = load_run_matrix_manifest(path)
    assert [row.execution_status for row in loaded] == ["FAIL", "NOT_RUN"]
    assert path.read_text(encoding="utf-8").count("\n") == 2

    drift = dict(second_payload)
    drift["attempt"] = 3
    drift["recorded_at_utc"] = "2026-09-03T12:00:03Z"
    drift["candidate"] = dict(drift["candidate"])
    drift["candidate"]["source_head"] = "3" * 40
    before = path.read_bytes()
    with pytest.raises(ValueError, match="identity drifted"):
        writer.append(drift)
    assert path.read_bytes() == before


def test_complete_five_workload_matrix_calls_coarse_comparator(tmp_path: Path) -> None:
    report = build_trajectory_suite_report(_complete_records(tmp_path))
    assert report["status"] == "PASS"
    assert report["workload_count"] == 5
    assert all(pair["summary"]["complete"] for pair in report["pairs"])
    assert all(pair["summary"]["contract_status"] == "PASS" for pair in report["pairs"])
    h3 = next(
        pair
        for pair in report["pairs"]
        if pair["workload_id"] == "minimax_h3_exact_vs_grid1000"
    )
    assert h3["comparison_kind"] == "approximation"
    assert h3["comparison"]["status"] == "APPROXIMATION_CURVE_CLOSE_50C_SINGLE_SEED"
    assert h3["fixed_claim_boundary"] == "APPROXIMATION_NOT_EXACT_EQUIVALENCE"


def test_missing_rank_fails_only_that_pair_and_keeps_matrix(tmp_path: Path) -> None:
    records = _complete_records(tmp_path)
    first = records[0]
    assert first.candidate.rank_jsonl_root is not None
    (Path(first.candidate.rank_jsonl_root) / "rank_15.jsonl").unlink()
    report = build_trajectory_suite_report(records)
    assert report["status"] == "FAIL"
    failed = report["pairs"][0]
    assert failed["matrix_status"] == "FAIL"
    assert failed["summary"]["comparison_status"] == "FAIL_EVIDENCE"
    assert "exactly one file per rank" in failed["evidence_error"]
    assert all(pair["matrix_status"] == "PASS" for pair in report["pairs"][1:])


def test_state_comparison_loader_accepts_evaluated_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            {
                "state_comparisons": [
                    {
                        "schema_version": "distillschedule.trajectory_state_comparison.v1",
                        "cycle": 50,
                        "component": "generator",
                        "state_kind": "parameter_delta",
                        "cosine": 0.9995,
                        "normalized_l2": 0.01,
                        "norm_ratio": 1.0,
                        "min_named_tensor_cosine": 0.999,
                        "finite": True,
                        "reference_digest": None,
                        "candidate_digest": None,
                        "status": "PASS",
                        "thresholds": {"min_cosine": 0.999},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = load_state_comparisons([path])
    assert len(rows) == 1
    assert rows[0].cycle == 50


def test_report_writes_single_matrix_and_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = build_trajectory_suite_report(_complete_records(tmp_path / "inputs"))
    paths = write_trajectory_suite_report(report, tmp_path / "report")
    assert set(paths) == {"json", "csv", "markdown", "png", "sha256"}
    assert json.loads(paths["json"].read_text())["status"] == "PASS"
    assert len(paths["csv"].read_text().splitlines()) == 6
    markdown = paths["markdown"].read_text()
    assert "MiniMax-H3 Exact vs Grid1000" in markdown
    assert "not labelled exact-equivalent" in markdown
    assert paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError):
        write_trajectory_suite_report(report, tmp_path / "report")
