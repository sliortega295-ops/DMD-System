from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from distillschedule.runtime.gan_execution_receipts import (
    build_gan_execution_receipts,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tensor_keys(slot: int, *, local_source_index: bool = False) -> list[str]:
    source_slot = 0 if local_source_index else slot
    source = [
        f"sample/{source_slot}/primary/x0",
        f"sample/{source_slot}/primary/history",
        f"sample/{source_slot}/primary/target",
        f"sample/{source_slot}/secondary/x0",
        f"sample/{source_slot}/secondary/history",
        f"sample/{source_slot}/secondary/target",
        f"sample/{source_slot}/text_embedding",
    ]
    keys = list(source)
    history = (
        "continuous_history",
        "history_short",
        "history_mid",
        "history_long",
        "indices_hidden_states",
        "indices_history_short",
        "indices_history_mid",
        "indices_history_long",
    )
    for step in range(1000, 1005):
        for suffix in history:
            keys.append(f"step/{step}/critic/{suffix}")
        keys.extend(
            [
                f"step/{step}/critic/entry_noise",
                f"step/{step}/critic/rollout_noise",
                f"step/{step}/critic/block_noise/1",
                f"step/{step}/critic/block_noise/2",
                f"step/{step}/critic/critic_timestep",
                f"step/{step}/critic/critic_noise",
                f"step/{step}/critic/gan_d_fake_crop",
                f"step/{step}/critic/gan_d_real_crop",
                f"step/{step}/critic/gan_d_r1_crop",
                f"step/{step}/critic/r1_epsilon",
            ]
        )
    for suffix in history:
        keys.append(f"step/1000/generator/{suffix}")
    keys.extend(
        [
            "step/1000/generator/entry_noise",
            "step/1000/generator/rollout_noise",
            "step/1000/generator/block_noise/1",
            "step/1000/generator/block_noise/2",
            "step/1000/generator/dmd_timestep",
            "step/1000/generator/dmd_noise",
            "step/1000/generator/gan_g_noise",
            "step/1000/generator/gan_g_crop",
        ]
    )
    return keys


def _rollout(key: str) -> dict:
    return {
        "key": key,
        "operation": "rollout",
        "control": {
            "stage_schedule_scope": "global_rank0_broadcast_equivalent",
            "call_count": 3,
            "pyramid_stage_call_sequence": [
                {"stage": stage, "denoising_exit": 0, "call_count": 1}
                for stage in range(3)
            ],
        },
        "tensor_keys": [],
    }


def _make_fixture(
    tmp_path: Path, *, local_source_index: bool = False
) -> tuple[Path, Path]:
    trace_root = tmp_path / "trace"
    run_root = tmp_path / "run"
    (run_root / "trace_access").mkdir(parents=True)
    (run_root / "block_streaming").mkdir()
    bundle_ranks = []
    for slot in range(16):
        trace_hash = _sha(f"trace-{slot}")
        rank_root = trace_root / f"rank{slot}"
        rank_root.mkdir(parents=True)
        keys = _tensor_keys(slot, local_source_index=local_source_index)
        operations = [_rollout("step/1000/generator/rollout")]
        operations.extend(
            _rollout(f"step/{step}/critic/rollout")
            for step in range(1000, 1005)
        )
        trace = {
            "trace_hash": trace_hash,
            "operations": operations,
            "tensors": {key: {"sha256": _sha(key)} for key in keys},
        }
        (rank_root / "trace.json").write_text(json.dumps(trace))
        bundle_ranks.append(
            {"rank": slot, "path": f"rank{slot}", "trace_hash": trace_hash}
        )
        access_path = run_root / "trace_access" / f"rank_{slot:03d}.jsonl"
        with access_path.open("w") as handle:
            for key in keys:
                handle.write(
                    json.dumps(
                        {
                            "rank": slot,
                            "key": key,
                            "sha256": _sha(key),
                            "trace_hash": trace_hash,
                        }
                    )
                    + "\n"
                )
        # Generator GPUs execute one differentiable rollout at step1000 plus
        # one Critic-data rollout at every step1000--1004: 3 + 5*3 = 18.
        for component, calls in (("generator", 18), ("score", 14)):
            path = run_root / "block_streaming" / f"rank_{slot:03d}_{component}.jsonl"
            rows = [
                {
                    "event": "streamer_installed",
                    "streamed_block_indices": [39],
                }
            ]
            for _ in range(calls):
                rows.append(
                    {
                        "event": "materialize_block",
                        "block_index": 39,
                        "generation": "EAGER_FORWARD",
                    }
                )
                rows.append({"event": "release_block", "block_index": 39})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace_root.mkdir(exist_ok=True)
    (trace_root / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": "distillschedule.operation_trace_bundle.v1",
                "bundle_hash": _sha("bundle"),
                "ranks": bundle_ranks,
            }
        )
    )
    return run_root, trace_root


def test_actual_consumption_and_physical_call_receipts_pass(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path)
    built = build_gan_execution_receipts(
        family="baseline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_streamed_block_count=1,
    )
    assert built["provenance"]["status"] == "PASS"
    assert len(built["fairness"]["entries"]) == 80
    assert len(built["compute"]["entries"]) == 80
    assert (
        built["provenance"]["physical_call_proof"]["rank_000_generator"][
            "observed_logical_forwards_per_block"
        ]
        == 18
    )


def test_rank_qualified_trace_may_use_local_sample_zero_keys(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    built = build_gan_execution_receipts(
        family="baseline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_streamed_block_count=1,
    )
    assert built["provenance"]["status"] == "PASS"
    assert len(built["fairness"]["entries"]) == 80


def test_manifest_value_without_actual_history_consumption_fails(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path)
    path = run_root / "trace_access" / "rank_000.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["key"]
        != "step/1000/generator/indices_hidden_states"
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="was not actually consumed"):
        build_gan_execution_receipts(
            family="baseline",
            run_root=run_root,
            trace_bundle=trace_root,
            expected_streamed_block_count=1,
        )


def test_physical_call_shortcut_is_rejected(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path)
    path = run_root / "block_streaming" / "rank_000_score.jsonl"
    rows = path.read_text().splitlines()
    # Remove one materialize/release pair while preserving balanced lifecycle.
    path.write_text("\n".join(rows[:-2]) + "\n")
    with pytest.raises(ValueError, match="physical logical call count mismatch"):
        build_gan_execution_receipts(
            family="baseline",
            run_root=run_root,
            trace_bundle=trace_root,
            expected_streamed_block_count=1,
        )


def _convert_fixture_to_pipeline(run_root: Path, trace_root: Path) -> None:
    access_root = run_root / "trace_access"
    for path in access_root.glob("rank_*.jsonl"):
        path.unlink()
    handles = {
        rank: (access_root / f"rank_{rank:03d}.jsonl").open("w")
        for rank in range(16)
    }
    try:
        for slot in range(16):
            trace = json.loads((trace_root / f"rank{slot}" / "trace.json").read_text())
            generator_rank = slot % 8
            score_rank = 8 + generator_rank
            for key, descriptor in trace["tensors"].items():
                if key.startswith("sample/"):
                    ranks = (generator_rank, score_rank)
                elif "/generator/" in key:
                    if key.endswith(("dmd_timestep", "dmd_noise", "gan_g_noise", "gan_g_crop")):
                        ranks = (score_rank,)
                    elif key.endswith(
                        (
                            "continuous_history",
                            "history_short",
                            "history_mid",
                            "history_long",
                            "indices_hidden_states",
                            "indices_history_short",
                            "indices_history_mid",
                            "indices_history_long",
                        )
                    ):
                        ranks = (generator_rank, score_rank)
                    else:
                        ranks = (generator_rank,)
                elif "/critic/" in key:
                    if key.endswith(
                        (
                            "critic_timestep",
                            "critic_noise",
                            "gan_d_fake_crop",
                            "gan_d_real_crop",
                            "gan_d_r1_crop",
                            "r1_epsilon",
                        )
                    ):
                        ranks = (score_rank,)
                    elif key.endswith(
                        (
                            "continuous_history",
                            "history_short",
                            "history_mid",
                            "history_long",
                            "indices_hidden_states",
                            "indices_history_short",
                            "indices_history_mid",
                            "indices_history_long",
                            "entry_noise",
                        )
                    ):
                        ranks = (generator_rank, score_rank)
                    else:
                        ranks = (generator_rank,)
                else:
                    raise AssertionError(key)
                for rank in ranks:
                    handles[rank].write(
                        json.dumps(
                            {
                                "rank": rank,
                                "key": key,
                                "sha256": descriptor["sha256"],
                                "trace_hash": trace["trace_hash"],
                            }
                        )
                        + "\n"
                    )
    finally:
        for handle in handles.values():
            handle.close()

    old = run_root / "block_streaming"
    for path in old.glob("*.jsonl"):
        path.unlink()
    old.rmdir()
    block_root = run_root / "results" / "block_streaming"
    block_root.mkdir(parents=True)
    for rank, component, calls in (
        *((rank, "generator", 36) for rank in range(8)),
        *((rank, "score", 28) for rank in range(8, 16)),
    ):
        rows = [{"event": "streamer_installed", "streamed_block_indices": [39]}]
        for _ in range(calls):
            rows.extend(
                (
                    {
                        "event": "materialize_block",
                        "block_index": 39,
                        "generation": "EAGER_FORWARD",
                    },
                    {"event": "release_block", "block_index": 39},
                )
            )
        (block_root / f"rank_{rank:03d}_{component}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_pipeline_actual_consumption_and_nested_physical_receipts_pass(
    tmp_path: Path,
) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    _convert_fixture_to_pipeline(run_root, trace_root)
    built = build_gan_execution_receipts(
        family="pipeline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_streamed_block_count=1,
    )
    assert built["provenance"]["status"] == "PASS"
    assert (
        built["provenance"]["physical_call_proof"]["rank_000_generator"][
            "observed_logical_forwards_per_block"
        ]
        == 36
    )
    assert (
        built["provenance"]["physical_call_proof"]["rank_008_score"][
            "observed_logical_forwards_per_block"
        ]
        == 28
    )


def test_pipeline_generator_phase_residency_is_reconciled_separately(
    tmp_path: Path,
) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    _convert_fixture_to_pipeline(run_root, trace_root)
    block_root = run_root / "results" / "block_streaming"
    for rank in range(8):
        rows = [{"event": "streamer_installed", "streamed_block_indices": [39]}]
        # Two logical samples, each with three Step1000 train-rollout calls.
        for _ in range(6):
            rows.extend(
                (
                    {
                        "event": "materialize_block",
                        "block_index": 39,
                        "generation": "EAGER_FORWARD",
                    },
                    {"event": "release_block", "block_index": 39},
                )
            )
        rows.extend(
            (
                {"event": "phase_resident_enter"},
                {
                    "event": "materialize_block",
                    "block_index": 39,
                    "generation": "PHASE_RESIDENT_NO_GRAD",
                },
                {"event": "release_block", "block_index": 39},
                {"event": "phase_resident_exit"},
            )
        )
        (block_root / f"rank_{rank:03d}_generator.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    built = build_gan_execution_receipts(
        family="pipeline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_generator_streamed_block_count=1,
        expected_score_streamed_block_count=1,
        generator_critic_phase_resident=True,
    )
    proof = built["provenance"]["physical_call_proof"]["rank_000_generator"]
    assert proof["expected_logical_forwards"] == 36
    assert proof["expected_eager_logical_forwards"] == 6
    assert proof["expected_no_grad_logical_forwards"] == 30
    assert proof["observed_phase_materializations_per_block"] == 1
    assert proof["generator_critic_phase_resident"] is True


def test_pipeline_role_specific_resident_score_is_proved_from_rank_results(
    tmp_path: Path,
) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    _convert_fixture_to_pipeline(run_root, trace_root)
    block_root = run_root / "results" / "block_streaming"
    for rank in range(8, 16):
        (block_root / f"rank_{rank:03d}_score.jsonl").unlink()
    zero_stats = {
        "captured_parameter_count": 0,
        "captured_bytes": 0,
        "streamed_block_count": 0,
        "materialize_count": 0,
        "materialized_bytes": 0,
        "release_count": 0,
        "released_bytes": 0,
        "resident_bytes": 0,
    }
    result_root = run_root / "results"
    for rank in range(16):
        (result_root / f"rank_{rank:03d}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank": rank,
                    "role": "generator" if rank < 8 else "score",
                    "streamer_stats": zero_stats,
                    "streaming_policy": {
                        "streamed_block_count": 1 if rank < 8 else 0,
                        "generator_critic_phase_resident": False,
                    },
                }
            )
        )
    built = build_gan_execution_receipts(
        family="pipeline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_generator_streamed_block_count=1,
        expected_score_streamed_block_count=0,
    )
    proof = built["provenance"]["physical_call_proof"]
    assert proof["rank_000_generator"]["streamed_block_indices"] == [39]
    assert proof["rank_008_score"]["streamed_block_indices"] == []


def test_pipeline_missing_shared_history_consumer_fails(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    _convert_fixture_to_pipeline(run_root, trace_root)
    path = run_root / "trace_access" / "rank_008.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if not (
            json.loads(line)["trace_hash"] == _sha("trace-0")
            and json.loads(line)["key"] == "step/1000/generator/continuous_history"
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="physical-consumer set drift"):
        build_gan_execution_receipts(
            family="pipeline",
            run_root=run_root,
            trace_bundle=trace_root,
            expected_streamed_block_count=1,
        )


def test_pipeline_resident_k0_proves_no_streamer_events(tmp_path: Path) -> None:
    run_root, trace_root = _make_fixture(tmp_path, local_source_index=True)
    _convert_fixture_to_pipeline(run_root, trace_root)
    block_root = run_root / "results" / "block_streaming"
    for path in block_root.glob("*.jsonl"):
        path.unlink()
    block_root.rmdir()
    zero_stats = {
        "captured_parameter_count": 0,
        "captured_bytes": 0,
        "streamed_block_count": 0,
        "materialize_count": 0,
        "materialized_bytes": 0,
        "release_count": 0,
        "released_bytes": 0,
        "resident_bytes": 0,
    }
    result_root = run_root / "results"
    for rank in range(16):
        (result_root / f"rank_{rank:03d}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank": rank,
                    "role": "generator" if rank < 8 else "score",
                    "streamer_stats": zero_stats,
                }
            )
        )
    built = build_gan_execution_receipts(
        family="pipeline",
        run_root=run_root,
        trace_bundle=trace_root,
        expected_streamed_block_count=0,
    )
    proof = built["provenance"]["physical_call_proof"]
    assert len(proof) == 16
    assert proof["rank_000_generator"]["streamed_block_indices"] == []
    assert proof["rank_008_score"]["physical_call_accounting"] == (
        "resident_model_no_stream_events"
    )
