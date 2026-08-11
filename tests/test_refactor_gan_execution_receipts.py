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


def _tensor_keys(slot: int) -> list[str]:
    source = [
        f"sample/{slot}/primary/x0",
        f"sample/{slot}/primary/history",
        f"sample/{slot}/primary/target",
        f"sample/{slot}/secondary/x0",
        f"sample/{slot}/secondary/history",
        f"sample/{slot}/secondary/target",
        f"sample/{slot}/text_embedding",
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


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    trace_root = tmp_path / "trace"
    run_root = tmp_path / "run"
    (run_root / "trace_access").mkdir(parents=True)
    (run_root / "block_streaming").mkdir()
    bundle_ranks = []
    for slot in range(16):
        trace_hash = _sha(f"trace-{slot}")
        rank_root = trace_root / f"rank{slot}"
        rank_root.mkdir(parents=True)
        keys = _tensor_keys(slot)
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
        for component, calls in (("generator", 15), ("score", 14)):
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
        == 15
    )


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
