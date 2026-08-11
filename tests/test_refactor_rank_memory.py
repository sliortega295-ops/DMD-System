from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/distillschedule/summarize_routea_rank_memory.py"
    spec = importlib.util.spec_from_file_location("rank_memory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rank_memory"] = module
    spec.loader.exec_module(module)
    return module


MEMORY = load_script()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_rank(root: Path, rank: int) -> None:
    write_jsonl(
        root / "phase_trace" / f"rank_{rank:03d}.jsonl",
        [
            {
                "phase": "training_step_start",
                "cuda": {"allocated_bytes": 100},
            },
            {
                "phase": "model_swap_end",
                "incoming_component": "generator",
                "outgoing_component": "score",
                "cuda": {"allocated_bytes": 700},
            },
            {
                "phase": "semantic_generator_pre_backward",
                "cuda": {
                    "allocated_bytes": 900,
                    "reserved_bytes": 1000,
                    "active_bytes": 880,
                    "inactive_split_bytes": 20,
                    "driver_free_bytes": 200,
                },
            }
        ],
    )
    write_jsonl(
        root
        / "activation_offload"
        / f"rank_{rank:03d}_generator_selected.jsonl",
        [
            {
                "event": "pack",
                "policy": "KEEP_BELOW_THRESHOLD",
                "device": "cuda:0",
                "storage_id": ["cuda:0", 1000, 100],
                "storage_bytes": 100,
            },
            {
                "event": "pack",
                "policy": "KEEP_BELOW_THRESHOLD",
                "device": "cuda:0",
                "storage_id": ["cuda:0", 1000, 100],
                "storage_bytes": 100,
            },
            {
                "event": "pack",
                "policy": "KEEP_BELOW_THRESHOLD",
                "device": "cuda:0",
                "storage_id": ["cuda:0", 2000, 100],
                "storage_bytes": 100,
            },
            {
                "event": "scope_end",
                "stats": {
                    "scope_count": 1,
                    "kept_below_threshold_storage_bytes": 100,
                    "offloaded_storage_bytes": 200,
                },
            },
            {
                "event": "scope_end",
                "stats": {
                    "scope_count": 2,
                    "pack_count": 7,
                    "kept_below_threshold_count": 5,
                    "kept_below_threshold_storage_bytes": 300,
                    "offloaded_storage_bytes": 400,
                    "peak_live_cpu_bytes": 400,
                },
            },
        ],
    )
    write_jsonl(
        root / "block_streaming" / f"rank_{rank:03d}_generator.jsonl",
        [
            {
                "event": "streamer_installed",
                "streamed_block_indices": [0, 1],
                "streamed_block_count": 2,
            },
            {
                "event": "materialize_block",
                "block_index": 0,
                "bytes": 100,
                "elapsed_seconds": 1.0,
            },
            {
                "event": "materialize_block",
                "block_index": 1,
                "bytes": 100,
                "elapsed_seconds": 2.0,
            },
            {
                "event": "materialize_block",
                "block_index": 0,
                "bytes": 100,
                "elapsed_seconds": 3.0,
            },
        ],
    )
    write_jsonl(
        root / "immutable_model_backing" / f"rank_{rank:03d}_generator.jsonl",
        [
            {
                "event": "immutable_model_backing_installed",
                "stats": {
                    "shared_immutable_bytes": 400,
                    "private_parameter_bytes": 100,
                    "externally_managed_parameter_bytes": 200,
                },
            }
        ],
    )


def test_rank_memory_uses_cumulative_final_scope_without_summing(tmp_path: Path) -> None:
    make_rank(tmp_path, 0)
    result = MEMORY.summarize(tmp_path, 1)
    row = result["per_rank"][0]
    assert row["generator_saved_activation"]["kept_below_threshold_pack_storage_bytes"] == 300
    assert row["generator_saved_activation"]["unique_kept_below_threshold_storage_bytes"] == 200
    assert row["generator_saved_activation"]["alias_pack_amplification"] == 1.5
    assert row["generator_frozen_blocks"]["unique_streamed_capacity_bytes"] == 200
    assert row["generator_frozen_blocks"]["issued_materialization_bytes"] == 300
    assert row["generator_frozen_blocks"]["materialization_count"] == 3
    assert row["value_comparison"]["unique_kept_activation_equivalent_blocks"] == 2.0
    assert row["pre_backward"]["allocator_reserved_unallocated_bytes"] == 100
    assert row["generator_model_residency"]["known_model_resident_at_pre_backward_bytes"] == 500
    assert row["generator_model_residency"]["allocated_minus_known_model_bytes"] == 400
    assert row["generator_model_residency"]["unclassified_after_unique_kept_activation_bytes"] == 200
    assert row["phase_delta_attribution"]["persistent_runtime_floor_before_step_bytes"] == 100
    assert row["phase_delta_attribution"]["forward_incremental_to_pre_backward_bytes"] == 200
    assert row["phase_delta_attribution"]["forward_unclassified_after_unique_kept_activation_bytes"] == 0


def test_rank_memory_fails_closed_on_missing_rank(tmp_path: Path) -> None:
    make_rank(tmp_path, 0)
    with pytest.raises(RuntimeError, match="rank evidence mismatch"):
        MEMORY.summarize(tmp_path, 2)
