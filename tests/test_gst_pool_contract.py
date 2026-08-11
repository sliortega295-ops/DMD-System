from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from runtime.gst_pool_contract import (
    NEGATIVE_PROMPT_SCHEMA,
    GeneratorOffloadCandidate,
    NegativePromptAsset,
    source_identity,
    write_negative_prompt_asset,
)
from runtime.gst_pool_exchange import (
    atomic_torch_save,
    exchange_paths,
    lane_source_ids,
    make_input_payload,
    validate_input_payload,
)
from runtime.gst_pool_trace import EXPECTED_TRACE_CONTRACT, validate_gst_trace_bundle
from runtime.operation_trace import (
    OperationTraceWriter,
    write_operation_trace_bundle_manifest,
)
from scripts.distillschedule import summarize_gst_pool_candidate
from scripts.distillschedule.probe_gst_pool_generator import _optimizer_config


def _candidate(tmp_path: Path, **changes) -> Path:
    value = {
        "schema_version": "distillschedule.g_offload_candidate.v1",
        "run_id": "candidate-a",
        "execution_graph": "target_no_gan_gst_pool",
        "student_residency": "resident",
        "teacher_residency": "resident",
        "generator_offload_blocks": [0, 3, 7],
        "generator_offload_block_count": 3,
        "order_digest": None,
        "contract_digest": "a" * 64,
        "activation_policy_mutable": False,
        "loss_optimizer_batch_trace_version_ema_mutable": False,
        "historical_candidate_substitution_allowed": False,
    }
    value.update(changes)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    return path


def test_candidate_is_fail_closed(tmp_path: Path) -> None:
    parsed = GeneratorOffloadCandidate.load(
        _candidate(tmp_path), expected_contract_digest="a" * 64
    )
    assert parsed.streamed_blocks == (0, 3, 7)
    assert len(parsed.manifest_sha256) == 64

    bad = _candidate(
        tmp_path,
        generator_offload_blocks=[0, 0],
        generator_offload_block_count=2,
    )
    with pytest.raises(ValueError, match="duplicates"):
        GeneratorOffloadCandidate.load(bad)


def test_generator_runtime_precision_matches_native_world8_checkpoint() -> None:
    config = _optimizer_config()
    assert config["fp16"]["enabled"] is False
    assert config["bf16"]["enabled"] is True
    assert config["zero_optimization"]["stage"] == 1


def test_source_identity_is_two_rank_qualified_waves() -> None:
    assert source_identity(0) == {
        "source_id": 0,
        "generator_rank": 0,
        "generator_wave": 0,
        "generator_world_size": 8,
        "generator_microbatches": 2,
        "global_batch": 16,
    }
    assert source_identity(15)["generator_rank"] == 7
    assert source_identity(15)["generator_wave"] == 1
    with pytest.raises(ValueError):
        source_identity(16)


def test_capacity_exchange_keeps_rank_identity_and_is_immutable(tmp_path: Path) -> None:
    assert lane_source_ids(0) == (0, 4, 8, 12)
    x0 = torch.zeros((1, 16, 9, 48, 80), dtype=torch.bfloat16)
    payload = make_input_payload(
        run_id="run",
        source_id=9,
        generator_version=3,
        trace_identity_sha256="b" * 64,
        x0=x0,
    )
    recovered = validate_input_payload(
        payload,
        run_id="run",
        source_id=9,
        trace_identity_sha256="b" * 64,
    )
    assert torch.equal(recovered, x0)
    path = exchange_paths(tmp_path, 9)["input"]
    atomic_torch_save(path, payload)
    with pytest.raises(FileExistsError):
        atomic_torch_save(path, payload)


def test_negative_prompt_asset_round_trip_and_rejects_drift(tmp_path: Path) -> None:
    root = tmp_path / "negative"
    tensor = torch.arange(512 * 4096, dtype=torch.int32).reshape(1, 512, 4096)
    tensor = (tensor % 127).to(torch.bfloat16)
    written = write_negative_prompt_asset(
        root, embedding=tensor, model_root=tmp_path / "weights"
    )
    assert written.manifest["schema_version"] == NEGATIVE_PROMPT_SCHEMA
    assert torch.equal(NegativePromptAsset.load(root).embedding, tensor)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        NegativePromptAsset.load(root)


def _write_rank_trace(root: Path, rank: int) -> tuple[str, str]:
    trace_root = root / f"rank{rank}"
    writer = OperationTraceWriter(
        classification="SYN-FEATURE", source={"rank": rank}
    )
    keys = {
        "sample/0/text_embedding",
        "sample/0/attention_mask",
        "step/15/generator/continuous_history",
        "step/15/generator/history_short",
        "step/15/generator/history_mid",
        "step/15/generator/history_long",
        "step/15/generator/indices_hidden_states",
        "step/15/generator/indices_history_short",
        "step/15/generator/indices_history_mid",
        "step/15/generator/indices_history_long",
        "step/15/generator/entry_noise",
        "step/15/generator/rollout_noise",
        "step/15/generator/block_noise/1",
        "step/15/generator/block_noise/2",
        "step/15/generator/dmd_timestep",
        "step/15/generator/dmd_noise",
    }
    for key in sorted(keys):
        writer.add_tensor(key, torch.tensor([rank]), role=key)
    writer.add_operation(
        "step/15/branch",
        component="control",
        operation="stage3_branch",
        control={
            "global_step": 15,
            "dfake_gen_update_ratio": 5,
            "train_generator": True,
            "train_critic": True,
            "gan": False,
            "reward": False,
            "vae_decode": False,
        },
    )
    writer.add_operation(
        "step/15/generator/rollout",
        component="generator",
        operation="selected_rollout",
        tensor_keys=sorted(key for key in keys if "dmd_" not in key),
        control={
            "gradient_enabled": True,
            "call_count": 4,
            "initial_synchronized_pyramid_stage_exit": [0, 1, 0],
            "pyramid_stage_call_sequence": [
                {"stage": 0, "denoising_exit": 0, "call_count": 1},
                {"stage": 1, "denoising_exit": 1, "call_count": 2},
                {"stage": 2, "denoising_exit": 0, "call_count": 1},
            ],
            "stage_schedule_scope": "global_rank0_broadcast_equivalent",
        },
    )
    writer.add_operation(
        "step/15/generator/dmd",
        component="score_shared_backbone",
        operation="detached_dmd_gradient",
        tensor_keys=[
            "step/15/generator/dmd_timestep",
            "step/15/generator/dmd_noise",
        ],
    )
    trace = writer.write(trace_root, contract=EXPECTED_TRACE_CONTRACT)
    return f"rank{rank}", trace.trace_hash


def test_target_trace_bundle_validates_all_16_rank_controls(tmp_path: Path) -> None:
    entries = []
    for rank in range(16):
        path, trace_hash = _write_rank_trace(tmp_path, rank)
        entries.append({"rank": rank, "path": path, "trace_hash": trace_hash})
    bundle = write_operation_trace_bundle_manifest(tmp_path, entries)
    result = validate_gst_trace_bundle(tmp_path)
    assert result["status"] == "PASS"
    assert result["bundle_hash"] == bundle["bundle_hash"]
    assert result["rank_count"] == 16


def test_target_candidate_summarizer_requires_all_role_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = _candidate(tmp_path)
    candidate = GeneratorOffloadCandidate.load(candidate_path)
    run_root = tmp_path / "run"
    rank_root = run_root / "rank_results"

    def samples(source_ids: tuple[int, ...]) -> list[dict[str, object]]:
        return [
            {
                "source_id": source_id,
                "trace_identity_sha256": f"trace-{source_id}",
                "x0_sha256": f"x0-{source_id}",
            }
            for source_id in source_ids
        ]

    common = {
        "status": "PASS",
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "contract_digest": candidate.contract_digest,
        "memory": [],
    }
    for rank in range(8):
        path = rank_root / "generator" / f"rank_{rank:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **common,
                    "rank": rank,
                    "streamed_blocks": list(candidate.streamed_blocks),
                    "waves": samples((rank, rank + 8)),
                    "final": {
                        "generator_version_before": 3,
                        "generator_version_after": 4,
                        "accumulation_boundaries": 1,
                    },
                    "block_profile": [],
                }
            )
        )
    for lane in range(4):
        source_ids = lane_source_ids(lane)
        student = rank_root / "student" / f"rank_{lane:02d}.json"
        student.parent.mkdir(parents=True, exist_ok=True)
        student.write_text(
            json.dumps({**common, "rank": lane, "samples": samples(source_ids)})
        )
        teacher_rank = lane + 4
        teacher = rank_root / "teacher" / f"rank_{teacher_rank:02d}.json"
        teacher.parent.mkdir(parents=True, exist_ok=True)
        teacher.write_text(
            json.dumps(
                {**common, "rank": teacher_rank, "samples": samples(source_ids)}
            )
        )

    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_gst_pool_candidate.py",
            "--run-root",
            str(run_root),
            "--candidate-manifest",
            str(candidate_path),
            "--contract-digest",
            candidate.contract_digest,
            "--generator-return-code",
            "0",
            "--score-return-code",
            "0",
            "--output",
            str(output),
        ],
    )
    assert summarize_gst_pool_candidate.main() == 0
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["validation"]["logical_samples"] == 16
    assert result["validation"]["trace_and_x0_identity"] == "exact"
