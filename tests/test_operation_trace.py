import json
import random
from argparse import Namespace

import pytest
import torch

from runtime.operation_trace import OperationTrace, OperationTraceWriter, tensor_sha256


def test_trace_round_trip_is_operation_keyed_and_rng_independent(tmp_path):
    writer = OperationTraceWriter(classification="MINI", source={"sample_id": "x"})
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    key = writer.add_tensor("step/0/noise", base, role="noise_tensor")
    writer.add_operation(
        "step/0/generator",
        component="generator",
        operation="selected_forward",
        tensor_keys=[key],
        control={"exit": 1},
    )
    trace = writer.write(tmp_path, contract={"resolution": [384, 640]})
    expected = tensor_sha256(base)

    random.seed(999)
    torch.manual_seed(999)
    _ = torch.randn(1024)
    replay = OperationTrace.load(tmp_path)

    assert replay.trace_hash == trace.trace_hash
    assert tensor_sha256(replay.tensor(key)) == expected
    assert replay.operation("step/0/generator")["control"] == {"exit": 1}
    assert replay.manifest["tensors"][key]["stride"] == list(base.stride())


def test_trace_rejects_manifest_tampering(tmp_path):
    writer = OperationTraceWriter(classification="MINI", source={})
    key = writer.add_tensor("x", torch.ones(2), role="input")
    writer.add_operation("op", component="x", operation="read", tensor_keys=[key])
    writer.write(tmp_path, contract={})
    path = tmp_path / "trace.json"
    manifest = json.loads(path.read_text())
    manifest["contract"]["resolution"] = [1, 1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trace hash mismatch"):
        OperationTrace.load(tmp_path)


def test_trace_rejects_payload_tampering(tmp_path):
    writer = OperationTraceWriter(classification="MINI", source={})
    key = writer.add_tensor("x", torch.ones(2), role="input")
    writer.add_operation("op", component="x", operation="read", tensor_keys=[key])
    writer.write(tmp_path, contract={})
    tensors = torch.load(tmp_path / "tensors.pt", weights_only=False)
    tensors["x"][0] = 7
    torch.save(tensors, tmp_path / "tensors.pt")
    with pytest.raises(ValueError, match="tensor descriptor mismatch"):
        OperationTrace.load(tmp_path)


def test_duplicate_keys_fail_closed(tmp_path):
    writer = OperationTraceWriter(classification="MINI", source={})
    writer.add_tensor("x", torch.ones(1), role="input")
    with pytest.raises(ValueError, match="duplicate"):
        writer.add_tensor("x", torch.zeros(1), role="input")
    writer.add_operation("op", component="x", operation="read", tensor_keys=["x"])
    with pytest.raises(ValueError, match="duplicate"):
        writer.add_operation("op", component="x", operation="read")


def test_rank_trace_bundle_resolves_rank_specific_trace(tmp_path):
    from runtime.operation_trace import (
        load_operation_trace_bundle_manifest,
        resolve_rank_trace_root,
        write_operation_trace_bundle_manifest,
    )

    entries = []
    for rank in range(2):
        root = tmp_path / f"rank{rank}"
        writer = OperationTraceWriter(classification="MINI", source={"sample_id": rank})
        key = writer.add_tensor("sample/0/primary/target", torch.tensor([rank]), role="target")
        writer.add_operation("sample/0/source", component="input", operation="sample", tensor_keys=[key])
        trace = writer.write(root, contract={})
        entries.append({"rank": rank, "path": f"rank{rank}", "trace_hash": trace.trace_hash})
    bundle = write_operation_trace_bundle_manifest(tmp_path, entries)
    assert load_operation_trace_bundle_manifest(tmp_path)["bundle_hash"] == bundle["bundle_hash"]
    assert resolve_rank_trace_root(tmp_path, 1) == tmp_path / "rank1"
    assert OperationTrace.load(resolve_rank_trace_root(tmp_path, 1)).tensor(
        "sample/0/primary/target"
    ).item() == 1


def test_rank_trace_bundle_rejects_out_of_range_rank(tmp_path):
    from runtime.operation_trace import resolve_rank_trace_root, write_operation_trace_bundle_manifest

    root = tmp_path / "rank0"
    writer = OperationTraceWriter(classification="MINI", source={})
    key = writer.add_tensor("x", torch.ones(1), role="input")
    writer.add_operation("op", component="input", operation="sample", tensor_keys=[key])
    trace = writer.write(root, contract={})
    write_operation_trace_bundle_manifest(
        tmp_path, [{"rank": 0, "path": "rank0", "trace_hash": trace.trace_hash}]
    )
    with pytest.raises(ValueError, match="outside operation trace bundle"):
        resolve_rank_trace_root(tmp_path, 1)


def test_prepare_m1_calibration_assets_reuses_one_operation_trace(tmp_path):
    from scripts.distillschedule.prepare_m1_calibration_assets import prepare
    from runtime.operation_trace import load_operation_trace_bundle_manifest

    trace_root = tmp_path / "source_trace"
    writer = OperationTraceWriter(classification="MINI", source={"sample_id": "cal"})
    key = writer.add_tensor("x", torch.arange(4), role="input")
    writer.add_operation("op", component="input", operation="sample", tensor_keys=[key])
    trace = writer.write(trace_root, contract={})
    dataset = tmp_path / "source_dataset"
    dataset.mkdir()
    for index in range(4):
        torch.save({"index": index}, dataset / f"source_{index:03d}.pt")
    output_root = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    result = prepare(
        Namespace(
            source_trace=trace_root,
            source_dataset=dataset,
            output_root=output_root,
            worlds=[1, 2, 4],
            manifest=manifest_path,
        )
    )
    assert result["trace_hash"] == trace.trace_hash
    assert result["worlds"]["4"]["dataset_file_count"] == 4
    bundle = load_operation_trace_bundle_manifest(
        output_root / "traces" / "cal_a_rank4"
    )
    assert [entry["trace_hash"] for entry in bundle["ranks"]] == [
        trace.trace_hash
    ] * 4


def test_stage3_source_replay_is_authoritative_and_returns_private_materializations(tmp_path):
    from runtime.operation_trace import STAGE3_SOURCE_KEYS, replay_stage3_source

    writer = OperationTraceWriter(classification="MINI", source={"sample_id": "source"})
    shapes = {
        "primary_x0": (2, 1, 4, 5),
        "primary_history": (2, 7, 4, 5),
        "primary_target": (2, 3, 4, 5),
        "secondary_x0": (2, 1, 4, 5),
        "secondary_history": (2, 7, 4, 5),
        "secondary_target": (2, 3, 4, 5),
        "text_embedding": (6, 7),
    }
    keys = []
    for offset, (name, key) in enumerate(STAGE3_SOURCE_KEYS.items()):
        value = torch.arange(torch.tensor(shapes[name]).prod().item(), dtype=torch.float32)
        keys.append(writer.add_tensor(key, value.reshape(shapes[name]) + offset, role=name))
    writer.add_operation(
        "sample/0/source", component="input", operation="official_stage3_feature_sample", tensor_keys=keys
    )
    trace = writer.write(
        tmp_path,
        contract={"full_latent_shape": [1, 2, 3, 4, 5], "text_shape": [1, 6, 7]},
    )

    torch.manual_seed(123)
    first = replay_stage3_source(trace, device="cpu", dtype=torch.bfloat16)
    _ = torch.randn(100)
    torch.manual_seed(999)
    second = replay_stage3_source(trace, device="cpu", dtype=torch.bfloat16)

    assert first.keys() == second.keys() == STAGE3_SOURCE_KEYS.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert first["primary_target"].shape == (1, 2, 3, 4, 5)
    assert first["text_embedding"].shape == (1, 6, 7)
    assert all(value.dtype == torch.bfloat16 for value in first.values())
    first["primary_target"].zero_()
    assert trace.tensor(STAGE3_SOURCE_KEYS["primary_target"], clone=False).count_nonzero() > 0
