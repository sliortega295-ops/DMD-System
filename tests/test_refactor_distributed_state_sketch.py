from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from distillschedule.evidence.distributed_state_sketch import (
    SCHEMA_VERSION,
    compare_state_sketch_roots,
    sketch_named_tensors,
    write_rank_state_anchor,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _sketch(
    rank: int,
    values: list[float],
    *,
    owned_indices: set[int] | None = None,
    expected_replica_count: int = 2,
) -> dict:
    coordinates = {
        f"layer.weight[{index}]": number
        for index, number in enumerate(values)
        if owned_indices is None or index in owned_indices
    }
    inventory = [
        {
            "name": "layer.weight",
            "shape": [len(values)],
            "numel": len(values),
            "dtype": "torch.float32",
        }
    ]
    value = {
        "schema_version": SCHEMA_VERSION,
        "global_rank": rank,
        "samples_per_tensor": len(values),
        "max_tensors": 96,
        "inventory_tensor_count": 1,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest(),
        "selected_tensor_names": ["layer.weight"],
        "tensor_contracts": {
            "layer.weight": {
                "shape": [len(values)],
                "numel": len(values),
                "dtype": "torch.float32",
                "sample_flat_indices": list(range(len(values))),
            }
        },
        "expected_replica_counts": {
            "layer.weight": expected_replica_count,
        },
        "values": coordinates,
    }
    value["sha256"] = _digest(value)
    return value


def _write_run(
    root: Path,
    *,
    run_id: str,
    variant: str,
    values: list[float],
    cycles: tuple[int, ...] = (1, 2),
    world_size: int = 2,
) -> None:
    for cycle in cycles:
        for rank in range(world_size):
            write_rank_state_anchor(
                root
                / "trajectory"
                / "state"
                / f"rank_{rank:02d}_cycle_{cycle:03d}.json",
                run_id=run_id,
                workload="helios14b_nogan",
                variant=variant,
                cycle=cycle,
                rank=rank,
                world_size=world_size,
                states={
                    "generator_parameter": _sketch(
                        rank, values, expected_replica_count=world_size
                    ),
                    "critic_4_gradient": _sketch(
                        rank, values, expected_replica_count=world_size
                    ),
                },
            )


def test_state_comparator_merges_replicas_and_emits_coarse_metrics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, run_id="base", variant="baseline", values=[1.0, 2.0, 3.0])
    _write_run(
        candidate,
        run_id="candidate",
        variant="optimized",
        values=[1.0, 2.0, 3.001],
    )

    result = compare_state_sketch_roots(
        reference, candidate, required_cycles=(1, 2)
    )

    assert result["status"] == "PASS"
    assert result["comparison_count"] == 4
    row = next(
        value
        for value in result["comparisons"]
        if value["cycle"] == 1 and value["state_key"] == "generator_parameter"
    )
    assert row["component"] == "generator"
    assert row["state_kind"] == "parameter"
    assert row["reference_merge"]["replica_agreement"] == "EXACT"
    assert row["reference_merge"]["replicated_coordinate_count"] == 3
    assert row["metrics"]["coordinate_count"] == 3
    assert row["cosine"] > 0.999
    assert row["min_named_tensor_cosine"] > 0.995


def test_comparator_matches_resident_replication_to_fsdp_shards(tmp_path: Path) -> None:
    reference = tmp_path / "resident"
    candidate = tmp_path / "fsdp"
    values = [1.0, 2.0, 3.0, 4.0]
    for rank in range(2):
        write_rank_state_anchor(
            reference / "trajectory/state" / f"rank_{rank:02d}_cycle_001.json",
            run_id="resident",
            workload="wan13b",
            variant="resident",
            cycle=1,
            rank=rank,
            world_size=2,
            states={
                "generator_parameter": _sketch(
                    rank, values, expected_replica_count=2
                )
            },
        )
        write_rank_state_anchor(
            candidate / "trajectory/state" / f"rank_{rank:02d}_cycle_001.json",
            run_id="fsdp",
            workload="wan13b",
            variant="world16-fsdp",
            cycle=1,
            rank=rank,
            world_size=2,
            states={
                "generator_parameter": _sketch(
                    rank,
                    values,
                    owned_indices={0, 1} if rank == 0 else {2, 3},
                    expected_replica_count=1,
                )
            },
        )

    result = compare_state_sketch_roots(
        reference, candidate, required_cycles=(1,)
    )
    assert result["status"] == "PASS"
    row = result["comparisons"][0]
    assert row["reference_merge"]["max_replica_count"] == 2
    assert row["candidate_merge"]["max_replica_count"] == 1
    assert row["cosine"] == pytest.approx(1.0)
    assert row["normalized_l2"] == pytest.approx(0.0)


def test_numerical_drift_is_review_and_cli_still_completes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(
        reference,
        run_id="base",
        variant="baseline",
        values=[1.0, 0.0],
        cycles=(1,),
    )
    _write_run(
        candidate,
        run_id="candidate",
        variant="optimized",
        values=[0.0, 1.0],
        cycles=(1,),
    )
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/compare_trajectory_state_sketches.py"),
            "--reference-root",
            str(reference),
            "--candidate-root",
            str(candidate),
            "--required-cycles",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "REVIEW"
    assert result["review_count"] == 2
    assert result["evaluation_policy"] == "POST_RUN_ONLY_NO_NUMERICAL_EARLY_STOP"


def test_comparator_rejects_replica_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, run_id="base", variant="baseline", values=[1.0], cycles=(1,))
    _write_run(
        candidate,
        run_id="candidate",
        variant="optimized",
        values=[1.0],
        cycles=(1,),
    )

    path = reference / "trajectory/state/rank_01_cycle_001.json"
    value = json.loads(path.read_text())
    sketch = value["states"]["generator_parameter"]["sketch"]
    sketch["values"]["layer.weight[0]"] = 2.0
    sketch_body = dict(sketch)
    sketch_body.pop("sha256")
    sketch["sha256"] = _digest(sketch_body)
    anchor_body = dict(value)
    anchor_body.pop("sha256")
    value["sha256"] = _digest(anchor_body)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="replicated ranks disagree"):
        compare_state_sketch_roots(reference, candidate, required_cycles=(1,))


def test_comparator_rejects_missing_rank_and_tampered_hash(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, run_id="base", variant="baseline", values=[1.0], cycles=(1,))
    _write_run(
        candidate,
        run_id="candidate",
        variant="optimized",
        values=[1.0],
        cycles=(1,),
    )
    missing = candidate / "trajectory/state/rank_01_cycle_001.json"
    missing.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        compare_state_sketch_roots(reference, candidate, required_cycles=(1,))

    path = reference / "trajectory/state/rank_00_cycle_001.json"
    value = json.loads(path.read_text())
    value["variant"] = "tampered"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="SHA256"):
        compare_state_sketch_roots(reference, reference, required_cycles=(1,))


def test_comparator_rejects_missing_replica_contribution(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(
        reference,
        run_id="base",
        variant="baseline",
        values=[1.0, 2.0],
        cycles=(1,),
    )
    _write_run(
        candidate,
        run_id="candidate",
        variant="optimized",
        values=[1.0, 2.0],
        cycles=(1,),
    )
    path = candidate / "trajectory/state/rank_01_cycle_001.json"
    value = json.loads(path.read_text())
    for record in value["states"].values():
        sketch = record["sketch"]
        sketch["values"] = {}
        sketch_body = dict(sketch)
        sketch_body.pop("sha256")
        sketch["sha256"] = _digest(sketch_body)
    anchor_body = dict(value)
    anchor_body.pop("sha256")
    value["sha256"] = _digest(anchor_body)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="replica contribution count"):
        compare_state_sketch_roots(reference, candidate, required_cycles=(1,))


def test_comparator_rejects_cross_anchor_sampling_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(
        reference, run_id="base", variant="baseline", values=[1.0, 2.0]
    )
    _write_run(
        candidate, run_id="candidate", variant="optimized", values=[1.0, 2.0]
    )
    for root in (reference, candidate):
        for rank in range(2):
            path = root / f"trajectory/state/rank_{rank:02d}_cycle_002.json"
            value = json.loads(path.read_text())
            for record in value["states"].values():
                sketch = record["sketch"]
                sketch["samples_per_tensor"] = 1
                sketch["tensor_contracts"]["layer.weight"][
                    "sample_flat_indices"
                ] = [0]
                sketch["values"] = {"layer.weight[0]": 1.0}
                sketch_body = dict(sketch)
                sketch_body.pop("sha256")
                sketch["sha256"] = _digest(sketch_body)
            anchor_body = dict(value)
            anchor_body.pop("sha256")
            value["sha256"] = _digest(anchor_body)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="changed across anchors"):
        compare_state_sketch_roots(reference, candidate, required_cycles=(1, 2))


def test_plain_resident_sketch_covers_all_global_coordinates() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = sketch_named_tensors(
        [("layer.weight", tensor)],
        global_rank=7,
        samples_per_tensor=4,
        max_tensors=1,
    )
    assert result["global_rank"] == 7
    assert result["values"] == {
        "layer.weight[0]": 1.0,
        "layer.weight[1]": 2.0,
        "layer.weight[2]": 3.0,
        "layer.weight[3]": 4.0,
    }
    body = dict(result)
    supplied = body.pop("sha256")
    assert supplied == _digest(body)
    with pytest.raises(ValueError, match="inventory cannot be empty"):
        sketch_named_tensors([], global_rank=0)


def test_write_anchor_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "rank_00_cycle_001.json"
    kwargs = {
        "run_id": "run",
        "workload": "helios14b_nogan",
        "variant": "baseline",
        "cycle": 1,
        "rank": 0,
        "world_size": 1,
        "states": {
            "generator_parameter": _sketch(
                0, [1.0], expected_replica_count=1
            )
        },
    }
    metadata = write_rank_state_anchor(path, **kwargs)
    assert metadata["state_sketch_path"] == str(path)
    assert len(metadata["state_sketch_sha256"]) == 64
    assert len(metadata["state_content_sha256"]) == 64
    assert metadata["state_sketch_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_rank_state_anchor(path, **kwargs)
