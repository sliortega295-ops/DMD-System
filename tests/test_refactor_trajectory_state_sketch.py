from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from distillschedule.evidence.trajectory_state_sketch import (
    SCHEMA_VERSION,
    collect_numeric_state_sketch,
    compare_numeric_state_sketches,
)


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.shape = (len(values),)
        self.dtype = "torch.float32"
        self.device = "cpu"

    def detach(self) -> "_FakeTensor":
        return self

    def contiguous(self) -> "_FakeTensor":
        return self

    def view(self, _shape: int) -> "_FakeTensor":
        return self

    def index_select(self, _dimension: int, indices: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor([self.values[int(index)] for index in indices.values])

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeDistributed:
    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def all_gather_object(self, output: list[object], value: object) -> None:
        output[0] = value


class _TwoReplicaDistributed(_FakeDistributed):
    def __init__(self, *, disagree: bool = False) -> None:
        self.disagree = disagree

    def get_world_size(self) -> int:
        return 2

    def all_gather_object(self, output: list[object], value: object) -> None:
        replica = copy.deepcopy(value)
        for row in replica["rows"]:
            row["contributor_rank"] = 1
            if self.disagree and row["values"]:
                first = next(iter(row["values"]))
                row["values"][first] += 1.0
        output[:] = [value, replica]


def test_full_replica_sketch_exposes_numeric_coordinate_values() -> None:
    torch = SimpleNamespace(
        distributed=_FakeDistributed(),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    sketch = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("module._orig_mod.layer.weight", _FakeTensor([1.0, 2.0, 3.0]), None)],
        samples_per_tensor=16,
    )
    assert sketch["schema_version"] == SCHEMA_VERSION
    assert sketch["inventory_tensor_count"] == 1
    assert sketch["coordinate_count"] == 3
    assert sketch["replica_validation"] == (
        "ALL_RANKS_CONTRIBUTE_SAMPLED_COORDINATES_MUST_MATCH"
    )
    assert sketch["replica_agreement_scope"] == "SAMPLED_COORDINATES_NOT_FULL_TENSOR_PROOF"
    assert sketch["coordinates"] == [
        {"tensor": "layer.weight", "flat_index": 0, "value_fp32": 1.0},
        {"tensor": "layer.weight", "flat_index": 1, "value_fp32": 2.0},
        {"tensor": "layer.weight", "flat_index": 2, "value_fp32": 3.0},
    ]


def test_schema_pinned_comparator_aligns_global_numeric_coordinates() -> None:
    torch = SimpleNamespace(
        distributed=_FakeDistributed(),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    reference = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("layer.weight", _FakeTensor([1.0, 2.0, 3.0]), None)],
    )
    candidate = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("layer.weight", _FakeTensor([1.0, 2.0, 4.0]), None)],
    )
    comparison = compare_numeric_state_sketches(
        reference,
        candidate,
        cycle=5,
        component="generator_trainable",
    )
    assert comparison.cycle == 5
    assert comparison.component == "generator_trainable"
    assert 0.0 < comparison.normalized_l2 < 1.0
    assert comparison.cosine < 1.0


def test_comparator_rejects_unversioned_replica_evidence() -> None:
    torch = SimpleNamespace(
        distributed=_FakeDistributed(),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    sketch = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("layer.weight", _FakeTensor([1.0]), None)],
    )
    stale = dict(sketch, schema_version="distillschedule.trajectory_numeric_state_sketch.v1")
    try:
        compare_numeric_state_sketches(
            stale,
            sketch,
            cycle=1,
            component="generator_trainable",
        )
    except ValueError as error:
        assert "schema" in str(error)
    else:  # pragma: no cover - assertion explains the fail-closed contract
        raise AssertionError("stale state-sketch schema was accepted")


def test_full_replica_sketch_verifies_all_rank_copies() -> None:
    torch = SimpleNamespace(
        distributed=_TwoReplicaDistributed(),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    sketch = collect_numeric_state_sketch(
        torch=torch,
        named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)],
    )
    assert sketch["world_size"] == 2
    assert sketch["tensors"][0]["contributor_ranks"] == [0, 1]


def test_full_replica_sketch_rejects_rank_disagreement() -> None:
    torch = SimpleNamespace(
        distributed=_TwoReplicaDistributed(disagree=True),
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    with pytest.raises(AssertionError, match="disagree"):
        collect_numeric_state_sketch(
            torch=torch,
            named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)],
        )
