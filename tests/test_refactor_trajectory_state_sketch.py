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
    def __init__(self) -> None:
        self.observed_group = None

    def is_initialized(self) -> bool:
        return True

    def get_rank(self, group: object | None = None) -> int:
        self.observed_group = group
        return 0

    def get_world_size(self, group: object | None = None) -> int:
        self.observed_group = group
        return 1

    def all_gather_object(
        self,
        output: list[object],
        value: object,
        group: object | None = None,
    ) -> None:
        self.observed_group = group
        output[0] = value


class _TwoReplicaDistributed(_FakeDistributed):
    def __init__(self, *, disagree: bool = False) -> None:
        super().__init__()
        self.disagree = disagree

    def get_world_size(self, group: object | None = None) -> int:
        self.observed_group = group
        return 2

    def all_gather_object(
        self,
        output: list[object],
        value: object,
        group: object | None = None,
    ) -> None:
        self.observed_group = group
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


def test_numeric_sketch_uses_explicit_object_collective_group() -> None:
    distributed = _FakeDistributed()
    torch = SimpleNamespace(
        distributed=distributed,
        int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )
    evidence_group = object()
    collect_numeric_state_sketch(
        torch=torch,
        named_values=[("layer.weight", _FakeTensor([1.0]), None)],
        object_collective_group=evidence_group,
    )
    assert distributed.observed_group is evidence_group


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


class _RolePoolDistributed(_FakeDistributed):
    """Eight world contributors, with only four physical model owners."""

    def __init__(self, *, owners: tuple[int, ...], rank: int = 0, failure: str = "") -> None:
        super().__init__()
        self.owners, self.rank, self.failure = owners, rank, failure

    def get_rank(self, group: object | None = None) -> int:
        return self.rank

    def get_world_size(self, group: object | None = None) -> int:
        return 8

    def all_gather_object(self, output: list[object], value: object, group=None) -> None:
        template = {
            "name": "layer.weight", "shape": [2], "dtype": "torch.float32",
            "numel": 2, "requested_indices": [0, 1], "values": {"0": 1.0, "1": 2.0},
            "representation": "FULL_REPLICA",
        }
        for rank in range(8):
            owns = rank in self.owners
            row = {**copy.deepcopy(template), "contributor_rank": rank}
            item = {
                **copy.deepcopy(value),
                "inventory_names": ["layer.weight"] if owns else [],
                "rows": [row] if owns else [],
            }
            if rank == self.owners[-1]:
                if self.failure == "missing_owner":
                    item.update(inventory_names=[], rows=[])
                elif self.failure == "owner_inventory":
                    item["inventory_names"] = ["other.weight"]
                elif self.failure == "owner_disagreement":
                    item["rows"][0]["values"]["0"] = 99.0
                elif self.failure == "owner_declaration":
                    item["owner_ranks"] = [rank]
            if self.failure == "nonowner_state" and rank == next(
                rank for rank in range(8) if rank not in self.owners
            ):
                item.update(inventory_names=["layer.weight"], rows=[row])
            output[rank] = item


def _role_pool_torch(distributed):
    return SimpleNamespace(
        distributed=distributed, int64="torch.int64",
        tensor=lambda values, **_kwargs: _FakeTensor(list(values)),
    )


@pytest.mark.parametrize("owners,rank", [((0, 1, 2, 3), 0), ((4, 5, 6, 7), 0), ((4, 5, 6, 7), 7)])
def test_role_pool_sketch_requires_every_declared_owner_not_every_world_rank(owners, rank):
    sketch = collect_numeric_state_sketch(
        torch=_role_pool_torch(_RolePoolDistributed(owners=owners, rank=rank)),
        named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)] if rank in owners else [],
        primary_rank=min(owners), owner_ranks=owners,
    )
    assert sketch["world_size"] == 8
    assert sketch["owner_ranks"] == list(owners)
    assert sketch["tensors"][0]["contributor_ranks"] == list(owners)
    assert sketch["replica_validation"] == "ALL_OWNER_RANKS_CONTRIBUTE_SAMPLED_COORDINATES_MUST_MATCH"
    reference = collect_numeric_state_sketch(
        torch=_role_pool_torch(_FakeDistributed()),
        named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)],
    )
    comparison = compare_numeric_state_sketches(reference, sketch, cycle=1, component="generator_trainable")
    assert comparison.normalized_l2 == 0.0


@pytest.mark.parametrize("failure,signature", [
    ("missing_owner", "inventory differs"), ("owner_inventory", "inventory differs"),
    ("owner_disagreement", "disagree"), ("nonowner_state", "non-owner"),
    ("owner_declaration", "owner rank declaration"),
])
def test_role_pool_sketch_does_not_weaken_owner_or_replica_gates(failure, signature):
    owners = (0, 1, 2, 3)
    with pytest.raises(AssertionError, match=signature):
        collect_numeric_state_sketch(
            torch=_role_pool_torch(_RolePoolDistributed(owners=owners, failure=failure)),
            named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)],
            primary_rank=0, owner_ranks=owners,
        )


@pytest.mark.parametrize("owners", [(), (0, 0), (-1, 0), (0, 8), (1, 2)])
def test_role_pool_sketch_rejects_invalid_ownership(owners):
    with pytest.raises(ValueError, match="owner"):
        collect_numeric_state_sketch(
            torch=_role_pool_torch(_FakeDistributed()), named_values=[], owner_ranks=owners,
        )


def test_unspecified_owners_still_require_every_world_rank_inventory():
    with pytest.raises(AssertionError, match="inventory differs"):
        collect_numeric_state_sketch(
            torch=_role_pool_torch(_RolePoolDistributed(owners=(0, 1, 2, 3))),
            named_values=[("layer.weight", _FakeTensor([1.0, 2.0]), None)],
        )
